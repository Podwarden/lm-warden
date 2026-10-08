"""Per-file pull plumbing (#85).

When the wizard pins a ``filename`` on the model row, ``run_pull`` must:

1. Compute ``allow_patterns`` = ``[filename, config.json, tokenizer*, *.txt, *.md]``.
2. Feed that into ``estimate_repo_bytes`` (so the disk-shortage check sees the
   filtered size, not the whole repo — a 19.8 GB single-file pull on a repo
   with a 200 GB total weight set must not trip a false shortage).
3. Feed it into ``insufficient_disk`` and ``_snapshot_download`` so the
   download is also narrowed.

When ``filename`` is None (legacy / whole-repo rows), every callsite gets
``allow_patterns=None`` — preserving the pre-#85 behaviour.
"""

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.models.pull_task import allow_patterns_for


def test_allow_patterns_for_with_filename():
    pats = allow_patterns_for("model.safetensors")
    assert pats is not None
    assert "model.safetensors" in pats
    assert "config.json" in pats
    assert "tokenizer*" in pats
    assert "*.txt" in pats
    assert "*.md" in pats


def test_allow_patterns_for_none_means_whole_repo():
    assert allow_patterns_for(None) is None


async def _seed_model(db_path, filename: str | None):
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id="m1",
                served_model_name="m1",
                hf_repo="o/r",
                hf_revision="main",
                gpu_indices=[0],
                tensor_parallel_size=1,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=[],
                extra_env={},
                status="registered",
                pulled_bytes=0,
                pulled_total=None,
                last_error=None,
                filename=filename,
            )
        )


async def test_run_pull_pinned_file_passes_allow_patterns_through(tmp_data_dir, monkeypatch):
    """``filename`` set on the row -> ``allow_patterns`` lands in every call."""
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path, filename="model-00001-of-00002.safetensors")

    seen: dict = {}

    async def fake_estimate(repo, revision, token, allow_patterns=None):
        seen["estimate"] = allow_patterns
        return 1024

    async def fake_disk_check(repo, revision, cache_dir, token, estimate=None, allow_patterns=None):
        seen["disk_check"] = allow_patterns
        return None

    async def fake_download(model_id, settings_, repo, revision, token, allow_patterns=None):
        seen["download"] = allow_patterns
        return "/tmp/fake"

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    # #111: picking a sharded safetensors member must expand to all sibling
    # shards plus the weight_map index — otherwise vLLM/transformers loads
    # ``model.safetensors.index.json`` and crashes because the referenced
    # sibling shards aren't on disk.
    expected = [
        "model-00001-of-00002.safetensors",
        "config.json",
        "tokenizer*",
        "*.txt",
        "*.md",
        "model-*-of-00002.safetensors",
        "*.safetensors.index.json",
    ]
    assert seen["estimate"] == expected
    assert seen["disk_check"] == expected
    assert seen["download"] == expected


async def test_run_pull_whole_repo_passes_none_allow_patterns(tmp_data_dir, monkeypatch):
    """``filename`` is None -> every callee sees ``allow_patterns=None``,
    i.e. behaviour identical to the pre-#85 whole-repo pull."""
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path, filename=None)

    seen: dict = {}

    async def fake_estimate(repo, revision, token, allow_patterns=None):
        seen["estimate"] = allow_patterns
        return 1024

    async def fake_disk_check(repo, revision, cache_dir, token, estimate=None, allow_patterns=None):
        seen["disk_check"] = allow_patterns
        return None

    async def fake_download(model_id, settings_, repo, revision, token, allow_patterns=None):
        seen["download"] = allow_patterns
        return "/tmp/fake"

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    assert seen["estimate"] is None
    assert seen["disk_check"] is None
    assert seen["download"] is None


async def test_estimate_repo_bytes_filters_siblings_by_allow_patterns(monkeypatch):
    """Verify ``estimate_repo_bytes`` only sums siblings matching the patterns.

    #111: picking a sharded safetensors member must size the *whole* shard
    set + weight_map index — not just the picked shard. The previous
    contract (one shard's bytes + extras) was the broken behaviour that
    caused only the first shard to actually download.
    """
    from app.models import pull_task as pt

    class _Sib:
        def __init__(self, name, size):
            self.rfilename = name
            self.size = size

    class _Info:
        siblings = [
            _Sib("model-00001-of-00002.safetensors", 10 * 1024**3),
            _Sib("model-00002-of-00002.safetensors", 10 * 1024**3),
            _Sib("model.safetensors.index.json", 32 * 1024),
            _Sib("config.json", 4 * 1024),
            _Sib("tokenizer.json", 2 * 1024**2),
            _Sib("README.md", 1024),
            _Sib("optimizer.pt", 50 * 1024**3),  # big tail we don't want
            _Sib("weird-extra.bin", 5 * 1024**3),  # also unwanted
        ]

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo, revision=None, files_metadata=False):
            return _Info()

    monkeypatch.setattr(pt, "HfApi", _FakeApi)

    # Whole-repo (no filter): sums everything.
    total_all = await pt.estimate_repo_bytes("o/r", "main", hf_token=None)
    assert total_all == sum(s.size for s in _Info.siblings)

    # Per-file: pin shard 1, expect BOTH shards + index.json + extras
    # (config + tokenizer + README). The big optimizer / weird-extra.bin
    # must NOT be included.
    pats = pt.allow_patterns_for("model-00001-of-00002.safetensors")
    total_filtered = await pt.estimate_repo_bytes("o/r", "main", hf_token=None, allow_patterns=pats)
    expected = (
        10 * 1024**3  # shard 1
        + 10 * 1024**3  # shard 2
        + 32 * 1024  # index.json
        + 4 * 1024  # config.json
        + 2 * 1024**2  # tokenizer.json
        + 1024  # README.md
    )
    assert total_filtered == expected
    # The big optimizer + extra.bin tail must NOT be included.
    assert total_filtered < 21 * 1024**3


def test_allow_patterns_for_sharded_safetensors_expands_to_full_set():
    """Sharded safetensors: include the shard glob + weight_map index (#111)."""
    pats = allow_patterns_for("model-00001-of-00005.safetensors")
    assert pats is not None
    # Base extras still present.
    assert "model-00001-of-00005.safetensors" in pats
    assert "config.json" in pats
    assert "tokenizer*" in pats
    assert "*.txt" in pats
    assert "*.md" in pats
    # New: full shard glob + safetensors weight_map index.
    assert "model-*-of-00005.safetensors" in pats
    assert "*.safetensors.index.json" in pats


def test_allow_patterns_for_sharded_gguf_expands_without_index_json():
    """Sharded GGUF: shard glob present, but NO safetensors index (#111).

    llama.cpp resolves GGUF shards by suffix and does not need a weight_map.
    """
    pats = allow_patterns_for("llama-Q5_K_M-00001-of-00003.gguf")
    assert pats is not None
    assert "llama-Q5_K_M-00001-of-00003.gguf" in pats
    assert "llama-Q5_K_M-*-of-00003.gguf" in pats
    assert "*.safetensors.index.json" not in pats


def test_allow_patterns_for_single_safetensors_unchanged():
    """Non-sharded files: legacy patterns only, no shard glob (#111)."""
    pats = allow_patterns_for("model.safetensors")
    assert pats == [
        "model.safetensors",
        "config.json",
        "tokenizer*",
        "*.txt",
        "*.md",
    ]


# ---------------------------------------------------------------------------
# Sub-project C: the multimodal projector is part of a per-file pull
# ---------------------------------------------------------------------------


def test_mmproj_is_included_in_a_per_file_pull():
    """Without this the projector is never downloaded and Task 6's resolver
    raises ModelFileNotFound at load time -- correct, but late and confusing,
    because the operator DID select a vision model."""
    pats = allow_patterns_for("model-IQ3_XXS.gguf", mmproj_filename="mmproj-BF16.gguf")
    assert "mmproj-BF16.gguf" in pats
    assert "model-IQ3_XXS.gguf" in pats


def test_mmproj_absent_leaves_the_patterns_exactly_as_before():
    """The vLLM path must not gain a file. Byte-for-byte the pre-C list."""
    assert allow_patterns_for("model.safetensors") == [
        "model.safetensors",
        "config.json",
        "tokenizer*",
        "*.txt",
        "*.md",
    ]


def test_whole_repo_pull_is_still_none():
    assert allow_patterns_for(None) is None
    assert allow_patterns_for(None, mmproj_filename="mmproj-BF16.gguf") is None


def test_sharded_gguf_still_expands_to_the_whole_family_with_an_mmproj():
    pats = allow_patterns_for("m-Q4_K_M-00001-of-00003.gguf", mmproj_filename="mmproj-BF16.gguf")
    assert any("-of-00003.gguf" in p and "*" in p for p in pats)
    assert "mmproj-BF16.gguf" in pats


def test_the_projector_is_appended_last_not_folded_into_the_shard_glob():
    """It is not a member of the weights shard family, and ordering matters
    only in that a reviewer reading the list should see it as an addition."""
    pats = allow_patterns_for("m.gguf", mmproj_filename="mmproj-BF16.gguf")
    assert pats[-1] == "mmproj-BF16.gguf"
