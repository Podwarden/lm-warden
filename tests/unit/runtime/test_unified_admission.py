"""Unified-memory admission (Metal design s10, "The GpuOwnership trap").

Every Mac model row carries ``gpu_indices == [0]``, so the discrete-GPU
``GpuOwnership`` refuses the SECOND model on gpu 0 -- silently serialising
the product to one model per Mac. On unified memory there is nothing to own:
the question is bytes. ``UnifiedMemoryAdmission`` refuses a load only when
``resident + estimate > budget``, and the supervisor takes the admission as
an injected collaborator (no platform check inside it). Linux keeps
``GpuOwnership`` byte-for-byte.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.capacity import unified_weights_budget
from app.runtime.backends import LaunchPlan
from app.runtime.gpu_ownership import GpuConflict, GpuOwnership
from app.runtime.supervisor import Supervisor
from app.runtime.unified_admission import UnifiedMemoryAdmission, admission_for_profile

GIB = 1024**3


def _memsize_only(mib: int):
    """A capacity reading with only hw.memsize (no ceiling, no sysctls): the
    pre-P6 stand-in, so the budgets below are unchanged by the real probe."""
    from app.models.capacity import UnifiedCapacity

    return UnifiedCapacity(memsize=mib * 2**20)


def _sizes(table):
    """An estimate callable that reads a model's size from a table by id."""
    return lambda model: table[model.id]


def _m(model_id, name=None):
    return SimpleNamespace(id=model_id, served_model_name=name or model_id)


def _claim(adm, model, gpus=(0,)):
    adm.claim(model.id, list(gpus), label=model.served_model_name, model=model)


# --- the admission itself --------------------------------------------------


def test_two_models_that_fit_both_load_on_gpu_0():
    """The trap: GpuOwnership refuses the second; unified memory must not."""
    adm = UnifiedMemoryAdmission(
        budget_bytes=20 * GIB, estimate=_sizes({"a": 8 * GIB, "b": 8 * GIB})
    )
    _claim(adm, _m("a"))
    _claim(adm, _m("b"))
    assert adm.resident() == {"a": 8 * GIB, "b": 8 * GIB}


def test_load_that_would_exceed_budget_is_refused_and_names_occupants():
    adm = UnifiedMemoryAdmission(
        budget_bytes=20 * GIB, estimate=_sizes({"a": 8 * GIB, "b": 8 * GIB, "c": 5 * GIB})
    )
    _claim(adm, _m("a", "Llama-A"))
    _claim(adm, _m("b", "Qwen-B"))
    with pytest.raises(GpuConflict) as ei:
        _claim(adm, _m("c", "Big-C"))
    msg = str(ei.value)
    assert "Big-C" in msg and "Llama-A" in msg and "Qwen-B" in msg
    assert "unload" in msg.lower()
    # A refusal admits nothing.
    assert "c" not in adm.resident()


def test_exactly_at_budget_is_admitted():
    adm = UnifiedMemoryAdmission(
        budget_bytes=16 * GIB, estimate=_sizes({"a": 8 * GIB, "b": 8 * GIB})
    )
    _claim(adm, _m("a"))
    _claim(adm, _m("b"))
    assert sum(adm.resident().values()) == 16 * GIB


@pytest.mark.parametrize("budget", [0, -3 * GIB])
def test_unreadable_or_negative_budget_fails_closed(budget):
    """The p6f probe returns 0 when hw.memsize is unreadable, and
    unified_weights_budget can go negative; neither may admit anything."""
    adm = UnifiedMemoryAdmission(budget_bytes=budget, estimate=_sizes({"a": 1}))
    with pytest.raises(GpuConflict) as ei:
        _claim(adm, _m("a"))
    assert "budget" in str(ei.value).lower()
    assert adm.resident() == {}


@pytest.mark.parametrize("size", [0, -1])
def test_unknown_estimate_fails_closed(size):
    """0 means the weights are not on disk / unmeasurable: refuse rather than
    admit a model at zero cost and overcommit the machine."""
    adm = UnifiedMemoryAdmission(budget_bytes=20 * GIB, estimate=_sizes({"a": size}))
    with pytest.raises(GpuConflict):
        _claim(adm, _m("a"))
    assert adm.resident() == {}


def test_estimate_that_raises_fails_closed():
    def boom(model):
        raise OSError("EACCES")

    adm = UnifiedMemoryAdmission(budget_bytes=20 * GIB, estimate=boom)
    with pytest.raises(GpuConflict):
        _claim(adm, _m("a"))


def test_release_frees_the_bytes():
    adm = UnifiedMemoryAdmission(
        budget_bytes=10 * GIB, estimate=_sizes({"a": 8 * GIB, "b": 8 * GIB})
    )
    _claim(adm, _m("a"))
    with pytest.raises(GpuConflict):
        _claim(adm, _m("b"))
    adm.release("a")
    _claim(adm, _m("b"))
    assert adm.resident() == {"b": 8 * GIB}
    adm.release("never-claimed")  # idempotent, like GpuOwnership.release


def test_reclaim_by_the_same_model_is_not_double_counted():
    adm = UnifiedMemoryAdmission(budget_bytes=10 * GIB, estimate=_sizes({"a": 8 * GIB}))
    _claim(adm, _m("a"))
    _claim(adm, _m("a"))
    assert adm.resident() == {"a": 8 * GIB}


# --- the supervisor seam ---------------------------------------------------


class _Settings:
    load_max_s = 7200.0


def _settings(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    s.hf_cache_dir = str(tmp_path / "hf-cache")
    (tmp_path / "tok").write_text("hf_x")
    return s


def test_supervisor_defaults_to_gpu_ownership(tmp_path):
    """Linux behaviour is unchanged: no admission passed -> GpuOwnership."""
    assert isinstance(Supervisor(_settings(tmp_path)).gpus, GpuOwnership)


def test_supervisor_uses_the_injected_admission(tmp_path):
    adm = UnifiedMemoryAdmission(budget_bytes=GIB, estimate=_sizes({}))
    assert Supervisor(_settings(tmp_path), admission=adm).gpus is adm


def _model_row(model_id):
    model = MagicMock()
    model.id = model_id
    model.served_model_name = model_id
    model.gpu_indices = [0]
    model.tensor_parallel_size = 1
    model.engine_image = None
    model.engine_channel = None
    model.engine_vllm_version = None
    model.backend = None
    model.mmproj_filename = None
    model.filename = None
    return model


async def _load(sup, model, port):
    proc = MagicMock()
    proc.pid = port
    proc.returncode = None
    plan = LaunchPlan(argv=["vllm", "serve"], env={}, image=None, port=port, gpu_indices=[0])
    with (
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch("app.runtime.backends.vllm.VllmBackend.plan", return_value=plan),
    ):
        await sup.load(model, port=port)


@pytest.mark.asyncio
async def test_supervisor_loads_two_models_on_gpu_0_under_unified_admission(tmp_path):
    adm = UnifiedMemoryAdmission(
        budget_bytes=20 * GIB, estimate=_sizes({"a": 8 * GIB, "b": 8 * GIB, "c": 8 * GIB})
    )
    sup = Supervisor(_settings(tmp_path), admission=adm)
    await _load(sup, _model_row("a"), 10001)
    await _load(sup, _model_row("b"), 10002)
    assert sup.is_running("a") and sup.is_running("b")
    with pytest.raises(GpuConflict):
        await _load(sup, _model_row("c"), 10003)
    assert not sup.is_running("c")
    for w in list(sup._watchers.values()):
        w.cancel()


@pytest.mark.asyncio
async def test_supervisor_with_default_admission_still_refuses_second_on_gpu_0(tmp_path):
    """Regression guard for the Linux contract: one model per GPU."""
    sup = Supervisor(_settings(tmp_path))
    await _load(sup, _model_row("a"), 10001)
    with pytest.raises(GpuConflict):
        await _load(sup, _model_row("b"), 10002)
    for w in list(sup._watchers.values()):
        w.cancel()


# --- choosing the admission from the deployment profile -------------------


@pytest.mark.asyncio
async def test_linux_profile_gets_gpu_ownership(tmp_path):
    probe = AsyncMock(return_value=_memsize_only(16 * 1024))
    with patch("app.system.gpu._apple_capacity", new=probe):
        adm = await admission_for_profile("linux", hf_cache_dir=tmp_path)
    assert isinstance(adm, GpuOwnership)
    probe.assert_not_called()  # no sysctl on Linux


@pytest.mark.asyncio
async def test_apple_profile_budget_is_unified_weights_budget_of_memsize(tmp_path):
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    assert isinstance(adm, UnifiedMemoryAdmission)
    assert adm.budget_bytes == unified_weights_budget(16 * GIB, 0, 16 * GIB, 0, 0)


@pytest.mark.asyncio
async def test_apple_profile_budget_uses_device_ceiling_and_usable(tmp_path):
    """P6: the real ceiling (recommendedMaxWorkingSetSize) and
    hw.memsize_usable replace the hw.memsize stand-in in the admission budget."""
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(
        memsize=16 * GIB,
        memsize_usable=16554000384,
        wired_limit_mb=0,
        device_ceiling=12713115648,
    )
    with patch("app.system.gpu._apple_capacity", new=AsyncMock(return_value=cap)):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    # The Metal ceiling already leaves the OS its share: no OS reserve on top.
    assert adm.budget_bytes == 12713115648
    assert adm.budget_bytes < unified_weights_budget(16 * GIB, 0, 16 * GIB, 0, 0)


@pytest.mark.asyncio
async def test_apple_profile_unreadable_memsize_fails_closed(tmp_path):
    with patch("app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(0))):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    assert adm.budget_bytes == 0
    with pytest.raises(GpuConflict):
        _claim(adm, _m("a"))


def _tree(tmp_path, files, snapshot="s1", slug="models--org--tiny"):
    """Blobs + a single snapshot of symlinks over them.

    ``files`` maps a snapshot filename to a blob size in bytes; each name
    gets its own blob and the snapshot links every one in. Returns (blobs,
    snapshot) so tests can add their own links.
    """
    blobs = tmp_path / slug / "blobs"
    snap = tmp_path / slug / "snapshots" / snapshot
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, size in files.items():
        b = blobs / name
        b.write_bytes(b"\0" * size)
        (snap / name).symlink_to(b)
    return blobs, snap


@pytest.mark.asyncio
async def test_apple_profile_estimate_sizes_only_the_pinned_quant(tmp_path):
    """The estimate sizes the files the row actually loads, not every blob."""
    _tree(
        tmp_path,
        {"Q4.gguf": 1000, "Q8.gguf": 5000, "config.json": 24},
    )
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(
        id="t", served_model_name="tiny", hf_repo="org/tiny", filename="Q4.gguf"
    )
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 1024


@pytest.mark.asyncio
async def test_estimate_whole_snapshot_when_no_filename(tmp_path):
    """A row with no pinned filename loads the whole repo: every file counts."""
    _tree(tmp_path, {"Q4.gguf": 1000, "Q8.gguf": 5000, "config.json": 24})
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(id="t", served_model_name="tiny", hf_repo="org/tiny", filename=None)
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 6024


@pytest.mark.asyncio
async def test_estimate_counts_gguf_shard_siblings(tmp_path):
    """A sharded GGUF pull counts the shard glob's siblings, not other quants."""
    _tree(
        tmp_path,
        {
            "m-00001-of-00002.gguf": 100,
            "m-00002-of-00002.gguf": 200,
            "other.gguf": 999,
        },
    )
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(
        id="t",
        served_model_name="tiny",
        hf_repo="org/tiny",
        filename="m-00001-of-00002.gguf",
    )
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 300


@pytest.mark.asyncio
async def test_estimate_counts_gguf_shards_in_a_subdirectory(tmp_path):
    """Many GGUF repos keep each quant's shard set in its own folder
    (``Q4_K_M/m-00001-of-00002.gguf``): the walk must descend into it, or the
    estimate is the config alone -- an UNDER-estimate that admits the load."""
    blobs, snap = _tree(tmp_path, {"config.json": 24})
    for rel, size in (
        ("Q4_K_M/m-00001-of-00002.gguf", 100),
        ("Q4_K_M/m-00002-of-00002.gguf", 200),
        ("Q8_0/m-00001-of-00002.gguf", 999),
    ):
        blob = blobs / rel.replace("/", "__")
        blob.write_bytes(b"\0" * size)
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).symlink_to(blob)
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(
        id="t",
        served_model_name="tiny",
        hf_repo="org/tiny",
        filename="Q4_K_M/m-00001-of-00002.gguf",
    )
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 324


@pytest.mark.asyncio
async def test_estimate_counts_mmproj(tmp_path):
    """The multimodal projector counts too; the non-selected quant does not."""
    _tree(tmp_path, {"Q4.gguf": 1000, "mmproj.gguf": 50, "Q8.gguf": 5000})
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(
        id="t",
        served_model_name="tiny",
        hf_repo="org/tiny",
        filename="Q4.gguf",
        mmproj_filename="mmproj.gguf",
    )
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 1050


@pytest.mark.asyncio
async def test_estimate_uses_newest_snapshot(tmp_path):
    """Two snapshots exist: the newest one (sorted reverse) is sized."""
    blobs, snap_a = _tree(tmp_path, {"a.gguf": 10}, snapshot="aaa")
    snap_b = tmp_path / "models--org--tiny" / "snapshots" / "bbb"
    snap_b.mkdir(parents=True)
    b = blobs / "b.gguf"
    b.write_bytes(b"\0" * 20)
    (snap_b / "b.gguf").symlink_to(b)
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(id="t", served_model_name="tiny", hf_repo="org/tiny", filename=None)
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 20


@pytest.mark.asyncio
async def test_estimate_root_beats_hub(tmp_path):
    """A repo present under both <root> and <root>/hub: the root wins."""
    _tree(tmp_path, {"x.gguf": 100}, slug="models--org--tiny")
    hub = tmp_path / "hub" / "models--org--tiny"
    hub_blob = hub / "blobs" / "x.gguf"
    hub_snap = hub / "snapshots" / "s1"
    hub_blob.parent.mkdir(parents=True)
    hub_snap.mkdir(parents=True)
    hub_blob.write_bytes(b"\0" * 4096)
    (hub_snap / "x.gguf").symlink_to(hub_blob)
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(id="t", served_model_name="tiny", hf_repo="org/tiny", filename=None)
    # The weights half of the estimate (the KV half: test_unified_admission_kv).
    assert adm._estimate(model) == 100


@pytest.mark.asyncio
async def test_estimate_missing_pinned_file_is_zero(tmp_path):
    """Fail closed: a pinned weights file that is absent is refused, not
    admitted at the size of its tokenizer."""
    _tree(tmp_path, {"config.json": 24})  # Q4.gguf is NOT in the snapshot
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(
        id="t", served_model_name="tiny", hf_repo="org/tiny", filename="Q4.gguf"
    )
    with pytest.raises(GpuConflict):
        _claim(adm, model)
    assert adm.resident() == {}


@pytest.mark.asyncio
async def test_estimate_no_snapshot_is_zero(tmp_path):
    """Blobs on disk but no snapshot directory: nothing to size, refuse."""
    blobs = tmp_path / "models--org--tiny" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "x").write_bytes(b"\0" * 1000)
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = SimpleNamespace(
        id="t", served_model_name="tiny", hf_repo="org/tiny", filename="Q4.gguf"
    )
    with pytest.raises(GpuConflict):
        _claim(adm, model)
    assert adm.resident() == {}


# --- every supervisor release path frees the bytes (a leak refuses forever) --


@pytest.mark.asyncio
async def test_supervisor_crash_unload_and_failed_spawn_all_free_unified_bytes(tmp_path):
    """Resident bytes are a ledger, so a missed release is permanent: the Mac
    would refuse every later load until the warden restarts. Engine exit
    (crash), unload and a failed spawn must each hand the bytes back."""
    adm = UnifiedMemoryAdmission(
        budget_bytes=16 * GIB,
        estimate=_sizes({"a": 8 * GIB, "b": 8 * GIB, "c": 8 * GIB, "d": 8 * GIB}),
    )
    sup = Supervisor(_settings(tmp_path), admission=adm)

    # Engine "a" exits on its own (a crash): the watcher must release it.
    crashed = MagicMock()
    crashed.pid = 10001
    crashed.returncode = None

    async def _exit():
        crashed.returncode = 139
        return 139

    crashed.wait = AsyncMock(side_effect=_exit)
    plan = LaunchPlan(argv=["vllm", "serve"], env={}, image=None, port=10001, gpu_indices=[0])
    with (
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=crashed)),
        patch("app.runtime.backends.vllm.VllmBackend.plan", return_value=plan),
    ):
        await sup.load(_model_row("a"), port=10001)
    await sup._watchers["a"]
    assert "a" not in adm.resident()

    # A spawn that raises must not keep the claim.
    with (
        patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=OSError("boom"))),
        patch("app.runtime.backends.vllm.VllmBackend.plan", return_value=plan),
        pytest.raises(OSError),
    ):
        await sup.load(_model_row("b"), port=10002)
    assert adm.resident() == {}

    # Two that fit, a third refused; unload one and the third fits.
    await _load(sup, _model_row("b"), 10002)
    await _load(sup, _model_row("c"), 10003)
    with pytest.raises(GpuConflict):
        await _load(sup, _model_row("d"), 10004)
    with patch.object(sup._driver, "terminate", new=AsyncMock()):
        await sup.unload("b", force=True)
    assert adm.resident() == {"c": 8 * GIB}
    await _load(sup, _model_row("d"), 10004)
    assert adm.resident() == {"c": 8 * GIB, "d": 8 * GIB}
    for w in list(sup._watchers.values()):
        w.cancel()


# --- the lifespan wires the profile's admission into the supervisor --------


@pytest.mark.parametrize(
    ("profile", "expected"), [("linux", GpuOwnership), ("apple", UnifiedMemoryAdmission)]
)
def test_lifespan_injects_the_profiles_admission(
    profile, expected, tmp_data_dir, migrated_db_template, monkeypatch
):
    import shutil

    from fastapi.testclient import TestClient

    from app import platform_profile
    from app.main import build_app

    monkeypatch.setattr(platform_profile, "PROFILE", profile)
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    with (
        patch(
            "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(16 * 1024))
        ),
        # The lifespan's stats sampler would spawn real nvidia-smi / ioreg
        # probes; cancelled mid-spawn at shutdown, their subprocess transports
        # outlive the TestClient loop and print "Event loop is closed" at GC.
        patch("app.runtime.stats_sampler.query_gpus", new=AsyncMock(return_value=[])),
        TestClient(app),
    ):
        assert type(app.state.supervisor.gpus) is expected
