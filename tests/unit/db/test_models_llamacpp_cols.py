"""Migration 0028: ``models.mmproj_filename`` and ``models.n_gpu_layers``.

The two knobs from design spec §6.6's field-by-field table that llama.cpp needs
and no existing column carries. Both nullable, appended last, no backfill --
the same shape as B's 0027, for the same reason.

There is deliberately NO ``split_mode`` column (decision D4): the GPU count is
already ``tensor_parallel_size == len(gpu_indices)``, and which llama.cpp flag
that becomes is ``LlamaCppBackend.plan()``'s job, not the schema's.
"""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="x",
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="registered",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


@pytest.mark.asyncio
async def test_migration_adds_both_columns(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(models)")).fetchall()}
    assert "mmproj_filename" in cols
    assert "n_gpu_layers" in cols


@pytest.mark.asyncio
async def test_llamacpp_columns_round_trip(tmp_path):
    """A llama.cpp row keeps its projector and its offload setting."""
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(
            _row(
                id="m-gguf",
                backend="llamacpp",
                filename="Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf",
                mmproj_filename="mmproj-Qwen3.8-27B-BF16.gguf",
                n_gpu_layers=None,
            )
        )
        got = await repo.get("m-gguf")
    assert got.mmproj_filename == "mmproj-Qwen3.8-27B-BF16.gguf"
    assert got.n_gpu_layers is None


@pytest.mark.asyncio
async def test_explicit_n_gpu_layers_round_trips(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(id="partial", backend="llamacpp", n_gpu_layers=20))
        got = await repo.get("partial")
    assert got.n_gpu_layers == 20


@pytest.mark.asyncio
async def test_pre_0028_rows_decode_with_nulls(tmp_path):
    """No backfill: every existing row reads back mmproj_filename=None and
    n_gpu_layers=None, which the llama.cpp argv builder treats as 'omit the
    flag and let mainline pick', not as a value."""
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(id="legacy"))
        got = await repo.get("legacy")
    assert got.mmproj_filename is None
    assert got.n_gpu_layers is None


@pytest.mark.asyncio
async def test_the_migration_writes_no_row_data(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(id="legacy"))
        raw = await (
            await db.execute("SELECT mmproj_filename, n_gpu_layers FROM models WHERE id = 'legacy'")
        ).fetchone()
    assert raw == (None, None)


def test_new_columns_are_appended_last():
    """app/db/repos/models.py:78-81 -- _MODEL_COLS and _decode_row are
    POSITIONAL and append-only. A reorder silently shifts every decoded field
    by one, which is a data-corruption bug that no other test would catch."""
    from app.db.repos.models import _MODEL_COLS

    cols = [c.strip() for c in _MODEL_COLS.split(",")]
    # 0038 (#286) appended three more after the llama.cpp pair.
    assert cols[-8:-5] == ["backend", "mmproj_filename", "n_gpu_layers"]
    # 0039 (#286 review) appended the boot-reconcile notice pair.
    assert cols[-5:-2] == ["data_parallel_size", "dp_affinity_enabled", "dp_spill_threshold"]
    assert cols[-2:] == ["layout_notice_level", "layout_notice"]
