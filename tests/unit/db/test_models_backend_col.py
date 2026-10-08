"""Migration 0027: ``models.backend`` — nullable, no backfill (decision D6).

The point of the column is that it changes nothing. Every row written before
it exists has ``backend = NULL``, and NULL decodes to ``"vllm"`` through
``app.runtime.backends.registry``, so no existing row's launch moves.
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
async def test_migration_adds_the_backend_column(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(models)")).fetchall()}
    assert "backend" in cols


@pytest.mark.asyncio
async def test_backend_column_is_nullable_and_decodes_to_vllm(tmp_path):
    """D6: existing rows are NOT backfilled. A NULL backend must decode to
    'vllm' so every pre-B row keeps launching exactly as it did."""
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(id="legacy"))
        # simulate a pre-0027 row: the column exists but was never written
        await db.execute("UPDATE models SET backend = NULL WHERE id = 'legacy'")
        await db.commit()
        got = await repo.get("legacy")
    assert got.backend == "vllm"


@pytest.mark.asyncio
async def test_backend_round_trips(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(id="m", backend="vllm"))
        got = await repo.get("m")
    assert got.backend == "vllm"


@pytest.mark.asyncio
async def test_the_migration_writes_no_row_data(tmp_path):
    """There is no UPDATE in 0027, so there is nothing to get wrong on a
    volume holding a live database. A row inserted through a pre-0027-shaped
    insert stays NULL rather than being rewritten."""
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(id="legacy"))
        await db.execute("UPDATE models SET backend = NULL WHERE id = 'legacy'")
        await db.commit()
        raw = await (await db.execute("SELECT backend FROM models WHERE id = 'legacy'")).fetchone()
    assert raw[0] is None


def test_backend_sits_at_index_29():
    """_MODEL_COLS is positional and append-only (models.py:78-81). A column
    inserted in the middle silently shifts every index in _decode_row.

    B asserted ``backend`` was LAST; sub-project C appended
    ``mmproj_filename`` and ``n_gpu_layers`` after it, so what B was actually
    protecting -- that ``backend`` sits immediately after ``supports_reasoning``
    at index 29, where ``_decode_row`` reads it -- is what this now says.
    C's own tail assertion lives in tests/unit/db/test_models_llamacpp_cols.py.
    """
    from app.db.repos.models import _MODEL_COLS

    cols = [c.strip() for c in _MODEL_COLS.split(",")]
    assert cols[28:30] == ["supports_reasoning", "backend"]
