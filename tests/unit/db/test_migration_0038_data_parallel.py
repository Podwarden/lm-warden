"""Migration 0038: ``models.data_parallel_size`` / ``dp_affinity_enabled`` /
``dp_spill_threshold`` (#286)."""

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
async def test_columns_exist_with_not_null_defaults(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        info = {r[1]: r for r in await (await db.execute("PRAGMA table_info(models)")).fetchall()}
    # PRAGMA row: (cid, name, type, notnull, dflt_value, pk)
    assert info["data_parallel_size"][3] == 1
    assert info["data_parallel_size"][4] == "1"
    assert info["dp_affinity_enabled"][3] == 1
    assert info["dp_affinity_enabled"][4] == "1"
    assert info["dp_spill_threshold"][3] == 0
    assert info["dp_spill_threshold"][4] is None


@pytest.mark.asyncio
async def test_pre_0038_insert_decodes_to_defaults(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, "
            "status, extra_env) VALUES ('old','old','a/b','main','[0]',1,0.9,0,'[]',"
            "'registered','{}')"
        )
        await db.commit()
        row = await ModelRepo(db).get("old")
    assert row is not None
    assert (row.data_parallel_size, row.dp_affinity_enabled, row.dp_spill_threshold) == (1, 1, None)


@pytest.mark.asyncio
async def test_insert_round_trips_the_three_columns(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(
            _row(
                gpu_indices=list(range(7)),
                data_parallel_size=7,
                dp_affinity_enabled=0,
                dp_spill_threshold=12,
            )
        )
        got = await repo.get("m1")
    assert got is not None
    assert (got.data_parallel_size, got.dp_affinity_enabled, got.dp_spill_threshold) == (7, 0, 12)


@pytest.mark.asyncio
async def test_update_parallel_layout_writes_and_bumps_updated_at(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(_row(gpu_indices=list(range(4)), tensor_parallel_size=4))
        await db.execute("UPDATE models SET updated_at = '2000-01-01 00:00:00' WHERE id = 'm1'")
        await db.commit()
        await repo.update_parallel_layout(
            "m1",
            tensor_parallel_size=2,
            data_parallel_size=2,
            extra_args=["--max-num-seqs", "8"],
        )
        got = await repo.get("m1")
    assert got is not None
    assert got.tensor_parallel_size == 2
    assert got.data_parallel_size == 2
    assert got.extra_args == ["--max-num-seqs", "8"]
    assert got.updated_at != "2000-01-01 00:00:00"
