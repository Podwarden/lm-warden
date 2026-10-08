import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


@pytest.mark.asyncio
async def test_migration_adds_engine_cols_and_tables(tmp_path):
    db_path = str(tmp_path / "t.db")
    async with open_db(db_path) as db:
        await apply_migrations(db)
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(models)")).fetchall()}
        assert {"engine_channel", "engine_vllm_version", "engine_image"} <= cols
        tables = {
            r[0]
            for r in await (
                await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        assert {"engine_templates", "stack_attempts"} <= tables


@pytest.mark.asyncio
async def test_model_row_engine_roundtrip(tmp_path):
    db_path = str(tmp_path / "t.db")
    async with open_db(db_path) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(
            ModelRow(
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
                engine_channel="cuda-stable",
                engine_vllm_version="0.20.0",
                engine_image=None,
            )
        )
        got = await repo.get("m1")
        assert got.engine_channel == "cuda-stable"
        assert got.engine_vllm_version == "0.20.0"
        assert got.engine_image is None
