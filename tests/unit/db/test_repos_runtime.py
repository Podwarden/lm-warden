import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_upsert_and_clear_runtime(db):
    from app.db.repos.models import ModelRow

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
        )
    )
    rt = RuntimeRepo(db)
    await rt.upsert("m1", pid=1234, port=10000, started_at="2026-05-08T00:00:00Z")
    row = await rt.get("m1")
    assert row.pid == 1234
    assert row.port == 10000

    await rt.clear_all()
    assert await rt.get("m1") is None
