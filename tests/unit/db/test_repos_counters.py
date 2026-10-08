import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.counters import CountersRepo
from app.db.repos.models import ModelRepo, ModelRow


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        await ModelRepo(conn).insert(
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
        yield conn


async def test_increment_accumulates(db):
    c = CountersRepo(db)
    await c.increment("m1", None, 100, 50)
    await c.increment("m1", None, 30, 10)
    cur = await db.execute(
        "SELECT requests, prompt_tokens, completion_tokens FROM counters "
        "WHERE model_id = 'm1' AND token_id IS NULL"
    )
    r = await cur.fetchone()
    assert r == (2, 130, 60)
