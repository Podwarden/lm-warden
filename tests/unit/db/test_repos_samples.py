import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.samples import SamplesRepo


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


async def test_model_samples_accumulate_per_minute(db):
    s = SamplesRepo(db)
    await s.add_model_sample("m1", 100, 5, 200, 100)
    await s.add_model_sample("m1", 100, 3, 50, 25)
    await s.add_model_sample("m1", 101, 1, 10, 5)
    rows = await s.model_samples_since("m1", 100)
    assert rows == [
        {"minute": 100, "requests": 8, "prompt_tokens": 250, "completion_tokens": 125},
        {"minute": 101, "requests": 1, "prompt_tokens": 10, "completion_tokens": 5},
    ]


async def test_model_samples_sum_cached_and_measured(db):
    s = SamplesRepo(db)
    await s.add_model_sample("m1", 100, 1, 200, 10, delta_cached=150)
    await s.add_model_sample("m1", 100, 1, 100, 10, delta_cached=0)  # measured zero
    await s.add_model_sample("m1", 100, 1, 50, 10)  # engine did not report it
    cur = await db.execute(
        "SELECT requests, prompt_tokens, cached_tokens, cached_measured_requests "
        "FROM model_samples WHERE model_id='m1' AND minute=100"
    )
    assert await cur.fetchone() == (3, 350, 150, 2)


async def test_gpu_samples_replace_per_minute(db):
    s = SamplesRepo(db)
    await s.add_gpu_sample(0, 100, 50, 4096, 24576)
    await s.add_gpu_sample(0, 100, 80, 8192, 24576)  # overwrites
    rows = await s.gpu_samples_since(100)
    assert len(rows) == 1
    assert rows[0]["utilization_pct"] == 80
    assert rows[0]["memory_used_mib"] == 8192


async def test_prune_older_than(db):
    s = SamplesRepo(db)
    await s.add_model_sample("m1", 50, 1, 10, 5)
    await s.add_model_sample("m1", 100, 1, 10, 5)
    await s.prune_older_than(80)
    rows = await s.model_samples_since("m1", 0)
    assert len(rows) == 1
    assert rows[0]["minute"] == 100
