"""Round-trip test for ModelRow.extra_env through insert() → get()."""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_extra_env_round_trips(db):
    repo = ModelRepo(db)
    await repo.insert(
        ModelRow(
            id="m1",
            served_model_name="gpt-oss-20b",
            hf_repo="openai/gpt-oss-20b",
            hf_revision="main",
            gpu_indices=[0, 1],
            tensor_parallel_size=2,
            dtype="bfloat16",
            max_model_len=32000,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            extra_args=[],
            extra_env={"VLLM_USE_V1": "1", "NCCL_IB_DISABLE": "1"},
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
        )
    )
    loaded = await repo.get("m1")
    assert loaded is not None
    assert loaded.extra_env == {"VLLM_USE_V1": "1", "NCCL_IB_DISABLE": "1"}


async def test_extra_env_empty_dict_round_trips(db):
    repo = ModelRepo(db)
    await repo.insert(
        ModelRow(
            id="m2",
            served_model_name="custom-model",
            hf_repo="org/repo",
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
    loaded = await repo.get("m2")
    assert loaded is not None
    assert loaded.extra_env == {}


async def test_extra_env_visible_via_list_all(db):
    repo = ModelRepo(db)
    await repo.insert(
        ModelRow(
            id="m3",
            served_model_name="model-list-test",
            hf_repo="org/repo",
            hf_revision="main",
            gpu_indices=[0],
            tensor_parallel_size=1,
            dtype=None,
            max_model_len=None,
            gpu_memory_utilization=0.9,
            trust_remote_code=False,
            extra_args=[],
            extra_env={"VLLM_MAX_NUM_SEQS": "64"},
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
        )
    )
    rows = await repo.list_all()
    m3 = next((r for r in rows if r.id == "m3"), None)
    assert m3 is not None
    assert m3.extra_env == {"VLLM_MAX_NUM_SEQS": "64"}
