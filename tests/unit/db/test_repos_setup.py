import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.setup import SetupRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_initial_state_welcome(db):
    s = await SetupRepo(db).get()
    assert s.step == "welcome"
    assert s.draft == {}


async def test_merge_draft_accumulates(db):
    repo = SetupRepo(db)
    await repo.merge_draft(allowed_gpu_indices=[1, 2])
    await repo.merge_draft(hf_token_present=True)
    s = await repo.get()
    assert s.draft == {"allowed_gpu_indices": [1, 2], "hf_token_present": True}


async def test_set_step_done_marks_done(db):
    repo = SetupRepo(db)
    assert not await repo.is_done()
    await repo.set_step("done")
    assert await repo.is_done()
