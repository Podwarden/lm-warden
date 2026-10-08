import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.templates import store
from app.templates.registry import EngineSpec, ModelTemplate


def _tpl(id="user-1"):
    return ModelTemplate(
        id=id,
        label="My Llama",
        hf_repo="meta/llama",
        hf_revision="main",
        dtype="auto",
        max_model_len=4096,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        trust_remote_code=False,
        engine=EngineSpec("cuda-stable", "0.20.0"),
        source="user",
    )


@pytest.mark.asyncio
async def test_save_and_get_user_template(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await store.save_user_template(db, _tpl())
        got = await store.get_template(db, "user-1")
        assert got.source == "user"
        assert got.engine == EngineSpec("cuda-stable", "0.20.0")


@pytest.mark.asyncio
async def test_list_merges_builtin_and_user(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await store.save_user_template(db, _tpl())
        ids = {t.id for t in await store.list_templates(db)}
        assert {"gpt-oss-20b", "user-1"} <= ids


@pytest.mark.asyncio
async def test_get_falls_back_to_builtin(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        assert (await store.get_template(db, "gpt-oss-20b")).source == "builtin"


@pytest.mark.asyncio
async def test_delete_user_template(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await store.save_user_template(db, _tpl())
        await store.delete_user_template(db, "user-1")
        assert await store.get_template(db, "user-1") is None


@pytest.mark.asyncio
async def test_delete_builtin_raises(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        with pytest.raises(ValueError):
            await store.delete_user_template(db, "gpt-oss-20b")
