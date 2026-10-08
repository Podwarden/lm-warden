import sqlite3

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.users import UserRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_and_get_user(db):
    repo = UserRepo(db)
    await repo.create("admin", "hashed-pw")
    u = await repo.get_by_username("admin")
    assert u is not None
    assert u.username == "admin"
    assert u.password_hash == "hashed-pw"


async def test_unique_username(db):
    repo = UserRepo(db)
    await repo.create("admin", "h1")
    with pytest.raises(sqlite3.IntegrityError):
        await repo.create("admin", "h2")


async def test_count_users(db):
    repo = UserRepo(db)
    assert await repo.count() == 0
    await repo.create("a", "h")
    await repo.create("b", "h")
    assert await repo.count() == 2
