import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo, hash_token


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_lookup_revoke(db):
    repo = TokenRepo(db)
    await repo.create("tok1", "ci-bot", "vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    found = await repo.find_by_plaintext("vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert found.id == "tok1"
    assert found.prefix == "vw_aaaaa"

    await repo.revoke("tok1")
    found = await repo.find_by_plaintext("vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert found.revoked_at is not None


async def test_only_hash_stored(db):
    plaintext = "vw_secret_token_value_xyz"
    await TokenRepo(db).create("tok2", "n", plaintext)
    cur = await db.execute("SELECT hash FROM api_tokens WHERE id = ?", ("tok2",))
    (stored,) = await cur.fetchone()
    assert stored == hash_token(plaintext)
    assert plaintext not in stored


async def _last_used(db, token_id):
    cur = await db.execute("SELECT last_used_at FROM api_tokens WHERE id = ?", (token_id,))
    (value,) = await cur.fetchone()
    return value


async def _set_last_used(db, token_id, modifier):
    await db.execute(
        "UPDATE api_tokens SET last_used_at = datetime('now', ?) WHERE id = ?",
        (modifier, token_id),
    )
    await db.commit()


async def test_touch_last_used_stamps_a_never_used_key(db):
    repo = TokenRepo(db)
    await repo.create("t1", "n", "vw_touch_one_aaaaaaaaaaaaaaaaaaaaaa")
    assert await _last_used(db, "t1") is None
    await repo.touch_last_used("t1")
    assert await _last_used(db, "t1") is not None


async def test_touch_last_used_leaves_a_recent_stamp_alone(db):
    # last_used_at is indexed (0035): rewriting it on every request costs a
    # page write + WAL fsync each, so a stamp younger than a minute stays.
    repo = TokenRepo(db)
    await repo.create("t1", "n", "vw_touch_two_aaaaaaaaaaaaaaaaaaaaaa")
    await _set_last_used(db, "t1", "-30 seconds")
    before = await _last_used(db, "t1")
    await repo.touch_last_used("t1")
    assert await _last_used(db, "t1") == before
    cur = await db.execute("SELECT total_changes()")
    changes_before = (await cur.fetchone())[0]
    await repo.touch_last_used("t1")
    cur = await db.execute("SELECT total_changes()")
    assert (await cur.fetchone())[0] == changes_before  # no row written at all


async def test_touch_last_used_refreshes_a_stamp_older_than_a_minute(db):
    repo = TokenRepo(db)
    await repo.create("t1", "n", "vw_touch_three_aaaaaaaaaaaaaaaaaaaa")
    await _set_last_used(db, "t1", "-61 seconds")
    before = await _last_used(db, "t1")
    await repo.touch_last_used("t1")
    after = await _last_used(db, "t1")
    assert after > before
