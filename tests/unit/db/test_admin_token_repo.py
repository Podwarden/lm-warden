"""TokenRepo for admin tokens (migration 0037):

  * create() records scope and created_by; inference rows keep NULL
  * rotate() of an admin row mints an admin successor -- vwa_ secret, same
    owner -- and "(old N)" numbering is per scope
  * list_all() is inference-only unless asked; list_admin() is the Settings
    list: live first, then tokens dead for at most 30 days
"""

from datetime import timedelta

import aiosqlite
import pytest

from app.auth.bearer import ADMIN_TOKEN_PREFIX, generate_admin_token, generate_bearer_token
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo, sqlite_utc_in, sqlite_utc_now


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def _admin(repo: TokenRepo, tid: str = "a1", name: str = "ci", days: int = 90) -> str:
    plaintext = generate_admin_token()
    await repo.create(tid, name, plaintext, scope="admin", expires_in_days=days, created_by="admin")
    return plaintext


async def _set(db: aiosqlite.Connection, tid: str, **cols: str | None) -> None:
    sets = ", ".join(f"{k} = ?" for k in cols)
    await db.execute(f"UPDATE api_tokens SET {sets} WHERE id = ?", (*cols.values(), tid))
    await db.commit()


async def test_create_records_scope_owner_and_prefix(db):
    repo = TokenRepo(db)
    plaintext = await _admin(repo)
    row = await repo.get("a1")
    assert row is not None
    assert (row.scope, row.created_by) == ("admin", "admin")
    assert row.prefix == plaintext[:8] and row.prefix.startswith(ADMIN_TOKEN_PREFIX)
    found = await repo.find_by_plaintext(plaintext)
    assert found is not None and found.id == "a1"


async def test_inference_rows_default_to_no_owner(db):
    repo = TokenRepo(db)
    await repo.create("i1", "bot", generate_bearer_token())
    row = await repo.get("i1")
    assert row is not None
    assert (row.scope, row.created_by) == ("inference", None)


async def test_rotating_an_admin_token_mints_an_admin_successor(db):
    repo = TokenRepo(db)
    await _admin(repo)
    new_id, plaintext, renamed_to = await repo.rotate("a1", grace_hours=1)
    assert plaintext.startswith(ADMIN_TOKEN_PREFIX)
    new = await repo.get(new_id)
    assert new is not None
    assert (new.scope, new.created_by, new.rotated_from, new.name) == ("admin", "admin", "a1", "ci")
    assert renamed_to == "ci (old 1)"
    old = await repo.get("a1")
    assert old is not None and old.rotated_at is not None
    assert old.revoked_at is not None and old.revoked_at > sqlite_utc_now()


async def test_rotating_an_inference_token_is_unchanged(db):
    repo = TokenRepo(db)
    await repo.create("i1", "bot", generate_bearer_token())
    new_id, plaintext, _ = await repo.rotate("i1")
    assert plaintext.startswith("vw_") and not plaintext.startswith(ADMIN_TOKEN_PREFIX)
    new = await repo.get(new_id)
    assert new is not None and (new.scope, new.created_by) == ("inference", None)


async def test_old_n_numbering_is_per_scope(db):
    repo = TokenRepo(db)
    await repo.create("i1", "ci", generate_bearer_token())
    await repo.rotate("i1")  # the inference "ci (old 1)"
    await _admin(repo, "a1", "ci")
    _, _, renamed_to = await repo.rotate("a1")
    assert renamed_to == "ci (old 1)"


async def test_list_all_is_inference_only_unless_asked(db):
    repo = TokenRepo(db)
    await repo.create("i1", "bot", generate_bearer_token())
    await _admin(repo)
    assert [r.id for r in await repo.list_all()] == ["i1"]
    assert [r.id for r in await repo.list_all(scope="admin")] == ["a1"]


async def test_list_admin_puts_live_tokens_first_and_drops_long_dead_ones(db):
    repo = TokenRepo(db)
    for tid in ("live-old", "live-new", "grace", "revoked-recent", "expired-recent",
                "revoked-long-ago", "dead-twice"):
        await _admin(repo, tid, tid)
    await repo.create("i1", "bot", generate_bearer_token())
    ago = lambda days: sqlite_utc_in(timedelta(days=-days))  # noqa: E731
    await _set(db, "live-old", created_at=ago(5))
    await _set(db, "live-new", created_at=ago(1))
    # A rotated predecessor inside its grace window still works: live.
    await _set(db, "grace", created_at=ago(3), rotated_at=ago(0),
               revoked_at=sqlite_utc_in(timedelta(hours=1)))
    await _set(db, "revoked-recent", created_at=ago(2), revoked_at=ago(2))
    await _set(db, "expired-recent", created_at=ago(4), expires_at=ago(1))
    await _set(db, "revoked-long-ago", revoked_at=ago(40))
    # Dead since it was revoked 40 days ago, whatever its later expiry says.
    await _set(db, "dead-twice", revoked_at=ago(40), expires_at=ago(1))

    rows = await repo.list_admin(now=sqlite_utc_now(), since=ago(30))

    assert [r.id for r in rows] == [
        "live-new", "grace", "live-old", "revoked-recent", "expired-recent",
    ]


def test_seed_admin_token_round_trips(migrated_db_template, tmp_path):
    import asyncio
    import shutil

    from tests.conftest import seed_admin_token

    db_path = tmp_path / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, db_path)
    tid, plaintext = seed_admin_token(db_path, name="helper")

    async def lookup():
        async with open_db(db_path) as conn:
            return await TokenRepo(conn).find_by_plaintext(plaintext)

    row = asyncio.run(lookup())
    assert row is not None
    assert (row.id, row.scope, row.created_by, row.name) == (tid, "admin", "admin", "helper")
