"""TokenRepo additions for the token details page (spec 2026-09-18 §3.2-3.3).

  * update() -- rename, and pause semantics: the first pause time is kept,
    resume clears it, _UNSET leaves every field alone
  * rotate() of a paused key -- the successor is a fresh key and starts
    unpaused, the predecessor keeps its pause
  * lineage() -- the rotation chain, oldest first, from any member
"""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_update_renames(db):
    repo = TokenRepo(db)
    await repo.create("t", "before", "vw_" + "a" * 32)
    assert await repo.update("t", name="after") is True
    assert (await repo.get("t")).name == "after"


async def test_pause_sets_paused_at_and_resume_clears_it(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "b" * 32)
    assert (await repo.get("t")).paused_at is None
    assert await repo.update("t", paused=True) is True
    assert (await repo.get("t")).paused_at is not None
    assert await repo.update("t", paused=False) is True
    assert (await repo.get("t")).paused_at is None


async def test_a_repeated_pause_keeps_the_original_time(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "c" * 32)
    await repo.update("t", paused=True)
    # Pin a recognisable first-pause time so the assertion cannot pass by
    # both calls landing in the same second.
    await db.execute("UPDATE api_tokens SET paused_at = '2026-01-01 00:00:00' WHERE id = 't'")
    await db.commit()
    await repo.update("t", paused=True)
    assert (await repo.get("t")).paused_at == "2026-01-01 00:00:00"


async def test_update_with_nothing_set_touches_nothing(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "d" * 32, priority=4)
    assert await repo.update("t") is True
    row = await repo.get("t")
    assert (row.name, row.priority, row.paused_at) == ("n", 4, None)


async def test_update_returns_false_for_unknown_token(db):
    assert await TokenRepo(db).update("nope", paused=True) is False


async def test_rotating_a_paused_key_gives_an_unpaused_successor(db):
    repo = TokenRepo(db)
    await repo.create("old", "n", "vw_" + "e" * 32)
    await repo.update("old", paused=True)
    new_id, _plaintext, _renamed = await repo.rotate(old_id="old")
    assert (await repo.get(new_id)).paused_at is None
    assert (await repo.get("old")).paused_at is not None


async def test_lineage_of_a_lone_key_is_itself(db):
    repo = TokenRepo(db)
    await repo.create("solo", "n", "vw_" + "f" * 32)
    assert [r.id for r in await repo.lineage("solo")] == ["solo"]


async def test_lineage_of_an_unknown_id_is_empty(db):
    assert await TokenRepo(db).lineage("nope") == []


async def test_lineage_is_oldest_first_from_any_member(db):
    repo = TokenRepo(db)
    await repo.create("a", "n", "vw_" + "g" * 32)
    b, _, _ = await repo.rotate(old_id="a")
    c, _, _ = await repo.rotate(old_id=b)
    for member in ("a", b, c):
        assert [r.id for r in await repo.lineage(member)] == ["a", b, c]


async def test_lineage_stops_at_a_deleted_predecessor(db):
    repo = TokenRepo(db)
    await repo.create("a", "n", "vw_" + "h" * 32)
    b, _, _ = await repo.rotate(old_id="a")
    c, _, _ = await repo.rotate(old_id=b)
    assert await repo.delete("a") is True
    assert [r.id for r in await repo.lineage(c)] == [b, c]


async def test_lineage_terminates_on_a_hand_edited_cycle(db):
    repo = TokenRepo(db)
    await repo.create("a", "n", "vw_" + "i" * 32)
    b, _, _ = await repo.rotate(old_id="a")
    await db.execute("UPDATE api_tokens SET rotated_from = ? WHERE id = 'a'", (b,))
    await db.commit()
    ids = [r.id for r in await repo.lineage(b)]
    assert sorted(ids) == sorted(["a", b])
    assert len(ids) == len(set(ids))
