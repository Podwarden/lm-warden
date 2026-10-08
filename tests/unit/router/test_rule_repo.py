"""RouterRuleRepo (#287)."""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.router_rules import RouterRuleRepo


@pytest.fixture
async def repo(tmp_path):
    async with open_db(tmp_path / "t.db") as db:
        await apply_migrations(db)
        yield RouterRuleRepo(db)


async def _mk(repo, pattern="claude-haiku*", **kw):
    args = dict(
        target_model_id="m1", enabled=True, fallback=True, strip_thinking=True, min_max_tokens=0
    )
    args.update(kw)
    return await repo.create(pattern, **args)


async def test_create_appends_positions(repo):
    a, b, c = await _mk(repo, "a-*"), await _mk(repo, "b-*"), await _mk(repo, "c-*")
    assert [a.position, b.position, c.position] == [0, 1, 2]
    assert [r.pattern for r in await repo.list_all()] == ["a-*", "b-*", "c-*"]
    assert a.enabled is True and a.fallback is True and a.strip_thinking is True
    assert a.min_max_tokens == 0 and len(a.id) == 32


async def test_get_and_missing(repo):
    a = await _mk(repo)
    assert (await repo.get(a.id)) == a
    assert await repo.get("nope") is None


async def test_update_changes_one_field_and_bumps_updated_at(repo):
    a = await _mk(repo)
    await repo.db.execute("UPDATE router_rules SET updated_at = '2000-01-01 00:00:00'")
    await repo.db.commit()
    b = await repo.update(a.id, enabled=False)
    assert b is not None
    assert b.enabled is False
    assert (b.pattern, b.fallback, b.strip_thinking, b.min_max_tokens) == (
        a.pattern,
        a.fallback,
        a.strip_thinking,
        a.min_max_tokens,
    )
    assert b.updated_at != "2000-01-01 00:00:00"
    assert b.created_at == a.created_at


async def test_update_unknown_returns_none(repo):
    assert await repo.update("nope", enabled=False) is None


async def test_update_rejects_unknown_field(repo):
    a = await _mk(repo)
    with pytest.raises(ValueError):
        await repo.update(a.id, position=9)


async def test_delete(repo):
    a = await _mk(repo)
    assert await repo.delete(a.id) is True
    assert await repo.delete(a.id) is False
    assert await repo.list_all() == []


async def test_position_after_delete_is_max_plus_one(repo):
    a, b = await _mk(repo, "a-*"), await _mk(repo, "b-*")
    await repo.delete(a.id)
    c = await _mk(repo, "c-*")
    assert c.position == b.position + 1


async def test_reorder_permutation(repo):
    a, b, c = await _mk(repo, "a-*"), await _mk(repo, "b-*"), await _mk(repo, "c-*")
    await repo.reorder([c.id, a.id, b.id])
    rows = await repo.list_all()
    assert [r.id for r in rows] == [c.id, a.id, b.id]
    assert [r.position for r in rows] == [0, 1, 2]


async def test_reorder_rejects_non_permutation(repo):
    a, b = await _mk(repo, "a-*"), await _mk(repo, "b-*")
    for bad in ([a.id], [a.id, b.id, "x"], [a.id, a.id], [a.id, "x"]):
        with pytest.raises(ValueError):
            await repo.reorder(bad)
    assert [r.id for r in await repo.list_all()] == [a.id, b.id]
