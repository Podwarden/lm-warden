"""Repo-level coverage for the S5 tokens-v2 additions (#104):

  * TokenRepo.create accepts priority
  * TokenRepo.rotate inherits priority from the predecessor in a single
    transaction
  * TokenRepo.update is a PATCH-style sentinel-aware updater
  * the removed per-token rate limit is neither written nor inherited
  * TokenUsageRepo.add / range / totals follow the half-open-interval contract
"""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import _UNSET, TokenRepo, TokenUsageRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def _raw_rate(db, token_id: str):
    cur = await db.execute("SELECT rate_limit_tps FROM api_tokens WHERE id = ?", (token_id,))
    return (await cur.fetchone())[0]


async def test_create_stores_priority(db):
    repo = TokenRepo(db)
    await repo.create("tok-rp", "rp", "vw_" + "a" * 32, priority=7)
    row = await repo.get("tok-rp")
    assert row is not None
    assert row.priority == 7
    # Per-token rate limits were removed; the column is left in place, unused.
    assert await _raw_rate(db, "tok-rp") is None


async def test_create_defaults_priority_5(db):
    repo = TokenRepo(db)
    await repo.create("tok-def", "def", "vw_" + "b" * 32)
    row = await repo.get("tok-def")
    assert row.priority == 5


async def test_create_no_longer_takes_a_rate_limit(db):
    with pytest.raises(TypeError):
        await TokenRepo(db).create(  # type: ignore[call-arg]
            "t", "n", "vw_" + "g" * 32, rate_limit_tps=100,
        )


async def test_rotate_inherits_priority_but_not_a_legacy_rate(db):
    repo = TokenRepo(db)
    await repo.create("old", "n", "vw_" + "c" * 32, priority=8)
    # A row written before the removal can still carry a value in the column.
    await db.execute("UPDATE api_tokens SET rate_limit_tps = 250 WHERE id = 'old'")
    await db.commit()
    # #150 — rotate() no longer takes new_name; successor keeps the
    # predecessor's name and the predecessor is renamed to "n (old 1)".
    new_id, _plaintext, renamed_to = await repo.rotate(old_id="old")
    new_row = await repo.get(new_id)
    assert new_row.priority == 8
    assert new_row.name == "n"
    assert renamed_to == "n (old 1)"
    assert await _raw_rate(db, new_id) is None


async def test_update_patches_priority_only(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "d" * 32, priority=5)
    ok = await repo.update("t", priority=2)
    assert ok is True
    row = await repo.get("t")
    assert row.priority == 2
    assert row.name == "n"  # untouched by the patch


async def test_update_no_longer_takes_a_rate_limit(db):
    with pytest.raises(TypeError):
        await TokenRepo(db).update("t", rate_limit_tps=100)  # type: ignore[call-arg]


async def test_update_unset_is_noop(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "f" * 32, priority=4)
    ok = await repo.update("t", name=_UNSET, priority=_UNSET)
    assert ok is True  # noop is idempotent success
    row = await repo.get("t")
    assert row.priority == 4


async def test_update_returns_false_for_unknown_priority_patch(db):
    ok = await TokenRepo(db).update("does-not-exist", priority=2)
    assert ok is False


async def test_unset_sentinel_is_a_real_singleton():
    """A regression would split _UNSET into multiple instances, breaking
    the isinstance() check inside update."""
    from app.db.repos.tokens import _Unset
    assert _UNSET is _Unset()


async def test_token_usage_add_creates_then_increments(db):
    usage = TokenUsageRepo(db)
    await usage.add("tok-u", minute=1000, prompt_tokens=10, completion_tokens=20)
    await usage.add("tok-u", minute=1000, prompt_tokens=5, completion_tokens=3)
    requests, prompt, completion = await usage.totals("tok-u", 1000, 1001)
    assert requests == 2
    assert prompt == 15
    assert completion == 23


async def test_token_usage_range_is_ordered_and_half_open(db):
    usage = TokenUsageRepo(db)
    for m in (1000, 1001, 1005, 1010):
        await usage.add("tok-u", minute=m, prompt_tokens=1, completion_tokens=1)
    rows = await usage.range("tok-u", since_minute=1000, until_minute=1005)
    # 1005 must NOT be included (until is exclusive); 1000 IS included.
    minutes = [r[0] for r in rows]
    assert minutes == [1000, 1001]
    rows = await usage.range("tok-u", since_minute=1000, until_minute=1011)
    minutes = [r[0] for r in rows]
    assert minutes == [1000, 1001, 1005, 1010]


async def test_token_usage_totals_isolates_tokens(db):
    usage = TokenUsageRepo(db)
    await usage.add("A", 100, prompt_tokens=10, completion_tokens=20)
    await usage.add("B", 100, prompt_tokens=99, completion_tokens=99)
    a_totals = await usage.totals("A", 0, 200)
    b_totals = await usage.totals("B", 0, 200)
    assert a_totals == (1, 10, 20)
    assert b_totals == (1, 99, 99)


async def test_token_usage_totals_zero_for_empty_range(db):
    """Querying a token with no rows in the range must return all-zero,
    not raise — the UI relies on this to render 'no usage' cells."""
    totals = await TokenUsageRepo(db).totals("nope", 0, 1)
    assert totals == (0, 0, 0)
