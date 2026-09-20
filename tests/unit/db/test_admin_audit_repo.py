"""admin_audit repository (migration 0037): record, newest-first paging with a
`before` cursor that never splits a timestamp (and never claims a further
page when the tie-break extension has actually exhausted the table), and the
pruner's age cut + per-token row cap (#257) + global row cap."""

import aiosqlite
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.admin_audit import PER_TOKEN_CAP_SQL, AdminAuditRepo, prune


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def _rec(repo: AdminAuditRepo, ts: float, token_id: str = "a1") -> None:
    await repo.record(
        ts=ts, token_id=token_id, username="admin", method="GET",
        path="/api/models/{model_id}", status=200, duration_ms=5,
        client_ip="127.0.0.1", peer_ip="10.0.0.9",
    )


async def test_a_page_is_newest_first_and_one_tokens_only(db):
    repo = AdminAuditRepo(db)
    for ts in (1.0, 2.0, 3.0):
        await _rec(repo, ts)
    await _rec(repo, 4.0, token_id="other")
    items, next_before = await repo.page("a1", limit=10, before=None)
    assert [r.ts for r in items] == [3.0, 2.0, 1.0]
    assert next_before is None
    first = items[0]
    assert (first.username, first.method, first.path, first.status, first.duration_ms,
            first.client_ip, first.peer_ip) == (
        "admin", "GET", "/api/models/{model_id}", 200, 5, "127.0.0.1", "10.0.0.9",
    )


async def test_before_pages_backwards(db):
    repo = AdminAuditRepo(db)
    for ts in (1.0, 2.0, 3.0, 4.0, 5.0):
        await _rec(repo, ts)
    items, nb = await repo.page("a1", limit=2, before=None)
    assert ([r.ts for r in items], nb) == ([5.0, 4.0], 4.0)
    items, nb = await repo.page("a1", limit=2, before=nb)
    assert ([r.ts for r in items], nb) == ([3.0, 2.0], 2.0)
    items, nb = await repo.page("a1", limit=2, before=nb)
    assert ([r.ts for r in items], nb) == ([1.0], None)


async def test_a_page_never_splits_rows_that_share_a_timestamp(db):
    repo = AdminAuditRepo(db)
    for ts in (5.0, 7.0, 7.0, 7.0, 9.0):
        await _rec(repo, ts)
    items, nb = await repo.page("a1", limit=2, before=None)
    assert ([r.ts for r in items], nb) == ([9.0, 7.0, 7.0, 7.0], 7.0)
    items, nb = await repo.page("a1", limit=2, before=nb)
    assert ([r.ts for r in items], nb) == ([5.0], None)


async def test_a_tied_boundary_that_exhausts_the_table_has_no_next_page(db):
    """Ruling: next_before must be set ONLY when a further row exists beyond
    the page. A naive "page is full" check would set it here even though the
    tie-break extension already pulled in every remaining row."""
    repo = AdminAuditRepo(db)
    for ts in (7.0, 7.0, 7.0):
        await _rec(repo, ts)
    items, nb = await repo.page("a1", limit=2, before=None)
    assert [r.ts for r in items] == [7.0, 7.0, 7.0]
    assert nb is None


async def test_prune_cuts_by_age_then_caps_the_row_count(db):
    repo = AdminAuditRepo(db)
    for ts in (1.0, 2.0, 3.0, 4.0, 5.0):
        await _rec(repo, ts)
    removed = await prune(db, cutoff=2.5, max_rows=2, per_token_max_rows=0)
    await db.commit()
    assert removed == 3  # 1.0 and 2.0 by age, then 3.0 over the cap
    items, _ = await repo.page("a1", limit=10, before=None)
    assert [r.ts for r in items] == [5.0, 4.0]


async def test_prune_per_token_cap_evicts_only_the_noisy_tokens_oldest_rows(db):
    """Issue #257: a token over PER_TOKEN_MAX_ROWS loses only its own oldest
    rows. A second, quiet token's history is untouched -- the whole point of
    the per-token cap is that a flooding token can no longer evict rows that
    are not its own."""
    repo = AdminAuditRepo(db)
    for ts in (1.0, 2.0, 3.0, 4.0, 5.0):
        await _rec(repo, ts, token_id="noisy")
    for ts in (10.0, 20.0):
        await _rec(repo, ts, token_id="quiet")
    removed = await prune(db, cutoff=0.0, max_rows=0, per_token_max_rows=3)
    await db.commit()
    assert removed == 2  # noisy's two oldest (1.0, 2.0)

    noisy, _ = await repo.page("noisy", limit=10, before=None)
    assert [r.ts for r in noisy] == [5.0, 4.0, 3.0]  # newest three survive
    quiet, _ = await repo.page("quiet", limit=10, before=None)
    assert [r.ts for r in quiet] == [20.0, 10.0]  # untouched


async def test_prune_per_token_cap_leaves_a_token_under_the_cap_alone(db):
    repo = AdminAuditRepo(db)
    for ts in (1.0, 2.0):
        await _rec(repo, ts, token_id="a1")
    removed = await prune(db, cutoff=0.0, max_rows=0, per_token_max_rows=20_000)
    await db.commit()
    assert removed == 0
    items, _ = await repo.page("a1", limit=10, before=None)
    assert [r.ts for r in items] == [2.0, 1.0]


async def test_prune_global_cap_still_applies_across_tokens_each_under_their_own_cap(db):
    """The per-token cap does not replace the global backstop: three tokens
    each well under PER_TOKEN_MAX_ROWS can still, together, exceed the global
    cap, and the oldest rows overall must go regardless of which token wrote
    them."""
    repo = AdminAuditRepo(db)
    for ts in (1.0, 2.0, 3.0):
        await _rec(repo, ts, token_id="a")
    for ts in (4.0, 5.0, 6.0):
        await _rec(repo, ts, token_id="b")
    for ts in (7.0, 8.0, 9.0):
        await _rec(repo, ts, token_id="c")
    removed = await prune(db, cutoff=0.0, max_rows=5, per_token_max_rows=10)
    await db.commit()
    assert removed == 4  # 1.0, 2.0, 3.0, 4.0 -- oldest overall, cross-token
    cur = await db.execute("SELECT ts FROM admin_audit ORDER BY ts DESC")
    assert [r[0] for r in await cur.fetchall()] == [9.0, 8.0, 7.0, 6.0, 5.0]


async def test_prune_runs_age_then_per_token_then_global_and_sums_the_count(db):
    """All three passes in one call: age first, then the per-token cap, then
    the global cap as a backstop -- and the returned count is their sum."""
    repo = AdminAuditRepo(db)
    # 'a': one stale row (goes by age), then 4 fresh -- per-token cap 2 trims
    # it to its newest 2.
    await _rec(repo, 1.0, token_id="a")  # stale, by age
    for ts in (100.0, 101.0, 102.0, 103.0):
        await _rec(repo, ts, token_id="a")
    # 'b': 3 fresh rows, also over the per-token cap of 2.
    for ts in (200.0, 201.0, 202.0):
        await _rec(repo, ts, token_id="b")
    # age cutoff = 50 removes the ts=1.0 row (1); per_token cap=2 then trims
    # 'a' from 4 to 2 (removes 100.0, 101.0) and 'b' from 3 to 2 (removes
    # 200.0) -- 3 more; the 4 survivors are already at the global cap=4, so
    # the backstop pass removes nothing.
    removed = await prune(db, cutoff=50.0, max_rows=4, per_token_max_rows=2)
    await db.commit()
    assert removed == 1 + 3 + 0
    cur = await db.execute("SELECT token_id, ts FROM admin_audit ORDER BY ts DESC")
    assert await cur.fetchall() == [
        ("b", 202.0), ("b", 201.0), ("a", 103.0), ("a", 102.0),
    ]


async def test_the_per_token_cap_delete_uses_the_token_ts_index_not_a_table_scan(
    tmp_data_dir,
):
    """Index-friendliness per #257: the per-token cap's DELETE must walk
    idx_admin_audit_token_ts as a single scan (one SQL round for every
    token_id, not a query per token) rather than scan the table unindexed."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as conn:
        await apply_migrations(conn)
    async with aiosqlite.connect(db_path) as raw:
        cur = await raw.execute("EXPLAIN QUERY PLAN " + PER_TOKEN_CAP_SQL, {"cap": 20_000})
        plan = [row[3] for row in await cur.fetchall()]
    assert any(
        "USING COVERING INDEX idx_admin_audit_token_ts" in line for line in plan
    ), plan
    # No unindexed full scan of the table anywhere in the plan.
    assert not any(
        line.startswith("SCAN admin_audit") and "INDEX" not in line for line in plan
    ), plan
