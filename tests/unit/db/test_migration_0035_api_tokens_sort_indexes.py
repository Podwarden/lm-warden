"""Migration 0035: indexes behind the paged, sortable token list.

Covers:
  * each index exists with the columns (and name collation) the list sorts by
  * SQLite's planner -- no ANALYZE -- walks that index for the page CTE of
    every index-servable sort, in both directions, instead of sorting the table
  * the page's decorations stay index lookups: the successor through
    idx_api_tokens_rotated_from, the page's 24h usage through
    token_usage_minute's primary key, the usage_24h sort's aggregate through
    idx_token_usage_minute_minute
"""

from pathlib import Path

import aiosqlite
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos import tokens as token_repo

INDEXES = {
    "idx_api_tokens_created_id": ["created_at", "id"],
    "idx_api_tokens_last_used_id": ["last_used_at", "id"],
    "idx_api_tokens_expires_id": ["expires_at", "id"],
    "idx_api_tokens_name_id": ["name", "id"],
    "idx_api_tokens_priority_id": ["priority", "id"],
    "idx_api_tokens_prefix_id": ["prefix", "id"],
    "idx_api_tokens_rotated_from": ["rotated_from"],
    "idx_api_tokens_hidden": ["expires_at", "revoked_at", "rotated_at"],
}

SERVED_BY = {
    "name": "idx_api_tokens_name_id",
    "prefix": "idx_api_tokens_prefix_id",
    "created": "idx_api_tokens_created_id",
    "expires": "idx_api_tokens_expires_id",
    "last_used": "idx_api_tokens_last_used_id",
    "priority": "idx_api_tokens_priority_id",
}


async def _migrated(tmp_data_dir: Path) -> Path:
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


# Real-looking values: the planner's choices must not hinge on NULL binds.
PARAMS = {
    "now": "2026-09-18 00:00:00", "in30": "2026-10-18 00:00:00",
    "since": 29_000_000, "until": 29_001_441, "limit": 50, "offset": 0, "q": "%bot%",
}


async def _plan(db_path: Path, sql: str) -> list[str]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("EXPLAIN QUERY PLAN " + sql, PARAMS)
        return [row[3] for row in await cur.fetchall()]


def _page_cte(plan: list[str]) -> list[str]:
    """The detail lines of the page CTE: between MATERIALIZE page and the next
    top-level MATERIALIZE (the outer query's usage subquery)."""
    start = plan.index("MATERIALIZE page")
    end = next(i for i in range(start + 1, len(plan)) if plan[i] == "MATERIALIZE us")
    return plan[start + 1:end]


async def test_the_redundant_single_column_indexes_are_gone(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        names = {row[0] for row in await cur.fetchall()}
    assert "idx_tokens_expires_at" not in names
    assert "idx_api_tokens_prefix" not in names


async def test_the_indexes_exist(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        for name, cols in INDEXES.items():
            cur = await db.execute(f"PRAGMA index_xinfo({name})")
            info = [row for row in await cur.fetchall() if row[5]]  # key columns only
            assert [row[2] for row in info] == cols, name
        cur = await db.execute("PRAGMA index_xinfo(idx_api_tokens_name_id)")
        assert (await cur.fetchone())[4] == "NOCASE"


@pytest.mark.parametrize("sort", sorted(SERVED_BY))
@pytest.mark.parametrize("desc", [False, True])
@pytest.mark.parametrize("search", [False, True])
async def test_index_servable_sorts_walk_their_index(
    tmp_data_dir: Path, sort: str, desc: bool, search: bool,
) -> None:
    db_path = await _migrated(tmp_data_dir)
    cte = _page_cte(await _plan(db_path, token_repo.token_page_sql(sort, desc, search)))
    assert cte[0] == f"SCAN t USING INDEX {SERVED_BY[sort]}", cte
    # The whole ORDER BY, id tie-break included, comes off the index.
    assert not any("TEMP B-TREE" in line for line in cte), cte


async def test_page_decorations_are_index_lookups(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    plan = await _plan(db_path, token_repo.token_page_sql("created", True))
    joined = " | ".join(plan)
    assert "SEARCH s USING INDEX idx_api_tokens_rotated_from (rotated_from=?)" in joined
    assert (
        "SEARCH token_usage_minute USING INDEX sqlite_autoindex_token_usage_minute_1 "
        "(token_id=? AND minute>? AND minute<?)"
    ) in joined
    assert "SEARCH t USING INDEX sqlite_autoindex_api_tokens_1 (id=?)" in joined


@pytest.mark.parametrize("sort", ["created", "expires"])
async def test_the_near_expiry_filter_seeks_the_expires_range(
    tmp_data_dir: Path, sort: str,
) -> None:
    # The 30-day window is a small slice of the list: the planner should
    # seek it on the expires index rather than walk another index and filter.
    db_path = await _migrated(tmp_data_dir)
    cte = _page_cte(await _plan(db_path, token_repo.token_page_sql(sort, True, False, True)))
    assert cte[0].startswith("SEARCH t USING INDEX idx_api_tokens_expires_id (expires_at>?"), cte


async def test_usage_sort_aggregates_only_the_last_day(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    cte = _page_cte(await _plan(db_path, token_repo.token_page_sql("usage_24h", True)))
    assert any(
        "SEARCH token_usage_minute USING INDEX idx_token_usage_minute_minute" in line
        for line in cte
    ), cte


async def test_the_unsearched_counts_scan_no_rows(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    plan = await _plan(db_path, token_repo.token_counts_sql())
    # COUNT(*) walks some covering index, the near-expiry window is a covering
    # range count, and the hidden keys come through an index -- nothing scans
    # the table to test the visibility rule.
    assert any(line.startswith("SCAN api_tokens USING COVERING INDEX") for line in plan), plan
    assert any("USING COVERING INDEX" in line and "expires_at>?" in line for line in plan), plan
    assert "SCAN t USING COVERING INDEX idx_api_tokens_hidden" in plan, plan
    assert any(
        line.startswith("SEARCH t USING COVERING INDEX idx_api_tokens_hidden (expires_at>?")
        for line in plan
    ), plan
    assert not any(line.startswith("SCAN t") and "COVERING" not in line for line in plan), plan


async def test_the_hidden_index_matches_the_visibility_rule(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_api_tokens_hidden'"
        )
        (sql,) = await cur.fetchone()
    assert "WHERE revoked_at IS NOT NULL AND rotated_at IS NULL" in sql


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("expiring", [False, True])
async def test_near_expiry_count_seeks_the_expires_index(
    tmp_data_dir: Path, search: bool, expiring: bool,
) -> None:
    db_path = await _migrated(tmp_data_dir)
    plan = await _plan(db_path, token_repo.token_counts_sql(search, expiring))
    assert any(line.startswith("SEARCH t USING") and "expires_at>?" in line for line in plan), plan
