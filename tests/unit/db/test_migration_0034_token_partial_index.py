"""Migration 0034: a partial index for ``latency_since`` (#251).

The token page polls ``GET /api/tokens/{id}/series`` every 10 s, and each call
asks when per-token history begins (``earliest_token_finished_at``). Without
this index that MIN() walks every pre-0033 row that carries no token_id.

Covers:
  * the index exists, over finished_at, partial on ``token_id IS NOT NULL``
  * SQLite's planner serves the exact production query from it -- no ANALYZE
    needed -- instead of idx_request_history_finished_at
  * the query still answers correctly around rows without a token_id
"""

import sqlite3

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats import request_history as rh

INDEX = "idx_request_history_token_finished"


async def _migrated(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


async def test_the_partial_index_exists(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(f"PRAGMA index_info({INDEX})")
        cols = [row[2] for row in await cur.fetchall()]
        cur = await db.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (INDEX,)
        )
        (sql,) = await cur.fetchone()
        cur = await db.execute("PRAGMA index_list(request_history)")
        partial = {row[1]: row[4] for row in await cur.fetchall()}
    assert cols == ["finished_at"]
    assert "WHERE token_id IS NOT NULL" in sql
    assert partial[INDEX] == 1


async def test_min_finished_at_is_served_by_the_partial_index(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("EXPLAIN QUERY PLAN " + rh.EARLIEST_TOKEN_FINISHED_SQL)
        plan = " | ".join(row[3] for row in await cur.fetchall())
    # A SEARCH through the partial index is the min() optimisation reading its
    # first entry; the old plan went through idx_request_history_finished_at
    # and skipped every token-less row on the way.
    assert f"USING COVERING INDEX {INDEX}" in plan or f"USING INDEX {INDEX}" in plan, plan
    assert "SCAN" not in plan, plan


async def test_latency_since_skips_rows_without_a_token_id(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_id, "
            "duration_s, started_iso) VALUES (?, ?, 'id-model-a', 'model-a', ?, 1.0, 'x')",
            [("pre-1", 10.0, None), ("pre-2", 20.0, None), ("t-1", 40.0, "tok-a"),
             ("t-2", 30.0, "tok-b")],
        )
        db.commit()
    async with open_db(db_path) as db:
        assert await rh.earliest_token_finished_at(db) == 30.0
