"""Migration 0036: ``model_variants``, ``token_model_usage_minute`` (the
per-(key, model variant) rollup) and ``request_history.variant_id``.

Covers:
  * the tables' columns and primary keys, and the nullable history column
  * the (key set, minute range) scan the series endpoint runs is served by
    idx_token_model_usage_minute_token_minute, not a walk over every minute
    of the key through the PK
  * by_model_since's store-wide MIN(minute) reads one index entry instead of
    scanning the table on every page poll
"""

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import EARLIEST_MODEL_MINUTE_SQL


async def _migrated(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


async def _cols(db_path, table):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(f"PRAGMA table_info({table})")
        return {row[1]: (row[2], row[3], row[5]) for row in await cur.fetchall()}


async def test_the_tables_and_their_primary_keys(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    assert await _cols(db_path, "model_variants") == {
        "id": ("TEXT", 0, 1),
        "model_id": ("TEXT", 1, 0),
        "served_model_name": ("TEXT", 1, 0),
        "descriptor": ("TEXT", 1, 0),
        "first_seen": ("REAL", 1, 0),
    }
    assert (await _cols(db_path, "request_history"))["variant_id"] == ("TEXT", 0, 0)
    cols = await _cols(db_path, "token_model_usage_minute")
    assert cols == {
        "token_id": ("TEXT", 1, 1),
        "variant_id": ("TEXT", 1, 2),
        "model_id": ("TEXT", 1, 0),
        "minute": ("INTEGER", 1, 3),
        "requests": ("INTEGER", 1, 0),
        "prompt_tokens": ("INTEGER", 1, 0),
        "completion_tokens": ("INTEGER", 1, 0),
    }


async def _plan(db_path, sql, params=()):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("EXPLAIN QUERY PLAN " + sql, params)
        return " | ".join(row[3] for row in await cur.fetchall())


async def test_a_key_range_scan_uses_the_token_minute_index(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    plan = await _plan(
        db_path,
        "SELECT model_id, SUM(prompt_tokens) FROM token_model_usage_minute "
        "WHERE token_id IN (?, ?) AND minute >= ? AND minute < ? GROUP BY model_id",
        ("a", "b", 1, 2),
    )
    # Not "USING PRIMARY KEY (token_id=?)", which reads every minute of the key.
    assert "USING INDEX idx_token_model_usage_minute_token_minute (token_id=? AND minute>" \
        in plan, plan


async def test_earliest_minute_reads_the_minute_index(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    plan = await _plan(db_path, EARLIEST_MODEL_MINUTE_SQL)
    assert "idx_token_model_usage_minute_minute" in plan, plan
    assert "SCAN" not in plan, plan
