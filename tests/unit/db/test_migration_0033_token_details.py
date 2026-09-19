"""Migration 0033: api_tokens.paused_at and request_history.token_id.

The token details page pauses keys and charts per-key timings. Both columns are
nullable with no backfill: a key that was never paused reads NULL, and a
history row written before this migration cannot be attributed to a key id,
because names are reused across rotations (spec 2026-09-18 §2).

Covers:
  * both columns appear, TEXT and NULLABLE, plus the (token_id, finished_at)
    index the series endpoint reads through
  * a database at 0032 with a token and a history row upgrades cleanly,
    TokenRow still unpacks the old row, and the old history row reads NULL
"""

import sqlite3

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo
from tests.unit.db.test_migration_0031_request_history import _apply_through

TYPE_IDX, NOTNULL_IDX = 2, 3


async def test_columns_and_index_exist(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(api_tokens)")
        tok_cols = {row[1]: row for row in await cur.fetchall()}
        cur = await db.execute("PRAGMA table_info(request_history)")
        rh_cols = {row[1]: row for row in await cur.fetchall()}
        cur = await db.execute("PRAGMA index_info(idx_request_history_token)")
        idx_cols = [row[2] for row in await cur.fetchall()]
    assert tok_cols["paused_at"][TYPE_IDX] == "TEXT"
    assert tok_cols["paused_at"][NOTNULL_IDX] == 0
    assert rh_cols["token_id"][TYPE_IDX] == "TEXT"
    assert rh_cols["token_id"][NOTNULL_IDX] == 0
    assert idx_cols == ["token_id", "finished_at"]


async def test_upgrade_from_0032_keeps_old_rows_readable(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    await _apply_through(db_path, "0032_request_history_queued.sql")
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('tok-old', 'ip-harness', 'vw_abcde', 'h-old', 'inference')"
        )
        db.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_name, "
            "duration_s, started_iso) VALUES ('r-old', 100.0, 'm-1', 'served-1', "
            "'ip-harness', 2.0, 'x')"
        )
        db.commit()

    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # idempotent
        row = await TokenRepo(db).get("tok-old")

    assert row is not None
    assert row.name == "ip-harness"
    assert row.paused_at is None
    with sqlite3.connect(db_path) as db:
        got = db.execute(
            "SELECT token_name, token_id FROM request_history WHERE id = 'r-old'"
        ).fetchone()
    # No backfill by name: the old row stays unattributed.
    assert got == ("ip-harness", None)
