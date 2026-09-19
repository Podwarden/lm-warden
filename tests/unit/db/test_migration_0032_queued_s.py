"""Migration 0032: the admission-queue wait column on request_history.

The proxy admits requests through a per-engine, strict-priority gate
(app/proxy/scheduler.py). Waiting there was invisible: the clock that ``ttft_s``
is measured from is read AFTER admission (app/proxy/routes.py), so queue wait
appears in no stored column at all — while the scheduler's own docstring warns
that priority-9 traffic can starve lower priorities indefinitely.

Covers:
  * the column appears, typed REAL and NULLABLE
  * an existing database at 0031 with history rows upgrades cleanly, and those
    rows read NULL — "not measured", which is not the claim "waited 0 seconds"
"""

import sqlite3

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations
from tests.unit.db.test_migration_0031_request_history import _apply_through


async def test_queued_s_column_exists_and_is_nullable(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(request_history)")
        cols = {row[1]: row for row in await cur.fetchall()}
    assert "queued_s" in cols
    type_idx, notnull_idx = 2, 3
    assert cols["queued_s"][type_idx] == "REAL"
    # Nullable on purpose — see the module docstring.
    assert cols["queued_s"][notnull_idx] == 0


async def test_rows_written_before_the_upgrade_read_null_not_zero(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    await _apply_through(db_path, "0031_request_history.sql")
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, duration_s, "
            "ttft_s, started_iso) VALUES ('old-1', 100.0, 'm-1', 'served-1', 2.0, 0.5, 'x')"
        )
        db.commit()

    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # idempotent

    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT ttft_s, queued_s FROM request_history WHERE id = 'old-1'"
        ).fetchone()
    assert row == (0.5, None)
