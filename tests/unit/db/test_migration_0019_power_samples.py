"""Migration 0019: per-GPU minute-bucketed power_samples table.

Covers (S7, #124):
  * table appears with the documented column set and types
  * (gpu_idx, minute) composite PK enforces one row per bucket
  * idx_power_samples_gpu_idx_ts exists for the time-series read path
  * UPSERT-style accumulation works (avg-watts = watts_sum / samples)
  * Migration is idempotent (re-applying is a no-op)
  * Manual rollback recipe in the migration header restores the schema
    (exercised here so we have CI coverage of the documented procedure)
"""

import sqlite3

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_power_samples_table_exists_with_columns(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(power_samples)")
        cols = {row[1]: row for row in await cur.fetchall()}
    assert set(cols.keys()) >= {"gpu_idx", "minute", "watts_sum", "samples"}
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk).
    type_idx, notnull_idx, pk_idx = 2, 3, 5
    assert cols["gpu_idx"][type_idx] == "INTEGER"
    assert cols["minute"][type_idx] == "INTEGER"
    assert cols["watts_sum"][type_idx] == "REAL"
    assert cols["samples"][type_idx] == "INTEGER"
    # Composite PK — both columns flagged pk > 0.
    assert cols["gpu_idx"][pk_idx] > 0
    assert cols["minute"][pk_idx] > 0
    # NOT NULL on all four — the bucket integer + sample counters never null.
    for c in ("gpu_idx", "minute", "watts_sum", "samples"):
        assert cols[c][notnull_idx] == 1, c


async def test_index_exists_on_gpu_idx_ts(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' " "AND tbl_name='power_samples'"
        )
        indices = {row[0] for row in await cur.fetchall()}
    assert "idx_power_samples_gpu_idx_ts" in indices


async def test_upsert_accumulates_within_bucket(tmp_data_dir):
    """The collector lands ~12 samples per minute (5s cadence). Two writes
    against the same (gpu_idx, minute) PK must accumulate rather than
    overwrite, so the read side can compute an average across the bucket.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        # First sample: 100 W.
        await db.execute(
            "INSERT INTO power_samples(gpu_idx, minute, watts_sum, samples) "
            "VALUES (0, 27500000, 100.0, 1) "
            "ON CONFLICT(gpu_idx, minute) DO UPDATE SET "
            "watts_sum = watts_sum + excluded.watts_sum, "
            "samples = samples + excluded.samples"
        )
        # Second sample: 200 W in the same minute.
        await db.execute(
            "INSERT INTO power_samples(gpu_idx, minute, watts_sum, samples) "
            "VALUES (0, 27500000, 200.0, 1) "
            "ON CONFLICT(gpu_idx, minute) DO UPDATE SET "
            "watts_sum = watts_sum + excluded.watts_sum, "
            "samples = samples + excluded.samples"
        )
        await db.commit()
        cur = await db.execute(
            "SELECT watts_sum, samples FROM power_samples "
            "WHERE gpu_idx = 0 AND minute = 27500000"
        )
        watts_sum, samples = await cur.fetchone()
    assert watts_sum == 300.0
    assert samples == 2
    # Read-side average = 150.0 W.
    assert watts_sum / samples == 150.0


async def test_migration_idempotent(tmp_data_dir):
    """Re-applying migrations must not double-create the table (which would
    error with 'table power_samples already exists') nor re-insert into
    schema_migrations."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second pass — no-op


async def test_manual_rollback_recipe_works(tmp_data_dir):
    """Exercise the rollback recipe documented in 0019's header so it can't
    silently rot. After rollback, the table and index are gone and the
    migration is no longer recorded as applied.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    with sqlite3.connect(db_path) as db:
        db.execute("BEGIN")
        db.execute("DROP INDEX IF EXISTS idx_power_samples_gpu_idx_ts")
        db.execute("DROP TABLE IF EXISTS power_samples")
        db.execute("DELETE FROM schema_migrations WHERE filename = '0019_power_samples.sql'")
        db.execute("COMMIT")
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "power_samples" not in tables
        indices = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_power_samples_gpu_idx_ts" not in indices
        applied = {
            row[0] for row in db.execute("SELECT filename FROM schema_migrations").fetchall()
        }
        assert "0019_power_samples.sql" not in applied
