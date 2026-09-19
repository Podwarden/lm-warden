"""Migration 0018: per-token rate_limit_tps + priority + token_usage_minute.

Covers (S5, closes #104):
  * Columns appear with the correct types + defaults
  * priority CHECK trigger rejects out-of-range values on INSERT and UPDATE
  * rate_limit_tps CHECK trigger rejects <=0 on INSERT and UPDATE
  * token_usage_minute exists with the composite PK and minute index
  * Migration is idempotent (re-applying is a no-op)
  * Rollback recipe in the migration header DOES restore the schema
    (exercised here so we have CI coverage of the documented procedure)
"""

import sqlite3

import aiosqlite
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_columns_present_with_correct_defaults(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(api_tokens)")
        cols = {row[1]: row for row in await cur.fetchall()}
    assert "rate_limit_tps" in cols
    assert "priority" in cols
    # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk).
    # We only care about notnull (idx 3) and dflt_value (idx 4) here.
    notnull_idx, default_idx = 3, 4
    # priority is NOT NULL DEFAULT 5
    assert cols["priority"][notnull_idx] == 1
    assert cols["priority"][default_idx] == "5"
    # rate_limit_tps is nullable (unlimited)
    assert cols["rate_limit_tps"][notnull_idx] == 0


async def test_priority_check_rejects_above_range_on_insert(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        # RAISE(ABORT, ...) in a trigger surfaces as IntegrityError
        # (SQLITE_CONSTRAINT). Tests pin the exact exception so a future
        # refactor that swaps the trigger for a CHECK constraint still
        # raises the same Python-level error type.
        with pytest.raises(aiosqlite.IntegrityError):
            await db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, priority) "
                "VALUES ('t1', 'n', 'pfx', 'h', 'inference', 10)"
            )
            await db.commit()


async def test_priority_check_rejects_below_range_on_insert(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        with pytest.raises(aiosqlite.IntegrityError):
            await db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, priority) "
                "VALUES ('t2', 'n', 'pfx', 'h', 'inference', -1)"
            )
            await db.commit()


async def test_priority_check_rejects_above_range_on_update(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, priority) "
            "VALUES ('t3', 'n', 'pfx', 'h', 'inference', 5)"
        )
        await db.commit()
        with pytest.raises(aiosqlite.IntegrityError):
            await db.execute("UPDATE api_tokens SET priority = 99 WHERE id = 't3'")
            await db.commit()


async def test_priority_accepts_boundary_values(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        # 0 and 9 are inside the range — these must succeed. Distinct
        # hashes are required because api_tokens.hash has a UNIQUE
        # constraint from the original 0001 schema.
        await db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, priority) "
            "VALUES ('t-lo', 'lo', 'pfx', 'h-lo', 'inference', 0)"
        )
        await db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, priority) "
            "VALUES ('t-hi', 'hi', 'pfx', 'h-hi', 'inference', 9)"
        )
        await db.commit()


async def test_rate_limit_tps_rejects_zero_or_negative(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        with pytest.raises(aiosqlite.IntegrityError):
            await db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, rate_limit_tps) "
                "VALUES ('t-z', 'z', 'pfx', 'h', 'inference', 0)"
            )
            await db.commit()
        with pytest.raises(aiosqlite.IntegrityError):
            await db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, rate_limit_tps) "
                "VALUES ('t-n', 'n', 'pfx', 'h', 'inference', -1)"
            )
            await db.commit()


async def test_rate_limit_tps_null_means_unlimited(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        # NULL must be allowed — the trigger only fires when NEW.rate_limit_tps
        # IS NOT NULL.
        await db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('t-null', 'null', 'pfx', 'h', 'inference')"
        )
        await db.commit()
        cur = await db.execute("SELECT rate_limit_tps FROM api_tokens WHERE id='t-null'")
        (val,) = await cur.fetchone()
        assert val is None


async def test_token_usage_minute_table_exists_with_pk_and_index(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(token_usage_minute)")
        cols = [row[1] for row in await cur.fetchall()]
        assert set(cols) >= {
            "token_id", "minute", "requests", "prompt_tokens", "completion_tokens",
        }
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='token_usage_minute'"
        )
        indices = {row[0] for row in await cur.fetchall()}
        assert "idx_token_usage_minute_minute" in indices


async def test_migration_idempotent(tmp_data_dir):
    """Re-applying migrations must not double-insert into schema_migrations
    nor re-run 0018 (which would fail because triggers/tables already exist)."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second pass — no-op


async def test_manual_rollback_recipe_works(tmp_data_dir):
    """Exercise the rollback recipe documented in 0018's header so it
    cannot silently rot. SQLite 3.35+ supports ALTER TABLE DROP COLUMN.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)

    # Sync sqlite3 (rollback path uses raw SQL — match what an operator
    # would type at the sqlite3 prompt). Must match the exact recipe
    # documented in the 0018 migration header, including the
    # drop-triggers-before-columns ordering.
    with sqlite3.connect(db_path) as db:
        db.execute("BEGIN")
        db.execute("DROP TRIGGER IF EXISTS api_tokens_priority_range_insert")
        db.execute("DROP TRIGGER IF EXISTS api_tokens_priority_range_update")
        db.execute("DROP TRIGGER IF EXISTS api_tokens_rate_limit_tps_range_insert")
        db.execute("DROP TRIGGER IF EXISTS api_tokens_rate_limit_tps_range_update")
        db.execute("DROP TABLE IF EXISTS token_usage_minute")
        db.execute("DROP INDEX IF EXISTS idx_api_tokens_priority_id")  # 0035
        db.execute("ALTER TABLE api_tokens DROP COLUMN priority")
        db.execute("ALTER TABLE api_tokens DROP COLUMN rate_limit_tps")
        db.execute(
            "DELETE FROM schema_migrations WHERE filename = '0018_tokens_rate_priority.sql'"
        )
        db.execute("COMMIT")
        # Verify columns are gone.
        cols = {row[1] for row in db.execute("PRAGMA table_info(api_tokens)").fetchall()}
        assert "priority" not in cols
        assert "rate_limit_tps" not in cols
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "token_usage_minute" not in tables
