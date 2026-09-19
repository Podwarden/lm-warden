
import pytest

from app.db.database import open_db
from app.db.migrations import SQL_DIR, apply_migrations


@pytest.mark.asyncio
async def test_columns_added_with_backfill(tmp_path):
    # Why: apply_migrations has no stop_after param; manually apply 0001-0008,
    # mark them in schema_migrations, then call apply_migrations() to run only 0009.
    db_path = tmp_path / "vw.db"
    async with open_db(db_path) as db:
        # Bootstrap schema_migrations so the runner can track state.
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  filename TEXT PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        await db.commit()

        # Apply migrations 0001-0008 manually and record them.
        for path in sorted(SQL_DIR.glob("*.sql")):
            if path.stem >= "0009":
                break
            await db.executescript(path.read_text(encoding="utf-8"))
            await db.execute(
                "INSERT INTO schema_migrations(filename) VALUES (?)", (path.name,)
            )
            await db.commit()

        # Insert a row representing pre-migration state (expires_at not yet a column).
        await db.execute(
            "INSERT INTO api_tokens (id, name, prefix, hash, scope, allowed_models, "
            "rate_limit_rpm, rate_limit_tpm, created_at) "
            "VALUES ('tok1', 'old', 'aa', 'hh', 'all', NULL, NULL, NULL, "
            "datetime('now', '-30 days'))"
        )
        await db.commit()

        # apply_migrations sees only 0009 is unapplied — runs it.
        await apply_migrations(db)

        row = await (await db.execute(
            "SELECT expires_at, rotated_at, rotated_from FROM api_tokens WHERE id='tok1'"
        )).fetchone()
        assert row[0] is not None  # backfilled by migration UPDATE
        assert row[1] is None
        assert row[2] is None

        # An expires_at index must exist. apply_migrations ran every later file
        # too, and 0035 replaces 0009's idx_tokens_expires_at with the
        # composite idx_api_tokens_expires_id.
        idx = await (await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name IN ('idx_tokens_expires_at', 'idx_api_tokens_expires_id')"
        )).fetchone()
        assert idx is not None
