"""Default-off for new installs, untouched for upgrades (spec 2026-10-07 §9)."""

import sqlite3
from pathlib import Path

from app.db.database import open_db
from app.db.migrations import apply_migrations


def _value(db_path: Path) -> str | None:
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT value FROM settings WHERE key='landing_page_enabled'").fetchone()
    return row[0] if row else None


async def _migrate(db_path: Path) -> None:
    async with open_db(db_path) as db:
        await apply_migrations(db)


async def test_fresh_database_starts_with_the_website_off(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    await _migrate(db)
    assert _value(db) == "false"


async def test_upgrade_keeps_an_enabled_website_on(tmp_path: Path) -> None:
    """An install that ran 0020 back when it seeded 'true' never re-runs it."""
    db = tmp_path / "old.db"
    await _migrate(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE settings SET value='true' WHERE key='landing_page_enabled'")
    await _migrate(db)  # the upgrade: every migration already recorded
    assert _value(db) == "true"
