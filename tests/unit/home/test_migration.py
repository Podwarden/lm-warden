"""Migration 0057: fresh installs get `builtin`; upgraded ones keep what they showed."""

import sqlite3
from pathlib import Path

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations

FILE = "0057_home_page_mode.sql"


async def _migrate(db_path: Path) -> None:
    async with open_db(db_path) as db:
        await apply_migrations(db)


def _settings(db_path: Path) -> dict[str, str]:
    with sqlite3.connect(db_path) as db:
        return dict(db.execute("SELECT key, value FROM settings"))


async def _upgraded(db_path: Path, landing: str | None) -> dict[str, str]:
    """A database that ran everything before 0057, with the given legacy value."""
    await _migrate(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("DELETE FROM schema_migrations WHERE filename = ?", (FILE,))
        db.execute("DELETE FROM settings WHERE key IN ('home_page_mode','home_page_off_action')")
        if landing is None:
            db.execute("DELETE FROM settings WHERE key = 'landing_page_enabled'")
        else:
            db.execute(
                "UPDATE settings SET value = ? WHERE key = 'landing_page_enabled'", (landing,)
            )
    await _migrate(db_path)
    return _settings(db_path)


async def test_fresh_database_gets_builtin_and_console(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    await _migrate(db)
    s = _settings(db)
    assert (s["home_page_mode"], s["home_page_off_action"]) == ("builtin", "console")
    assert s["landing_page_enabled"] == "false"  # untouched, inert


@pytest.mark.parametrize(
    ("legacy", "mode"),
    [("true", "builtin"), ("false", "off"), (None, "off")],
    ids=["true-maps-to-builtin", "false-maps-to-off", "missing-row-maps-to-off"],
)
async def test_upgraded_database_keeps_what_it_showed(
    tmp_path: Path, legacy: str | None, mode: str
) -> None:
    s = await _upgraded(tmp_path / "old.db", legacy)
    assert (s["home_page_mode"], s["home_page_off_action"]) == (mode, "console")
    assert s.get("landing_page_enabled") == legacy  # row left alone


async def test_rerun_does_not_overwrite_an_admins_choice(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    await _migrate(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE settings SET value='off' WHERE key='home_page_mode'")
    await _migrate(db)
    assert _settings(db)["home_page_mode"] == "off"
