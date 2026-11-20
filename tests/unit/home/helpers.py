"""Helpers shared by the home-page tests."""

import sqlite3
from pathlib import Path


def put_setting(tmp_data_dir: Path, key: str, value: str) -> None:
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db", isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")
