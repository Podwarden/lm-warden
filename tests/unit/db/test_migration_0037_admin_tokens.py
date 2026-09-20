"""Migration 0037: admin tokens and their audit trail.

  * api_tokens.created_by -- nullable; rows written before 0037 read NULL
  * idx_api_tokens_scope over api_tokens(scope)
  * admin_audit, with (token_id, ts DESC) for one token's page and (ts) for
    the pruner; client_ip (proxy-reported) and peer_ip (the socket peer)
    both live next to each other
"""

import shutil
from pathlib import Path

import aiosqlite
import pytest

from app.db import migrations
from app.db.database import open_db
from app.db.migrations import apply_migrations


async def _migrated(tmp_data_dir: Path) -> Path:
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


async def test_created_by_is_a_nullable_text_column(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(api_tokens)")
        cols = {row[1]: row for row in await cur.fetchall()}
    assert cols["created_by"][2] == "TEXT"
    assert cols["created_by"][3] == 0  # notnull flag


async def test_scope_is_indexed(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA index_info(idx_api_tokens_scope)")
        assert [row[2] for row in await cur.fetchall()] == ["scope"]


async def test_admin_audit_table_and_indexes(tmp_data_dir: Path) -> None:
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(admin_audit)")
        assert [row[1] for row in await cur.fetchall()] == [
            "id", "ts", "token_id", "username", "method", "path",
            "status", "duration_ms", "client_ip", "peer_ip",
        ]
        cur = await db.execute("PRAGMA index_xinfo(idx_admin_audit_token_ts)")
        # (name, desc) of the key columns only.
        key = [(row[2], row[3]) for row in await cur.fetchall() if row[5]]
        assert key == [("token_id", 0), ("ts", 1)]
        cur = await db.execute("PRAGMA index_info(idx_admin_audit_ts)")
        assert [row[2] for row in await cur.fetchall()] == ["ts"]


async def test_an_inference_row_from_before_0037_survives_the_upgrade(
    tmp_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_sql = tmp_path / "sql-0036"
    old_sql.mkdir()
    for f in sorted(migrations.SQL_DIR.glob("*.sql")):
        if f.name < "0037":
            shutil.copy(f, old_sql / f.name)
    db_path = tmp_data_dir / "vllm-warden.db"
    real_sql = migrations.SQL_DIR
    monkeypatch.setattr(migrations, "SQL_DIR", old_sql)
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash) VALUES ('t1', 'bot', 'vw_abcde', 'h1')"
        )
        await db.commit()
    # Not monkeypatch.undo(): that would also undo tmp_data_dir's env vars.
    monkeypatch.setattr(migrations, "SQL_DIR", real_sql)
    async with open_db(db_path) as db:
        await apply_migrations(db)
        cur = await db.execute("SELECT scope, created_by FROM api_tokens WHERE id = 't1'")
        assert await cur.fetchone() == ("inference", None)
