"""Migration 0053: sessions pages columns, the toolsets and tool-name tables, indexes."""

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_0053_adds_session_columns_toolsets_and_indexes(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(request_history)")
        cols = {r[1]: r[2] for r in await cur.fetchall()}
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        idx = {r[0] for r in await cur.fetchall()}
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in await cur.fetchall()}
        cur = await db.execute("PRAGMA table_info(session_tools)")
        st = {r[1]: (r[2], r[3], r[5]) for r in await cur.fetchall()}
    for name, typ in {
        "cobs2_session": "TEXT",
        "cobs2_conv": "TEXT",
        "cobs2_shape": "TEXT",
        "cobs2_parent_req": "TEXT",
        "cobs2_sess_ref": "TEXT",
        "cobs2_toolset": "TEXT",
        "relayed": "INTEGER",
        "upstream": "TEXT",
        "served_gpu_tokens": "INTEGER",
        "served_ram_tokens": "INTEGER",
        "served_nvme_tokens": "INTEGER",
        "served_remote_tokens": "INTEGER",
        "load_s": "REAL",
        "node_id": "TEXT",
    }.items():
        assert cols.get(name) == typ, name
    assert {
        "idx_request_history_cobs2_session",
        "idx_request_history_sess_ref",
        "idx_request_history_parent_session",
        "idx_session_tools_last_seen",
    } <= idx
    assert {"cobs2_toolsets", "session_tools"} <= tables
    # names only: no description or schema column, ever
    assert set(st) == {"token_id", "tool_hmac", "name", "family", "size", "first_seen", "last_seen"}
    assert st["token_id"][2] == 1 and st["tool_hmac"][2] == 2
