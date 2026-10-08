import sqlite3

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_0045_adds_forest_columns_and_indexes(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    with sqlite3.connect(db_path) as db:
        cols = {r[1]: r[2] for r in db.execute("PRAGMA table_info(request_history)")}
        idx = {r[1] for r in db.execute("PRAGMA index_list(request_history)")}
    for name, typ in {
        "session_key": "TEXT",
        "parent_session_key": "TEXT",
        "turn_index": "INTEGER",
        "batch_id": "TEXT",
        "tools_out": "TEXT",
        "tools_in": "TEXT",
    }.items():
        assert cols.get(name) == typ, name
    assert "idx_request_history_session_key" in idx
    assert "idx_request_history_token_batch" in idx
