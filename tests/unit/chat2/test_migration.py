import sqlite3
from pathlib import Path

from tests.conftest import seed_admin_user


def _conn(db_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def test_chat2_tables_exist(tmp_data_dir, client) -> None:
    client.get("/healthz")  # boots the app, runs migrations
    c = _conn(tmp_data_dir / "vllm-warden.db")
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "chats",
        "messages",
        "attachments",
        "user_chat_defaults",
        "rate_cards",
        "usage_ledger",
    } <= names
    cols = {r[1] for r in c.execute("PRAGMA table_info(models)")}
    assert {"supports_tools", "supports_vision", "supports_reasoning"} <= cols


def test_chat_user_fk_is_integer_and_enforced(tmp_data_dir, client) -> None:
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    c = _conn(tmp_data_dir / "vllm-warden.db")
    (uid,) = c.execute("SELECT id FROM users LIMIT 1").fetchone()
    c.execute(
        "INSERT INTO chats(id,user_id,title,settings_json,created_at,updated_at) "
        "VALUES ('c1',?,'t','{}','2026-01-01T00:00:00.000Z','2026-01-01T00:00:00.000Z')",
        (uid,),
    )
    try:
        c.execute(
            "INSERT INTO chats(id,user_id,title,settings_json,created_at,updated_at) "
            "VALUES ('c2',999,'t','{}','2026-01-01T00:00:00.000Z','2026-01-01T00:00:00.000Z')"
        )
        raise AssertionError("FK not enforced")
    except sqlite3.IntegrityError:
        pass


def test_message_seq_unique_per_chat(tmp_data_dir, client) -> None:
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    c = _conn(tmp_data_dir / "vllm-warden.db")
    (uid,) = c.execute("SELECT id FROM users LIMIT 1").fetchone()
    ts = "2026-01-01T00:00:00.000Z"
    c.execute(
        "INSERT INTO chats(id,user_id,title,settings_json,created_at,updated_at) "
        "VALUES ('c1',?,'t','{}',?,?)",
        (uid, ts, ts),
    )
    row = ("m1", "c1", 1, "user", "[]", "{}", ts)
    c.execute(
        "INSERT INTO messages(id,chat_id,seq,role,parts_json,settings_snapshot_json,"
        "created_at) VALUES (?,?,?,?,?,?,?)",
        row,
    )
    try:
        c.execute(
            "INSERT INTO messages(id,chat_id,seq,role,parts_json,settings_snapshot_json,"
            "created_at) VALUES (?,?,?,?,?,?,?)",
            ("m2",) + row[1:],
        )
        raise AssertionError("UNIQUE(chat_id, seq) not enforced")
    except sqlite3.IntegrityError:
        pass
