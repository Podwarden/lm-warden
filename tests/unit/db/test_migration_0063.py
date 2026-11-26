"""Migration 0063: Prompt Lens layer 1 and response fields on request_history,
all nullable, so every row written before it still loads."""

import sqlite3

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations

COLUMNS = {
    "lens_break": "TEXT",
    "lens_cause": "TEXT",
    "lens_span_class": "TEXT",
    "lens_span_line": "INTEGER",
    "lens_lost_tokens": "INTEGER",
    "lens_cf_tokens": "INTEGER",
    "lens_other_breaks": "INTEGER",
    "cache_write_tokens": "INTEGER",
    "cache_marks": "INTEGER",
    "lens_repeat_results": "INTEGER",
    "max_tokens_req": "INTEGER",
    "reasoning_tokens": "INTEGER",
    "reply_chars": "INTEGER",
    "calls_n": "INTEGER",
    "calls_bad": "TEXT",
    "tools_lint": "TEXT",
}


async def _migrated(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


async def test_migration_0063_columns_nullable(tmp_data_dir):
    db_path = await _migrated(tmp_data_dir)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(request_history)")
        cols = {r[1]: (r[2], r[3], r[4]) for r in await cur.fetchall()}
    for name, typ in COLUMNS.items():
        # (type, notnull, default): nullable, no default.
        assert cols.get(name) == (typ, 0, None), name


async def test_rows_without_lens_fields_still_load(tmp_data_dir):
    """A row written without the new fields (as before 0063) reads back with
    every lens field None; a row with them reads them."""
    from app.sessions.store import request_row
    from app.stats.request_history import RequestHistoryStore

    db_path = await _migrated(tmp_data_dir)
    store = RequestHistoryStore(db_path)
    base = {
        "model": "m",
        "model_id": "m",
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "duration_s": 1.0,
        "started_iso": "2026-11-23T00:00:00Z",
    }
    store.record({**base, "id": "old"}, finished_at=1000.0)
    store.record(
        {
            **base,
            "id": "new",
            "lens_break": "sys",
            "lens_cause": "sys_changed",
            "lens_span_class": "path",
            "lens_span_line": 41,
            "lens_lost_tokens": 3000,
            "lens_cf_tokens": 7920,
            "lens_other_breaks": 1,
            "cache_write_tokens": 28434,
            "cache_marks": 2,
        },
        finished_at=1001.0,
    )
    assert await store.flush() == 2
    with sqlite3.connect(db_path) as db:
        old = request_row(db, "old")
        new = request_row(db, "new")
    assert old is not None and new is not None
    for name in COLUMNS:
        assert getattr(old, name) is None, name
    assert (new.lens_break, new.lens_cause, new.lens_span_class, new.lens_span_line) == (
        "sys",
        "sys_changed",
        "path",
        41,
    )
    assert (new.lens_lost_tokens, new.lens_cf_tokens, new.lens_other_breaks) == (3000, 7920, 1)
    assert (new.cache_write_tokens, new.cache_marks) == (28434, 2)


async def test_unknown_codes_are_not_stored(tmp_data_dir):
    """The writer keeps only known codes: a stray string never lands as text."""
    from app.stats.request_history import RequestHistoryStore

    db_path = await _migrated(tmp_data_dir)
    store = RequestHistoryStore(db_path)
    store.record(
        {
            "id": "x",
            "model": "m",
            "model_id": "m",
            "started_iso": "2026-11-23T00:00:00Z",
            "lens_break": "the system prompt said hi",
            "lens_cause": "CANARY-7f3e9b21-lens",
            "lens_span_class": "CANARY-7f3e9b21-lens",
            "lens_span_line": "x",
        },
        finished_at=1000.0,
    )
    assert await store.flush() == 1
    with sqlite3.connect(db_path) as db:
        got = db.execute(
            "SELECT lens_break, lens_cause, lens_span_class, lens_span_line FROM request_history"
        ).fetchone()
    assert got == (None, None, None, None)
