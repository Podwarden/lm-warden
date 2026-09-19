"""Migration 0031: the per-request history table.

Covers:
  * the table appears with the documented columns, types and nullability
  * both indexes exist for the newest-first-within-a-window read path
  * an EXISTING database -- one migrated through 0030 with rows in the tables
    the pruner and the stats page already use -- upgrades cleanly and keeps
    its rows (migrations are forward-only; there is no other path to prod)
  * no foreign key to models: history outlives a deleted model row
"""

import sqlite3

import aiosqlite

from app.db.database import open_db
from app.db.migrations import SQL_DIR, apply_migrations


async def _apply_through(db_path, last: str) -> None:
    """Migrate a fresh file through ``last`` only, the way a pre-upgrade
    deployment's database looks the moment the new image boots."""
    import app.db.migrations as m

    real = m.SQL_DIR
    files = sorted(p.name for p in real.glob("*.sql"))
    assert last in files
    keep = files[: files.index(last) + 1]
    async with open_db(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  filename TEXT PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        for name in keep:
            statements = m._split_statements((real / name).read_text(encoding="utf-8"))
            await db.execute("BEGIN IMMEDIATE")
            for stmt in statements:
                await db.execute(stmt)
            await db.execute("INSERT INTO schema_migrations(filename) VALUES (?)", (name,))
            await db.commit()


async def test_request_history_table_exists_with_columns(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(request_history)")
        cols = {row[1]: row for row in await cur.fetchall()}
    # (cid, name, type, notnull, dflt_value, pk)
    type_idx, notnull_idx, pk_idx = 2, 3, 5
    expected = {
        "id": "TEXT", "finished_at": "REAL", "model_id": "TEXT", "model": "TEXT",
        "token_name": "TEXT", "client_ip": "TEXT", "prompt_tokens": "INTEGER",
        "completion_tokens": "INTEGER", "duration_s": "REAL", "ttft_s": "REAL",
        "finish_reason": "TEXT", "orphan": "INTEGER", "started_iso": "TEXT",
        # Added by 0032; asserted here too because this test pins the shape of
        # the table as it actually ships, not as 0031 left it.
        "queued_s": "REAL",
        # Added by 0033 (token details page), for the same reason.
        "token_id": "TEXT",
        # Added by 0036 (per-variant usage), likewise.
        "variant_id": "TEXT",
    }
    assert set(cols) == set(expected)
    for name, typ in expected.items():
        assert cols[name][type_idx] == typ, name
    assert cols["id"][pk_idx] == 1
    for required in ("finished_at", "model_id", "model", "duration_s", "started_iso"):
        assert cols[required][notnull_idx] == 1, required
    # Nullable on purpose: no first token, no token, no reason observed.
    for optional in (
        "ttft_s", "token_name", "client_ip", "finish_reason", "queued_s", "token_id",
        "variant_id",
    ):
        assert cols[optional][notnull_idx] == 0, optional


async def test_indexes_exist(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='request_history'"
        )
        indices = {row[0] for row in await cur.fetchall()}
    assert "idx_request_history_finished_at" in indices
    assert "idx_request_history_model_finished" in indices


async def test_existing_database_upgrades_cleanly_and_keeps_its_rows(tmp_data_dir):
    """The deployed DB is at 0030 with real rows. Booting the new image must
    add the table without touching what is already there."""
    db_path = tmp_data_dir / "vllm-warden.db"
    await _apply_through(db_path, "0030_stress_progress.sql")
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-1', 'served-1', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, "
            "completion_tokens) VALUES ('m-1', 1, 3, 30, 300)"
        )
        db.commit()
        tables_before = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "request_history" not in tables_before

    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # idempotent

    with sqlite3.connect(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "request_history" in tables
        assert tables_before <= tables
        assert db.execute("SELECT requests FROM model_samples").fetchone() == (3,)
        applied = {r[0] for r in db.execute("SELECT filename FROM schema_migrations")}
    assert applied == {p.name for p in SQL_DIR.glob("*.sql")}


async def test_history_outlives_a_deleted_model_row(tmp_data_dir):
    """No FK: deleting a model must not erase what it served. The served name
    is on the row, so it stays readable."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-gone', 'served-gone', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'registered')"
        )
        db.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, duration_s, "
            "started_iso) VALUES ('r1', 100.0, 'm-gone', 'served-gone', 1.5, 'x')"
        )
        db.execute("DELETE FROM models WHERE id = 'm-gone'")
        db.commit()
        assert db.execute("SELECT model FROM request_history").fetchone() == ("served-gone",)
