import sqlite3

import aiosqlite
import pytest

from app.db.database import open_db
from app.db.migrations import (
    SQL_DIR,
    _split_statements,
    apply_migrations,
)


async def test_migrations_create_schema_table_and_run_files(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        assert "schema_migrations" in tables
        assert "users" in tables


async def test_migrations_idempotent(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second call must be a no-op
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM schema_migrations")
        (count,) = await cur.fetchone()
        # 0001–0015 + 0017 (drop_bench; 0016 reserved) + 0018 (tokens-v2,
        # S5/#104) + 0019 (power_samples, S7/#124) + 0020 (landing_page_enabled
        # seed, #155) + 0021 (public_url doc-only no-op, #154) + 0022
        # (engine_templates + stack_attempts, #162) + 0023 (drop ghost
        # hf_cache_dir setting, 2026-06-15 ENOSPC follow-up) + 0024
        # (models.prior_status, so recovery stops keying on an error string) +
        # 0025 (chat2: chats/messages/attachments/user_chat_defaults/
        # rate_cards/usage_ledger + models.supports_* columns) + 0026
        # (chats.title_prev, so a reassessed auto-title can be undone) + 0027
        # (models.backend, sub-project B -- nullable, no backfill, NULL means
        # vllm) + 0028 (models.mmproj_filename + models.n_gpu_layers,
        # sub-project C -- the two llama.cpp knobs, both nullable, no backfill)
        # + 0029 (stress_runs: measured operating limits, keyed on
        # (model_id, fingerprint) so the same model at two context sizes keeps
        # two valid rows; a separate table rather than `models` columns because
        # _PATCHABLE_MODEL_FIELDS is derived and would make a machine-written
        # measurement hand-editable) + 0030 (stress_runs.progress: mid-run
        # phase and probe count, overwritten in place by the heartbeat, so a
        # multi-hour run is not a blank modal) + 0031 (request_history: one row
        # per completed /v1 request -- the proxy's own TTFT and duration, so
        # llama.cpp gets a latency panel and the requests chart honours the
        # window instead of a 15-minute in-memory ring) + 0032
        # (request_history.queued_s: the wait at the proxy's admission gate,
        # which no stored column held -- the clock ttft_s and duration_s are
        # measured from is read AFTER the slot is granted, so queueing was
        # invisible while the scheduler can starve low priorities) + 0033
        # (api_tokens.paused_at + request_history.token_id: the token details
        # page's pause and its per-key timings, no backfill by name) + 0034
        # (a partial index on request_history(finished_at) WHERE token_id IS
        # NOT NULL, so the series endpoint's latency_since stops walking every
        # token-less row on each poll -- #251) + 0035 (api_tokens sort
        # indexes ending in id, plus rotated_from, so the paged token list
        # reads one page off an index instead of sorting the table) + 0036
        # (model_variants + token_model_usage_minute + request_history.variant_id:
        # the per-(key, model variant) minute rollup behind the token page's
        # "Usage by model" card, no backfill) + 0037 (api_tokens.created_by +
        # idx_api_tokens_scope + admin_audit: admin tokens for the control API
        # and the audit trail of every request they make).
        assert count == 36


async def test_migrations_create_all_v2_tables(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        for t in ["users", "setup_state", "models", "model_runtime"]:
            assert t in tables


async def test_migrations_create_full_v2_schema(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        for t in ["api_tokens", "counters", "model_samples", "gpu_samples"]:
            assert t in tables
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in await cur.fetchall()}
        assert "idx_model_samples_minute" in indices
        assert "idx_gpu_samples_minute" in indices


async def test_0017_drops_bench_tables_and_is_idempotent(tmp_data_dir):
    """0017 must drop bench_run / bench_load_config_attempt / bench_cell —
    even when a pre-overhaul DB has 0012's tables already present, and
    even when migrations are re-applied (the DROP IF EXISTS in 0017
    becomes a no-op the second time around)."""
    db_path = tmp_data_dir / "vllm-warden.db"
    # First pass: pre-create the bench tables AND mark 0012 as already
    # applied so apply_migrations skips re-creating them. Simulates a
    # pre-overhaul DB.
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            "CREATE TABLE schema_migrations(filename TEXT PRIMARY KEY, applied_at TEXT);"
            "INSERT INTO schema_migrations(filename, applied_at) VALUES "
            "  ('0012_bench.sql', datetime('now'));"
            "CREATE TABLE bench_run(run_id TEXT PRIMARY KEY);"
            "CREATE TABLE bench_load_config_attempt(attempt_id TEXT PRIMARY KEY);"
            "CREATE TABLE bench_cell(cell_id TEXT PRIMARY KEY);"
        )
        await db.commit()
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # idempotency
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'bench_%'"
        )
        bench_tables = {row[0] for row in await cur.fetchall()}
        assert bench_tables == set(), bench_tables


# ---------------------------------------------------------------------------
# #231 — a migration and its schema_migrations row must commit together
# ---------------------------------------------------------------------------
#
# The old runner used ``executescript``, which issues an implicit COMMIT before
# it runs the script; the bookkeeping INSERT that followed landed in a separate
# transaction. A pod evicted (or an ENOSPC on /data) between the two left the
# DDL half-applied while the migration still read as unapplied, so the next boot
# replayed it, failed, and the pod never started again without hand-editing the
# SQLite file. These tests pin the atomicity that replaced it.


class _DieAfter:
    """aiosqlite connection proxy that dies once a statement has been run.

    Deliberately raises *after* forwarding the statement, because that is the
    real failure shape: the write reached SQLite, then the process went away.
    A proxy is used rather than a synthetic SQL_DIR so the interruption is
    exercised against a migration we actually ship.
    """

    def __init__(self, db, needle: str, occurrence: int = 1):
        self._db = db
        self._needle = needle
        self._occurrence = occurrence
        self._seen = 0
        self.tripped = False

    def __getattr__(self, name):
        return getattr(self._db, name)

    async def execute(self, sql, *args, **kwargs):
        cur = await self._db.execute(sql, *args, **kwargs)
        if self._needle in sql:
            self._seen += 1
            if self._seen == self._occurrence:
                self.tripped = True
                raise RuntimeError("simulated pod eviction mid-migration")
        return cur


async def _table_names(db_path):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in await cur.fetchall()}


async def _applied(db_path):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in await cur.fetchall()}


async def test_interrupted_migration_rolls_back_and_next_boot_recovers(tmp_data_dir):
    """Killed between two DDL statements of 0006, the next run must still boot.

    0006 creates counters, model_samples and gpu_samples in that order. We die
    after model_samples, i.e. with two of the three tables written. Under the
    old executescript runner those two tables survived while 0006 stayed
    unrecorded, so the replay died on "table counters already exists" and the
    pod was unbootable.
    """
    db_path = tmp_data_dir / "vllm-warden.db"

    async with open_db(db_path) as db:
        proxy = _DieAfter(db, "CREATE TABLE model_samples")
        with pytest.raises(RuntimeError, match="simulated pod eviction"):
            await apply_migrations(proxy)
        assert proxy.tripped

    # Rolled back: neither the partial DDL nor the bookkeeping row survived.
    tables = await _table_names(db_path)
    assert "counters" not in tables, tables
    assert "model_samples" not in tables, tables
    applied = await _applied(db_path)
    assert "0006_counters_samples.sql" not in applied
    # Migrations that fully committed before the interruption are kept — the
    # transaction is per file, not per run.
    assert "0001_users.sql" in applied

    # The whole point: the next boot recovers with no manual intervention.
    async with open_db(db_path) as db:
        await apply_migrations(db)

    tables = await _table_names(db_path)
    for t in ("counters", "model_samples", "gpu_samples", "token_usage_minute"):
        assert t in tables, tables
    assert await _applied(db_path) == {p.name for p in SQL_DIR.glob("*.sql")}


async def test_interrupted_between_ddl_and_bookkeeping_row_rolls_back(tmp_data_dir):
    """The exact window executescript could not close.

    Dying on the ``INSERT INTO schema_migrations`` — after every statement of
    the migration has run — is the worst case: the schema is fully changed and
    the runner is about to record it. That INSERT is inside the same
    transaction as the DDL now, so the whole migration unwinds.
    """
    db_path = tmp_data_dir / "vllm-warden.db"

    async with open_db(db_path) as db:
        # 3rd bookkeeping INSERT of the run == 0003_models.sql.
        proxy = _DieAfter(db, "INSERT INTO schema_migrations", occurrence=3)
        with pytest.raises(RuntimeError, match="simulated pod eviction"):
            await apply_migrations(proxy)
        assert proxy.tripped

    tables = await _table_names(db_path)
    assert "models" not in tables, tables
    assert "setup_state" in tables, tables  # 0002 committed cleanly
    assert await _applied(db_path) == {"0001_users.sql", "0002_setup_state.sql"}

    async with open_db(db_path) as db:
        await apply_migrations(db)
    assert "models" in await _table_names(db_path)
    assert await _applied(db_path) == {p.name for p in SQL_DIR.glob("*.sql")}


async def test_interrupted_migration_leaves_connection_usable(tmp_data_dir):
    """The failed migration must not strand an open transaction.

    main.py writes to the same DB straight after apply_migrations returns
    (boot reconciliation, mark_runtime_dead_on_startup, RuntimeRepo.clear_all),
    so a transaction left open by the error path would deadlock the next
    writer against itself.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        proxy = _DieAfter(db, "CREATE TABLE model_samples")
        with pytest.raises(RuntimeError):
            await apply_migrations(proxy)
        # Would raise "cannot start a transaction within a transaction" if the
        # rollback had not happened.
        await db.execute("BEGIN IMMEDIATE")
        await db.execute("INSERT INTO schema_migrations(filename) VALUES ('probe')")
        await db.commit()
    assert "probe" in await _applied(db_path)


# ---------------------------------------------------------------------------
# #231 — statement splitting
# ---------------------------------------------------------------------------


def test_split_keeps_trigger_bodies_whole():
    """0018's triggers contain semicolons inside BEGIN ... END.

    A ``split(";")`` splitter cuts each of the four triggers in half and every
    half is a syntax error. It would also treat the ``BEGIN;`` / ``COMMIT;``
    lines in 0018's commented-out rollback recipe as statements.
    """
    sql = (SQL_DIR / "0018_tokens_rate_priority.sql").read_text(encoding="utf-8")
    statements = _split_statements(sql)

    # Match on the closing END; rather than on the text "CREATE TRIGGER": the
    # splitter keeps each statement's preceding comment block attached, and
    # 0018's header prose mentions CREATE TRIGGER.
    triggers = [s for s in statements if s.rstrip().endswith("END;")]
    assert len(triggers) == 4, [s.strip()[:40] for s in statements]
    for trigger in triggers:
        assert "CREATE TRIGGER api_tokens_" in trigger
        assert "RAISE(ABORT" in trigger
    # Two ALTERs + four triggers + table + index; nothing from the comments.
    assert len(statements) == 8, [s.strip()[:40] for s in statements]


def test_split_ignores_semicolons_in_literals_and_comments():
    sql = (
        "-- rollback: BEGIN; DROP TABLE t; COMMIT;\n"
        "CREATE TABLE t (v TEXT);\n"
        "/* block; comment; */\n"
        "INSERT INTO t(v) VALUES ('a;b');\n"
    )
    statements = _split_statements(sql)
    assert len(statements) == 2, statements
    assert "CREATE TABLE t" in statements[0]
    assert "'a;b'" in statements[1]


def test_every_shipped_migration_splits_into_executable_statements():
    """Guards the whole directory, not just today's known-tricky file.

    Each chunk must be a single complete statement — that is what
    Connection.execute accepts — and must start with a real SQL keyword rather
    than a fragment of the file's comment prose.
    """
    for path in sorted(SQL_DIR.glob("*.sql")):
        statements = _split_statements(path.read_text(encoding="utf-8"))
        assert statements, f"{path.name} produced no statements"
        for stmt in statements:
            assert sqlite3.complete_statement(stmt), (path.name, stmt[:80])


async def test_transaction_hostile_migration_is_rejected_loudly(tmp_data_dir, monkeypatch):
    """A future migration carrying a PRAGMA must fail on a test DB, not in prod.

    PRAGMAs are silently ignored inside a transaction, so accepting one would
    mean the migration appears to succeed while doing nothing — or, worse,
    tempts someone to reintroduce executescript to make it work.
    """
    sql_dir = tmp_data_dir / "fake-sql"
    sql_dir.mkdir()
    (sql_dir / "0001_ok.sql").write_text("CREATE TABLE ok (id INTEGER);\n")
    (sql_dir / "0002_bad.sql").write_text("-- looks harmless\nPRAGMA journal_mode = DELETE;\n")
    monkeypatch.setattr("app.db.migrations.SQL_DIR", sql_dir)

    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        with pytest.raises(RuntimeError, match="0002_bad.sql"):
            await apply_migrations(db)

    assert "ok" in await _table_names(db_path)
    assert await _applied(db_path) == {"0001_ok.sql"}
