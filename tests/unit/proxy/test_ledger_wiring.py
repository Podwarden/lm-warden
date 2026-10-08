"""#279 stage 3 -- the request path only accumulates; the ledger writes later."""

import asyncio
import shutil
import sqlite3
import time

from fastapi.testclient import TestClient

from app.proxy.ledger import Ledger
from tests.unit.proxy.ledger_helpers import flush_ledger
from tests.unit.proxy.test_dp_routing_forward import _post, _ready


def test_request_path_does_not_write_ledger_tables_until_flush(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    # No background loop: only the explicit flush below may write.
    client.app.state.ledger = Ledger(tmp_data_dir / "vllm-warden.db")
    _post(client)
    db = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM counters").fetchone()[0] == 0
    flush_ledger(client)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT requests FROM counters").fetchone()[0] == 1


def test_auth_does_not_commit_on_request_path(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir, dp=1)
    client.app.state.ledger = Ledger(tmp_data_dir / "vllm-warden.db")
    from app.db.repos.tokens import TokenRepo

    def boom(*a, **k):
        raise AssertionError("touch_last_used must not run on the request path")

    monkeypatch.setattr(TokenRepo, "touch_last_used", boom)
    r, _ = _post(client)
    assert r.status_code == 200
    assert client.app.state.ledger.pending()
    assert "tok1" in client.app.state.ledger.touched()


async def _idle(self, interval=1.0):
    await asyncio.Event().wait()


def test_lifespan_shutdown_flushes_ledger(tmp_data_dir, migrated_db_template, monkeypatch):
    from app.main import build_app

    # With the loop idle, only the lifespan's final flush can write the row.
    monkeypatch.setattr(Ledger, "run_forever", _idle)
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    db = tmp_data_dir / "vllm-warden.db"
    with TestClient(app) as c:
        _ready(c, tmp_data_dir, dp=1)
        r, _ = _post(c)
        assert r.status_code == 200
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM counters").fetchone()[0] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT requests FROM counters").fetchone()[0] == 1


def test_shutdown_with_a_held_write_lock_is_bounded(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    from app.main import build_app
    from app.stats.request_history import RequestHistoryStore

    monkeypatch.setattr(Ledger, "run_forever", _idle)
    # The history store's own final flush waits out the same held lock (its own
    # budget, not the ledger's); keep it out so this pins the ledger's bound.
    monkeypatch.setattr(RequestHistoryStore, "run_forever", _idle)
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    db = tmp_data_dir / "vllm-warden.db"
    lock = sqlite3.connect(db, isolation_level=None)
    t0 = None
    try:
        with TestClient(app) as c:
            _ready(c, tmp_data_dir, dp=1)
            r, _ = _post(c)
            assert r.status_code == 200
            lock.execute("BEGIN IMMEDIATE")
            t0 = time.monotonic()
        elapsed = time.monotonic() - t0
        assert elapsed < 7, elapsed
        # The batch is still pending (never lost, nothing raised).
        assert app.state.ledger.pending()
    finally:
        lock.execute("ROLLBACK") if lock.in_transaction else None
        lock.close()


def _history_row(i):
    return {
        "id": f"held-lock-{i}",
        "model": "qwen",
        "model_id": "qwen",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "duration_s": 0.1,
        "finish_reason": "stop",
    }


def test_whole_shutdown_with_a_held_write_lock_fits_dockers_stop_grace(
    tmp_data_dir, migrated_db_template
):
    """#295: the real loops, nothing idled. Under a write lock held by another
    connection, the lifespan's ledger flush, the ledger loop's last flush and
    the request-history writer (caught mid-write AND with rows still queued)
    all wait on the same lock. Together they must end inside Docker's 10 s
    stop grace, or the container is SIGKILLed instead of stopped."""
    from app.main import build_app

    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    db = tmp_data_dir / "vllm-warden.db"
    lock = sqlite3.connect(db, isolation_level=None)
    t0 = None
    try:
        with TestClient(app) as c:
            _ready(c, tmp_data_dir, dp=1)
            r, _ = _post(c)
            assert r.status_code == 200
            lock.execute("BEGIN IMMEDIATE")
            history = app.state.request_history
            assert history.record(_history_row(0))
            # Past the writer's batching sleep: it is now inside a write that
            # waits on the lock. Then queue more for its final flush.
            time.sleep(history.flush_interval_s + 0.5)
            assert history.record(_history_row(1))
            r, _ = _post(c)  # a ledger batch the lock keeps pending
            assert r.status_code == 200
            assert app.state.ledger.pending()
            t0 = time.monotonic()
        elapsed = time.monotonic() - t0
        # Docker's default stop grace is 10 s; keep a margin for a slow runner.
        assert elapsed < 9, elapsed
    finally:
        lock.execute("ROLLBACK") if lock.in_transaction else None
        lock.close()


async def test_a_failing_variant_lookup_still_records_the_request(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    """#279 final review M2: the variant is best-effort; the counters, the
    minute sample and the key's usage are recorded without it."""
    from types import SimpleNamespace

    from app.proxy import routes

    db = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, db)
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices) "
            "VALUES ('qwen', 'qwen', 'o/r', '[0]')"
        )
        c.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('tok1', 't', 'vw_x', 'h', 'inference')"
        )
        c.commit()

    def boom(state, model):
        raise RuntimeError("no variant")

    monkeypatch.setattr(routes, "running_variant", boom)
    ledger = Ledger(db)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ledger=ledger)))
    await routes._record_counters(request, SimpleNamespace(id="qwen"), "tok1", 5, 2)
    assert ledger.pending() is True
    await ledger.flush()
    with sqlite3.connect(db) as c:
        assert c.execute(
            "SELECT model_id, token_id, requests, prompt_tokens, completion_tokens FROM counters"
        ).fetchall() == [("qwen", "tok1", 1, 5, 2)]
        assert c.execute("SELECT model_id, requests FROM model_samples").fetchall() == [("qwen", 1)]
        assert c.execute("SELECT token_id, requests FROM token_usage_minute").fetchall() == [
            ("tok1", 1)
        ]
        assert c.execute("SELECT COUNT(*) FROM token_model_usage_minute").fetchone()[0] == 0
