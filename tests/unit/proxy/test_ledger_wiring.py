"""#279 stage 3 -- the request path only accumulates; the ledger writes later."""

import asyncio
import shutil
import sqlite3
import threading
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


def _catch_the_pruner_mid_write(monkeypatch):
    """Pin the race behind the old ~1-in-3 flake: a loop caught mid-write.

    The stats pruner runs a real prune at boot. When it had not reached its
    first DELETE before the test took the write lock, it sat in SQLite's busy
    handler at shutdown; cancelled there, its open_db close waited the full
    10 s _CLOSE_ACK_TIMEOUT behind that statement, so a ~2 s shutdown took
    12.0 s. Here the real prune_once runs only once the lock is held, so every
    run takes that path. Returns (go, in_write): set `go` with the lock held,
    then wait for `in_write`."""
    from app.runtime import stats_pruner

    go, in_write = threading.Event(), threading.Event()
    prune_once = stats_pruner.prune_once

    async def pruner(settings):
        while not go.is_set():  # noqa: ASYNC110 -- set from the test thread, not the loop
            await asyncio.sleep(0.01)
        in_write.set()
        await prune_once(settings)  # blocks on the held lock (30 s busy_timeout)
        await asyncio.Event().wait()

    monkeypatch.setattr(stats_pruner, "run_pruner_forever", pruner)
    return go, in_write


def _hold_the_pruner(go, in_write):
    go.set()
    assert in_write.wait(10)
    time.sleep(0.5)  # past connect and pragmas: inside the DELETE's busy wait


def _wait_for_sqlite_threads(before):
    """The abandoned prune ends once the lock is released; let its worker
    thread finish before the next test, so nothing outlives this one."""
    import aiosqlite

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        live = {t for t in threading.enumerate() if isinstance(t, aiosqlite.Connection)}
        if not (live - before):
            return
        time.sleep(0.01)


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
    go, in_write = _catch_the_pruner_mid_write(monkeypatch)
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    db = tmp_data_dir / "vllm-warden.db"
    lock = sqlite3.connect(db, isolation_level=None)
    before = set(threading.enumerate())
    t0 = None
    try:
        with TestClient(app) as c:
            _ready(c, tmp_data_dir, dp=1)
            r, _ = _post(c)
            assert r.status_code == 200
            lock.execute("BEGIN IMMEDIATE")
            _hold_the_pruner(go, in_write)
            t0 = time.monotonic()
        elapsed = time.monotonic() - t0
        # ~2.5 s: the ledger's 2 s busy_timeout, then the pruner's cancelled
        # close (0.5 s). 12.0 s when that close waited out its 10 s ack.
        assert elapsed < 7, elapsed
        # The batch is still pending (never lost, nothing raised).
        assert app.state.ledger.pending()
    finally:
        lock.execute("ROLLBACK") if lock.in_transaction else None
        lock.close()
        _wait_for_sqlite_threads(before)


def test_the_shutdown_flush_is_bounded_by_its_own_timeout_too(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    """The lifespan's `asyncio.timeout(5)` around the final ledger flush is a
    bound of its own, not a restatement of the ledger's 2 s busy_timeout: with
    the busy wait longer than the whole budget, only the timeout ends the
    flush, and the cancelled flush's close must not then wait 10 s for its
    acknowledgement. ~5.5 s; without the timeout, 30 s; without the short
    cancelled-close ack, 15 s."""
    from app.main import build_app
    from app.proxy import ledger
    from app.runtime import stats_pruner
    from app.stats.request_history import RequestHistoryStore

    monkeypatch.setattr(ledger, "LEDGER_BUSY_TIMEOUT_MS", 30_000)
    monkeypatch.setattr(Ledger, "run_forever", _idle)
    monkeypatch.setattr(RequestHistoryStore, "run_forever", _idle)

    async def idle_pruner(settings):
        await asyncio.Event().wait()

    monkeypatch.setattr(stats_pruner, "run_pruner_forever", idle_pruner)
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    db = tmp_data_dir / "vllm-warden.db"
    lock = sqlite3.connect(db, isolation_level=None)
    before = set(threading.enumerate())
    t0 = None
    try:
        with TestClient(app) as c:
            _ready(c, tmp_data_dir, dp=1)
            r, _ = _post(c)
            assert r.status_code == 200
            lock.execute("BEGIN IMMEDIATE")
            t0 = time.monotonic()
        elapsed = time.monotonic() - t0
        assert 4.5 < elapsed < 7, elapsed
        assert app.state.ledger.pending()
    finally:
        lock.execute("ROLLBACK") if lock.in_transaction else None
        lock.close()
        _wait_for_sqlite_threads(before)


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
    tmp_data_dir, migrated_db_template, monkeypatch
):
    """#295: the real loops, nothing idled. Under a write lock held by another
    connection, the lifespan's ledger flush, the ledger loop's last flush and
    the request-history writer (caught mid-write AND with rows still queued)
    all wait on the same lock, and so does the stats pruner, caught mid-prune
    every run (see _catch_the_pruner_mid_write). Together they must end inside
    Docker's 10 s stop grace, or the container is SIGKILLed instead of
    stopped."""
    from app.main import build_app

    go, in_write = _catch_the_pruner_mid_write(monkeypatch)
    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()
    db = tmp_data_dir / "vllm-warden.db"
    lock = sqlite3.connect(db, isolation_level=None)
    before = set(threading.enumerate())
    t0 = None
    try:
        with TestClient(app) as c:
            _ready(c, tmp_data_dir, dp=1)
            r, _ = _post(c)
            assert r.status_code == 200
            lock.execute("BEGIN IMMEDIATE")
            _hold_the_pruner(go, in_write)
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
        _wait_for_sqlite_threads(before)


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
