"""200 concurrent proxied requests against a fake upstream: zero failures and exact
ledger totals after flushing, with a writer holding the DB lock for the whole run (#279)."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from app.proxy import ledger as ledger_mod
from tests.unit.proxy.ledger_helpers import flush_ledger
from tests.unit.proxy.test_dp_routing_forward import (
    AUTH,
    _ready,
    _resp,
    _sender,
)

N = 200


def test_200_concurrent_requests_under_a_held_write_lock_lose_nothing(
    tmp_data_dir, client, monkeypatch
):
    _ready(client, tmp_data_dir, dp=1, affinity=0)
    db_path = tmp_data_dir / "vllm-warden.db"
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    send, _seen = _sender(_resp())
    ledger = client.app.state.ledger

    # The background flush must actually run into the lock: a short busy wait
    # turns the held lock into "database is locked" well inside the hold, and
    # the ledger's busy hook (which logs the I1 retry warning) is the signal.
    monkeypatch.setattr(ledger_mod, "LEDGER_BUSY_TIMEOUT_MS", 50)
    busy_hit = threading.Event()
    real_note_busy = ledger._note_busy

    def note_busy():
        busy_hit.set()
        real_note_busy()

    monkeypatch.setattr(ledger, "_note_busy", note_busy)

    locked, release = threading.Event(), threading.Event()

    def hold_lock():
        con = sqlite3.connect(db_path, timeout=5, isolation_level=None)
        try:
            con.execute("BEGIN IMMEDIATE")
            locked.set()
            release.wait(30)
            con.execute("COMMIT")
        finally:
            con.close()

    def one(_):
        return client.post("/v1/chat/completions", headers=AUTH, json=body).status_code

    holder = threading.Thread(target=hold_lock)
    try:
        with patch("httpx.AsyncClient.send", new=send):
            holder.start()
            assert locked.wait(5)
            # Every request is served while the write lock is held.
            with ThreadPoolExecutor(32) as pool:
                statuses = list(pool.map(one, range(N)))
            assert statuses == [200] * N
            # Still held: the totals are in memory, not on disk, and the
            # ledger's flush has hit the lock (and kept the batch).
            assert ledger.pending() is True
            assert busy_hit.wait(10), "the ledger flush never ran into the held lock"
            assert ledger.pending() is True
            with sqlite3.connect(db_path) as db:
                (on_disk,) = db.execute("SELECT COALESCE(SUM(requests),0) FROM counters").fetchone()
            assert on_disk == 0
    finally:
        release.set()
        holder.join()

    # flush() never raises; with the lock gone one call lands everything.
    flush_ledger(client)
    assert ledger.pending() is False

    with sqlite3.connect(db_path) as db:
        (reqs,) = db.execute("SELECT COALESCE(SUM(requests),0) FROM counters").fetchone()
        (minute,) = db.execute(
            "SELECT COALESCE(SUM(requests),0) FROM token_usage_minute"
        ).fetchone()
    assert reqs == N
    assert minute == N
