"""The proxy's finished-request write path into the persisted history store.

Reported by the operator, 2026-09-05: "after a request ends it seems that all
charts and graphs reset". The registry holds only in-flight requests and drops
each one in the streaming `finally`, so duration, TTFT and finish reason — the
three things that only become knowable at that instant — were discarded there.
They now go to SQLite through ``app.state.request_history``.

The endpoints that read the store are covered in test_v2_requests.py.
"""

import asyncio
import sqlite3
import time

from app.proxy.request_registry import LiveRequest, finished_record
from app.stats.request_history import RequestHistoryStore
from tests.conftest import jwt_login, seed_admin_user


def test_the_app_boots_a_store_and_a_writer(tmp_data_dir, client):
    client.get("/healthz")
    store = client.app.state.request_history
    assert isinstance(store, RequestHistoryStore)
    assert store.db_path == tmp_data_dir / "vllm-warden.db"


def test_a_recorded_request_reaches_sqlite_without_the_request_waiting(tmp_data_dir, client):
    """`record` returns at once; the background writer lands the row."""
    client.get("/healthz")
    store = client.app.state.request_history
    req = LiveRequest(
        id="req-1", token_id="t1", token_name="key-a", client_ip="10.0.0.1",
        model="model-a", model_row_id="id-model-a", path="/v1/chat/completions",
        prompt_tokens=12, max_model_len=4096, started_monotonic=100.0,
        started_iso="2026-09-05T18:59:00Z", completion_tokens=7,
        first_token_monotonic=100.4, finish_reason="stop",
    )
    assert store.record(finished_record(req, now=102.0)) is True
    db_path = tmp_data_dir / "vllm-warden.db"
    # 30s, not 5: what is under test is that `record` RETURNS at once and the
    # row lands eventually -- neither claim is about how fast the background
    # writer drains. At 5s this failed intermittently under `pytest -n auto`
    # on a loaded runner, which read as a broken store rather than a busy box.
    deadline = time.time() + 30
    row = None
    while time.time() < deadline:
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT model_id, model, ttft_s, duration_s, finish_reason "
                "FROM request_history WHERE id = 'req-1'"
            ).fetchone()
        if row:
            break
        time.sleep(0.05)
    assert row == ("id-model-a", "model-a", 0.4, 2.0, "stop")


def test_a_build_without_the_store_says_so_rather_than_500ing(tmp_data_dir, client):
    """The old panel's contract survives: the endpoint reports itself absent
    and the page renders its empty state and explains itself."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    if hasattr(client.app.state, "request_history"):
        delattr(client.app.state, "request_history")
    body = client.get("/api/stats/live/finished", headers=auth).json()
    assert body == {"requests": [], "retained_seconds": 0, "available": False}


async def test_record_is_safe_to_call_from_the_streaming_finally(tmp_path):
    """No loop-bound work, no awaits, no exceptions -- even before the writer
    has started, and even for a record the store cannot use."""
    store = RequestHistoryStore(tmp_path / "x.db")
    assert store.record({"id": "a", "model_id": "m", "model": "m"}) is True
    assert store.record({}) is False
    assert store.record(None) is False  # type: ignore[arg-type]
    assert store.pending() == 1
    # A writer started later still drains what was queued before it existed.
    task = asyncio.create_task(store.run_forever())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---- admission-queue wait (0032) -------------------------------------------


def test_finished_record_carries_the_admission_queue_wait():
    req = LiveRequest(
        id="req-q", token_id="t1", token_name="key-a", client_ip="10.0.0.1",
        model="model-a", model_row_id="id-model-a", path="/v1/chat/completions",
        prompt_tokens=12, max_model_len=4096, started_monotonic=100.0,
        started_iso="2026-09-10T18:59:00Z", completion_tokens=7,
        first_token_monotonic=100.4, finish_reason="stop", queued_s=1.2345,
    )
    rec = finished_record(req, now=102.0)
    assert rec["queued_s"] == 1.234  # rounded like every other timing field
    # It is NOT inside ttft_s: the clock ttft is measured from is read after
    # admission, so the wait was never added in and must not be subtracted.
    assert rec["ttft_s"] == 0.4
    assert rec["duration_s"] == 2.0


def test_an_unmeasured_queue_wait_stays_none_rather_than_becoming_zero():
    # A request registered by a build or a path that never set it. Zero would
    # be a claim ("the engine had a free slot"); None is the honest absence.
    req = LiveRequest(
        id="req-n", token_id="t1", token_name="key-a", client_ip="10.0.0.1",
        model="model-a", model_row_id="id-model-a", path="/v1/chat/completions",
        prompt_tokens=12, max_model_len=4096, started_monotonic=100.0,
        started_iso="2026-09-10T18:59:00Z",
    )
    assert finished_record(req, now=101.0)["queued_s"] is None


def test_the_queue_wait_round_trips_through_the_store(tmp_data_dir, client):
    client.get("/healthz")
    store = client.app.state.request_history
    req = LiveRequest(
        id="req-rt", token_id="t1", token_name="key-a", client_ip="10.0.0.1",
        model="model-a", model_row_id="id-model-a", path="/v1/chat/completions",
        prompt_tokens=12, max_model_len=4096, started_monotonic=100.0,
        started_iso="2026-09-10T18:59:00Z", completion_tokens=7,
        first_token_monotonic=100.4, queued_s=0.75,
    )
    assert store.record(finished_record(req, now=102.0)) is True
    db_path = tmp_data_dir / "vllm-warden.db"
    # Generous deadline: this waits on the store's BACKGROUND writer to drain,
    # not on anything latency-sensitive. Five seconds (the older tests in this
    # file) is not enough under `pytest -n auto` on a loaded runner, which is
    # how this flaked in CI.
    deadline = time.time() + 30
    row = None
    while time.time() < deadline:
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT queued_s FROM request_history WHERE id = 'req-rt'"
            ).fetchone()
        if row:
            break
        time.sleep(0.05)
    assert row == (0.75,)
