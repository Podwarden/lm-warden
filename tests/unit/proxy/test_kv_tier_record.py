"""KV-cache tier (step 1): the per-turn tier columns of ``request_history``.

Rulings R16a / R24 / R23: each prefix block is counted once. ``served_ram_tokens`` /
``served_nvme_tokens`` / ``load_s`` are the hits the connector recorded in the store
daemon under the LM Warden request id; vLLM's ``cached_tokens`` already counts those, so
``served_gpu_tokens = max(cached - ram - nvme, 0)``. A daemon that answered with nothing
recorded (``{}``) means GPU only: ram = nvme = 0, ``load_s`` NULL. A daemon that is down,
slow or failing (None) leaves all four NULL and never delays the request.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any
from unittest.mock import AsyncMock, patch

from app.proxy.kv import apply_hits, fetch_hits
from app.proxy.request_id import REQUEST_ID_HEADER
from app.proxy.request_registry import LiveRequest, finished_record
from tests.conftest import wait_until
from tests.unit.proxy.test_kv_params import _kv_on
from tests.unit.proxy.test_messages import AUTH, _json_upstream, _ready, _sse_upstream

CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
USAGE = {
    "prompt_tokens": 6000,
    "completion_tokens": 1,
    "total_tokens": 6001,
    "prompt_tokens_details": {"cached_tokens": 5216},
}
OK = {
    "id": "c1",
    "object": "chat.completion",
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": USAGE,
}
HITS = {"ram_tokens": 3168, "nvme_tokens": 0, "load_s": 0.12}
COLS = "served_gpu_tokens, served_ram_tokens, served_nvme_tokens, load_s"


class FakeKv:
    """Stands in for ``lmw_kv.client.Client``: ``pop_hits`` only."""

    def __init__(self, hits: Any = HITS, *, raises: bool = False, gate=None) -> None:
        self.hits = hits
        self.raises = raises
        self.gate = gate
        self.tags: list[str] = []
        self.threads: list[int] = []
        self.returned = 0

    def pop_hits(self, tag: str) -> dict[str, Any] | None:
        self.threads.append(threading.get_ident())
        self.tags.append(tag)
        try:
            if self.gate is not None:
                self.gate.wait(10)
            if self.raises:
                raise OSError("daemon gone")
            return self.hits
        finally:
            self.returned += 1


def _daemon(client, tmp_data_dir, fake: FakeKv | None) -> None:
    _kv_on(client, tmp_data_dir)
    sup = client.app.state.kv_store
    sup.client = lambda: fake  # healthy with ``fake``; None = daemon down


def _go(client, body=CHAT, upstream=None):
    send = AsyncMock(return_value=upstream or _json_upstream(OK))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers=AUTH, json=body)
    return r


def _row(tmp_data_dir, rid: str):
    db_path = tmp_data_dir / "vllm-warden.db"

    def _get():
        with sqlite3.connect(db_path) as db:
            return db.execute(
                f"SELECT {COLS}, cached_tokens FROM request_history WHERE id = ?", (rid,)
            ).fetchone()

    return wait_until(_get, what="history row")


# -- pure -----------------------------------------------------------------------


def _live(**kw) -> LiveRequest:
    return LiveRequest(
        id="r1",
        token_id="t",
        token_name="t",
        client_ip=None,
        model="m",
        model_row_id="m",
        path="/v1/chat/completions",
        prompt_tokens=1,
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="x",
        **kw,
    )


def test_finished_record_emits_the_tier_fields():
    req = _live()
    rec = finished_record(req, now=1.0)
    assert [rec[k] for k in ("served_gpu_tokens", "served_ram_tokens")] == [None, None]
    assert rec["served_nvme_tokens"] is None and rec["load_s"] is None
    req.served_gpu_tokens, req.served_ram_tokens, req.served_nvme_tokens = 1, 2, 3
    req.load_s = 0.5
    rec = finished_record(req, now=1.0)
    assert (rec["served_gpu_tokens"], rec["served_ram_tokens"]) == (1, 2)
    assert (rec["served_nvme_tokens"], rec["load_s"]) == (3, 0.5)


def test_apply_hits_gpu_is_cached_minus_the_store_hits():
    req = _live(cached_tokens=5216)
    apply_hits(req, HITS)
    assert (req.served_gpu_tokens, req.served_ram_tokens, req.served_nvme_tokens) == (
        2048,
        3168,
        0,
    )
    assert req.load_s == 0.12
    req = _live(cached_tokens=4000)
    apply_hits(req, {"ram_tokens": 1000, "nvme_tokens": 2000, "load_s": 0.3})
    assert (req.served_gpu_tokens, req.served_ram_tokens, req.served_nvme_tokens) == (
        1000,
        1000,
        2000,
    )


def test_apply_hits_gpu_clamps_at_zero():
    req = _live(cached_tokens=100)
    apply_hits(req, {"ram_tokens": 3168, "nvme_tokens": 512, "load_s": 0.2})
    assert (req.served_gpu_tokens, req.served_ram_tokens, req.served_nvme_tokens) == (
        0,
        3168,
        512,
    )


def test_apply_hits_empty_means_gpu_only():
    req = _live(cached_tokens=2048)
    apply_hits(req, {})
    assert (req.served_gpu_tokens, req.served_ram_tokens, req.served_nvme_tokens) == (
        2048,
        0,
        0,
    )
    assert req.load_s is None


def test_apply_hits_none_or_malformed_leaves_all_four_none():
    for bad in (None, {"ram_tokens": "x", "nvme_tokens": 0, "load_s": 0.1}, {"ram_tokens": 1}, [1]):
        req = _live(cached_tokens=10)
        apply_hits(req, bad)  # type: ignore[arg-type]
        assert req.served_gpu_tokens is None and req.served_ram_tokens is None
        assert req.served_nvme_tokens is None and req.load_s is None


def test_fetch_hits_fails_open():
    assert fetch_hits(FakeKv(), "r") == HITS
    assert fetch_hits(FakeKv(raises=True), "r") is None
    assert fetch_hits(None, "r") is None


# -- forward --------------------------------------------------------------------


def test_nonstream_turn_records_the_tiers(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    fake = FakeKv()
    _daemon(client, tmp_data_dir, fake)
    r = _go(client)
    assert r.status_code == 200
    rid = r.headers[REQUEST_ID_HEADER]
    assert _row(tmp_data_dir, rid) == (2048, 3168, 0, 0.12, 5216)
    assert fake.tags == [rid]
    # Off the event loop: the client blocks on a socket.
    loop_thread = client.portal.call(threading.get_ident)
    assert fake.threads and fake.threads[0] != loop_thread


def test_stream_turn_records_the_tiers(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    fake = FakeKv({"ram_tokens": 512, "nvme_tokens": 1024, "load_s": 0.4})
    _daemon(client, tmp_data_dir, fake)
    up = _sse_upstream(
        [
            {"choices": [{"index": 0, "delta": {"content": "he"}}]},
            {"choices": [{"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": USAGE},
        ]
    )
    send = AsyncMock(return_value=up)
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream(
            "POST", "/v1/chat/completions", headers=AUTH, json={**CHAT, "stream": True}
        ) as r:
            b"".join(r.iter_bytes())
            rid = r.headers[REQUEST_ID_HEADER]
    assert _row(tmp_data_dir, rid) == (3680, 512, 1024, 0.4, 5216)
    assert fake.tags == [rid]


def test_daemon_down_leaves_the_tiers_null(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _daemon(client, tmp_data_dir, None)
    r = _go(client)
    assert r.status_code == 200
    assert _row(tmp_data_dir, r.headers[REQUEST_ID_HEADER]) == (None, None, None, None, 5216)


def test_daemon_failing_leaves_the_tiers_null(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _daemon(client, tmp_data_dir, FakeKv(raises=True))
    r = _go(client)
    assert r.status_code == 200
    assert _row(tmp_data_dir, r.headers[REQUEST_ID_HEADER]) == (None, None, None, None, 5216)


def test_no_hits_recorded_is_a_gpu_only_turn(tmp_data_dir, client):
    # R23: the daemon answered {} -- nothing was restored from the store this turn.
    _ready(client, tmp_data_dir)
    _daemon(client, tmp_data_dir, FakeKv({}))
    r = _go(client)
    assert r.status_code == 200
    assert _row(tmp_data_dir, r.headers[REQUEST_ID_HEADER]) == (5216, 0, 0, None, 5216)


def test_a_failed_pop_leaves_the_tiers_null(tmp_data_dir, client):
    # R23: None is "the call failed or timed out", never "nothing recorded".
    _ready(client, tmp_data_dir)
    _daemon(client, tmp_data_dir, FakeKv(None))
    r = _go(client)
    assert r.status_code == 200
    assert _row(tmp_data_dir, r.headers[REQUEST_ID_HEADER]) == (None, None, None, None, 5216)


def test_kv_off_never_asks_the_daemon(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    fake = FakeKv()
    client.app.state.kv_store.client = lambda: fake
    r = _go(client)
    assert r.status_code == 200
    assert _row(tmp_data_dir, r.headers[REQUEST_ID_HEADER]) == (None, None, None, None, 5216)
    assert fake.tags == []


def test_a_slow_daemon_never_delays_the_response(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    gate = threading.Event()
    fake = FakeKv(gate=gate)
    _daemon(client, tmp_data_dir, fake)
    try:
        r = _go(client)  # returns while pop_hits is still blocked on the gate
        assert r.status_code == 200
        assert fake.returned == 0  # the response did not wait for the daemon
        wait_until(lambda: fake.tags, what="pop_hits called")
    finally:
        gate.set()
    assert _row(tmp_data_dir, r.headers[REQUEST_ID_HEADER]) == (2048, 3168, 0, 0.12, 5216)
