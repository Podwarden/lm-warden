"""A backend with no metrics endpoint is "not supported", not a scrape failure.

MLX has ``metrics_path=None``. The live frame must say so with
``metrics_supported: false`` and NO ``scrape_error`` (which drives the amber
"Metrics scrape failed" banner), and busy/idle must come from the proxy's own
in-flight registry. A vLLM-like backend whose scrape really fails keeps its
``scrape_error``.
"""

from __future__ import annotations

import httpx
import pytest

from app.proxy.request_registry import LiveRequest, RequestRegistry
from app.stats.live_engine import (
    _default_fetch,
    _null_frame,
    proxy_in_flight,
    unsupported_frame,
)
from tests.unit.stats.test_live_scrape_no_metrics_path import _FakeBackend


def _req(rid, model_row_id, phase="answering"):
    return LiveRequest(
        id=rid,
        token_id=None,
        token_name="k",
        client_ip=None,
        model="served",
        model_row_id=model_row_id,
        path="/v1/chat/completions",
        prompt_tokens=1,
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="2026-01-01T00:00:00Z",
        phase=phase,
    )


def test_unsupported_frame_is_not_a_scrape_error():
    f = unsupported_frame("qwen", "m1", 4096, "mlx", running=4, waiting=0)
    assert f["scrape_error"] is None
    assert f["metrics_supported"] is False
    assert "no metrics endpoint" in f["metrics_unsupported_reason"]
    assert f["backend"] == "mlx"
    assert f["engine"]["num_requests_running"] == 4
    assert f["engine"]["num_requests_waiting"] == 0
    # Everything the engine would report stays null, never zero.
    assert f["engine"]["kv_cache_usage_perc"] is None
    assert f["throughput"] is None and f["latency"] is None


@pytest.mark.asyncio
async def test_failing_scrape_on_a_metrics_backend_keeps_scrape_error(monkeypatch):
    def handler(request):
        return httpx.Response(500, text="boom")

    real = httpx.AsyncClient

    def client(*a, **kw):
        kw.setdefault("transport", httpx.MockTransport(handler))
        return real(*a, **kw)

    monkeypatch.setattr("app.stats.live_engine.httpx.AsyncClient", client)
    res = await _default_fetch("127.0.0.1", 8001, 1.0, _FakeBackend(metrics_path="/metrics"))
    assert res.metrics is None and res.error
    frame = _null_frame("vllm-model", "m2", 4096, res.error, "vllm")
    assert frame["scrape_error"] == res.error
    assert frame.get("metrics_supported") is None  # not flagged unsupported


@pytest.mark.asyncio
async def test_proxy_in_flight_counts_per_model_and_splits_queued():
    reg = RequestRegistry()
    for r in (
        _req("a", "m1"),
        _req("b", "m1", "prefill"),
        _req("c", "m1", "queued"),
        _req("d", "other"),
    ):
        await reg.register(r)
    assert proxy_in_flight(reg, "m1") == (2, 1)
    assert proxy_in_flight(reg, "nobody") == (0, 0)
    assert proxy_in_flight(None, "m1") == (0, 0)
