"""A backend with no metrics endpoint must not be scraped.

``_default_fetch`` builds the scrape URL from
``backend.capabilities.metrics_path``. When that is ``None`` (an engine that
exposes no ``/metrics``), the old code interpolated the literal ``"None"``
into ``http://host:portNone``, which httpx rejects as an invalid port — the
scrape_error frame then carried that misleading text. The fix short-circuits
before any HTTP: no request, an honest error naming that the backend exposes
no metrics endpoint.

This pins two things:

  * ``metrics_path is None``  → no HTTP request, ``metrics=None``, error set.
  * ``metrics_path == "/metrics"`` → unchanged: exactly one GET to that path.
"""

from __future__ import annotations

import httpx
import pytest

from app.runtime.backends import BackendCapabilities
from app.stats.live_engine import ScrapeResult, _default_fetch


class _FakeBackend:
    """A Backend-Protocol stand-in whose only moving part is metrics_path."""

    def __init__(self, *, metrics_path: str | None) -> None:
        self.capabilities = BackendCapabilities(
            name="fake",
            display_name="Fake",
            supports_tensor_parallel=False,
            supports_pipeline_parallel=False,
            max_gpus=None,
            openai_paths=frozenset(),
            health_path="/health",
            metrics_path=metrics_path,
            supports_request_priority=False,
            supports_lora=False,
            supports_vision=False,
            needs_local_model_path=False,
            supports_version_pin=False,
            env_prefixes=(),
            env_exact=frozenset(),
            vram_cap_fraction=None,
        )
        self.parse_metrics_calls = 0

    def parse_metrics(self, body: str):  # noqa: ARG002 — shape matches the Protocol
        self.parse_metrics_calls += 1
        return None


@pytest.mark.asyncio
async def test_none_metrics_path_makes_no_request(monkeypatch):
    """A backend exposing ``metrics_path=None`` is skipped, not scraped.

    Fails on the unpatched tree: ``_default_fetch`` interpolates the literal
    ``"None"`` into the URL, httpx refuses it ("Invalid port: '8001None'")
    before the transport, and the error does not name the missing metrics
    endpoint. The no-request leg kills a fix that still sends ``GET /None``.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="# no metrics\n")

    # Patch the AsyncClient constructor used inside _default_fetch so any
    # scrape attempt is forced through the recording transport. If the fix
    # short-circuits before constructing a client, the constructor is never
    # called and `requests` stays empty — which is what we assert.
    real_async_client = httpx.AsyncClient

    def recording_client(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("app.stats.live_engine.httpx.AsyncClient", recording_client)

    backend = _FakeBackend(metrics_path=None)
    at = 12.5
    res = await _default_fetch("127.0.0.1", 8001, at, backend)

    assert isinstance(res, ScrapeResult)
    # No HTTP request was issued — the whole point of the fix.
    assert requests == [], f"expected no request, got {[r.url for r in requests]}"
    # parse_metrics is never called when there is no metrics path to scrape.
    assert backend.parse_metrics_calls == 0
    # metrics is None and the error names the missing endpoint.
    assert res.metrics is None
    assert res.error is not None
    assert "metrics" in res.error.lower()
    # The monotonic clock passes through untouched.
    assert res.monotonic == at


@pytest.mark.asyncio
async def test_metrics_path_still_scrapes(monkeypatch):
    """A backend WITH a metrics path is scraped as before — one GET, no skip.

    This is the unchanged-behaviour leg: the fix must not regress backends that
    DO expose ``/metrics``.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="# metrics\n")

    real_async_client = httpx.AsyncClient

    def recording_client(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("app.stats.live_engine.httpx.AsyncClient", recording_client)

    backend = _FakeBackend(metrics_path="/metrics")
    at = 99.0
    res = await _default_fetch("127.0.0.1", 8002, at, backend)

    # Exactly one GET to the backend's metrics path.
    assert len(requests) == 1
    assert requests[0].url.path == "/metrics"
    assert requests[0].url.host == "127.0.0.1"
    assert requests[0].url.port == 8002
    # No error, metrics parsed (parse_metrics returned None, which is the
    # reading for an unrecognised body — fine: the contract is "no error").
    assert res.error is None
    assert res.monotonic == at
