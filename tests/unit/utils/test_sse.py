"""Unit tests for ``app.utils.sse``.

The header dict is small and the helper is intentionally trivial — these
tests exist mainly to pin the contract so a "let's simplify" pass can't
quietly drop one of the headers and re-introduce the proxy-buffering
regression fixed for log streams in #50.
"""

from __future__ import annotations

from app.utils.sse import sse_headers


def test_sse_headers_contains_x_accel_buffering_off():
    """X-Accel-Buffering: no is the load-bearing header — without it
    nginx (and lookalikes) buffer text/event-stream responses and
    swallow events until a buffer threshold is hit."""
    h = sse_headers()
    assert h["X-Accel-Buffering"] == "no"


def test_sse_headers_contains_cache_control_no_cache():
    """Cache-Control: no-cache is defensive cover for CDNs/proxies
    that don't recognise the SSE media type and would otherwise
    apply default caching to the response body."""
    h = sse_headers()
    assert h["Cache-Control"] == "no-cache"


def test_sse_headers_returns_fresh_dict_each_call():
    """Each caller gets its own dict. If we ever returned a module-level
    constant by reference, a caller doing
    ``StreamingResponse(headers={**sse_headers(), **extra})`` would be
    fine but a careless ``h = sse_headers(); h.update(...)`` would
    mutate the shared default for every other endpoint."""
    a = sse_headers()
    b = sse_headers()
    assert a is not b
    a["X-Accel-Buffering"] = "TAMPERED"
    assert b["X-Accel-Buffering"] == "no"
