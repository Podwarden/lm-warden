"""In-process live request registry (Plane B).

Owner: dev-2. See docs/live-stats-spec.md § "Plane B".

Tracks every in-flight /v1 proxy request with token name, client IP, model,
context tokens (prompt + streamed completion) vs max_model_len, elapsed, phase
(prefill/decode), and an orphan flag (client disconnected but forward still
draining). Single uvicorn worker; lock-light — a short ``asyncio.Lock`` guards
insert/delete only, field updates during streaming are plain attribute writes
directly on the ``LiveRequest`` the caller holds (last-write-wins tolerated by
the reader, same rationale as ``ActiveRequestCounter.count()``). Absorbs
ActiveRequestCounter's ``count()`` so ``GET /api/admin/active-requests`` and its
Playwright test stay green.

Registration must be FAIL-OPEN: no registry error may ever break a proxied
request — every hook call in ``app/proxy/routes.py`` wraps this in try/except.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class LiveRequest:
    """One in-flight /v1 request. Metadata only — never a token secret.

    ``completion_tokens`` / ``phase`` / ``orphan`` are mutated in place by the
    streaming loop in ``_forward`` without taking the registry lock (the reader
    tolerates a one-tick-stale field). ``prompt_tokens`` and identity fields are
    fixed at register time.
    """

    id: str
    token_id: str | None
    token_name: str | None
    client_ip: str | None
    #: The SERVED model name -- what the client asked for and what a human
    #: reads in the requests table.
    model: str
    #: The models table's row id. Kept separately because it, not the served
    #: name, is what every stats endpoint filters on: /api/stats/v2/overview
    #: validates ``?models=`` against this id (``_resolve_selection``), so a
    #: finished record keyed by the served name silently matches nothing
    #: whenever the two differ.
    model_row_id: str
    path: str
    prompt_tokens: int
    max_model_len: int | None
    started_monotonic: float
    started_iso: str
    completion_tokens: int = 0
    phase: str = "prefill"
    orphan: bool = False
    #: Monotonic clock at the first streamed frame — the proxy's own TTFT.
    #: Measured here rather than taken from the engine because llama.cpp
    #: publishes no latency histogram at all, and this is the one vantage point
    #: that sees every backend identically.
    first_token_monotonic: float | None = None
    #: Set from the terminal SSE frame (or the non-streaming body) when it
    #: carries one. None means "not observed", never "stop" by assumption.
    finish_reason: str | None = None
    #: Seconds spent waiting at the proxy's per-engine admission gate
    #: (app/proxy/scheduler.py) before this request was forwarded. Measured in
    #: `_forward` around the acquire alone, so it excludes tokenization, which
    #: precedes it.
    #:
    #: NOT part of `ttft_s`: `started_monotonic` above is read once the slot is
    #: held, so the wait was never inside TTFT and subtracting it would be
    #: wrong. It is the WARDEN's queue only -- an engine's own waiting queue
    #: (vLLM's continuous batching) is inside TTFT and invisible from here.
    #:
    #: None means not measured, which is not the reading "waited zero seconds".
    queued_s: float | None = None
    #: The model VARIANT the running engine was launched as (0036,
    #: app/runtime/variants.py). None when not resolved.
    variant_id: str | None = None


class RequestRegistry:
    """Live in-flight /v1 request registry.

    The dict is the single source of truth. ``register`` / ``deregister`` take
    a short lock so an insert and a delete that land in the same tick can't
    corrupt the mapping; ``snapshot`` / ``count`` are lock-free reads (a racy
    snapshot may be one entry off mid-insert, which the ~1.5s poller tolerates).
    Live field updates are done by mutating the returned ``LiveRequest`` object
    directly, no registry method required.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._reqs: dict[str, LiveRequest] = {}

    async def register(self, req: LiveRequest) -> None:
        async with self._lock:
            self._reqs[req.id] = req

    async def deregister(self, req_id: str) -> None:
        async with self._lock:
            self._reqs.pop(req_id, None)

    def get(self, req_id: str) -> LiveRequest | None:
        return self._reqs.get(req_id)

    def count(self) -> int:
        """Back-compat with ActiveRequestCounter — number in flight."""
        return len(self._reqs)

    def snapshot(self) -> list[LiveRequest]:
        """Lock-free copy of the current in-flight requests."""
        return list(self._reqs.values())


def finished_record(req: LiveRequest, *, now: float) -> dict[str, Any]:
    """The row the finished-request ring keeps once ``req`` has ended.

    Extracted from the proxy's ``_deregister`` so the record's shape is
    testable on its own. It was inline, and the ring's tests inject records of
    their own making, so nothing checked what the proxy actually wrote --
    which is how ``model_id`` came to hold the served model name.

    ``now`` is a monotonic reading, taken by the caller at the moment the
    request ended.
    """
    ttft = (
        req.first_token_monotonic - req.started_monotonic
        if req.first_token_monotonic is not None
        else None
    )
    return {
        "id": req.id,
        # Served name for display, row id for filtering. See LiveRequest.
        "model": req.model,
        "model_id": req.model_row_id,
        "token_name": req.token_name,
        # The row id, not just the name: names are reused across rotations
        # (migration 0033), so only the id says which key made the request.
        "token_id": req.token_id,
        "client_ip": req.client_ip,
        "prompt_tokens": req.prompt_tokens,
        "completion_tokens": req.completion_tokens,
        "duration_s": round(now - req.started_monotonic, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "finish_reason": req.finish_reason,
        "orphan": req.orphan,
        "started_iso": req.started_iso,
        "queued_s": round(req.queued_s, 3) if req.queued_s is not None else None,
        "variant_id": req.variant_id,
    }
