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
from dataclasses import dataclass, field
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
    #: Lifecycle: ``queued`` (waiting at the warden's admission gate), ``prefill``
    #: (forwarded, no content-bearing frame yet), then whichever kind the LATEST
    #: delta was: ``thinking`` (reasoning), ``answering`` (text), ``tool_call``.
    #: Older rows used ``decode`` for all three. A non-streaming request is
    #: ``waiting`` from admission on: the engine reports nothing until done.
    phase: str = "prefill"
    #: Whether the CLIENT asked for a stream (a non-stream client may still be
    #: streamed upstream by the runaway detector; it sees nothing either way).
    is_stream: bool = True
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
    #: Client session id (app/proxy/session_id.py::extract_session), shown on the
    #: dashboard. None when the client sent none.
    session_id: str | None = None
    #: ESTIMATED prompt tokens already in the engine's prefix cache, from this
    #: conversation's previous request on the same replica (PrefixMemory). vLLM
    #: reports the true figure only on the final usage chunk, so while a request
    #: is in flight this is a guess. None means no basis for one.
    cache_est_tokens: int | None = None
    #: MEASURED ``usage.prompt_tokens_details.cached_tokens`` from the final
    #: usage frame / non-stream body. None means the engine did not report it.
    cached_tokens: int | None = None
    #: Which signal produced the session id (or "prompt_hash" when the
    #: conversation was only inferred); see app/proxy/session_id.py.
    session_source: str | None = None
    #: Parent session of a subagent, display only.
    parent_session_id: str | None = None
    #: Session forest (spec 2026-10-05 §3). Hashed keys only; raw ids never leave the proxy.
    session_key: str | None = None
    parent_session_key: str | None = None
    turn_index: int | None = None
    batch_id: str | None = None
    tools_in: list[Any] = field(default_factory=list)
    tools_out: list[Any] = field(default_factory=list)
    #: Filled at deregister by app/cache_obs (spec section 3); None = not observed.
    cache_obs: dict[str, Any] | None = None
    #: The replica the request was routed to (None: unrouted / single replica).
    dp_rank: int | None = None
    #: Other admitted requests on the same model + replica when this one was
    #: admitted. 0 = it prefilled alone. None = not measured.
    inflight_same_rank_at_start: int | None = None
    #: Seconds since this conversation's previous request finished, when the
    #: prefix memory had a usable entry.
    gap_s: float | None = None
    #: True when the estimate was scaled down because the idle gap is one the
    #: learned model says the engine usually evicts (prefill_model.py).
    est_decayed: bool = False
    #: The estimate BEFORE any decay, so history can audit the raw guess.
    cache_est_raw_tokens: int | None = None


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


#: The cache-observation columns (app/cache_obs/record.py::observe_fields).
CACHE_OBS_KEYS: tuple[str, ...] = (
    "reusable_tokens",
    "reusable_tokens_fleet",
    "reusable_tokens_own",
    "cache_outcome",
    "cache_outcome_own",
    "diverged_at",
    "diverged_at_own",
    "cached_source",
    "cached_ttft_est_tokens",
)


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
    rec: dict[str, Any] = {
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
        # Measured vs estimated side by side, so the estimate can be audited.
        "cached_tokens": req.cached_tokens,
        "cache_est_tokens": (
            req.cache_est_raw_tokens
            if req.cache_est_raw_tokens is not None
            else req.cache_est_tokens
        ),
        "session_source": req.session_source,
        "parent_session_id": req.parent_session_id,
        "session_key": req.session_key,
        "parent_session_key": req.parent_session_key,
        "turn_index": req.turn_index,
        "batch_id": req.batch_id,
        "tools_in": list(req.tools_in),
        "tools_out": list(req.tools_out),
        "dp_rank": req.dp_rank,
        "inflight_same_rank_at_start": req.inflight_same_rank_at_start,
        "gap_s": round(req.gap_s, 3) if req.gap_s is not None else None,
    }
    obs = req.cache_obs or {}
    for k in CACHE_OBS_KEYS:
        rec[k] = obs.get(k)
    return rec
