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
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.cobs2.shape import Toolset


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
    #: ``waiting`` from admission until its body is forwarded, then
    #: ``running_no_stream``: the engine holds it but reports nothing until done.
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
    #: cobs2: content ids (exact / loose) of the assistant message this request
    #: generated; None = not a chat request or not captured.
    cobs2_reply_msg: str | None = None
    cobs2_reply_loose_msg: str | None = None
    #: cobs2 L1 (plan 3 task 5): the request's head id, last body node (exact),
    #: the lineage edge and the node it continued / forked / was compacted from,
    #: and whether it asked for a summary. None = not computed (fail-open).
    cobs2_head: str | None = None
    cobs2_node: str | None = None
    cobs2_parent_node: str | None = None
    cobs2_edge: str | None = None
    cobs2_compaction_request: bool | None = None
    #: The loose chain matched deeper than the exact one (Lineage._find): the
    #: harness rewrote history. In memory only, no column; feeds the
    #: harness_rewrite miss cause (app/cache_obs/classify.py).
    cobs2_loose_beyond_exact: bool = False
    #: Sessions pages (migration 0053): the lineage session used, the
    #: conversation id, the shape JSON (sizes/counts), the request that produced
    #: the parent node, the keyed (token, session) lookup ref, the offered
    #: toolset (hashes, families, sizes; names in memory only, for
    #: ``session_tools``), and whether the request was relayed upstream.
    cobs2_session: str | None = None
    cobs2_conv: str | None = None
    cobs2_shape: str | None = None
    cobs2_parent_req: str | None = None
    cobs2_sess_ref: str | None = None
    cobs2_toolset: Toolset | None = None
    relayed: bool = False
    upstream: str | None = None
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
    #: True when the thinking policy (token or model "off") forced
    #: enable_thinking=false on this request.
    thinking_forced_off: bool = False
    #: What the key's prompt diet cut from this request (JSON), or None.
    prompt_diet: str | None = None
    #: What the sensitive data guard did (JSON, no values: app/guard/proxy.py), or None.
    guard: str | None = None
    #: KV-cache tiers (ruling R16a: each prefix block counted once). ``served_gpu_tokens``
    #: is the engine's own ``cached_tokens``; the RAM / NVMe figures and ``load_s`` are
    #: the hits the KV connector recorded for this request (app/proxy/kv.py). All four
    #: stay None when the model has no KV tier or the store daemon did not answer.
    served_gpu_tokens: int | None = None
    served_ram_tokens: int | None = None
    served_nvme_tokens: int | None = None
    load_s: float | None = None
    #: Per-key KV sharing (plan 2): True when the request ran under an isolated key's scope,
    #: False for a sharing key, None when no KV tier applied. ``kv_iso_cost_tokens`` is the
    #: prompt tokens an isolated request recomputed that sharing would have served.
    kv_isolated: bool | None = None
    kv_iso_cost_tokens: int | None = None
    #: Prompt Lens layer 1 (analysis plan Task 1.3, migration 0063): the first
    #: broken part (``sys``, ``tools``, ``node:<i>``), its cause code, the span
    #: class and line of a changed system prompt line, the counterfactual and
    #: lost tokens and the count of further breaks. Codes and ints only; None =
    #: not computed (kvsig off, or a lens fault: fail-open).
    lens_break: str | None = None
    lens_cause: str | None = None
    lens_span_class: str | None = None
    lens_span_line: int | None = None
    lens_lost_tokens: int | None = None
    lens_cf_tokens: int | None = None
    lens_other_breaks: int | None = None
    #: Relays: the upstream's cache_creation_input_tokens and the cache_control
    #: marks counted in the relayed body.
    cache_write_tokens: int | None = None
    cache_marks: int | None = None
    #: Layer 3 (filled by analysis Task 1.4): repeated tool-result tokens,
    #: the request's max_tokens, the reply's reasoning tokens, size and calls,
    #: the per-call verdicts (``[[hash8, code], ...]``, at most 16) and the
    #: per-tool lint ints.
    lens_repeat_results: int | None = None
    max_tokens_req: int | None = None
    reasoning_tokens: int | None = None
    reply_chars: int | None = None
    calls_n: int | None = None
    calls_bad: list[Any] | None = None
    tools_lint: list[int] | None = None
    #: Prompt version (analysis Task 3.1): ``h:<name>`` from X-Prompt-Version,
    #: else the detected 8 hex id; None with kvsig off.
    lens_version: str | None = None
    #: Live out-token readings ``(monotonic, tokens_so_far)``, one per live tick
    #: (0.5 s) while tokens arrive: the source of the row's 5 s decode speed
    #: (app/proxy/live_tokens.py). Bounded; display only, never persisted.
    out_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=32))


#: Phases where the request has not reached the engine yet.
_NOT_FORWARDED = frozenset({"queued", "waiting", "blocked"})


def engine_progress_tokens(req: LiveRequest) -> int | None:
    """Tokens generated so far, as far as the warden can honestly tell.

    A streamed request's count is observed frame by frame. A non-streaming
    one (``running_no_stream``) is invisible until it ends, and vLLM 0.32
    exports no per-request series to read it from, so it is None (unknown),
    never 0. A request not yet forwarded has no engine progress either.
    """
    if req.phase in _NOT_FORWARDED or not req.is_stream:
        return None
    return req.completion_tokens


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
    "cobs2_divergence_cause",
    "cobs2_divergence_cause_own",
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
        "cobs2_reply_msg": req.cobs2_reply_msg,
        "cobs2_reply_loose_msg": req.cobs2_reply_loose_msg,
        "cobs2_head": req.cobs2_head,
        "cobs2_node": req.cobs2_node,
        "cobs2_parent_node": req.cobs2_parent_node,
        "cobs2_edge": req.cobs2_edge,
        "cobs2_compaction_request": req.cobs2_compaction_request,
        "cobs2_loose_beyond_exact": req.cobs2_loose_beyond_exact,
        "cobs2_session": req.cobs2_session,
        "cobs2_conv": req.cobs2_conv,
        "cobs2_shape": req.cobs2_shape,
        "cobs2_parent_req": req.cobs2_parent_req,
        "cobs2_sess_ref": req.cobs2_sess_ref,
        "cobs2_toolset": req.cobs2_toolset.digest if req.cobs2_toolset else None,
        "cobs2_toolset_tools": (
            [list(e) for e in req.cobs2_toolset.entries] if req.cobs2_toolset else None
        ),
        "cobs2_toolset_names": list(req.cobs2_toolset.names) if req.cobs2_toolset else None,
        "relayed": req.relayed,
        "upstream": req.upstream,
        "dp_rank": req.dp_rank,
        "inflight_same_rank_at_start": req.inflight_same_rank_at_start,
        "gap_s": round(req.gap_s, 3) if req.gap_s is not None else None,
        "thinking_forced_off": req.thinking_forced_off,
        "prompt_diet": req.prompt_diet,
        "guard": req.guard,
        "served_gpu_tokens": req.served_gpu_tokens,
        "served_ram_tokens": req.served_ram_tokens,
        "served_nvme_tokens": req.served_nvme_tokens,
        "load_s": req.load_s,
        "kv_isolated": None if req.kv_isolated is None else int(req.kv_isolated),
        "kv_iso_cost_tokens": req.kv_iso_cost_tokens,
        "lens_break": req.lens_break,
        "lens_cause": req.lens_cause,
        "lens_span_class": req.lens_span_class,
        "lens_span_line": req.lens_span_line,
        "lens_lost_tokens": req.lens_lost_tokens,
        "lens_cf_tokens": req.lens_cf_tokens,
        "lens_other_breaks": req.lens_other_breaks,
        "cache_write_tokens": req.cache_write_tokens,
        "cache_marks": req.cache_marks,
        "lens_repeat_results": req.lens_repeat_results,
        "max_tokens_req": req.max_tokens_req,
        "reasoning_tokens": req.reasoning_tokens,
        "reply_chars": req.reply_chars,
        "calls_n": req.calls_n,
        "calls_bad": list(req.calls_bad) if req.calls_bad is not None else None,
        "tools_lint": list(req.tools_lint) if req.tools_lint is not None else None,
        "lens_version": req.lens_version,
    }
    obs = req.cache_obs or {}
    for k in CACHE_OBS_KEYS:
        rec[k] = obs.get(k)
    return rec
