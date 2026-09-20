"""Live engine-metrics SSE — ``GET /api/stats/live`` (Plane A).

Scrapes EVERY loaded model's engine ``/metrics`` endpoint (aggregate Prometheus
text) and emits a parsed JSON frame over SSE, mirroring the header-metrics SSE
in ``app/header/routes_api.py`` (ticket auth, ``sse_headers``, ~2s cadence, 15s
keepalive, shared TTL cache so multi-tab collapses to one scrape).

FRAME SHAPE. ``{...leading model's block, "models": [block, ...]}`` — one block
per loaded model, each exactly what ``build_frame`` has always produced, plus
the leading block spread at the top level for a UI bundle older than the
``models`` key. ``build_frame`` itself is untouched and stays pinned by
tests/fixtures/live_frame_golden.json: eleven dashboard panels read it field by
field, so multi-model support had to be a wrapper around that shape rather than
a change to it.

WHICH models a viewer sees is the CLIENT's choice. Filtering here would put a
selection into the SSE ticket path, so every checkbox click would tear down and
re-mint the stream, and the frame already carries the per-model data — there is
nothing to gain by moving the choice to the server. Combining several blocks
into one number is likewise the client's job, and the ``| None`` contract below
is what makes that safe to do.

Two data planes back the live-stats dashboard (see docs/live-stats-spec.md).
This module is Plane A: the *aggregate* engine truth the engine exposes on
``127.0.0.1:{engine_port}/metrics``. Per-request KV/context truth is Plane B
(``app/stats/live_requests.py``), a separate module — they never share a file.

Design notes:

* **Dialect** — this module names **no engine metric at all**. Sub-project C
  moved the metric NAMES into each backend (``app/runtime/backends/*/metrics.py``)
  and the shared Prometheus text parser into ``app/stats/prometheus.py``. What
  arrives here is an ``EngineReading``: one field per concept, every field
  ``float | None``, where ``None`` means *this engine does not report it* — never
  zero. That is what lets llama.cpp, which publishes no KV gauge, no preemption
  counter and no latency histograms, degrade instead of lying.
* **Cache** — ``_MetricsCache`` keyed on ``model_id`` with a ~1.5s TTL and a
  single ``asyncio.Lock`` collapses N concurrent tabs to one scrape, mirroring
  ``_ProbeCache`` in ``app/system/routes_gpus.py``. The cached entry carries the
  scrape's monotonic timestamp so every tab derives per-second rates against a
  consistent clock.
* **Rates** — counters are cumulative; per-second rates (generation/prompt
  tokens, preemptions) and the interval prefix-cache hit rate are deltas vs the
  *connection's* previous frame. The first frame emits ``null`` rates. A counter
  going backwards (engine restart) also yields ``null`` for that tick.
* **Percentiles** — histogram p50/p90/p99 via cumulative ``_bucket`` count
  interpolation; means via ``_sum`` / ``_count``.
* **Fail-open** — a scrape failure is caught and surfaced as ``scrape_error`` on
  an otherwise-null frame; it never kills the stream (mirrors the header SSE).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.auth.stream_guard import guard_stream
from app.db.database import open_db
from app.models.routes_logs import require_sse_ticket
from app.runtime.backends import registry
from app.runtime.backends.metrics import EngineReading

# Re-exported: this module owned the parser until sub-project C, and
# tests/unit/stats/test_live_engine.py -- plus any operator snippet -- still
# imports the names from here. The parser now lives in app/stats/prometheus.py
# because both backends need it and neither should have to import the SSE route.
from app.stats.prometheus import (  # noqa: F401
    Metrics,
    PromSample,
    hist_mean,
    hist_quantile,
    parse_prometheus,
)
from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/stats", tags=["stats-live"])

# --------------------------------------------------------------------------- #
# Cadence / cache tuning
# --------------------------------------------------------------------------- #

# SSE emit cadence. Override via ``VW_STATS_LIVE_INTERVAL_S`` (float seconds).
# Floor at 0.5s so a misconfigured env can't pin a CPU (same clamp as the
# header metrics stream).
def _interval_seconds() -> float:
    raw = os.environ.get("VW_STATS_LIVE_INTERVAL_S")
    if not raw:
        return 2.0
    try:
        v = float(raw)
    except ValueError:
        return 2.0
    return max(0.5, v)

# SSE keepalive cadence — same rationale as the header stream. With the default
# 2s emit interval this never fires; it covers an operator who bumps the emit
# interval up for a quiet dashboard.
KEEPALIVE_INTERVAL_S: float = 15.0

# Shared-cache TTL. ~1.5s absorbs multi-tab burst polling into one scrape while
# staying fresh enough to feel live at the 2s emit cadence.
STATS_LIVE_CACHE_TTL_S: float = 1.5

# httpx timeout for the /metrics scrape. Short — the endpoint is loopback and a
# slow scrape should surface as scrape_error, not stall the whole stream.
SCRAPE_TIMEOUT_S: float = 5.0

def _pct(hist, q: float) -> float | None:
    if hist is None:
        return None
    buckets, _sum, _count = hist
    return hist_quantile(buckets, q)

def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den

def _int_or_none(v: float | None) -> int | None:
    return int(round(v)) if v is not None else None

# --------------------------------------------------------------------------- #
# Frame construction
# --------------------------------------------------------------------------- #

@dataclass
class RateState:
    """Per-connection snapshot of cumulative counters + scrape clock, used to
    derive per-second rates on the *next* frame."""

    monotonic: float
    gen: float | None
    prompt: float | None
    preempt: float | None
    prefix_hits: float | None
    prefix_queries: float | None

def _rate(cur: float | None, prev: float | None, dt: float) -> float | None:
    """Per-second rate of a cumulative counter, or ``None``.

    ``None`` when either endpoint is missing, ``dt`` is non-positive, or the
    counter went backwards (an engine restart reset it — a spurious spike we
    refuse to emit).
    """
    if cur is None or prev is None or dt <= 0 or cur < prev:
        return None
    return (cur - prev) / dt

#: What ``Metrics.histogram()`` returns: cumulative ``(le, count)`` pairs plus
#: the histogram's own sum and count. Named so the two helpers below can state
#: what they accept instead of taking a bare tuple.
Hist = tuple[list[tuple[float, float]], float | None, float | None]


def buckets_payload(hist: Hist | None) -> dict[str, Any] | None:
    """The engine's histogram buckets, as JSON.

    Emitted alongside the quantiles rather than instead of them. TTFT here is
    bimodal by construction -- prefix-cache hit against cold prefill -- so a
    single percentile lands in the valley between the two humps and describes a
    request that never happened. The buckets let the client draw the measured
    distribution and put the percentiles on it as markers.

    ``+Inf`` is encoded as ``None``: JSON has no infinity, and dropping that
    bucket would silently discard the entire tail, which for a latency chart is
    the half worth looking at.

    None for a missing histogram -- llama.cpp reports none at all, and zeros
    would render as a distribution in which every request was instantaneous.
    """
    if hist is None:
        return None
    buckets, total_sum, total_count = hist
    if not buckets:
        return None
    return {
        "le": [None if le == float("inf") else le for le, _ in buckets],
        "counts": [c for _, c in buckets],
        "count": total_count,
        "sum": total_sum,
    }

def bucket_deltas(
    newer: dict[str, Any] | None, older: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Two cumulative reads into "what happened between them", or None.

    Returns None rather than a wrong number in every case where subtraction
    would lie:

    * no earlier read -- the first frame's only honest answer is "not yet".
      Showing the lifetime total under a "last 5 minutes" heading is precisely
      the lifetime-as-live mislabelling this page is being rebuilt to remove.
    * counters went backwards -- an engine restart zeroes them, and subtracting
      gives negative bars.
    * boundaries moved -- an engine upgrade can change the bucket edges, and
      subtracting across them compares different bins while looking fine.

    A window in which nothing happened is NOT None: it is a distribution whose
    counts are all zero, so the panel can say "no requests in this window"
    instead of going blank, which is a different and more useful statement.
    """
    if newer is None or older is None:
        return None
    if newer.get("le") != older.get("le"):
        return None
    a, b = newer.get("counts") or [], older.get("counts") or []
    if len(a) != len(b):
        return None
    if any(x < y for x, y in zip(a, b, strict=True)):
        return None
    n_count, o_count = newer.get("count"), older.get("count")
    if n_count is None or o_count is None or n_count < o_count:
        return None
    n_sum, o_sum = newer.get("sum") or 0.0, older.get("sum") or 0.0
    return {
        "le": list(newer["le"]),
        "counts": [x - y for x, y in zip(a, b, strict=True)],
        "count": n_count - o_count,
        "sum": max(0.0, n_sum - o_sum),
    }

def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def _null_frame(
    model, model_id, max_model_len, scrape_error: str, backend: str | None = None
) -> dict:
    return {
        "ts": _now_iso(),
        "model": model,
        "model_id": model_id,
        # Sub-project C: the same additive key build_frame emits, so the two
        # frame shapes stay identical and the frontend never has to branch on
        # which one it got. ``None`` when no model could be resolved at all.
        "backend": backend,
        "max_model_len": max_model_len,
        "engine": None,
        "throughput": None,
        "cache": None,
        "latency": None,

        "finished": None,
        "scrape_error": scrape_error,
    }

def build_frame(
    r,  # EngineReading
    *,
    model: str | None,
    model_id: str | None,
    max_model_len: int | None,
    prev: RateState | None,
    scrape_monotonic: float,
    backend: str = "vllm",
) -> tuple[dict, RateState]:
    """Build the ``data:`` frame from one EngineReading; return ``(frame, state)``.

    This function no longer knows any engine's metric names -- the backend put
    them into ``r`` (app/runtime/backends/*/metrics.py). What stays here is what
    is the same for every engine: per-second rates against the connection's
    previous snapshot, histogram quantiles, and the frame schema the dashboard
    reads field by field.

    A None in ``r`` means the engine does not report that number. It flows
    straight through as JSON null and is NEVER defaulted to 0; the frontend
    renders null as an em dash or hides the row entirely.
    """
    dt = (scrape_monotonic - prev.monotonic) if prev else 0.0

    kv_used = (
        int(round(r.kv_cache_usage_perc * r.kv_tokens_total))
        if (r.kv_cache_usage_perc is not None and r.kv_tokens_total is not None)
        else None
    )

    engine = {
        "num_requests_running": _int_or_none(r.requests_running),
        "num_requests_waiting": _int_or_none(r.requests_waiting),
        "waiting_by_reason": {
            "capacity": _int_or_none(r.waiting_capacity),
            "deferred": _int_or_none(r.waiting_deferred),
        },
        "kv_cache_usage_perc": r.kv_cache_usage_perc,
        "kv_tokens_used": kv_used,
        "kv_tokens_total": _int_or_none(r.kv_tokens_total),
        "engine_sleep_state": _int_or_none(r.engine_sleep_state),
        "preemptions_total": _int_or_none(r.preemptions_total),
        "preemptions_per_s": _rate(
            r.preemptions_total, prev.preempt if prev else None, dt
        ),
    }

    throughput = {
        "prompt_tokens_per_s": _rate(
            r.prompt_tokens_total, prev.prompt if prev else None, dt
        ),
        "generation_tokens_per_s": _rate(
            r.generation_tokens_total, prev.gen if prev else None, dt
        ),
        "prompt_tokens_total": _int_or_none(r.prompt_tokens_total),
        "generation_tokens_total": _int_or_none(r.generation_tokens_total),
    }

    # Interval prefix-cache hit rate = delta hits / delta queries over the tick.
    interval_hit_rate: float | None = None
    if prev is not None and dt > 0:
        dh = _rate(r.prefix_cache_hits, prev.prefix_hits, dt)  # per-s, reuse reset guard
        dq = _rate(r.prefix_cache_queries, prev.prefix_queries, dt)
        if dh is not None and dq is not None and dq > 0:
            interval_hit_rate = dh / dq
    cache = {
        "prefix_hit_rate": interval_hit_rate,
        "prefix_hit_rate_cumulative": _ratio(
            r.prefix_cache_hits, r.prefix_cache_queries
        ),
        "mm_hit_rate_cumulative": _ratio(r.mm_cache_hits, r.mm_cache_queries),
        "external_prefix_hit_rate_cumulative": _ratio(
            r.external_prefix_cache_hits, r.external_prefix_cache_queries
        ),
    }

    ttft, itl, tpot, e2e = r.ttft_hist, r.itl_hist, r.tpot_hist, r.e2e_hist
    latency: dict[str, Any] = {
        "ttft_p50": _pct(ttft, 0.5),
        "ttft_p90": _pct(ttft, 0.9),
        "ttft_p99": _pct(ttft, 0.99),
        "ttft_mean": hist_mean(ttft[1], ttft[2]) if ttft else None,
        "itl_p50": _pct(itl, 0.5),
        "itl_p99": _pct(itl, 0.99),
        "tpot_p50": _pct(tpot, 0.5),
        "e2e_p50": _pct(e2e, 0.5),
        "e2e_p90": _pct(e2e, 0.9),
        "e2e_p99": _pct(e2e, 0.99),
    }

    # The raw buckets, so the client can render the measured distribution
    # rather than five numbers reduced from it. See buckets_payload.
    latency["buckets"] = {
        "ttft": buckets_payload(ttft),
        "itl": buckets_payload(itl),
        "tpot": buckets_payload(tpot),
        "e2e": buckets_payload(e2e),
    }

    # MFU is gone, panel and field together. `mfu_estimate` was a literal None
    # left for "a later iteration" -- nothing ever resolved peak device FLOPs --
    # and the cumulative FLOPs counter beside it was rendered through a
    # per-second formatter, so its unit was wrong even when its value was not
    # zero. Measured GPU utilisation and power (the host charts) answer the
    # question MFU was meant to answer, with numbers the box actually reports.

    finished = {
        "stop": _int_or_none(r.finished_stop),
        "length": _int_or_none(r.finished_length),
        "abort": _int_or_none(r.finished_abort),
    }

    frame = {
        "ts": _now_iso(),
        "model": model,
        "model_id": model_id,
        # Sub-project C's one additive field. The dashboard needs it to say
        # "llama.cpp does not report this" rather than showing a blank tile of
        # unknown provenance -- design spec §9.1.
        "backend": backend,
        "max_model_len": max_model_len,
        "engine": engine,
        "throughput": throughput,
        "cache": cache,
        "latency": latency,

        "finished": finished,
        "scrape_error": None,
    }
    state = RateState(
        monotonic=scrape_monotonic,
        gen=r.generation_tokens_total,
        prompt=r.prompt_tokens_total,
        preempt=r.preemptions_total,
        prefix_hits=r.prefix_cache_hits,
        prefix_queries=r.prefix_cache_queries,
    )
    return frame, state

# --------------------------------------------------------------------------- #
# Shared TTL scrape cache (mirror of ``_ProbeCache``, keyed by model_id)
# --------------------------------------------------------------------------- #

@dataclass
class ScrapeResult:
    # Sub-project C: an EngineReading, not a Metrics. The backend did the
    # name lookup; what reaches the stats layer is already dialect-free.
    metrics: EngineReading | None
    error: str | None
    monotonic: float

async def _default_fetch(host: str, port: int, at: float, backend) -> ScrapeResult:
    """Scrape the backend's metrics path once and parse. Never raises — a failed
    scrape is a ``ScrapeResult`` with ``error`` set and ``metrics=None``.

    The path comes from ``backend.capabilities.metrics_path`` rather than a
    literal: llama.cpp happens to use ``/metrics`` too, but it only serves it
    when ``--metrics`` was passed, and a future backend need not agree at all.
    """
    url = f"http://{host}:{port}{backend.capabilities.metrics_path}"
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
            r = await client.get(url)
        r.raise_for_status()
        return ScrapeResult(metrics=backend.parse_metrics(r.text), error=None, monotonic=at)
    except Exception as exc:  # noqa: BLE001 — any scrape failure is surfaced, not raised
        return ScrapeResult(
            metrics=None, error=str(exc) or exc.__class__.__name__, monotonic=at
        )

class _MetricsCache:
    """One in-process cache for the vLLM ``/metrics`` scrape, keyed by model_id.

    A single ``asyncio.Lock`` guards the cache read and the underlying scrape so
    concurrent requests inside the TTL window collapse to one HTTP GET — the
    multi-tab de-dup goal. The cached ``ScrapeResult`` carries the scrape's
    monotonic timestamp so every consumer derives rates against a shared clock.
    Lives on ``app.state.stats_live_cache`` so tests can inject a clock + fetch.
    """

    def __init__(
        self,
        *,
        ttl: float = STATS_LIVE_CACHE_TTL_S,
        clock=time.monotonic,
        fetch=_default_fetch,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._fetch = fetch
        self._lock = asyncio.Lock()
        self._cache: dict[str, ScrapeResult] = {}
        self.invocations = 0  # exposed for tests

    async def get(
        self, model_id: str, host: str, port: int, backend
    ) -> ScrapeResult:
        async with self._lock:
            now = self._clock()
            hit = self._cache.get(model_id)
            if hit is not None and (now - hit.monotonic) < self._ttl:
                return hit
            self.invocations += 1
            res = await self._fetch(host, port, now, backend)
            self._cache[model_id] = res
            return res

def _get_cache(request: Request) -> _MetricsCache:
    cache = getattr(request.app.state, "stats_live_cache", None)
    if cache is None:
        cache = _MetricsCache()
        request.app.state.stats_live_cache = cache
    return cache

async def _loaded_models(
    db_path,
) -> list[tuple[str, str, int | None, str | None]]:
    """``(model_id, served_model_name, max_model_len, backend)`` for EVERY
    loaded model, ordered by served name. Empty list when nothing is loaded.

    Same 'loaded' semantics as the header stream: ``models.status='loaded'``
    with a ``model_runtime`` row.

    This was ``_loaded_model``, singular, with ``LIMIT 1``. On a box running two
    engines the live dashboard therefore showed one of them, and *which* one was
    whatever SQLite happened to return first -- not a stable choice, let alone a
    stated one. The ordering is by served name for the same reason the header's
    is: this feeds a view that repaints every two seconds, and a list that
    reorders under the operator cannot be read.

    ``backend`` is selected raw -- NULL stays NULL here and ``registry.get``
    resolves it to the default (D6), so the fallback lives in exactly one place.
    """
    async with open_db(db_path) as db:
        cur = await db.execute(
            "SELECT m.id, m.served_model_name, m.max_model_len, m.backend "
            "FROM models m JOIN model_runtime r ON r.model_id = m.id "
            "WHERE m.status = 'loaded' "
            "ORDER BY m.served_model_name ASC"
        )
        rows = await cur.fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]

# --------------------------------------------------------------------------- #
# SSE endpoint
# --------------------------------------------------------------------------- #

def envelope(blocks: list[dict]) -> dict:
    """Wrap per-model blocks into the frame the dashboard receives.

    ``blocks`` are ``build_frame`` / ``_null_frame`` outputs, one per model,
    UNCHANGED -- which is the point. ``build_frame``'s shape is pinned by a
    committed golden (tests/fixtures/live_frame_golden.json) because eleven
    panels read it field by field, so multi-model support is a wrapper around
    it and never an edit to it.

    The envelope is the per-model list PLUS the leading block's keys spread at
    the top level. The duplication is deliberate and is the same bargain the
    header frame strikes: ui and api ship as separate images and skew in both
    directions, so a bundle that predates ``models`` keeps reading the frame it
    knows instead of rendering nothing while an engine serves.

    The client picks WHICH models it displays. Filtering here would mean the
    SSE ticket path carrying a selection, so every checkbox click would tear
    down and re-mint a stream -- and the data is already per-model in the
    frame, so there is nothing to gain by moving the choice to the server.
    Combining several models into one number is the client's job too, and the
    None-vs-zero rule below is what makes that safe.
    """
    lead = blocks[0] if blocks else _null_frame(None, None, None, "no model loaded")
    return {**lead, "models": blocks}

@router.get("/live")
async def stream_live(
    request: Request, stream_key: str = Depends(require_sse_ticket)
) -> StreamingResponse:
    """Stream live engine metrics as SSE events (one JSON frame per tick).

    Structure mirrors the header-metrics SSE: an immediate first frame, then an
    emit every ``_interval_seconds()`` with ``is_disconnected`` checks and a
    belt-and-suspenders 15s keepalive. Each tick resolves EVERY loaded model,
    pulls a (possibly cached) scrape per model, and builds one block each —
    computing rates against this connection's previous frame, per model.

    Rate state is per model_id, not per connection. Two engines' cumulative
    counters share no clock and no origin; a single previous snapshot would
    have produced a rate for whichever model happened to come second that was
    computed against the first model's totals -- a plausible-looking number
    with no meaning at all.
    """
    cache = _get_cache(request)
    settings = request.app.state.settings
    supervisor = request.app.state.supervisor
    interval = _interval_seconds()

    async def _one_block(
        row: tuple[str, str, int | None, str | None],
        prev: RateState | None,
    ) -> tuple[dict, RateState | None]:
        """Scrape one model (cached) and build its block.

        Returns ``(block, new_state_or_None)``. On failure (no port, scrape
        error) returns a null block with ``scrape_error`` set and ``None``
        state, so rate accounting for THAT model resumes cleanly once its
        engine is back without disturbing any other model's.
        """
        model_id, model_name, max_model_len, backend_name = row
        # The row's OWN backend decides both the metrics path and the metric
        # dialect. An UnknownBackendError here would be a row asking for a
        # backend this build lacks -- which the load path already refuses, so it
        # cannot happen for a *loaded* model; the SSE loop's own try/except
        # would surface it as a scrape_error rather than killing the stream.
        backend = registry.get(backend_name)
        port = supervisor.get_port(model_id)
        if port is None:
            return (
                _null_frame(
                    model_name,
                    model_id,
                    max_model_len,
                    "engine not running",
                    backend.capabilities.name,
                ),
                None,
            )
        host = supervisor.get_host(model_id) or "127.0.0.1"
        res = await cache.get(model_id, host, port, backend)
        if res.metrics is None:
            return (
                _null_frame(
                    model_name,
                    model_id,
                    max_model_len,
                    res.error or "scrape failed",
                    backend.capabilities.name,
                ),
                None,
            )
        return build_frame(
            res.metrics,
            model=model_name,
            model_id=model_id,
            max_model_len=max_model_len,
            prev=prev,
            scrape_monotonic=res.monotonic,
            backend=backend.capabilities.name,
        )

    async def _one_frame(prev: dict[str, RateState]) -> dict:
        """One tick: every loaded model, one block each. Mutates ``prev``."""
        rows = await _loaded_models(settings.db_path)
        if not rows:
            # `models: []`, and a top level that still carries the old
            # "no model loaded" scrape_error for a client that reads only that.
            return envelope([])
        blocks = []
        for row in rows:
            block, state = await _one_block(row, prev.get(row[0]))
            if state is not None:
                prev[row[0]] = state
            else:
                # A model whose scrape failed must not derive its next rate
                # against a snapshot from before the gap; drop its history the
                # same way the single-model loop dropped the connection's.
                prev.pop(row[0], None)
            blocks.append(block)
        return envelope(blocks)

    async def gen():
        # Registered for logout / revoke, and an admin token re-checked while
        # the stream is open (app/auth/stream_guard.py).
        async with guard_stream(request, stream_key):
            prev: dict[str, RateState] = {}
            last_yield_at = time.monotonic()
            # Immediate first frame so the consumer isn't blank for ``interval``s.
            try:
                yield f"data: {json.dumps(await _one_frame(prev))}\n\n"
                last_yield_at = time.monotonic()
            except Exception:  # noqa: BLE001 — first-tick failure falls through to the loop
                pass

            while True:
                if await request.is_disconnected():
                    return
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    return
                if await request.is_disconnected():
                    return
                try:
                    yield f"data: {json.dumps(await _one_frame(prev))}\n\n"
                    last_yield_at = time.monotonic()
                except Exception as exc:  # noqa: BLE001 — never let a frame error kill the stream
                    # `models: []`, NOT `[err]`: a block with no model_id would
                    # render as an anonymous panel in a per-model view. The error
                    # belongs to the tick, not to a model.
                    err = _null_frame(
                        None, None, None, str(exc) or exc.__class__.__name__
                    )
                    yield f"data: {json.dumps({**err, 'models': []})}\n\n"
                    last_yield_at = time.monotonic()
                if time.monotonic() - last_yield_at >= KEEPALIVE_INTERVAL_S:
                    yield ": keepalive\n\n"
                    last_yield_at = time.monotonic()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers())
