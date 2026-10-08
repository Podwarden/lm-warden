"""Live header metrics SSE: ``GET /api/header/metrics/stream``.

Powers the compact VRAM% + GPU% + active-model badge mounted in the nav
chrome (frontend/src/components/header-metrics.tsx). Emits one JSON event
every ``VW_HEADER_METRICS_INTERVAL_S`` seconds (default 2.0).

The payload is intentionally a small superset of what the badge renders
today so future iterations (e.g. tooltip with per-GPU breakdown) don't
need a new endpoint::

    {
      "ts": "2026-05-23T19:01:02.345Z",
      "gpus": [{"index": 0, "name": "NVIDIA RTX A4000",
                "memory_used_mib": 12450, "memory_total_mib": 16376,
                "utilization_pct": 87}],
      "vram_used_mib": 12450,
      "vram_total_mib": 16376,
      "vram_pct": 76,                   # int 0-100, pooled across all cards
      "gpu_util_pct": 87,               # the BUSIEST card, not an average
      "active_models": [                # EVERY model, most significant first
        {"id": "61c82bbc...", "served_model_name": "llama-3.1-8b",
         "status": "loaded"},
        {"id": "da98f08a...", "served_model_name": "qwen3.8-27b",
         "status": "loaded"}
      ],
      # The three below predate multi-model and mirror active_models[0]. Kept
      # so an older UI bundle against a newer api shows one correct model
      # instead of "idle" while two engines serve.
      "active_model": "llama-3.1-8b",   # or null
      "active_model_id": "61c82bbc...", # or null
      "active_model_status": "loaded",  # 'loaded'|'failed'|'loading'|null
      "probe_error": null               # passthrough from nvidia-smi probe
                                        # (the four vram/gpu numbers are null
                                        # when it is set or no card answered)
    }

Auth: SSE ticket (same pattern as model log streams). The widget is
hidden on /login and /setup on the client; this endpoint stays JWT-gated
via ticket mint so the contract is symmetric — no GPU info leaks to
unauthenticated callers.

Data source: shares ``app.state.gpu_probe_cache`` with
``/api/system/gpus`` (2s TTL absorbs burst polling so multiple tabs
collapse to one ``nvidia-smi`` invocation). Active-model lookup joins
``models`` to ``model_runtime``: a ``status='loaded'`` row surfaces only
with a runtime sibling, while ``loading`` and ``failed`` rows surface
without one (they have none yet) so the badge can read "loading" or go
red on a crash instead of claiming "idle". See ``_active_models``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.auth.stream_guard import guard_stream
from app.db.database import open_db
from app.models.routes_logs import require_sse_ticket
from app.system.routes_gpus import _ProbeCache
from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/header", tags=["header"])


# Default emit cadence. Override via ``VW_HEADER_METRICS_INTERVAL_S`` env var
# (float seconds). Floor at 0.5s so a misconfigured env can't pin a CPU.
def _interval_seconds() -> float:
    raw = os.environ.get("VW_HEADER_METRICS_INTERVAL_S")
    if not raw:
        return 2.0
    try:
        v = float(raw)
    except ValueError:
        return 2.0
    return max(0.5, v)


# SSE keepalive cadence — same rationale as routes_logs.KEEPALIVE_INTERVAL_S.
# Header metrics emit every 2s by default so the keepalive practically never
# fires, but a slow probe (e.g. nvidia-smi cold start) could still leave a
# gap long enough for an idle proxy to drop the connection.
KEEPALIVE_INTERVAL_S: float = 15.0


def _get_cache(request: Request) -> _ProbeCache:
    """Reuse the same probe cache as ``/api/system/gpus``.

    The cache lives on ``app.state.gpu_probe_cache`` (lazy-init in
    ``app/system/routes_gpus._get_cache``). We avoid importing the
    private helper directly so a future relocation of the cache owner
    doesn't break this module silently.
    """
    cache = getattr(request.app.state, "gpu_probe_cache", None)
    if cache is None:
        cache = _ProbeCache()
        request.app.state.gpu_probe_cache = cache
    return cache


async def _active_models(
    db_path: str,
) -> list[tuple[str, str, str]]:
    """Every model the badge should report, as ``(id, served_model_name,
    status)``, most significant first. Empty list when the box is idle.

    This was ``_active_model``, singular, ending in ``LIMIT 1`` under a comment
    reading "the supervisor enforces single-model loading, so multiple
    ``loaded`` rows would be a defect we don't paper over here". That is no
    longer the product. The operator serves a vLLM model on one card and a
    llama.cpp model on the other, simultaneously, and the header named exactly
    one of them. The LIMIT was not a display choice -- it was a single-model
    assumption compiled into SQL, so no amount of frontend work could have
    shown the second engine.

    Three states qualify, and they are NOT symmetric:

    - ``loaded`` — requires a ``model_runtime`` sibling. That row is what
      proves the supervisor actually owns a live engine; a ``loaded`` row
      without one is stale state we must not advertise as serving.
    - ``loading`` — has NO ``model_runtime`` sibling yet, because the
      supervisor writes that only once the engine answers. Surfacing it is
      the whole point: a 27B load takes minutes, and reporting "idle" for
      that window tells the operator the opposite of what is happening.
    - ``failed`` — also has no sibling. This is the state that matters most
      and used to be the most invisible: after an engine death the row sits
      in ``failed`` indefinitely (nothing auto-restores it), and a header
      reading "idle" was indistinguishable from a clean, healthy box.

    Hence the LEFT JOIN plus an explicit per-status predicate rather than a
    plain INNER JOIN — the strictness is kept exactly where it matters.

    Precedence is ``loaded`` > ``loading`` > ``failed``, so a live engine is
    never repainted by an unrelated stale failure and an operator's retry
    (``failed`` -> ``loading``) is visibly acknowledged. With several rows that
    precedence stops being a filter and becomes an ORDER: everything is
    reported, and the leading entry is the one the legacy singular fields name.

    The within-class tiebreak is ``served_model_name``, NOT ``updated_at``. The
    badge repaints every two seconds; ``updated_at`` moves under a running
    fleet, so ordering by it would visibly reshuffle a multi-model header
    whenever a row was merely touched. A name is stable, and it also makes the
    chip order match the order the operator reads elsewhere.
    """
    async with open_db(db_path) as db:
        cur = await db.execute(
            "SELECT m.id, m.served_model_name, m.status "
            "FROM models m LEFT JOIN model_runtime r ON r.model_id = m.id "
            "WHERE (m.status = 'loaded' AND r.model_id IS NOT NULL) "
            "   OR m.status IN ('loading', 'failed') "
            "ORDER BY CASE m.status "
            "         WHEN 'loaded' THEN 0 "
            "         WHEN 'loading' THEN 1 "
            "         ELSE 2 END, "
            "         m.served_model_name ASC"
        )
        rows = await cur.fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _payload(snap, actives: list[tuple[str, str, str]]) -> dict:
    """One SSE frame.

    ``actives`` is ``_active_models``' output. The legacy singular fields are
    DERIVED from its first entry rather than passed alongside it, so the frame
    cannot contain a plural list and a singular name that disagree.
    """
    gpus = [
        {
            "index": g.index,
            "name": g.name,
            "memory_used_mib": g.memory_used_mib,
            "memory_total_mib": g.memory_total_mib,
            "utilization_pct": g.utilization_pct,
        }
        for g in snap.gpus
    ]
    # Both aggregates are questions about the BOX, not about a model, so N
    # models across N cards does not change what either one means.
    #
    #   vram_pct  pooled fill: used across all cards / total across all cards.
    #             Still the right number with several models -- it answers "is
    #             there room for another one", which is a whole-box question.
    #   gpu_util  the BUSIEST card, and it must stay a max rather than become
    #             an average. Averaging 90% and 0% reports 45%, a number that
    #             describes neither card and reads as comfortable while a model
    #             saturates its GPU. The per-card breakdown is in the tooltip,
    #             where a number per card can be labelled with its card.
    #
    # No reading is not a zero (#255). When the probe failed, or reported no
    # cards, all four are null so the badge shows "--" instead of a calm 0 %
    # beside a model that is still serving -- the state a revoked GPU grant
    # produces, where the next model load is about to fail.
    used: int | None = None
    total: int | None = None
    vram_pct: int | None = None
    util: int | None = None
    if snap.gpus and not snap.probe_error:
        used_mib = sum(g.memory_used_mib for g in snap.gpus)
        total_mib = sum(g.memory_total_mib for g in snap.gpus)
        used, total = used_mib, total_mib
        vram_pct = int(round(100.0 * used_mib / total_mib)) if total_mib else None
        util = max(g.utilization_pct for g in snap.gpus)
    first = actives[0] if actives else (None, None, None)
    return {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gpus": gpus,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "vram_pct": vram_pct,
        "gpu_util_pct": util,
        # EVERY model the box is serving, starting, or has crashed. Always a
        # list, never null, so a client can map over it without a null check.
        "active_models": [
            {"id": mid, "served_model_name": name, "status": status}
            for mid, name, status in actives
        ],
        # The three fields that predate multi-model, kept because a UI bundle
        # can outrun its api pod and vice versa. They name the LEADING entry,
        # so an older bundle shows one correct model instead of "idle" while
        # two engines serve.
        "active_model": first[1],
        "active_model_id": first[0],
        # 'loaded' | 'loading' | 'failed' | null. Always present so the
        # frontend never has to tell "key absent" (old build) from "idle".
        "active_model_status": first[2],
        "probe_error": snap.probe_error,
    }


@router.get("/metrics/stream")
async def stream_metrics(
    request: Request, stream_key: str = Depends(require_sse_ticket)
) -> StreamingResponse:
    """Stream live VRAM%, GPU utilisation, and active-model name as SSE events."""
    cache = _get_cache(request)
    settings = request.app.state.settings
    interval = _interval_seconds()

    async def gen():
        # Registered for logout / revoke, and an admin token re-checked while
        # the stream is open (app/auth/stream_guard.py).
        async with guard_stream(request, stream_key):
            last_yield_at = time.monotonic()
            # Emit one immediate frame so the consumer doesn't sit blank for
            # ``interval`` seconds waiting on the first tick.
            try:
                snap = await cache.get()
                actives = await _active_models(settings.db_path)
                yield f"data: {json.dumps(_payload(snap, actives))}\n\n"
                last_yield_at = time.monotonic()
            except Exception:  # noqa: BLE001
                # If the very first probe errors, fall through to the loop
                # below; the next attempt will surface a probe_error payload.
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
                    snap = await cache.get()
                    actives = await _active_models(settings.db_path)
                    yield f"data: {json.dumps(_payload(snap, actives))}\n\n"
                    last_yield_at = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    # A probe / DB failure shouldn't kill the stream — emit a
                    # probe_error event and keep ticking. The consumer renders
                    # the badge in a "degraded" state on probe_error != null.
                    err_payload = {
                        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "gpus": [],
                        # Nothing was measured: null, never 0 (#255).
                        "vram_used_mib": None,
                        "vram_total_mib": None,
                        "vram_pct": None,
                        "gpu_util_pct": None,
                        # A list, matching the happy path. A client that maps over
                        # active_models must not crash on the one frame shape it
                        # only ever sees when something is already going wrong.
                        "active_models": [],
                        "active_model": None,
                        "active_model_id": None,
                        "active_model_status": None,
                        "probe_error": str(exc) or exc.__class__.__name__,
                    }
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    last_yield_at = time.monotonic()
                # Belt-and-suspenders keepalive — when the emit interval is
                # tuned shorter than KEEPALIVE_INTERVAL_S (the default 2s
                # case) this never fires; it exists for the operator who
                # bumps the env var to 30s for a quiet dashboard.
                if time.monotonic() - last_yield_at >= KEEPALIVE_INTERVAL_S:
                    yield ": keepalive\n\n"
                    last_yield_at = time.monotonic()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
