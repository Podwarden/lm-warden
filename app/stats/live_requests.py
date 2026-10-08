"""Live per-request registry snapshot — ``GET /api/stats/requests`` (Plane B).

Owner: dev-2. See docs/live-stats-spec.md § "Plane B".

JWT-gated plain-JSON snapshot of the in-flight request registry, aggregated by
token and by client IP. The frontend polls this ~1.5s (no SSE — keep the hot
path free of stream fan-out). Token *id* and *name* and client IP are
metadata only (the id is an opaque row id, used to link a row to the token's
details page — never emit token plaintext/hash/secret columns).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.proxy.request_registry import LiveRequest
from app.stats.routes_forest import ForestViewer, forest_viewer

router = APIRouter(prefix="/api/stats", tags=["stats-live"])


DEFAULT_PREFILL_TOK_S = 2000.0


def _prefill_slow(req: LiveRequest, elapsed_s: float, prefill_tok_s: float) -> bool:
    """HEURISTIC: still prefilling well past what the uncached part of the
    prompt should take. ``prefill_tok_s`` is a configured guess
    (VW_PREFILL_TOK_S_HINT), not a measurement of this engine."""
    if req.phase != "prefill" or not req.is_stream or prefill_tok_s <= 0:
        return False
    fresh = max(0, req.prompt_tokens - (req.cache_est_tokens or 0))
    return elapsed_s > max(3.0, 4.0 * fresh / prefill_tok_s)


def _fresh_tokens(req: LiveRequest) -> int:
    """Prompt tokens the engine must really compute (not estimated cached)."""
    return max(0, req.prompt_tokens - (req.cache_est_tokens or 0))


def _ahead_of(req: LiveRequest, live: list[LiveRequest]) -> tuple[int, int]:
    """``(count, fresh_tokens)`` of requests on the SAME model and replica that
    were admitted earlier and are still prefilling: what this one waits behind
    in the engine. Single-replica models all share rank None, so they compare
    on the model alone. O(n) per row over the handful of live requests."""
    count = fresh = 0
    for o in live:
        if (
            o is not req
            and o.phase == "prefill"
            and o.model_row_id == req.model_row_id
            and o.dp_rank == req.dp_rank
            and o.started_monotonic < req.started_monotonic
        ):
            count += 1
            fresh += _fresh_tokens(o)
    return count, fresh


def slow_cause(
    *,
    elapsed_s: float,
    slow: bool,
    ahead_count: int,
    expected_start_s: float | None,
    cache_est_pct: float | None,
) -> str | None:
    """Why a prefill is taking its time, for prefill rows only:
    ``queued`` -- others admitted earlier are still prefilling and we are within
    1.5x (+3 s) of the time that predicts; ``evicted_likely`` -- slow, alone,
    and a mostly-cached prompt that evidently was not; ``slow`` -- slow for no
    reason the warden can see."""
    if (
        ahead_count > 0
        and expected_start_s is not None
        and elapsed_s <= 1.5 * expected_start_s + 3.0
    ):
        return "queued"
    if slow and ahead_count == 0 and (cache_est_pct or 0.0) >= 0.5:
        return "evicted_likely"
    if slow:
        return "slow"
    return None


def _serialize(
    req: LiveRequest,
    now_monotonic: float,
    prefill_tok_s: float = DEFAULT_PREFILL_TOK_S,
    *,
    ahead: tuple[int, int] = (0, 0),
    rate_source: str = "hint",
    est_accuracy: float | None = None,
) -> dict[str, Any]:
    """One request row. ``context_tokens`` and ``context_pct`` are derived;
    only metadata (token id, token name, client IP) crosses the wire — no
    secrets. ``token_id`` is the row id the frontend links to the token's
    details page, not a hash/plaintext column."""
    context_tokens = req.prompt_tokens + req.completion_tokens
    context_pct: float | None = None
    if req.max_model_len:
        context_pct = round(context_tokens / req.max_model_len, 4)
    cache_est_pct: float | None = None
    if req.cache_est_tokens is not None and req.prompt_tokens:
        # The estimate is in engine tokens and the prompt may still be the
        # warden's own count (corrected on the first engine usage), so cap it.
        cache_est_pct = round(min(1.0, req.cache_est_tokens / req.prompt_tokens), 4)
    elapsed = now_monotonic - req.started_monotonic
    slow = _prefill_slow(req, elapsed, prefill_tok_s)
    prefilling = req.phase == "prefill" and req.is_stream
    expected_start_s = (
        round((ahead[1] + _fresh_tokens(req)) / prefill_tok_s, 1)
        if prefilling and prefill_tok_s > 0
        else None
    )
    cause = (
        slow_cause(
            elapsed_s=elapsed,
            slow=slow,
            ahead_count=ahead[0],
            expected_start_s=expected_start_s,
            cache_est_pct=cache_est_pct,
        )
        if prefilling
        else None
    )
    return {
        "id": req.id,
        "token_id": req.token_id,
        "token_name": req.token_name,
        "client_ip": req.client_ip,
        "model": req.model,
        "path": req.path,
        "prompt_tokens": req.prompt_tokens,
        "completion_tokens": req.completion_tokens,
        "context_tokens": context_tokens,
        "max_model_len": req.max_model_len,
        "context_pct": context_pct,
        "elapsed_s": round(now_monotonic - req.started_monotonic, 1),
        "queued_s": round(req.queued_s, 3) if req.queued_s is not None else None,
        "prefill_slow": slow,
        # Queue-ahead (live) and the model behind the numbers.
        "dp_rank": req.dp_rank,
        "ahead_count": ahead[0] if prefilling else 0,
        "ahead_fresh_tokens": ahead[1] if prefilling else 0,
        "expected_start_s": expected_start_s,
        "slow_cause": cause,
        "prefill_tok_s": round(prefill_tok_s, 1),
        "prefill_rate_source": rate_source,
        "est_decayed": req.est_decayed,
        "est_accuracy": est_accuracy,
        "phase": req.phase,
        "is_stream": req.is_stream,
        "orphan": req.orphan,
        "session_id": req.session_id,
        "session_source": req.session_source,
        "parent_session_id": req.parent_session_id,
        # The forest's id for this session: HMAC(cookie_secret, raw)[:16], the session_key its
        # history row gets (session forest spec §6.4). Hashed, so safe for a key holder too.
        "forest_session": req.session_key,
        # An ESTIMATE (vLLM reports the real hit only after the request ends).
        "cache_est_tokens": req.cache_est_tokens,
        "cache_est_pct": cache_est_pct,
    }


def _aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fold the serialized rows into by-token and by-IP buckets. Cheap in-memory
    pass — the poller hits this ~1.5s so it stays O(n) over live requests.

    Grouped by ``token_id`` (not name): names are reused across rotations
    (migration 0033), so two distinct tokens sharing a name must not collapse
    into one row. ``token_name`` rides along for display. ``token_id is None``
    is the anonymous bucket, same as the un-keyed request itself.
    """
    by_token: dict[str | None, dict] = {}
    by_ip: dict[str | None, dict] = {}
    for r in rows:
        tid = r["token_id"]
        t = by_token.setdefault(
            tid,
            {
                "token_id": tid,
                "token_name": r["token_name"],
                "requests": 0,
                "context_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )
        t["requests"] += 1
        t["context_tokens"] += r["context_tokens"]
        t["prompt_tokens"] += r["prompt_tokens"]
        t["completion_tokens"] += r["completion_tokens"]

        ip = r["client_ip"]
        p = by_ip.setdefault(ip, {"client_ip": ip, "requests": 0, "context_tokens": 0})
        p["requests"] += 1
        p["context_tokens"] += r["context_tokens"]

    by_token_list = sorted(by_token.values(), key=lambda x: x["context_tokens"], reverse=True)
    by_ip_list = sorted(by_ip.values(), key=lambda x: x["context_tokens"], reverse=True)
    return by_token_list, by_ip_list


@router.get("/requests")
async def live_requests(
    request: Request, viewer: ForestViewer = Depends(forest_viewer)
) -> dict[str, Any]:
    """Snapshot of in-flight /v1 requests + by-token / by-IP aggregations.

    A key holder's forest token sees only its own key's requests (and the
    aggregates are computed from those rows alone); admin sees all."""
    reg = getattr(request.app.state, "request_registry", None)
    live = reg.snapshot() if reg is not None else []
    if viewer.token_id is not None:
        live = [r for r in live if r.token_id == viewer.token_id]
    now = time.monotonic()
    hint = getattr(getattr(request.app.state, "settings", None), "prefill_tok_s_hint", None)
    rate = hint if isinstance(hint, int | float) and hint > 0 else DEFAULT_PREFILL_TOK_S
    learned = getattr(request.app.state, "prefill_model", None)
    rows = []
    for r in live:
        r_rate, source = (
            learned.rate_for(r.model_row_id, rate) if learned is not None else (rate, "hint")
        )
        accuracy = learned.accuracy_for(r.model_row_id) if learned is not None else None
        rows.append(
            _serialize(
                r,
                now,
                r_rate,
                ahead=_ahead_of(r, live) if r.phase == "prefill" else (0, 0),
                rate_source=source,
                est_accuracy=accuracy,
            )
        )
    by_token, by_ip = _aggregate(rows)
    return {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "count": len(rows),
        "requests": rows,
        "by_token": by_token,
        "by_ip": by_ip,
    }


@router.get("/v2/prefill-model")
async def prefill_model(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """What the warden has learned about this box's prefill and cache (updated
    about once a minute from finished requests; see app/stats/prefill_model.py)."""
    state = getattr(request.app.state, "prefill_model", None)
    hint = getattr(getattr(request.app.state, "settings", None), "prefill_tok_s_hint", None)
    rate = hint if isinstance(hint, int | float) and hint > 0 else DEFAULT_PREFILL_TOK_S
    if state is None:
        return {"updated_at": None, "hint_tok_s": rate, "models": {}}
    snap: dict[str, Any] = state.snapshot(rate)
    return snap
