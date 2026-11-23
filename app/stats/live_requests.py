"""Live per-request registry snapshot — ``GET /api/stats/requests`` (Plane B).

Owner: dev-2. See docs/live-stats-spec.md § "Plane B".

JWT-gated plain-JSON snapshot of the in-flight request registry, aggregated by
token and by client IP. The frontend polls this ~1.5s (no SSE — keep the hot
path free of stream fan-out). Token *id* and *name* and client IP are
metadata only (the id is an opaque row id, used to link a row to the token's
details page — never emit token plaintext/hash/secret columns).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.proxy.live_tokens import decode_tps
from app.proxy.request_registry import LiveRequest, engine_progress_tokens
from app.stats.prefill_model import own_state as _own_learned
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


def _own_estimate(req: LiveRequest, learned: Any) -> LiveRequest:
    """The in-flight cache estimate with the idle-gap decay learned from this
    key's own requests (the proxy applied the fleet's)."""
    raw = req.cache_est_raw_tokens
    if raw is None:
        return req
    factor = learned.decay_for(req.model_row_id, req.gap_s) if learned is not None else None
    est = int(raw * factor) if factor is not None and raw else raw
    return dataclasses.replace(
        req, cache_est_tokens=est, est_decayed=factor is not None and bool(raw)
    )


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
    # B1: engine TTFT (slot held -> first frame, includes the engine's own
    # queue) and the client's TTFT (the warden's admission queue added).
    ttft_s = (
        round(req.first_token_monotonic - req.started_monotonic, 3)
        if req.first_token_monotonic is not None
        else None
    )
    ttft_total_s = round(ttft_s + (req.queued_s or 0.0), 3) if ttft_s is not None else None
    tps = decode_tps(
        list(getattr(req, "out_samples", ()) or ()),
        first_token=req.first_token_monotonic,
        now=now_monotonic,
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
        "ttft_s": ttft_s,
        "ttft_total_s": ttft_total_s,
        # Tokens of every generated channel so far (content, thinking, tool
        # calls), and this request's decode speed over the last 5 s (B1).
        "out_tokens": req.completion_tokens,
        "decode_tps_5s": round(tps, 1) if tps is not None else None,
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
        "thinking_forced_off": req.thinking_forced_off,
        "est_accuracy": est_accuracy,
        "phase": req.phase,
        # Generated tokens so far; null for a non-streaming request in flight
        # (``running_no_stream``) -- the engine is working, the warden cannot see.
        "engine_progress_tokens": engine_progress_tokens(req),
        "is_stream": req.is_stream,
        "orphan": req.orphan,
        "session_id": req.session_id,
        "session_source": req.session_source,
        "parent_session_id": req.parent_session_id,
        # The forest's id for this session: HMAC(cookie_secret, raw)[:16], the session_key its
        # history row gets (session forest spec §6.4). Hashed, so safe for a key holder too.
        "forest_session": req.session_key,
        # Sessions pages: a link to this request's session page (resolved by the
        # API once its first turn is recorded); null without a session.
        "session_ref": ("r_" + req.id) if getattr(req, "cobs2_session", None) else None,
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


def _groups(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    """The Requests table's live Group by (stats redesign B2).

    ``group`` is ``key``, ``ip`` or ``model``. Each group carries how many of
    its requests are in flight, how many wait at the warden's queue, their
    context tokens, and ``live_tps``: the SUM of the running requests' 5 s
    decode speeds (None when none is decoding: idle, not zero). A key group
    never mixes models -- one group per key and model -- so a speed always
    belongs to one model. An IP group carries no model.
    """
    acc: dict[tuple[Any, Any], dict[str, Any]] = {}
    for r in rows:
        if group == "key":
            k: tuple[Any, Any] = (r["token_id"], r["model"])
            label = r["token_name"]
        elif group == "model":
            k = (r["model"], r["model"])
            label = r["model"]
        else:
            k = (r["client_ip"], None)
            label = r["client_ip"]
        g = acc.get(k)
        if g is None:
            g = acc[k] = {
                "key": k[0],
                "label": label,
                "model": k[1],
                "in_flight": 0,
                "queued": 0,
                "context_tokens": 0,
                "live_tps": None,
            }
        g["in_flight"] += 1
        if r.get("phase") == "queued":
            g["queued"] += 1
        g["context_tokens"] += r.get("context_tokens") or 0
        tps = r.get("decode_tps_5s")
        if tps is not None:
            g["live_tps"] = round((g["live_tps"] or 0.0) + tps, 1)
    return sorted(acc.values(), key=lambda g: (-g["in_flight"], str(g["label"])))


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
    tok = viewer.token_id
    own = tok is not None
    if tok is not None and live:
        # A key holder: everything learned comes from its own requests only.
        learned = await asyncio.to_thread(_own_learned, request.app.state.settings.db_path, tok)
    rows = []
    for r in live:
        r_rate, source = (
            learned.rate_for(r.model_row_id, rate) if learned is not None else (rate, "hint")
        )
        accuracy = learned.accuracy_for(r.model_row_id) if learned is not None else None
        ahead = _ahead_of(r, live) if r.phase == "prefill" else (0, 0)
        row = _serialize(
            _own_estimate(r, learned) if own else r,
            now,
            r_rate,
            ahead=ahead,
            rate_source=source,
            est_accuracy=accuracy,
        )
        if own:
            # Never another key's queue; no learned rate is a null, not the hint.
            row["ahead_count"] = None
            row["ahead_fresh_tokens"] = None
            # "queued" rests on the queue count (D6: never shown to a key
            # holder); the expected start stays, the cause goes.
            if row.get("slow_cause") == "queued":
                row["slow_cause"] = None
            if source != "learned":
                row["prefill_tok_s"] = None
                row["prefill_rate_source"] = None
        rows.append(row)
    by_token, by_ip = _aggregate(rows)
    return {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "count": len(rows),
        "requests": rows,
        "by_token": by_token,
        "by_ip": by_ip,
        # B2: the live columns of the Requests table's Group by.
        "groups": {g: _groups(rows, g) for g in ("key", "ip", "model")},
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
