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

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.proxy.request_registry import LiveRequest

router = APIRouter(prefix="/api/stats", tags=["stats-live"])


def _serialize(req: LiveRequest, now_monotonic: float) -> dict:
    """One request row. ``context_tokens`` and ``context_pct`` are derived;
    only metadata (token id, token name, client IP) crosses the wire — no
    secrets. ``token_id`` is the row id the frontend links to the token's
    details page, not a hash/plaintext column."""
    context_tokens = req.prompt_tokens + req.completion_tokens
    context_pct: float | None = None
    if req.max_model_len:
        context_pct = round(context_tokens / req.max_model_len, 4)
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
        "phase": req.phase,
        "orphan": req.orphan,
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
            {"token_id": tid, "token_name": r["token_name"], "requests": 0,
             "context_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        t["requests"] += 1
        t["context_tokens"] += r["context_tokens"]
        t["prompt_tokens"] += r["prompt_tokens"]
        t["completion_tokens"] += r["completion_tokens"]

        ip = r["client_ip"]
        p = by_ip.setdefault(
            ip, {"client_ip": ip, "requests": 0, "context_tokens": 0}
        )
        p["requests"] += 1
        p["context_tokens"] += r["context_tokens"]

    by_token_list = sorted(by_token.values(), key=lambda x: x["context_tokens"], reverse=True)
    by_ip_list = sorted(by_ip.values(), key=lambda x: x["context_tokens"], reverse=True)
    return by_token_list, by_ip_list


@router.get("/requests")
async def live_requests(request: Request, _user: str = Depends(require_jwt)):
    """Snapshot of in-flight /v1 requests + by-token / by-IP aggregations."""
    reg = getattr(request.app.state, "request_registry", None)
    live = reg.snapshot() if reg is not None else []
    now = time.monotonic()
    rows = [_serialize(r, now) for r in live]
    by_token, by_ip = _aggregate(rows)
    return {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "count": len(rows),
        "requests": rows,
        "by_token": by_token,
        "by_ip": by_ip,
    }
