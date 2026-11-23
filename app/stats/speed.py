"""Per-request latency and speed for the Stats page (stats redesign B3).

One place defines every figure the page's Latency and speed tiles, the latency
chart and the Requests table's window columns show, so the operator's view and
the key holder's view can never disagree about what "decode speed" means.

Definitions (docs/live-stats-spec.md, design section 3.3):

  ttft      ``queued_s + ttft_s``: the warden's admission queue plus slot held
            to the first token (which includes the engine's own queue). What
            the client waited. Streaming requests only.
  decode    ``(completion - 1) / (duration - ttft_s)``, completion >= 16.
            One request's share of a batched engine, not the engine total.
  prefill   ``(prompt - cached) / ttft_s``: COMPUTED tokens only, so a cache
            hit cannot inflate it. Needs the engine-measured ``cached_tokens``
            and at least 512 fresh tokens. Only requests that prefilled alone
            (``inflight_same_rank_at_start == 0``) count toward its summary;
            the others are returned as points marked ``prefill_alone=false``.
  prompt    ``prompt / ttft_s`` (cache effects included). The KEY HOLDER's
            figure: computing tokens needs ``cached_tokens``, which on a
            shared prefix cache can reveal another key's prompt.
  duration  ``duration_s``, slot held to the last byte.

Low-volume rules (``summarize``): n = 0 none; 1 to 4 the values themselves;
5 to 19 p50 and max; 20 and more p50 and p95; p99 from 100.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import aiosqlite

from app.stats.request_history import LOCAL_ONLY, _model_clause, quantile

MIN_DECODE_TOKENS = 16
MIN_FRESH_PREFILL = 512
POINTS_LIMIT = 2000
GROUPS = ("key", "ip", "model")


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def row_metrics(r: dict[str, Any]) -> dict[str, Any]:
    """Every per-request figure of one history row (None = not defined for it)."""
    ttft = _num(r.get("ttft_s"))
    queued = _num(r.get("queued_s")) or 0.0
    dur = _num(r.get("duration_s"))
    prompt = int(r.get("prompt_tokens") or 0)
    comp = int(r.get("completion_tokens") or 0)
    cached = r.get("cached_tokens")
    out: dict[str, Any] = {
        "ttft": round(ttft + queued, 3) if ttft is not None else None,
        "decode": None,
        "prefill": None,
        "prefill_alone": r.get("inflight_same_rank_at_start") == 0,
        "prompt_speed": None,
        "duration": dur,
    }
    if ttft is not None and dur is not None and comp >= MIN_DECODE_TOKENS and dur > ttft:
        out["decode"] = round((comp - 1) / (dur - ttft), 1)
    if ttft is not None and ttft > 0:
        out["prompt_speed"] = round(prompt / ttft, 1) if prompt > 0 else None
        if cached is not None:
            fresh = prompt - int(cached)
            if fresh >= MIN_FRESH_PREFILL:
                out["prefill"] = round(fresh / ttft, 1)
    return out


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """The low-volume rules. ``values`` in arrival order."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mode": "none"}
    if n < 5:
        return {"n": n, "mode": "list", "values": [round(v, 3) for v in values]}
    s = sorted(values)
    p50 = quantile(s, 0.5)
    if n < 20:
        return {"n": n, "mode": "max", "p50": p50, "max": s[-1]}
    return {
        "n": n,
        "mode": "p95",
        "p50": p50,
        "p95": quantile(s, 0.95),
        "p99": quantile(s, 0.99) if n >= 100 else None,
    }


def _p50(values: list[float]) -> float | None:
    """A group's p50, only when 5 or more requests back it."""
    return quantile(sorted(values), 0.5) if len(values) >= 5 else None


_COLS = (
    "id",
    "finished_at",
    "model_id",
    "model",
    "token_id",
    "token_name",
    "client_ip",
    "prompt_tokens",
    "completion_tokens",
    "duration_s",
    "ttft_s",
    "queued_s",
    "cached_tokens",
    "inflight_same_rank_at_start",
    "finish_reason",
)


async def query_speed_rows(
    db: aiosqlite.Connection,
    *,
    since: float,
    model_ids: Sequence[str] | None,
    token_id: str | None = None,
    until: float | None = None,
) -> list[dict[str, Any]]:
    """Every local row of the window, oldest first (uncapped: a percentile over a
    sample is not the window's). ``token_id`` narrows to one key's rows;
    ``until`` ends the window (default: now)."""
    where, args = _model_clause(model_ids)
    sql = "SELECT " + ", ".join(_COLS) + " FROM request_history WHERE finished_at >= ?" + where
    params: tuple[Any, ...] = (since, *args)
    if until is not None:
        sql += " AND finished_at <= ?"
        params += (until,)
    if token_id is not None:
        sql += " AND token_id = ?"
        params += (token_id,)
    cur = await db.execute(sql + " ORDER BY finished_at ASC, id ASC", params)
    return [dict(zip(_COLS, r, strict=True)) for r in await cur.fetchall()]


#: Window rows are re-read at most once per bucket: the page refreshes every
#: 30 s and, with Group by on, asks twice (the model and the group), and a 7d
#: window on a busy install is a large read (review finding 19).
CACHE_BUCKET_S = 15.0
_CACHE_MAX = 32
_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def clear_cache() -> None:
    _cache.clear()


async def cached_speed_rows(
    db: aiosqlite.Connection,
    db_key: str,
    *,
    since: float,
    model_ids: Sequence[str] | None,
    token_id: str | None = None,
    until: float | None = None,
) -> list[dict[str, Any]]:
    """``query_speed_rows`` with a per-bucket cache. The key holds the window
    start rounded to CACHE_BUCKET_S, the selection and the key's id, so one
    key's lens never serves another's."""
    key = (
        db_key,
        int(since // CACHE_BUCKET_S),
        tuple(model_ids) if model_ids is not None else None,
        token_id,
        int(until // CACHE_BUCKET_S) if until is not None else None,
    )
    hit = _cache.get(key)
    if hit is not None:
        return hit
    rows = await query_speed_rows(
        db, since=since, model_ids=model_ids, token_id=token_id, until=until
    )
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = rows
    return rows


async def last_finished_at(
    db: aiosqlite.Connection, *, model_ids: Sequence[str] | None, token_id: str | None = None
) -> float | None:
    """When the scope's newest request finished, whatever the window ("last
    request 3 h ago" in an empty window)."""
    where, args = _model_clause(model_ids)
    sql = "SELECT MAX(finished_at) FROM request_history WHERE 1" + where
    params: tuple[Any, ...] = tuple(args)
    if token_id is not None:
        sql += " AND token_id = ?"
        params += (token_id,)
    cur = await db.execute(sql, params)
    row = await cur.fetchone()
    return row[0] if row else None


def speed_payload(
    rows: list[dict[str, Any]],
    *,
    key_view: bool = False,
    points_limit: int = POINTS_LIMIT,
) -> dict[str, Any]:
    """Points for the chart and tile strips, plus the tile summaries.

    ``key_view``: the key holder's lens. Points carry no key, name, client IP or
    cached figure, and ``prompt`` (prompt / TTFT) replaces ``prefill``.
    """
    metrics = [row_metrics(r) for r in rows]
    ttft = [m["ttft"] for m in metrics if m["ttft"] is not None]
    decode = [m["decode"] for m in metrics if m["decode"] is not None]
    dur = [m["duration"] for m in metrics if m["duration"] is not None]
    queue = [
        float(r["queued_s"])
        for r, m in zip(rows, metrics, strict=True)
        if m["ttft"] is not None and r.get("queued_s") is not None
    ]
    summary: dict[str, Any] = {
        "ttft": summarize(ttft),
        "decode": summarize(decode),
        "duration": summarize(dur),
        "warden_queue": summarize(queue),
    }
    if key_view:
        summary["prompt"] = summarize(
            [m["prompt_speed"] for m in metrics if m["prompt_speed"] is not None]
        )
    else:
        summary["prefill"] = summarize(
            [m["prefill"] for m in metrics if m["prefill"] is not None and m["prefill_alone"]]
        )
    stride = max(1, math.ceil(len(rows) / max(1, points_limit)))
    points = []
    # Newest rows are always kept: stride from the end.
    for idx in range(len(rows) - 1, -1, -stride):
        r, m = rows[idx], metrics[idx]
        p: dict[str, Any] = {
            "id": r["id"],
            "finished_at": r["finished_at"],
            "model_id": r["model_id"],
            "model": r["model"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "finish_reason": r["finish_reason"],
            "queued_s": r["queued_s"],
            "ttft": m["ttft"],
            "decode": m["decode"],
            "duration": m["duration"],
        }
        if key_view:
            p["prompt_speed"] = m["prompt_speed"]
        else:
            p.update(
                token_id=r["token_id"],
                token_name=r["token_name"],
                client_ip=r["client_ip"],
                prefill=m["prefill"],
                prefill_alone=m["prefill_alone"],
            )
        points.append(p)
    points.reverse()
    return {
        "count": len(rows),
        "stride": stride,
        "summary": summary,
        "points": points,
    }


def speed_groups(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    """Window columns of the Requests table's Group by (operator only).

    A group never mixes models: ``key`` groups split by model (one row per key
    and model); a ``model`` group is one model; an ``ip`` row spans models, so it
    carries counts only, never a speed.
    """
    if group not in GROUPS:
        raise ValueError(group)
    acc: dict[tuple[Any, Any], dict[str, Any]] = {}
    for r in rows:
        if group == "key":
            k = (r["token_id"], r["model_id"])
            label = r["token_name"]
        elif group == "model":
            k = (r["model_id"], r["model_id"])
            label = r["model"]
        else:
            k = (r["client_ip"], None)
            label = r["client_ip"]
        g = acc.get(k)
        if g is None:
            g = acc[k] = {
                "key": k[0],
                "label": label,
                "model_id": k[1],
                "model": r["model"] if k[1] is not None else None,
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "_ttft": [],
                "_decode": [],
                "_prefill": [],
            }
        g["requests"] += 1
        g["prompt_tokens"] += int(r["prompt_tokens"] or 0)
        g["completion_tokens"] += int(r["completion_tokens"] or 0)
        if group == "ip":
            continue  # an IP spans models: counts only, never a mixed speed
        m = row_metrics(r)
        if m["ttft"] is not None:
            g["_ttft"].append(m["ttft"])
        if m["decode"] is not None:
            g["_decode"].append(m["decode"])
        if m["prefill"] is not None and m["prefill_alone"]:
            g["_prefill"].append(m["prefill"])
    out = []
    for g in acc.values():
        g["n_ttft"] = len(g["_ttft"])
        g["p50_ttft"] = _p50(g.pop("_ttft"))
        g["n_decode"] = len(g["_decode"])
        g["p50_decode"] = _p50(g.pop("_decode"))
        g["n_prefill"] = len(g["_prefill"])
        g["p50_prefill"] = _p50(g.pop("_prefill"))
        out.append(g)
    out.sort(key=lambda g: (-g["requests"], str(g["label"])))
    return out


__all__ = [
    "GROUPS",
    "LOCAL_ONLY",
    "last_finished_at",
    "query_speed_rows",
    "row_metrics",
    "speed_groups",
    "speed_payload",
    "summarize",
]
