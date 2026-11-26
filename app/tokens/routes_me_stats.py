"""The key holder's Stats view (stats redesign B8, B9, B11).

Signed in with their own ``vw_`` key (the forest login), a key holder sees how
THEIR requests went: speed, tokens per minute, their own-lens cache, their
requests in flight, and -- when both the platform and the admin allow it --
the captured prompt and reply of their own requests.

Owner decisions D2, D5, D6, D8 bind every route here:

* Only this key's own requests. The key comes from the sign-in (``key_holder``),
  never from a request parameter; no route reads a token id from the query.
* Nothing fleet-wide and nothing derived from other keys: no server load, no
  install, engine or host figure, no other key's name, IP, session or prompt,
  no fleet-learned prefill rate, no "requests ahead" count. The waiting estimate
  is computed from the key's own history only.
* No client IP, not even their own (D5): the session identifies a request.
* No measured ``cached_tokens``: on a shared prefix cache it can reveal that
  another key sent the same text. The cache card uses the own lens
  (``min(cached, reusable_own)``: reuse of the key's OWN earlier prompts).
* Prompt Lens (owner decision 2026-11-22): ``/api/me/requests/{id}/capture``
  answers whenever the platform has Prompt Lens on, and only for the caller's
  own requests. Unknown, expired and foreign ids all get the same 404.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.cache_obs.classify import OUTCOMES
from app.db.database import open_db
from app.proxy import prompt_lens_kept
from app.proxy.ledger import spread_over_minutes
from app.proxy.live_tokens import decode_tps
from app.proxy.request_registry import engine_progress_tokens
from app.stats import speed
from app.stats.request_history import LOCAL_ONLY, quantile, query_cache_summary
from app.tokens.routes_me import key_holder, prompt_lens_view_available
from app.tokens.routes_me_prompt_lens import kept_read_allowed

router = APIRouter(prefix="/api/me")

_RANGES = {"1h": 3600.0, "6h": 6 * 3600.0, "24h": 86400.0, "7d": 7 * 86400.0}
_RANGE = Query(default="1h", pattern=r"^(1h|6h|24h|7d)$")

#: Own requests needed before the waiting estimate is offered.
OWN_RATE_MIN_SAMPLES = 3
OWN_RATE_WINDOW_S = 86400.0

CAPTURE_NOTE = (
    "Captures are kept briefly in server memory, not on disk, and the platform's admins "
    "can see them too. You only ever see captures of requests made with your key."
)
#: CAPTURE_NOTE for a key an admin set to keep captures (keep_captures, G-R14):
#: its last ``prompt_lens_kept.KEPT_PER_KEY`` finished captures are on disk.
CAPTURE_NOTE_KEPT = (
    "Captures are kept briefly in server memory, and the platform's admins can see them "
    "too. An admin set this key to keep captures: the last "
    f"{prompt_lens_kept.KEPT_PER_KEY} captures are kept on disk for this key. You only ever "
    "see captures of requests made with your key."
)


#: ``until`` (ISO 8601) ends the window at that time instead of now: the key
#: holder's page anchors an empty window to the key's last request (owner
#: request 2026-11-22). Clamped to now; every query stays on the caller's key.
_UNTIL = Query(default=None, max_length=40, description="ISO 8601 end of the window; default now")


def _window(range_: str, until: str | None) -> tuple[float, float, float | None]:
    """``(since, end, until_epoch)`` of the window. 422 on an unreadable time;
    a time in the future reads as now."""
    now = time.time()
    end = now
    anchored: float | None = None
    if until is not None:
        try:
            dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(422, "until must be an ISO 8601 time") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        end = min(now, dt.timestamp())
        anchored = end
    return end - _RANGES[range_], end, anchored


async def _own_models(db: Any, tid: str, since: float, end: float) -> list[dict[str, Any]]:
    """Models this key used in the window, most used first."""
    cur = await db.execute(
        "SELECT model_id, MAX(model), COUNT(*) FROM request_history "
        "WHERE token_id = ? AND finished_at >= ? AND finished_at <= ?"
        + LOCAL_ONLY
        + " GROUP BY model_id "
        "ORDER BY COUNT(*) DESC, model_id",
        (tid, since, end),
    )
    return [{"model_id": m, "model": n, "requests": c} for m, n, c in await cur.fetchall()]


@router.get("/stats/speed")
async def my_speed(
    request: Request,
    range: str = _RANGE,
    model: str | None = None,
    until: str | None = _UNTIL,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """The key's own latency and speed for ONE model (key-holder lens of
    /api/stats/v2/speed: ``prompt`` speed instead of computed prefill, no
    names, IPs or cached figures). ``last_request_at`` is when this key's
    newest request finished, any model, whatever the window: the page anchors
    an empty window there."""
    since, end, anchored = _window(range, until)
    async with open_db(request.app.state.settings.db_path) as db:
        models = await _own_models(db, tid, since, end)
        sel = [model] if model else None
        rows = await speed.cached_speed_rows(
            db,
            str(request.app.state.settings.db_path),
            since=since,
            model_ids=sel,
            token_id=tid,
            until=anchored,
        )
        last = await speed.last_finished_at(db, model_ids=sel, token_id=tid)
        last_any = await speed.last_finished_at(db, model_ids=None, token_id=tid)
    body: dict[str, Any] = {
        "range": range,
        "model_id": model,
        "until": anchored,
        "since_epoch": since,
        "end_epoch": end,
        "models": models,
        "last_finished_at": last,
        "last_request_at": last_any,
    }
    if model:
        body.update(speed.speed_payload(rows, key_view=True))
    else:
        body.update({"count": len(rows), "stride": 1, "summary": None, "points": []})
    return body


@router.get("/stats/tokens")
async def my_tokens(
    request: Request,
    range: str = _RANGE,
    model: str | None = None,
    until: str | None = _UNTIL,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """Prompt and completion tokens per minute for this key (and model), booked
    to the minutes they were produced (D3): the prompt at the first-token
    minute, the completion spread to the finish. No cached split."""
    since, now, anchored = _window(range, until)
    sql = (
        "SELECT finished_at, duration_s, ttft_s, prompt_tokens, completion_tokens "
        "FROM request_history WHERE token_id = ? AND finished_at >= ? AND finished_at <= ?"
        + LOCAL_ONLY
    )
    args: tuple[Any, ...] = (tid, since, now)
    if model:
        sql += " AND model_id = ?"
        args += (model,)
    async with open_db(request.app.state.settings.db_path) as db:
        cur = await db.execute(sql, args)
        rows = await cur.fetchall()
    acc: dict[int, list[int]] = {}
    for fin, dur, ttft, prompt, comp in rows:
        end = float(fin)
        first = end - float(dur) + float(ttft) if dur is not None and ttft is not None else end
        first = min(max(first, end - float(dur or 0.0)), end)
        a = acc.setdefault(int(end // 60), [0, 0, 0])
        a[0] += 1
        acc.setdefault(int(first // 60), [0, 0, 0])[1] += int(prompt or 0)
        for m, n in spread_over_minutes(int(comp or 0), start=first, end=end).items():
            acc.setdefault(m, [0, 0, 0])[2] += n
    minutes = [
        {"minute": m, "requests": v[0], "prompt_tokens": v[1], "completion_tokens": v[2]}
        for m, v in sorted(acc.items())
    ]
    return {
        "range": range,
        "model_id": model,
        "until": anchored,
        "since_minute": int(since // 60),
        "now_minute": int(now // 60),
        "minutes": minutes,
    }


@router.get("/stats/cache")
async def my_cache(
    request: Request,
    range: str = _RANGE,
    model: str | None = None,
    until: str | None = _UNTIL,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """The key's own-lens cache: tokens reused from its OWN earlier prompts.

    Prefill saved is estimated at the CONFIGURED prefill rate
    (``VW_PREFILL_TOK_S_HINT``), not the rate the warden learned from all keys'
    traffic, so no figure here is derived from another key."""
    since, _end, anchored = _window(range, until)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await query_cache_summary(
            db, since=since, model_ids=[model] if model else None, token_id=tid, until=anchored
        )
    hint = float(getattr(settings, "prefill_tok_s_hint", 0) or 0)
    reused = sum(int(r.get("own_reused_tokens") or 0) for r in rows)
    outcomes = dict.fromkeys((o for o in OUTCOMES if o != "misrouted"), 0)
    for r in rows:
        for o, n in (r.get("outcomes") or {}).items():
            if o in outcomes:
                outcomes[o] += int(n or 0)
    cache = {
        "requests": sum(int(r["requests"]) for r in rows),
        "prompt_tokens": sum(int(r["prompt_tokens"]) for r in rows),
        "own_reused_tokens": reused,
        "own_known_prompt_tokens": sum(int(r.get("own_known_prompt_tokens") or 0) for r in rows),
        "own_known_requests": sum(int(r.get("own_known_requests") or 0) for r in rows),
        "outcomes": outcomes,
        "prefill_saved_s": (reused / hint) if hint > 0 and rows else None,
        "prefill_saved_basis": "configured",
        "prefill_rate_tok_s": hint if hint > 0 else None,
    }
    return {"range": range, "model_id": model, "until": anchored, "cache": cache}


async def _own_prompt_rate(db: Any, tid: str) -> dict[str, float]:
    """Per model: the key's OWN median prompt speed (prompt / TTFT) over the
    last 24 h, when at least OWN_RATE_MIN_SAMPLES requests back it."""
    cur = await db.execute(
        "SELECT model_id, prompt_tokens, ttft_s FROM request_history WHERE token_id = ? "
        "AND finished_at >= ? AND ttft_s > 0 AND prompt_tokens > 0" + LOCAL_ONLY,
        (tid, time.time() - OWN_RATE_WINDOW_S),
    )
    by: dict[str, list[float]] = {}
    for mid, p, t in await cur.fetchall():
        by.setdefault(mid, []).append(float(p) / float(t))
    out: dict[str, float] = {}
    for mid, v in by.items():
        if len(v) >= OWN_RATE_MIN_SAMPLES:
            q = quantile(sorted(v), 0.5)
            if q:
                out[mid] = q
    return out


def _own_row(r: Any, now: float, mine: list[Any], rates: dict[str, float]) -> dict[str, Any]:
    ctx = r.prompt_tokens + r.completion_tokens
    ttft = (
        round(r.first_token_monotonic - r.started_monotonic, 3)
        if r.first_token_monotonic is not None
        else None
    )
    tps = decode_tps(
        list(getattr(r, "out_samples", ()) or ()), first_token=r.first_token_monotonic, now=now
    )
    expected = None
    waiting = r.phase in ("queued", "prefill") and r.is_stream
    rate = rates.get(r.model_row_id)
    if waiting and rate:
        # The key's own earlier work on the same model that has not started
        # producing yet, plus this prompt, at the key's own prompt speed.
        ahead = sum(
            o.prompt_tokens
            for o in mine
            if o is not r
            and o.model_row_id == r.model_row_id
            and o.phase in ("queued", "prefill")
            and o.started_monotonic < r.started_monotonic
        )
        expected = round((ahead + r.prompt_tokens) / rate, 1)
    return {
        "id": r.id,
        "model": r.model,
        "model_id": r.model_row_id,
        "phase": r.phase,
        "is_stream": r.is_stream,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "out_tokens": r.completion_tokens,
        "engine_progress_tokens": engine_progress_tokens(r),
        "context_tokens": ctx,
        "max_model_len": r.max_model_len,
        "context_pct": round(ctx / r.max_model_len, 4) if r.max_model_len else None,
        "elapsed_s": round(now - r.started_monotonic, 1),
        "queued_s": round(r.queued_s, 3) if r.queued_s is not None else None,
        "ttft_s": ttft,
        "ttft_total_s": round(ttft + (r.queued_s or 0.0), 3) if ttft is not None else None,
        "decode_tps_5s": round(tps, 1) if tps is not None else None,
        "session_id": r.session_id,
        "forest_session": r.session_key,
        "expected_start_s": expected,
        "expected_start_basis": "own" if expected is not None else None,
    }


@router.get("/stats/requests")
async def my_requests(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    """The key's requests in flight, all its models. ``state`` per model:
    ``waiting`` (one of them waits to start), ``running`` or absent (idle)."""
    reg = getattr(request.app.state, "request_registry", None)
    mine = [r for r in (reg.snapshot() if reg is not None else []) if r.token_id == tid]
    now = time.monotonic()
    rates: dict[str, float] = {}
    if any(r.phase in ("queued", "prefill") for r in mine):
        async with open_db(request.app.state.settings.db_path) as db:
            rates = await _own_prompt_rate(db, tid)
    rows = [_own_row(r, now, mine, rates) for r in mine]
    state: dict[str, str] = {}
    for r in rows:
        waiting = r["phase"] in ("queued", "prefill")
        if waiting or r["model"] not in state:
            state[r["model"]] = "waiting" if waiting else "running"
    return {"count": len(rows), "requests": rows, "state": state}


def _capture_events(hub: Any, rid: str, tid: str) -> tuple[list[dict[str, Any]], bool] | None:
    """The request's events, own key only, and whether the key's window already
    dropped some of them (``partial``). None when the key has no capture with
    this id.

    Read from the key's OWN replay window (``PromptLensHub.key_events`` /
    ``key_start_of``), not the shared ring: whether a capture is still there,
    and whether it is partial, then depends on the key's own traffic only. On
    the shared ring both would move with every other key's traffic (review I1)."""
    events = hub.key_events(tid)
    start = next(
        (
            e
            for e in events
            if e.get("type") == "request_start"
            and e.get("request_id") == rid
            and e.get("token_id") == tid
        ),
        None,
    )
    partial = False
    if start is None:
        kept = hub.key_start_of(tid, rid)
        if kept is None or kept.get("token_id") != tid:
            return None
        start, partial = kept, True
    req_id = start.get("req_id")
    rest = [
        e
        for e in events
        if e.get("req_id") == req_id and e.get("token_id") == tid and e is not start
    ]
    return [start, *rest], partial


_NOT_FOUND = "no capture of yours with this id"


def _kept_events(
    events: list[Any], reply: str, reply_dropped: int, thinking: str, thinking_dropped: int
) -> tuple[list[dict[str, Any]], bool] | None:
    """A kept capture as ``_capture_events`` returns one: its events in the key
    holder's view, the whole reply and reasoning as one delta each, and
    ``partial`` when the store had already dropped the oldest text. None for a
    record with no request_start."""
    view = prompt_lens_kept.holder_events(events)
    if not view or view[0].get("type") != "request_start":
        return None
    start, *rest = view
    deltas: list[dict[str, Any]] = []
    for channel, text in (("content", reply), ("reasoning", thinking)):
        if text:
            deltas.append({"type": "delta", "channel": channel, "text": text})
    return [start, *deltas, *rest], bool(reply_dropped or thinking_dropped)


@router.get("/requests/{request_id}/capture", deprecated=True)
async def my_capture(
    request_id: str, request: Request, tid: str = Depends(key_holder)
) -> dict[str, Any]:
    """The captured prompt and reply of one of the key's OWN requests (D8).

    Deprecated (2026-11-22): the Prompt Lens dock no longer calls it (it reads
    the key stream and GET /api/me/prompt-lens/requests/{req_id}/section/{kind});
    it stays for one release and goes in the next.
    """
    async with open_db(request.app.state.settings.db_path) as db:
        if not await prompt_lens_view_available(request, tid):
            raise HTTPException(404, _NOT_FOUND)
        found = _capture_events(request.app.state.prompt_lens_hub, request_id, tid)
        kept = False
        if not found:
            # Out of memory: the key's OWN kept capture (keep_captures), if any.
            kept_read_allowed(request, tid)
            texts = await prompt_lens_kept.texts_by_request(db, request_id, tid)
            found = _kept_events(*texts) if texts is not None else None
            if not found:
                raise HTTPException(404, _NOT_FOUND)
            kept = True
        events, partial = found
        cur = await db.execute(
            "SELECT finished_at, queued_s, ttft_s, duration_s, prompt_tokens, completion_tokens, "
            "cached_tokens, reusable_tokens_own, finish_reason FROM request_history "
            "WHERE id = ? AND token_id = ?",
            (request_id, tid),
        )
        hist = await cur.fetchone()
        cur = await db.execute("SELECT keep_captures FROM api_tokens WHERE id = ?", (tid,))
        keep_row = await cur.fetchone()
    keeps = bool(keep_row and keep_row[0])
    start = events[0]
    end = next((e for e in events if e.get("type") == "request_end"), None)
    reply = "".join(
        e.get("text") or ""
        for e in events
        if e.get("type") == "delta" and e.get("channel") == "content"
    )
    reasoning = "".join(
        e.get("text") or ""
        for e in events
        if e.get("type") == "delta" and e.get("channel") == "reasoning"
    )
    timing: dict[str, Any] = {}
    if hist is not None:
        fin, queued, ttft, dur, prompt, comp, cached, reuse_own, reason = hist
        m = speed.row_metrics(
            {
                "queued_s": queued,
                "ttft_s": ttft,
                "duration_s": dur,
                "prompt_tokens": prompt,
                "completion_tokens": comp,
                "cached_tokens": None,
            }
        )
        timing = {
            "finished_at": fin,
            "queued_s": queued,
            "ttft_s": m["ttft"],
            "duration_s": dur,
            "decode_tps": m["decode"],
            "prompt_tokens": prompt,
            "completion_tokens": comp,
            "own_reused_tokens": (
                min(int(cached), int(reuse_own))
                if cached is not None and reuse_own is not None
                else None
            ),
            "finish_reason": reason,
        }
    else:
        reg = getattr(request.app.state, "request_registry", None)
        live = reg.get(request_id) if reg is not None else None
        if live is not None and live.token_id == tid:
            now = time.monotonic()
            row_ = _own_row(live, now, [live], {})
            timing = {
                "queued_s": row_["queued_s"],
                "ttft_s": row_["ttft_total_s"],
                "elapsed_s": row_["elapsed_s"],
                "decode_tps": row_["decode_tps_5s"],
                "prompt_tokens": row_["prompt_tokens"],
                "completion_tokens": row_["completion_tokens"],
            }
    return {
        "id": request_id,
        "model": start.get("served_name") or start.get("model"),
        "started_ts": start.get("ts"),
        "ended_ts": end.get("ts") if end else None,
        "live": end is None,
        "prompt": start.get("prompt"),
        "prompt_elided": start.get("prompt_elided"),
        "reply": reply,
        "reasoning": reasoning,
        "finish_reason": end.get("finish_reason") if end else None,
        "prompt_tokens": end.get("prompt_tokens") if end else None,
        "completion_tokens": end.get("completion_tokens") if end else None,
        "timing": timing,
        #: The server already dropped the oldest part of this capture (a busy
        #: server keeps the newest events): the reply shown is its latest part.
        "partial": partial,
        #: Served from the key's kept captures (on disk; an admin's switch).
        "kept": kept,
        "note": CAPTURE_NOTE_KEPT if keeps else CAPTURE_NOTE,
    }
