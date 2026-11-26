"""Reads behind the Prompt Lens API that the UI routes do not already have (analysis Task 4.2):
the key's recent requests, one request with its findings, a turn diff and leanprompt's measured
score. Worker-thread SQLite, scoped by token (``token_id`` None = the admin: any key), text-free
(A-10): ids, sizes, counts, codes, positions.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import closing
from typing import Any

from app.lens import leanprompt_score, service
from app.lens.leanprompt_score import ScoreRow
from app.sessions import store
from app.sessions.refs import sess_ref, session_id
from app.stats.routes_forest import _connect

#: Requests listed at most per page.
MAX_REQUESTS = 200
#: Calls the measured score reads at most (newest first).
MAX_SCORE_ROWS = 200_000
DEFAULT_SCORE_WINDOW_S = 7 * 24 * 3600.0


def _lens(cause: str | None, *, strip: bool) -> str | None:
    return None if strip and cause in service.ADMIN_ONLY_CAUSES else cause


def _list_work(
    db_path: Any,
    token_id: str | None,
    key: str | None,
    since: float,
    before: float | None,
    model: str | None,
    limit: int,
) -> dict[str, Any]:
    scope = token_id if token_id is not None else key
    sql = (
        "SELECT id, finished_at, duration_s, model, prompt_tokens, cached_tokens,"
        " completion_tokens, finish_reason, cobs2_sess_ref, lens_cause, lens_break,"
        " lens_lost_tokens, lens_version, token_id FROM request_history WHERE finished_at >= ?"
    )
    args: list[Any] = [since]
    if scope is not None:
        sql += " AND token_id = ?"
        args.append(scope)
    if before is not None:
        sql += " AND finished_at < ?"
        args.append(before)
    if model:
        sql += " AND model = ?"
        args.append(model)
    with closing(_connect(db_path)) as db:
        got = db.execute(sql + " ORDER BY finished_at DESC LIMIT ?", (*args, limit + 1)).fetchall()
    page = got[:limit]
    items = [
        {
            "request_id": t[0],
            "finished_at": t[1],
            "duration_s": t[2],
            "model": t[3],
            "prompt_tokens": int(t[4] or 0),
            "cached_tokens": t[5],
            "completion_tokens": int(t[6] or 0),
            "finish_reason": t[7],
            "session_id": session_id(t[8]) if t[8] else None,
            "cause": _lens(t[9], strip=token_id is not None),
            "break": t[10],
            "lost_tokens": t[11],
            "version": t[12],
            **({"key_id": t[13]} if token_id is None else {}),
        }
        for t in page
    ]
    return {
        "requests": items,
        "next_before": page[-1][1] if len(got) > limit and page else None,
    }


async def requests_list(
    state: Any,
    *,
    token_id: str | None,
    key: str | None = None,
    since: float | None = None,
    before: float | None = None,
    model: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """The key's finished requests, newest first (``before`` pages on). A key holder's list is
    its own; ``key`` is the admin's filter."""
    return await asyncio.to_thread(
        _list_work,
        state.settings.db_path,
        token_id,
        key if token_id is None else None,
        since if since is not None else time.time() - 7 * 24 * 3600.0,
        before,
        model,
        max(1, min(int(limit), MAX_REQUESTS)),
    )


def _summary(r: store.Row, *, strip: bool) -> dict[str, Any]:
    return {
        "request_id": r.id,
        "finished_at": r.finished_at,
        "model": r.model,
        "relayed": r.relayed,
        "finish_reason": r.finish_reason,
        "prompt_tokens": r.prompt_tokens,
        "cached_tokens": r.cached_tokens,
        "completion_tokens": r.completion_tokens,
        "reusable_tokens": r.reusable_tokens_own if strip else r.reusable_tokens,
        "edge": r.cobs2_edge,
        "ttft_s": r.ttft_s,
        "break": r.lens_break,
        "cause": _lens(r.lens_cause, strip=strip),
        "span_class": r.lens_span_class,
        "span_line": r.lens_span_line,
        "lost_tokens": r.lens_lost_tokens,
        "counterfactual_tokens": r.lens_cf_tokens,
        "other_breaks": r.lens_other_breaks,
        "version": r.lens_version,
    }


def _detail_work(
    state: Any, req_id: str, token_id: str | None, held: tuple[str, str | None] | None
) -> dict[str, Any] | None:
    with closing(_connect(state.settings.db_path)) as db:
        row = service._find_row(db, req_id, token_id, held)
        if row is None:
            return None
        if token_id is not None:
            row = service._for_key_holder(db, token_id, [row])[0]
        out = _summary(row, strip=token_id is not None)
        ref = (
            sess_ref(state.settings.cookie_secret, row.token_id, row.cobs2_session)
            if row.cobs2_session and row.token_id
            else None
        )
        out["session_id"] = session_id(ref) if ref else None
        return out


async def request_detail(state: Any, *, token_id: str | None, req_id: str) -> dict[str, Any] | None:
    """One request: its cache figures and layer 1 verdict, plus the findings that touch it.
    None (404) when it is not the caller's to read."""
    body = await service.request_findings_body(state, token_id=token_id, req_id=req_id)
    if body is None:
        return None
    sections = getattr(state, "prompt_lens_sections", None)
    held = sections.request_of(req_id) if sections is not None else None
    out = await asyncio.to_thread(_detail_work, state, req_id, token_id, held)
    if out is None:
        return None
    return {**out, "turn": body["turn"], "findings": body["findings"], "names": body["names"]}


def _diff_work(
    state: Any,
    req_id: str,
    against: str | None,
    token_id: str | None,
    held: tuple[str, str | None] | None,
) -> dict[str, Any] | None:
    with closing(_connect(state.settings.db_path)) as db:
        row = service._find_row(db, req_id, token_id, held)
        if row is None:
            return None
        other_id = against or row.cobs2_parent_req
        other = service._find_row(db, other_id, token_id, None) if other_id else None
        if against and other is None:
            return None  # an unknown or foreign ``against``: the same 404
        rows = [row] + ([other] if other is not None else [])
        if token_id is not None:
            rows = service._for_key_holder(db, token_id, rows)
    b = rows[0]
    a = rows[1] if len(rows) > 1 else None
    strip = token_id is not None
    continues = a is not None and b.cobs2_parent_req == a.id
    out: dict[str, Any] = {
        "request_id": b.id,
        "against": a.id if a is not None else None,
        # False: ``against`` is not the request's parent turn, so a changed head is not shown.
        "continues": continues,
        "first_change": None,
        "tokens": None,
        "b": _summary(b, strip=strip),
        "a": _summary(a, strip=strip) if a is not None else None,
    }
    if a is not None:
        out["tokens"] = {
            "prompt_delta": b.prompt_tokens - a.prompt_tokens,
            "cached_delta": (b.cached_tokens or 0) - (a.cached_tokens or 0),
        }
    if continues and b.lens_break:
        out["first_change"] = {
            "part": b.lens_break,
            "cause": _lens(b.lens_cause, strip=strip),
            "span_class": b.lens_span_class,
            "line": b.lens_span_line,
            "lost_tokens": b.lens_lost_tokens,
            "counterfactual_tokens": b.lens_cf_tokens,
        }
    return out


async def request_diff(
    state: Any, *, token_id: str | None, req_id: str, against: str | None
) -> dict[str, Any] | None:
    """What changed between a request and its parent turn (or ``against``, another request of
    the same caller): the first part that broke reuse, with its cause and position, and the
    token deltas. From stored codes and counts, so it works after the ring forgot the text and
    never shows any (A-10)."""
    if not service._ID.fullmatch(req_id) or (against and not service._ID.fullmatch(against)):
        return None
    sections = getattr(state, "prompt_lens_sections", None)
    held = sections.request_of(req_id) if sections is not None else None
    return await asyncio.to_thread(_diff_work, state, req_id, against, token_id, held)


# -- leanprompt's measured score ---------------------------------------------------------


def _score_work(db_path: Any, scope: str | None, since: float, model: str | None) -> dict[str, Any]:
    sql = (
        "SELECT prompt_tokens, cached_tokens, completion_tokens, reusable_tokens_own,"
        " reusable_tokens, cobs2_parent_req, finish_reason, calls_bad FROM request_history"
        " WHERE finished_at >= ?"
    )
    args: list[Any] = [since]
    if scope is not None:
        sql += " AND token_id = ?"
        args.append(scope)
    if model:
        sql += " AND model = ?"
        args.append(model)
    with closing(_connect(db_path)) as db:
        got = db.execute(sql + " LIMIT ?", (*args, MAX_SCORE_ROWS)).fetchall()
    rows = [
        ScoreRow(
            prompt_tokens=int(t[0] or 0),
            cached_tokens=t[1],
            completion_tokens=int(t[2] or 0),
            reusable_tokens=t[3] if t[3] is not None else t[4],
            continues=bool(t[5]),
            finish_reason=t[6],
            bad_call=_has_bad(t[7]),
        )
        for t in got
    ]
    return leanprompt_score.leanprompt_score(rows)


def _has_bad(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    try:
        v = json.loads(raw)
    except ValueError:
        return False
    return bool(v)


async def score(
    state: Any,
    *,
    token_id: str | None,
    key: str | None = None,
    since: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """leanprompt's measured-score JSON for the key (``npx leanprompt score --from lmwarden``)."""
    got = await asyncio.to_thread(
        _score_work,
        state.settings.db_path,
        token_id if token_id is not None else key,
        since if since is not None else time.time() - DEFAULT_SCORE_WINDOW_S,
        model,
    )
    out = {**got, "source": "lm-warden"}
    if since is not None:
        out["since"] = since
    if model:
        out["model"] = model
    return out
