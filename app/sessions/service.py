"""Async glue for the sessions API: scoping by viewer, ref resolution, the
5 s list cache, and the bounded store reads run in a worker thread."""

from __future__ import annotations

import asyncio
import base64
import time
from contextlib import closing
from typing import Any

from app.sessions import store
from app.sessions.listing import GROUP_PAGE, apply_list
from app.sessions.model import (
    MAX_TURNS,
    ListContext,
    build_session,
    segment,
    summarize_list,
    turn_json,
)
from app.sessions.refs import ParsedRef, is_fallback, sess_ref, session_id
from app.stats.routes_forest import ForestViewer
from app.stats.routes_forest import _connect as _forest_connect

RANGES = {"1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}
LIST_CACHE_S = 5.0
_LIST_CACHE_MAX = 64
#: Rows read on each side of a session-less (fallback) segment's anchor row.
_FALLBACK_SIDE = MAX_TURNS
_list_cache: dict[
    tuple[Any, ...], tuple[float, list[dict[str, Any]], float | None, float | None]
] = {}


class NotRecordedYet(Exception):
    """The ref is a request still in flight whose session has no rows yet."""


def _connect(db_path: Any) -> Any:
    return _forest_connect(db_path)


def _tok(t: str | None) -> str | None:
    return t if t else None


def _encode_cursor(last_at: float, sid: str) -> str:
    return base64.urlsafe_b64encode(f"{last_at!r}:{sid}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[float, str] | None:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        at, sid = raw.split(":", 1)
        return float(at), sid
    except Exception:  # noqa: BLE001
        return None


async def _scan(
    state: Any, viewer: ForestViewer, *, range_s: int, tok: str | None, now: float
) -> tuple[float, list[dict[str, Any]], float | None, float | None]:
    """The window's session summaries (with private pause marks), cached 5 s per
    (db, viewer, range, token): filters and grouping run on it without a rescan."""
    db_path = state.settings.db_path
    key = (str(db_path), viewer.token_id, range_s, tok)
    hit = _list_cache.get(key)
    if hit is not None and now - hit[0] < LIST_CACHE_S:
        return hit
    secret = state.settings.cookie_secret

    def work() -> tuple[list[dict[str, Any]], float | None, float | None]:
        with closing(_connect(db_path)) as db:
            rows, trunc = store.list_rows(db, since=now - range_s, token_id=tok)
            ctx = ListContext(
                toolsets=store.toolsets(db, {r.toolset for r in rows if r.toolset}),
                engines=store.model_engines(db, {r.model_id for r in rows if r.model_id}),
                cold_rates=_cold_rates(state, {r.model_id for r in rows if r.model_id}),
            )
            items = summarize_list(
                rows,
                secret=secret,
                root_of=lambda t, s: store.root_of(db, _tok(t), s),
                now=now,
                ctx=ctx,
            )
            oldest = store.oldest_finished_at(db)
            before = rows[-1].finished_at if trunc and rows else None
        return items, before, oldest

    items, before, oldest = await asyncio.to_thread(work)
    entry = (now, items, before, oldest)
    if len(_list_cache) >= _LIST_CACHE_MAX:
        _list_cache.clear()
    _list_cache[key] = entry
    return entry


async def list_sessions(
    state: Any,
    viewer: ForestViewer,
    *,
    range_s: int,
    model: str | None,
    token: str | None,
    q: str | None,
    cursor: str | None,
    limit: int,
    group: str | None = None,
    group_key: str | None = None,
    view: str = "all",
    where: str = "all",
    sort: str | None = None,
    since: float | None = None,
    until: float | None = None,
    offset: int | None = None,
    per_group: int = GROUP_PAGE,
    tz_offset_min: int = 0,
) -> dict[str, Any]:
    tok = viewer.token_id if viewer.token_id is not None else _tok(token)
    now = time.time()
    scanned_at, items, before, oldest = await _scan(
        state, viewer, range_s=range_s, tok=tok, now=now
    )
    legacy = group is None and sort is None and offset is None
    listed = apply_list(
        items,
        now=scanned_at,
        range_start=scanned_at - range_s,
        view=view,
        where=where,
        q=q,
        model=model,
        since=since,
        until=until,
        group="none" if legacy else (group or "key"),
        group_key=group_key,
        sort=sort or "newest",
        offset=offset or 0,
        limit=limit if not legacy else len(items) + 1,
        per_group=per_group,
        tz_offset_min=tz_offset_min,
    )
    out: dict[str, Any] = {
        "truncated_before": before,
        "oldest_data_at": oldest,
        "total": listed["total"],
        "views": listed["views"],
        "histogram": listed["histogram"],
        "pause_summary": listed["pause_summary"],
    }
    if not legacy:
        out["groups"] = listed["groups"]
        out["sessions"] = listed["sessions"]
        out["next_cursor"] = None
        return out
    # Legacy (R12): newest activity first, cursor paging.
    page_items = sorted(listed["sessions"], key=lambda i: (-i["last_at"], i["session_id"]))
    if cursor:
        c = _decode_cursor(cursor)
        if c is not None:
            page_items = [
                i
                for i in page_items
                if i["last_at"] < c[0] or (i["last_at"] == c[0] and i["session_id"] > c[1])
            ]
    page = page_items[:limit]
    out["sessions"] = page
    out["next_cursor"] = (
        _encode_cursor(page[-1]["last_at"], page[-1]["session_id"])
        if len(page_items) > limit and page
        else None
    )
    return out


def _visible(viewer: ForestViewer, token_id: str | None) -> bool:
    return viewer.token_id is None or viewer.token_id == token_id


def _cold_rates(state: Any, model_ids: set[str]) -> dict[str, float | None]:
    pm = getattr(state, "prefill_model", None)
    if pm is None:
        return {}
    out: dict[str, float | None] = {}
    for m in model_ids:
        try:
            out[m] = pm.cold_rate_for(m)
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def _session_rows(
    db: Any, secret: str, token_id: str | None, sess: str, *, anchor: str | None, seg: str | None
) -> tuple[list[store.Row], bool, str, str] | None:
    """Rows, truncated, canonical session id, root session."""
    if is_fallback(sess):
        ref = sess_ref(secret, token_id, sess)
        if ref is None:
            return None
        # Anchor on the named row (the request asked for, or the row the segment id
        # names) and read around it, never "the newest N rows of the whole session":
        # those slide, so a long or older segment would 404 (verified finding #20).
        arow = (
            store.request_row(db, anchor)
            if anchor is not None
            else store.anchor_row(db, token_id, sess, seg)
            if seg is not None
            else None
        )
        if arow is None:
            return None
        rows, more_before, more_after = store.rows_around(
            db, token_id, sess, arow.finished_at, _FALLBACK_SIDE
        )
        for part in segment(rows):
            if any(r.id == arow.id for r in part):
                trunc = (
                    len(part) > MAX_TURNS
                    or (more_before and part[0] is rows[0])
                    or (more_after and part[-1] is rows[-1])
                )
                # A segment id answers with itself, so list and detail agree.
                sid = session_id(ref, arow.id if anchor is None else part[0].id)
                return part[-MAX_TURNS:], trunc, sid, sess
        return None
    root = store.root_of(db, token_id, sess)
    ref = sess_ref(secret, token_id, root)
    if ref is None:
        return None
    rows, trunc = store.rows_for(db, token_id, store.descendants(db, token_id, root), MAX_TURNS)
    return rows, trunc, session_id(ref), root


async def get_session(state: Any, viewer: ForestViewer, ref: ParsedRef) -> dict[str, Any] | None:
    db_path = state.settings.db_path
    secret = state.settings.cookie_secret
    live = None
    if ref.kind == "request":
        reg = getattr(state, "request_registry", None)
        for o in reg.snapshot() if reg is not None else ():
            if o.id == ref.ref:
                live = (o.token_id, getattr(o, "cobs2_session", None))
                break

    def work() -> dict[str, Any] | None:
        with closing(_connect(db_path)) as db:
            anchor: str | None = None
            if ref.kind == "request":
                row = store.request_row(db, ref.ref)
                if row is not None:
                    token_id, sess, anchor = row.token_id, row.cobs2_session, row.id
                elif live is not None and live[1]:
                    token_id, sess = live
                    if not _visible(viewer, token_id):
                        return None
                    if not sess or not store.rows_for(db, token_id, [sess], 1)[0]:
                        raise NotRecordedYet
                else:
                    return None
            else:
                found = store.lookup_ref(db, ref.ref)
                if found is None:
                    return None
                token_id, sess = found
            if sess is None or not _visible(viewer, token_id):
                return None
            got = _session_rows(db, secret, token_id, sess, anchor=anchor, seg=ref.segment)
            if got is None or not got[0]:
                return None
            rows, trunc, sid, root = got
            digests = {r.toolset for r in rows if r.toolset}
            ts = store.toolsets(db, digests)
            hmacs = {h for d in ts.values() for h, _f, _c in d} | {
                h for r in rows for _f, h in r.tools_out
            }
            names = store.tool_names(db, token_id, hmacs)
            oldest = store.oldest_finished_at(db)
            engines = store.model_engines(db, {r.model_id for r in rows})
        return build_session(
            rows,
            session_id=sid,
            token_name=rows[-1].token_name,
            truncated=trunc,
            oldest_data_at=oldest,
            cold_rates=_cold_rates(state, {r.model_id for r in rows}),
            toolsets=ts,
            tool_names=names,
            root_session=root,
            engines=engines,
        )

    return await asyncio.to_thread(work)


async def get_request(
    state: Any, viewer: ForestViewer, request_id: str, wait_s: float
) -> dict[str, Any] | None:
    db_path = state.settings.db_path
    secret = state.settings.cookie_secret

    def read() -> dict[str, Any] | None:
        with closing(_connect(db_path)) as db:
            row = store.request_row(db, request_id)
            if row is None or not _visible(viewer, row.token_id):
                return None
            sid = None
            turn = 1
            if row.cobs2_session:
                got = _session_rows(
                    db, secret, row.token_id, row.cobs2_session, anchor=row.id, seg=None
                )
                if got is not None:
                    sid = got[2]
                    same = [r for r in got[0] if r.cobs2_conv == row.cobs2_conv]
                    turn = next((i + 1 for i, r in enumerate(same) if r.id == row.id), 1)
        out = turn_json(
            row,
            conversation_id=row.cobs2_conv,
            turn=turn,
            lineage_unknown=row.cobs2_edge is None and row.cobs2_node is None,
            side_call=False,
        )
        out["session_id"] = sid
        return out

    got = await asyncio.to_thread(read)
    if got is None and wait_s > 0:
        hist = getattr(state, "request_history", None)
        if hist is not None and hasattr(hist, "wait_for"):
            await hist.wait_for(request_id, wait_s)
            got = await asyncio.to_thread(read)
    if got is None:
        # Still in flight (and the viewer's own): say so, so a client can tell
        # a slow commit from a wrong id.
        reg = getattr(state, "request_registry", None)
        for o in reg.snapshot() if reg is not None else ():
            if o.id == request_id and _visible(viewer, o.token_id):
                raise NotRecordedYet
    return got
