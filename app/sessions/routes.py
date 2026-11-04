"""Sessions API: the dashboard's sessions pages and the test agent's read-back.

All three routes take the admin session or a key holder's forest JWT
(``forest_viewer``); a key holder sees only its own token's sessions and gets
404 -- never 403 -- for anyone else's. Plain JSON, ids only, never prompt text.
A read-path failure answers 503 and leaves the rest of the dashboard alone.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.auth.deps import require_jwt
from app.sessions import listing, service
from app.sessions.refs import parse_ref
from app.stats.routes_forest import ForestViewer, forest_viewer

logger = logging.getLogger(__name__)
router = APIRouter()

_UNAVAILABLE = "sessions unavailable"


@router.get("/api/sessions")
async def sessions_list(
    request: Request,
    range: str = "24h",
    model: str | None = None,
    token: str | None = None,
    q: str | None = None,
    cursor: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    group: str | None = None,
    group_key: str | None = None,
    view: str = "all",
    where: str = "all",
    sort: str | None = None,
    since: float | None = None,
    until: float | None = None,
    offset: int | None = Query(None, ge=0),
    per_group: int = Query(listing.GROUP_PAGE, ge=1, le=100),
    tz_offset_min: int = Query(0, ge=-840, le=840),
    viewer: ForestViewer = Depends(forest_viewer),
) -> dict[str, Any]:
    """Sessions with views, an activity histogram and the pause summary.
    With ``group``, ``sort`` or ``offset`` the answer is grouped and paged by
    offset; without them, newest activity first and ``next_cursor`` pages on."""
    if range not in service.RANGES:
        raise HTTPException(400, f"range must be one of {sorted(service.RANGES)}")
    for name, val, allowed in (
        ("group", group, listing.GROUPS),
        ("view", view, listing.VIEWS),
        ("where", where, listing.WHERES),
        ("sort", sort, listing.SORTS),
    ):
        if val is not None and val not in allowed:
            raise HTTPException(400, f"{name} must be one of {list(allowed)}")
    try:
        return await service.list_sessions(
            request.app.state,
            viewer,
            range_s=service.RANGES[range],
            model=model,
            token=token,
            q=q,
            cursor=cursor,
            limit=limit,
            group=group,
            group_key=group_key,
            view=view,
            where=where,
            sort=sort,
            since=since,
            until=until,
            offset=offset,
            per_group=per_group,
            tz_offset_min=tz_offset_min,
        )
    except Exception:  # noqa: BLE001 -- the dashboard keeps working
        logger.debug("sessions: list failed", exc_info=True)
        raise HTTPException(503, _UNAVAILABLE) from None


@router.get("/api/sessions/{session_id}")
async def session_detail(
    request: Request,
    session_id: str,
    viewer: ForestViewer = Depends(forest_viewer),
) -> dict[str, Any]:
    """One session by ``s_…`` (canonical) or ``r_<request id>``."""
    ref = parse_ref(session_id)
    if ref is None:
        raise HTTPException(404, "session not found")
    try:
        out = await service.get_session(request.app.state, viewer, ref)
    except service.NotRecordedYet:
        raise HTTPException(404, "not recorded yet") from None
    except Exception:  # noqa: BLE001
        logger.debug("sessions: detail failed", exc_info=True)
        raise HTTPException(503, _UNAVAILABLE) from None
    if out is None:
        raise HTTPException(404, "session not found")
    return out


@router.get("/api/requests/{request_id}")
async def request_detail(
    request: Request,
    request_id: str,
    wait_ms: int = Query(0, ge=0, le=5000),
    viewer: ForestViewer = Depends(forest_viewer),
) -> dict[str, Any]:
    """One turn by its ``X-LMW-Request-Id``; ``wait_ms`` waits for the commit."""
    ref = parse_ref("r_" + request_id)
    if ref is None:
        raise HTTPException(404, "request not found")
    try:
        out = await service.get_request(request.app.state, viewer, ref.ref, wait_ms / 1000.0)
    except service.NotRecordedYet:
        raise HTTPException(404, "not recorded yet") from None
    except Exception:  # noqa: BLE001
        logger.debug("sessions: request failed", exc_info=True)
        raise HTTPException(503, _UNAVAILABLE) from None
    if out is None:
        raise HTTPException(404, "request not found")
    return out


@router.get("/api/capture/{request_id}")
async def capture_read(
    request: Request, request_id: str, _user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """Admin only: one captured request (exact bodies and raw response) of a key
    whose debug capture was on (app/proxy/capture.py). 404 when not captured."""
    store = getattr(request.app.state, "capture_store", None)
    rec: dict[str, Any] | None = (
        await asyncio.to_thread(store.get, request_id) if store is not None else None
    )
    if rec is None:
        raise HTTPException(404, "not captured")
    return rec
