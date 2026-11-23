"""Prompt Lens live-stream SSE endpoint — ``GET /api/admin/prompt-lens/stream``.

Mirrors the model-loading log stream (``app/models/routes_logs.py``): guarded by
the same single-use SSE ticket (minted by the ``require_jwt``-gated
``POST /api/auth/sse-ticket``), so EventSource — which cannot send an
Authorization header — composes with the existing proxy/Caddy SSE path. There is
no viewer/operator/admin RBAC in this project; the warden's single privileged UI
session (a valid access JWT → a ticket) is the gate.

On connect: send a ``: connected`` comment, a ``ring`` frame (what the replay
ring holds; repeated every 30 s), replay the hub's ring snapshot as ``data:``
frames, then stream live events from the subscriber queue. Idle windows emit a
``: keepalive`` comment so intermediate proxies don't sever the connection; a
client that fell behind its bounded queue gets one ``lagged`` frame per gap.
The subscriber is removed in a ``finally`` on disconnect / cancellation /
close. The generator is shared with the key holder's own stream
(app/proxy/prompt_lens_stream.py).

``?token_ids=a,b`` (token details page, spec 2026-09-18 §3.5) narrows both the
replay and the live events to those keys. Absent, the stream is byte-for-byte
what it was before the parameter existed -- scripts use it that way -- apart
from the leading ``: connected`` comment (#251), which every SSE parser skips,
and the ``ring`` (G7) and ``lagged`` frames: ``data:`` frames of two new
``type`` values, which a script that reads request events by ``type`` skips.
"""

import base64
import binascii
import dataclasses
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.models.routes_logs import require_sse_ticket
from app.proxy import prompt_lens_kept, prompt_lens_switch
from app.proxy.prompt_lens import MAX_TOKEN_IDS, parse_token_ids
from app.proxy.prompt_lens_sections import SectionKind, SectionView
from app.proxy.prompt_lens_stream import admin_view, prompt_lens_sse

router = APIRouter(prefix="/api/admin/prompt-lens", tags=["prompt-lens"])

# Full path of the live-stream endpoint. Exported so the shared SSE-ticket
# mint (``POST /api/auth/sse-ticket``) can recognise a Prompt Lens ticket request
# and refuse it up front when the feature is off — the browser's EventSource
# can never read the stream endpoint's own 409 (``onerror`` carries no HTTP
# status), so the viewer classifies the disabled state from the *mint*
# response. Keep this in sync with the router prefix + route below.
STREAM_PATH = "/api/admin/prompt-lens/stream"

# Keepalive cadence: emit an SSE comment line after this many idle seconds so
# intermediate proxies (nginx/Caddy default 60s) and EventSource's own timers
# don't tear down a quiet stream. Module-level so tests can monkeypatch a
# sub-second value rather than waiting real time. Mirrors routes_logs.py.
KEEPALIVE_INTERVAL_S: float = 15.0

# The stream's very first bytes (#251). EventSource fires ``open`` -- the
# viewer's "Live" badge -- only once bytes arrive, and with an empty ring and a
# quiet hub the first ones used to be a keepalive up to KEEPALIVE_INTERVAL_S
# later. An SSE comment is ignored by every parser, so it carries no event.
CONNECTED_FRAME = ": connected\n\n"

# Media ids are secrets.token_hex(8) -> exactly 16 lowercase hex chars. The
# regex is defense-in-depth (the store is a flat dict, so no traversal is
# possible anyway) and lets us 404 junk without a store lookup.
#: The 409 detail of every admin Prompt Lens route while it is off.
DISABLED = "Prompt Lens is disabled (Settings, or VW_PROMPT_LENS_ENABLED)"

_MEDIA_ID_RE = re.compile(r"^[0-9a-f]{16}\Z")


@router.get("/media/{media_id}")
async def prompt_lens_media(media_id: str, request: Request, user: str = Depends(require_jwt)):
    """Serve one stored Prompt Lens image (spec 2026-08-03, §5). Plain authed
    fetch (NOT EventSource), so require_jwt applies directly — no SSE ticket.
    Decoding happens here, off the proxy hot path."""
    store = getattr(request.app.state, "prompt_lens_media", None)
    if store is None or not prompt_lens_switch.enabled(request.app):
        raise HTTPException(409, DISABLED)
    if not _MEDIA_ID_RE.match(media_id):
        raise HTTPException(404, "unknown media id")
    item = store.get(media_id)
    if item is None:
        raise HTTPException(404, "media evicted or unknown")
    mime, b64 = item
    try:
        data = base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error) as err:
        raise HTTPException(404, "media payload undecodable") from err
    return Response(
        content=data,
        media_type=mime,
        headers={"Cache-Control": "private, max-age=3600"},
    )


class SectionResponse(BaseModel):
    """One whole section of a request (G9), as ``GET .../section/{kind}`` returns it.

    ``offset`` is the absolute char offset ``text`` starts at (at least
    ``dropped_chars``, the oldest text the store already dropped), so
    ``offset + len(text)`` is the section's length: the pop-up refetches with
    ``?from=`` that and appends live deltas after it. ``seq`` is the hub seq of
    the last event folded in; null for a key holder (it counts every key's
    events). ``calls`` for tool_calls, by index; ``messages`` for tool_results
    and prompt (``sys`` and ``tools`` first); ``older`` the prompt's collapsed
    summary. ``kept``: served from the key's kept captures (on disk)."""

    kind: SectionKind
    seq: int | None = None
    live: bool
    dropped_chars: int
    offset: int
    text: str | None = None
    calls: list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]] | None = None
    dropped_calls: int = 0
    older: dict[str, Any] | None = None
    kept: bool = False


def section_body(view: SectionView, *, key_holder: bool) -> dict[str, Any]:
    """``view`` as the route returns it: never the key id; no hub seq for a key holder."""
    body = dataclasses.asdict(view)
    body.pop("token_id", None)
    if key_holder:
        body["seq"] = None
    return body


#: ``?from=``: the absolute char offset to read a text section from.
FROM_QUERY = Query(0, alias="from", ge=0, description="Absolute char offset (text sections).")


@router.get("/requests/{req_id}/section/{kind}", response_model=SectionResponse)
async def prompt_lens_section(
    request: Request,
    req_id: Annotated[str, Path(max_length=64)],
    kind: SectionKind,
    from_: int = FROM_QUERY,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """One whole section of any key's request (G9): the dock's pop-up. 404 when
    neither memory nor the kept captures have it. While Prompt Lens is off only
    kept captures (on disk) are served: an admin can still read them."""
    settings = request.app.state.settings
    sections = getattr(request.app.state, "prompt_lens_sections", None)
    view = None
    if sections is not None and prompt_lens_switch.enabled(request.app):
        view = sections.section(req_id, kind, start=from_)
    if view is None:
        # Not in memory any more: a kept capture (keep_captures), of any key.
        async with open_db(settings.db_path) as db:
            view = await prompt_lens_kept.section(db, req_id, kind, start=from_, token_id=None)
    if view is None:
        raise HTTPException(404, "no such capture")
    return section_body(view, key_holder=False)


class KeptSummary(BaseModel):
    """One kept capture in a list (Prompt Lens keep captures, newest first)."""

    req_id: str
    request_id: str | None = None
    model: str | None = None
    stream: bool | None = None
    session_id: str | None = None
    turn: int | None = None
    started_ts: float | None = None
    ended_ts: float | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    kept: bool = True


class AdminKeptSummary(KeptSummary):
    token_id: str


class KeptList(BaseModel):
    kept: list[KeptSummary]


class AdminKeptList(BaseModel):
    kept: list[AdminKeptSummary]


class KeptCapture(BaseModel):
    """One kept capture: its request_start, kvsig and request_end events (as
    the caller may see them) and each section's size; the sections themselves
    come from ``.../section/{kind}``."""

    req_id: str
    kept: bool = True
    events: list[dict[str, Any]]
    sections: dict[str, Any]


class AdminKeptCapture(KeptCapture):
    token_id: str


#: Most kept captures one admin list returns (50 per key are kept).
KEPT_LIST_MAX = 1000


@router.get("/kept", response_model=AdminKeptList)
async def prompt_lens_kept_list(
    request: Request,
    token_ids: Annotated[
        str | None,
        Query(
            description=f"Comma-separated api_tokens ids, at most {MAX_TOKEN_IDS}; omit for all."
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=KEPT_LIST_MAX)] = 200,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Kept captures (keys with keep_captures on), newest first. Also while
    Prompt Lens is off: what is on disk stays readable to an admin."""
    try:
        wanted = parse_token_ids(token_ids)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await prompt_lens_kept.list_for(db, wanted, limit=limit)
    return {"kept": rows}


@router.get("/kept/{req_id}", response_model=AdminKeptCapture)
async def prompt_lens_kept_capture(
    request: Request,
    req_id: Annotated[str, Path(max_length=64)],
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """One kept capture of any key, as the admin sees it (also while Prompt Lens
    is off). 404 unknown."""
    async with open_db(request.app.state.settings.db_path) as db:
        found = await prompt_lens_kept.get_events(db, req_id, token_id=None)
    if found is None:
        raise HTTPException(404, "no such capture")
    tid, events, summary = found
    return {
        "req_id": req_id,
        "token_id": tid,
        "kept": True,
        "events": prompt_lens_kept.admin_events(events),
        "sections": summary.get("sections") or {},
    }


class PromptLensChange(BaseModel):
    """One save of the switch: when, by whom, through what, before and after."""

    ts: float
    actor: str
    via: str
    old_enabled: bool
    new_enabled: bool
    old_source: str
    new_source: str


class PromptLensStatus(BaseModel):
    """Whether Prompt Lens is on, and where that comes from (the Settings page
    shows it): ``source`` is ``setting`` (the runtime setting
    ``prompt_lens_enabled``), ``env`` (``env_var``: ``VW_PROMPT_LENS_ENABLED``
    or the deprecated ``VW_GODMODE_ENABLED``) or ``default`` (off)."""

    enabled: bool
    source: Literal["setting", "env", "default"]
    setting: bool | None = None
    env_var: str | None = None
    env_value: bool | None = None
    #: The newest save of the switch (prompt_lens_switch_audit), or None.
    last_change: PromptLensChange | None = None


@router.get("/status", response_model=PromptLensStatus)
async def prompt_lens_status(request: Request, user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Whether Prompt Lens is on. The token page asks before it renders the dock
    at all (spec 2026-09-18 §3.5). A plain authed fetch, like ``/media``: no
    SSE ticket, and nothing is subscribed."""
    out = prompt_lens_switch.status(request.app)
    async with open_db(request.app.state.settings.db_path) as db:
        out["last_change"] = await prompt_lens_switch.last_change(db)
    return out


@router.get("/stream")
async def prompt_lens_stream(
    request: Request,
    user: str = Depends(require_sse_ticket),
    token_ids: Annotated[
        str | None,
        Query(
            description=(
                f"Comma-separated api_tokens ids, at most {MAX_TOKEN_IDS}. "
                "Only those keys' events are replayed and streamed. "
                "Omit for every key."
            ),
        ),
    ] = None,
):
    hub = getattr(request.app.state, "prompt_lens_hub", None)
    # Clear "disabled" signal (not a dead stream) so the UI can render a
    # "Prompt Lens is off" placeholder rather than spinning on an empty
    # EventSource.
    if hub is None or not prompt_lens_switch.enabled(request.app):
        raise HTTPException(409, DISABLED)
    # Parsed BEFORE subscribing so a refused filter leaves no subscriber.
    try:
        wanted = parse_token_ids(token_ids)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return await prompt_lens_sse(
        request,
        hub,
        user=user,
        wanted=wanted,
        transform=admin_view,
        ring_meta=True,
        recheck=None,
    )
