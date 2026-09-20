"""God mode live-stream SSE endpoint — ``GET /api/admin/godmode/stream``.

Mirrors the model-loading log stream (``app/models/routes_logs.py``): guarded by
the same single-use SSE ticket (minted by the ``require_jwt``-gated
``POST /api/auth/sse-ticket``), so EventSource — which cannot send an
Authorization header — composes with the existing proxy/Caddy SSE path. There is
no viewer/operator/admin RBAC in this project; the warden's single privileged UI
session (a valid access JWT → a ticket) is the gate.

On connect: send a ``: connected`` comment, replay the hub's ring snapshot as
``data:`` frames, then stream live events from the subscriber queue. Idle windows emit a ``: keepalive`` comment so
intermediate proxies don't sever the connection. The subscriber is removed in a
``finally`` on disconnect / cancellation / close.

``?token_ids=a,b`` (token details page, spec 2026-09-18 §3.5) narrows both the
replay and the live events to those keys. Absent, the stream is byte-for-byte
what it was before the parameter existed -- scripts use it that way -- apart
from the leading ``: connected`` comment (#251), which every SSE parser skips.
"""

import asyncio
import base64
import binascii
import json
import re
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from app.auth.deps import require_jwt
from app.auth.stream_guard import guard_stream
from app.models.routes_logs import require_sse_ticket
from app.proxy.godmode import MAX_TOKEN_IDS, event_matches, parse_token_ids
from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/admin/godmode", tags=["godmode"])

# Full path of the live-stream endpoint. Exported so the shared SSE-ticket
# mint (``POST /api/auth/sse-ticket``) can recognise a god-mode ticket request
# and refuse it up front when the feature is off — the browser's EventSource
# can never read the stream endpoint's own 409 (``onerror`` carries no HTTP
# status), so the viewer classifies the disabled state from the *mint*
# response. Keep this in sync with the router prefix + route below.
STREAM_PATH = "/api/admin/godmode/stream"

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
_MEDIA_ID_RE = re.compile(r"^[0-9a-f]{16}\Z")


@router.get("/media/{media_id}")
async def godmode_media(
    media_id: str, request: Request, user: str = Depends(require_jwt)
):
    """Serve one stored god-mode image (spec 2026-08-03, §5). Plain authed
    fetch (NOT EventSource), so require_jwt applies directly — no SSE ticket.
    Decoding happens here, off the proxy hot path."""
    settings = request.app.state.settings
    store = getattr(request.app.state, "godmode_media", None)
    if store is None or not settings.godmode_enabled:
        raise HTTPException(409, "god mode is disabled (VW_GODMODE_ENABLED)")
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


@router.get("/status")
async def godmode_status(request: Request, user: str = Depends(require_jwt)) -> dict[str, bool]:
    """Whether god mode is on. The token page asks before it renders the dock
    at all (spec 2026-09-18 §3.5). A plain authed fetch, like ``/media``: no
    SSE ticket, and nothing is subscribed."""
    settings = request.app.state.settings
    hub = getattr(request.app.state, "godmode_hub", None)
    return {"enabled": hub is not None and bool(settings.godmode_enabled)}


@router.get("/stream")
async def godmode_stream(
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
    settings = request.app.state.settings
    hub = getattr(request.app.state, "godmode_hub", None)
    # Clear "disabled" signal (not a dead stream) so the UI can render a
    # "god mode is disabled (VW_GODMODE_ENABLED)" placeholder rather than
    # spinning on an empty EventSource.
    if hub is None or not settings.godmode_enabled:
        raise HTTPException(409, "god mode is disabled (VW_GODMODE_ENABLED)")
    # Parsed BEFORE subscribing so a refused filter leaves no subscriber.
    try:
        wanted = parse_token_ids(token_ids)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    queue, snapshot = hub.subscribe()

    async def gen():
        try:
            # Registered for logout / revoke, and an admin token re-checked
            # while the stream is open (app/auth/stream_guard.py).
            async with guard_stream(request, user):
                yield CONNECTED_FRAME
                # Replay the ring snapshot first so a freshly-opened page
                # immediately shows the last few requests.
                for event in snapshot:
                    if event_matches(event, wanted):
                        yield f"data: {json.dumps(event)}\n\n"
                # Live loop: block on the queue until the next keepalive is due;
                # on timeout emit a comment frame and re-check the connection.
                #
                # The wait is the time REMAINING until that deadline, not a fresh
                # KEEPALIVE_INTERVAL_S, and only a frame this client receives
                # moves the deadline. A filtered stream can be busy with OTHER
                # keys' events: each one wakes this loop without reaching the
                # client, so a full-interval wait after it would let the gap grow
                # to about twice the interval (#251). Unfiltered, every event is
                # sent and resets the deadline, so the cadence is as before.
                keepalive_due = time.monotonic() + KEEPALIVE_INTERVAL_S
                while True:
                    if await request.is_disconnected():
                        return
                    remaining = keepalive_due - time.monotonic()
                    try:
                        if remaining <= 0:
                            raise TimeoutError
                        event = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        keepalive_due = time.monotonic() + KEEPALIVE_INTERVAL_S
                        continue
                    if event_matches(event, wanted):
                        yield f"data: {json.dumps(event)}\n\n"
                        keepalive_due = time.monotonic() + KEEPALIVE_INTERVAL_S
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
