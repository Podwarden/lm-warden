"""G7: the key holder's live Prompt Lens stream, bound to the key on the server.

* ``POST /api/me/prompt-lens/ticket`` -- the key's forest sign-in (``key_holder``)
  mints a single-use, 60 s SSE ticket for ``ME_STREAM_PATH``. Its subject is
  ``key_stream_key(token id)``, and the sign-in's session id rides inside it,
  signed. 404 ``"Prompt Lens is off"`` unless platform Prompt Lens is on
  and the admin allowed Prompt Lens for the key (``prompt_lens_view_available``, the key's
  ``prompt_lens`` permission, owner decision 2026-11-26).
* ``GET /api/me/prompt-lens/stream?ticket=`` -- ``require_key_ticket`` spends the
  ticket and returns ``key:<id>``; the stream registers under it, so revoking,
  rotating or pausing the key and a key sign-out end it at once
  (``stream_registry.cancel_user``). It is the shared generator
  (app/proxy/prompt_lens_stream.py) with ``wanted = {id}`` -- the stream takes no
  key parameter -- ``transform = for_key_holder``, no ``ring`` frames, and a
  re-check on its own timer: the key still usable (``token_refusal``), the
  sign-in still active and not past its ``exp``, and platform Prompt Lens still
  on, else the stream ends.

Limits per key (review I2): at most ``MAX_STREAMS_PER_KEY`` open streams and
``MINTS_PER_MINUTE`` ticket mints a minute, else 429 with Retry-After. The
stream replays the key's own window only and numbers its frames per
connection (app/proxy/prompt_lens_stream.py).

Residual, accepted by the spec: a key's own ``progress.phase`` and its
request_end ``queued_s`` / ``ttft_s`` show the server's load, other keys'
traffic included, as any client timing its own requests would.

Spec 2026-11-22 section 4 (isolation rules 1-6). Nothing here logs a ticket or
any content.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse

from app.auth.deps import (
    KEY_STREAM_PREFIX,
    bearer_token,
    key_stream_key,
    require_jwt,
    token_refusal,
)
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, sqlite_utc_now
from app.proxy import prompt_lens_kept
from app.proxy.prompt_lens_scope import for_key_holder
from app.proxy.prompt_lens_sections import SectionKind
from app.proxy.prompt_lens_stream import prompt_lens_sse
from app.proxy.routes_prompt_lens import (
    FROM_QUERY,
    KeptCapture,
    KeptList,
    SectionResponse,
    section_body,
)
from app.tokens.routes_me import key_holder, prompt_lens_view_available

router = APIRouter(prefix="/api/me/prompt-lens", tags=["prompt-lens"])

TICKET_PATH = "/api/me/prompt-lens/ticket"
ME_STREAM_PATH = "/api/me/prompt-lens/stream"

#: Open live streams one key may hold (several tabs or devices); more is 429.
MAX_STREAMS_PER_KEY = 4
#: Ticket mints one key may make a minute; more is 429.
MINTS_PER_MINUTE = 30

_OFF = "Prompt Lens is off"
_STREAMS_FULL = "this key already has the most live streams open"
#: The one 404 of every capture read: off, unknown, released and another key's
#: id all look the same (as /api/me/requests/{id}/capture).
_NOT_YOURS = "no capture of yours with this id"


def _now() -> float:
    """Wall-clock seconds (a JWT ``exp`` is one). Tests patch it."""
    return time.time()


def _too_many(detail: str, retry_after: int) -> HTTPException:
    return HTTPException(429, detail, headers={"Retry-After": str(max(1, retry_after))})


def _mint_allowed(request: Request, tid: str) -> int:
    """0 when ``tid`` may mint now (and counts the mint), else seconds to wait."""
    book: dict[str, deque[float]] | None = getattr(request.app.state, "key_ticket_mints", None)
    if book is None:
        book = {}
        request.app.state.key_ticket_mints = book
    now = time.monotonic()
    if len(book) > 1024:  # forget keys with no mint in the last minute
        for k in [k for k, d in book.items() if not d or now - d[-1] > 60.0]:
            del book[k]
    d = book.setdefault(tid, deque(maxlen=MINTS_PER_MINUTE))
    if len(d) == MINTS_PER_MINUTE and now - d[0] < 60.0:
        return int(60.0 - (now - d[0])) + 1
    d.append(now)
    return 0


async def _key_usable(request: Request, tid: str, sid: Any, exp: Any) -> HTTPException | None:
    """Why the key ``tid`` signed in as ``sid`` (expiring at ``exp``) may not
    stream now, or None."""
    if not isinstance(exp, int | float) or isinstance(exp, bool) or _now() >= exp:
        return HTTPException(401, "session expired")
    sessions = getattr(request.app.state, "auth_sessions", None)
    if sessions is None or not sessions.is_active(sid, kind="key", subject=tid):
        return HTTPException(401, "session ended")
    async with open_db(request.app.state.settings.db_path) as db:
        row = await TokenRepo(db).get(tid)
    if row is None or row.scope != "inference":
        return HTTPException(401, "unknown token")
    return token_refusal(row, sqlite_utc_now())


async def require_key_ticket(
    request: Request,
    ticket: str | None = Query(
        default=None,
        description="Single-use ticket from POST /api/me/prompt-lens/ticket (the key's sign-in).",
    ),
) -> str:
    """Authenticate the key holder's stream; return its registry key ``key:<id>``.

    Ticket only (ruling G-F2). A ticket presented is spent first, whatever
    follows, so one that reached a log is already dead. A header credential is
    then refused: an admin session or admin token 403 (this is the key
    holder's own stream), a forest sign-in, an inference key or anything else
    401. The ticket must be for ``ME_STREAM_PATH`` and carry a key subject; the
    key must still be usable and the sign-in that minted it still active and
    unexpired.
    """
    claims: dict[str, Any] | None = None
    error = "missing ticket"
    if ticket is not None:
        try:
            claims = request.app.state.sse_tickets.consume_claims(ticket, ME_STREAM_PATH)
        except ValueError as exc:
            error = str(exc)
    if bearer_token(request) is not None:
        await require_jwt(request)  # 401 unless an admin session / admin token
        raise HTTPException(403, "sign in with your API key to use this stream")
    if claims is None:
        raise HTTPException(401, error)
    sub = str(claims["sub"])
    tid = sub[len(KEY_STREAM_PREFIX) :] if sub.startswith(KEY_STREAM_PREFIX) else ""
    if not tid:
        raise HTTPException(401, "not a key holder's ticket")
    sid, exp = claims.get("sid"), claims.get("exp")
    refusal = await _key_usable(request, tid, sid, exp)
    if refusal is not None:
        raise refusal
    request.state.key_stream = (tid, sid, exp)
    return key_stream_key(tid)


@router.post("/ticket")
async def mint_ticket(request: Request, tid: str = Depends(key_holder)) -> dict[str, str]:
    """A single-use ticket for the key holder's live stream (60 s). At most
    30 a minute per key, and none while the key has 4 streams open (429 with
    Retry-After)."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, _OFF)
    # The stream cap again here (ruling G-R15): the stream's own 429 comes on
    # the EventSource GET, whose status a browser cannot read; this one it
    # can. Checked before the mint budget, so a refusal spends no mint.
    if request.app.state.stream_registry.count(key_stream_key(tid)) >= MAX_STREAMS_PER_KEY:
        raise _too_many(_STREAMS_FULL, 5)
    wait = _mint_allowed(request, tid)
    if wait:
        raise _too_many("too many stream tickets for this key; try again shortly", wait)
    sid = getattr(request.state, "session_id", None)
    exp = getattr(request.state, "session_exp", None)
    ticket = request.app.state.sse_tickets.mint(
        key_stream_key(tid), ME_STREAM_PATH, sid=sid, exp=exp if isinstance(exp, int) else None
    )
    return {"ticket": ticket}


@router.get("/stream")
async def key_stream(
    request: Request, user: str = Depends(require_key_ticket)
) -> StreamingResponse:
    """The signed-in key's own requests, live (SSE): the stream is bound to the key
    that minted the ticket and takes no key parameter. Ends when the key is
    revoked, rotated or paused, the sign-in ends or expires, or Prompt Lens is turned
    off. At most 4 open per key (429)."""
    tid, sid, exp = request.state.key_stream
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, _OFF)
    if request.app.state.stream_registry.count(user) >= MAX_STREAMS_PER_KEY:
        raise _too_many(_STREAMS_FULL, 5)
    hub = request.app.state.prompt_lens_hub

    async def recheck() -> bool:
        if not await prompt_lens_view_available(request, tid):
            return False
        return await _key_usable(request, tid, sid, exp) is None

    # No await between the count above and the registration inside
    # prompt_lens_sse: concurrent connects cannot both slip under the cap.
    return await prompt_lens_sse(
        request,
        hub,
        user=user,
        wanted=frozenset({tid}),
        transform=for_key_holder,
        ring_meta=False,
        recheck=recheck,
        key_id=tid,
    )


def kept_read_allowed(request: Request, tid: str) -> None:
    """429 past ``prompt_lens_kept.READS_PER_MINUTE`` DB-backed reads a minute for
    this key (review I3). Counted before the lookup, so an unknown or another
    key's id costs the same as an own one."""
    book = getattr(request.app.state, "prompt_lens_kept_reads", None)
    if book is None:
        book = request.app.state.prompt_lens_kept_reads = prompt_lens_kept.ReadBook()
    wait = book.wait(tid)
    if wait:
        raise _too_many("too many kept capture reads for this key; try again shortly", wait)


@router.get("/requests/{req_id}/section/{kind}", response_model=SectionResponse)
async def my_section(
    request: Request,
    req_id: Annotated[str, Path(max_length=64)],
    kind: SectionKind,
    from_: int = FROM_QUERY,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """One whole section of one of the signed-in key's OWN requests (G9).

    One 404 ``"no capture of yours with this id"`` for Prompt Lens off, an
    unknown id, a released or expired one, and another key's. The body never
    carries the key id or the hub's seq."""
    sections = getattr(request.app.state, "prompt_lens_sections", None)
    if sections is None or not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, _NOT_YOURS)
    view = sections.section(req_id, kind, start=from_)
    if view is None or view.token_id != tid:
        # Not in memory (or not this key's): the key's OWN kept captures only.
        kept_read_allowed(request, tid)
        async with open_db(request.app.state.settings.db_path) as db:
            view = await prompt_lens_kept.section(db, req_id, kind, start=from_, token_id=tid)
    if view is None or view.token_id != tid:
        raise HTTPException(404, _NOT_YOURS)
    return section_body(view, key_holder=True)


@router.get("/kept", response_model=KeptList)
async def my_kept(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    """The signed-in key's kept captures, newest first (at most 50: an admin
    turned keep_captures on for this key). 404 when Prompt Lens is off."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, _NOT_YOURS)
    kept_read_allowed(request, tid)
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await prompt_lens_kept.list_for(
            db, frozenset({tid}), limit=prompt_lens_kept.KEPT_PER_KEY
        )
    return {"kept": [{k: v for k, v in r.items() if k != "token_id"} for r in rows]}


@router.get("/kept/{req_id}", response_model=KeptCapture)
async def my_kept_capture(
    request: Request,
    req_id: Annotated[str, Path(max_length=64)],
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """One of the key's OWN kept captures, in the key holder's view
    (``for_key_holder`` on every event). One 404 for off, unknown and another
    key's id."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, _NOT_YOURS)
    kept_read_allowed(request, tid)
    async with open_db(request.app.state.settings.db_path) as db:
        found = await prompt_lens_kept.get_events(db, req_id, token_id=tid)
    if found is None or found[0] != tid:
        raise HTTPException(404, _NOT_YOURS)
    _tid, events, summary = found
    return {
        "req_id": req_id,
        "kept": True,
        "events": prompt_lens_kept.holder_events(events),
        "sections": summary.get("sections") or {},
    }
