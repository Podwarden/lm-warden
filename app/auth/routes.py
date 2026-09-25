import asyncio
import weakref

import bcrypt
import jwt as pyjwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.auth.cookies import cookie_secure
from app.auth.deps import require_session, session_stream_key
from app.auth.jwt import decode, mint_access, mint_refresh
from app.auth.origin import origin_check_dep
from app.auth.throttle import (
    LOGIN_THROTTLED_MESSAGE,
    login_throttle,
    throttle_client,
    too_many_attempts,
)
from app.db.database import open_db
from app.db.repos.users import UserRepo

# bcrypt.checkpw against this constant when the user is unknown, so the
# response time matches the "user exists, wrong password" path. Otherwise an
# attacker can enumerate usernames by timing /api/auth/login.
_DUMMY_HASH = bcrypt.hashpw(b"timing-equalizer", bcrypt.gensalt()).decode()

router = APIRouter(prefix="/api/auth", tags=["auth"])

#: bcrypt checks running at once. Each takes a CPU for a quarter of a second
#: or so; they run in worker threads (bcrypt releases the GIL) so a burst of
#: logins no longer stalls the event loop -- and with it /v1 -- and this cap
#: keeps a burst from taking every core.
BCRYPT_SLOTS = 2
_bcrypt_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


async def check_password(password: str, password_hash: str) -> bool:
    """``bcrypt.checkpw`` off the event loop, at most BCRYPT_SLOTS at once."""
    loop = asyncio.get_running_loop()
    slots = _bcrypt_slots.get(loop)
    if slots is None:
        slots = _bcrypt_slots[loop] = asyncio.Semaphore(BCRYPT_SLOTS)
    async with slots:
        return await asyncio.to_thread(
            bcrypt.checkpw, password.encode("utf-8"), password_hash.encode("utf-8")
        )


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.post(
    "/login",
    responses={
        401: {"description": "Invalid credentials."},
        429: {
            "description": (
                "Too many failed logins from this (public, proxy-vouched) "
                "address, or a failed login while too many others are being "
                "held back. Never the answer to a correct password unless "
                "the address is locked. Wait for the Retry-After header "
                "(seconds). See app/auth/throttle.py for the limits."
            ),
        },
    },
)
async def login(body: LoginBody, request: Request, response: Response):
    settings = request.app.state.settings
    # Brute-force throttle (app/auth/throttle.py). The LOCK is checked before
    # the database and bcrypt, so a locked caller costs nothing -- and only a
    # public client address the trusted proxies vouch for can be locked. It
    # answers the same for a known and an unknown username, so it leaks
    # nothing the timing equalizer below hides.
    throttle = login_throttle(request)
    client = throttle_client(request)
    wait = throttle.retry_after(client)
    if wait > 0:
        raise too_many_attempts(wait, LOGIN_THROTTLED_MESSAGE)
    async with open_db(settings.db_path) as db:
        user = await UserRepo(db).get_by_username(body.username)
    # An unknown user is checked against a constant hash, so the response
    # time matches "user exists, wrong password" (and is refused whatever
    # the constant's password).
    ok = await check_password(
        body.password, user.password_hash if user is not None else _DUMMY_HASH
    )
    if user is None or not ok:
        # The THROTTLE: a failed attempt is answered late (escalating, capped
        # at a few seconds), or 429 when too many are already being held.
        # Only failures ever get here, so no throttle state refuses a correct
        # password.
        await throttle.failed(client, body.username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    throttle.success(client, body.username)
    secret = request.app.state.jwt_secret
    access_ttl = settings.session_access_ttl_minutes
    refresh_ttl = settings.session_refresh_ttl_days
    access = mint_access(user.username, secret, ttl_minutes=access_ttl)
    refresh = mint_refresh(user.username, secret, ttl_days=refresh_ttl)
    # `secure` is derived, never hardcoded: a Secure cookie is discarded by
    # the browser on a plain-http:// origin, which is the documented quick
    # start. See app/auth/cookies.py for the derivation and the bug it fixes.
    response.set_cookie(
        "vw_refresh", refresh,
        max_age=refresh_ttl * 86400,
        httponly=True, secure=cookie_secure(request), samesite="strict",
        path="/api/auth",
    )
    return {"access_token": access, "expires_in": access_ttl * 60}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT,
             dependencies=[Depends(origin_check_dep)])
async def logout(request: Request, response: Response,
                 user: str = Depends(require_session)):
    # This session's streams (keyed session:<username>); an admin token's
    # streams are keyed by the token and outlive a browser logout.
    request.app.state.stream_registry.cancel_user(session_stream_key(user))
    # Match every attribute the cookie was SET with. A browser keys a cookie
    # on (name, domain, path), so the expiry alone does the deletion -- but a
    # deletion that disagrees on `secure`/`samesite` is rejected outright by
    # some browsers, which would leave a live refresh token behind on logout.
    response.delete_cookie(
        "vw_refresh", path="/api/auth",
        httponly=True, secure=cookie_secure(request), samesite="strict",
    )
    return None


@router.post("/refresh", dependencies=[Depends(origin_check_dep)])
async def refresh(
    request: Request,
    vw_refresh: str | None = Cookie(default=None),
):
    if vw_refresh is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing refresh cookie")
    secret = request.app.state.jwt_secret
    try:
        claims = decode(vw_refresh, secret)
    except pyjwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token") from exc
    if claims.get("typ") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing subject")
    access_ttl = request.app.state.settings.session_access_ttl_minutes
    access = mint_access(sub, secret, ttl_minutes=access_ttl)
    return {"access_token": access, "expires_in": access_ttl * 60}


class TicketBody(BaseModel):
    path: str


@router.post("/sse-ticket")
async def mint_sse_ticket(
    body: TicketBody, request: Request, user: str = Depends(require_session)
):
    # God mode is the one stream that can be switched off at runtime. The
    # browser's EventSource can never read the stream endpoint's own 409
    # (`onerror` exposes no HTTP status), so the viewer classifies the
    # disabled state from THIS mint response. Refuse a god-mode ticket up
    # front when the feature is off. Scoped to the god-mode path only — every
    # other stream (e.g. model logs) mints unconditionally as before.
    from app.proxy.routes_godmode import STREAM_PATH as GODMODE_STREAM_PATH

    if body.path == GODMODE_STREAM_PATH:
        settings = request.app.state.settings
        hub = getattr(request.app.state, "godmode_hub", None)
        if hub is None or not settings.godmode_enabled:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "god mode is disabled (VW_GODMODE_ENABLED)",
            )

    return {"ticket": request.app.state.sse_tickets.mint(user, body.path)}
