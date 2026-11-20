import hmac
import secrets
from collections.abc import Awaitable, Callable
from hashlib import sha256

from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

from app.auth.bearer import ADMIN_TOKEN_PREFIX, INFERENCE_TOKEN_PREFIX, parse_bearer_header
from app.auth.cookies import cookie_secure
from app.auth.origin import browser_origin, settings_origin_allowed
from app.auth.public_routes import mints_csrf

CallNext = Callable[[Request], Awaitable[Response]]


def generate_csrf_token(session_id: str, *, secret: str) -> str:
    return hmac.new(secret.encode(), session_id.encode(), sha256).hexdigest()


def verify_csrf_token(token: str, session_id: str, *, secret: str) -> bool:
    expected = generate_csrf_token(session_id, secret=secret)
    return hmac.compare_digest(token, expected)


# ---------------------------------------------------------------------------
# Middleware: ensure_csrf_id
# ---------------------------------------------------------------------------
# Runs outermost. Guarantees request.state.csrf_id and request.state.csrf_token
# are populated for every request before any route handler fires.

_CSRF_COOKIE = "vw_csrf_id"
_SESSION_COOKIE = "vw_session"

# Public pages and static documents (the website, /agent-guide.md) never carry
# a form or a mutating call, so an anonymous visitor there gets no cookie.
# Minting one made every response -- including the year-long immutable video
# and image assets -- carry Set-Cookie, which a CDN in front of a public
# instance treats as "do not cache", and handed a cookie to every ad click
# that never goes near the console. The UI mints its own through /api/csrf.
# Which paths those are is registered (app/auth/public_routes.py: PublicRoute
# with mint_csrf=False), not listed here.


async def ensure_csrf_id(request: Request, call_next: CallNext) -> Response:
    """Populate request.state.csrf_id / csrf_token; auto-mint vw_csrf_id when needed."""
    # Prefer the session cookie as the HMAC binding ID (gives per-user tokens).
    session_val = request.cookies.get(_SESSION_COOKIE)
    csrf_id_val = request.cookies.get(_CSRF_COOKIE)

    minted_new = False
    if session_val:
        binding_id = session_val
    elif csrf_id_val:
        binding_id = csrf_id_val
    else:
        # No existing identity — mint a fresh anonymous CSRF ID.
        binding_id = secrets.token_urlsafe(16)
        minted_new = True

    request.state.csrf_id = binding_id
    secret = request.app.state.settings.cookie_secret
    request.state.csrf_token = generate_csrf_token(binding_id, secret=secret)

    response: Response = await call_next(request)

    if minted_new and mints_csrf(request.app, request.url.path):
        # Same derivation as the refresh cookie (app/auth/cookies.py). These
        # two used to disagree -- refresh hardcoded Secure, this one hardcoded
        # not-Secure -- so on the documented plain-HTTP quick start the
        # browser kept the CSRF cookie and dropped the refresh cookie, and
        # the session looked half-alive instead of absent. They must agree:
        # they describe the same connection.
        response.set_cookie(
            _CSRF_COOKIE,
            binding_id,
            httponly=True,
            samesite="strict",
            secure=cookie_secure(request),
            max_age=60 * 60 * 24,
        )

    return response


# ---------------------------------------------------------------------------
# Middleware: csrf_check  (conventions: security.csrf@1)
# ---------------------------------------------------------------------------
# Runs innermost (added after ensure_csrf_id in middleware stack).
#
# Gated on COOKIES, not on a list of paths. A browser attaches cookies to a
# request another site starts; it cannot attach an Authorization header (that
# needs a CORS preflight this server never grants). So:
#
# 1. A request carrying none of this app's cookies has nothing ambient to
#    abuse. It skips this layer and its route's own auth decides -- this is
#    what keeps every SDK, CLI and curl caller working.
# 2. A request authenticated by an API key (vw_) or an admin token (vwa_) is
#    the same: the bearer is the credential, whatever cookies ride along.
# 3. Every other state-changing request must come from an allowed origin
#    (Origin, else Referer; app/auth/origin.py's one allow-list), when the
#    browser names one -- on EVERY path, the token-exempt ones included.
# 4. ... and must echo the HMAC token, except on the few routes below that a
#    browser reaches before it can hold one, or whose credential is a header.

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
#: Every cookie this app sets that a request can ride on. A new cookie that
#: authenticates or binds anything must be added here. lw_lang (the website's
#: language choice, site/) is deliberately absent: it grants nothing.
APP_COOKIES = (_SESSION_COOKIE, "vw_refresh", _CSRF_COOKIE)
#: Routes that keep the ORIGIN check but skip the TOKEN check (/v1/ never gets
#: here at all: app/main.py's PathBypass sends it round this pair, because it
#: takes an API key header and nothing else):
#:   /api/auth/    login (no token yet), refresh/logout (origin-checked by
#:                 their own dependency, Origin required), the SSE ticket;
#:   /api/setup/   the first-run wizard, before any account exists (guarded
#:                 by the setup code, app/setup/code.py);
#:   /api/me/      the key holder's area: a bearer key-holder token only.
_TOKEN_EXEMPT_PREFIXES = ("/api/auth/", "/api/setup/", "/api/me/")
# Exact paths (not prefixes). Key-holder login and sign-out: the credential is an
# explicit `Authorization: Bearer ...` header (the key, then the key holder's
# JWT), not a cookie.
_TOKEN_EXEMPT_PATHS = frozenset({"/api/forest/login", "/api/forest/logout"})
#: The largest form body read to find a hidden _csrf field.
_FORM_CSRF_MAX_BYTES = 64 * 1024
_ORIGIN_REFUSED = "origin mismatch"
_TOKEN_REFUSED = "csrf token invalid"


def _key_bearer(request: Request) -> bool:
    """``Authorization: Bearer vw_...`` or ``vwa_...``: an API key or an admin
    token (spec 2026-09-19, decision 8).

    CSRF defends the credentials a browser attaches by itself -- cookies. A
    bearer header is not one. An invalid key is still refused, by the route's
    own 401. The browser UI's session JWT is a header too, but that request
    keeps the checks: it is the console, and the console has the cookies.

    Matching on the prefix alone (no lookup here) is safe ONLY because this
    app installs no CORS middleware: a cross-site page can neither set an
    Authorization header without a preflight nobody answers, nor read any
    answer. Adding CORSMiddleware means revisiting this exemption first;
    tests/unit/auth/test_csrf_origin_gate.py fails until someone does.
    """
    token = parse_bearer_header(request.headers.get("authorization"))
    return token is not None and (
        token.startswith(ADMIN_TOKEN_PREFIX) or token.startswith(INFERENCE_TOKEN_PREFIX)
    )


def _refused(detail: str) -> Response:
    return JSONResponse({"detail": detail}, status_code=403)


async def csrf_check(request: Request, call_next: CallNext) -> Response:
    """Refuse a state-changing cookie request from a foreign origin, or one
    that lacks a valid CSRF token."""
    if request.method in _SAFE_METHODS:
        return await call_next(request)
    if not any(name in request.cookies for name in APP_COOKIES) or _key_bearer(request):
        return await call_next(request)

    origin = browser_origin(request)
    if origin is not None and not settings_origin_allowed(request, origin):
        return _refused(_ORIGIN_REFUSED)

    path = request.url.path
    if path in _TOKEN_EXEMPT_PATHS or path.startswith(_TOKEN_EXEMPT_PREFIXES):
        return await call_next(request)

    token = request.headers.get("X-CSRF-Token")
    if not token:
        # Fallback: plain HTML form posts may use a hidden _csrf field. Only a small,
        # url-encoded body with a declared length is read for it (verified finding #6): a
        # multipart or chunked body was read whole into memory, before any auth, to be refused.
        ct = request.headers.get("content-type", "")
        try:
            declared = int(request.headers.get("content-length", ""))
        except ValueError:
            declared = -1
        if "application/x-www-form-urlencoded" in ct and 0 <= declared <= _FORM_CSRF_MAX_BYTES:
            from urllib.parse import parse_qs

            body = await request.body()  # reads + caches bytes in request._body
            vals = parse_qs(body.decode("utf-8", errors="replace")).get("_csrf")
            token = vals[0] if vals else None

    binding_id = getattr(request.state, "csrf_id", None)
    if not token or not binding_id:
        return _refused(_TOKEN_REFUSED)

    secret = request.app.state.settings.cookie_secret
    if not verify_csrf_token(token, binding_id, secret=secret):
        return _refused(_TOKEN_REFUSED)

    return await call_next(request)
