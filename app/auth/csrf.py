import hmac
import secrets
from hashlib import sha256

from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

from app.auth.bearer import ADMIN_TOKEN_PREFIX, parse_bearer_header
from app.auth.cookies import cookie_secure


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


async def ensure_csrf_id(request: Request, call_next) -> Response:
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

    if minted_new:
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
# Middleware: csrf_check
# ---------------------------------------------------------------------------
# Runs innermost (added after ensure_csrf_id in middleware stack).
# Validates X-CSRF-Token header (or _csrf form field) on mutating requests.

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_BYPASS_PREFIXES = (
    "/v1/", "/login", "/logout", "/healthz", "/static", "/api/auth", "/api/setup",
)


def _admin_bearer(request: Request) -> bool:
    """``Authorization: Bearer vwa_...`` (spec 2026-09-19, decision 8).

    CSRF defends the credentials a browser attaches by itself -- cookies. A
    bearer header is not one: a cross-site page cannot set it without a CORS
    preflight this server never grants. An invalid admin token is still
    refused, by the auth dependency's 401. Admin tokens only: the browser UI's
    session JWT is a header too, but that request keeps the check as before.
    """
    token = parse_bearer_header(request.headers.get("authorization"))
    return token is not None and token.startswith(ADMIN_TOKEN_PREFIX)


async def csrf_check(request: Request, call_next) -> Response:
    """Reject mutating requests that lack a valid CSRF token."""
    if request.method in _SAFE_METHODS:
        return await call_next(request)

    path = request.url.path
    if any(path.startswith(p) for p in _BYPASS_PREFIXES) or _admin_bearer(request):
        return await call_next(request)

    token = request.headers.get("X-CSRF-Token")
    if not token:
        # Fallback: plain HTML form posts may use a hidden _csrf field.
        # Read the raw body and cache it on request._body so FastAPI can re-read it.
        ct = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
            body = await request.body()  # reads + caches bytes in request._body
            # For url-encoded forms, parse _csrf directly from raw bytes.
            if "application/x-www-form-urlencoded" in ct:
                from urllib.parse import parse_qs
                qs = parse_qs(body.decode("utf-8", errors="replace"))
                vals = qs.get("_csrf")
                token = vals[0] if vals else None

    if not token:
        return JSONResponse({"detail": "csrf token invalid"}, status_code=403)

    binding_id = getattr(request.state, "csrf_id", None)
    if not binding_id:
        return JSONResponse({"detail": "csrf token invalid"}, status_code=403)

    secret = request.app.state.settings.cookie_secret
    if not verify_csrf_token(token, binding_id, secret=secret):
        return JSONResponse({"detail": "csrf token invalid"}, status_code=403)

    return await call_next(request)
