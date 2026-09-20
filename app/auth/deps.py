"""Authentication dependencies for the control API (/api/*).

``require_jwt`` guards every /api route that is not public (most ``Depends``
sites in the app). Despite its name it accepts two credentials
(docs/superpowers/specs/2026-09-19-admin-tokens-design.md):

* a session access JWT from ``POST /api/auth/login`` -- ``principal = "session"``;
* an admin token, ``vwa_...`` -- ``principal = "admin_token:<id>"``. It acts as
  the user that issued it (``api_tokens.created_by``).

Either way it returns a username, so chat ownership and every other caller
work unchanged; ``request.state.principal`` records which credential it was.

``require_session`` is the session-only variant for
app/auth/policy.py::SESSION_ONLY_ROUTES: it identifies an admin token (so the
attempt lands in that token's audit trail) and answers it with 403
``session_only`` instead of trying it.

One path handles an admin secret everywhere -- require_jwt, require_session,
the SSE streams (app/models/routes_logs.py::require_sse_ticket) and the
streams' periodic re-check (app/auth/stream_guard.py):
``identify_admin_token`` says WHICH token it is, ``token_refusal`` says
whether that token may be used right now.
"""

from typing import Any

import jwt as pyjwt
from fastapi import HTTPException, Request, status

from app.auth.admin_audit import STATE_KEY, AdminAuditContext
from app.auth.bearer import ADMIN_TOKEN_PREFIX, parse_bearer_header
from app.auth.jwt import decode
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, TokenRow, sqlite_utc_now
from app.utils.client_ip import client_ip

SESSION_PRINCIPAL = "session"
_ADMIN_PRINCIPAL_PREFIX = "admin_token:"

SESSION_ONLY_DETAIL: dict[str, str] = {
    "error_code": "session_only",
    "message": (
        "This endpoint needs a signed-in session; admin tokens are refused here. "
        "An admin token cannot issue, refresh, revoke or inspect admin tokens, "
        "sign a session out, or mint SSE tickets, so a leaked one cannot copy, "
        "extend or hide itself."
    ),
}


# #256, widened by the C1/I1 follow-up review. trust_remote_code makes the
# engine execute the target Hugging Face repository's Python inside the warden
# container, with the warden's own privileges (VW_JWT_SECRET, api_tokens,
# admin_audit) -- so an admin token that can set it anywhere can mint itself a
# session and undo the session-only fence around token management. Every write
# path to a true value is therefore session-only, and they live here rather
# than next to one route because three modules refuse with them:
# app/models/routes_api.py (register, register-via-template, template create,
# load) and app/settings/routes_api.py (the settings PATCH).
#
# The first review of #256 guarded only the literal ``trust_remote_code: true``
# body key on POST /api/models plus the load, which left the settings PATCH,
# the template create and a template's own merged value wide open. Keep this
# wording and the code in step: `documents/API.md` quotes it.
TRUST_REMOTE_CODE_SET_MESSAGE = (
    "trust_remote_code runs the target repository's Python inside the warden "
    "container. Setting it to true needs a signed-in session: an admin token "
    "cannot register a model with it (directly or via a template), save a "
    "template that carries it, or patch it onto an existing model."
)
TRUST_REMOTE_CODE_LOAD_MESSAGE = (
    "This model has trust_remote_code=true, so loading it runs the target "
    "repository's Python inside the warden container. Loading it needs a "
    "signed-in session; an admin token cannot load it."
)


def is_session_principal(request: Request) -> bool:
    """Whether the request's credential is a signed-in session, not an admin
    token. Set on ``request.state.principal`` by ``require_jwt`` /
    ``require_session`` before any route body runs."""
    return getattr(request.state, "principal", None) == SESSION_PRINCIPAL


def refuse_session_only(
    request: Request, *, message: str, keys: list[str] | None = None
) -> None:
    """Raise 403 ``session_only`` unless the request's principal is a
    signed-in session -- i.e. refuse an admin token.

    The shared "is this a session? if not, 403" mechanism behind every
    session-only guardrail that is not itself a whole route (whole routes go
    through app/auth/policy.py::SESSION_ONLY_ROUTES / ``require_session``
    instead): the runtime-settings keys in
    app/settings/routes_api.py::_refuse_session_only_keys, and the
    trust_remote_code refusals in app/models/routes_api.py and
    app/settings/routes_api.py (#256 -- an admin token registering, patching or
    loading a trust_remote_code model runs that repository's Python inside the
    warden container, which is full code execution, not a settings change; see
    TRUST_REMOTE_CODE_SET_MESSAGE above). No-op for a session; callers check
    whatever they're refusing (a key was sent, a flag is true, ...) before
    calling this, so a session never pays for the check.
    """
    if is_session_principal(request):
        return
    detail: dict[str, Any] = {"error_code": "session_only", "message": message}
    if keys is not None:
        detail["keys"] = keys
    raise HTTPException(status_code=403, detail=detail)


def admin_principal(token_id: str) -> str:
    """``request.state.principal`` of an admin token -- also the key its SSE
    streams register under in app.state.stream_registry."""
    return f"{_ADMIN_PRINCIPAL_PREFIX}{token_id}"


def session_stream_key(username: str) -> str:
    """The stream-registry key of a browser session's streams. Prefixed, so a
    username can never collide with an ``admin_principal`` key."""
    return f"session:{username}"


def stream_key(request: Request) -> str:
    """The stream-registry key of an authenticated request (set by
    require_jwt / require_sse_ticket): ``admin_token:<id>`` for an admin
    token, ``session:<username>`` for a session. Revoking the token, or
    logging the session out, cancels the streams registered under it
    (app/auth/stream_guard.py)."""
    return str(request.state.stream_key)


def admin_token_id(principal: str) -> str | None:
    """The token id inside an ``admin_principal``; None for anything else
    (a username, ``"session"``)."""
    if principal.startswith(_ADMIN_PRINCIPAL_PREFIX):
        return principal[len(_ADMIN_PRINCIPAL_PREFIX):] or None
    return None


def is_admin_token(token: str) -> bool:
    return token.startswith(ADMIN_TOKEN_PREFIX)


def bearer_token(request: Request) -> str | None:
    """The bearer credential, or None. The scheme is case-insensitive."""
    return parse_bearer_header(request.headers.get("authorization"))


def _required_bearer(request: Request) -> str:
    token = bearer_token(request)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    return token


def _session_subject(request: Request, token: str) -> str:
    try:
        claims = decode(token, request.app.state.jwt_secret)
    except pyjwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    if claims.get("typ") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing subject")
    request.state.principal = SESSION_PRINCIPAL
    request.state.stream_key = session_stream_key(str(sub))
    return str(sub)


def token_refusal(row: TokenRow, now: str) -> HTTPException | None:
    """Why ``row`` may not be used at ``now`` (a SQLite UTC string), or None.

    The one ladder for every api_tokens credential -- an inference key on /v1
    (app/proxy/auth.py::require_bearer) and an admin token on /api -- so both
    answer alike: expired 401, revoked 401, paused 403.

    * ``revoked_at`` is set to a FUTURE timestamp by rotate() to implement a
      grace window during which the predecessor must keep working, so only
      ``revoked_at <= now`` refuses. The ``<=`` is load-bearing for immediate
      revoke (#185): rotate with grace_hours=0 writes revoked_at = now, and
      both sides are second-granularity strings, so a request landing in the
      same second compares EQUAL. Tightening this to ``<`` would silently give
      a hard-revoked token an up-to-1s admission window.
    * Paused is 403, not 401, so a client can tell "your key is paused" from
      "your key is wrong". Checked AFTER expiry and revocation so a key that
      is both dead and paused still gets the 401: the dead state wins.
    """
    if row.expires_at is not None and row.expires_at <= now:
        return HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired")
    if row.revoked_at is not None and row.revoked_at <= now:
        return HTTPException(status.HTTP_401_UNAUTHORIZED, "token revoked")
    if row.paused_at is not None:
        return HTTPException(status.HTTP_403_FORBIDDEN, "token paused")
    return None


async def identify_admin_token(
    request: Request, repo: TokenRepo, plaintext: str
) -> tuple[TokenRow, str]:
    """Which admin token ``plaintext`` is: ``(row, owner username)``.

    Identification only -- the row may be expired, revoked or paused (see
    ``token_refusal``). A secret matching no row, a row whose scope is not
    'admin' (an inference key never unlocks /api) and an ownerless row are
    all 401 "unknown token", so a probe learns nothing.

    Sets the request's audit context (app/auth/admin_audit.py): from here on
    the secret is a known admin token, so even a refused attempt belongs in
    its trail.
    """
    row = await repo.find_by_plaintext(plaintext)
    if row is None or row.scope != "admin" or not row.created_by:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown token")
    setattr(
        request.state,
        STATE_KEY,
        AdminAuditContext(
            token_id=row.id,
            username=row.created_by,
            client_ip=client_ip(request),
            peer_ip=request.client.host if request.client else None,
        ),
    )
    return row, row.created_by


async def authenticate_admin_token(request: Request, plaintext: str) -> str:
    """Validate an admin token; return the username it acts as.

    The checks, their order and their answers are require_bearer's for an
    inference key: unknown 401, expired 401, revoked 401 -- a FUTURE
    revoked_at is a refresh's grace window and still works -- paused 403.
    A refused request is not a use: ``last_used_at`` is stamped only on
    success. Sets ``request.state.principal`` and ``request.state.stream_key``.
    """
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        row, owner = await identify_admin_token(request, repo, plaintext)
        refusal = token_refusal(row, sqlite_utc_now())
        if refusal is not None:
            raise refusal
        await repo.touch_last_used(row.id)
    request.state.principal = admin_principal(row.id)
    request.state.stream_key = request.state.principal
    return owner


async def require_jwt(request: Request) -> str:
    """A session JWT or an admin token; returns the username (module docstring)."""
    token = _required_bearer(request)
    if is_admin_token(token):
        return await authenticate_admin_token(request, token)
    return _session_subject(request, token)


async def require_session(request: Request) -> str:
    """A session JWT only.

    An admin token gets 403 ``session_only`` -- the route refuses the KIND of
    credential, so an expired or paused one gets the same answer. It is
    identified first, so the refusal is filed in that token's audit trail; a
    ``vwa_`` secret that matches no admin token is 401 "unknown token", as on
    every other route.
    """
    token = _required_bearer(request)
    if is_admin_token(token):
        async with open_db(request.app.state.settings.db_path) as db:
            await identify_admin_token(request, TokenRepo(db), token)
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=dict(SESSION_ONLY_DETAIL))
    return _session_subject(request, token)
