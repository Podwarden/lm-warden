"""Credentials of the Prompt Lens API and MCP (analysis Task 4.1).

``/api/lens/me/*`` takes a key holder's credential, either of:

* an inference key (``vw_...``), accepted exactly as ``/v1/chat/completions`` accepts it
  (``require_bearer``: unknown, expired, revoked and rotated-away keys 401, a paused key 403);
* the key holder's sign-in token (the forest JWT of ``POST /api/forest/login``).

An admin credential (session or ``vwa_`` token) gets 403 there: the admin has
``/api/lens/admin/*`` (A-1). ``POST /api/lens/mcp`` takes the same two, and an admin credential
too, which gets the admin tool set.

``lens_credential`` and ``lens_mcp_credential`` only say WHO; ``gate`` then applies the key's
``prompt_lens`` permission and the platform switch (the same 404 as the UI routes when off, so
off, unknown and foreign look alike) and the rate limit shared by REST and MCP.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Literal, TypeVar

from fastapi import HTTPException, Request

from app.auth.bearer import ADMIN_TOKEN_PREFIX, INFERENCE_TOKEN_PREFIX, parse_bearer_header
from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, TokenRow
from app.lens.limits import limiter_of
from app.proxy.auth import LMWARDEN_KEY_HEADER, require_bearer
from app.stats.routes_forest import forest_viewer
from app.tokens.permissions import prompt_lens_allowed

log = logging.getLogger("app.lens.api")

NOT_YOURS = "no request of yours with this id"
ADMIN_ON_ME = "an admin credential uses /api/lens/admin; this route reads one key's own traffic"

T = TypeVar("T")


@dataclass(frozen=True)
class LensCred:
    #: "key" (an inference key), "forest" (a key holder's sign-in) or "admin".
    via: Literal["key", "forest", "admin"]
    #: The key's id; None for an admin.
    token_id: str | None
    #: The admin's user name; None for a key holder.
    user: str | None = None

    @property
    def admin(self) -> bool:
        return self.via == "admin"

    @property
    def credential_id(self) -> str:
        return f"admin:{self.user}" if self.admin else str(self.token_id)


def _bearer(request: Request) -> str | None:
    header = (request.headers.get(LMWARDEN_KEY_HEADER) or "").strip()
    return header or parse_bearer_header(request.headers.get("authorization"))


async def _credential(request: Request, *, allow_admin: bool) -> LensCred:
    tok = _bearer(request)
    if tok is not None and tok.startswith(INFERENCE_TOKEN_PREFIX):
        row: TokenRow = await require_bearer(request)
        return LensCred("key", row.id)
    if tok is not None and tok.startswith(ADMIN_TOKEN_PREFIX):
        user = await require_jwt(request)
        if not allow_admin:
            raise HTTPException(403, ADMIN_ON_ME)
        return LensCred("admin", None, user)
    viewer = await forest_viewer(request)  # a forest JWT; else an admin session, or 401
    if viewer.token_id is None:
        if not allow_admin:
            raise HTTPException(403, ADMIN_ON_ME)
        return LensCred("admin", None, "session")
    return LensCred("forest", viewer.token_id)


async def lens_credential(request: Request) -> LensCred:
    """A key holder's credential for ``/api/lens/me/*``: an inference key or a sign-in."""
    return await _credential(request, allow_admin=False)


async def lens_mcp_credential(request: Request) -> LensCred:
    """The MCP endpoint's credential: a key holder's, or an admin's."""
    return await _credential(request, allow_admin=True)


def key_hash(cred: LensCred) -> str:
    """A short hash of the credential's id for logs (never the id or the secret)."""
    return hashlib.sha256(cred.credential_id.encode()).hexdigest()[:8]


async def key_row(request: Request, token_id: str) -> TokenRow | None:
    async with open_db(request.app.state.settings.db_path) as db:
        return await TokenRepo(db).get(token_id)


def too_many(wait: int) -> HTTPException:
    return HTTPException(
        429,
        "too many Prompt Lens calls; try again shortly",
        headers={"Retry-After": str(max(1, wait))},
    )


async def gate(request: Request, cred: LensCred, *, diff: bool = False) -> None:
    """The key's Prompt Lens permission and the platform switch (404 when either is off, as
    the UI routes answer), then the rate limit shared by REST and MCP (429 + Retry-After)."""
    if not cred.admin and not await prompt_lens_allowed(request, str(cred.token_id)):
        raise HTTPException(404, NOT_YOURS)
    wait = limiter_of(request.app).check(cred.credential_id, admin=cred.admin, diff=diff)
    if wait is not None:
        raise too_many(wait)


async def logged(request: Request, cred: LensCred, route: str, work: Awaitable[T]) -> T:
    """Await ``work`` and log one line: the route template, a hash of the credential, the
    status and the milliseconds. Never a query, a body or a credential."""
    t0 = time.monotonic()
    status = 200
    try:
        return await work
    except HTTPException as exc:
        status = exc.status_code
        raise
    except Exception:
        status = 500
        raise
    finally:
        log.info(
            "lens-api route=%s key=%s status=%d ms=%d",
            route,
            key_hash(cred),
            status,
            int((time.monotonic() - t0) * 1000),
        )
