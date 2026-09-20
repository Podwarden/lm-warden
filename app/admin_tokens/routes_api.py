"""Admin tokens: issue, list, refresh and revoke
(docs/superpowers/specs/2026-09-19-admin-tokens-design.md, "API").

An admin token is an ``api_tokens`` row with ``scope = 'admin'`` and a ``vwa_``
secret; it calls the control API as the user that issued it (see
app/auth/deps.py::require_jwt). Every route here is SESSION-ONLY
(``require_session``; app/auth/policy.py::SESSION_ONLY_ROUTES): an admin token
can do anything else, but it cannot copy itself, extend itself, or hide by
revoking its siblings (decision 3). The inference-token routes under
/api/tokens never show these rows (app/tokens/routes_api.py).
"""

import secrets
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from app.auth.bearer import generate_admin_token
from app.auth.deps import admin_principal, require_session
from app.db.database import open_db
from app.db.repos.admin_audit import AdminAuditRepo
from app.db.repos.tokens import TokenRepo, TokenRow, sqlite_utc_in, sqlite_utc_now

router = APIRouter(prefix="/api/admin-tokens", tags=["admin-tokens"])

#: Revoked and expired tokens stay in the list, dimmed, this long -- so their
#: audit trail stays one click away (spec "API").
DEAD_VISIBLE_DAYS = 30

_SQLITE_FMT = "%Y-%m-%d %H:%M:%S"

AdminTokenStatus = Literal["active", "grace", "revoked", "expired"]


class AdminTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    # Required, and JSON null means "never expires": the one value a client
    # has to send on purpose (decision 5). Omitting the key is a 422.
    expires_in_days: Literal[30, 90, 365] | None

    @field_validator("name", mode="before")
    @classmethod
    def _trim(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v


class AdminTokenRotate(BaseModel):
    """How long the old secret keeps working after a refresh (decision 6)."""

    grace_hours: Literal[0, 1, 24] = 1


class AdminToken(BaseModel):
    id: str
    name: str
    prefix: str
    created_by: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    revoked_at: str | None
    rotated_from: str | None
    status: AdminTokenStatus


class AdminTokenIssued(AdminToken):
    plaintext: str = Field(
        description="The secret. Returned only here, once; only its SHA-256 hash is stored."
    )


class AdminTokenList(BaseModel):
    items: list[AdminToken]


def admin_status(row: TokenRow, now: str) -> AdminTokenStatus:
    """require_jwt's verdict as a badge, judged at ``now``: expiry first, then
    revocation -- a FUTURE revoked_at is a refresh's grace window."""
    if row.expires_at is not None and row.expires_at <= now:
        return "expired"
    if row.revoked_at is not None:
        return "revoked" if row.revoked_at <= now else "grace"
    return "active"


def term_days(row: TokenRow) -> int:
    """The token's lifetime in whole days (at least 1), or 0 for never. A
    refresh gives the successor the same term, starting now (Rulings #4)."""
    if row.expires_at is None:
        return 0
    created = datetime.strptime(row.created_at, _SQLITE_FMT)
    expires = datetime.strptime(row.expires_at, _SQLITE_FMT)
    return max(1, round((expires - created).total_seconds() / 86400))


def _out(row: TokenRow, now: str) -> AdminToken:
    return AdminToken(
        id=row.id,
        name=row.name,
        prefix=row.prefix,
        created_by=row.created_by,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        rotated_from=row.rotated_from,
        status=admin_status(row, now),
    )


def _issued(row: TokenRow, plaintext: str) -> AdminTokenIssued:
    return AdminTokenIssued(**_out(row, sqlite_utc_now()).model_dump(), plaintext=plaintext)


async def admin_row_or_404(repo: TokenRepo, token_id: str) -> TokenRow:
    """The admin token ``token_id``; 404 for an unknown id AND for an
    inference key -- this API does not see those, as /api/tokens does not see
    admin rows."""
    row = await repo.get(token_id)
    if row is None or row.scope != "admin":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "admin token not found")
    return row


@router.get("", response_model=AdminTokenList)
async def list_admin_tokens(
    request: Request, _user: str = Depends(require_session)
) -> AdminTokenList:
    """Live admin tokens first (active, or in a refresh's grace window), then
    those revoked or expired within the last 30 days; newest first in each."""
    now = sqlite_utc_now()
    since = sqlite_utc_in(timedelta(days=-DEAD_VISIBLE_DAYS))
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await TokenRepo(db).list_admin(now=now, since=since)
    return AdminTokenList(items=[_out(r, now) for r in rows])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=AdminTokenIssued)
async def issue_admin_token(
    body: AdminTokenCreate, request: Request, user: str = Depends(require_session)
) -> AdminTokenIssued:
    """Issue an admin token acting as the signed-in user. ``plaintext`` is in
    this response and nowhere else, ever."""
    plaintext = generate_admin_token()
    token_id = secrets.token_hex(16)
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        await repo.create(
            token_id=token_id,
            name=body.name,
            plaintext=plaintext,
            scope="admin",
            expires_in_days=body.expires_in_days or 0,
            created_by=user,
        )
        row = await repo.get(token_id)
    if row is None:  # defensive -- the insert above committed
        raise HTTPException(500, "admin token vanished after create")
    return _issued(row, plaintext)


@router.post(
    "/{token_id}/rotate", status_code=status.HTTP_201_CREATED, response_model=AdminTokenIssued
)
async def rotate_admin_token(
    token_id: str,
    body: AdminTokenRotate,
    request: Request,
    _user: str = Depends(require_session),
) -> AdminTokenIssued:
    """Refresh: a new secret with the same name, owner and term (starting
    now); the old one keeps working for ``grace_hours``. 409 for a token that
    was already refreshed, or that is revoked or expired -- a dead token is
    not brought back by a refresh; issue a new one."""
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        old = await admin_row_or_404(repo, token_id)
        state = admin_status(old, sqlite_utc_now())
        if old.rotated_at is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Admin token '{old.name}' was already refreshed; refresh its successor.",
            )
        if state in ("revoked", "expired"):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Admin token '{old.name}' is {state}; issue a new one.",
            )
        try:
            new_id, plaintext, _renamed = await repo.rotate(
                old_id=token_id, grace_hours=body.grace_hours, expires_in_days=term_days(old)
            )
        except ValueError as exc:  # a racing refresh flipped rotated_at first
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        new = await repo.get(new_id)
    if new is None:  # defensive -- rotate() committed the successor
        raise HTTPException(500, "admin token vanished after refresh")
    if body.grace_hours == 0:
        # No grace: the old secret is dead now, and so are its open streams.
        request.app.state.stream_registry.cancel_user(admin_principal(token_id))
    return _issued(new, plaintext)


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_admin_token(
    token_id: str, request: Request, _user: str = Depends(require_session)
) -> None:
    """Revoke now: the next request with it gets 401 and its open streams are
    cancelled. The row stays (dimmed in the list) so its audit trail stays
    readable. Revoking a revoked token is a no-op that keeps the original
    time; revoking a predecessor in its grace window ends the grace."""
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        row = await admin_row_or_404(repo, token_id)
        if row.revoked_at is None or row.revoked_at > sqlite_utc_now():
            await repo.revoke(token_id)
    request.app.state.stream_registry.cancel_user(admin_principal(token_id))


class AdminAuditRow(BaseModel):
    id: int
    ts: float = Field(description="When the request arrived, epoch seconds.")
    method: str
    path: str = Field(description="The route template, e.g. /api/models/{model_id}/load.")
    status: int
    duration_ms: int
    client_ip: str | None
    peer_ip: str | None = Field(
        description="The socket peer (a reverse proxy's address when behind one); "
        "cannot be set by the client."
    )
    username: str


class AdminAuditPage(BaseModel):
    items: list[AdminAuditRow]
    next_before: float | None = Field(
        description="Pass as ?before= for the next, older page; null when there is none."
    )


@router.get("/{token_id}/audit", response_model=AdminAuditPage)
async def admin_token_audit(
    token_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    before: float | None = Query(
        default=None, description="Epoch seconds; only rows strictly older are returned."
    ),
    _user: str = Depends(require_session),
) -> AdminAuditPage:
    """What an admin token has done, newest first (decision 7). Rows sharing a
    timestamp are never split across pages."""
    async with open_db(request.app.state.settings.db_path) as db:
        await admin_row_or_404(TokenRepo(db), token_id)
        rows, next_before = await AdminAuditRepo(db).page(token_id, limit=limit, before=before)
    return AdminAuditPage(
        items=[
            AdminAuditRow(
                id=r.id, ts=r.ts, method=r.method, path=r.path, status=r.status,
                duration_ms=r.duration_ms, client_ip=r.client_ip, peer_ip=r.peer_ip,
                username=r.username,
            )
            for r in rows
        ],
        next_before=next_before,
    )
