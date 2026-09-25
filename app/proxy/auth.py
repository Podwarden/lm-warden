from fastapi import HTTPException, Request

from app.auth.bearer import INFERENCE_TOKEN_PREFIX, parse_bearer_header
from app.auth.deps import token_refusal
from app.auth.throttle import bearer_throttle, check_bearer
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, TokenRow, sqlite_utc_now


async def require_bearer(request: Request) -> TokenRow:
    """Validate Bearer token, return the full TokenRow, update last_used_at.

    Raises 401 if missing/malformed/unknown/expired/revoked, 403 if the key
    is paused (token details page, migration 0033), and 429 with Retry-After
    when this address has presented too many unknown secrets
    (app/auth/throttle.py).
    """
    # The same parser as the control API (app/auth/deps.py::bearer_token):
    # the scheme is case-insensitive and any whitespace separates it from the
    # credential.
    plaintext = parse_bearer_header(request.headers.get("authorization"))
    if not plaintext:
        raise HTTPException(401, "missing bearer token")
    if not plaintext.startswith(INFERENCE_TOKEN_PREFIX):
        raise HTTPException(401, "invalid token format")
    # Unknown-secret throttle (app/auth/throttle.py): an address spraying
    # secrets that match no row gets 429 + Retry-After before the lookup. A
    # secret already seen to match a row is never throttled.
    address = check_bearer(request, plaintext)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = TokenRepo(db)
        row = await repo.find_by_plaintext(plaintext)
        if row is None:
            bearer_throttle(request).unknown(address)
        else:
            bearer_throttle(request).matched(plaintext)
        # An admin token is never a /v1 credential (spec 2026-09-19, decision
        # 2). The vw_ prefix check above already refuses every vwa_ secret;
        # this refuses any non-inference row whose secret happens to look like
        # vw_ -- as "unknown", so a probe learns nothing.
        if row is None or row.scope != "inference":
            raise HTTPException(401, "unknown token")
        # Expired / revoked (a FUTURE revoked_at is a rotation's grace window)
        # / paused: the ladder the control API's admin tokens share, with its
        # reasons (app/auth/deps.py::token_refusal). Requests already in
        # flight are not touched, and there is no cache to flush -- this reads
        # the row on every request. Raised before touch_last_used: a refused
        # request is not a use.
        refusal = token_refusal(row, sqlite_utc_now())
        if refusal is not None:
            raise refusal
        await repo.touch_last_used(row.id)
    return row


def token_allows(token: TokenRow, served_name: str) -> bool:
    """Return True if the token permits access to served_name.

    - allowed_models is None  → all models allowed (unrestricted token)
    - allowed_models is a CSV → served_name must appear as one of the items
      (each item is stripped of whitespace; empty items never match)
    """
    if token.allowed_models is None:
        return True
    allowed = {item.strip() for item in token.allowed_models.split(",") if item.strip()}
    return served_name in allowed
