from fastapi import HTTPException, Request

from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, TokenRow, sqlite_utc_now


async def require_bearer(request: Request) -> TokenRow:
    """Validate Bearer token, return the full TokenRow, update last_used_at.

    Raises 401 if missing/malformed/unknown/expired/revoked, and 403 if the
    key is paused (token details page, migration 0033).
    """
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    plaintext = auth[7:].strip()
    if not plaintext.startswith("vw_"):
        raise HTTPException(401, "invalid token format")
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = TokenRepo(db)
        row = await repo.find_by_plaintext(plaintext)
        if row is None:
            raise HTTPException(401, "unknown token")
        if row.expires_at is not None and row.expires_at <= sqlite_utc_now():
            raise HTTPException(401, "token expired")
        # `revoked_at` is set to a FUTURE timestamp by rotate() to implement
        # a grace window during which the predecessor must keep working.
        # Reject only once the grace window has elapsed (revoked_at <= now).
        # The `<=` is load-bearing for immediate revoke (#185): rotate with
        # grace_hours=0 writes revoked_at = now, and both sides are
        # second-granularity strings, so a request landing in the same second
        # compares EQUAL. Tightening this to `<` would silently give a
        # hard-revoked token an up-to-1s admission window.
        if row.revoked_at is not None and row.revoked_at <= sqlite_utc_now():
            raise HTTPException(401, "token revoked")
        # Paused by the operator. 403, not 401, so a client can tell "your key
        # is paused" from "your key is wrong". Checked AFTER expiry and
        # revocation so a key that is both dead and paused still gets the 401:
        # the dead state wins. Requests already in flight are not touched, and
        # there is no cache to flush -- this reads the row on every request.
        # Raised before touch_last_used: a refused request is not a use.
        if row.paused_at is not None:
            raise HTTPException(403, "token paused")
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
