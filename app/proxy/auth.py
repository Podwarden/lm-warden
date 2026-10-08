from typing import Any

from fastapi import HTTPException, Request

from app.auth.bearer import INFERENCE_TOKEN_PREFIX, parse_bearer_header
from app.auth.deps import token_refusal
from app.auth.throttle import bearer_throttle, check_bearer
from app.db.database import open_db
from app.db.repos.tokens import TokenRepo, TokenRow, hash_token, sqlite_utc_now
from app.proxy.token_cache import MISS, generation

#: The dedicated header Claude Code carries the LM Warden key in (#287).
LMWARDEN_KEY_HEADER = "x-lmwarden-key"


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
    #
    # #287: `X-LMWarden-Key`, when present and non-empty, IS the LM Warden
    # credential and nothing else is tried: `Authorization` / `x-api-key` then
    # belong to the client's upstream (Anthropic) account and are never
    # validated here. Absent: today's behaviour, exactly.
    header_key = (request.headers.get(LMWARDEN_KEY_HEADER) or "").strip()
    request.state.lmwarden_key_header = bool(header_key)
    plaintext = header_key or parse_bearer_header(request.headers.get("authorization"))
    if not plaintext and _is_anthropic_route(request):
        # The Anthropic SDKs send the key as `x-api-key` (Claude Code does
        # with ANTHROPIC_API_KEY; ANTHROPIC_AUTH_TOKEN sends a Bearer header).
        # Same secret, same checks below -- only where it is read from, and
        # only on the Anthropic-shaped routes (#281).
        plaintext = (request.headers.get("x-api-key") or "").strip() or None
    if not plaintext:
        raise HTTPException(401, "missing bearer token")
    if not plaintext.startswith(INFERENCE_TOKEN_PREFIX):
        raise HTTPException(401, "invalid token format")
    # Unknown-secret throttle (app/auth/throttle.py): an address spraying
    # secrets that match no row gets 429 + Retry-After before the lookup. A
    # secret already seen to match a row is never throttled.
    address = check_bearer(request, plaintext)
    settings = request.app.state.settings
    secret_hash = hash_token(plaintext)
    row: TokenRow | None = _cache_get(request, secret_hash)
    if row is MISS:
        # Snapshot BEFORE the read: a revoke committing while this lookup is in
        # flight bumps the generation, and the stale row is then dead on arrival.
        gen = generation()
        async with open_db(settings.db_path) as db:
            row = await TokenRepo(db).find_by_plaintext(plaintext)
        # Only a row is cached: an unknown secret must reach the lookup (and
        # the throttle) every time.
        if row is not None and row.scope == "inference":
            _cache_put(request, secret_hash, row, gen)
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
    # reasons (app/auth/deps.py::token_refusal), evaluated per request from
    # the (possibly cached) row. Raised before ledger.touch: a refused
    # request is not a use. Rows are cached for TOKEN_CACHE_TTL_S and dropped
    # on every TokenRepo change (app/proxy/token_cache.py).
    refusal = token_refusal(row, sqlite_utc_now())
    if refusal is not None:
        raise refusal
    ledger = getattr(request.app.state, "ledger", None)
    if ledger is not None:
        ledger.touch(row.id)
    return row


def _cache_get(request: Request, secret_hash: str) -> Any:
    # Fail open: any cache trouble is a plain DB lookup.
    try:
        cache = getattr(request.app.state, "token_cache", None)
        return MISS if cache is None else cache.get(secret_hash)
    except Exception:
        return MISS


def _cache_put(request: Request, secret_hash: str, row: TokenRow, gen: int) -> None:
    try:
        cache = getattr(request.app.state, "token_cache", None)
        if cache is not None:
            cache.put(secret_hash, row, gen)
    except Exception:
        pass


def upstream_credential_headers(request: Request) -> dict[str, str]:
    """The client's own upstream credential (#287), lower-case names.

    ``authorization`` / ``x-api-key`` as sent, but ONLY when the request
    authenticated through ``X-LMWarden-Key`` -- otherwise those headers hold
    (or may hold) the LM Warden key itself, which must never be relayed.
    """
    if not getattr(request.state, "lmwarden_key_header", False):
        return {}
    out: dict[str, str] = {}
    for name in ("authorization", "x-api-key"):
        value = request.headers.get(name)
        if value:
            out[name] = value
    return out


#: Route templates (app/proxy/routes_messages.py) that also accept `x-api-key`.
ANTHROPIC_ROUTES = frozenset({"/v1/messages", "/v1/messages/count_tokens"})


def _is_anthropic_route(request: Request) -> bool:
    # The matched route's template, not request.url.path: FastAPI puts the
    # route in the scope before dependencies run, and the template does not
    # change with a root_path or a trailing query (`?beta=true`).
    return getattr(request.scope.get("route"), "path", None) in ANTHROPIC_ROUTES


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
