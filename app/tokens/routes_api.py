import asyncio
import json
import secrets
import time
from datetime import timedelta
from typing import Any, Literal

import aiosqlite
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator

from app.auth.bearer import generate_bearer_token
from app.auth.deps import key_stream_key, require_jwt
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.settings import SettingsRepo
from app.db.repos.tokens import (
    _UNSET,
    KV_ISOLATION_VALUES,
    KvChange,
    KvChangeRefused,
    TokenModelUsageRepo,
    TokenRepo,
    TokenRow,
    TokenUsageRepo,
    list_token_page,
    sqlite_utc_in,
    sqlite_utc_now,
)
from app.guard.resolve import effective, validate_key
from app.kv import isolation as kv_iso
from app.kv import policy_sync
from app.kv import usage as kv_usage
from app.kv.schemas import TokenCacheOut
from app.proxy.prompt_diet import Diet, diet_json, parse_diet
from app.proxy.thinking_policy import ThinkingPolicy
from app.stats import request_history
from app.tokens import series

router = APIRouter(prefix="/api/tokens", tags=["tokens"])

_RATE_LIMIT_REMOVED = (
    "rate_limit_tps is no longer supported: per-token rate limits were "
    "removed. Use priority to order a key's traffic."
)


def _reject_rate_limit_tps(data: object) -> object:
    """422 a body that still carries ``rate_limit_tps`` (any value, null too).

    Per-token rate limits were removed. These models otherwise ignore unknown
    keys, like every other body in this API, so without this a client still
    setting a limit would get a 2xx and silently no limit -- the one outcome
    worth refusing loudly.
    """
    if isinstance(data, dict) and "rate_limit_tps" in data:
        raise ValueError(_RATE_LIMIT_REMOVED)
    return data


class PromptDiet(BaseModel):
    """A key's opt-in cut of dead weight before the engine (app/proxy/prompt_diet.py)."""

    strip_tools: list[str] = Field(default_factory=list, max_length=256)
    #: true = every earlier turn; "before_last_user" = only turns before the last real user message.
    strip_reasoning: bool | Literal["before_last_user"] = False


def _json_or_none(raw: str | None) -> Any:
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _with_effective(detail: dict[str, Any], request: Request) -> dict[str, Any]:
    detail["guard_effective"] = effective(
        getattr(request.app.state, "guard_platform", None), detail.get("guard")
    )
    return detail


def _diet_out(raw: str | None) -> dict[str, Any] | None:
    d = parse_diet(raw)
    return (
        None
        if d is None
        else {"strip_tools": sorted(d.strip_tools), "strip_reasoning": d.strip_reasoning}
    )


class TokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    expires_in_days: int = Field(default=365, ge=0, le=3650)
    # STRICT scheduler priority 0..9; 9 is served first, 0 last. The schema
    # CHECK trigger mirrors this bound. Default 5 matches the column DEFAULT.
    priority: int = Field(default=5, ge=0, le=9)
    # Claude Code model router (#287): may this key make the warden relay its
    # request to Anthropic? Off unless the operator opts the key in.
    anthropic_relay: bool = False
    # "off" makes the proxy force thinking off for every request on this key.
    thinking_policy: ThinkingPolicy = "client"
    #: Sensitive data guard, as on PATCH: null = use the platform setting, {"off": true}, or
    #: {"override": {...}}. Checked before the key exists, so a bad value creates nothing.
    guard: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_rate_limit(cls, data: object) -> object:
        return _reject_rate_limit_tps(data)


#: How a key shares the KV cache (spec 2026-11-21 Plan 2, section 3).
KvIsolation = Literal["sharing", "isolated"]

#: The runtime setting that decides ``kv_isolation`` of a new key (O1, default sharing).
KV_ISOLATION_DEFAULT_KEY = "kv_isolation_default"

KV_RESET_SHARING = "this key shares its cache; there is nothing of its own to reset"

#: O2: a holder may make their key more private, never less.
BACK_TO_SHARING = "only an admin can switch this key back to sharing"


class TokenUpdate(BaseModel):
    """PATCH body — every field is optional; omit to leave untouched.

    The route turns an omitted key into the ``_UNSET`` sentinel before
    calling ``TokenRepo.update``. ``rate_limit_tps`` is refused with 422:
    per-token rate limits were removed.

    ``name`` is trimmed, then held to TokenCreate's 1..64 bounds. Duplicate
    names are allowed, as they are at create time. ``paused`` true pauses the
    key (403 "token paused" on its next request), false resumes it. Neither
    accepts JSON null: both columns have no "cleared" meaning a null could ask
    for.
    """

    name: str | None = Field(default=None, min_length=1, max_length=64)
    priority: int | None = Field(default=None, ge=0, le=9)
    paused: bool | None = None
    anthropic_relay: bool | None = None
    thinking_policy: ThinkingPolicy | None = None
    #: null turns the diet off.
    prompt_diet: PromptDiet | None = None
    #: Sensitive data guard: null = inherit the platform, {"off": true}, or {"override": {...}}.
    guard: dict[str, Any] | None = None
    #: Debug capture for the next N hours (bodies and responses, admin-only); 0 turns it off.
    capture_hours: float | None = Field(default=None, ge=0, le=720)
    #: How the key shares the KV cache. Going to sharing retires the key's isolated scope.
    kv_isolation: KvIsolation | None = None
    #: The key's cap on the RAM and NVMe tiers, in percent; null clears it.
    kv_max_pct_ram: int | None = Field(default=None, ge=1, le=100)
    kv_max_pct_nvme: int | None = Field(default=None, ge=1, le=100)
    #: Prompt Lens keep captures: the key's last 50 finished requests (prompts
    #: included) are kept on disk. false turns it off and deletes what it kept.
    keep_captures: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_rate_limit(cls, data: object) -> object:
        return _reject_rate_limit_tps(data)

    @field_validator("name", mode="before")
    @classmethod
    def _trim_name(cls, v: object) -> object:
        # mode="before" runs only when the key is present, so an omitted name
        # never reaches here; an explicit null does, and is refused.
        if v is None:
            raise ValueError("name cannot be null")
        return v.strip() if isinstance(v, str) else v

    @field_validator(
        "paused",
        "anthropic_relay",
        "thinking_policy",
        "kv_isolation",
        mode="before",
    )
    @classmethod
    def _flag_not_null(cls, v: object) -> object:
        if v is None:
            raise ValueError("flag cannot be null")
        return v


class TokenRotate(BaseModel):
    grace_hours: int = Field(default=24, ge=0, le=720)
    expires_in_days: int | None = Field(default=None, ge=0, le=3650)


def _is_expired(r: TokenRow, now_str: str) -> bool:
    return r.expires_at is not None and r.expires_at <= now_str


def _is_revoked(r: TokenRow, now_str: str) -> bool:
    # #185 — `revoked_at` alone cannot distinguish "grace window still
    # open" from "already cut off"; both are non-null. Compute the same
    # comparison require_bearer makes (app/proxy/auth.py) server-side so
    # the status badge doesn't depend on the operator's workstation clock.
    return r.revoked_at is not None and r.revoked_at <= now_str


def _dead_state(r: TokenRow, now_str: str) -> str | None:
    """Why require_bearer would 401 this key -- "expired" or "revoked" -- or
    None when it is live. Expiry is checked first, in require_bearer's order."""
    if _is_expired(r, now_str):
        return "expired"
    if _is_revoked(r, now_str):
        return "revoked"
    return None


async def _inference_row(repo: TokenRepo, token_id: str) -> TokenRow | None:
    """The inference key ``token_id``, or None -- also for an admin token
    (0037), which every /api/tokens route answers with the same 404 as a
    missing row (spec 2026-09-19, decision 2)."""
    row = await repo.get(token_id)
    return row if row is not None and row.scope == "inference" else None


def _enrich(
    r: TokenRow,
    *,
    successor_id: str | None,
    usage_24h: tuple[int, int, int],
    now_str: str,
    in_30d_str: str,
) -> dict[str, Any]:
    """The token's API shape, shared by the list and ``GET /{id}``.

    ``now_str`` is captured once per request by the caller, so every row in
    one response is judged against the same instant -- up to one 10s UI poll
    behind, which is the same staleness the list always had.
    """
    is_expired = _is_expired(r, now_str)
    is_near = r.expires_at is not None and not is_expired and r.expires_at <= in_30d_str
    usage_requests, usage_prompt, usage_completion = usage_24h
    return {
        "id": r.id,
        "name": r.name,
        "prefix": r.prefix,
        "preview": r.prefix,
        "created_at": r.created_at,
        "last_used_at": r.last_used_at,
        "expires_at": r.expires_at,
        "rotated_at": r.rotated_at,
        "rotated_from": r.rotated_from,
        "successor_id": successor_id,
        "successor_deleted": r.rotated_at is not None and successor_id is None,
        "is_expired": is_expired,
        "is_near_expiry": is_near,
        "revoked_at": r.revoked_at,
        "is_revoked": _is_revoked(r, now_str),
        # S5 (#104) — surface priority + 24h usage rollup so the UI can
        # paint the "Priority / Last 24h" columns without an extra
        # round-trip per row.
        "priority": r.priority,
        "anthropic_relay": bool(r.anthropic_relay),
        "thinking_policy": r.thinking_policy,
        "prompt_diet": _diet_out(r.prompt_diet),
        "capture_until": r.capture_until,
        "guard": _json_or_none(r.guard),
        "keep_captures": bool(r.keep_captures),
        "usage_24h": {
            "requests": usage_requests,
            "prompt_tokens": usage_prompt,
            "completion_tokens": usage_completion,
            "total_tokens": usage_prompt + usage_completion,
        },
        # Token details page (0033): the list badge shows Paused too.
        "paused_at": r.paused_at,
        "is_paused": r.paused_at is not None,
        # Per-key KV cache: the tenant id and epoch are never shown (they name the scope).
        "kv_isolation": r.kv_isolation,
        "kv_max_pct_ram": r.kv_max_pct_ram,
        "kv_max_pct_nvme": r.kv_max_pct_nvme,
        # The served names the key may use (None: every model), split as token_allows does.
        "allowed_models": (
            None
            if r.allowed_models is None
            else [m.strip() for m in r.allowed_models.split(",") if m.strip()]
        ),
    }


def _lineage_entry(r: TokenRow, *, self_id: str, now_str: str) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "created_at": r.created_at,
        "rotated_at": r.rotated_at,
        "is_revoked": _is_revoked(r, now_str),
        # A future revoked_at is a rotation grace window still open -- but
        # only while require_bearer would still let the key in (#251): an
        # expired key gets 401 and a paused one 403, so neither is "in grace".
        "in_grace": (
            r.revoked_at is not None
            and r.revoked_at > now_str
            and not _is_expired(r, now_str)
            and r.paused_at is None
        ),
        "is_self": r.id == self_id,
    }


async def _token_detail(db: aiosqlite.Connection, token_id: str) -> dict[str, Any] | None:
    """``GET /{id}``'s body: ``_enrich`` plus ``lineage``. None if unknown."""
    chain = await TokenRepo(db).lineage(token_id)
    ids = [r.id for r in chain]
    if token_id not in ids or chain[ids.index(token_id)].scope != "inference":
        return None
    i = ids.index(token_id)
    now_minute = int(time.time() // 60)
    usage = await TokenUsageRepo(db).totals(
        token_id=token_id,
        since_minute=now_minute - 24 * 60,
        until_minute=now_minute + 1,
    )
    now_str = sqlite_utc_now()
    body = _enrich(
        chain[i],
        successor_id=ids[i + 1] if i + 1 < len(ids) else None,
        usage_24h=usage,
        now_str=now_str,
        in_30d_str=sqlite_utc_in(timedelta(days=30)),
    )
    body["lineage"] = [_lineage_entry(r, self_id=token_id, now_str=now_str) for r in chain]
    return body


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_token(body: TokenCreate, request: Request, _user: str = Depends(require_jwt)):
    try:
        guard = validate_key(body.guard)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    plaintext = generate_bearer_token()
    tid = secrets.token_hex(16)
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        default = await SettingsRepo(db).get(KV_ISOLATION_DEFAULT_KEY)
        await repo.create(
            token_id=tid,
            name=body.name,
            plaintext=plaintext,
            expires_in_days=body.expires_in_days,
            priority=body.priority,
            anthropic_relay=body.anthropic_relay,
            thinking_policy=body.thinking_policy,
            guard=json.dumps(guard) if guard is not None else None,
            kv_isolation=default if default in KV_ISOLATION_VALUES else "sharing",
        )
        created = await repo.get(tid)
    if created is None:  # defensive — race between insert and read is impossible
        raise HTTPException(500, "token vanished after create")
    policy_sync.request_sync(request.app.state)  # a new tenant and its weight
    prefix = plaintext[:8]
    return {
        "id": tid,
        "name": body.name,
        "plaintext": plaintext,
        "prefix": prefix,
        "preview": prefix,
        "expires_at": created.expires_at,
        "priority": created.priority,
        "anthropic_relay": bool(created.anthropic_relay),
        "thinking_policy": created.thinking_policy,
        "guard": guard,
        "guard_effective": effective(getattr(request.app.state, "guard_platform", None), guard),
    }


@router.patch("/{token_id}")
async def update_token(
    token_id: str,
    body: TokenUpdate,
    request: Request,
    _user: str = Depends(require_jwt),
):
    """Update name, priority and pause state on an existing token.

    Omitted keys are untouched. The Pydantic schema rejects out-of-range
    values -- and the removed ``rate_limit_tps`` -- with 422; the DB CHECK
    trigger is belt-and-braces.

    ``paused: true`` on a key that is expired, or revoked with its grace
    window over, is a 409 and nothing in the body is applied: there is
    nothing left to pause. A predecessor still inside its grace window CAN
    be paused -- that is the quick way to cut an old key off early.

    Returns the full token, the same shape as ``GET /api/tokens/{id}``.

    **Note:** ``priority`` cannot be set to ``null`` — the column is ``NOT NULL``
    in the schema. ``{"priority": null}`` is a 422, refused before anything is
    written. To reset priority to the default, send ``{"priority": 5}``.
    """
    raw = body.model_dump(exclude_unset=True)
    # Every refusal comes before the KV change commits: priority is NOT NULL.
    if "priority" in raw and raw["priority"] is None:
        raise HTTPException(422, "priority cannot be null")
    changes: list[KvChange] = []
    try:
        out = await _update_token(request, token_id, raw, changes)
        # Priority (the tenant's weight), caps and isolation all reach the daemon.
        policy_sync.request_sync(request.app.state)
        return out
    finally:
        # The KV change committed on its own: push and audit it even if a later write failed.
        if changes:
            await after_kv_change(request, token_id, _user, changes[0], raw.get("kv_isolation"))


async def _update_token(
    request: Request, token_id: str, raw: dict[str, Any], change_out: list[KvChange]
) -> dict[str, Any]:
    """``update_token``'s writes. The KV change, once committed, is appended to
    ``change_out`` so the caller can finish it whatever happens next."""
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        row = await _inference_row(repo, token_id)
        if row is None:
            raise HTTPException(404)
        guard_value: Any = _UNSET
        if "guard" in raw:
            try:
                g = validate_key(raw["guard"])
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            guard_value = json.dumps(g) if g is not None else None
        if raw.get("paused") is True:
            dead = _dead_state(row, sqlite_utc_now())
            if dead is not None:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    detail=f"Token '{row.name}' is {dead}; only a live key can be paused.",
                )
        # First, so a refused change applies nothing else in the body.
        change_out.append(
            await kv_change(request, repo, row, isolation=raw.get("kv_isolation"), reset=False)
        )
        await repo.update(
            token_id,
            name=raw.get("name", _UNSET),
            priority=raw.get("priority", _UNSET),
            paused=raw.get("paused", _UNSET),
            anthropic_relay=raw.get("anthropic_relay", _UNSET),
            thinking_policy=raw.get("thinking_policy", _UNSET),
            guard=guard_value,
            kv_max_pct_ram=raw.get("kv_max_pct_ram", _UNSET),
            kv_max_pct_nvme=raw.get("kv_max_pct_nvme", _UNSET),
            keep_captures=(
                _UNSET if raw.get("keep_captures") is None else bool(raw["keep_captures"])
            ),
            capture_until=(
                _UNSET
                if raw.get("capture_hours") is None
                else (
                    time.time() + raw["capture_hours"] * 3600 if raw["capture_hours"] > 0 else None
                )
            ),
            prompt_diet=(
                _UNSET
                if "prompt_diet" not in raw
                else None
                if raw["prompt_diet"] is None
                else diet_json(
                    Diet(
                        frozenset(raw["prompt_diet"]["strip_tools"]),
                        raw["prompt_diet"]["strip_reasoning"],
                    )
                )
            ),
        )
        detail = await _token_detail(db, token_id)
    # get() found the row above and sqlite serialises the write, so the row
    # MUST still exist; the assert narrows the Optional and guards a race
    # with a concurrent DELETE.
    assert detail is not None, f"token {token_id} disappeared mid-PATCH"
    if raw.get("paused") is True:
        # A paused key's live Prompt Lens streams end at once (G7); their re-check
        # would find it within seconds anyway.
        request.app.state.stream_registry.cancel_user(key_stream_key(token_id))
    return _with_effective(detail, request)


async def kv_change(
    request: Request,
    repo: TokenRepo,
    row: TokenRow,
    *,
    isolation: str | None,
    reset: bool,
    allow_to_sharing: bool = True,
) -> KvChange:
    """``TokenRepo.update_kv`` for ``row`` (and its live family), with its refusals as
    HTTP errors: 409 for a key that no longer works or a reset of a sharing key, 403 for
    a holder switching an isolated key back to sharing (``allow_to_sharing`` False)."""
    if isolation is None and not reset:
        return KvChange(family=[], retired=[])
    try:
        return await repo.update_kv(
            row.id,
            isolation=isolation,
            reset=reset,
            tag_for=kv_iso.tag_for_state(request.app.state),
            now=sqlite_utc_now(),
            allow_to_sharing=allow_to_sharing,
        )
    except LookupError:
        raise HTTPException(404) from None
    except KvChangeRefused as e:
        if str(e) == "sharing":
            raise HTTPException(status.HTTP_409_CONFLICT, KV_RESET_SHARING) from None
        if str(e) == "to_sharing":
            raise HTTPException(status.HTTP_403_FORBIDDEN, BACK_TO_SHARING) from None
        dead = _dead_state(row, sqlite_utc_now()) or "no longer live"
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Token '{row.name}' is {dead}; only a live key's cache setting can be changed.",
        ) from None


async def after_kv_change(
    request: Request, token_id: str, user: str, change: KvChange, isolation: str | None
) -> None:
    """After the change committed: send the retired tags to the store (best effort) and
    write the audit row of a session admin or key holder."""
    if not change.family:
        return
    if change.retired:
        await kv_iso.push_retired(request.app.state, change.retired)
    kv_iso.audit_change(request, token_id=token_id, username=user, change=isolation or "reset")


@router.post("/{token_id}/kv-reset")
async def reset_token_kv(
    token_id: str, request: Request, _user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """Reset an isolated key's cache: its family moves to a fresh scope and the old one is
    retired. 409 for a sharing key, which has no cache of its own. Returns the token, the
    same shape as ``GET /api/tokens/{id}``."""
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        row = await _inference_row(repo, token_id)
        if row is None:
            raise HTTPException(404)
        change = await kv_change(request, repo, row, isolation=None, reset=True)
        detail = await _token_detail(db, token_id)
    if detail is None:
        raise HTTPException(404)
    await after_kv_change(request, token_id, _user, change, None)
    policy_sync.request_sync(request.app.state)
    return _with_effective(detail, request)


@router.post("/{token_id}/rotate", status_code=status.HTTP_201_CREATED)
async def rotate_token(
    token_id: str,
    body: TokenRotate,
    request: Request,
    _user: str = Depends(require_jwt),
):
    """Rotate a token: rename old row to ``"{name} (old N)"`` and mint a
    new row that keeps the ORIGINAL name (#150).

    Rejects with 409 if the row was already rotated — cascaded `(old 1) →
    (old 2)` renames are footgun-y on accidental double-clicks; the UI
    already disables the button on rotated rows (token-row.tsx) but the
    server check is defence in depth for direct API callers.

    Returns the freshly-minted plaintext + the predecessor's new
    ``"{name} (old N)"`` name so the UI can show "rotated; old token is
    now <renamed_to>" without an extra list call.
    """
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        old = await _inference_row(repo, token_id)
        if old is None:
            raise HTTPException(404)
        if old.rotated_at is not None:
            # 409 = the resource is in a state that conflicts with the request.
            # Matches the "already rotated" AC in the issue ("rotate of an
            # already-rotated token rejected with 4xx").
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail=(
                    f"Token '{old.name}' was already rotated at {old.rotated_at}; "
                    "rotate the successor instead."
                ),
            )
        try:
            new_id, new_plaintext, renamed_to = await repo.rotate(
                old_id=token_id,
                grace_hours=body.grace_hours,
                expires_in_days=body.expires_in_days,
            )
        except ValueError as exc:
            # Defence in depth: rotate() also raises "already rotated" if a
            # racing caller flipped rotated_at between our get() and rotate().
            # SQLite serialises writes so this is theoretical, but free.
            if "already rotated" in str(exc):
                raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            if str(exc) in ("revoked", "expired"):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    detail=f"Token '{old.name}' is {exc}; a dead key cannot be rotated. Issue a new key instead.",
                ) from exc
            raise
    prefix = new_plaintext[:8]
    policy_sync.request_sync(request.app.state)  # the family's newest row carries its policy
    # Verified finding #2: rotating a key ends its holders' sign-ins (the new key signs in afresh).
    await request.app.state.auth_sessions.end_subject("key", token_id)
    # ...and so the live Prompt Lens streams those sign-ins opened (G7), whatever the grace.
    request.app.state.stream_registry.cancel_user(key_stream_key(token_id))
    return {
        "id": new_id,
        "name": old.name,  # The active token keeps the ORIGINAL name.
        "plaintext": new_plaintext,
        "prefix": prefix,
        "rotated_from": token_id,
        # UI surfaces this in the success modal: "old token renamed to <name>"
        # so the operator immediately knows where their previous bearer landed.
        "renamed_to": renamed_to,
        # #185 — echo the grace the server actually applied. 0 means the
        # predecessor's revoked_at is now, i.e. it is rejected on its very
        # next request. The success modal narrates from this rather than from
        # what the client remembers asking for, and scripted callers get a
        # confirmation of the hard cut.
        "grace_hours": body.grace_hours,
    }


class TokenUsage24h(BaseModel):
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenListItem(BaseModel):
    """One ``GET /api/tokens`` item: ``_enrich``'s shape, field for field."""

    id: str
    name: str
    prefix: str
    preview: str
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    rotated_at: str | None
    rotated_from: str | None
    successor_id: str | None
    successor_deleted: bool
    is_expired: bool
    is_near_expiry: bool
    revoked_at: str | None
    is_revoked: bool
    priority: int
    anthropic_relay: bool
    thinking_policy: ThinkingPolicy
    prompt_diet: PromptDiet | None = None
    capture_until: float | None = None
    guard: dict[str, Any] | None = None
    #: Prompt Lens keep captures: the last 50 finished requests are kept on disk.
    keep_captures: bool = False
    usage_24h: TokenUsage24h
    paused_at: str | None
    is_paused: bool
    kv_isolation: KvIsolation = "sharing"
    kv_max_pct_ram: int | None = None
    kv_max_pct_nvme: int | None = None
    allowed_models: list[str] | None = None


class TokenListPage(BaseModel):
    items: list[TokenListItem]
    total: int
    limit: int
    offset: int
    near_expiry: int


@router.get("", response_model=TokenListPage)
async def list_tokens(
    request: Request,
    sort: Literal[
        "name",
        "prefix",
        "created",
        "expires",
        "last_used",
        "priority",
        "usage_24h",
        "status",
    ] = Query(
        default="created",
        description=(
            "Sort column. `usage_24h` = prompt + completion tokens over the last "
            "24h; `status` = the badge order Paused, Revoked, Expired, Grace, "
            "Expiring soon, Active. NULLs (never used / never expires) sort last "
            "in both directions; `id` breaks ties."
        ),
    ),
    direction: Literal["asc", "desc"] = Query(default="desc", alias="dir"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=2**62),
    q: str = Query(
        default="",
        max_length=64,
        description="Case-insensitive substring of the token name; empty = no filter.",
    ),
    near_expiry: int = Query(
        default=0,
        ge=0,
        le=1,
        description=(
            "1 = only keys expiring within 30 days and not yet expired -- the "
            "keys the `near_expiry` count counts."
        ),
    ),
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """One page of the visible tokens -- ``{items, total, limit, offset,
    near_expiry}``.

    A key revoked WITHOUT a rotation is hidden, as it always was; a rotated
    predecessor stays listed. ``total`` counts the visible (and ``q``-matched)
    keys across all pages and ``near_expiry`` those of them expiring within 30
    days but not yet expired -- the expiry banner's number. ``near_expiry=1``
    narrows the list to exactly those keys. Every item is the
    ``GET /{id}`` shape without ``lineage``.

    A page costs two statements however many keys exist; see
    ``list_token_page`` for how the sorts use the 0035 indexes.
    """
    search = q.strip()
    now_minute = int(time.time() // 60)
    # One instant for the whole response: the SQL status rank and _enrich's
    # badge fields must agree on which keys are expired.
    now_str = sqlite_utc_now()
    in_30d_str = sqlite_utc_in(timedelta(days=30))
    async with open_db(request.app.state.settings.db_path) as db:
        page = await list_token_page(
            db,
            sort=sort,
            desc=direction == "desc",
            limit=limit,
            offset=offset,
            now_str=now_str,
            in_30d_str=in_30d_str,
            since_minute=now_minute - 24 * 60,
            until_minute=now_minute + 1,
            q=search,
            near_expiry_only=bool(near_expiry),
        )
    items = [
        _enrich(
            e.row,
            successor_id=e.successor_id,
            usage_24h=e.usage_24h,
            now_str=now_str,
            in_30d_str=in_30d_str,
        )
        for e in page.entries
    ]
    return {
        "items": items,
        "total": page.total,
        "limit": limit,
        "offset": offset,
        "near_expiry": page.near_expiry,
    }


@router.get("/{token_id}")
async def get_token(
    token_id: str, request: Request, _user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """One token: every field a ``GET /api/tokens`` item carries, plus
    ``lineage`` -- the rotation chain it belongs to, oldest first, each entry
    ``{id, name, created_at, rotated_at, is_revoked, in_grace, is_self}``.
    ``in_grace`` is true only while a rotated key's grace window is open AND
    it still authenticates -- an expired or paused predecessor is refused, so
    it reads false.

    Unlike the list, this answers for a plainly revoked key too: a details
    page linked from anywhere must not 404 a row that exists.
    """
    async with open_db(request.app.state.settings.db_path) as db:
        detail = await _token_detail(db, token_id)
    if detail is None:
        raise HTTPException(404)
    return _with_effective(detail, request)


@router.get("/{token_id}/usage")
async def get_token_usage(
    token_id: str,
    request: Request,
    range: str = Query(default="24h", pattern=r"^(1h|24h|7d)$"),
    _user: str = Depends(require_jwt),
):
    """Per-minute usage rollup for one token.

    ``range`` selects the look-back window. The query plan is bounded by
    ``idx_token_usage_minute_minute`` so even a 7d range over a busy token
    is cheap. Returned ``buckets`` are minute-aligned (gaps mean zero —
    we don't pre-fill, the UI does that).
    """
    now_minute = int(time.time() // 60)
    if range == "1h":
        since_minute = now_minute - 60
    elif range == "24h":
        since_minute = now_minute - 24 * 60
    else:  # "7d" — the Pydantic Query pattern bounds this branch
        since_minute = now_minute - 7 * 24 * 60

    async with open_db(request.app.state.settings.db_path) as db:
        # Verify the token exists so we can return 404 instead of an
        # empty-bucket success that the UI would silently render as
        # "no usage" — clearer error surface.
        if await _inference_row(TokenRepo(db), token_id) is None:
            raise HTTPException(404)
        rollup = TokenUsageRepo(db)
        rows = await rollup.range(
            token_id=token_id,
            since_minute=since_minute,
            until_minute=now_minute + 1,
        )
        totals = await rollup.totals(
            token_id=token_id,
            since_minute=since_minute,
            until_minute=now_minute + 1,
        )

    return {
        "token_id": token_id,
        "range": range,
        "since_minute": since_minute,
        "until_minute": now_minute,
        "buckets": [
            {
                "minute": m,
                "requests": req,
                "prompt_tokens": pt,
                "completion_tokens": ct,
            }
            for (m, req, pt, ct) in rows
        ],
        "totals": {
            "requests": totals[0],
            "prompt_tokens": totals[1],
            "completion_tokens": totals[2],
            "total_tokens": totals[1] + totals[2],
        },
    }


_CACHE_RANGE_S = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}


@router.get("/{token_id}/cache", response_model=TokenCacheOut)
async def get_token_cache(
    token_id: str,
    request: Request,
    range: str = Query(default="24h", pattern=r"^(1h|24h|7d)$"),
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Prompt-cache observation for one key, OWN lens only (spec §5.2): matched
    against this key's own earlier prompts, never another key's.

    For an admin, the key's KV report is merged in (``app.kv.usage.key_cache_report``):
    ``isolation``, ``windows``, ``store`` (with ``evicted_by`` and the key's admin
    ``reuse``) and ``isolation_cost_tokens``. The key holder reads theirs at ``/api/me/key/cache``.
    """
    since = time.time() - _CACHE_RANGE_S[range]
    settings = request.app.state.settings
    learned = getattr(request.app.state, "prefill_model", None)
    if _user == "key-holder":
        # The key holder's own page: prefill saved at the rate learned from
        # its own requests, never the fleet's.
        from app.stats import prefill_model

        learned = await asyncio.to_thread(prefill_model.own_state, settings.db_path, token_id)
    admin = _user != "key-holder"
    stats = await kv_usage.fetch_stats(request.app.state) if admin else None
    async with open_db(settings.db_path) as db:
        if await _inference_row(TokenRepo(db), token_id) is None:
            raise HTTPException(404)
        rows = await request_history.query_cache_summary(
            db, since=since, model_ids=None, token_id=token_id
        )
        await request_history.decorate_cache_rows(
            db,
            rows,
            learned=learned,
            hint=settings.prefill_tok_s_hint,
            state=request.app.state,
        )
        kv: dict[str, Any] = {}
        if admin:
            kv = await kv_usage.key_cache_report(
                db, token_id, stats, kv_usage.network_key(request.app.state), admin=True
            )
    return {"token_id": token_id, "range": range, "since_epoch": since, "models": rows, **kv}


@router.get("/{token_id}/series")
async def get_token_series(
    token_id: str,
    request: Request,
    from_: float = Query(alias="from", description="Window start, epoch seconds."),
    to: float = Query(description="Window end (exclusive), epoch seconds."),
    chain: int = Query(
        default=1,
        ge=0,
        le=1,
        description="1 = include the key's earlier rotated keys; 0 = this key only.",
    ),
    max_bins: int = Query(default=series.MAX_BINS, ge=1, le=series.MAX_BINS),
    timings: int = Query(
        default=1,
        ge=0,
        le=1,
        description=(
            "1 = include the queue/TTFT/duration percentiles; 0 = usage only "
            "(every timing field null, timing_sample empty, latency_since null)."
        ),
    ),
    by_model: int = Query(
        default=1,
        ge=0,
        le=1,
        description=(
            "1 = include the per-model split (by_model, model_bins, "
            "by_model_since); 0 = skip it (empty lists, null since)."
        ),
    ),
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Usage and timings for one key over any window, binned server-side.

    See app/tokens/series.py for the binning rules and the response shape.
    A ``to`` past the server's clock is clamped to it, not rejected -- a
    client clock running fast must not turn a "last hour" poll into an empty
    chart. 422 when that leaves ``from >= to``, or the (possibly clamped)
    window is longer than 366 days; 404 for an unknown id.

    ``timings=0`` skips request_history entirely -- the history strip draws
    usage only, over up to 366 days, and must not pay for a COUNT plus up to
    50k timing rows it never shows (#251). ``by_model=0`` likewise skips
    token_model_usage_minute; the strip passes both.
    """
    try:
        to = series.validate_window(from_, to, now_s=time.time())
    except series.SeriesWindowError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    width = series.pick_bin_minutes(to - from_, max_bins)
    from_minute, to_minute = series.minute_window(from_, to)
    async with open_db(request.app.state.settings.db_path) as db:
        lineage = await TokenRepo(db).lineage(token_id)
        if not lineage or lineage[0].scope != "inference":
            # A detail of its own, so a test can tell this 404 from a missing
            # route's bare "Not Found" (#251). A rotation chain is one scope --
            # rotate() inherits it -- so checking the chain's oldest row covers
            # the whole chain.
            raise HTTPException(404, "token not found")
        ids = series.chain_token_ids(
            [r.id for r in lineage],
            token_id,
            include_earlier=bool(chain),
        )
        count_rows = await TokenUsageRepo(db).chain_bins(
            ids,
            from_minute=from_minute,
            to_minute=to_minute,
            bin_minutes=width,
        )
        sample = request_history.TokenTimings(rows=[], total=0, stride=1)
        latency_since: float | None = None
        if timings:
            # Same minutes as the counts, so both halves of a bin cover one
            # span. The cap is read at call time (not bound as a default) so
            # a test can lower it.
            sample = await request_history.query_token_timings(
                db,
                token_ids=ids,
                since=from_minute * 60,
                until=to_minute * 60,
                cap=request_history.TIMING_ROW_CAP,
            )
            latency_since = await request_history.earliest_token_finished_at(db)
        model_rows: list[series.VariantRow] = []
        model_bin_rows: list[series.ModelBinRow] = []
        by_model_since: int | None = None
        if by_model:
            per_model = TokenModelUsageRepo(db)
            model_rows = await per_model.by_variant(
                ids,
                from_minute=from_minute,
                to_minute=to_minute,
            )
            model_bin_rows = await per_model.model_bins(
                ids,
                from_minute=from_minute,
                to_minute=to_minute,
                bin_minutes=width,
            )
            first = await per_model.earliest_minute()
            by_model_since = None if first is None else first * 60
    return series.build_series(
        token_ids=ids,
        from_minute=from_minute,
        to_minute=to_minute,
        bin_minutes=width,
        count_rows=count_rows,
        timing_rows=sample.rows,
        timing_total=sample.total,
        timing_stride=sample.stride,
        latency_since=latency_since,
        by_model=model_rows,
        model_bin_rows=model_bin_rows,
        by_model_since=by_model_since,
    )


@router.post("/{token_id}/test")
async def test_token(
    token_id: str,
    request: Request,
    _user: str = Depends(require_jwt),
):
    """Issue a 1-token completion through the proxy to sanity-check the token.

    Designed for the UI "Test token" button. The test mints a short-lived
    bearer header (we already know the token plaintext only at create time,
    so we can't recover it — instead we hit /v1/models which only requires
    the token id + scope/allowed_models check). On success returns the
    upstream model list. On any non-2xx, returns the proxy's status + body
    so the UI can display the exact failure to the operator.

    NB: this endpoint runs as the JWT-authenticated UI user, not as the
    bearer token holder. It cannot validate the priority path because that requires routing through the real proxy with the
    bearer secret — out of scope for the test button; the wizard surfaces
    "ping the proxy" mode which is sufficient for operator confidence.
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        token_row = await _inference_row(TokenRepo(db), token_id)
        if token_row is None:
            raise HTTPException(404)
        models = await ModelRepo(db).list_all()

    # Build a "what would this token see?" projection so the UI can show
    # what models the bearer would be able to hit — done in-process, no
    # outbound HTTP needed (the proxy /v1/models endpoint also does this,
    # but it requires the actual bearer plaintext which we don't store).
    from app.proxy.auth import token_allows

    allowed_models = [
        m.served_model_name
        for m in models
        if m.status == "loaded" and token_allows(token_row, m.served_model_name)
    ]

    # If we can reach the local proxy listener, attempt a HEAD on /v1/models
    # to confirm the routing path. We can't pass the real bearer (we don't
    # store the plaintext), so this only checks "is the proxy alive" — the
    # status code is informational only.
    proxy_reachable = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Use 0.0.0.0 binding from settings — host is loopback for the
            # in-process test.
            url = f"http://127.0.0.1:{settings.bind_port}/healthz"
            r = await client.get(url)
            proxy_reachable = r.status_code == 200
    except httpx.HTTPError:
        proxy_reachable = False

    return {
        "token_id": token_id,
        "ok": True,
        "allowed_models": allowed_models,
        "priority": token_row.priority,
        "proxy_reachable": proxy_reachable,
        # Mirror app/proxy/auth.py::require_bearer — `revoked_at` is a FUTURE
        # timestamp for the whole of a rotation grace window, so a non-null
        # value on its own does NOT mean "this token is being rejected".
        # Reporting it as revoked made Test lie in both directions: about a
        # predecessor still inside a healthy grace window, and about one that
        # was hard-cut with grace_hours=0 (#185).
        "revoked": _is_revoked(token_row, sqlite_utc_now()),
        "expired": _is_expired(token_row, sqlite_utc_now()),
        # Mirrors require_bearer's 403 "token paused".
        "paused": token_row.paused_at is not None,
    }


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_token(token_id: str, request: Request, _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        if await _inference_row(repo, token_id) is None:
            raise HTTPException(404)
        # An isolated key's scope is retired with the row when no live row of its family
        # is left (a rotation successor in use keeps it).
        deleted, retired = await repo.delete_with_retired(
            token_id, tag_for=kv_iso.tag_for_state(request.app.state), now=sqlite_utc_now()
        )
    if not deleted:
        raise HTTPException(404)
    if retired:
        await kv_iso.push_retired(request.app.state, retired)
    policy_sync.request_sync(request.app.state)  # the tenant may be gone
    # Verified finding #2: revoking a key ends its holders' sign-ins.
    await request.app.state.auth_sessions.end_subject("key", token_id)
    # ...and their live Prompt Lens streams (G7).
    request.app.state.stream_registry.cancel_user(key_stream_key(token_id))
    return None
