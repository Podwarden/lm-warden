"""Runtime + per-model settings API.

GET/PATCH `/api/settings/runtime` is the single read/write surface for the
runtime tunables seeded by migration 0010 (plus the admin_username derived
from the users table and the admin_password sentinel). The PATCH response
classifies each changed key into one of:

  * "none"           — takes effect immediately, no restart needed
  * "model-reload"   — already-loaded models won't pick this up until
                       they're unloaded + reloaded (e.g. hf_token)
  * "warden-restart" — process-level config bound at import time
                       (e.g. session TTLs); operator must restart warden

The classification is returned in `requires_restart_kinds` so the FE can
nudge the operator to take the appropriate follow-up action.

The legacy `POST /api/settings` shape (allowed_gpu_indices + hf_token) is
removed. Phase 1 of the redesign moved the wizard onto `/api/setup`, so
nothing in-tree depended on the old POST contract. HF-token validation
semantics are preserved here: a PATCH that supplies `hf_token` calls
`validate_hf_token` first, and on ValueError returns 422 with the message
(same shape as the prior endpoint).

Admin credentials (`admin_username`, `admin_password`) are routed to the
`users` table — NOT the settings KV — because the login path reads only
from `users`. A PATCH that writes them into settings KV would be a
write-only operation that quietly breaks future password changes AND
leaks the plaintext into the DB. See routing block in `patch_runtime`.

Per-model settings (Task 3.3) live on a sibling router `model_settings_router`
mounted under `/api/models/{model_id}/settings`.
"""
import dataclasses
import json
import re
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth.deps import (
    TRUST_REMOTE_CODE_SET_MESSAGE,
    refuse_session_only,
    require_jwt,
)
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.settings import SettingsRepo
from app.db.repos.setup import SetupRepo
from app.system.hf import validate_hf_token

router = APIRouter(prefix="/api/settings", tags=["settings"])


# (key, kind) where kind is one of "none" | "model-reload" | "warden-restart".
# Drives both PATCH validation (unknown keys → 400) and the
# requires_restart_kinds echo.
RUNTIME_KEYS: dict[str, str] = {
    "admin_username": "none",
    "admin_password": "none",            # writes invalidate sessions but no restart
    "hf_token": "model-reload",
    "default_gpu_indices": "none",
    "default_token_expiration_days": "none",
    "rotation_grace_hours": "none",
    "session_access_ttl_minutes": "warden-restart",
    "session_refresh_ttl_days": "warden-restart",
    "sse_ticket_ttl_seconds": "none",
    "vllm_version": "warden-restart",
    "log_retention_lines": "none",
    # #155 unified-port: public landing-page opt-out. Read on every
    # /_landing request, so a flip takes effect immediately — no restart.
    "landing_page_enabled": "none",
    # #154 settings redesign (subsumes #151): canonical externally-reachable
    # base URL. Read on every snippet render in the FE — no restart needed.
    # Absent row = "use window.location.origin" (FE-side fallback).
    "public_url": "none",
    # Engine watchdog (2026-08-17). Read from the KV on every tick, so a change
    # takes effect on the next pass -- no restart kind.
    "watchdog_enabled": "none",
    "watchdog_restore_on_boot": "none",
    "watchdog_interval_s": "none",
    "watchdog_failure_threshold": "none",
    "watchdog_max_restarts": "none",
}

# Keys that should never be returned to the client in plaintext.
_SECRET_KEYS = frozenset({"hf_token", "admin_password"})

# Sentinel returned in place of secret/credential values.
_MASKED = "***"

# Username pattern — matches auth/routes.py LoginBody.username constraints
# (length 1–64). Restricted character set keeps the value safe to embed in
# JWT subjects, log lines, and SQL identifiers.
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Coercers: each one returns the canonical TEXT representation to persist, or
# raises ValueError with a human-readable reason. Coercers run BEFORE any DB
# write so a single bad value aborts the whole PATCH with 422 (no partial
# writes). Bounds mirror the constraints on the related token-create payload
# (app/tokens/routes_api.py) so persisting a setting can't yield a future
# invalid Pydantic Field.
# ---------------------------------------------------------------------------


def _nonneg_int(v: Any) -> str:
    iv = int(v)  # raises ValueError on non-numeric strings, bools-as-non-ints
    if isinstance(v, bool):  # bool is an int subclass; reject explicitly
        raise ValueError("must be an integer, not a boolean")
    if iv < 0:
        raise ValueError("must be >= 0")
    return str(iv)


def _pos_int(v: Any) -> str:
    if isinstance(v, bool):
        raise ValueError("must be an integer, not a boolean")
    iv = int(v)
    if iv <= 0:
        raise ValueError("must be > 0")
    return str(iv)


def _bounded_int(min_v: int, max_v: int):
    def _check(v: Any) -> str:
        if isinstance(v, bool):
            raise ValueError("must be an integer, not a boolean")
        iv = int(v)
        if iv < min_v or iv > max_v:
            raise ValueError(f"must be between {min_v} and {max_v}")
        return str(iv)

    return _check


def _gpu_list(v: Any) -> str:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError("must be a JSON list of non-negative ints") from exc
    if not isinstance(v, list):
        raise ValueError("must be a JSON list of non-negative ints")
    for i in v:
        # bool is an int subclass; exclude it explicitly.
        if isinstance(i, bool) or not isinstance(i, int) or i < 0:
            raise ValueError("must be a JSON list of non-negative ints")
    return json.dumps(v)


def _nonempty_str(v: Any) -> str:
    if not isinstance(v, str) or v == "":
        raise ValueError("must be a non-empty string")
    return v


# #155 — canonical bool coercer. Accepts real Python bools and the common
# canonical truthy/falsy strings. Stored as 'true' / 'false' (lowercase)
# so the route layer can `raw.strip().lower() in {"true",...}` without
# re-parsing JSON. Free-form strings (e.g. "maybe", "1.5") raise so junk
# values can't land in the settings KV.
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _bool(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        lowered = v.strip().lower()
        if lowered in _TRUE:
            return "true"
        if lowered in _FALSE:
            return "false"
    raise ValueError(
        "must be a boolean (true/false) or one of "
        "'true'/'false'/'yes'/'no'/'on'/'off'/'1'/'0'"
    )


# #154 settings redesign (subsumes #151).
#
# Coerces a `public_url` setting payload. Accepts an absolute http(s) URL
# with a non-empty netloc. The trailing slash is stripped before persist so
# downstream `f"{base}/v1/chat"` concatenation never produces `//v1/chat`.
#
# Rejects:
#   * non-string / empty / whitespace-only values (URL must be explicit)
#   * non-http(s) schemes (e.g. ftp://, file://) — we only embed the URL
#     in user-facing HTTP snippets and the SSE/proxy stack already binds
#     to http(s)
#   * URLs without a netloc (e.g. "http://", "https:///foo") — these
#     parse but are semantically empty
#   * URLs > 2048 bytes — protects log lines and snippet rendering from
#     pathological inputs. Matches the de-facto IE-era limit; nothing
#     legitimate needs more.
#
# What we deliberately DO NOT validate:
#   * Reachability — we don't fetch the URL. Operators may configure the
#     warden's public URL before DNS / cert is live; warden auth doesn't
#     depend on it. This matches `landing_page_enabled`, which also
#     accepts arbitrary values without round-tripping.
#   * Same-origin to the warden — by design, this setting EXISTS for the
#     case where the public URL is different from `window.location.origin`.
def _url(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError("must be a string")
    s = v.strip()
    if not s:
        raise ValueError("must be a non-empty URL")
    if len(s) > 2048:
        raise ValueError("must be <= 2048 characters")
    try:
        parts = urlsplit(s)
    except ValueError as exc:
        raise ValueError(f"is not a parseable URL: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError("must use http:// or https:// scheme")
    if not parts.netloc:
        raise ValueError("must include a host (non-empty netloc)")
    # Persist without trailing slash so concatenated snippets are clean.
    return s.rstrip("/")


# Per-spec, lines 350–363 of docs/superpowers/specs/2026-05-11-vllm-warden-ui-redesign-design.md:
#   default_token_expiration_days: matches TokenCreate Field(ge=0, le=3650)
#   rotation_grace_hours:          matches TokenRotate Field(ge=0, le=720)
#   session_access_ttl_minutes:    minutes, must be positive
#   session_refresh_ttl_days:      days, must be positive
#   sse_ticket_ttl_seconds:        seconds, must be positive
#   log_retention_lines:           lines, must be positive
_COERCERS = {
    "default_gpu_indices": _gpu_list,
    "default_token_expiration_days": _bounded_int(0, 3650),
    "rotation_grace_hours": _bounded_int(0, 720),
    "session_access_ttl_minutes": _pos_int,
    "session_refresh_ttl_days": _pos_int,
    "sse_ticket_ttl_seconds": _pos_int,
    "vllm_version": _nonempty_str,
    "log_retention_lines": _pos_int,
    # #155 unified-port: public landing-page opt-out.
    "landing_page_enabled": _bool,
    # #154 settings redesign (subsumes #151): canonical externally-reachable
    # base URL for user-facing snippets.
    "public_url": _url,
    # Engine watchdog. Bounds are deliberately tight: an interval under 5s would
    # hammer the engine, and a threshold of 1 would restart on a single GC pause.
    "watchdog_enabled": _bool,
    "watchdog_restore_on_boot": _bool,
    "watchdog_interval_s": _bounded_int(5, 3600),
    "watchdog_failure_threshold": _bounded_int(2, 20),
    "watchdog_max_restarts": _bounded_int(0, 20),
    # hf_token, admin_username, admin_password are handled out-of-band below.
}


#: Runtime keys only a signed-in session may change (spec 2026-09-19,
#: decision 3). They are the session credential itself -- the operator's
#: username and password -- and the lifetimes of the session's credentials
#: (access JWT, refresh cookie, SSE ticket). An admin token that could change
#: the password would lock the owner out of the session-only revoke that ends
#: it, so a leaked token could keep itself alive. Every other runtime key is
#: ordinary configuration an admin token may change.
#:
#: `public_url` joined this set after final-review finding I1 (2026-09-19),
#: as defence in depth: it is the host client-facing URLs are built from, and
#: a leaked admin token must not be able to steer it to a lookalike host it
#: controls. (The admin-token reveal screen uses the page origin instead, so
#: no stored setting can redirect a freshly revealed secret.)
SESSION_ONLY_RUNTIME_KEYS: frozenset[str] = frozenset({
    "admin_username",
    "admin_password",
    "session_access_ttl_minutes",
    "session_refresh_ttl_days",
    "sse_ticket_ttl_seconds",
    "public_url",
})


def _refuse_session_only_keys(request: Request, body: dict[str, Any]) -> None:
    """403 ``session_only`` when a non-session credential (an admin token)
    sends any SESSION_ONLY_RUNTIME_KEYS key. Runs before any validation or
    write, so a refused PATCH changes nothing at all. Delegates the
    "is this a session?" check and the 403 shape to app/auth/deps.py::
    refuse_session_only, shared with the trust_remote_code refusals (#256)."""
    keys = sorted(SESSION_ONLY_RUNTIME_KEYS.intersection(body))
    if keys:
        refuse_session_only(
            request,
            message=(
                f"{', '.join(keys)} can only be changed from a signed-in session. "
                "Admin tokens cannot change the operator's credentials or session "
                "lifetimes, so a leaked token cannot lock the owner out of revoking "
                "it. Nothing in this request was applied."
            ),
            keys=keys,
        )


async def _first_admin_id(db) -> int | None:
    """Return the id of the single-admin row, or None if no admin exists yet."""
    cur = await db.execute("SELECT id FROM users ORDER BY id LIMIT 1")
    row = await cur.fetchone()
    return row[0] if row else None


@router.get("/runtime")
async def get_runtime(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Return the full runtime-settings surface.

    Sources:
      * `admin_username` / `admin_password` — first row of the users table
        (single-admin model; password is never returned in plaintext)
      * everything else — `settings` table, falling back to None when a key
        has not yet been written
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        kv = await SettingsRepo(db).get_many(
            [k for k in RUNTIME_KEYS if k not in ("admin_username", "admin_password")]
        )
        # Single-admin model — first row.
        cur = await db.execute(
            "SELECT username FROM users ORDER BY id LIMIT 1"
        )
        admin_row = await cur.fetchone()

    out: dict[str, Any] = {k: kv.get(k) for k in RUNTIME_KEYS}
    out["admin_username"] = admin_row[0] if admin_row else None
    # admin_password — sentinel when an admin exists, else None.
    out["admin_password"] = _MASKED if admin_row else None
    # Mask hf_token if a value was persisted; leave None if absent.
    if out.get("hf_token") is not None:
        out["hf_token"] = _MASKED
    return out


@router.patch("/runtime")
async def patch_runtime(
    body: dict[str, Any],
    request: Request,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Patch a subset of runtime settings.

    Validation order (all checks run before ANY write — no partial writes):
      1. Unknown keys → 400. SESSION_ONLY_RUNTIME_KEYS from an admin token
         → 403 ``session_only``.
      2. `hf_token` empty-string → 422 (clearing is not a supported op).
      3. `hf_token` non-empty → validated against the HF API; ValueError → 422.
      4. `admin_username` → must match `_USERNAME_RE` → 422 on failure.
      5. `admin_password` → must be non-empty string → 422 on failure.
      6. Every other supplied key → run its coercer (type + bounds) → 422 on failure.

    Routing:
      * `admin_username` / `admin_password` → UPDATE users (NOT settings KV).
        Login reads from `users` only, so writing creds into the KV would be
        a no-op for auth AND leak plaintext.
      * Everything else → SettingsRepo.set(key, coerced_value).

    Response: `{"ok": True, "requires_restart_kinds": [...], "requires_restart": [...]}`
    where both fields contain the unique non-"none" kinds in sorted order.
    `requires_restart` is retained for the FE that the plan author wrote
    against; `requires_restart_kinds` is the spec name.
    """
    bad = [k for k in body if k not in RUNTIME_KEYS]
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown keys: {sorted(bad)}")
    # Credentials and session lifetimes are session-only -- refused whole,
    # before anything is validated or written.
    _refuse_session_only_keys(request, body)

    # --- Pre-write validation ------------------------------------------------

    # hf_token: empty-string is a 422 (clear-token affordance is not in spec).
    if "hf_token" in body:
        if not isinstance(body["hf_token"], str) or body["hf_token"] == "":
            raise HTTPException(
                status_code=422, detail="hf_token cannot be empty"
            )
        try:
            await validate_hf_token(body["hf_token"])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # admin_username: pattern check before we open the DB.
    if "admin_username" in body:
        v = body["admin_username"]
        if not isinstance(v, str) or not _USERNAME_RE.fullmatch(v):
            raise HTTPException(
                status_code=422,
                detail=(
                    "admin_username: must be 1–64 chars, "
                    "alphanumeric, underscore, or hyphen"
                ),
            )

    # admin_password: non-empty string. (No max bound here — bcrypt truncates
    # at 72 bytes regardless, and the login Field caps incoming attempts at 256.)
    if "admin_password" in body:
        v = body["admin_password"]
        if not isinstance(v, str) or v == "":
            raise HTTPException(
                status_code=422, detail="admin_password cannot be empty"
            )

    # Coerce every other key. Build the dict of canonical TEXT values to write
    # so the inner DB block can stay short.
    to_write: dict[str, str] = {}
    for k, v in body.items():
        if k in ("admin_username", "admin_password", "hf_token"):
            continue  # handled above / out-of-band
        coercer = _COERCERS.get(k)
        if coercer is None:
            # Defensive: a key in RUNTIME_KEYS but missing from _COERCERS would
            # silently bypass validation. Today that set is empty, but keep the
            # guard so a future RUNTIME_KEYS addition without a coercer fails loudly.
            raise HTTPException(
                status_code=500,
                detail=f"internal: no coercer registered for key {k!r}",
            )
        try:
            to_write[k] = coercer(v)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=f"{k}: {e}") from e

    # hf_token is written as-is (already validated), but only after the rest.
    if "hf_token" in body:
        to_write["hf_token"] = body["hf_token"]

    # --- Writes --------------------------------------------------------------

    kinds: set[str] = set()
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        # Admin creds → users table.
        if "admin_username" in body or "admin_password" in body:
            admin_id = await _first_admin_id(db)
            if admin_id is None:
                # No admin row yet — should never happen in a running system
                # (setup wizard seeds it), but fail cleanly rather than UPDATE
                # 0 rows silently.
                raise HTTPException(
                    status_code=409,
                    detail="no admin user exists yet; complete setup first",
                )
            if "admin_username" in body:
                await db.execute(
                    "UPDATE users SET username = ? WHERE id = ?",
                    (body["admin_username"], admin_id),
                )
            if "admin_password" in body:
                pw_hash = bcrypt.hashpw(
                    body["admin_password"].encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8")
                await db.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (pw_hash, admin_id),
                )
            await db.commit()
            # admin_* are classified "none" in RUNTIME_KEYS — no kind to add.
            # Keep the lookup centralised so a future re-classification is
            # picked up by the loop above.
            for k in ("admin_username", "admin_password"):
                if k in body:
                    kind = RUNTIME_KEYS[k]
                    if kind != "none":
                        kinds.add(kind)

        # Everything else → settings KV.
        repo = SettingsRepo(db)
        for k, stored in to_write.items():
            await repo.set(k, stored)
            kind = RUNTIME_KEYS[k]
            if kind != "none":
                kinds.add(kind)

    out_kinds = sorted(kinds)
    return {
        "ok": True,
        "requires_restart_kinds": out_kinds,
        "requires_restart": out_kinds,
    }


# ---------------------------------------------------------------------------
# Per-model settings (Task 3.3)
# ---------------------------------------------------------------------------

# #110 — Derived PATCH allowlist for /api/models/{id}/settings.
#
# Historically `_PATCHABLE_MODEL_FIELDS` was hand-maintained, which let
# new ModelRow columns (e.g. #85's filename / parallelism_strategy /
# max_batch_size, #106's hf_config_repo / tokenizer_repo) silently fall
# off the patchable surface even though every other layer treats them
# as operator-tunable. The fix is to derive the allowlist from
# `dataclasses.fields(ModelRow)` MINUS an explicit `_NEVER_PATCH`
# blocklist that names lifecycle-owned and DB-managed columns. New
# columns become patchable by default; locking one requires adding it
# to `_NEVER_PATCH` with a comment explaining why.
#
# (The plan §S3 wording says `ModelRow.__fields__` — that's the
# Pydantic accessor; ModelRow is a `@dataclass`, so we use
# `dataclasses.fields` instead. Same semantics for our purpose: the
# canonical field name registry.)
_NEVER_PATCH: frozenset[str] = frozenset({
    # Identifier — opaque, set at insert time.
    "id",
    # Lifecycle columns owned by the pull task / load runner. Mutating
    # these out-of-band corrupts the state machine (#11, #29).
    "status",
    "pulled_bytes",
    "pulled_total",
    "last_error",
    # DB-managed audit columns. `updated_at` is bumped by every UPDATE
    # in this module; `created_at` lives in the SQL schema only (not
    # in ModelRow today) but stays on the blocklist as belt-and-
    # suspenders should it ever be promoted into the dataclass.
    "updated_at",
    "created_at",
    # Watchdog CONTROL state, not a user setting (migration 0024). It records
    # the status a serving engine was interrupted in, and it is the flag
    # app/runtime/watchdog.py::wants_restart keys on -- its only writers are
    # boot reconciliation and the crash path, and the restart sweep clears it.
    # Making it operator-editable was how an admin token could arm the sweep on
    # a `failed` row and have the watchdog call start_engine for it, which (with
    # `trust_remote_code` patched on in the same breath) was remote code
    # execution without ever calling /load. Nothing should PATCH it, session or
    # token: the state machine owns it, exactly like `status`.
    "prior_status",
})


def _derive_patchable_model_fields() -> frozenset[str]:
    """Build the PATCH allowlist as (ModelRow fields) - _NEVER_PATCH.

    Runs at import time. Asserts every name in `_NEVER_PATCH` that is
    NOT one of the documented SQL-only exceptions matches a real
    ModelRow field — so a future rename (e.g. `status` → `lifecycle`)
    on the dataclass fails loudly here instead of silently letting the
    old name through as patchable.
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    # Names allowed in `_NEVER_PATCH` without a backing dataclass field —
    # currently just `created_at`, which lives in SQL only.
    SQL_ONLY_EXCEPTIONS = {"created_at"}
    stale = [
        n for n in _NEVER_PATCH
        if n not in row_fields and n not in SQL_ONLY_EXCEPTIONS
    ]
    if stale:
        raise RuntimeError(
            f"_NEVER_PATCH contains names {stale} that are not ModelRow "
            f"fields and not in SQL_ONLY_EXCEPTIONS={sorted(SQL_ONLY_EXCEPTIONS)}. "
            f"Did a column get renamed? Update _NEVER_PATCH to match."
        )
    return frozenset(row_fields - _NEVER_PATCH)


_PATCHABLE_MODEL_FIELDS: frozenset[str] = _derive_patchable_model_fields()

# ModelRow fields persisted as JSON in the underlying column.
_MODEL_JSON_FIELDS = frozenset({"gpu_indices", "extra_args", "extra_env"})

# Capability flags are TRI-state, not boolean: 1 = yes, 0 = no, NULL = "nobody
# has stated an answer". The third state is load-bearing — app/chat2/catalog.py
# only auto-detects vision from the on-disk HF config while the column is NULL,
# so collapsing NULL into 0 is what silently turned every pasted image into
# "[image omitted]" on a model that could read it perfectly well.
#
# JSON has exactly one way to say "no value", so the wire contract is:
#   * `true` / `false`  -> explicit 1 / 0, the operator's answer wins forever
#   * `null`            -> back to auto (NULL)
#   * key omitted       -> left exactly as it was
# The last of those is already how every other field on this endpoint behaves,
# so there is no extra sentinel to learn. The body is an untyped dict, so the
# value is normalised through `int(bool(v))` — the same coercion
# `trust_remote_code` uses below — rather than being written raw.
_MODEL_TRISTATE_FIELDS = frozenset({
    "supports_tools", "supports_vision", "supports_reasoning",
})


def _coerce_tristate(field: str, v: Any) -> int | None:
    """Validate one capability flag, or raise 400.

    Deliberately NOT `int(bool(v))` (what `trust_remote_code` uses): that reads
    the string `"no"` -- a perfectly plausible thing for a shell script to
    send -- as an explicit YES, the exact opposite of what was asked, and
    silently. Only real JSON booleans, their case-insensitive string spellings
    (form-encoded clients and `curl` recipes), and null get through; anything
    else is a loud 400.
    """
    if v is None:
        return None
    if isinstance(v, bool):  # must precede the int check -- bool IS an int
        return int(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return int(v.strip().lower() == "true")
    raise HTTPException(
        status_code=400,
        detail=f"{field} must be true, false, or null (got {v!r})",
    )


def _coerce_backend(v: Any) -> str | None:
    """Validate models.backend, or raise 400.

    Necessary because the PATCH allowlist is DERIVED from ModelRow's fields, so
    a new column is patchable the moment it exists. Without this, an operator
    could write 'sglang' and the failure would surface from inside
    Supervisor.load as an UnknownBackendError -- after the GPU claim, as an
    opaque last_error, at load time rather than at write time.
    ``registry.is_known`` is the single source of truth, so this can never
    drift from what the build can actually run.

    None is accepted: it is the D6 default (NULL decodes to 'vllm') and is how
    an operator returns a row to the default without naming it. Note the
    explicit ``v and`` -- ``is_known("")`` is True by design (it treats falsy
    as "unset"), and letting an empty string through would write a value that
    is neither NULL nor a backend name.
    """
    from app.runtime.backends import registry

    if v is None:
        return None
    if isinstance(v, str) and v and registry.is_known(v):
        return v
    raise HTTPException(
        status_code=400,
        detail=(
            f"backend must be null or one of {list(registry.available())} "
            f"(got {v!r})"
        ),
    )


model_settings_router = APIRouter(prefix="/api/models", tags=["model-settings"])


@model_settings_router.get("/{model_id}/settings")
async def get_model_settings(
    model_id: str,
    request: Request,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Return the whole persisted ModelRow.

    The `supports_*` capability flags come back RAW (null / 0 / 1) rather than
    coerced to a bool, so a client can tell "the operator said no" from "nobody
    said, so it is auto-detected" — see `_MODEL_TRISTATE_FIELDS`.
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        m = await ModelRepo(db).get(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="model not found")
    return asdict(m)


@model_settings_router.patch("/{model_id}/settings")
async def patch_model_settings(
    model_id: str,
    body: dict[str, Any],
    request: Request,
    _user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Patch a model's persistent settings.

    Refuses to mutate a model that is currently `status == 'loaded'` — that
    column is authoritative state set by the supervisor on transitions. The
    operator must unload the model first, which is a 409.

    One carve-out: a body whose keys are ALL capability flags is allowed
    through on a loaded model. Those columns are inert metadata that the load
    runner and the engine never read (only the chat2 catalog does, per
    request), and the loaded model is precisely the one an operator is looking
    at when they notice vision is set wrong — making them unload it to fix a
    label is backwards. Mixing in any real engine setting puts the whole patch
    back under the guard.

    ``trust_remote_code: true`` is session-only here (#256, C1), for exactly the
    reason it is session-only on register and on load: the row is what a later
    load — or the watchdog's restart sweep — reads, so patching the flag on is
    the same code-execution grant by a different verb. Refused first, before
    validation and before any write, so a refused PATCH changes nothing at all.
    Turning the flag off, and every other setting, stays open to an admin token.
    """
    if body.get("trust_remote_code"):
        # Truthy, not ``is True``: the write below coerces with ``int(bool(v))``,
        # so anything truthy would have persisted a 1.
        refuse_session_only(request, message=TRUST_REMOTE_CODE_SET_MESSAGE)

    bad = [k for k in body if k not in _PATCHABLE_MODEL_FIELDS]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"unknown or non-patchable keys: {sorted(bad)}",
        )
    capabilities_only = bool(body) and set(body) <= _MODEL_TRISTATE_FIELDS

    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        m = await ModelRepo(db).get(model_id)
        if not m:
            raise HTTPException(status_code=404, detail="model not found")
        if m.status == "loaded" and not capabilities_only:
            raise HTTPException(
                status_code=409,
                detail="model must be unloaded before editing settings",
            )

        # Mirror the create-time and load-time allowlist checks. Without this
        # the PATCH surface is the one write path that can persist a GPU index
        # the host does not permit: `_derive_patchable_model_fields` made
        # `gpu_indices` patchable by default, and the row below writes it
        # verbatim. The row then loads nowhere -- the operator sees
        # "gpu_indices [...] not subset of allowed [...]" at load time, far
        # from the edit that caused it, on a page that offers no way to see
        # the allowlist. Same 400 and same wording as POST /api/models so the
        # frontend's existing handling applies unchanged.
        if "gpu_indices" in body:
            requested = body["gpu_indices"]
            if not isinstance(requested, list) or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in requested
            ):
                raise HTTPException(
                    status_code=400, detail="gpu_indices must be a list of integers"
                )
            if not requested:
                raise HTTPException(
                    status_code=400, detail="gpu_indices must name at least one GPU"
                )
            allowed = set((await SetupRepo(db).get()).draft.get("allowed_gpu_indices", []))
            if not set(requested).issubset(allowed):
                bad = sorted(set(requested) - allowed)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"GPU indices {bad} not in allowed_gpu_indices "
                        f"{sorted(allowed)}"
                    ),
                )

        # Build the SET clause from the allowlist intersected with the body.
        # Iterating over `_PATCHABLE_MODEL_FIELDS` (not `body.items()`) makes
        # the identifier source explicit — keys can't slip into the SQL string
        # without passing through the allowlist constant.
        set_parts: list[str] = []
        values: list[Any] = []
        for k in _PATCHABLE_MODEL_FIELDS:
            if k not in body:
                continue
            v = body[k]
            if k in _MODEL_JSON_FIELDS:
                values.append(json.dumps(v))
            elif k in _MODEL_TRISTATE_FIELDS:
                # `null` is a deliberate reset to auto, not a missing value —
                # only an omitted key means "don't touch". Validated (not
                # coerced) so junk is a 400 before anything is written.
                values.append(_coerce_tristate(k, v))
            elif k == "backend":
                values.append(_coerce_backend(v))
            elif k == "trust_remote_code":
                values.append(int(bool(v)))
            else:
                values.append(v)
            set_parts.append(f"{k} = ?")
        if set_parts:
            values.append(model_id)
            await db.execute(
                f"UPDATE models SET {', '.join(set_parts)}, updated_at = datetime('now') "
                f"WHERE id = ?",
                tuple(values),
            )
            await db.commit()

    return {"ok": True}
