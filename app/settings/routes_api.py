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
import json
import re
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth.deps import refuse_session_only, require_jwt
from app.auth.principal import principal_of
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.settings import SettingsRepo
from app.models.policy import OPERATOR_FIELDS, TRISTATE_FIELDS
from app.models.writer import apply_model_change, refuse_session_to_set
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

# #110, restructured by the model-write-path step 0. The PATCH allowlist for
# /api/models/{id}/settings is now READ OFF the per-column policy table in
# app/models/policy.py rather than derived here.
#
# History, because the shape matters: `_PATCHABLE_MODEL_FIELDS` started
# hand-maintained and drifted from `ModelRow` every time a column was added
# (#85's filename / parallelism_strategy / max_batch_size and #106's
# hf_config_repo / tokenizer_repo were never wired up). #110 replaced it with
# `dataclasses.fields(ModelRow)` MINUS a `_NEVER_PATCH` blocklist, which fixed
# the rot by inverting the default: new columns became patchable unless someone
# remembered to exclude them. That default is how `trust_remote_code` (#256) and
# `prior_status` (#260) arrived on this surface.
#
# `MODEL_FIELD_POLICY` keeps #110's drift guard -- it must name every ModelRow
# column or the import fails -- without the writable-by-default inversion: an
# unclassified column is a hard startup failure, not a patchable field. The
# per-column reasoning, including why `extra_args` / `engine_image` stay open and
# why `prior_status` is locked to the state machine, lives there as `why=`.
_PATCHABLE_MODEL_FIELDS: frozenset[str] = OPERATOR_FIELDS

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
# so there is no extra sentinel to learn.
#
# Which columns those are is a property of the column, so it is read off the
# policy table's `tristate=True` entries. The coercion itself now lives in
# app/models/writer.py, which every write path shares.
_MODEL_TRISTATE_FIELDS = TRISTATE_FIELDS


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

    Every key is optional; an omitted key is left exactly as it was. The body is
    merged into the current row and the RESULT is validated as a whole row by
    `app/models/schemas.py::ModelSpec`, so this endpoint applies the same rules
    `POST /api/models` does -- including the cross-field ones a partial body
    cannot express, like `tensor_parallel_size` matching `len(gpu_indices)` and
    `extra_env` matching the row's backend.

    That is #262's fix. This route used to carry a hand-written subset of
    register's validation: three coercers and a `gpu_indices` check, with
    `json.dumps(v)` for the JSON columns and the raw JSON value for everything
    else. A string `extra_args` therefore round-tripped as a string and
    `args += extra_args` appended it to argv one CHARACTER at a time; slug
    patterns, length bounds and numeric bounds were not applied at all; and a
    `served_model_name` that collided with another row -- the column is UNIQUE
    in SQL -- reached SQLite as an IntegrityError and came back as a 500.

    Answers:

    * 400 -- a key no operator may write (a `Writer.RUNTIME` or `Writer.DB`
      column, or an unknown one), or one of the three untyped-body shapes this
      endpoint has always answered 400 for (`gpu_indices`, `backend`, a
      capability flag), or a GPU the host does not permit;
    * 403 `session_only` -- an admin token setting `trust_remote_code` true;
    * 404 -- no such model;
    * 409 -- the model is loaded (unload first), or the new
      `served_model_name` is taken;
    * 422 -- a value that breaks a rule of the merged row.

    Refuses to mutate a model that is currently `status == 'loaded'` — that
    column is authoritative state set by the supervisor on transitions. The
    operator must unload the model first, which is a 409. One carve-out: a body
    whose keys are ALL capability flags is allowed through on a loaded model.
    Those columns are inert metadata that the load runner and the engine never
    read (only the chat2 catalog does, per request), and the loaded model is
    precisely the one an operator is looking at when they notice vision is set
    wrong — making them unload it to fix a label is backwards. Mixing in any
    real engine setting puts the whole patch back under the guard.

    ``trust_remote_code: true`` is session-only here (#256, C1), for exactly the
    reason it is session-only on register and on load: patching the flag on
    persists the same grant by a different verb. (#264: no engine ever reads
    this column, so "the row is what a later load reads" was not the mechanism;
    what read it was the warden's own proxy tokenizer, and that is gone too.
    The refusal stays -- an admin token must not be able to write the grant.)
    Refused first, before the database is even opened, so a refused PATCH
    changes nothing at all -- not even a 404 lookup. Turning the flag off, and
    every other setting, stays open to an admin token.

    Nothing is written unless every check passes: `apply_model_change` refuses
    before it writes, never part-way through.
    """
    principal = principal_of(request)
    # Before the DB is touched, so the answer to an admin token is the refusal
    # rather than a 404 for a model it may not know exists. The writer runs the
    # same check again on the merged change; this one is about ordering.
    refuse_session_to_set(principal, body)

    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        m = await ModelRepo(db).get(model_id)
        if not m:
            raise HTTPException(status_code=404, detail="model not found")
        await apply_model_change(
            db,
            principal=principal,
            current=m,
            changes=body,
            # The body is raw JSON, so the three columns whose wire form is
            # ambiguous (`gpu_indices`, `backend`, a capability flag) go through
            # the writer's coercers and keep this endpoint's existing 400s.
            from_untyped_body=True,
        )

    return {"ok": True}
