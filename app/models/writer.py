"""The single operator-facing writer of a ``ModelRow`` (#262, #265).

Why this module exists, in one sentence: **the row is what the runtime trusts,
and until now nothing in the codebase was responsible for the row.**
``backend.plan(row)`` builds argv and env from it, ``Supervisor.
_resolve_engine_image(row)`` picks the container image, and the watchdog re-reads
it and restarts it with nobody at the keyboard -- while every rule about what may
be *on* a row was attached to a *request*: the full ruleset in ``ModelCreate``, a
thinner copy in ``TemplateCreate``, a hand-written subset in
``patch_model_settings``, and none at all in ``try_stack`` and stress-apply. So
the same hole kept reappearing one door at a time: ``trust_remote_code`` through
three doors (#256), ``extra_env`` through three (#261), ``prior_status`` through
one (#260), and #262's PATCH skipping essentially all of register's validation.

``apply_model_change`` is the choke point. Every operator-facing producer of a
row goes through it, so a new column is governed by two things that are complete
by construction -- ``app/models/policy.py``'s per-column policy table (which
fails at import if a ``ModelRow`` column is unclassified) and
``app/models/schemas.py``'s ``ModelSpec`` (which a test pins against the table's
operator-writable set) -- rather than by whichever route a reviewer happened to
read.

What it does NOT change (be honest about this, API.md already is): ``extra_args``
and ``engine_image`` stay operator-writable, because ``POST /api/models`` already
accepts both from an admin token verbatim, so fencing them here would move that
power one route over rather than remove it. For any principal that can register a
model and start it, both are code execution -- see the ``why=`` on those columns
in the policy table and "treat an admin token as equivalent to code execution in
the warden container" in API.md. This module makes the write path legible and
stops the *next* column from being writable-by-accident; it does not change the
threat model for a leaked admin token.

The state machine is deliberately NOT a caller. ``status``, ``prior_status``,
``pulled_bytes``, ``pulled_total`` and ``last_error`` are ``Writer.RUNTIME`` and
are written only by the named ``ModelRepo`` methods that own them
(``update_status``, ``set_prior_status``, ``update_pull_progress``,
``mark_runtime_dead_on_startup``). The watchdog and boot reconciliation have no
principal, so they cannot call this function at all: ``SYSTEM`` raises.
``tests/unit/models/test_single_row_writer.py`` pins both halves.
"""
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import aiosqlite
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from app.auth.deps import TRUST_REMOTE_CODE_SET_MESSAGE
from app.auth.principal import Principal
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.setup import SetupRepo
from app.models.policy import (
    LOADED_EDITABLE_FIELDS,
    OPERATOR_FIELDS,
    SESSION_TO_SET_FIELDS,
    TRISTATE_FIELDS,
)
from app.models.schemas import ModelSpec


class ModelChangeRefused(HTTPException):
    """A change that was not applied, with the status the caller answers.

    An ``HTTPException`` subclass rather than a bespoke error: five routes share
    the writer, and none of them should be able to turn a refusal into a 500 by
    forgetting a ``try``. The statuses and detail shapes are the ones those
    routes already returned -- 400 for a key or a placement the host does not
    permit, 403 ``session_only``, 409 for unload-first and for a name collision,
    422 for a value that breaks a rule.
    """


#: How each ``SESSION_TO_SET`` column's value is coerced on its way to the row,
#: derived from ``ModelSpec`` so the refusal below cannot drift from the write.
#: Without it the two disagreed: ``refuse_session_to_set`` tested raw Python
#: truthiness while ``ModelSpec`` coerces, so an admin token CLEARING
#: ``trust_remote_code`` with the string ``"false"`` -- which the row would have
#: stored as 0 -- was refused 403 for "setting" it.
_SESSION_TO_SET_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    name: TypeAdapter(ModelSpec.model_fields[name].annotation)
    for name in SESSION_TO_SET_FIELDS
}


def _would_write_truthy(key: str, value: Any) -> bool:
    """Whether ``value`` would land on the row as something truthy.

    Coerced exactly as ``ModelSpec`` will coerce it, because the policy is about
    what the ROW ends up holding, not about the spelling a client used. A value
    the spec cannot coerce at all falls back to raw truthiness and is therefore
    refused: it is about to be a 422 anyway, and guessing in the permissive
    direction is the wrong way to be wrong about a session-only grant.
    """
    try:
        return bool(_SESSION_TO_SET_ADAPTERS[key].validate_python(value))
    except ValidationError:
        return bool(value)


def refuse_session_to_set(principal: Principal, changes: Mapping[str, Any]) -> None:
    """Refuse an admin token that is setting a ``SESSION_TO_SET`` column truthy.

    The one implementation of what used to be four hand-copied
    ``refuse_session_only`` call sites (#256's register body key, its merged
    template value, the template create, and the settings PATCH). "A session may
    set it, anyone may clear it" stops being a comment on a route and becomes the
    definition of the policy table's enum member -- which means the two halves of
    that sentence must agree on what "set" means, hence ``_would_write_truthy``.

    Exported because two callers need it BEFORE they have a row to merge into:
    register and the settings PATCH refuse the body key before opening the
    database, and ``POST /api/models/templates`` writes a template rather than a
    row but is the same grant by a different verb. ``apply_model_change`` runs it
    again on the merged change, which is the check that actually protects the
    row.
    """
    if principal.is_session:
        return
    for key in sorted(SESSION_TO_SET_FIELDS & set(changes)):
        if _would_write_truthy(key, changes[key]):
            raise ModelChangeRefused(
                status_code=403,
                detail={
                    "error_code": "session_only",
                    "message": TRUST_REMOTE_CODE_SET_MESSAGE,
                },
            )


# ---------------------------------------------------------------------------
# Wire coercion
# ---------------------------------------------------------------------------
#
# ``PATCH /api/models/{id}/settings`` takes an untyped JSON object, so three
# columns need a decision about what a client may send that ``ModelSpec`` (which
# validates what a ROW may HOLD) cannot make on its own. They keep the 400s and
# the messages that endpoint already answered with; the coercers are here rather
# than in the route because they are now shared by every caller.


def _coerce_tristate_wire(field: str, v: Any) -> int | None:
    """Validate one capability flag as a CLIENT may send it, or raise 400.

    Deliberately NOT ``int(bool(v))``: that reads the string ``"no"`` -- a
    perfectly plausible thing for a shell script to send -- as an explicit YES,
    the exact opposite of what was asked, and silently. Only real JSON booleans,
    their case-insensitive string spellings (form-encoded clients and ``curl``
    recipes), and null get through.

    Narrower than ``ModelSpec``'s own tri-state validator, which also accepts the
    bare ``0``/``1`` the DATABASE holds: a bare ``1`` from a client is ambiguous
    enough to be worth refusing, a ``1`` read back off a row is not.
    """
    if v is None:
        return None
    if isinstance(v, bool):  # must precede the int check -- bool IS an int
        return int(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return int(v.strip().lower() == "true")
    raise ModelChangeRefused(
        status_code=400,
        detail=f"{field} must be true, false, or null (got {v!r})",
    )


def _coerce_backend_wire(v: Any) -> str | None:
    """Validate ``models.backend``, or raise 400.

    ``registry.is_known`` is the single source of truth, so this can never drift
    from what the build can actually run. None is accepted: it is the D6 default
    (NULL decodes to the registry default) and is how an operator returns a row
    to the default without naming it. Note the explicit ``v and`` --
    ``is_known("")`` is True by design (it treats falsy as "unset"), and letting
    an empty string through would write a value that is neither NULL nor a
    backend name.
    """
    from app.runtime.backends import registry

    if v is None:
        return None
    if isinstance(v, str) and v and registry.is_known(v):
        return v
    raise ModelChangeRefused(
        status_code=400,
        detail=(
            f"backend must be null or one of {list(registry.available())} "
            f"(got {v!r})"
        ),
    )


def _coerce_gpu_indices_wire(v: Any) -> list[int]:
    """Validate the shape of ``gpu_indices``, or raise 400.

    ``ModelSpec`` refuses the same things with a 422; this keeps the 400 and the
    wording the settings PATCH has answered with since #110, because the
    frontend's placement editor handles it.
    """
    if not isinstance(v, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in v
    ):
        raise ModelChangeRefused(
            status_code=400, detail="gpu_indices must be a list of integers"
        )
    if not v:
        raise ModelChangeRefused(
            status_code=400, detail="gpu_indices must name at least one GPU"
        )
    return v


def _coerce_wire(changes: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise the three columns an untyped JSON body can express ambiguously.

    Only for a caller that hands over raw JSON -- today just the settings PATCH,
    which is why ``apply_model_change`` takes ``from_untyped_body``. A caller
    whose values came through a pydantic request model has already resolved the
    ambiguity, and running this over those values would misread it: a capability
    flag arrives as the integer 1 there, which is exactly what this refuses from
    a client.
    """
    out: dict[str, Any] = {}
    for key, value in changes.items():
        if key in TRISTATE_FIELDS:
            out[key] = _coerce_tristate_wire(key, value)
        elif key == "backend":
            out[key] = _coerce_backend_wire(value)
        elif key == "gpu_indices":
            out[key] = _coerce_gpu_indices_wire(value)
        else:
            out[key] = value
    return out


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    """FastAPI's 422 shape, built by hand because this one is raised from a route
    body rather than by the body parser. ``ctx`` is dropped: it can carry the
    original exception object, which is not JSON-serialisable."""
    return [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False)
    ]


# ---------------------------------------------------------------------------
# The choke point
# ---------------------------------------------------------------------------


async def _check_model_change(
    db: aiosqlite.Connection,
    *,
    principal: Principal,
    current: ModelRow | None,
    changes: Mapping[str, Any],
    from_untyped_body: bool,
    assume_unloaded: bool,
) -> tuple[ModelSpec, dict[str, Any]]:
    """Every refusal ``apply_model_change`` can raise, and none of its writes.

    Returns the validated merged row and the coerced change. Read-only: it opens
    no transaction and touches nothing, which is what lets a caller run it as a
    dry run *before* doing something it cannot undo (see
    :func:`dry_run_model_change`). ``apply_model_change`` is this function plus
    the write, so the two can never come to different conclusions.
    """
    if principal.kind == "system":
        raise RuntimeError(
            "the SYSTEM principal may not write operator columns: the state "
            "machine owns status / prior_status / pulled_* / last_error and "
            "writes them through the named ModelRepo methods. A caller with no "
            "principal must not be able to reach an operator field."
        )

    # 1. principal
    refuse_session_to_set(principal, changes)

    # 2. key policy
    unwritable = sorted(set(changes) - OPERATOR_FIELDS)
    if unwritable:
        raise ModelChangeRefused(
            status_code=400,
            detail=f"unknown or non-patchable keys: {unwritable}",
        )

    # 3. loaded guard
    if (
        current is not None
        and current.status == "loaded"
        and not assume_unloaded
        and not (changes and set(changes) <= LOADED_EDITABLE_FIELDS)
    ):
        raise ModelChangeRefused(
            status_code=409,
            detail="model must be unloaded before editing settings",
        )

    # 4. wire coercion
    coerced = _coerce_wire(changes) if from_untyped_body else dict(changes)

    # 5. merge and validate
    base: dict[str, Any] = {}
    if current is not None:
        row_values = asdict(current)
        base = {k: row_values[k] for k in OPERATOR_FIELDS}
    try:
        spec = ModelSpec.model_validate({**base, **coerced})
    except ValidationError as exc:
        raise ModelChangeRefused(
            status_code=422, detail=_validation_detail(exc)
        ) from exc

    # 6. cross-row checks
    if "gpu_indices" in coerced:
        allowed = set((await SetupRepo(db).get()).draft.get("allowed_gpu_indices", []))
        if not set(spec.gpu_indices).issubset(allowed):
            bad = sorted(set(spec.gpu_indices) - allowed)
            raise ModelChangeRefused(
                status_code=400,
                detail=(
                    f"GPU indices {bad} not in allowed_gpu_indices "
                    f"{sorted(allowed)}"
                ),
            )
    if "served_model_name" in coerced:
        clash = await ModelRepo(db).get_by_served_name(spec.served_model_name)
        if clash is not None and (current is None or clash.id != current.id):
            raise ModelChangeRefused(
                status_code=409,
                detail=f"served_model_name '{spec.served_model_name}' already exists",
            )

    return spec, coerced


async def dry_run_model_change(
    db: aiosqlite.Connection,
    *,
    principal: Principal,
    current: ModelRow | None,
    changes: Mapping[str, Any],
    from_untyped_body: bool = False,
    assume_unloaded: bool = False,
) -> None:
    """Raise whatever ``apply_model_change`` would raise, and write nothing.

    For a caller whose write is preceded by a step it cannot take back. **Never
    put a destructive step before a validation that can fail for reasons
    unrelated to the request.** ``apply_model_change`` validates the whole merged
    row, so it can refuse over a column the caller never touched -- an
    ``extra_env`` key written by a pre-#261 PATCH, say, which loads and serves
    perfectly well because ``filter_extra_env`` drops it at launch. A caller that
    has already torn an engine down by then turns that stale column into an
    outage: `POST /api/models/{id}/stress/apply` did exactly that, and the
    watchdog does not recover a row that is `pulled` with no `prior_status`,
    because that is what an operator unloading on purpose looks like.

    ``assume_unloaded=True`` skips the loaded guard, for a caller that is about
    to unload the model itself. It is the one check that is legitimately false
    now and true by the time the real write runs; every other refusal this
    raises would have been raised then too.

    This is advisory, not a lock: the row can still change between the dry run
    and the write. The caller keeps whatever it does on a late refusal -- the dry
    run shrinks the window to a race, it does not remove it.
    """
    await _check_model_change(
        db,
        principal=principal,
        current=current,
        changes=changes,
        from_untyped_body=from_untyped_body,
        assume_unloaded=assume_unloaded,
    )


async def apply_model_change(
    db: aiosqlite.Connection,
    *,
    principal: Principal,
    current: ModelRow | None,
    changes: Mapping[str, Any],
    model_id: str | None = None,
    from_untyped_body: bool = False,
) -> ModelRow:
    """Apply ``changes`` to a model row, or refuse without writing anything.

    ``current=None`` is a register (``model_id`` is then required); otherwise an
    update. The steps, in this order and for these reasons:

    1. **principal** -- a ``SESSION_TO_SET`` column being set truthy by an admin
       token is 403 ``session_only``. First, ahead of the key policy, because a
       body that also names an unwritable column must still get the security
       answer: #256's exploit body was ``{"trust_remote_code": true,
       "prior_status": "loaded"}`` and a 400 about ``prior_status`` would tell a
       token holder the flag itself was fine.
    2. **key policy** -- a key that is not operator-writable (``RUNTIME``, ``DB``
       or unknown) is 400, consulting ``MODEL_FIELD_POLICY``.
    3. **loaded guard** -- a change to a ``status == 'loaded'`` row is 409 unless
       every key is ``editable_while_loaded``. Before validation, so "unload
       first" is the instruction an operator gets rather than a detail about a
       value they cannot apply yet anyway.
    4. **wire coercion** -- the untyped-body normalisations above (400), for a
       caller that passes ``from_untyped_body`` (the settings PATCH). Every
       other caller arrives through a pydantic request model, which has already
       resolved those ambiguities.
    5. **merge and validate** -- the change is merged into the current row and
       the RESULT is validated by ``ModelSpec`` (422). Merge-then-validate is the
       whole point: the cross-field checks (``tensor_parallel_size`` against
       ``len(gpu_indices)``, ``extra_env`` against the row's backend) are
       properties of a row, and a partial body cannot express them.
    6. **cross-row checks** -- ``gpu_indices`` must be a subset of setup's
       ``allowed_gpu_indices`` (400), and ``served_model_name`` must be unique
       (409; the column is ``UNIQUE`` in SQL, so without this the settings PATCH
       answered a collision with an ``IntegrityError`` and a 500).
    7. **write** -- insert, or update exactly the named columns.

    Nothing is written before step 7, so a refusal at any step changes nothing.
    Returns the row as stored.

    Steps 1-6 are :func:`_check_model_change`, which
    :func:`dry_run_model_change` exposes on its own: a caller that must disrupt
    something before it can write (stress-apply unloads the engine first) runs
    them BEFORE the disruption, because step 5 validates the whole merged row and
    can refuse over a column the request never touched.
    """
    spec, coerced = await _check_model_change(
        db,
        principal=principal,
        current=current,
        changes=changes,
        from_untyped_body=from_untyped_body,
        assume_unloaded=False,
    )
    repo = ModelRepo(db)

    # 7. write
    written = {key: getattr(spec, key) for key in coerced}
    if current is None:
        if not model_id:
            raise ValueError("model_id is required to register a row")
        row = ModelRow(
            id=model_id,
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
            **{key: getattr(spec, key) for key in OPERATOR_FIELDS},
        )
        await repo.insert(row)
        return row

    await repo.update_fields(current.id, written)
    updated = await repo.get(current.id)
    assert updated is not None  # the row was read moments ago in the same txn
    return updated
