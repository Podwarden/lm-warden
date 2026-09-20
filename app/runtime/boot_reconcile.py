"""Boot reconciliation for rows stranded in a transient status (#236).

WHY THIS EXISTS
---------------
``loading``, ``unloading`` and ``pulling`` all describe work owned by an
in-process object: the supervisor's engine handle, or the pull task's
asyncio task. Neither survives a restart of the warden process, so after a
restart a row in one of those statuses describes something that no longer
exists anywhere.

That is not merely cosmetic — every route out of those statuses is closed:

    POST /api/models/{id}/load    -> 409 (only 'pulled'/'failed' may load)
    POST /api/models/{id}/unload  -> 409 (only 'loaded'/'failed' may unload)

Observed live on a client install on 2026-09-03: ``docker compose up -d``
recreated the ``api`` container mid-load, the GPUs came back free (the
engine subprocess died with the container) and the row read ``loading``
with a null ``last_error`` across two further restarts. The only escape was
``UPDATE models SET status='pulled'`` by hand, which an operator without a
shell on the container does not have.

So on boot we demote every such row to a status an operator can act on:
``pulled`` when the weights are in the HF cache, ``registered`` when they
are not, with ``last_error`` saying why the status moved.

WHAT IS DELIBERATELY LEFT ALONE
-------------------------------
1. Any model the supervisor actually holds a handle for. At boot there is
   never one — the supervisor is freshly constructed and nothing has been
   spawned yet — but the check is the honest predicate for "is this load
   real?", and it keeps the function safe to call from anywhere later.

2. A ``loading`` row carrying a was-serving ``prior_status``: that is the
   watchdog restoring an engine that WAS serving when it died
   (``app/runtime/watchdog.py::_restart``). Demoting it to ``pulled``
   would quietly disable the automatic restore that
   ``restore_after_warden_restart`` exists for — prod stayed down for
   11.5h on 2026-08-18 without it. Those rows fall through to
   ``ModelRepo.mark_runtime_dead_on_startup`` in the same boot, which
   marks them ``failed`` + ``prior_status`` so the watchdog picks them up.

3. Terminal rows (``loaded``, ``failed``, ``pulled``, ``registered``).
   ``loaded`` is ``mark_runtime_dead_on_startup``'s business, not ours:
   it must become ``failed`` WITH a ``prior_status`` so the model is
   restored automatically, and stealing it here would break that.

ORDERING: this runs BEFORE ``mark_runtime_dead_on_startup`` in the
lifespan. After that call every transient row is already ``failed`` and
there would be nothing left for us to see.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.runtime.backends.paths import _snapshot_dir
from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

logger = logging.getLogger(__name__)

# Statuses whose truth lives in this process's memory, not on disk.
TRANSIENT_STATUSES = ("loading", "unloading", "pulling")

# ``prior_status`` values that mean "this model was SERVING" — the flag
# watchdog.wants_restart keys on. A 'loading' row carrying one is a restore
# in flight, not an interrupted first load.
_WAS_SERVING = ("loaded", "loading")


def weights_present(hf_cache_dir: Path | str, hf_repo: str) -> bool:
    """True when ``hf_repo`` looks like a COMPLETE download in the HF cache.

    Reuses the resolver the launch path uses (``backends.paths``) so that
    "we think the weights are there" cannot drift from "the engine can
    find them" — including its search of both ``<root>`` and
    ``<root>/hub``.

    A snapshot directory alone is not enough: an interrupted ``pulling``
    row leaves one behind with ``*.incomplete`` blobs in it, and calling
    that ``pulled`` would offer the operator a Load button that fails.
    An empty snapshot dir is treated the same way.
    """
    try:
        snap = _snapshot_dir(Path(hf_cache_dir), hf_repo)
        if snap is None:
            return False
        if not any(snap.iterdir()):
            return False
        blobs = snap.parent.parent / "blobs"
        if blobs.is_dir() and any(blobs.glob("*.incomplete")):
            return False
    except OSError:
        # An unreadable cache is not evidence that the weights are there.
        logger.warning("boot reconcile: cache probe failed for %s", hf_repo,
                       exc_info=True)
        return False
    return True


def _is_held(supervisor: Any, model_id: str) -> bool:
    """Does the supervisor actually hold a live engine for this model?

    Defensive ``getattr``: callers pass ``None`` at the very start of boot
    (nothing can be in flight yet) and tests pass stand-ins.
    """
    if supervisor is None:
        return False
    is_running = getattr(supervisor, "is_running", None)
    if is_running is not None and is_running(model_id):
        return True
    get_state = getattr(supervisor, "get_state", None)
    return get_state is not None and get_state(model_id) is not None


async def reconcile_stranded_models(
    settings: Any, supervisor: Any = None
) -> list[tuple[str, str]]:
    """Demote transient rows with no backing process. Returns [(id, new)].

    See the module docstring for the full rationale and the two
    deliberate exemptions.
    """
    async with open_db(settings.db_path) as db:
        repo = ModelRepo(db)
        rows = await repo.list_all()
        moved: list[tuple[str, str]] = []
        for row in rows:
            if row.status not in TRANSIENT_STATUSES:
                continue
            if _is_held(supervisor, row.id):
                continue
            if row.status == "loading" and (row.prior_status or "") in _WAS_SERVING:
                # Watchdog restore of a model that was serving — leave the
                # was-serving path to mark_runtime_dead_on_startup. Same set
                # watchdog.wants_restart keys on, deliberately.
                continue
            new_status = (
                "pulled"
                if weights_present(settings.hf_cache_dir, row.hf_repo)
                else "registered"
            )
            logger.warning(
                "boot reconcile: %s was stranded in '%s' with no backing "
                "process — demoting to '%s'",
                row.id,
                row.status,
                new_status,
            )
            await repo.update_status(
                row.id,
                new_status,
                last_error=(
                    f"recovered from an interrupted {row.status} after a "
                    f"restart: no engine was running for this model"
                ),
            )
            if row.status == "pulling":
                # Same reasoning as mark_runtime_dead_on_startup: the pull
                # does not resume, it re-fetches, so stale progress on a
                # task that no longer exists would lie to the UI.
                await repo.update_pull_progress(row.id, 0, 0)
            moved.append((row.id, new_status))
    return moved


# ---------------------------------------------------------------------------
# #266 — rows poisoned before tonight's extra_env write-path fix
# ---------------------------------------------------------------------------
#
# Until tonight, ``PATCH /api/models/{id}/settings`` validated NOTHING about
# ``extra_env``: no per-backend key allowlist, no hard-locked-key refusal, not
# even a check that the value was a JSON object of strings. ``POST
# /api/models`` (``ModelCreate``) always enforced the full shape through
# pydantic, so every row register ever wrote was clean — the PATCH door was
# the only way a bad value could reach the table. That door is closed now
# (``validate_extra_env_for_backend`` runs on both write paths, see
# ``app/models/schemas.py``), but nothing sweeps rows a pre-fix PATCH already
# wrote.
#
# A row is in exactly one of two poisoned shapes:
#
#   1. ``extra_env`` is a well-formed ``dict[str, str]`` that carries one of
#      ``HARD_LOCKED_ENV_KEYS`` (LD_PRELOAD, PYTHONPATH, HF_TOKEN, PATH, ...).
#      ``filter_extra_env`` checks the hard-locked set FIRST, unconditionally,
#      before anything else -- so this row refuses to load with a
#      ``ValueError`` EVERY time, on EVERY backend, regardless of what else is
#      on the row. There is no code path on which it loads with the key still
#      there.
#
#   2. Either column is not the shape register would ever have accepted at
#      all -- ``extra_env`` not a string-to-string object, or ``extra_args``
#      not a list of strings (a PATCH body of ``{"extra_args": "--foo"}`` was
#      accepted verbatim: the PATCH route treats both columns as opaque JSON,
#      ``json.dumps(v)`` on whatever ``v`` is). This is WORSE than case 1 and
#      more urgent than the load-time failure the issue opens with:
#      ``ModelOut`` types the two columns ``dict[str, str]`` / ``list[str]``,
#      so FastAPI's response validation raises on ANY row shaped wrong —
#      and because ``GET /api/models`` returns every row in one response, one
#      poisoned row 500s the *entire model list*, not just its own detail
#      page. The dashboard goes blank for every model on the install, not
#      just the bad one. It is also broken at load time in its own right:
#      ``app/runtime/backends/vllm/args.py`` does
#      ``list(getattr(model, "extra_args", []) or [])``, which on a non-empty
#      STRING iterates it character by character into argv, and on some other
#      JSON scalar (e.g. an int) raises an uncaught ``TypeError`` instead of
#      ``filter_extra_env``'s clean ``ValueError``.
#
# Both shapes are unusable on every code path that reads them — never merely
# "an operator might disagree with this value" — which is the bar #266's
# ruling sets for mutating instead of only logging. So this strips rather
# than just reporting: case 1 removes only the hard-locked key(s), keeping
# every allowlisted key the operator actually meant to set; case 2 resets the
# column to its empty default (``{}`` / ``[]``), which is exactly what a
# fresh register with the field omitted would have written. Every strip is
# WARNING-logged with the model id and precisely what was removed/reset, so
# an operator reading the boot log — before this process ever accepts an HTTP
# request, hence before any load attempt or list poll can hit the failure
# this explains — knows which model was affected and why, without needing DB
# access to find out.
#
# Deliberately NOT folded into ``reconcile_stranded_models``: that function's
# whole predicate is "does a transient status describe a live process", and
# runs against ``TRANSIENT_STATUSES`` only. Poisoning here is independent of
# status — a 'pulled' or 'failed' row can carry it exactly as well as a
# 'registered' one — so this sweeps every row, unconditionally, once per
# boot.


def _is_str_str_dict(value: object) -> bool:
    """True iff ``value`` is a ``dict[str, str]`` — the wire contract
    ``ModelOut.extra_env`` and every write-path validator assume."""
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _is_str_list(value: object) -> bool:
    """True iff ``value`` is a ``list[str]`` — ``ModelOut.extra_args``'s
    wire contract, and what ``args.py`` assumes it can append to argv."""
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


async def reconcile_poisoned_extra_fields(settings: Any) -> list[tuple[str, list[str]]]:
    """Strip extra_env/extra_args poison a pre-#266 settings PATCH could
    write. Returns ``[(model_id, [what changed, ...])]``.

    See the module comment above this function for the two poisoned shapes,
    why each is unusable on every code path (not just an operator judgment
    call), and why that clears the "mutate, don't just log" bar #266 set.
    Runs against every row regardless of status — unlike
    ``reconcile_stranded_models``, this poisoning has nothing to do with
    whether a process is or was live.
    """
    async with open_db(settings.db_path) as db:
        repo = ModelRepo(db)
        rows = await repo.list_all()
        moved: list[tuple[str, list[str]]] = []
        for row in rows:
            changes: list[str] = []

            new_env = row.extra_env
            if not _is_str_str_dict(row.extra_env):
                changes.append(
                    f"extra_env was not an object of string to string "
                    f"(got {type(row.extra_env).__name__}: {row.extra_env!r}) "
                    f"-- reset to {{}}"
                )
                new_env = {}
            else:
                locked = sorted(k for k in row.extra_env if k in HARD_LOCKED_ENV_KEYS)
                if locked:
                    new_env = {
                        k: v for k, v in row.extra_env.items()
                        if k not in HARD_LOCKED_ENV_KEYS
                    }
                    changes.append(
                        f"extra_env carried hard-locked key(s) {locked} -- removed "
                        f"(filter_extra_env refuses to load ANY model with one of "
                        f"these set, on every backend, every time)"
                    )

            new_args = row.extra_args
            if not _is_str_list(row.extra_args):
                changes.append(
                    f"extra_args was not a list of strings "
                    f"(got {type(row.extra_args).__name__}: {row.extra_args!r}) "
                    f"-- reset to []"
                )
                new_args = []

            if not changes:
                continue

            logger.warning(
                "boot reconcile: model '%s' carries extra_env/extra_args a "
                "pre-#266-fix settings PATCH could write but POST /api/models "
                "would have refused, and it breaks the model on every code "
                "path (load and, for a wrong-shaped column, the models list "
                "itself) -- cleaning it up: %s",
                row.id,
                "; ".join(changes),
            )
            await repo.update_extra_fields(row.id, extra_env=new_env, extra_args=new_args)
            moved.append((row.id, changes))
    return moved
