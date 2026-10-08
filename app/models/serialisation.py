"""The single place a ``ModelRow`` becomes an API response.

Why this module exists, and why it is not just three more keys in two dicts:

``GET /api/models/{id}`` and ``GET /api/models`` each hand-built a dict literal.
Migration 0027 added ``backend`` and 0028 added ``mmproj_filename`` /
``n_gpu_layers``; ``ModelRow`` and ``_decode_row`` grew all three, the write path
persisted them correctly, and both read paths silently dropped them. The
operator saw the consequence as "every model reports Engine: vllm" on a model
that was demonstrably running ``llama-server`` — the UI's ``backend ?? "vllm"``
fallback firing on a key that was never sent.

That is a drift bug, not a typo, and it recurs on every migration. The remedy is
to make the *whole row* the default and require an explicit decision to withhold
a column:

* ``_serialise`` starts from ``dataclasses.asdict`` and subtracts. A new column
  is therefore published unless someone names it in ``_PRIVATE_FIELDS`` or
  ``_ENGINE_FIELDS``, and ``tests/unit/models/test_model_serialisation.py``
  fails the build if it is in neither.
* ``model_detail`` and ``model_summary`` both call ``_serialise`` and return the
  SAME dict. #237 is why: ``/api/models`` used to hand-maintain a second, leaner
  allowlist (``SUMMARY_FIELDS``) on the theory that the list endpoint is polled
  every 2s and cheapness there is a real constraint. In practice that second
  allowlist is exactly what let ``max_model_len`` / ``gpu_memory_utilization`` /
  ``extra_args`` — and, worse, all three tri-state capability flags — reach
  ``GET /api/models/{id}`` while staying invisible on ``GET /api/models``, so a
  list view could not show engine sizing at all and a write-then-verify against
  the list endpoint looked like the API silently ignored the write. A model row
  is a few dozen scalars; serialising it twice is not the expensive part of a 2s
  poll. Keeping two allowlists in lockstep by hand was. One function, called
  from both routes, makes that drift structurally impossible instead of merely
  fixed.

Sub-project E appends ``actions`` and ``backend_verdicts`` to both responses.
Those are computed, not persisted, so they belong at the call site on top of
what these functions return -- this module deliberately does not grow a notion
of lifecycle legality, which is E's alone.
"""

import dataclasses
from typing import Any

from app.db.repos.models import ModelRow
from app.runtime.backends.registry import DEFAULT_BACKEND

# Folded into the nested ``engine`` object rather than published flat. Kept as a
# set (not just a comment) so the coverage test can account for them.
_ENGINE_FIELDS = frozenset({"engine_channel", "engine_vllm_version", "engine_image"})

# Folded into the nested ``layout_notice`` object (#286): the boot reconcile's
# operator-visible outcome, ``null`` when there is nothing to say.
_LAYOUT_NOTICE_FIELDS = frozenset({"layout_notice_level", "layout_notice"})

# Deliberately withheld from the models API.
#
# ``updated_at`` and ``prior_status`` are bookkeeping for the GC sweep and the
# crash-restart sweep respectively; neither is an operator-facing fact about the
# model.
_PRIVATE_FIELDS = frozenset(
    {
        "updated_at",
        "prior_status",
    }
)

# Capability flags are TRI-state, not boolean: 1 = operator said yes, 0 =
# operator said no, NULL = nobody has said. The third state is load-bearing --
# see the comment on ``ModelRow`` in app/db/repos/models.py and
# app/chat2/catalog.py, which only auto-detects ``supports_vision`` from the
# on-disk HF config while the column is NULL. ``dataclasses.asdict`` hands these
# back as the raw ``int | None`` SQLite gave them (1 / 0 / None); ``_tristate``
# is the one place that turns them into real JSON booleans without collapsing
# the NULL state, so every consumer of these two functions gets the same
# ``true`` / ``false`` / ``null`` contract.
_CAPABILITY_FIELDS = frozenset(
    {
        "supports_tools",
        "supports_vision",
        "supports_reasoning",
    }
)

# Plain stored booleans (0/1 in SQLite) published as real JSON booleans.
_BOOL_FIELDS = frozenset({"dp_affinity_enabled"})


def _tristate(value: int | None) -> bool | None:
    """Map a raw 1/0/NULL capability column onto true/false/null.

    Deliberately not ``bool(value)`` -- that reads NULL as ``False`` and erases
    "nobody has said" into "operator said no", which is precisely the
    distinction ``app/chat2/catalog.py`` depends on to auto-detect vision.
    """
    return None if value is None else bool(value)


def _backend_of(row: ModelRow) -> str:
    """``backend``, never NULL.

    ``_decode_row`` already applies ``DEFAULT_BACKEND`` on the way out of
    SQLite, so in production this is a no-op. It is repeated here because the
    UI's own ``backend ?? "vllm"`` fallback is exactly what made the missing key
    look harmless for two migrations, and the serialiser is the last place that
    can guarantee the client never has to guess. D6: NULL means vLLM.
    """
    return row.backend or DEFAULT_BACKEND


def _serialise(row: ModelRow) -> dict[str, Any]:
    """Everything ``ModelRow`` carries, minus ``_PRIVATE_FIELDS``.

    The engine pin is presented as a nested object (``None`` until a channel is
    pinned) rather than three flat columns -- the shape the frontend already
    reads -- and the three ``supports_*`` columns are coerced through
    ``_tristate`` so the tri-state survives the trip to JSON. ``model_detail``
    and ``model_summary`` are both thin wrappers around this so the two routes
    can never again disagree about what a model row contains -- see the module
    docstring (#237).
    """
    body = {
        k: v
        for k, v in dataclasses.asdict(row).items()
        if k not in _PRIVATE_FIELDS and k not in _ENGINE_FIELDS and k not in _LAYOUT_NOTICE_FIELDS
    }
    body["backend"] = _backend_of(row)
    for name in _CAPABILITY_FIELDS:
        body[name] = _tristate(getattr(row, name))
    for name in _BOOL_FIELDS:
        body[name] = bool(getattr(row, name))
    body["layout_notice"] = (
        {"level": row.layout_notice_level, "message": row.layout_notice}
        if row.layout_notice_level and row.layout_notice
        else None
    )
    body["engine"] = (
        None
        if not row.engine_channel
        else {
            "channel": row.engine_channel,
            "vllm_version": row.engine_vllm_version,
            "image": row.engine_image,
        }
    )
    return body


def model_summary(row: ModelRow) -> dict[str, Any]:
    """One row of ``GET /api/models``.

    Field-for-field identical to ``model_detail`` -- see the module docstring
    for why the two used to diverge and why that was a bug, not a design.
    """
    return _serialise(row)


def model_detail(row: ModelRow) -> dict[str, Any]:
    """The body of ``GET /api/models/{id}``."""
    return _serialise(row)
