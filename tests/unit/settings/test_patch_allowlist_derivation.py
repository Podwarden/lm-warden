"""The reviewed artifact for what may be written to a ``ModelRow`` is
``app/models/policy.py``'s ``MODEL_FIELD_POLICY`` — this file pins it, and pins
that ``/api/models/{id}/settings``'s PATCH surface is read off it.

History (the file name is kept so ``git log --follow`` still lands here):
``_PATCHABLE_MODEL_FIELDS`` was hand-maintained and drifted from ``ModelRow``
every time a column was added — #85 and #106 added five patchable-looking fields
that were never wired up. #110 replaced it with ``dataclasses.fields(ModelRow)``
minus a ``_NEVER_PATCH`` blocklist, which cured the drift by inverting the
default: every new column became HTTP-writable unless someone remembered to
exclude it. ``trust_remote_code`` (#256) and ``prior_status`` (#260) arrived on
this endpoint exactly that way.

The policy table keeps #110's drift guard — it must name every ``ModelRow``
column or the import raises, so app startup and this test run both stop — while
removing the writable-by-default inversion: an unclassified column is writable by
nobody. The tests below are therefore in two halves:

  * the **completeness tripwire**: the table covers ``ModelRow`` exactly, and the
    checker's failure message tells the next person what to do;
  * the **spelled-out sets**: the operator-writable columns, the session-to-set
    column, the tri-state columns and the state-machine columns, written out by
    hand so adding a column produces a reviewable diff here rather than a silent
    widening of the PATCH surface.
"""
import dataclasses

import pytest

from app.db.repos.models import ModelRow
from app.models.policy import (
    LOADED_EDITABLE_FIELDS,
    MODEL_FIELD_POLICY,
    OPERATOR_FIELDS,
    RUNTIME_FIELDS,
    SESSION_TO_SET_FIELDS,
    TRISTATE_FIELDS,
    FieldPolicy,
    Writer,
    _check_policy_covers_model_row,
)
from app.settings.routes_api import (
    _MODEL_TRISTATE_FIELDS,
    _PATCHABLE_MODEL_FIELDS,
)

# ---------------------------------------------------------------------------
# The completeness tripwire
# ---------------------------------------------------------------------------


def test_policy_names_every_model_row_column_exactly_once():
    """The table and ``ModelRow`` agree, in both directions.

    ``app/models/policy.py`` asserts this at import time, so in practice a
    violation is an ImportError long before pytest collects anything. This test
    states the invariant where a reader looks for it, and catches the case where
    someone "fixes" the import-time check by weakening it.
    """
    row_columns = {f.name for f in dataclasses.fields(ModelRow)}
    assert set(MODEL_FIELD_POLICY) == row_columns


def test_every_column_has_a_writer():
    assert all(isinstance(p.writer, Writer) for p in MODEL_FIELD_POLICY.values())


def test_unclassified_column_fails_the_check_with_actionable_advice(monkeypatch):
    """Adding a ``ModelRow`` column without classifying it must stop the app.

    This is the whole point of the module: the next person cannot get a writable
    field by forgetting. Simulated by pointing the checker at a row shape that
    has one extra column, rather than by editing ``ModelRow``.
    """
    fake = dataclasses.make_dataclass(
        "FakeModelRow",
        [(f.name, f.type) for f in dataclasses.fields(ModelRow)]
        + [("wants_a_classification", "str | None")],
    )
    monkeypatch.setattr("app.models.policy.ModelRow", fake)
    with pytest.raises(RuntimeError) as exc:
        _check_policy_covers_model_row()
    message = str(exc.value)
    assert "wants_a_classification" in message
    assert "app/models/policy.py" in message
    # The message must say what to do, not just that something is wrong.
    for writer in ("OPERATOR", "SESSION_TO_SET", "RUNTIME", "DB"):
        assert writer in message


def test_renamed_or_dropped_column_fails_the_check(monkeypatch):
    """The other direction — the guard #110's ``_NEVER_PATCH`` drift test had.

    A rename on ``ModelRow`` used to silently promote the old name back into the
    patchable set; here it is a startup failure that names the stale key.
    """
    kept = [f for f in dataclasses.fields(ModelRow) if f.name != "prior_status"]
    fake = dataclasses.make_dataclass(
        "FakeModelRow", [(f.name, f.type) for f in kept]
    )
    monkeypatch.setattr("app.models.policy.ModelRow", fake)
    with pytest.raises(RuntimeError) as exc:
        _check_policy_covers_model_row()
    assert "prior_status" in str(exc.value)
    assert "renamed or dropped" in str(exc.value)


# ---------------------------------------------------------------------------
# The spelled-out sets
# ---------------------------------------------------------------------------

# Every column an operator-facing write path may set, classified. The rule the
# #256 / #260 reviews applied, and the one to apply to the next column added to
# ``ModelRow``:
#
#   a column may be operator-writable only if an admin token can ALREADY set it
#   at `POST /api/models` (so no other write path grants anything the register
#   does not) and it is not control state some part of the runtime owns.
#
# `trust_remote_code` is the one column that fails the first half -- register
# refuses it for an admin token -- so it stays operator-writable (a session may
# set it, and anyone may clear it) via `Writer.SESSION_TO_SET`, rather than being
# locked outright. `prior_status` fails the second half and is `Writer.RUNTIME`.
# Everything else is a plain user setting that register already accepts from a
# token.
#
# Deliberately spelled out rather than derived from the table, so a new ModelRow
# column fails this test and forces that classification instead of arriving in
# silence. The per-column rationale itself lives in `app/models/policy.py` as
# `why=`; this list is the review checkpoint.
_EXPECTED_OPERATOR_WRITABLE = {
    # Identity / source
    "served_model_name", "hf_repo", "hf_revision", "filename",
    "hf_config_repo", "tokenizer_repo", "mmproj_filename",
    # Sizing and placement
    "gpu_indices", "tensor_parallel_size", "parallelism_strategy",
    "max_batch_size", "max_model_len", "gpu_memory_utilization", "dtype",
    "n_gpu_layers",
    # Engine axis
    "backend", "engine_channel", "engine_vllm_version", "engine_image",
    # Free-form engine start-up inputs. Open because `POST /api/models` already
    # takes both from an admin token verbatim, so fencing them on one route
    # would move the same power one route over, not remove it.
    "extra_args", "extra_env",
    # Session-only when set to a truthy value (#256 / C1) -- see
    # tests/unit/models/test_trust_remote_code_session_only.py. Clearing it is
    # open to an admin token, which is why it is not RUNTIME.
    "trust_remote_code",
    # Inert capability metadata the engine never reads (tri-state).
    "supports_tools", "supports_vision", "supports_reasoning",
}

# Lifecycle columns owned by the pull task / load runner / crash path. Mutating
# these out-of-band corrupts the state machine (#11, #29). `prior_status` is the
# newest and the one that mattered: an admin token that could write it armed
# `restart_crashed_models` on a `failed` row, and the sweep re-reads the row and
# calls `start_engine` -- so with `trust_remote_code` patched on in the same
# breath it was remote code execution with no /load call at all (#260).
_EXPECTED_RUNTIME = {
    "status", "pulled_bytes", "pulled_total", "last_error", "prior_status",
}

# Never written by application code. `id` is set once at insert; `updated_at` is
# bumped by the UPDATE statements themselves.
#
# Note what is NOT here: `created_at`. It lives in the SQL schema only and is
# never decoded into `ModelRow`, so it has no place in a `ModelRow` policy. The
# old `_NEVER_PATCH` listed it "should the dataclass ever grow it", which needed
# a `SQL_ONLY_EXCEPTIONS` escape hatch to keep its own drift check honest; the
# completeness tripwire above covers that case without one.
_EXPECTED_DB = {"id", "updated_at"}


def test_operator_writable_surface_is_the_reviewed_set():
    """A new ModelRow column must be classified, not defaulted in.

    If this fails, decide what the new column is and record it in
    ``app/models/policy.py``: a plain user setting register already accepts from
    an admin token (``Writer.OPERATOR``, and add it here), a privilege register
    does not grant a token (``Writer.SESSION_TO_SET``), engine/watchdog control
    state (``Writer.RUNTIME``), or DB bookkeeping (``Writer.DB``).
    """
    assert OPERATOR_FIELDS == _EXPECTED_OPERATOR_WRITABLE


def test_session_to_set_is_exactly_trust_remote_code():
    """The one column a token may clear but not set (#256).

    Widening this set means granting an admin token a new way to reach code
    execution, or narrowing it means a session-only fence disappeared. Either is
    a decision, not a side effect.
    """
    assert SESSION_TO_SET_FIELDS == {"trust_remote_code"}


def test_state_machine_columns_are_runtime():
    assert RUNTIME_FIELDS == _EXPECTED_RUNTIME


def test_db_managed_columns():
    db_fields = {
        name for name, p in MODEL_FIELD_POLICY.items() if p.writer is Writer.DB
    }
    assert db_fields == _EXPECTED_DB


def test_tristate_and_loaded_editable_are_the_capability_flags():
    """Recorded as two properties even though they coincide today.

    ``editable_while_loaded`` is the settings PATCH's carve-out from its
    unload-first 409; ``tristate`` is the true/false/null wire contract. The next
    inert-but-not-boolean column must be able to have one without the other.
    """
    capability_flags = {"supports_tools", "supports_vision", "supports_reasoning"}
    assert LOADED_EDITABLE_FIELDS == capability_flags
    assert TRISTATE_FIELDS == capability_flags


def test_no_state_machine_or_db_column_is_operator_writable():
    """Belt-and-suspenders — catches the worst-case regression even if the
    derivation helpers above are refactored away.
    """
    leaked = OPERATOR_FIELDS & (_EXPECTED_RUNTIME | _EXPECTED_DB)
    assert not leaked, f"lifecycle/DB columns leaked into the writable set: {sorted(leaked)}"


def test_every_why_is_a_single_line():
    """``why=`` is one line of rationale, not a paragraph; the long-form
    reasoning belongs in the grouping comments above each block."""
    multiline = [
        name for name, p in MODEL_FIELD_POLICY.items() if "\n" in p.why
    ]
    assert not multiline


def test_field_policy_is_frozen():
    """The table is read at import and shared; nothing may mutate an entry."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        FieldPolicy(Writer.OPERATOR).writer = Writer.RUNTIME  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The settings PATCH reads its allowlist off the table
# ---------------------------------------------------------------------------


def test_patch_allowlist_is_the_policy_table():
    """``/api/models/{id}/settings`` has no allowlist of its own any more.

    These two identities are the step-0 equivalence proof: both sets were
    byte-identical to what ``_derive_patchable_model_fields`` and the
    hand-written ``_MODEL_TRISTATE_FIELDS`` literal produced before the policy
    table existed, so this commit changed no behaviour.
    """
    assert _PATCHABLE_MODEL_FIELDS == OPERATOR_FIELDS
    assert _MODEL_TRISTATE_FIELDS == TRISTATE_FIELDS


def test_patchable_surface_pins_the_pre_policy_frozenset():
    """The 25 columns ``(ModelRow fields) - _NEVER_PATCH`` produced at
    a5529c3d, pinned literally. A later step of the write-path plan may well
    shrink this set -- when it does, the change must show up as a diff here.
    """
    assert _PATCHABLE_MODEL_FIELDS == frozenset({
        "backend", "dtype", "engine_channel", "engine_image",
        "engine_vllm_version", "extra_args", "extra_env", "filename",
        "gpu_indices", "gpu_memory_utilization", "hf_config_repo", "hf_repo",
        "hf_revision", "max_batch_size", "max_model_len", "mmproj_filename",
        "n_gpu_layers", "parallelism_strategy", "served_model_name",
        "supports_reasoning", "supports_tools", "supports_vision",
        "tensor_parallel_size", "tokenizer_repo", "trust_remote_code",
    })
    assert len(_PATCHABLE_MODEL_FIELDS) == 25


@pytest.mark.parametrize(
    "field",
    [
        # The five fields that #85 and #106 added but the hand-maintained
        # allowlist never picked up. The #110 fix's value to operators is
        # exactly that these are patchable; the policy table must not lose it.
        "filename",
        "parallelism_strategy",
        "max_batch_size",
        "hf_config_repo",
        "tokenizer_repo",
    ],
)
def test_fields_patchable_since_110_stay_patchable(field):
    assert field in _PATCHABLE_MODEL_FIELDS
