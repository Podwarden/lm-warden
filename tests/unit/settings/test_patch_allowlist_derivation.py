"""Tests for the derived PATCH allowlist on ``/api/models/{id}/settings``
(#110).

The hand-maintained ``_PATCHABLE_MODEL_FIELDS`` set drifted from
``ModelRow`` every time a new column was added (#85 / #106 added five
new patchable-looking fields that were never wired up). The mitigation
is to derive the allowlist from ``ModelRow``'s dataclass fields minus
an explicit ``_NEVER_PATCH`` blocklist for the lifecycle-owned
columns; the blocklist is validated at import time so a rename of any
blocklisted field on ``ModelRow`` fails the test suite (and app
startup) rather than silently letting the field through as patchable.
"""
import dataclasses

import pytest

from app.db.repos.models import ModelRow
from app.settings.routes_api import (
    _NEVER_PATCH,
    _PATCHABLE_MODEL_FIELDS,
)


def test_never_patch_contains_lifecycle_owned_fields():
    """The blocklist must include every lifecycle column. If a field
    moves out of this set the PATCH endpoint would start accepting it,
    which is exactly the failure mode #110 guards against.
    """
    expected = {
        "id", "status", "pulled_bytes", "pulled_total", "last_error",
        # C1 (2026-09-19 follow-up review): the watchdog's restart flag. An
        # admin token that could write it armed restart_crashed_models on a
        # `failed` row, and the sweep re-reads the row and calls start_engine --
        # so with `trust_remote_code` patched on in the same breath it was
        # remote code execution with no /load call.
        "prior_status",
    }
    assert expected.issubset(_NEVER_PATCH)


def test_never_patch_names_exist_on_modelrow_dataclass():
    """Drift guard — every name in ``_NEVER_PATCH`` MUST correspond to a
    real ``ModelRow`` field (or be a known SQL-only column kept here
    for safety, like ``created_at``). A rename of a blocklisted field
    on ``ModelRow`` without updating ``_NEVER_PATCH`` would silently
    promote it back into the patchable set; this test catches that.
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    # ``created_at`` is the documented exception — it lives in the SQL
    # schema only, never decoded into ``ModelRow``, and stays on the
    # blocklist as a belt-and-suspenders measure should the dataclass
    # ever grow it.
    SQL_ONLY_EXCEPTIONS = {"created_at"}
    bad = [n for n in _NEVER_PATCH if n not in row_fields and n not in SQL_ONLY_EXCEPTIONS]
    assert not bad, (
        f"_NEVER_PATCH names {bad} are not ModelRow fields and not in "
        f"SQL_ONLY_EXCEPTIONS — did a column get renamed?"
    )


def test_patchable_fields_are_derived_minus_blocklist():
    """The allowlist is exactly (ModelRow fields) - _NEVER_PATCH. If a
    new column is added to ModelRow it becomes patchable by default;
    operators wanting to lock it MUST add it to _NEVER_PATCH (with a
    comment explaining why).
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    expected = row_fields - _NEVER_PATCH
    assert _PATCHABLE_MODEL_FIELDS == expected


def test_no_lifecycle_field_in_patchable_set():
    """Belt-and-suspenders — the lifecycle-owned columns MUST NOT be in
    the patchable set. Catches the worst-case regression even if the
    derivation logic itself gets refactored.
    """
    lifecycle = {
        "id", "status", "pulled_bytes", "pulled_total", "last_error", "updated_at",
        "prior_status",
    }
    leaked = _PATCHABLE_MODEL_FIELDS & lifecycle
    assert not leaked, f"lifecycle fields leaked into patch allowlist: {sorted(leaked)}"


# Every key this PATCH accepts, classified. The rule the review applied, and
# the one to apply to the next column added to ``ModelRow``:
#
#   a key may be patchable only if an admin token can ALREADY set it at
#   `POST /api/models` (so the PATCH grants nothing the register does not) and
#   it is not control state some part of the runtime owns.
#
# `trust_remote_code` is the one field that fails the first half -- register
# refuses it for an admin token -- so it stays patchable (a session may set it,
# and anyone may clear it) but the route refuses a truthy value from a token,
# rather than being blocklisted outright. `prior_status` fails the second half
# and is blocklisted. Everything else is a plain user setting that register
# already accepts from a token.
#
# Deliberately spelled out rather than derived, so a new ModelRow column fails
# this test and forces that classification instead of becoming patchable in
# silence.
_EXPECTED_PATCHABLE = {
    # Identity / source
    "served_model_name", "hf_repo", "hf_revision", "filename",
    "hf_config_repo", "tokenizer_repo", "mmproj_filename",
    # Sizing and placement
    "gpu_indices", "tensor_parallel_size", "parallelism_strategy",
    "max_batch_size", "max_model_len", "gpu_memory_utilization", "dtype",
    "n_gpu_layers",
    # Engine axis
    "backend", "engine_channel", "engine_vllm_version", "engine_image",
    # Free-form engine start-up inputs. Patchable because `POST /api/models`
    # already takes both from an admin token verbatim, so blocking them here
    # would move the same power one route over, not remove it.
    "extra_args", "extra_env",
    # Session-only when set to a truthy value (#256 / C1) -- see
    # tests/unit/models/test_trust_remote_code_session_only.py. Clearing it is
    # open to an admin token, which is why it is not on the blocklist.
    "trust_remote_code",
    # Inert capability metadata the engine never reads (tri-state).
    "supports_tools", "supports_vision", "supports_reasoning",
}


def test_the_patchable_surface_is_the_reviewed_set():
    """A new ModelRow column must be classified, not defaulted in.

    The derivation (#110) makes new columns patchable automatically, which is
    the right default for ergonomics and the wrong one for security review --
    C1 was `trust_remote_code` and `prior_status` arriving that way. This test
    is the checkpoint: if it fails, decide whether the new column is a plain
    user setting (add it here), engine/watchdog control state (add it to
    ``_NEVER_PATCH``), or a privilege register does not grant a token (refuse it
    in ``patch_model_settings`` like `trust_remote_code`).
    """
    assert _PATCHABLE_MODEL_FIELDS == _EXPECTED_PATCHABLE


@pytest.mark.parametrize(
    "field",
    [
        # The five fields that #85 and #106 added but the hand-maintained
        # allowlist never picked up. The fix's value to operators is
        # exactly that these become patchable.
        "filename",
        "parallelism_strategy",
        "max_batch_size",
        "hf_config_repo",
        "tokenizer_repo",
    ],
)
def test_newly_patchable_fields_are_in_allowlist(field):
    assert field in _PATCHABLE_MODEL_FIELDS, (
        f"{field} should be patchable post-#110 derivation"
    )
