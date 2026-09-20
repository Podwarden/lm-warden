"""``apply_model_change`` -- the one way an operator-facing path writes a row.

The write path had six producers of a ``ModelRow`` (#262, #265): register, the
settings PATCH, the template merge inside register, ``try_stack``, stress-apply,
and the state machine. One carried the full ruleset and the others carried
whatever a reviewer remembered to copy, which is why the same hole kept
reappearing one door at a time -- ``trust_remote_code`` through three doors
(#256), ``extra_env`` through three (#261), ``prior_status`` through one (#260).

This module tests the choke point directly, at the level where its ORDER is
visible: a refusal at any step must leave the row untouched, and which refusal
fires first is a contract, not an accident. The HTTP-level consequences are in
``tests/unit/settings/test_model_settings_validation.py`` and each caller's own
file.
"""
import asyncio
import dataclasses

import pytest
from fastapi import HTTPException

from app.auth.principal import SYSTEM, Principal
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.models.writer import (
    ModelChangeRefused,
    apply_model_change,
    dry_run_model_change,
)
from tests.conftest import seed_admin_user

SESSION = Principal("session", id="admin")
TOKEN = Principal("admin_token", id="tok1")


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="m1",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="pulled",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


@pytest.fixture
def db_path(tmp_data_dir, migrated_db_template):
    import shutil

    p = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, p)
    seed_admin_user(p, allowed_gpu_indices=[0, 1])
    return str(p)


def _seed(db_path, row: ModelRow) -> None:
    async def _go():
        async with open_db(db_path) as db:
            await ModelRepo(db).insert(row)

    asyncio.run(_go())


def _apply(db_path, *, principal=SESSION, current=None, changes=None, model_id=None,
           from_untyped_body=False):
    async def _go():
        async with open_db(db_path) as db:
            return await apply_model_change(
                db,
                principal=principal,
                current=current,
                changes=changes or {},
                model_id=model_id,
                from_untyped_body=from_untyped_body,
            )

    return asyncio.run(_go())


def _read(db_path, model_id="m1") -> ModelRow | None:
    async def _go():
        async with open_db(db_path) as db:
            return await ModelRepo(db).get(model_id)

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# The happy paths
# ---------------------------------------------------------------------------


def test_a_register_inserts_the_validated_row(db_path):
    row = _apply(db_path, changes={
        "served_model_name": "fresh", "hf_repo": "org/repo", "gpu_indices": [1, 0],
    }, model_id="new1")
    assert row.id == "new1"
    assert row.status == "registered"
    # ModelSpec canonicalises, so the stored list is sorted and tp is derived.
    assert row.gpu_indices == [0, 1]
    assert row.tensor_parallel_size == 2
    assert _read(db_path, "new1").served_model_name == "fresh"


def test_an_update_writes_only_the_changed_columns(db_path):
    _seed(db_path, _row())
    _apply(db_path, current=_row(), changes={"max_model_len": 4096})
    after = _read(db_path)
    assert after.max_model_len == 4096
    assert after.dtype == "auto" and after.gpu_indices == [0]


# ---------------------------------------------------------------------------
# Step order. Each of these sends a body that trips TWO rules; which answer
# comes back is the contract.
# ---------------------------------------------------------------------------


def test_a_session_only_column_is_refused_before_the_key_policy(db_path):
    """#256's C1 body -- ``trust_remote_code`` plus the watchdog's restart flag.

    The 403 must win over the 400 for ``prior_status``: the security refusal is
    the answer an operator needs, and the 400 would tell a token holder that the
    flag itself was acceptable.
    """
    _seed(db_path, _row(status="failed"))
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, principal=TOKEN, current=_row(status="failed"),
               changes={"trust_remote_code": True, "prior_status": "loaded"})
    assert exc.value.status_code == 403
    assert exc.value.detail["error_code"] == "session_only"
    assert _read(db_path).trust_remote_code is False


def test_clearing_a_session_only_column_is_open_to_a_token(db_path):
    _seed(db_path, _row(trust_remote_code=True))
    _apply(db_path, principal=TOKEN, current=_row(trust_remote_code=True),
           changes={"trust_remote_code": False})
    assert _read(db_path).trust_remote_code is False


def test_a_session_may_set_a_session_only_column(db_path):
    _seed(db_path, _row())
    _apply(db_path, current=_row(), changes={"trust_remote_code": True})
    assert _read(db_path).trust_remote_code is True


@pytest.mark.parametrize("key", ["status", "prior_status", "pulled_bytes",
                                 "last_error", "id", "updated_at", "nonsense"])
def test_a_column_no_operator_may_write_is_refused(db_path, key):
    _seed(db_path, _row())
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=_row(), changes={key: "loaded"})
    assert exc.value.status_code == 400
    assert key in str(exc.value.detail)


def test_a_loaded_row_refuses_anything_but_the_capability_flags(db_path):
    _seed(db_path, _row(status="loaded"))
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=_row(status="loaded"), changes={"max_model_len": 4096})
    assert exc.value.status_code == 409
    assert _read(db_path).max_model_len == 2048


def test_a_loaded_row_accepts_a_capabilities_only_change(db_path):
    _seed(db_path, _row(status="loaded"))
    _apply(db_path, current=_row(status="loaded"), changes={"supports_vision": True})
    assert _read(db_path).supports_vision == 1


def test_the_loaded_guard_runs_before_validation(db_path):
    """A loaded row with an invalid change gets the 409, not the 422: unload
    first is the instruction that matters."""
    _seed(db_path, _row(status="loaded"))
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=_row(status="loaded"), changes={"max_model_len": 0})
    assert exc.value.status_code == 409


def test_validation_runs_on_the_merged_row_not_the_change(db_path):
    """The cross-field check is the reason merge-then-validate exists: changing
    ``gpu_indices`` alone contradicts a ``tensor_parallel_size`` nothing in the
    body mentions."""
    _seed(db_path, _row())
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=_row(), changes={"gpu_indices": [0, 1]})
    assert exc.value.status_code == 422
    assert "tensor_parallel_size" in str(exc.value.detail)
    assert _read(db_path).gpu_indices == [0]
    # ...and stating both is accepted.
    _apply(db_path, current=_row(), changes={"gpu_indices": [0, 1],
                                             "tensor_parallel_size": 2})
    assert _read(db_path).gpu_indices == [0, 1]


def test_a_colliding_served_model_name_is_a_409(db_path):
    _seed(db_path, _row())
    _seed(db_path, _row(id="m2", served_model_name="m2"))
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=_row(id="m2", served_model_name="m2"),
               changes={"served_model_name": "m1"})
    assert exc.value.status_code == 409
    assert "already exists" in str(exc.value.detail)
    assert _read(db_path, "m2").served_model_name == "m2"


def test_renaming_a_row_to_its_own_name_is_not_a_collision(db_path):
    _seed(db_path, _row())
    _apply(db_path, current=_row(), changes={"served_model_name": "m1"})
    assert _read(db_path).served_model_name == "m1"


def test_gpu_indices_must_be_a_subset_of_the_setups_allowlist(db_path):
    _seed(db_path, _row())
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=_row(),
               changes={"gpu_indices": [2, 3], "tensor_parallel_size": 2})
    assert exc.value.status_code == 400
    assert "not in allowed_gpu_indices" in str(exc.value.detail)
    assert _read(db_path).gpu_indices == [0]


# ---------------------------------------------------------------------------
# The principal
# ---------------------------------------------------------------------------


def test_the_system_principal_cannot_reach_the_writer(db_path):
    """The watchdog, boot reconciliation and the pull task have no principal at
    all. They own the RUNTIME columns through named repo methods; letting them
    near an operator column would make "no principal" a way to write one.

    A RuntimeError rather than an HTTP refusal on purpose: there is no operator
    to answer, so this is a programming error, not a request.
    """
    _seed(db_path, _row())
    with pytest.raises(RuntimeError, match="SYSTEM"):
        _apply(db_path, principal=SYSTEM, current=_row(),
               changes={"max_model_len": 4096})
    assert _read(db_path).max_model_len == 2048


def test_a_refusal_is_an_http_exception_so_every_route_answers_alike(db_path):
    """``ModelChangeRefused`` IS an ``HTTPException``: five routes share the
    writer, and none of them should be able to turn one of its refusals into a
    500 by forgetting to catch it."""
    assert issubclass(ModelChangeRefused, HTTPException)


# ---------------------------------------------------------------------------
# Rows that predate the rules
# ---------------------------------------------------------------------------


def test_a_row_that_predates_the_rules_can_still_be_repaired(db_path):
    """Validation is a write-path rule, so a row written before it existed keeps
    reading (``GET`` never validates) -- but a change to it is refused until the
    offending column is fixed, which the same PATCH can do.
    """
    legacy = _row(extra_env={"FOO_BAR": "1"})
    _seed(db_path, legacy)
    assert _read(db_path).extra_env == {"FOO_BAR": "1"}

    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, current=legacy, changes={"max_model_len": 4096})
    assert exc.value.status_code == 422
    assert "allowlist" in str(exc.value.detail)

    _apply(db_path, current=legacy, changes={"extra_env": {}, "max_model_len": 4096})
    after = _read(db_path)
    assert after.extra_env == {} and after.max_model_len == 4096


def test_nothing_is_written_when_the_change_is_empty(db_path):
    _seed(db_path, _row())
    before = dataclasses.asdict(_read(db_path))
    _apply(db_path, current=_row(), changes={})
    assert dataclasses.asdict(_read(db_path)) == before


# ---------------------------------------------------------------------------
# The dry run: every refusal, none of the writes
#
# For a caller whose write is preceded by a step it cannot take back. The rule
# it exists to enforce: never put a destructive step before a validation that
# can fail for reasons unrelated to the request. `apply_model_change` validates
# the WHOLE merged row, so it can refuse over a column the request never
# touched -- which is how `POST /{id}/stress/apply` came to unload a serving
# model and then answer 422 about a stale `extra_env` key, leaving it down.
# ---------------------------------------------------------------------------


def _dry_run(db_path, *, principal=SESSION, current=None, changes=None, **kw):
    async def _go():
        async with open_db(db_path) as db:
            return await dry_run_model_change(
                db, principal=principal, current=current, changes=changes or {}, **kw
            )

    return asyncio.run(_go())


@pytest.mark.parametrize("changes,status", [
    ({"max_model_len": 0}, 422),
    ({"extra_env": {"FOO_BAR": "1"}}, 422),
    ({"gpu_indices": [0, 1]}, 422),
    ({"prior_status": "loaded"}, 400),
    ({"gpu_indices": [7, 8], "tensor_parallel_size": 2}, 400),
    ({"served_model_name": "taken"}, 409),
])
def test_the_dry_run_raises_what_the_write_would(db_path, changes, status):
    _seed(db_path, _row())
    _seed(db_path, _row(id="m2", served_model_name="taken"))

    with pytest.raises(ModelChangeRefused) as dry:
        _dry_run(db_path, current=_row(), changes=changes)
    with pytest.raises(ModelChangeRefused) as wet:
        _apply(db_path, current=_row(), changes=changes)
    assert dry.value.status_code == wet.value.status_code == status
    assert str(dry.value.detail) == str(wet.value.detail)


def test_the_dry_run_writes_nothing_when_it_passes(db_path):
    _seed(db_path, _row())
    before = dataclasses.asdict(_read(db_path))
    _dry_run(db_path, current=_row(), changes={"max_model_len": 4096})
    assert dataclasses.asdict(_read(db_path)) == before


def test_the_dry_run_refuses_the_system_principal_too(db_path):
    _seed(db_path, _row())
    with pytest.raises(RuntimeError, match="SYSTEM"):
        _dry_run(db_path, principal=SYSTEM, current=_row(),
                 changes={"max_model_len": 4096})


def test_assume_unloaded_skips_only_the_loaded_guard(db_path):
    """For a caller that is about to unload the model itself -- the one check
    that is legitimately false now and true by the time the write runs."""
    _seed(db_path, _row(status="loaded"))
    loaded = _row(status="loaded")

    with pytest.raises(ModelChangeRefused) as guarded:
        _dry_run(db_path, current=loaded, changes={"max_model_len": 4096})
    assert guarded.value.status_code == 409

    # Skipped...
    _dry_run(db_path, current=loaded, changes={"max_model_len": 4096},
             assume_unloaded=True)
    # ...and nothing else is.
    with pytest.raises(ModelChangeRefused) as still:
        _dry_run(db_path, current=loaded, changes={"max_model_len": 0},
                 assume_unloaded=True)
    assert still.value.status_code == 422


# ---------------------------------------------------------------------------
# refuse_session_to_set agrees with the write about what "set" means
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [False, "false", "False", "no", 0, 0.0, "0"])
def test_a_token_may_clear_a_session_only_column_however_it_spells_false(
    db_path, value
):
    """"A session may set it, anyone may clear it" -- and whether a value counts
    as clearing is decided by what the ROW will hold, not by the spelling.

    The refusal used to test raw Python truthiness while ``ModelSpec`` coerces,
    so clearing the flag with the string ``"false"`` -- which the row would have
    stored as 0 -- was refused 403 for "setting" it.
    """
    _seed(db_path, _row(trust_remote_code=True))
    _apply(db_path, principal=TOKEN, current=_row(trust_remote_code=True),
           changes={"trust_remote_code": value}, from_untyped_body=True)
    assert _read(db_path).trust_remote_code is False


@pytest.mark.parametrize("value", [True, "true", "True", 1, "yes"])
def test_a_token_still_cannot_set_it_however_it_spells_true(db_path, value):
    _seed(db_path, _row())
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, principal=TOKEN, current=_row(),
               changes={"trust_remote_code": value}, from_untyped_body=True)
    assert exc.value.status_code == 403
    assert _read(db_path).trust_remote_code is False


def test_a_value_the_spec_cannot_coerce_is_refused_rather_than_guessed(db_path):
    """Fail-safe: a value that is about to be a 422 anyway must not be read as
    "clearing" on the way past a session-only fence."""
    _seed(db_path, _row())
    with pytest.raises(ModelChangeRefused) as exc:
        _apply(db_path, principal=TOKEN, current=_row(),
               changes={"trust_remote_code": "banana"}, from_untyped_body=True)
    assert exc.value.status_code == 403
