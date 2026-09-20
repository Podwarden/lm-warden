"""#266 -- rows poisoned by a settings PATCH door that (before tonight's
fix) accepted an ``extra_env`` key ``POST /api/models`` refuses, or wrote
either ``extra_env``/``extra_args`` in a shape register would have refused.

Nothing swept those rows after the write-path fix landed. The consequence:
``filter_extra_env`` fails closed at LOAD time on a hard-locked key, and
malformed ``extra_env``/``extra_args`` breaks the READ path outright --
``GET /api/models`` / ``GET /api/models/{id}`` 500 because ``ModelOut``
types them ``dict[str, str]`` / ``list[str]``, and a bad row poisons the
whole list response, not just itself. Both failure modes are deterministic
-- there is no code path on which the row loads or lists successfully while
the poison is still there -- so this sweep strips the poison and logs
exactly what it did, the same "log loudly and mutate only when the row is
provably unusable otherwise" shape as ``reconcile_stranded_models``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.boot_reconcile import reconcile_poisoned_extra_fields


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(db_path=tmp_path / "vllm-warden.db")


async def _seed(tmp_path: Path, model_id: str, *, extra_env=None, extra_args=None) -> None:
    async with open_db(tmp_path / "vllm-warden.db") as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id=model_id,
                served_model_name=model_id,
                hf_repo="org/model",
                hf_revision="main",
                gpu_indices=[0],
                tensor_parallel_size=1,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                # ModelRow is a plain dataclass -- nothing stops us seeding
                # exactly the shapes a pre-fix PATCH could have persisted.
                extra_args=extra_args if extra_args is not None else [],
                extra_env=extra_env if extra_env is not None else {},
                status="pulled",
                pulled_bytes=0,
                pulled_total=None,
                last_error=None,
            )
        )


async def _row(tmp_path: Path, model_id: str) -> ModelRow:
    async with open_db(tmp_path / "vllm-warden.db") as db:
        row = await ModelRepo(db).get(model_id)
    assert row is not None
    return row


async def test_hard_locked_key_is_stripped_and_the_removal_is_logged(tmp_path, caplog):
    await _seed(
        tmp_path,
        "poisoned",
        extra_env={"VLLM_LOGGING_LEVEL": "DEBUG", "LD_PRELOAD": "/evil.so"},
    )

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert [m for m, _ in moved] == ["poisoned"]
    row = await _row(tmp_path, "poisoned")
    # The allowlisted key survives; only the hard-locked one is removed.
    assert row.extra_env == {"VLLM_LOGGING_LEVEL": "DEBUG"}

    text = caplog.text
    assert "poisoned" in text
    assert "LD_PRELOAD" in text


async def test_multiple_hard_locked_keys_are_all_named_in_the_log(tmp_path, caplog):
    await _seed(
        tmp_path,
        "poisoned2",
        extra_env={"HF_TOKEN": "hf_x", "PYTHONPATH": "/x", "PATH": "/bin"},
    )

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert [m for m, _ in moved] == ["poisoned2"]
    row = await _row(tmp_path, "poisoned2")
    assert row.extra_env == {}

    for key in ("HF_TOKEN", "PYTHONPATH", "PATH"):
        assert key in caplog.text


async def test_extra_env_wrong_shape_is_reset_and_logged(tmp_path, caplog):
    """A pre-#266 PATCH validated NOTHING about extra_env's shape -- an
    operator (or a scripted client) could have written a bare string, and
    that breaks GET /api/models for every model in the list, not just this
    one (``ModelOut.extra_env: dict[str, str]`` fails response validation)."""
    await _seed(tmp_path, "shapeless", extra_env="not-a-dict")

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert [m for m, _ in moved] == ["shapeless"]
    row = await _row(tmp_path, "shapeless")
    assert row.extra_env == {}
    assert "shapeless" in caplog.text


async def test_extra_args_wrong_shape_is_reset_and_logged(tmp_path, caplog):
    """A string extra_args is the #266 issue's other verification target:
    ``args.py`` does ``list(getattr(model, "extra_args", []) or [])``, which
    on a non-empty string iterates it CHARACTER BY CHARACTER into argv."""
    await _seed(tmp_path, "chars", extra_args="--enforce-eager")

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert [m for m, _ in moved] == ["chars"]
    row = await _row(tmp_path, "chars")
    assert row.extra_args == []
    assert "chars" in caplog.text


async def test_extra_args_with_non_string_elements_is_reset(tmp_path, caplog):
    await _seed(tmp_path, "mixedlist", extra_args=["--foo", 123, None])

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert [m for m, _ in moved] == ["mixedlist"]
    row = await _row(tmp_path, "mixedlist")
    assert row.extra_args == []


async def test_both_columns_poisoned_on_one_row_are_fixed_in_one_pass(tmp_path, caplog):
    await _seed(
        tmp_path,
        "double",
        extra_env={"LD_PRELOAD": "/evil.so"},
        extra_args="--danger",
    )

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert [m for m, _ in moved] == ["double"]
    row = await _row(tmp_path, "double")
    assert row.extra_env == {}
    assert row.extra_args == []


async def test_clean_rows_are_left_untouched_and_silent(tmp_path, caplog):
    await _seed(
        tmp_path,
        "clean1",
        extra_env={"VLLM_LOGGING_LEVEL": "DEBUG"},
        extra_args=["--enforce-eager"],
    )
    await _seed(tmp_path, "clean2")

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert moved == []
    assert caplog.text == ""
    row = await _row(tmp_path, "clean1")
    assert row.extra_env == {"VLLM_LOGGING_LEVEL": "DEBUG"}
    assert row.extra_args == ["--enforce-eager"]


async def test_a_key_not_matching_the_allowlist_but_not_hard_locked_is_left_alone(
    tmp_path, caplog
):
    """Not this sweep's job: an unrecognised-but-not-hard-locked key is
    dropped by ``filter_extra_env`` at LOAD time (silent, INFO-logged, not a
    refusal) -- exactly as it always has been for any operator-typo'd key.
    Only hard-locked keys and wrong shapes make the row unusable, and only
    those are this sweep's business."""
    await _seed(tmp_path, "typo", extra_env={"VLLM_FOO": "1", "NOT_ON_ANY_ALLOWLIST": "x"})

    with caplog.at_level(logging.WARNING, logger="app.runtime.boot_reconcile"):
        moved = await reconcile_poisoned_extra_fields(_settings(tmp_path))

    assert moved == []
    row = await _row(tmp_path, "typo")
    assert row.extra_env == {"VLLM_FOO": "1", "NOT_ON_ANY_ALLOWLIST": "x"}
