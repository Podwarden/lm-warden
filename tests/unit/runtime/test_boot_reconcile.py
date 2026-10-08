"""#236 — a warden restart must never leave a row in a transient status.

The client install of 2026-09-03: ``docker compose up -d`` recreated the api
container mid-load, the engine subprocess died with it, and the row read
``loading`` forever. ``load`` 409s from ``loading``, ``unload`` 409s from
``loading``, so the only escape was hand-editing SQLite.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.boot_reconcile import reconcile_stranded_models


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        db_path=tmp_path / "vllm-warden.db",
        hf_cache_dir=tmp_path / "hf-cache",
    )


def _plant_weights(tmp_path: Path, repo: str) -> None:
    """Write the HF cache layout the launch path resolves against."""
    slug = "models--" + repo.replace("/", "--")
    snap = tmp_path / "hf-cache" / slug / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")


async def _seed(tmp_path: Path, **rows: str) -> None:
    """Insert one model per ``id=status`` kwarg."""
    async with open_db(tmp_path / "vllm-warden.db") as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        for model_id, status in rows.items():
            await repo.insert(
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
                    extra_args=[],
                    extra_env={},
                    status=status,
                    pulled_bytes=0,
                    pulled_total=None,
                    last_error=None,
                )
            )


async def _status(tmp_path: Path, model_id: str) -> ModelRow:
    async with open_db(tmp_path / "vllm-warden.db") as db:
        row = await ModelRepo(db).get(model_id)
    assert row is not None
    return row


class _FakeSupervisor:
    """Stands in for the real thing: holds engines for the given ids."""

    def __init__(self, *held: str) -> None:
        self._held = set(held)

    def is_running(self, model_id: str) -> bool:
        return model_id in self._held

    def get_state(self, model_id: str):  # noqa: ANN201 — mirrors the real API
        return "LOADING" if model_id in self._held else None


async def test_stranded_loading_with_weights_becomes_pulled(tmp_path):
    """The exact client symptom: 'loading' with no engine and the weights
    already on disk. The operator must get a model they can simply load."""
    await _seed(tmp_path, stranded="loading")
    _plant_weights(tmp_path, "org/model")

    moved = await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor())

    assert moved == [("stranded", "pulled")]
    row = await _status(tmp_path, "stranded")
    assert row.status == "pulled"
    assert "interrupted loading" in (row.last_error or "")


async def test_stranded_loading_without_weights_becomes_registered(tmp_path):
    """No weights in the cache — 'pulled' would offer a Load button that
    cannot work, so the row goes back to 'registered' (pull first)."""
    await _seed(tmp_path, stranded="loading")

    moved = await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor())

    assert moved == [("stranded", "registered")]
    assert (await _status(tmp_path, "stranded")).status == "registered"


async def test_unloading_and_pulling_are_reconciled_too(tmp_path):
    """Both other transient statuses describe in-process work that a restart
    destroyed, and both dead-end the same way."""
    await _seed(tmp_path, half_unloaded="unloading", half_pulled="pulling")
    _plant_weights(tmp_path, "org/model")

    await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor())

    assert (await _status(tmp_path, "half_unloaded")).status == "pulled"
    # This pull left a complete snapshot behind (nothing incomplete in the
    # cache), so 'pulled' is the truthful demotion. The incomplete case is
    # covered by the next test.
    assert (await _status(tmp_path, "half_pulled")).status == "pulled"


async def test_interrupted_pull_with_incomplete_blobs_is_not_called_pulled(tmp_path):
    """huggingface_hub leaves ``*.incomplete`` blobs behind when a download
    is killed. A snapshot dir alone must not be read as 'weights present'."""
    await _seed(tmp_path, half_pulled="pulling")
    _plant_weights(tmp_path, "org/model")
    blobs = tmp_path / "hf-cache" / "models--org--model" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc123.incomplete").write_bytes(b"partial")

    await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor())

    row = await _status(tmp_path, "half_pulled")
    assert row.status == "registered"
    # Progress counters for a task that no longer exists would lie to the UI.
    assert row.pulled_bytes == 0
    assert row.pulled_total == 0


async def test_terminal_rows_are_left_untouched(tmp_path):
    """'loaded' belongs to mark_runtime_dead_on_startup (it must become
    'failed' WITH a prior_status so the watchdog restores it), and 'failed'
    is already an operator-actionable status."""
    await _seed(
        tmp_path,
        serving="loaded",
        broken="failed",
        idle="pulled",
        fresh="registered",
    )
    _plant_weights(tmp_path, "org/model")

    moved = await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor())

    assert moved == []
    assert (await _status(tmp_path, "serving")).status == "loaded"
    assert (await _status(tmp_path, "broken")).status == "failed"
    assert (await _status(tmp_path, "idle")).status == "pulled"
    assert (await _status(tmp_path, "fresh")).status == "registered"


async def test_a_load_the_supervisor_actually_holds_is_never_clobbered(tmp_path):
    """Boot means nothing is in flight, but the predicate is the honest one:
    if the supervisor holds an engine for this model, the load is real."""
    await _seed(tmp_path, live="loading")
    _plant_weights(tmp_path, "org/model")

    moved = await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor("live"))

    assert moved == []
    assert (await _status(tmp_path, "live")).status == "loading"


async def test_a_watchdog_restore_in_flight_is_left_to_the_was_serving_path(tmp_path):
    """A 'loading' row carrying a was-serving prior_status is the watchdog
    reloading an engine that WAS serving. Demoting it here would silently
    disable restore_after_warden_restart — the 11.5h outage of 2026-08-18."""
    await _seed(tmp_path, restoring="loading")
    _plant_weights(tmp_path, "org/model")
    async with open_db(tmp_path / "vllm-warden.db") as db:
        await ModelRepo(db).set_prior_status("restoring", "loaded")

    moved = await reconcile_stranded_models(_settings(tmp_path), _FakeSupervisor())

    assert moved == []
    assert (await _status(tmp_path, "restoring")).status == "loading"


@pytest.mark.parametrize("supervisor", [None])
async def test_a_missing_supervisor_is_treated_as_holding_nothing(tmp_path, supervisor):
    await _seed(tmp_path, stranded="loading")

    moved = await reconcile_stranded_models(_settings(tmp_path), supervisor)

    assert moved == [("stranded", "registered")]
