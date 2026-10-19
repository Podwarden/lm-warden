"""Load history: remember per-step load durations for the "usual" bars.

Covers ``LoadHistoryRepo`` (record + usual + cascade) and ``load_fingerprint``
(changes with the right fields, ignores the wrong ones, includes the baked
engine version for the in-container engine).
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.load_history import LoadHistoryRepo, load_fingerprint
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.load_progress.parse import Step

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _model(
    model_id: str = "m1",
    *,
    hf_repo: str = "o/r",
    hf_revision: str = "main",
    filename: str | None = None,
    dtype: str | None = None,
    gpu_indices: list[int] | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    backend: str | None = None,
    data_parallel_size: int = 1,
    engine_image: str | None = None,
    engine_vllm_version: str | None = None,
    engine_channel: str | None = None,
    served_model_name: str | None = None,
) -> ModelRow:
    return ModelRow(
        id=model_id,
        served_model_name=served_model_name or model_id,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        gpu_indices=gpu_indices or [0],
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        extra_args=extra_args or [],
        status="pulled",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env=extra_env or {},
        filename=filename,
        backend=backend,
        data_parallel_size=data_parallel_size,
        engine_image=engine_image,
        engine_vllm_version=engine_vllm_version,
        engine_channel=engine_channel,
    )


async def _seed_model(db, model_id: str = "m1") -> ModelRow:
    row = _model(model_id)
    await ModelRepo(db).insert(row)
    return row


# ---------------------------------------------------------------------------
# load_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_changes_with_tp_and_max_len():
    """TP size and max_model_len are part of the fingerprint."""
    base = _model()
    tp2 = _model(tensor_parallel_size=2)
    long_ctx = _model(max_model_len=32768)
    assert load_fingerprint(base, None) != load_fingerprint(tp2, None)
    assert load_fingerprint(base, None) != load_fingerprint(long_ctx, None)


def test_fingerprint_ignores_unrelated_fields():
    """``served_model_name`` does not affect the fingerprint."""
    a = _model("m1", served_model_name="alias-a")
    b = _model("m1", served_model_name="alias-b")
    assert load_fingerprint(a, None) == load_fingerprint(b, None)


def test_fingerprint_changes_with_image():
    """A different engine image pin starts a fresh history."""
    m = _model()
    assert load_fingerprint(m, "vllm/vllm-openai:v0.26") != load_fingerprint(
        m, "vllm/vllm-openai:v0.32"
    )


def test_fingerprint_changes_with_vllm_env():
    """A ``VLLM_`` env var key is part of the fingerprint."""
    a = _model(extra_env={"VLLM_GPU_MEMORY_UTILIZATION": "0.8"})
    b = _model(extra_env={"VLLM_GPU_MEMORY_UTILIZATION": "0.9"})
    assert load_fingerprint(a, None) != load_fingerprint(b, None)


def test_fingerprint_ignores_non_vllm_env():
    """Non-``VLLM_`` env vars (secrets) do not affect the fingerprint."""
    a = _model(extra_env={"HF_TOKEN": "secret-a"})
    b = _model(extra_env={"HF_TOKEN": "secret-b"})
    assert load_fingerprint(a, None) == load_fingerprint(b, None)


def test_engine_version_changes_fingerprint():
    """When no image pin is set, the baked engine version is part of the
    fingerprint, so an engine bump (e.g. vLLM 0.26 -> 0.32) starts a fresh
    history."""
    m = _model()
    fp_old = load_fingerprint(m, None, baked_engine_version="0.26.0")
    fp_new = load_fingerprint(m, None, baked_engine_version="0.32.0")
    assert fp_old != fp_new


# ---------------------------------------------------------------------------
# LoadHistoryRepo.record / usual
# ---------------------------------------------------------------------------


async def test_no_history_gives_none(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        repo = LoadHistoryRepo(db)
        usual_total, usual_steps = await repo.usual("m1", "fp1")
    assert usual_total is None
    assert usual_steps == {}


async def test_usual_is_median_of_last_five(tmp_data_dir):
    """``usual`` returns the median of the last ``n=5`` totals and per-step
    medians."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        repo = LoadHistoryRepo(db)
        # Record 5 loads with known step durations.
        # WEIGHTS: [10, 20, 30, 40, 50] -> median 30
        # total:   [60, 70, 80, 90, 100] -> median 80
        for i in range(5):
            await repo.record(
                "m1",
                "fp1",
                total_s=60 + i * 10,
                steps={Step.WEIGHTS: 10.0 + i * 10},
            )
        usual_total, usual_steps = await repo.usual("m1", "fp1")
    assert usual_total == pytest.approx(80.0)
    assert usual_steps[Step.WEIGHTS] == pytest.approx(30.0)


async def test_record_keeps_newest_twenty(tmp_data_dir):
    """``record`` keeps only the newest 20 rows per model."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        repo = LoadHistoryRepo(db)
        for i in range(25):
            await repo.record(
                "m1",
                "fp1",
                total_s=float(i),
                steps={Step.WEIGHTS: float(i)},
            )
        cur = await db.execute("SELECT COUNT(*) FROM model_load_history WHERE model_id = 'm1'")
        (count,) = await cur.fetchone()
    assert count == 20


async def test_record_never_fails_the_load(tmp_data_dir):
    """A recording error (e.g. a bad model_id FK) must not raise."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        repo = LoadHistoryRepo(db)
        # Non-existent model_id: FK violation. Must not raise.
        await repo.record(
            "nonexistent",
            "fp1",
            total_s=42.0,
            steps={Step.WEIGHTS: 10.0},
        )


async def test_delete_model_cascades(tmp_data_dir):
    """Deleting a model cascades to its load history (FK ON DELETE CASCADE)."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        repo = LoadHistoryRepo(db)
        await repo.record("m1", "fp1", total_s=60.0, steps={Step.WEIGHTS: 10.0})
        await ModelRepo(db).delete("m1")
    with sqlite3.connect(db_path) as c:
        (count,) = c.execute(
            "SELECT COUNT(*) FROM model_load_history WHERE model_id = 'm1'"
        ).fetchone()
    assert count == 0


async def test_usual_separates_by_fingerprint(tmp_data_dir):
    """Different fingerprints get independent histories."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        repo = LoadHistoryRepo(db)
        await repo.record("m1", "fp-a", total_s=60.0, steps={Step.WEIGHTS: 10.0})
        await repo.record("m1", "fp-b", total_s=120.0, steps={Step.WEIGHTS: 20.0})
        total_a, steps_a = await repo.usual("m1", "fp-a")
        total_b, steps_b = await repo.usual("m1", "fp-b")
    assert total_a == pytest.approx(60.0)
    assert steps_a[Step.WEIGHTS] == pytest.approx(10.0)
    assert total_b == pytest.approx(120.0)
    assert steps_b[Step.WEIGHTS] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Wiring: recording never fails the load; usual_total_s reaches the API
# ---------------------------------------------------------------------------


async def test_record_failure_never_fails_the_load(tmp_data_dir):
    """A recording error inside LoadHistoryRepo.record must not propagate.

    Simulates the ``on_loaded`` path in ``start_engine``: even if the DB
    write fails (here a closed connection), the call must return cleanly
    rather than raising and failing the load.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
    # Use a stale (closed) connection: every operation will fail.
    await db.close()
    stale_repo = LoadHistoryRepo(db)
    # Must not raise despite the closed connection.
    await stale_repo.record(
        "m1",
        "fp1",
        total_s=42.0,
        steps={Step.WEIGHTS: 10.0},
    )


async def test_usual_total_reaches_the_api(tmp_data_dir):
    """``usual_total_s`` stored on the tracker reaches the API snapshot via
    ``build_snapshot``."""
    from datetime import UTC, datetime

    from app.runtime.load_progress.snapshot import build_snapshot
    from app.runtime.load_progress.tracker import LoadTracker

    tracker = LoadTracker(
        model_id="m1",
        generation=0,
        backend="vllm",
        started_at=100.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
        usual={Step.WEIGHTS: 10.0},
        usual_total_s=60.0,
    )
    snap = build_snapshot(
        tracker,
        verdict=None,
        now=110.0,
        usual_total_s=tracker.usual_total_s,
        usual_steps=tracker.usual,
        now_wall=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )
    assert snap["usual_total_s"] == 60.0
    # ready_eta = now_wall + (usual_total_s - elapsed_s) = 00:01 + (60 - 10) = 00:01:50
    assert snap["ready_eta"] is not None
    expected_eta = datetime(2026, 1, 1, 0, 1, tzinfo=UTC) + timedelta(seconds=50.0)
    assert snap["ready_eta"] == expected_eta


# ---------------------------------------------------------------------------
# Fix round 1: usual times fill in off the load path
# ---------------------------------------------------------------------------


def _tracker():
    from datetime import UTC, datetime

    from app.runtime.load_progress.tracker import LoadTracker

    return LoadTracker(
        model_id="m1",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def test_read_usual_runs_off_the_event_loop(tmp_data_dir, monkeypatch):
    """The history read runs in a worker thread, never on the loop's thread."""
    import threading

    import app.models.routes_api as mod

    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        await LoadHistoryRepo(db).record("m1", "fp1", 60.0, {Step.WEIGHTS: 10.0})

    seen: list[threading.Thread] = []
    real_read = mod._read_usual_sync

    def _spy(path, model_id, fingerprint):
        seen.append(threading.current_thread())
        return real_read(path, model_id, fingerprint)

    monkeypatch.setattr(mod, "_read_usual_sync", _spy)
    tracker = _tracker()
    await mod._fill_usual(tracker, db_path, "m1", "fp1", lambda: True)

    assert seen, "the read was not called"
    assert seen[0] is not threading.current_thread(), "read ran on the event loop thread"
    assert tracker.usual_total_s == pytest.approx(60.0)


async def test_fill_usual_sets_values_when_history_exists(tmp_data_dir):
    import app.models.routes_api as mod

    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        await LoadHistoryRepo(db).record("m1", "fp1", 60.0, {Step.WEIGHTS: 10.0})
    tracker = _tracker()
    await mod._fill_usual(tracker, db_path, "m1", "fp1", lambda: True)
    assert tracker.usual_total_s == pytest.approx(60.0)
    assert tracker.usual[Step.WEIGHTS] == pytest.approx(10.0)


async def test_fill_usual_drops_values_for_a_superseded_load(tmp_data_dir):
    import app.models.routes_api as mod

    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed_model(db)
        await LoadHistoryRepo(db).record("m1", "fp1", 60.0, {Step.WEIGHTS: 10.0})
    tracker = _tracker()
    await mod._fill_usual(tracker, db_path, "m1", "fp1", lambda: False)
    assert tracker.usual_total_s is None
    assert tracker.usual == {}


async def test_failing_read_leaves_usual_none_and_does_not_raise(tmp_data_dir, monkeypatch):
    import sqlite3

    import app.models.routes_api as mod

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(mod, "_read_usual_sync", _boom)
    tracker = _tracker()
    await mod._fill_usual(tracker, tmp_data_dir / "vllm-warden.db", "m1", "fp1", lambda: True)
    assert tracker.usual_total_s is None
    assert tracker.usual == {}


def test_read_usual_on_missing_db_returns_no_usual(tmp_data_dir):
    import app.models.routes_api as mod

    assert mod._read_usual_sync(str(tmp_data_dir / "absent.db"), "m1", "fp1") == (None, {})
