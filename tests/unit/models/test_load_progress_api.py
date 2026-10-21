"""Tests for the load-progress API (Task 6).

Two layers:
  * Pure :func:`build_snapshot` tests — the serialiser must be total (never
    raise) on a fresh tracker, must mark the right step running/stalled, and
    must compute ``ready_eta`` / ``give_up_at`` only when the inputs are known.
  * Route tests via the ``client`` fixture — 404 when no active load, the
    snapshot shape matches the schema, the list returns only active loads,
    auth is required, and ``GET /api/models/load-progress`` is NOT shadowed by
    the ``/{model_id}`` route (the registration-order hazard).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.runtime.load_progress.parse import STEP_LABELS, VLLM_STEPS, Step
from app.runtime.load_progress.policy import Verdict
from app.runtime.load_progress.snapshot import build_snapshot
from app.runtime.load_progress.tracker import LoadTracker, VitalsSample
from tests.conftest import jwt_login, seed_admin_user


@pytest.fixture(autouse=True)
def _no_real_gpu_probe_in_lifespan(monkeypatch):
    # The lifespan's stats sampler would spawn real nvidia-smi / ioreg
    # probes; cancelled mid-spawn at shutdown, their subprocess transports
    # outlive the TestClient loop and print "Event loop is closed" at GC.
    monkeypatch.setattr("app.runtime.stats_sampler.query_gpus", AsyncMock(return_value=[]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    *,
    model_id: str = "m",
    backend: str = "vllm",
    started_at: float = 0.0,
    started_wall: datetime | None = None,
    usual: dict[Step, float] | None = None,
    dp_size: int | None = None,
) -> LoadTracker:
    return LoadTracker(
        model_id=model_id,
        generation=0,
        backend=backend,
        started_at=started_at,
        started_wall=started_wall or datetime(2026, 1, 1, tzinfo=UTC),
        usual=usual,
        dp_size=dp_size,
    )


def _progressing_verdict() -> Verdict:
    return Verdict(
        state="progressing",
        stalled_for_s=0.0,
        give_up_at_s=1200.0,
        reason=None,
        hint=None,
    )


def _stalled_verdict() -> Verdict:
    return Verdict(
        state="stalled",
        stalled_for_s=300.0,
        give_up_at_s=1200.0,
        reason="no progress for 5 min",
        hint=None,
    )


# ---------------------------------------------------------------------------
# build_snapshot — total on a fresh tracker
# ---------------------------------------------------------------------------


def test_build_snapshot_total_on_fresh_tracker():
    """build_snapshot must not raise on a tracker with zero events/vitals."""
    t = _make_tracker()
    snap = build_snapshot(
        t,
        None,
        now=10.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    # verdict None ⇒ progressing, stalled_for_s 0, no give_up_at.
    assert snap["state"] == "progressing"
    assert snap["stalled_for_s"] == 0.0
    assert snap["give_up_at"] is None
    assert snap["reason"] is None
    assert snap["hint"] is None
    assert snap["model_id"] == "m"
    assert snap["backend"] == "vllm"
    assert snap["elapsed_s"] == 10.0
    assert snap["layout"] is None  # tracker has no TP info
    assert snap["usual_total_s"] is None
    assert snap["ready_eta"] is None  # no history
    # Steps: all vLLM steps, first one (START) running, rest pending.
    assert len(snap["steps"]) == len(VLLM_STEPS)
    assert snap["steps"][0]["status"] == "running"
    assert snap["steps"][0]["key"] == Step.START.value
    assert all(s["status"] == "pending" for s in snap["steps"][1:])
    # current_step is the first step's label.
    assert snap["current_step"] == STEP_LABELS[Step.START]
    # Vitals: empty series, None log.
    assert snap["vitals"]["series"] == []
    assert snap["vitals"]["last_log_age_s"] is None
    assert snap["vitals"]["last_log_line"] is None
    assert snap["vitals"]["log_quiet_normal"] is False
    assert snap["vitals"]["gpu_mem_total_mib"] is None


# ---------------------------------------------------------------------------
# Step statuses: running and stalled
# ---------------------------------------------------------------------------


def test_step_statuses_running_and_stalled():
    """The first not-done step is 'running' when progressing, 'stalled' when
    the verdict is stalled."""
    t = _make_tracker(started_at=0.0)
    # Feed a log line so last_log_at is set; START.started_at is already 0.0.
    t.feed([], now=1.0, line="starting engine")
    snap = build_snapshot(
        t,
        _progressing_verdict(),
        now=5.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert snap["steps"][0]["status"] == "running"
    # START started at 0.0 (constructor), now=5.0 → elapsed 5.0.
    assert snap["steps"][0]["elapsed_s"] == pytest.approx(5.0)
    assert all(s["status"] == "pending" for s in snap["steps"][1:])

    # Now with a stalled verdict: the running step becomes stalled.
    snap = build_snapshot(
        t,
        _stalled_verdict(),
        now=5.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert snap["steps"][0]["status"] == "stalled"
    assert snap["state"] == "stalled"
    assert snap["stalled_for_s"] == 300.0
    assert all(s["status"] == "pending" for s in snap["steps"][1:])


def test_step_done_marks_completed_step():
    """A step the tracker says is done is 'done' with a duration_s."""
    t = _make_tracker(started_at=0.0)
    t.feed([], now=1.0, line="starting engine")
    # End START by starting a later step (WEIGHTS) at now=3.0.
    from app.runtime.load_progress.parse import LogEvent

    t.feed(
        [LogEvent("", "step_start", Step.WEIGHTS, None, None, "loading")],
        now=3.0,
        line="loading weights",
    )
    snap = build_snapshot(
        t,
        _progressing_verdict(),
        now=5.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert snap["steps"][0]["status"] == "done"
    # START started at 0.0 (constructor), ended at 3.0 → duration 3.0.
    assert snap["steps"][0]["duration_s"] == pytest.approx(3.0)
    # WEIGHTS is now the running step.
    weights_step = next(s for s in snap["steps"] if s["key"] == Step.WEIGHTS.value)
    assert weights_step["status"] == "running"


# ---------------------------------------------------------------------------
# ready_eta and give_up_at
# ---------------------------------------------------------------------------


def test_eta_and_give_up_at_fields():
    """ready_eta only when usual_total_s known; give_up_at = started_wall + s."""
    started_wall = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t = _make_tracker(started_at=0.0, started_wall=started_wall)
    now_wall = datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC)
    verdict = Verdict(
        state="progressing",
        stalled_for_s=0.0,
        give_up_at_s=1200.0,
        reason=None,
        hint=None,
    )
    # usual_total_s known (e.g. 600s), elapsed 10s → ready_eta = now + 590s.
    snap = build_snapshot(
        t, verdict, now=10.0, usual_total_s=600.0, usual_steps=None, now_wall=now_wall
    )
    assert snap["usual_total_s"] == 600.0
    assert snap["ready_eta"] == now_wall + timedelta(seconds=590.0)
    # give_up_at = started_wall + 1200s.
    assert snap["give_up_at"] == started_wall + timedelta(seconds=1200.0)

    # Without history: no ready_eta, but give_up_at still set from verdict.
    snap = build_snapshot(
        t, verdict, now=10.0, usual_total_s=None, usual_steps=None, now_wall=now_wall
    )
    assert snap["usual_total_s"] is None
    assert snap["ready_eta"] is None
    assert snap["give_up_at"] == started_wall + timedelta(seconds=1200.0)

    # verdict None: no give_up_at.
    snap = build_snapshot(
        t, None, now=10.0, usual_total_s=None, usual_steps=None, now_wall=now_wall
    )
    assert snap["give_up_at"] is None
    assert snap["ready_eta"] is None


def test_ready_eta_clamped_to_zero_when_past_usual():
    """ready_eta = now_wall + max(0, usual_total_s - elapsed)."""
    started_wall = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t = _make_tracker(started_at=0.0, started_wall=started_wall)
    now_wall = datetime(2026, 1, 1, 12, 10, 0, tzinfo=UTC)  # 600s later
    snap = build_snapshot(
        t,
        _progressing_verdict(),
        now=600.0,
        usual_total_s=300.0,
        usual_steps=None,
        now_wall=now_wall,
    )
    # elapsed 600 > usual 300 → remaining 0 → ready_eta == now_wall.
    assert snap["ready_eta"] == now_wall


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------


def test_vitals_cpu_cores_from_samples():
    """series: each point has t_s, cpu_cores (ΔCPU-s / Δt), gpu_mem_mib."""
    t = _make_tracker()
    t.add_vitals(VitalsSample(at=1.0, cpu_seconds=0.0, gpu_mem_mib=100, read_bytes=0))
    t.add_vitals(VitalsSample(at=2.0, cpu_seconds=1.0, gpu_mem_mib=100, read_bytes=0))
    t.add_vitals(VitalsSample(at=3.0, cpu_seconds=2.5, gpu_mem_mib=200, read_bytes=0))
    snap = build_snapshot(
        t,
        None,
        now=4.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    series = snap["vitals"]["series"]
    assert len(series) == 3
    # First point: t_s=1.0 (relative to started_at=0.0), cpu_cores=None (no prev)
    assert series[0]["t_s"] == pytest.approx(1.0)
    assert series[0]["cpu_cores"] is None
    assert series[0]["gpu_mem_mib"] == 100
    # Second: (1.0-0.0)/1.0 = 1.0
    assert series[1]["t_s"] == pytest.approx(2.0)
    assert series[1]["cpu_cores"] == pytest.approx(1.0)
    assert series[1]["gpu_mem_mib"] == 100
    # Third: (2.5-1.0)/1.0 = 1.5
    assert series[2]["t_s"] == pytest.approx(3.0)
    assert series[2]["cpu_cores"] == pytest.approx(1.5)
    assert series[2]["gpu_mem_mib"] == 200


def test_vitals_cpu_none_when_sample_missing():
    """cpu_cores is None when either sample's cpu_seconds is None or Δt <= 0."""
    t = _make_tracker()
    t.add_vitals(VitalsSample(at=1.0, cpu_seconds=None, gpu_mem_mib=100, read_bytes=0))
    t.add_vitals(VitalsSample(at=2.0, cpu_seconds=1.0, gpu_mem_mib=100, read_bytes=0))
    snap = build_snapshot(
        t,
        None,
        now=3.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    series = snap["vitals"]["series"]
    assert len(series) == 2
    assert series[0]["cpu_cores"] is None  # first point, no prev
    assert series[1]["cpu_cores"] is None  # prev had cpu_seconds=None


def test_vitals_log_quiet_normal():
    """log_quiet_normal when log age > 60s and CPU/GPU moved in the last 60s."""
    t = _make_tracker()
    t.feed([], now=0.0, line="starting")  # last_log_at = 0.0
    # Samples within the last 60s of now=200 showing CPU busy.
    t.add_vitals(VitalsSample(at=140.0, cpu_seconds=0.0, gpu_mem_mib=100, read_bytes=0))
    t.add_vitals(VitalsSample(at=150.0, cpu_seconds=10.0, gpu_mem_mib=100, read_bytes=0))
    # now=200 → log age 200s > 60s; CPU 1.0 cores > 0.5 → quiet_normal True.
    snap = build_snapshot(
        t,
        None,
        now=200.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert snap["vitals"]["log_quiet_normal"] is True


def test_vitals_log_quiet_normal_false_when_log_recent():
    """log_quiet_normal False when log age <= 60s."""
    t = _make_tracker()
    t.feed([], now=50.0, line="starting")  # last_log_at = 50.0
    t.add_vitals(VitalsSample(at=50.0, cpu_seconds=0.0, gpu_mem_mib=100, read_bytes=0))
    t.add_vitals(VitalsSample(at=51.0, cpu_seconds=1.0, gpu_mem_mib=100, read_bytes=0))
    snap = build_snapshot(
        t,
        None,
        now=55.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    # log age 5s < 60s → False.
    assert snap["vitals"]["log_quiet_normal"] is False


def test_vitals_series_downsampled_to_120():
    """When there are more than 120 samples, the series is downsampled to
    ≤ 120 points, keeping evenly spaced samples plus the newest."""
    t = _make_tracker()
    for i in range(200):
        t.add_vitals(VitalsSample(at=float(i), cpu_seconds=float(i), gpu_mem_mib=i, read_bytes=0))
    snap = build_snapshot(
        t,
        None,
        now=200.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    series = snap["vitals"]["series"]
    assert len(series) <= 120
    # The first point and the newest point are both kept.
    assert series[0]["t_s"] == pytest.approx(0.0)
    assert series[-1]["t_s"] == pytest.approx(199.0)


def test_vitals_series_t_s_monotonic():
    """t_s is monotonically non-decreasing across the series."""
    t = _make_tracker()
    for i in range(10):
        t.add_vitals(
            VitalsSample(at=float(i + 1), cpu_seconds=float(i), gpu_mem_mib=i, read_bytes=0)
        )
    snap = build_snapshot(
        t,
        None,
        now=20.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    series = snap["vitals"]["series"]
    for i in range(1, len(series)):
        assert (
            series[i]["t_s"] >= series[i - 1]["t_s"]
        ), f"t_s not monotonic at index {i}: {series[i - 1]['t_s']} -> {series[i]['t_s']}"


def test_vitals_gpu_mem_total_mib_from_tracker():
    """gpu_mem_total_mib is read from the tracker (set by the route)."""
    t = _make_tracker()
    t.gpu_mem_total_mib = 8 * 190 * 1024  # 8 × B300 ≈ 190 GiB each
    t.add_vitals(VitalsSample(at=1.0, cpu_seconds=0.0, gpu_mem_mib=100, read_bytes=0))
    snap = build_snapshot(
        t,
        None,
        now=2.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert snap["vitals"]["gpu_mem_total_mib"] == 8 * 190 * 1024


# ---------------------------------------------------------------------------
# Step status: skipped
# ---------------------------------------------------------------------------


def test_step_skipped_for_tune_without_autotuner():
    """A step the engine never reports (TUNE on a model with no AutoTuner) is
    ``skipped`` once a later step has started."""
    t = _make_tracker(started_at=0.0)
    from app.runtime.load_progress.parse import LogEvent

    # Feed WEIGHTS counter to start WEIGHTS (closes START).
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 1, 10, "w")], now=1.0, line="w")
    # Feed GRAPHS counter (after KV done) to start GRAPHS — TUNE is between KV
    # and GRAPHS and was never started, so it should be skipped.
    t.feed([LogEvent("r0", "step_end", Step.KV, None, None, "kv")], now=2.0, line="kv")
    t.feed([LogEvent("r0", "counter", Step.GRAPHS, 1, 83, "g")], now=3.0, line="g")
    snap = build_snapshot(
        t,
        _progressing_verdict(),
        now=4.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    tune_step = next(s for s in snap["steps"] if s["key"] == Step.TUNE.value)
    assert tune_step["status"] == "skipped"


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


def _jwt_login(client):
    return jwt_login(client)


def test_requires_auth(tmp_data_dir, client):
    """Both endpoints 401 without a JWT."""
    client.get("/healthz")
    assert client.get("/api/models/load-progress").status_code == 401
    assert client.get("/api/models/some-id/load-progress").status_code == 401


def test_404_when_not_loading(tmp_data_dir, client):
    """GET /api/models/{id}/load-progress 404s when no active load exists."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.get("/api/models/no-such-model/load-progress", headers=auth)
    assert r.status_code == 404


def test_snapshot_shape_matches_schema(tmp_data_dir, client):
    """A registered tracker returns a snapshot with all the schema fields."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    reg = client.app.state.load_progress
    t = LoadTracker(
        model_id="m1",
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
    )
    reg.start(t)
    try:
        r = client.get("/api/models/m1/load-progress", headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_id"] == "m1"
        assert body["backend"] == "vllm"
        assert body["state"] == "progressing"
        assert body["stalled_for_s"] == 0.0
        assert body["give_up_at"] is None
        assert body["reason"] is None
        assert body["hint"] is None
        assert body["usual_total_s"] is None
        assert body["ready_eta"] is None
        assert isinstance(body["elapsed_s"], float)
        assert isinstance(body["current_step"], str)
        assert isinstance(body["steps"], list)
        assert len(body["steps"]) == len(VLLM_STEPS)
        step = body["steps"][0]
        for key in (
            "key",
            "label",
            "status",
            "duration_s",
            "elapsed_s",
            "count",
            "total",
            "usual_s",
            "detail",
        ):
            assert key in step
        vitals = body["vitals"]
        for key in (
            "series",
            "last_log_age_s",
            "last_log_line",
            "log_quiet_normal",
            "gpu_mem_total_mib",
        ):
            assert key in vitals
    finally:
        reg.finish("m1", 0)


def test_list_returns_only_active(tmp_data_dir, client):
    """GET /api/models/load-progress returns only active loads, empty when none."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    # Empty when nothing loading.
    r = client.get("/api/models/load-progress", headers=auth)
    assert r.status_code == 200, r.text
    assert r.json() == []

    reg = client.app.state.load_progress
    t1 = LoadTracker(
        model_id="a",
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
    )
    t2 = LoadTracker(
        model_id="b",
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
    )
    reg.start(t1)
    reg.start(t2)
    try:
        r = client.get("/api/models/load-progress", headers=auth)
        assert r.status_code == 200, r.text
        ids = sorted(m["model_id"] for m in r.json())
        assert ids == ["a", "b"]
    finally:
        reg.finish("a", 0)
        reg.finish("b", 0)


def test_eta_absent_without_history(tmp_data_dir, client):
    """ready_eta is None when usual_total_s is unknown (no history yet)."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    reg = client.app.state.load_progress
    t = LoadTracker(
        model_id="m2",
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
    )
    reg.start(t)
    try:
        r = client.get("/api/models/m2/load-progress", headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ready_eta"] is None
        assert body["usual_total_s"] is None
    finally:
        reg.finish("m2", 0)


def test_load_progress_list_route_not_shadowed_by_model_id(tmp_data_dir, client):
    """GET /api/models/load-progress must NOT be captured by the /{model_id}
    route.  FastAPI matches in registration order; the list route is registered
    before /{model_id}, so 'load-progress' is not treated as a model id."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    # No active loads: the list route returns [] (200), not a 404 from get_model.
    r = client.get("/api/models/load-progress", headers=auth)
    assert r.status_code == 200, r.text
    assert r.json() == []

    # With an active load named "load-progress" (adversarial): the list route
    # still wins and returns it, rather than the per-model route 404ing.
    reg = client.app.state.load_progress
    t = LoadTracker(
        model_id="load-progress",
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
    )
    reg.start(t)
    try:
        r = client.get("/api/models/load-progress", headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        # The list route returns a list containing the "load-progress" model.
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["model_id"] == "load-progress"
    finally:
        reg.finish("load-progress", 0)


def test_layout_derived_from_model_row_tp(tmp_data_dir, client):
    """The layout string is derived from the tracker's dp_size and the model
    row's tensor_parallel_size (the tracker knows dp_size, not TP)."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1, 2, 3])
    auth = _jwt_login(client)
    # Create a model with tp=2, dp=2 (4 GPUs).
    from tests.conftest import csrf_header

    r = client.post(
        "/api/models",
        json={
            "served_model_name": "tp2dp2",
            "hf_repo": "Qwen/Qwen3-8B",
            "gpu_indices": [0, 1, 2, 3],
            "tensor_parallel_size": 2,
            "data_parallel_size": 2,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    model_id = r.json()["id"]

    reg = client.app.state.load_progress
    t = LoadTracker(
        model_id=model_id,
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
        dp_size=2,
    )
    reg.start(t)
    try:
        r = client.get(f"/api/models/{model_id}/load-progress", headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["layout"] == "2 replicas × TP2"
    finally:
        reg.finish(model_id, 0)


# ---------------------------------------------------------------------------
# TUNE detail order and lazy GPU capacity
# ---------------------------------------------------------------------------


def test_tune_detail_reads_name_then_op_count():
    """The TUNE step detail reads "<op name> · op N", the brief's order."""
    from app.runtime.load_progress.parse import LogEvent

    t = _make_tracker()
    t.feed([LogEvent("", "step_end", Step.KV, None, None, "kv")], now=1.0, line="kv")
    t.feed(
        [LogEvent("", "counter", Step.TUNE, 0, 22, "trtllm_batch_decode_mla")],
        now=2.0,
        line="t1",
    )
    t.feed(
        [LogEvent("", "counter", Step.TUNE, 6, 22, "trtllm_fp4_block_scale_moe")],
        now=3.0,
        line="t2",
    )
    snap = build_snapshot(
        t,
        None,
        now=4.0,
        usual_total_s=None,
        usual_steps=None,
        now_wall=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    tune = next(s for s in snap["steps"] if s["key"] == Step.TUNE.value)
    assert tune["detail"] == "trtllm_fp4_block_scale_moe · op 2"
    assert tune["count"] == 6 and tune["total"] == 22


class _FakeGpu:
    def __init__(self, index: int, memory_total_mib: int) -> None:
        self.index = index
        self.memory_total_mib = memory_total_mib


class _FakeSnap:
    def __init__(self, gpus: list[_FakeGpu]) -> None:
        self.gpus = gpus


class _FakeProbe:
    """Stands in for the GPU probe cache; counts calls, can fail."""

    def __init__(self, gpus: list[_FakeGpu] | None = None, *, error: bool = False) -> None:
        self._gpus = gpus or []
        self._error = error
        self.calls = 0

    async def get(self) -> _FakeSnap:
        self.calls += 1
        if self._error:
            raise RuntimeError("nvidia-smi exploded")
        return _FakeSnap(self._gpus)


def _tracked_model(client, tmp_data_dir, probe: _FakeProbe):
    """A model on GPUs 0 and 1 with a tracker that has NO GPU total yet."""
    from tests.conftest import csrf_header

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1, 2, 3])
    auth = _jwt_login(client)
    client.app.state.gpu_probe_cache = probe
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "gputotal",
            "hf_repo": "Qwen/Qwen3-8B",
            "gpu_indices": [0, 1],
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    model_id = r.json()["id"]
    reg = client.app.state.load_progress
    t = LoadTracker(
        model_id=model_id,
        generation=0,
        backend="vllm",
        started_at=time.monotonic(),
        started_wall=datetime.now(UTC),
        dp_size=1,
    )
    assert t.gpu_mem_total_mib is None
    reg.start(t)
    return auth, model_id, reg, t


def test_gpu_total_filled_on_first_snapshot_and_probed_once(tmp_data_dir, client):
    probe = _FakeProbe([_FakeGpu(0, 192000), _FakeGpu(1, 192000)])
    auth, model_id, reg, t = _tracked_model(client, tmp_data_dir, probe)
    try:
        for _ in range(2):
            r = client.get(f"/api/models/{model_id}/load-progress", headers=auth)
            assert r.status_code == 200, r.text
            assert r.json()["vitals"]["gpu_mem_total_mib"] == 384000
        assert t.gpu_mem_total_mib == 384000
        assert probe.calls == 1, "the GPU probe must run once per load, not per poll"
    finally:
        reg.finish(model_id, 0)


def test_failed_gpu_probe_gives_none_and_is_not_retried(tmp_data_dir, client):
    probe = _FakeProbe(error=True)
    auth, model_id, reg, t = _tracked_model(client, tmp_data_dir, probe)
    try:
        for _ in range(3):
            r = client.get(f"/api/models/{model_id}/load-progress", headers=auth)
            assert r.status_code == 200, r.text
            assert r.json()["vitals"]["gpu_mem_total_mib"] is None
        assert probe.calls == 1, "a failed probe must be remembered, not retried each poll"
    finally:
        reg.finish(model_id, 0)


def test_partial_gpu_probe_gives_none_not_a_short_sum(tmp_data_dir, client):
    # The model is on GPUs 0 and 1, but the probe only reports GPU 0.
    probe = _FakeProbe([_FakeGpu(0, 192000)])
    auth, model_id, reg, t = _tracked_model(client, tmp_data_dir, probe)
    try:
        r = client.get(f"/api/models/{model_id}/load-progress", headers=auth)
        assert r.status_code == 200, r.text
        assert r.json()["vitals"]["gpu_mem_total_mib"] is None
    finally:
        reg.finish(model_id, 0)
