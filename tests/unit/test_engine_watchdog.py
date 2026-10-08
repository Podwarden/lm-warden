"""Engine watchdog: the wrapper outliving the engine must not go unnoticed.

Regression cover for 2026-08-17, when EngineCore died four times while
`vllm serve` stayed alive, so `handle.wait()` never returned, the row kept
status='loaded', and the proxy forwarded to a dead engine indefinitely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from app.runtime.watchdog import (
    RestartBudget,
    capture_evidence,
    probe_health,
    prune_crash_dirs,
)


@dataclass
class FakeSettings:
    data_dir: Path
    watchdog_crash_keep: int = 3
    watchdog_max_restarts: int = 3
    watchdog_restart_window_s: float = 3600.0
    # Must mirror every Settings field capture_evidence() reads. Omitting one
    # does not raise here — capture_evidence guards each step with a broad
    # `except Exception`, so a missing attribute is swallowed as a warning and
    # the artefact is silently absent. Keep this stub in sync with
    # app/config.py or the evidence tests fail somewhere unrelated.
    watchdog_crash_log_bytes: int = 4 * 1024 * 1024

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vllm-warden.db"

    @property
    def crashes_dir(self) -> Path:
        return self.data_dir / "crashes"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


# --- detection --------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_reports_healthy_on_200(httpx_mock):
    httpx_mock.add_response(url="http://h:9/health", status_code=200, text="OK")
    healthy, detail = await probe_health("h", 9)
    assert healthy and "200" in detail


@pytest.mark.asyncio
async def test_probe_reports_unhealthy_on_500(httpx_mock):
    """A dead EngineCore under a live wrapper answers 500, not a refused socket."""
    httpx_mock.add_response(url="http://h:9/health", status_code=500, text="dead")
    healthy, detail = await probe_health("h", 9)
    assert not healthy
    assert "500" in detail, "the observed status must reach the crash report"


@pytest.mark.asyncio
async def test_probe_reports_unhealthy_on_connection_error(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    healthy, detail = await probe_health("h", 9)
    assert not healthy and "ConnectError" in detail


# --- crash-loop guard -------------------------------------------------------


def test_budget_allows_up_to_max_then_refuses():
    b = RestartBudget(max_restarts=3, window_s=3600)
    for i in range(3):
        assert b.allow("m", now=100 + i), f"restart {i + 1} should be allowed"
        b.record("m", now=100 + i)
    assert not b.allow("m", now=103), "the 4th restart in the window must be refused"


def test_budget_forgets_events_outside_the_window():
    b = RestartBudget(max_restarts=1, window_s=60)
    b.record("m", now=0)
    assert not b.allow("m", now=30)
    assert b.allow("m", now=100), "an old crash must not block a much later restart"


def test_budget_reset_on_recovery():
    b = RestartBudget(max_restarts=1, window_s=3600)
    b.record("m", now=0)
    assert not b.allow("m", now=1)
    b.reset("m")
    assert b.allow("m", now=2)


# --- evidence ---------------------------------------------------------------


def test_capture_writes_report_and_engine_log(tmp_path):
    s = FakeSettings(data_dir=tmp_path)
    s.logs_dir.mkdir(parents=True)
    (s.logs_dir / "m1.log").write_text("EngineCore died here\n")

    out = capture_evidence(s, "m1", report={"model_id": "m1", "wrapper_returncode": -11})

    assert out is not None
    report = json.loads((out / "report.json").read_text())
    assert report["wrapper_returncode"] == -11, "the fatal signal must be preserved"
    assert (
        "EngineCore died here" in (out / "engine.log").read_text()
    ), "the live log is reused by the next load, so the crash tail must be copied"


def test_capture_survives_a_missing_engine_log(tmp_path):
    """Evidence collection must never block recovery."""
    s = FakeSettings(data_dir=tmp_path)  # no logs dir at all
    out = capture_evidence(s, "m1", report={"model_id": "m1"})
    assert out is not None and (out / "report.json").exists()


def test_prune_keeps_only_the_newest_dirs(tmp_path):
    s = FakeSettings(data_dir=tmp_path, watchdog_crash_keep=2)
    root = s.crashes_dir / "m1"
    for stamp in ("20260817T010000Z", "20260817T020000Z", "20260817T030000Z"):
        (root / stamp).mkdir(parents=True)

    prune_crash_dirs(s, "m1")

    kept = sorted(d.name for d in root.iterdir())
    assert kept == ["20260817T020000Z", "20260817T030000Z"], "oldest must go first"


# --- check_once: threshold, ordering, crash-loop --------------------------


@dataclass
class FakeRow:
    id: str = "m1"
    status: str = "loaded"
    served_model_name: str = "m"
    hf_repo: str = "org/m"
    gpu_indices: list = field(default_factory=lambda: [0])
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    extra_args: list = field(default_factory=list)


class FakeSup:
    def __init__(self):
        self._handles = {"m1": type("H", (), {"returncode": None})()}
        self.unloaded = []
        self._generation = {}
        self._load_runners = {}
        self._claims = {}

    def claim(self, mid, holder):
        current = self._claims.get(mid)
        if current is None or current == holder:
            self._claims[mid] = holder
            return None
        return current

    def release_claim(self, mid, holder):
        if self._claims.get(mid) == holder:
            del self._claims[mid]

    def get_port(self, mid):
        return 9

    def get_host(self, mid):
        return "h"

    def get_pid(self, mid):
        return 4242

    def get_overrides(self, mid):
        return {"max_model_len": 4096}

    def get_generation(self, mid):
        return self._generation.get(mid, 0)

    def register_load_runner(self, mid, task):
        self._load_runners[mid] = task

    def is_running(self, mid):
        h = self._handles.get(mid)
        return h is not None and h.returncode is None

    async def unload(self, mid, *, force=False):
        self.unloaded.append((mid, force))
        # Mirror Supervisor.unload popping the handle, so a failed spawn is not
        # mistaken for a superseded one (is_running would otherwise stay True).
        self._handles.pop(mid, None)


@pytest.fixture
def wd_env(tmp_path, monkeypatch):
    """Patch check_once's DB and restart so the loop can be driven in isolation."""
    from app.runtime import watchdog

    s = FakeSettings(data_dir=tmp_path)
    s.logs_dir.mkdir(parents=True)
    (s.logs_dir / "m1.log").write_text("boom\n")
    s.watchdog_failure_threshold = 3

    calls = {"restarts": [], "order": []}

    class FakeRepo:
        def __init__(self, db):
            pass

        async def list_all(self):
            return [FakeRow()]

        async def get(self, mid):
            return FakeRow()

        async def update_status(self, mid, status, last_error=None):
            calls["order"].append(f"status:{status}")

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(watchdog, "ModelRepo", FakeRepo)
    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _NullDB())

    real_capture = watchdog.capture_evidence

    def spy_capture(settings, model_id, *, report):
        calls["order"].append("capture")
        return real_capture(settings, model_id, report=report)

    async def fake_restart(settings, app_state, model_id, overrides, *, recheck=None):
        calls["order"].append("restart")
        calls["restarts"].append((model_id, overrides))
        return "loaded"

    monkeypatch.setattr(watchdog, "capture_evidence", spy_capture)
    monkeypatch.setattr(watchdog, "_restart", fake_restart)

    app_state = type("S", (), {"supervisor": FakeSup()})()
    return watchdog, s, app_state, calls


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.asyncio
async def test_below_threshold_does_not_restart(wd_env, httpx_mock):
    watchdog, s, app_state, calls = wd_env
    httpx_mock.add_response(url="http://h:9/health", status_code=500, is_reusable=True)
    state, budget = {}, RestartBudget(3, 3600)

    for _ in range(2):
        await watchdog.check_once(s, app_state, state, budget)

    assert calls["restarts"] == [], "two failures under a threshold of three must wait"
    assert state["m1"] == 2


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.asyncio
async def test_threshold_reached_captures_then_restarts(wd_env, httpx_mock):
    """Ordering matters: recovery destroys the scene, so evidence comes first."""
    watchdog, s, app_state, calls = wd_env
    httpx_mock.add_response(url="http://h:9/health", status_code=500, is_reusable=True)
    state, budget = {}, RestartBudget(3, 3600)

    for _ in range(3):
        await watchdog.check_once(s, app_state, state, budget)

    assert len(calls["restarts"]) == 1
    assert calls["order"].index("capture") < calls["order"].index("restart")
    # The auto-capped context must be carried through, or the restart silently
    # differs from the load it is replacing.
    assert calls["restarts"][0][1] == {"max_model_len": 4096}


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.asyncio
async def test_recovery_resets_the_failure_streak(wd_env, httpx_mock):
    watchdog, s, app_state, calls = wd_env
    httpx_mock.add_response(url="http://h:9/health", status_code=500)
    httpx_mock.add_response(url="http://h:9/health", status_code=200)
    state, budget = {}, RestartBudget(3, 3600)

    await watchdog.check_once(s, app_state, state, budget)
    await watchdog.check_once(s, app_state, state, budget)

    assert state["m1"] == 0, "a transient blip must not accumulate toward a restart"
    assert calls["restarts"] == []


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.asyncio
async def test_crash_loop_guard_stops_restarting_and_marks_failed(wd_env, httpx_mock):
    watchdog, s, app_state, calls = wd_env
    httpx_mock.add_response(url="http://h:9/health", status_code=500, is_reusable=True)
    state = {}
    budget = RestartBudget(3, 3600)
    for _ in range(3):
        budget.record("m1")  # budget already spent

    for _ in range(3):
        await watchdog.check_once(s, app_state, state, budget)

    assert calls["restarts"] == [], "an engine dying on startup must not loop forever"
    assert "status:failed" in calls["order"]


# --- orphaned GPU holders ---------------------------------------------------
# Moved to tests/unit/runtime/test_engine_watchdog.py (#218). The five tests
# that lived here called reap_orphan_gpu_holders/wait_for_gpu_release with a
# bare set of live pids; both now take an EngineOwnership, because attributing
# a GPU process by a 'vllm' substring in its cmdline was killing other tenants'
# processes. The replacements cover the same behaviours against the new
# signature, plus the stranger cases the substring match used to fail.

# --- restore after a warden restart -----------------------------------------
# The engine is a child of the warden process, so a warden restart takes the
# engine with it. Boot reconciliation marks the row failed and nothing brought it
# back: a routine redeploy on 2026-08-17 left the model unloaded and every
# request 404'd until a human loaded it by hand.


@dataclass
class FailedRow:
    id: str
    status: str
    last_error: str | None
    # Recovery keys on THIS, not on last_error. Boot reconciliation sets it to
    # whichever status was interrupted; the crash path sets it to 'loaded'.
    prior_status: str | None = None


def _restore_env(monkeypatch, rows):
    from app.runtime import watchdog

    restarted = []

    class FakeRepo:
        def __init__(self, db):
            pass

        async def list_all(self):
            return rows

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    async def fake_restart(settings, app_state, model_id, overrides, *, recheck=None):
        restarted.append(model_id)
        return "loaded"

    monkeypatch.setattr(watchdog, "ModelRepo", FakeRepo)
    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _NullDB())
    monkeypatch.setattr(watchdog, "_restart", fake_restart)
    return watchdog, restarted


@pytest.mark.asyncio
async def test_restores_a_model_that_was_loaded(monkeypatch, tmp_path):
    wd, restarted = _restore_env(
        monkeypatch,
        [
            FailedRow(
                "m1",
                "failed",
                "process not running after restart (was loaded)",
                prior_status="loaded",
            ),
        ],
    )
    out = await wd.restore_after_warden_restart(FakeSettings(data_dir=tmp_path), None)
    assert restarted == ["m1"]
    assert out == ["m1:loaded"]


@pytest.mark.asyncio
async def test_does_not_restore_a_model_that_was_only_pulling(monkeypatch, tmp_path):
    """A mid-pull model must not be started; only what was actually serving."""
    wd, restarted = _restore_env(
        monkeypatch,
        [
            FailedRow(
                "m2",
                "failed",
                "process not running after restart (was pulling)",
                prior_status="pulling",
            ),
        ],
    )
    await wd.restore_after_warden_restart(FakeSettings(data_dir=tmp_path), None)
    assert restarted == []


@pytest.mark.asyncio
async def test_does_not_restore_a_genuine_failure(monkeypatch, tmp_path):
    """A model that failed on its own merits stays down for a human to look at."""
    wd, restarted = _restore_env(
        monkeypatch,
        [
            FailedRow("m3", "failed", "health timeout; subprocess still holding GPUs"),
            FailedRow("m4", "pulled", None),
            # A model that crashed mid-load via on_exit: it never proved it could
            # serve, so it carries NO prior_status and is a config problem for a
            # human rather than something to retry in a loop. (A 'loading' flag
            # would mean something different -- see
            # test_wants_restart_accepts_a_model_interrupted_while_restarting.)
            FailedRow("m5", "failed", "engine exited (rc=1)", prior_status=None),
        ],
    )
    await wd.restore_after_warden_restart(FakeSettings(data_dir=tmp_path), None)
    assert restarted == []


# --- live settings ----------------------------------------------------------
# The knobs live in the settings KV and are re-read every tick, so an operator's
# change in the Settings UI applies on the next pass rather than at the next
# warden restart.


@dataclass
class LiveSettings:
    data_dir: Path
    watchdog_enabled: bool = True
    watchdog_interval_s: float = 30.0
    watchdog_failure_threshold: int = 3
    watchdog_max_restarts: int = 3
    watchdog_restart_window_s: float = 3600.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db.sqlite"


def _kv_env(monkeypatch, kv):
    from app.runtime import watchdog

    class FakeSettingsRepo:
        def __init__(self, db):
            pass

        async def get_many(self, keys):
            return {k: v for k, v in kv.items() if k in keys}

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(watchdog, "SettingsRepo", FakeSettingsRepo)
    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _NullDB())
    return watchdog


@pytest.mark.asyncio
async def test_live_config_defaults_to_env_when_kv_is_empty(monkeypatch, tmp_path):
    wd = _kv_env(monkeypatch, {})
    cfg = await wd.live_config(LiveSettings(data_dir=tmp_path))
    assert cfg == {
        "enabled": True,
        "restore_on_boot": True,
        "interval_s": 30.0,
        "failure_threshold": 3,
        "max_restarts": 3,
    }


@pytest.mark.asyncio
async def test_kv_overrides_env(monkeypatch, tmp_path):
    wd = _kv_env(
        monkeypatch,
        {
            "watchdog_enabled": "0",
            "watchdog_restore_on_boot": "0",
            "watchdog_interval_s": "60",
            "watchdog_failure_threshold": "5",
            "watchdog_max_restarts": "0",
        },
    )
    cfg = await wd.live_config(LiveSettings(data_dir=tmp_path))
    assert cfg["enabled"] is False
    assert cfg["restore_on_boot"] is False
    assert cfg["interval_s"] == 60
    assert cfg["failure_threshold"] == 5
    assert cfg["max_restarts"] == 0, "0 = detect and report, never auto-restart"


@pytest.mark.asyncio
async def test_a_corrupt_kv_value_falls_back_instead_of_crashing(monkeypatch, tmp_path):
    """A bad row must not take the watchdog down — that would be worse than the
    failure it guards against."""
    wd = _kv_env(monkeypatch, {"watchdog_failure_threshold": "not-a-number"})
    cfg = await wd.live_config(LiveSettings(data_dir=tmp_path))
    assert cfg["failure_threshold"] == 3


def test_budget_reconfigure_keeps_history(monkeypatch):
    b = RestartBudget(max_restarts=3, window_s=3600)
    b.record("m", now=0)
    b.reconfigure(1)
    assert not b.allow("m", now=1), "tightening the limit must apply to past crashes"


def test_settings_registry_exposes_every_watchdog_key():
    """The UI renders from RUNTIME_KEYS; a key missing there is invisible and
    a key missing a coercer 500s on PATCH."""
    from app.runtime.watchdog import WATCHDOG_KEYS
    from app.settings.routes_api import _COERCERS, RUNTIME_KEYS

    for key in WATCHDOG_KEYS:
        assert key in RUNTIME_KEYS, f"{key} not exposed in settings"
        assert key in _COERCERS, f"{key} has no coercer"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
@pytest.mark.asyncio
async def test_a_loading_model_is_not_probed(wd_env, httpx_mock):
    """A reload takes minutes. If the row is not 'loading' during it, the watchdog
    probes the half-started engine, fails, and restarts the load it just began —
    a loop bounded only by the crash-loop guard. Observed in production."""
    watchdog, s, app_state, calls = wd_env
    httpx_mock.add_response(url="http://h:9/health", status_code=500, is_reusable=True)

    class LoadingRepo:
        def __init__(self, db):
            pass

        async def list_all(self):
            return [FailedRow("m1", "loading", None)]

        async def get(self, mid):
            return FailedRow("m1", "loading", None)

        async def update_status(self, mid, status, last_error=None):
            calls["order"].append(f"status:{status}")

    monkeypatch_target = watchdog
    original = monkeypatch_target.ModelRepo
    monkeypatch_target.ModelRepo = LoadingRepo
    try:
        await watchdog.check_once(s, app_state, {}, RestartBudget(3, 3600))
    finally:
        monkeypatch_target.ModelRepo = original

    assert calls["restarts"] == [], "a model mid-load must never be restarted"


def test_watchdog_restart_marks_the_row_loading():
    """The route marks 'loading' before calling start_engine; the watchdog calls
    start_engine directly, so it must do the same or the row reads 'loaded' for
    the whole reload.

    Deliberately NOT inside start_engine: an extra write at the top of the shared
    path serializes against SQLite's 30s busy_timeout and delays every spawn —
    measured as a 7-test regression in the load endpoint."""
    import inspect

    from app.models.routes_api import start_engine
    from app.runtime.watchdog import _restart

    assert 'update_status(model_id, "loading")' in inspect.getsource(_restart)
    assert 'update_status(model_id, "loading")' not in inspect.getsource(
        start_engine
    ), "the shared load path must stay free of an entry-time DB write"


# --- recovery keyed on prior_status, not on an error string -----------------


def _row(**kw):
    """Minimal duck-typed row for wants_restart()."""
    from types import SimpleNamespace

    base = {"id": "m1", "status": "failed", "prior_status": "loaded", "last_error": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_wants_restart_accepts_a_crashed_serving_model():
    from app.runtime.watchdog import wants_restart

    assert wants_restart(_row()) is True


def test_wants_restart_survives_a_diagnosis_in_last_error():
    """The regression that cost 1h43m of downtime on 2026-08-18.

    Recovery used to compare last_error against an exact sentinel. The crash
    path writes a DIAGNOSIS there instead, so the sentinel never appeared and
    nothing restarted the model. prior_status must be immune to whatever
    last_error happens to say.
    """
    from app.runtime.watchdog import wants_restart

    diagnosed = _row(last_error="This model requires trust_remote_code to load. (rc=0)")
    assert wants_restart(diagnosed) is True


def test_wants_restart_ignores_a_model_that_never_served():
    """A model that failed on its own merits is left for a human.

    on_exit sets prior_status ONLY for rows that were 'loaded', so a failure
    during a user-initiated load carries no flag at all and is never retried.
    That -- not the status value -- is what keeps a bad config from becoming a
    crash loop.
    """
    from app.runtime.watchdog import wants_restart

    assert wants_restart(_row(prior_status=None)) is False


def test_wants_restart_accepts_a_model_interrupted_while_restarting():
    """prior_status='loading' can only be written by boot reconciliation, i.e.
    "the warden restarted mid-load". Excluding it made a model interrupted
    during its own auto-restart permanently unrecoverable -- prod stayed dead
    11.5h that way on 2026-08-18/19."""
    from app.runtime.watchdog import wants_restart

    assert wants_restart(_row(prior_status="loading")) is True


def test_wants_restart_ignores_pulls_and_deliberate_unloads():
    """A half-finished pull is not a serving model, and an unload is the
    operator's stated intent."""
    from app.runtime.watchdog import wants_restart

    assert wants_restart(_row(prior_status="pulling")) is False
    assert wants_restart(_row(prior_status="unloading")) is False


def test_wants_restart_ignores_a_deliberately_unloaded_model():
    """An operator who unloads a model must not have it resurrected."""
    from app.runtime.watchdog import wants_restart

    assert wants_restart(_row(status="pulled", prior_status=None)) is False
    assert wants_restart(_row(status="loaded", prior_status=None)) is False


# --- the exit-path sweep: restart engines that died with their wrapper ------


def _sweep_env(monkeypatch, tmp_path, rows):
    from app.runtime import watchdog

    calls = {"restarts": [], "captured": [], "cleared": []}

    class FakeRepo:
        def __init__(self, db):
            pass

        async def list_all(self):
            return rows

        async def set_prior_status(self, mid, prior):
            calls["cleared"].append((mid, prior))

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    async def fake_restart(settings, app_state, model_id, overrides, *, recheck=None):
        calls["restarts"].append(model_id)
        return "loaded"

    def fake_capture(settings, model_id, *, report):
        calls["captured"].append((model_id, report))
        return None

    monkeypatch.setattr(watchdog, "ModelRepo", FakeRepo)
    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _NullDB())
    monkeypatch.setattr(watchdog, "_restart", fake_restart)
    monkeypatch.setattr(watchdog, "capture_evidence", fake_capture)
    monkeypatch.setattr(watchdog, "prune_crash_dirs", lambda *a, **k: None)
    return watchdog, calls


@pytest.mark.asyncio
async def test_sweep_restarts_an_engine_that_died_while_serving(monkeypatch, tmp_path):
    """The 2026-08-18 gap: check_once only probes rows the DB calls 'loaded',
    so an engine whose wrapper exited (row already 'failed') was invisible to
    every recovery path and stayed dead until a human noticed."""
    rows = [FailedRow("m1", "failed", "vllm subprocess exited (rc=0)", prior_status="loaded")]
    wd, calls = _sweep_env(monkeypatch, tmp_path, rows)
    out = await wd.restart_crashed_models(
        FakeSettings(data_dir=tmp_path), None, RestartBudget(3, 3600)
    )
    assert calls["restarts"] == ["m1"]
    assert out == ["m1:loaded"]


@pytest.mark.asyncio
async def test_sweep_captures_evidence_before_restarting(monkeypatch, tmp_path):
    """The restart overwrites the live engine log with a fresh startup, so the
    crash scene has to be copied first. The exit path skipped this entirely,
    which is why /data/crashes was empty after a real crash."""
    rows = [FailedRow("m1", "failed", "boom", prior_status="loaded")]
    wd, calls = _sweep_env(monkeypatch, tmp_path, rows)
    await wd.restart_crashed_models(FakeSettings(data_dir=tmp_path), None, RestartBudget(3, 3600))
    assert [m for m, _ in calls["captured"]] == ["m1"]
    assert calls["captured"][0][1]["last_error"] == "boom"


@pytest.mark.asyncio
async def test_sweep_respects_the_restart_budget(monkeypatch, tmp_path):
    """A crash loop must not be restarted forever, and once the budget is spent
    the flag is cleared so the sweep stops retrying it every single tick."""
    rows = [FailedRow("m1", "failed", "boom", prior_status="loaded")]
    wd, calls = _sweep_env(monkeypatch, tmp_path, rows)
    budget = RestartBudget(max_restarts=1, window_s=3600)
    settings = FakeSettings(data_dir=tmp_path)

    first = await wd.restart_crashed_models(settings, None, budget)
    second = await wd.restart_crashed_models(settings, None, budget)

    assert first == ["m1:loaded"]
    assert second == ["m1:budget-exhausted"]
    assert calls["restarts"] == ["m1"], "the 2nd pass must not spawn an engine"
    assert calls["cleared"] == [("m1", None)], "the flag must be retired"


@pytest.mark.asyncio
async def test_sweep_ignores_rows_it_must_not_touch(monkeypatch, tmp_path):
    rows = [
        FailedRow("healthy", "loaded", None),
        FailedRow("unloaded-by-hand", "pulled", None),
        FailedRow("bad-config", "failed", "boom", prior_status=None),
        FailedRow("half-pulled", "failed", "boom", prior_status="pulling"),
        FailedRow("genuine-failure", "failed", "OOM"),
    ]
    wd, calls = _sweep_env(monkeypatch, tmp_path, rows)
    out = await wd.restart_crashed_models(
        FakeSettings(data_dir=tmp_path), None, RestartBudget(3, 3600)
    )
    assert calls["restarts"] == [] and out == []


# --- a failed respawn must not strand the row in 'loading' ------------------


@pytest.mark.asyncio
async def test_restart_repairs_state_when_the_spawn_fails(monkeypatch, tmp_path):
    """The 11.5-hour outage of 2026-08-18/19.

    _restart marks the row 'loading' before calling start_engine. If the spawn
    throws, nothing else ever resets it: on_exit never fires (no process was
    supervised), and no recovery path inspects 'loading'. The row must go back
    to 'failed' AND keep its restart flag so the next sweep retries it under
    the budget.
    """
    from app.runtime import watchdog

    writes = []

    class FakeRepo:
        def __init__(self, db):
            pass

        async def get(self, mid):
            from types import SimpleNamespace

            return SimpleNamespace(id=mid, prior_status="loaded")

        async def update_status(self, mid, status, last_error=None):
            writes.append(("status", status, last_error))

        async def set_prior_status(self, mid, prior):
            writes.append(("prior", prior, None))

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    class FakeAlloc:
        def __init__(self):
            self.released = []

        def allocate(self):
            return 10000

        def release(self, p):
            self.released.append(p)

    async def boom(*a, **k):
        raise RuntimeError("no GPUs free")

    monkeypatch.setattr(watchdog, "ModelRepo", FakeRepo)
    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _NullDB())
    monkeypatch.setattr(watchdog, "reap_orphan_gpu_holders", lambda pids: [])
    monkeypatch.setattr("app.models.routes_api.start_engine", boom)

    alloc = FakeAlloc()
    app_state = type("S", (), {"supervisor": FakeSup(), "port_allocator": alloc})()

    with pytest.raises(RuntimeError):
        await watchdog._restart(FakeSettings(data_dir=tmp_path), app_state, "m1", None)

    assert ("status", "loading", None) in writes, "should have marked loading first"
    failed = [w for w in writes if w[0] == "status" and w[1] == "failed"]
    assert failed, "a failed spawn must not leave the row in 'loading'"
    assert "no GPUs free" in (failed[-1][2] or ""), "the spawn error must survive"
    assert ("prior", "loaded", None) in writes, (
        "the restart flag must be restored, or the row silently leaves the "
        "recoverable set and never retries"
    )
    assert alloc.released == [10000], "the port must not leak on a failed spawn"


@pytest.mark.asyncio
async def test_restart_keeps_the_restart_flag_lit_while_the_spawn_is_in_flight(
    monkeypatch, tmp_path
):
    """#236 -- the SUCCESS path must put prior_status straight back after its
    'loading' write, not only in the except branch the test above drives.

    ``ModelRepo.update_status('loading')`` clears prior_status by design, so
    without the re-assert the row spends the whole spawn as 'loading' with no
    flag -- indistinguishable from an operator's first-ever load, which boot
    reconciliation now demotes to 'pulled'. A warden restart mid-restore would
    then quietly park a model that WAS serving instead of restoring it.

    Asserted from INSIDE start_engine, against a real ModelRepo on a real DB:
    a fake repo would have to mirror exactly what update_status clears, and
    the moment it drifted this test would pass for the wrong reason.
    """
    from app.db.database import open_db
    from app.db.migrations import apply_migrations
    from app.db.repos.models import ModelRepo, ModelRow
    from app.runtime import watchdog

    settings = FakeSettings(data_dir=tmp_path)
    async with open_db(settings.db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id="m1",
                served_model_name="m1",
                hf_repo="org/m",
                hf_revision="main",
                gpu_indices=[0],
                tensor_parallel_size=1,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=[],
                extra_env={},
                status="failed",
                pulled_bytes=0,
                pulled_total=None,
                last_error="process not running after restart (was loaded)",
            )
        )
        # What mark_runtime_dead_on_startup / on_exit leave behind for the sweep.
        await ModelRepo(db).set_prior_status("m1", "loaded")

    seen = {}

    async def fake_start_engine(
        settings, sup, port_alloc, model, port, overrides=None, on_spawn=None
    ):
        async with open_db(settings.db_path) as db:
            row = await ModelRepo(db).get("m1")
        seen["during_spawn"] = (row.status, row.prior_status)
        # The shared load path's success write.
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status("m1", "loaded")

    class FakeAlloc:
        def allocate(self):
            return 10000

        def release(self, p):
            pass

    monkeypatch.setattr(watchdog, "reap_orphan_gpu_holders", lambda pids: [])
    monkeypatch.setattr("app.models.routes_api.start_engine", fake_start_engine)
    app_state = type("S", (), {"supervisor": FakeSup(), "port_allocator": FakeAlloc()})()

    out = await watchdog._restart(settings, app_state, "m1", None)

    assert seen["during_spawn"] == ("loading", "loaded"), (
        "while start_engine is in flight the row must read 'loading' AND still carry "
        "prior_status='loaded'; without the flag a warden restart mid-restore demotes "
        f"it to 'pulled' and auto-restore is silently off (saw {seen['during_spawn']})"
    )
    assert out == "loaded"
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get("m1")
    assert (
        row is not None and row.prior_status is None
    ), "reaching 'loaded' retires the flag -- it must be lit only for the attempt"
