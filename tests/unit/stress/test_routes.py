"""The stress endpoints, tested hermetically (design §7.1, §7.4, §7.5).

Everything here is seeded straight into SQLite and served through the real
FastAPI app with a stubbed ``app.state``. Nothing touches a GPU, a subprocess
or the runner: ``app.state.stress_start_run`` is the seam the route calls
instead of ``app.stress.runner.start_run``, so these tests keep passing while
the runner is still being written and fail if the ADMISSION rules regress,
which is what they are for.

The fingerprint is never hard-coded. Every test that needs a matching run row
asks ``GET /capabilities`` for the current fingerprint first and seeds with
that. A literal would pass while agreeing with nothing.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import pytest

from app.db.repos.tokens import hash_token
from app.stress.routes_api import COOLDOWN_ENV_VAR, STRESS_DISCLAIMER
from tests.conftest import csrf_header, jwt_login, seed_admin_user

API_TOKEN = "vw_validtoken1234567890abcdef12345"


# ---------------------------------------------------------------------------
# Fakes for the two things a fingerprint would otherwise shell out for
# ---------------------------------------------------------------------------


@dataclass
class _FakeGpuLive:
    index: int
    uuid: str
    name: str = "NVIDIA RTX A4000"
    memory_total_mib: int = 16376
    memory_used_mib: int = 0
    memory_free_mib: int = 16376
    utilization_pct: int = 0
    compute_cap: float | None = 8.6


@dataclass
class _FakeSnap:
    gpus: list = field(default_factory=list)
    apps: list = field(default_factory=list)
    probe_error: str | None = None


class _FakeProbeCache:
    def __init__(self, snap):
        self._snap = snap

    async def get(self):
        return self._snap


def _stub_state(client, *, indices=(0,), probe_error=None):
    """No nvidia-smi, no driver lookup — both are pure inputs to the hash."""
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(
            gpus=[_FakeGpuLive(index=i, uuid=f"GPU-{i:08d}") for i in indices],
            probe_error=probe_error,
        )
    )
    client.app.state.stress_driver_version = "550.54.15"
    client.app.state.stress_driver_version_resolved = True


class _FakeRunner:
    """Stands in for ``app.stress.runner.start_run``."""

    def __init__(self, run_id="run-new"):
        self.run_id = run_id
        self.calls: list[dict] = []

    async def __call__(self, app_state, model_id, *, mode, force):
        self.calls.append({"model_id": model_id, "mode": mode, "force": force})
        return self.run_id


def _install_runner(client, run_id="run-new") -> _FakeRunner:
    runner = _FakeRunner(run_id)
    client.app.state.stress_start_run = runner
    return runner


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _seed(db_path, *, status="loaded", model_id="qwen", gpus=(0,), max_model_len=32768):
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1, 2, 3])
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?,?,?,?,?)",
            ("tok1", "test", API_TOKEN[:8], hash_token(API_TOKEN), "inference"),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES (?,?,'org/repo','main',?,1,'auto',?,0.9,0,'[]',?,0,NULL,NULL)",
            (model_id, model_id, json.dumps(list(gpus)), max_model_len, status),
        )
        db.commit()


def _seed_run(
    db_path,
    *,
    run_id,
    fingerprint,
    model_id="qwen",
    status="completed",
    limits=None,
    recommended_config=None,
    non_monotone=0,
    traffic_observed=0,
    started_at=None,
):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO stress_runs(id, model_id, fingerprint, mode, status, "
            "non_monotone, traffic_observed, limits, recommended_config, seed, "
            "probe_suite_hash, algorithm_version, started_at, finished_at) "
            "VALUES (?,?,?,'conservative',?,?,?,?,?,7,'suite-v1',1,"
            "COALESCE(?, datetime('now')), datetime('now'))",
            (
                run_id,
                model_id,
                fingerprint,
                status,
                non_monotone,
                traffic_observed,
                json.dumps(limits) if limits is not None else None,
                json.dumps(recommended_config) if recommended_config is not None else None,
                started_at,
            ),
        )
        db.commit()


_MEASURED_LIMITS = {
    "quality_context": {
        "value": 122880,
        "unit": "tokens",
        "confidence": "confirmed",
        "limited_by": "quality",
        "gate_tripped": "abrupt",
        "raw_confirmed": 136533,
        "first_observed_failure": 147456,
        "oracle": {
            "n_consecutive": 3,
            "contour_p": 0.21,
            "confirm_m": 5,
            "safety_factor": 0.9,
        },
    },
    "recommended_concurrency": {"value": 24, "unit": "requests", "limited_by": "preemption"},
}


def _fingerprint(client, headers, model_id="qwen") -> str:
    r = client.get(f"/api/models/{model_id}/capabilities", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["fingerprint"]


def _post(client, headers, body, model_id="qwen"):
    return client.post(
        f"/api/models/{model_id}/stress",
        json=body,
        headers={**headers, **csrf_header(client)},
    )


@pytest.fixture
def env(tmp_data_dir, client):
    """A loaded model, an admin JWT, a stubbed probe and a stubbed runner."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path)
    _stub_state(client)
    runner = _install_runner(client)
    headers = jwt_login(client)
    return {"db": db_path, "headers": headers, "runner": runner, "client": client}


# ---------------------------------------------------------------------------
# Authorization — the hole v1 left open
# ---------------------------------------------------------------------------


def test_an_api_token_cannot_start_a_run(env):
    """A `vw_` token is a data-plane credential; a run crashes the engine."""
    r = env["client"].post(
        "/api/models/qwen/stress",
        json={"acknowledge_disruption": True},
        headers={"Authorization": f"Bearer {API_TOKEN}", **csrf_header(env["client"])},
    )
    assert r.status_code == 403
    assert r.json()["detail"]["error_code"] == "operator_only"
    assert env["runner"].calls == []


def test_an_api_token_cannot_read_capabilities(env):
    r = env["client"].get(
        "/api/models/qwen/capabilities",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    assert r.status_code == 403


def test_no_credential_at_all_is_rejected(env):
    r = env["client"].post(
        "/api/models/qwen/stress",
        json={"acknowledge_disruption": True},
        headers=csrf_header(env["client"]),
    )
    assert r.status_code == 401
    assert env["runner"].calls == []


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_a_run_is_refused_without_an_acknowledgement(env):
    r = _post(env["client"], env["headers"], {})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error_code"] == "disruption_not_acknowledged"
    assert detail["disclaimer"] == STRESS_DISCLAIMER
    assert env["runner"].calls == []


def test_a_run_is_refused_when_the_model_is_not_loaded(tmp_data_dir, client):
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db", status="pulled")
    _stub_state(client)
    runner = _install_runner(client)
    headers = jwt_login(client)
    r = _post(client, headers, {"acknowledge_disruption": True})
    assert r.status_code == 409
    assert r.json()["detail"]["error_code"] == "model_not_loaded"
    assert runner.calls == []


def test_an_unknown_model_is_a_404(env):
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True}, model_id="nope")
    assert r.status_code == 404


def test_a_second_run_for_the_same_model_is_refused(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-live", fingerprint=fp, status="running")
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 409
    assert r.json()["detail"]["error_code"] == "run_already_active"
    assert r.json()["detail"]["run_id"] == "run-live"
    assert env["runner"].calls == []


def test_a_held_lease_refuses_a_new_run(env):
    env["client"].app.state.stress_leases.take("qwen", run_id="run-live", gpu_uuids=["GPU-00000000"])
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 409
    assert r.json()["detail"]["error_code"] == "run_already_active"


def test_a_run_on_a_shared_gpu_is_refused(env):
    """Two runs sharing a card each attribute the other's pressure to itself."""
    env["client"].app.state.stress_leases.take(
        "someone-else", run_id="run-other", gpu_uuids=["GPU-00000000"]
    )
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 409
    assert r.json()["detail"]["error_code"] == "gpu_set_busy"
    assert env["runner"].calls == []


def test_a_lease_on_a_different_gpu_does_not_block(tmp_data_dir, client):
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db", gpus=(0,))
    _stub_state(client, indices=(0, 1))
    runner = _install_runner(client)
    headers = jwt_login(client)
    client.app.state.stress_leases.take("other", run_id="r", gpu_uuids=["GPU-00000001"])
    r = _post(client, headers, {"acknowledge_disruption": True})
    assert r.status_code == 202
    assert runner.calls == [{"model_id": "qwen", "mode": "conservative", "force": False}]


def test_a_run_starts_and_returns_202(env):
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True, "mode": "quick"})
    assert r.status_code == 202
    body = r.json()
    assert body["run_id"] == "run-new"
    assert body["mode"] == "quick"
    assert body["reused"] is False
    assert body["estimate_minutes"] > 0
    assert body["disclaimer"] == STRESS_DISCLAIMER
    assert env["runner"].calls == [{"model_id": "qwen", "mode": "quick", "force": False}]


def test_an_unknown_mode_is_rejected(env):
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True, "mode": "brutal"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Cooldown — nothing else stops a run per minute
# ---------------------------------------------------------------------------


def test_a_fresh_measurement_is_returned_instead_of_re_running(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-old", fingerprint=fp, limits=_MEASURED_LIMITS)
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 200
    body = r.json()
    assert body["reused"] is True
    assert body["run_id"] == "run-old"
    assert body["cooldown_expires_at"]
    assert env["runner"].calls == []


def test_force_overrides_the_cooldown(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-old", fingerprint=fp, limits=_MEASURED_LIMITS)
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True, "force": True})
    assert r.status_code == 202
    assert env["runner"].calls == [{"model_id": "qwen", "mode": "conservative", "force": True}]


def test_an_expired_cooldown_starts_a_new_run(env, monkeypatch):
    monkeypatch.setenv(COOLDOWN_ENV_VAR, "60")
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"],
        run_id="run-old",
        fingerprint=fp,
        limits=_MEASURED_LIMITS,
        started_at="2020-01-01 00:00:00",
    )
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 202
    assert env["runner"].calls


def test_a_run_under_a_different_fingerprint_does_not_trigger_the_cooldown(env):
    _seed_run(env["db"], run_id="run-old", fingerprint="sha256:other", limits=_MEASURED_LIMITS)
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 202


def test_a_tainted_run_does_not_trigger_the_cooldown(env):
    """A traffic-tainted run is recorded and never published — nor re-used."""
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"],
        run_id="run-tainted",
        fingerprint=fp,
        limits=_MEASURED_LIMITS,
        traffic_observed=1,
    )
    r = _post(env["client"], env["headers"], {"acknowledge_disruption": True})
    assert r.status_code == 202


# ---------------------------------------------------------------------------
# GET /capabilities
# ---------------------------------------------------------------------------


def test_capabilities_reports_nothing_measured_before_a_run(env):
    r = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["provenance"] == "none"
    assert body["limits"] == {}
    assert body["recommended_config"] is None
    assert body["declared"]["max_model_len"] == 32768


def test_capabilities_publishes_a_matching_measurement_with_full_provenance(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"],
        run_id="run-1",
        fingerprint=fp,
        limits=_MEASURED_LIMITS,
        recommended_config={"max_model_len": 98304, "fingerprint": "sha256:other"},
    )
    body = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"]).json()
    assert body["provenance"] == "measured"
    assert body["stale_reason"] is None

    ctx = body["limits"]["quality_context"]
    assert ctx["value"] == 122880
    assert ctx["provenance"] == "measured"
    assert ctx["confidence"] == "confirmed"
    assert ctx["limited_by"] == "quality"
    assert ctx["raw_confirmed"] == 136533
    assert ctx["first_observed_failure"] == 147456
    assert ctx["oracle"] == {
        "n_consecutive": 3,
        "contour_p": 0.21,
        "confirm_m": 5,
        "safety_factor": 0.9,
    }
    assert ctx["run_id"] == "run-1"
    assert ctx["fingerprint"] == fp
    # Advisory, and only ever here.
    assert body["recommended_config"]["max_model_len"] == 98304


def test_every_limit_is_an_object_even_when_the_run_recorded_a_scalar(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-1", fingerprint=fp, limits={"quality_context": 4096})
    body = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"]).json()
    ctx = body["limits"]["quality_context"]
    assert ctx["value"] == 4096
    # Absent oracle parameters read as null; they are never invented.
    assert ctx["oracle"] == {
        "n_consecutive": None,
        "contour_p": None,
        "confirm_m": None,
        "safety_factor": None,
    }
    assert ctx["confidence"] is None


def test_client_effective_concurrency_is_bounded_by_the_proxy(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-1", fingerprint=fp, limits=_MEASURED_LIMITS)
    body = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"]).json()
    # The run probes the engine directly; PriorityScheduler admits 16 per engine.
    assert body["client_effective_concurrency"] == 16


def test_run_history_is_returned_including_unpublishable_runs(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-bad", fingerprint=fp, non_monotone=1)
    body = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"]).json()
    ids = [r["id"] for r in body["runs"]]
    assert ids == ["run-bad"]
    assert body["runs"][0]["publishable"] is False
    assert body["provenance"] == "none"


# ---------------------------------------------------------------------------
# Staleness — the rule that must not fall back to `declared`
# ---------------------------------------------------------------------------


def test_a_measurement_from_another_configuration_is_stale_not_discarded(env):
    _seed_run(env["db"], run_id="run-1", fingerprint="sha256:elsewhere", limits=_MEASURED_LIMITS)
    body = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"]).json()
    assert body["provenance"] == "stale"
    assert body["stale_reason"]
    ctx = body["limits"]["quality_context"]
    assert ctx["provenance"] == "stale"
    assert ctx["stale_reason"]
    # The measured number survives as a ceiling. Falling back to `declared`
    # would silently WIDEN the advertised context back to the value the
    # measurement corrected downward.
    assert ctx["value"] <= 122880
    assert ctx["value"] != 131072


def test_a_stale_ceiling_is_clamped_down_to_the_declared_context(tmp_data_dir, client):
    """A 96k measurement must not be advertised while the engine runs at 8k."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path, max_model_len=8192)
    _stub_state(client)
    headers = jwt_login(client)
    _seed_run(db_path, run_id="run-1", fingerprint="sha256:elsewhere", limits=_MEASURED_LIMITS)
    body = client.get("/api/models/qwen/capabilities", headers=headers).json()
    ctx = body["limits"]["quality_context"]
    assert ctx["value"] == 8192
    assert ctx["clamped_to_declared"] is True


def test_a_stale_value_is_never_raised_to_the_declared_value(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path, max_model_len=131072)
    _stub_state(client)
    headers = jwt_login(client)
    _seed_run(db_path, run_id="run-1", fingerprint="sha256:elsewhere", limits=_MEASURED_LIMITS)
    body = client.get("/api/models/qwen/capabilities", headers=headers).json()
    ctx = body["limits"]["quality_context"]
    assert ctx["value"] == 122880
    assert ctx["clamped_to_declared"] is False


def test_an_interrupted_run_publishes_nothing(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"], run_id="run-1", fingerprint=fp, status="interrupted", limits=_MEASURED_LIMITS
    )
    body = env["client"].get("/api/models/qwen/capabilities", headers=env["headers"]).json()
    assert body["provenance"] == "none"
    assert body["limits"] == {}


# ---------------------------------------------------------------------------
# GET /v1/models — additive, and never advisory
# ---------------------------------------------------------------------------


def _v1_models(client):
    return client.get("/v1/models", headers={"Authorization": f"Bearer {API_TOKEN}"})


def test_v1_models_keeps_its_openai_shape(env):
    r = _v1_models(env["client"])
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert entry["id"] == "qwen"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "vllm-warden"


def test_v1_models_is_untouched_for_a_model_that_was_never_tested(env):
    entry = _v1_models(env["client"]).json()["data"][0]
    assert "warden" not in entry


def test_v1_models_publishes_a_measured_limit(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-1", fingerprint=fp, limits=_MEASURED_LIMITS)
    entry = _v1_models(env["client"]).json()["data"][0]
    warden = entry["warden"]
    assert warden["provenance"] == "measured"
    assert warden["stale"] is False
    assert warden["limits"]["quality_context"]["value"] == 122880
    assert warden["run_id"] == "run-1"
    assert warden["client_effective_concurrency"] == 16


def test_v1_models_never_publishes_recommended_config(env):
    """It names a configuration this engine is NOT running (§6.2)."""
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"],
        run_id="run-1",
        fingerprint=fp,
        limits=_MEASURED_LIMITS,
        recommended_config={"max_model_len": 98304},
    )
    payload = _v1_models(env["client"]).json()
    assert "recommended_config" not in json.dumps(payload)
    assert "98304" not in json.dumps(payload)


def test_v1_models_flags_a_stale_measurement_rather_than_dropping_it(env):
    _seed_run(env["db"], run_id="run-1", fingerprint="sha256:elsewhere", limits=_MEASURED_LIMITS)
    warden = _v1_models(env["client"]).json()["data"][0]["warden"]
    assert warden["provenance"] == "stale"
    assert warden["stale"] is True
    assert warden["stale_reason"]
    assert warden["limits"]["quality_context"]["value"] == 32768  # clamped, never widened


def test_v1_models_publishes_nothing_from_a_tainted_run(env):
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"],
        run_id="run-1",
        fingerprint=fp,
        limits=_MEASURED_LIMITS,
        traffic_observed=1,
    )
    warden = _v1_models(env["client"]).json()["data"][0]["warden"]
    assert warden["provenance"] == "none"
    assert warden["limits"] == {}


# ---------------------------------------------------------------------------
# Boot wiring
# ---------------------------------------------------------------------------


def test_boot_creates_an_empty_lease_registry(env):
    leases = env["client"].app.state.stress_leases
    assert leases is not None
    assert leases.is_held("qwen") is False


def test_boot_marks_a_running_row_interrupted(tmp_data_dir):
    """A run dies with the warden; a row left `running` blocks the cooldown."""
    from fastapi.testclient import TestClient

    from app.main import build_app

    db_path = tmp_data_dir / "vllm-warden.db"
    with TestClient(build_app()) as first:
        first.get("/healthz")
        _seed(db_path)
        _seed_run(db_path, run_id="run-dead", fingerprint="sha256:x", status="running")

    with TestClient(build_app()) as second:
        second.get("/healthz")
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT status, last_error, finished_at FROM stress_runs WHERE id='run-dead'"
            ).fetchone()
    assert row[0] == "interrupted"
    assert "warden restarted" in row[1]
    assert row[2] is not None


# ---------------------------------------------------------------------------
# Fingerprint stability — the property every rule above depends on
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_stable_across_calls(env):
    a = _fingerprint(env["client"], env["headers"])
    b = _fingerprint(env["client"], env["headers"])
    assert a == b


def test_a_neighbour_loading_onto_the_same_gpu_changes_the_fingerprint(env):
    before = _fingerprint(env["client"], env["headers"])
    with sqlite3.connect(env["db"]) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('nb','nb','o/r','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        db.commit()
    assert _fingerprint(env["client"], env["headers"]) != before


def test_a_missing_card_makes_the_measurement_stale_rather_than_current(tmp_data_dir, client):
    """A probe that cannot see the configured GPU cannot vouch for the hardware."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path, gpus=(0,))
    _stub_state(client, indices=(0,))
    headers = jwt_login(client)
    fp = _fingerprint(client, headers)
    _seed_run(db_path, run_id="run-1", fingerprint=fp, limits=_MEASURED_LIMITS)
    assert client.get("/api/models/qwen/capabilities", headers=headers).json()[
        "provenance"
    ] == "measured"

    _stub_state(client, indices=(), probe_error="nvidia-smi not found")
    body = client.get("/api/models/qwen/capabilities", headers=headers).json()
    assert body["hardware_incomplete"] is True
    assert body["provenance"] == "stale"
    assert "could not confirm" in body["stale_reason"]


def test_capabilities_names_the_run_its_limits_came_from(env):
    """The UI needs to attribute the record to a run, not merely receive it.

    `runs[]` carries only `_run_summary`, which has no limits and no
    recommendation, so the modal reads the TOP-LEVEL record instead. It may
    only do that when the record belongs to the run being shown — the record
    is keyed on the FINGERPRINT, so a run that published nothing leaves an
    earlier run's numbers standing here (see the next test).
    """
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="run-1", fingerprint=fp, limits=_MEASURED_LIMITS)

    body = env["client"].get(
        "/api/models/qwen/capabilities", headers=env["headers"]
    ).json()
    assert body["record_run_id"] == "run-1"
    assert body["limits"], "the record itself must still be published"


def test_capabilities_reports_no_record_run_when_nothing_is_published(env):
    """None, not omitted: the UI branches on it and an absent key would read
    as 'this run's numbers' under `?? undefined` comparison."""
    body = env["client"].get(
        "/api/models/qwen/capabilities", headers=env["headers"]
    ).json()
    assert "record_run_id" in body
    assert body["record_run_id"] is None


def test_a_run_that_measured_nothing_does_not_hold_the_cooldown(env):
    """The retest-blocking bug, 2026-09-04.

    A run that completed having confirmed nothing satisfied every condition
    `current_for` checked, so it held the six-hour cooldown: the POST returned
    `200 reused` naming the dead run, and the UI reveals `force` only on a 409,
    so the button was inert with no way out. A run that learned nothing is the
    strongest reason to allow another.
    """
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(
        env["db"],
        run_id="empty-1",
        fingerprint=fp,
        limits={"quality_context": {"raw_confirmed": 0,
                                    "invalid_reason": "no_confirmed_value"}},
        recommended_config=None,
    )
    r = _post(env["client"], env["headers"],
              {"mode": "quick", "acknowledge_disruption": True})
    assert r.status_code == 202, r.text
    assert r.json()["run_id"] != "empty-1"


def test_a_run_that_measured_something_still_holds_the_cooldown(env):
    """The cooldown itself is not weakened: a real measurement is still reused
    rather than paying for hours of GPU time to reconfirm it."""
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="real-1", fingerprint=fp, limits=_MEASURED_LIMITS)
    r = _post(env["client"], env["headers"],
              {"mode": "quick", "acknowledge_disruption": True})
    assert r.status_code == 200, r.text
    assert r.json()["run_id"] == "real-1"
    assert r.json()["reused"] is True


def test_a_running_run_reports_its_progress(env):
    """A thorough run takes hours; without this the modal has nothing to show.

    Reported with a screenshot on 2026-09-04: "0 probe(s)", "No probes
    recorded yet", and a bar rendered FULL — which reads as finished.
    """
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="live", fingerprint=fp, status="running")
    import json as _json
    import sqlite3

    con = sqlite3.connect(env["db"])
    con.execute(
        "UPDATE stress_runs SET progress = ? WHERE id = 'live'",
        (_json.dumps({"phase": "length", "note": "probing 8,192 tokens",
                      "step": 3, "steps_total": 12, "eta_s": 480.0}),),
    )
    con.commit()
    con.close()

    body = env["client"].get(
        "/api/models/qwen/capabilities", headers=env["headers"]
    ).json()
    live = next(r for r in body["runs"] if r["id"] == "live")
    assert live["progress"]["phase"] == "length"
    assert live["progress"]["note"] == "probing 8,192 tokens"
    assert live["progress"]["eta_s"] == 480.0


def test_apply_refuses_without_the_disruption_acknowledgement(env):
    """Applying reloads the engine, so it takes the same acknowledgement as
    starting a run: refuse loudly rather than surprise someone."""
    r = env["client"].post(
        "/api/models/qwen/stress/apply",
        json={"run_id": "whatever"},
        headers={**env["headers"], **csrf_header(env["client"])},
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error_code"] == "disruption_not_acknowledged"


def test_apply_refuses_a_run_from_a_different_model(env):
    r = env["client"].post(
        "/api/models/qwen/stress/apply",
        json={"run_id": "does-not-exist", "acknowledge_disruption": True},
        headers={**env["headers"], **csrf_header(env["client"])},
    )
    assert r.status_code == 404, r.text


def test_apply_refuses_a_run_with_no_recommendation(env):
    """`choose_recommendation` returns None rather than guessing; that silence
    must not become a setting."""
    fp = _fingerprint(env["client"], env["headers"])
    _seed_run(env["db"], run_id="norec", fingerprint=fp,
              limits=_MEASURED_LIMITS, recommended_config=None)
    r = env["client"].post(
        "/api/models/qwen/stress/apply",
        json={"run_id": "norec", "acknowledge_disruption": True},
        headers={**env["headers"], **csrf_header(env["client"])},
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error_code"] == "no_recommendation"


def test_apply_refuses_a_measurement_from_different_hardware(env):
    """The fingerprint covers GPUs, engine build and co-resident set."""
    _seed_run(env["db"], run_id="stale", fingerprint="sha256:somewhere-else",
              limits=_MEASURED_LIMITS,
              recommended_config={"max_model_len": 98304})
    r = env["client"].post(
        "/api/models/qwen/stress/apply",
        json={"run_id": "stale", "acknowledge_disruption": True},
        headers={**env["headers"], **csrf_header(env["client"])},
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error_code"] == "fingerprint_changed"


# ---------------------------------------------------------------------------
# #265: applying goes through the one row writer
#
# `POST /{id}/stress/apply` wrote `max_model_len` with a raw `UPDATE models` and
# then RESTARTED the engine off what it wrote. `plan_apply` bounds the number,
# but nothing checked that the rest of the row it was about to serve still
# satisfied its own rules -- it was the sixth producer of a ModelRow and the
# only one that also started a process from it.
# ---------------------------------------------------------------------------


def _stub_restart_and_unload(monkeypatch):
    """Stub the two destructive steps and COUNT them.

    ``unload`` is counted as well as ``restart`` because the question the tests
    below ask is not only "was the model left down" but "was it taken down at
    all" -- a refusal that never touches the engine is the fix; a refusal that
    unloads and then restarts is only the fallback.
    """
    from unittest.mock import AsyncMock

    restart = AsyncMock(return_value="loaded")
    unload = AsyncMock(return_value=None)
    monkeypatch.setattr("app.runtime.watchdog._restart", restart)
    monkeypatch.setattr("app.runtime.supervisor.Supervisor.unload", unload)
    return restart, unload


def _apply(env, run_id="rec"):
    return env["client"].post(
        "/api/models/qwen/stress/apply",
        json={"run_id": run_id, "acknowledge_disruption": True},
        headers={**env["headers"], **csrf_header(env["client"])},
    )


def _seed_recommendation(env, max_model_len=98304):
    _seed_run(
        env["db"],
        run_id="rec",
        fingerprint=_fingerprint(env["client"], env["headers"]),
        limits=_MEASURED_LIMITS,
        recommended_config={"max_model_len": max_model_len},
    )


def _status(db_path, model_id="qwen"):
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT status, prior_status, max_model_len FROM models WHERE id = ?",
            (model_id,),
        ).fetchone()


def test_apply_writes_the_measured_context_and_restarts(env, monkeypatch):
    restart, _ = _stub_restart_and_unload(monkeypatch)
    _seed_recommendation(env)

    r = _apply(env)
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == {"max_model_len": 98304}
    with sqlite3.connect(env["db"]) as db:
        assert db.execute(
            "SELECT max_model_len FROM models WHERE id = 'qwen'"
        ).fetchone()[0] == 98304
    assert restart.await_count == 1


def test_apply_refuses_a_row_register_would_refuse_and_does_not_restart(
    env, monkeypatch
):
    """A row carrying a value register never accepted cannot be written and
    restarted: the engine would come back serving a configuration the API would
    refuse to create."""
    restart, unload = _stub_restart_and_unload(monkeypatch)
    # Poisoned BEFORE the run is seeded: extra_env feeds the fingerprint, and a
    # measurement taken under a different one is refused earlier, for an
    # unrelated reason.
    with sqlite3.connect(env["db"]) as db:
        db.execute(
            "UPDATE models SET extra_env = ? WHERE id = 'qwen'",
            (json.dumps({"LD_PRELOAD": "/tmp/evil.so"}),),
        )
        db.commit()
    _seed_recommendation(env)

    r = _apply(env)
    assert r.status_code == 422, r.text
    assert "hard-locked" in r.text.lower()
    with sqlite3.connect(env["db"]) as db:
        assert db.execute(
            "SELECT max_model_len FROM models WHERE id = 'qwen'"
        ).fetchone()[0] == 32768
    assert restart.await_count == 0
    # And -- the point of the fix -- the engine was never touched. The refusal
    # is raised while the model is still serving, so it costs the operator an
    # error message and nothing else.
    assert unload.await_count == 0
    assert _status(env["db"]) == ("loaded", None, 32768)


def test_apply_does_not_take_a_serving_model_down_to_report_a_stale_column(
    env, monkeypatch
):
    """The finding, with the row shape that is live in the field.

    A well-formed `extra_env` key that is simply not on the backend's allowlist
    is the one poisoned shape #266's boot sweep deliberately leaves alone, and
    `filter_extra_env` drops it silently at launch -- so the model loads and
    serves perfectly well and nothing warns anybody. `apply_model_change`
    validates the WHOLE merged row, so it refuses over that column even though
    the request only changes `max_model_len`.

    With the validation after the unload, clicking "Apply and reload" answered
    422 with the engine already torn down, `status` committed as `pulled` and no
    restart -- and the watchdog does not recover a `pulled` row with no
    `prior_status`, because that is what a deliberate unload looks like. The
    model stayed down until a human noticed.
    """
    restart, unload = _stub_restart_and_unload(monkeypatch)
    with sqlite3.connect(env["db"]) as db:
        db.execute(
            "UPDATE models SET extra_env = ? WHERE id = 'qwen'",
            (json.dumps({"FOO_BAR": "1"}),),
        )
        db.commit()
    _seed_recommendation(env)

    r = _apply(env)
    assert r.status_code == 422, r.text
    assert "FOO_BAR" in r.text
    # Still serving, still at its old context, still recoverable by the
    # watchdog if anything else kills it.
    assert _status(env["db"]) == ("loaded", None, 32768)
    assert unload.await_count == 0
    assert restart.await_count == 0


def test_a_refusal_that_lands_after_the_unload_still_restarts(env, monkeypatch):
    """The fallback for the race the dry run cannot close.

    The dry run reads the row, then the engine comes down, then the real write
    re-reads it. Something else may have changed the row in between. If the
    write refuses at that point the engine is already gone, so the route brings
    it back on the configuration still in the row rather than leaving a dead
    model behind a 422. Simulated by poisoning the row DURING the unload.
    """
    from unittest.mock import AsyncMock

    restart = AsyncMock(return_value="loaded")
    monkeypatch.setattr("app.runtime.watchdog._restart", restart)

    async def _poison_then_unload(self, model_id, *, force=False):
        with sqlite3.connect(env["db"]) as db:
            db.execute(
                "UPDATE models SET extra_env = ? WHERE id = 'qwen'",
                (json.dumps({"FOO_BAR": "1"}),),
            )
            db.commit()

    monkeypatch.setattr("app.runtime.supervisor.Supervisor.unload", _poison_then_unload)
    _seed_recommendation(env)

    r = _apply(env)
    assert r.status_code == 422, r.text
    assert "FOO_BAR" in r.text
    # The context was not applied...
    assert _status(env["db"])[2] == 32768
    # ...but the engine was put back rather than left down.
    assert restart.await_count == 1


def test_the_unload_is_recorded_even_if_the_teardown_raises(env, monkeypatch):
    """`Supervisor.unload` releases the handle, the port and the GPU claim in a
    `finally` of its own, so the engine is gone whether it returned or raised.
    The row must say so either way: a row claiming `loaded` with no process
    behind it is the lie this write exists to prevent."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr("app.runtime.watchdog._restart", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        "app.runtime.supervisor.Supervisor.unload",
        AsyncMock(side_effect=RuntimeError("teardown blew up")),
    )
    _seed_recommendation(env)

    with pytest.raises(RuntimeError, match="teardown blew up"):
        _apply(env)
    assert _status(env["db"])[0] == "pulled"
