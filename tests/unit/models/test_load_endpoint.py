import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

from app.runtime.port_alloc import PortAllocator
from tests.conftest import csrf_header, jwt_login, seed_admin_user, wait_until


@dataclass
class _FakeGpuLive:
    index: int
    memory_total_mib: int = 16376


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


def _stub_probe(client, indices):
    """Install a probe cache reporting exactly ``indices`` as present.

    The load pre-flight now fails closed (#175): a configured GPU absent
    from the probe — or no probe at all — 422s. Tests that exercise the
    happy load path must therefore stub a probe that confirms their
    configured GPUs are present.
    """
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[_FakeGpuLive(index=i) for i in indices])
    )


def _seed_done_with_pulled_model(db_path, *, allowed, model_id="qwen", gpus=None):
    """Seed setup-done admin + a pre-pulled model with given gpu_indices.

    #55 fix — admin user + setup_state via shared barrier-aware helper.
    """
    seed_admin_user(db_path, allowed_gpu_indices=allowed)
    if gpus is not None:
        with sqlite3.connect(db_path) as db:
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status, "
                "pulled_bytes, pulled_total, last_error) "
                "VALUES (?, ?, 'o/r', 'main', ?, ?, NULL, NULL, 0.9, 0, '[]', 'pulled', 0, NULL, NULL)",
                (model_id, model_id, json.dumps(gpus), len(gpus)),
            )
            db.commit()


def _jwt_login(client, username="admin", password="hunter2"):
    # #55 fix — delegate to the barrier+retry helper.
    return jwt_login(client, username=username, password=password)


def test_load_validates_gpus_against_allowed_set(tmp_data_dir, client):
    """If gpu_indices ⊄ allowed_gpu_indices, must return 422."""
    client.get("/healthz")
    _seed_done_with_pulled_model(tmp_data_dir / "vllm-warden.db", allowed=[1, 2, 3], gpus=[0, 1])
    auth = _jwt_login(client)
    r = client.post("/api/models/qwen/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    assert "allowed" in r.json()["detail"].lower()


def test_load_calls_supervisor_then_health_check(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done_with_pulled_model(tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1])
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post("/api/models/qwen/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202
    # supervisor.load is invoked from a background task, so allow a short window
    wait_until(lambda: sup_load.await_count >= 1, what="Supervisor.load to be invoked")
    assert sup_load.await_count == 1


def test_unload_returns_404_for_unknown_model(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done_with_pulled_model(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    r = client.post("/api/models/missing/unload", headers={**auth, **csrf_header(client)})
    assert r.status_code == 404


class _RecordingPortAllocator(PortAllocator):
    """The app's own allocator plus a record of every ``release`` call, so a
    test can assert the port came back through the public API instead of
    peeking at the free list."""

    def __init__(self) -> None:
        super().__init__(start=10000, end=10999)
        self.released: list[int] = []

    def release(self, port: int) -> None:
        self.released.append(port)
        super().release(port)


def _drive_engine_crash(client, db_path, auth, *, model_id, rc, serving):
    """Load ``model_id`` through the route, fire ``on_exit(rc)`` from INSIDE
    the supervisor stub -- on the app's own event loop -- and snapshot every
    side effect ``on_exit`` owns at the moment it returns.

    Two things about the shape are deliberate:

    * ``on_exit`` runs where the real supervisor would run it. The previous
      version of this test captured the callback and drove it from the test
      thread with ``asyncio.get_event_loop().run_until_complete(...)``, which
      raises ``RuntimeError: There is no current event loop`` the moment any
      earlier ``asyncio.run()`` has touched the main thread's policy (the
      session-scoped ``migrated_db_template`` fixture does exactly that).
      Same pattern as ``test_load_crash_reports_diagnosed_last_error``.
    * The DB and allocator are read inside the stub, not from the test
      thread. Once ``sup.load`` returns the runner keeps going (health wait,
      then its own ``failed`` write on the same row), so a read from outside
      would race those writes. A ``threading.Event`` hands the snapshot back;
      the wait for it is a bounded ``wait_until`` (R20) budgeted at the 30 s
      ``busy_timeout`` every aiosqlite connection sets (app/db/database.py),
      because the ``on_exit`` closure's write can legitimately hold the DB's
      lock that long under xdist load (R30).

    ``serving=True`` flips the row to 'loaded' first, as the driver does once
    the engine is up; ``False`` lands the crash on a row still 'loading' -- an
    engine that never came up.
    """
    alloc = _RecordingPortAllocator()
    client.app.state.port_allocator = alloc
    seen: dict = {}
    fired = threading.Event()
    runner_done = threading.Event()

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        with sqlite3.connect(db_path) as db:
            if serving:
                db.execute("UPDATE models SET status='loaded' WHERE id=?", (model_id,))
            # What a live engine leaves in model_runtime; on_exit must clear it.
            db.execute(
                "INSERT INTO model_runtime(model_id, pid, port) VALUES (?, 4242, ?)",
                (model_id, port),
            )
            db.commit()
        seen["port"] = port
        seen["released_before"] = list(alloc.released)
        try:
            await on_exit(rc)
        finally:
            with sqlite3.connect(db_path) as db:
                seen["row"] = db.execute(
                    "SELECT status, last_error, prior_status FROM models WHERE id=?",
                    (model_id,),
                ).fetchone()
                (seen["runtime_rows"],) = db.execute(
                    "SELECT COUNT(*) FROM model_runtime WHERE model_id=?", (model_id,)
                ).fetchone()
            seen["released_after"] = list(alloc.released)
            fired.set()

    async def fake_health(**kwargs):
        # The engine is gone: a real health wait would sit on a dead socket for
        # load_timeout_s. Report the miss and let the runner finish.
        runner_done.set()
        return False

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=fake_load),
        patch("app.models.routes_api.wait_for_health", new=fake_health),
    ):
        r = client.post(f"/api/models/{model_id}/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        # R30: the bare 10 s Event.wait flaked because the on_exit closure's
        # aiosqlite write can hold the DB's lock up to the 30 s busy_timeout
        # (app/db/database.py) under xdist load; the budget matches that.
        # Which connection holds the lock is not identified (open question).
        wait_until(
            fired.is_set,
            timeout_s=30,
            what="on_exit to fire inside the supervisor stub",
        )
        wait_until(
            runner_done.is_set,
            timeout_s=30,
            what="the load runner to reach its health wait",
        )
    return seen


def test_on_exit_callback_flips_status_to_failed_and_releases_port(tmp_data_dir, client):
    """When the supervisor fires ``on_exit`` for an engine that WAS serving,
    the closure in ``start_engine`` owns three side effects, and each one is
    load-bearing for recovery:

    * ``status='failed'`` with the rc in ``last_error`` (the operator's view);
    * ``prior_status='loaded'`` -- the machine-readable "was serving" flag
      ``watchdog.wants_restart`` keys on. The 2026-08-18 11.5-hour outage was
      this write missing in effect: on_exit wrote a diagnosed last_error, the
      old string sentinel was displaced, and nothing ever restarted the model;
    * the ``model_runtime`` row cleared and the port handed back, or the next
      load finds the port taken and the UI keeps showing a dead pid.

    Until now only ``wants_restart(prior_status=...)`` was tested -- the
    consumer -- which passes whether or not anything ever sets the flag.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="crash-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    seen = _drive_engine_crash(client, db_path, auth, model_id="crash-model", rc=139, serving=True)

    status, last_error, prior_status = seen["row"]
    assert status == "failed", f"expected failed, got {status!r}"
    assert "rc=139" in (last_error or ""), f"rc must reach last_error, got {last_error!r}"
    assert prior_status == "loaded", (
        "on_exit must record that the row WAS serving, or wants_restart() never "
        "picks it up and the engine stays dead until a human loads it"
    )
    assert seen["runtime_rows"] == 0, "model_runtime must not keep advertising a dead pid"
    assert seen["released_before"] == [], "sanity: nothing released before on_exit"
    assert seen["released_after"] == [
        seen["port"]
    ], "on_exit must hand the port back to the allocator"


def test_on_exit_during_a_first_load_does_not_flag_the_row_for_restart(tmp_data_dir, client):
    """The other half of the guard in ``on_exit``: a row that was still
    'loading' when the engine died never came up, so it earns no automatic
    restart -- retrying a bad config is a crash loop with a nicer name. The
    row still fails with the rc, the runtime row still goes, and the port
    still comes back."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="never-up")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    seen = _drive_engine_crash(client, db_path, auth, model_id="never-up", rc=1, serving=False)

    status, last_error, prior_status = seen["row"]
    assert status == "failed"
    assert "rc=1" in (last_error or "")
    assert (
        prior_status is None
    ), "a load that never reached 'loaded' must not be marked as having served"
    assert seen["runtime_rows"] == 0
    assert seen["released_after"] == [seen["port"]]


def test_load_writes_health_ok_after_warmup_probe_succeeds(tmp_data_dir, client):
    """#29 — when wait_for_health succeeds AND the warmup probe succeeds,
    the runner must persist ``health_ok=True`` into ``model_runtime`` so
    callers (status badges, /stats, future telemetry) can distinguish
    'process up but never proved serving' from 'process up and served at
    least one request'. Previously the column was never written and stuck
    at its default 0.
    """
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 1],
        model_id="health-ok-model",
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # Inject a fake process record so the runner's upsert(pid=...)
        # path has a PID to read. The real Supervisor.load does this;
        # AsyncMock would not, leaving sup._handles empty.
        class _FakeProc:
            pid = 1234

        self._handles[model.id] = _FakeProc()
        self._ports[model.id] = port

    sup_mark_warming = AsyncMock()
    sup_mark_ready = AsyncMock()
    health = AsyncMock(return_value=True)
    probe = AsyncMock(return_value=ProbeResult(ok=True, detail=None))

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=fake_load),
        patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming),
        patch("app.runtime.supervisor.Supervisor.mark_ready", new=sup_mark_ready),
        patch("app.models.routes_api.wait_for_health", new=health),
        patch("app.models.routes_api.warmup_probe", new=probe),
    ):
        r = client.post(
            "/api/models/health-ok-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    def _runtime_row():
        if probe.await_count < 1:
            return None
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute(
                "SELECT health_ok, last_health_at "
                "FROM model_runtime WHERE model_id = 'health-ok-model'"
            ).fetchone()
        return row if row and row[0] == 1 else None

    row = wait_until(_runtime_row, what="model_runtime to record health_ok=1")
    assert row[0] == 1, f"expected health_ok=1, got {row[0]}"
    assert row[1] is not None, "expected last_health_at ISO timestamp"
    # Sanity: timestamp is in ISO-8601 with a 'T' separator (datetime.isoformat)
    assert "T" in row[1], f"expected ISO timestamp, got {row[1]!r}"


def test_load_runs_warmup_probe_before_flipping_to_loaded(tmp_data_dir, client):
    """Status must remain 'loading' until the warmup probe succeeds, not
    just when /health returns 200. Regression for 2026-05-20 Qwen3-VL
    crash loop."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 1],
        model_id="probe-model",
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult

    sup_load = AsyncMock()
    sup_mark_warming = AsyncMock()
    sup_mark_ready = AsyncMock()
    health = AsyncMock(return_value=True)
    probe = AsyncMock(return_value=ProbeResult(ok=True, detail=None))

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming),
        patch("app.runtime.supervisor.Supervisor.mark_ready", new=sup_mark_ready),
        patch("app.models.routes_api.wait_for_health", new=health),
        patch("app.models.routes_api.warmup_probe", new=probe),
    ):
        r = client.post(
            "/api/models/probe-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    wait_until(lambda: probe.await_count >= 1, what="the warmup probe to run")
    assert probe.await_count == 1
    sup_mark_warming.assert_awaited_once()
    sup_mark_ready.assert_awaited_once()


def test_load_probe_failure_marks_failed_without_unloading(tmp_data_dir, client):
    """When the warmup probe fails, the model row goes to 'failed' but
    the subprocess is left running (no SIGTERM). Operator must
    force-unload to release GPUs."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 1],
        model_id="probe-fail-model",
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    sup_mark_warming = AsyncMock()
    health = AsyncMock(return_value=True)
    probe = AsyncMock(return_value=ProbeResult(ok=False, detail="HTTP 503"))

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming),
        patch("app.models.routes_api.wait_for_health", new=health),
        patch("app.models.routes_api.warmup_probe", new=probe),
    ):
        r = client.post(
            "/api/models/probe-fail-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    # await_count increments before the runner's DB write completes, so wait
    # on the DB state, not the count.
    def _failed_row():
        if probe.await_count < 1:
            return None
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id = 'probe-fail-model'"
            ).fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(_failed_row, what="the model row to reach 'failed' with the probe detail")
    assert row[0] == "failed"
    assert "503" in row[1]
    sup_unload.assert_not_awaited()


def test_load_health_timeout_marks_failed_without_unloading(tmp_data_dir, client):
    """When wait_for_health times out, the model row goes to 'failed'
    but the subprocess is left running. Removes the legacy auto-SIGTERM
    that was the original race trigger."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 1],
        model_id="health-timeout-model",
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    health = AsyncMock(return_value=False)

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        # #282 — a genuine still-stuck timeout (vs. the engine having died,
        # which on_exit alone reports) is now distinguished by is_running();
        # this test is about the former, so the subprocess must read as alive.
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post(
            "/api/models/health-timeout-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    # await_count increments before the runner's DB write completes, so wait
    # on the DB state, not the count.
    def _failed_row():
        if health.await_count < 1:
            return None
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id = 'health-timeout-model'"
            ).fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(_failed_row, what="the model row to reach 'failed' on the health timeout")
    assert row[0] == "failed"
    assert "health timeout" in row[1]
    sup_unload.assert_not_awaited()


def test_load_wires_liveness_aware_hard_timeout_into_wait_for_health(tmp_data_dir, client):
    """#282 — start_engine must pass an ``alive`` probe and an extended
    ``hard_timeout_s`` to wait_for_health, not just the soft load_timeout_s.
    Without this wiring a slow-but-live load (e.g. NVFP4/Blackwell FlashInfer
    autotuning past 600s, #275) is judged by the soft deadline alone again."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 1],
        model_id="wiring-model",
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post(
            "/api/models/wiring-model/load",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202
        wait_until(lambda: health.await_count >= 1 or None, what="wait_for_health to be awaited")

        # Evaluate alive() while the Supervisor.is_running patch is still
        # active -- the closure reads it lazily, so calling it after the
        # `with` block exits would hit the real (unpatched) method instead.
        _, kwargs = health.call_args
        assert callable(kwargs["alive"])
        assert kwargs["alive"]() is True
        assert kwargs["hard_timeout_s"] == pytest.approx(3.0 * kwargs["timeout_s"])
        assert callable(kwargs["on_soft_timeout"])


def test_load_health_timeout_unrecognised_log_appends_bounded_raw_tail(tmp_data_dir, client):
    """#274 — a health timeout whose engine log matches NO known failure
    pattern (a genuine stall, e.g. hung capturing CUDA graphs) used to leave
    last_error as the bare generic string. A bounded slice of the raw
    engine-log tail must be appended, clearly separated, so the stall is
    self-diagnosing from last_error alone."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="stall-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    # A log matching NONE of diagnose_engine_log's patterns: chatty progress
    # lines ending in a distinctive capture line. The early marker sits far
    # enough back that the bounded tail must cut it off.
    logs_dir = tmp_data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    early = "EARLY-MARKER-XYZ-000"
    filler = "\n".join(
        f"INFO loading weights shard {i:03d} of 200 padding-padding-padding" for i in range(120)
    )
    distinct = "Capturing CUDA graph shapes:  45%|####---- | 450/1000 [00:42<00:51, 10.7it/s]"
    (logs_dir / "stall-model.log").write_text(early + "\n" + filler + "\n" + distinct + "\n")

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    health = AsyncMock(return_value=False)

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post(
            "/api/models/stall-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    def _failed_row():
        if health.await_count < 1:
            return None
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id = 'stall-model'"
            ).fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(
        _failed_row, what="the stall-model row to reach 'failed' on the health timeout"
    )
    _, last_error = row
    # The generic message is still there...
    assert "health timeout; subprocess still holding GPUs — force-unload to release" in last_error
    # ...and the unrecognised tail is appended, so the stall is self-diagnosing.
    assert distinct in last_error
    # Bounded: the early part of the log is cut off by the tail limit.
    assert early not in last_error
    sup_unload.assert_not_awaited()


def test_load_health_timeout_recognised_log_keeps_pattern_diagnosis_only(tmp_data_dir, client):
    """#274 no-regression — the SAME health-timeout path with a log tail that
    DOES match a known pattern must still produce exactly today's
    pattern-diagnosed message, with no appended raw tail."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="kv-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    logs_dir = tmp_data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "kv-model.log").write_text(
        "RAW-TAIL-SENTINEL-999 must never appear in a pattern-matched diagnosis\n"
        "ValueError: To serve at least one request with the models's max seq "
        "len (262144), (16.0 GiB KV cache is needed, which is larger than the "
        "available KV cache memory (9.6 GiB). Based on the available memory, "
        "the estimated maximum model length is 157216. Try increasing "
        "`gpu_memory_utilization` or decreasing `max_model_len`.\n"
    )

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    health = AsyncMock(return_value=False)

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post(
            "/api/models/kv-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    def _failed_row():
        if health.await_count < 1:
            return None
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id = 'kv-model'"
            ).fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(_failed_row, what="the kv-model row to reach 'failed' on the health timeout")
    _, last_error = row
    # The pattern diagnosis replaced the generic string, exactly as today...
    assert "health timeout" not in last_error
    assert "157216" in last_error
    assert "max_model_len" in last_error
    # ...with the force-unload suffix, and NO raw tail appended.
    assert last_error.endswith("(subprocess still holding GPUs — force-unload to release)")
    assert "RAW-TAIL-SENTINEL-999" not in last_error
    sup_unload.assert_not_awaited()


def test_unload_returns_409_when_supervisor_refuses(tmp_data_dir, client):
    """When the supervisor raises UnloadRefused, the route returns 409
    with the current state in the body."""
    from app.runtime.supervisor import ModelState, UnloadRefused

    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1],
        gpus=[0],
        model_id="refused-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='refused-model'")
        db.commit()
    auth = _jwt_login(client)

    # #166 — the refusal is now surfaced synchronously via the fast
    # ``ensure_unloadable`` pre-flight check (the slow teardown moved to a
    # background task), so the 409 originates there rather than from ``unload``.
    sup_check = AsyncMock(side_effect=UnloadRefused("refused-model", ModelState.WARMING))
    with patch("app.runtime.supervisor.Supervisor.ensure_unloadable", new=sup_check):
        r = client.post(
            "/api/models/refused-model/unload",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 409
    assert "WARMING" in r.json()["detail"]


_GIB = 1024**3

# A config whose model-max (262144) blows the KV budget on a single A4000 —
# mirrors the tencent/Hy-MT2-1.8B footgun the preflight exists to catch.
_HUGE_CTX_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}
# Inputs that make decide_preflight return cap/block for the huge-ctx config.
_A4000_VRAM = 16 * _GIB
_WEIGHTS_2B = 4 * _GIB


def _set_explicit_max_model_len(db_path, model_id, value):
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE models SET max_model_len = ? WHERE id = ?", (value, model_id))
        db.commit()


def test_load_blocks_explicit_max_model_len_that_wont_fit(tmp_data_dir, client):
    """FEATURE 1, Case B: an EXPLICIT max_model_len that won't fit must block
    with HTTP 422 ``wont_fit`` BEFORE any engine is spawned."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="toobig")
    # #175 physical-presence pre-flight fails closed on an empty probe; confirm
    # the seeded GPU is present so the flow reaches the KV-budget preflight.
    _stub_probe(client, [0])
    _set_explicit_max_model_len(db_path, "toobig", 262144)
    auth = _jwt_login(client)

    async def fake_inputs(settings, model, request):
        return _HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B

    sup_load = AsyncMock()
    with (
        patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs),
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
    ):
        r = client.post("/api/models/toobig/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error_code"] == "wont_fit"
    assert "recommended_max_model_len" in detail
    assert detail["breakdown"]["total_vram"] == _A4000_VRAM
    # Engine must NOT have been spawned, and the row must be re-loadable.
    sup_load.assert_not_awaited()
    with sqlite3.connect(db_path) as db:
        status = db.execute("SELECT status FROM models WHERE id='toobig'").fetchone()[0]
    assert status == "pulled"


def test_load_auto_caps_null_max_model_len(tmp_data_dir, client):
    """FEATURE 1, Case A: a NULL max_model_len whose model-max won't fit is
    auto-capped — sup.load gets overrides={'max_model_len': <cap>} and the 202
    body carries ``context_capped``."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(
        db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="capme"
    )  # max_model_len stays NULL from the seed
    # #175 physical-presence pre-flight fails closed on an empty probe; confirm
    # the seeded GPU is present so the flow reaches the KV-budget preflight.
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    async def fake_inputs(settings, model, request):
        return _HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B

    captured = {}

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        captured["overrides"] = overrides

    health = AsyncMock(return_value=True)
    with (
        patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs),
        patch("app.runtime.supervisor.Supervisor.load", new=fake_load),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post("/api/models/capme/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202
    cc = r.json()["context_capped"]
    assert cc["from"] == 262144
    assert cc["to"] < 262144
    assert cc["reason"] == "kv_cache_exceeds_vram"

    wait_until(lambda: "overrides" in captured, what="Supervisor.load to receive the overrides")
    assert captured.get("overrides") == {"max_model_len": cc["to"]}


def test_load_crash_reports_diagnosed_last_error(tmp_data_dir, client):
    """FEATURE 2: when the engine crashes, the runner reads the engine-log tail
    and reports the parsed, actionable message as ``last_error`` (not the bare
    generic text)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="crashy")
    # #175 physical-presence pre-flight fails closed on an empty probe; confirm
    # the seeded GPU is present so the flow reaches the engine-spawn/crash path.
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    # Seed a fake engine log that the diagnostics parser will recognise as a
    # KV-cache overflow (real observed string, including vLLM's own fit estimate).
    logs_dir = tmp_data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "crashy.log").write_text(
        "ValueError: To serve at least one request with the models's max seq "
        "len (262144), (16.0 GiB KV cache is needed, which is larger than the "
        "available KV cache memory (9.6 GiB). Based on the available memory, "
        "the estimated maximum model length is 157216. Try increasing "
        "`gpu_memory_utilization` or decreasing `max_model_len`.\n"
    )

    async def fake_inputs(settings, model, request):
        # Fail open so the preflight does not pre-empt the crash path.
        return {}, 0, 0

    captured_on_exit = []

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        captured_on_exit.append(on_exit)
        # Flip to loaded as the real driver would, then fire on_exit(rc=1).
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE models SET status='loaded' WHERE id='crashy'")
            db.commit()
        if on_exit is not None:
            await on_exit(1)

    with (
        patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs),
        patch("app.runtime.supervisor.Supervisor.load", new=fake_load),
    ):
        r = client.post("/api/models/crashy/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202

    def _crash_row():
        with sqlite3.connect(db_path) as db:
            row = db.execute("SELECT status, last_error FROM models WHERE id='crashy'").fetchone()
        return row if row and row[0] == "failed" and row[1] else None

    row = wait_until(
        _crash_row, what="the crashy row to reach 'failed' with a diagnosed last_error"
    )
    assert row[0] == "failed"
    # The diagnosed message (with vLLM's estimate) replaced the generic text...
    assert "157216" in row[1]
    assert "max_model_len" in row[1]
    # ...and the rc is still appended for log correlation.
    assert "rc=1" in row[1]


def test_unload_force_query_param_bypasses_refusal(tmp_data_dir, client):
    """?force=true must call sup.unload with force=True and succeed even
    when the supervisor would otherwise refuse."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1],
        gpus=[0],
        model_id="force-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='force-model'")
        db.commit()
    auth = _jwt_login(client)

    captured = {}

    async def fake_unload(self, model_id, *, force=False):
        captured["force"] = force

    with patch("app.runtime.supervisor.Supervisor.unload", new=fake_unload):
        r = client.post(
            "/api/models/force-model/unload?force=true",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202
    # #166 — teardown now runs in a background task, so the force flag is
    # observed asynchronously; wait for it rather than asserting inline.
    wait_until(
        lambda: "force" in captured,
        what="the background unload to run with the force flag",
    )
    assert captured["force"] is True


def test_unload_returns_202_immediately_even_when_teardown_is_slow(tmp_data_dir, client):
    """#166 regression — the unload route must NOT block the HTTP response on
    the (potentially many-second) engine teardown. A slow ``sup.unload`` used
    to keep the request open until the client/proxy disconnected, at which
    point the CSRF ``BaseHTTPMiddleware`` raised Starlette's
    ``RuntimeError("No response returned.")`` → HTTP 500, stranding the row in
    'unloading'. Teardown must run in a background task; the route returns 202
    promptly and the row reaches the terminal 'pulled' state asynchronously.
    """
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1],
        gpus=[0],
        model_id="slow-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='slow-model'")
        db.commit()
    auth = _jwt_login(client)

    async def slow_unload(self, model_id, *, force=False):
        await asyncio.sleep(1.0)  # simulate a large multi-GPU engine teardown

    with patch("app.runtime.supervisor.Supervisor.unload", new=slow_unload):
        start = time.monotonic()
        r = client.post(
            "/api/models/slow-model/unload",
            headers={**auth, **csrf_header(client)},
        )
        elapsed = time.monotonic() - start
    assert r.status_code == 202
    # The response must come back well before the 1.0s teardown completes.
    assert elapsed < 0.5, f"unload blocked on teardown for {elapsed:.2f}s"

    # The background task must still drive the row to the terminal state.
    def _pulled_row():
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute("SELECT status FROM models WHERE id='slow-model'").fetchone()
        return row if row and row[0] == "pulled" else None

    wait_until(
        _pulled_row,
        what="the row to reach terminal 'pulled' after the slow teardown",
    )


def test_load_422_when_configured_gpu_absent_from_probe(tmp_data_dir, client):
    """A model whose gpu_indices is allow-listed but physically absent must
    422 gpu_index_missing before the row flips to 'loading'."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 2],
        model_id="ghost-gpu",
    )
    # Probe sees only GPU 0 — index 2 is gone (card pulled / re-indexed).
    client.app.state.gpu_probe_cache = _FakeProbeCache(_FakeSnap(gpus=[_FakeGpuLive(index=0)]))
    auth = _jwt_login(client)

    r = client.post("/api/models/ghost-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "gpu_index_missing"
    assert body["detail"]["available"] == [0]
    assert "2" in body["detail"]["message"] or "[2]" in body["detail"]["message"]
    # Row must NOT have advanced to loading.
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        status = db.execute("SELECT status FROM models WHERE id='ghost-gpu'").fetchone()[0]
    assert status == "pulled"


def test_load_passes_preflight_when_all_gpus_present(tmp_data_dir, client):
    """Allow-listed AND present gpu_indices passes the probe pre-flight (202)."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[0, 1],
        model_id="present-gpu",
    )
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[_FakeGpuLive(index=0), _FakeGpuLive(index=1)])
    )
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post("/api/models/present-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202


def test_load_422_when_probe_errored(tmp_data_dir, client):
    """When the probe itself errored there is no ground truth that any
    configured GPU is present, so the pre-flight must FAIL CLOSED (422
    gpu_index_missing) per spec — every configured index is unconfirmed."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1, 2, 3],
        gpus=[2],
        model_id="probe-err-gpu",
    )
    # GPU 2 is configured but the probe reports an error and zero GPUs;
    # with no ground truth the guard must 422 (fail closed).
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[], probe_error="nvidia-smi unavailable")
    )
    auth = _jwt_login(client)
    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post("/api/models/probe-err-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "gpu_index_missing"
    assert body["detail"]["probe_error"] == "nvidia-smi unavailable"
    assert body["detail"]["available"] == []
    # Row must NOT have advanced to loading.
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        status = db.execute("SELECT status FROM models WHERE id='probe-err-gpu'").fetchone()[0]
    assert status == "pulled"


def test_unload_teardown_error_still_lands_terminal(tmp_data_dir, client):
    """#166 — if the engine teardown raises an unexpected error, the row must
    NOT be left stranded in 'unloading'. The background task drives it to a
    terminal state (the engine is gone either way) so the operator never needs
    a control-plane restart to recover.
    """
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db",
        allowed=[0, 1],
        gpus=[0],
        model_id="boom-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='boom-model'")
        db.commit()
    auth = _jwt_login(client)

    async def boom_unload(self, model_id, *, force=False):
        raise RuntimeError("driver blew up mid-teardown")

    with patch("app.runtime.supervisor.Supervisor.unload", new=boom_unload):
        r = client.post(
            "/api/models/boom-model/unload",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    def _terminal_row():
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute("SELECT status FROM models WHERE id='boom-model'").fetchone()
        return row if row and row[0] in ("pulled", "failed") else None

    wait_until(
        _terminal_row,
        what="the row to reach a terminal status after the teardown error",
    )


# --- #236: force-unload must mean force -------------------------------------


def _seed_model_in_status(db_path, model_id, status):
    """Seed setup-done admin + one model parked in ``status``."""
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id=model_id)
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE models SET status=? WHERE id=?", (status, model_id))
        db.commit()


def _wait_for_status(db_path, model_id, wanted):
    def _status():
        with sqlite3.connect(db_path) as db:
            row = db.execute("SELECT status FROM models WHERE id=?", (model_id,)).fetchone()
        return row[0] if row and row[0] in wanted else None

    return wait_until(_status, what=f"model '{model_id}' to reach one of {wanted}")


def test_force_unload_succeeds_from_loading(tmp_data_dir, client):
    """#236 — a warden that died mid-load leaves the row in 'loading' with no
    engine anywhere. Plain unload refuses (correctly); ``?force=true`` is the
    operator's escape hatch and used to refuse too, leaving hand-editing
    SQLite as the ONLY way out.

    The supervisor deliberately holds nothing here: that is exactly the state
    a restart leaves behind, and force must still drive the row terminal.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "stranded-model", "loading")
    auth = _jwt_login(client)

    r = client.post(
        "/api/models/stranded-model/unload?force=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 202, r.text
    assert _wait_for_status(db_path, "stranded-model", ("pulled",)) == "pulled"


def test_force_unload_succeeds_from_unloading(tmp_data_dir, client):
    """The other transient status an interrupted teardown can strand."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "half-unloaded", "unloading")
    auth = _jwt_login(client)

    r = client.post(
        "/api/models/half-unloaded/unload?force=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 202, r.text
    assert _wait_for_status(db_path, "half-unloaded", ("pulled",)) == "pulled"


def test_force_unload_kills_what_the_supervisor_still_holds(tmp_data_dir, client):
    """Force is not just a status rewrite: whatever the supervisor holds gets
    torn down, so a load that IS still running releases its GPUs."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "live-load", "loading")
    auth = _jwt_login(client)

    sup_unload = AsyncMock()
    with patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload):
        r = client.post(
            "/api/models/live-load/unload?force=true",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202, r.text
        assert _wait_for_status(db_path, "live-load", ("pulled",)) == "pulled"
    assert sup_unload.await_count == 1
    assert sup_unload.await_args.kwargs["force"] is True


def test_plain_unload_still_refuses_from_loading(tmp_data_dir, client):
    """The guard is unchanged without ``force``: unloading a load that is
    genuinely in flight is still a 409, not a silent kill."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "loading-model", "loading")
    auth = _jwt_login(client)

    r = client.post(
        "/api/models/loading-model/unload",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 409
    assert "loading" in r.json()["detail"]


def test_force_unload_still_refuses_from_pulling(tmp_data_dir, client):
    """No engine is involved in a download, and this route's terminal status
    is 'pulled' — a lie for a half-finished pull. Boot reconciliation owns
    that case."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "pulling-model", "pulling")
    auth = _jwt_login(client)

    r = client.post(
        "/api/models/pulling-model/unload?force=true",
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 409


# --- #275 root cause: a superseded load must never clobber a newer one -----


def test_superseded_load_generation_guard_prevents_clobbering_a_newer_load(tmp_data_dir, client):
    """#275 root cause (owner follow-up, note_92307) — force-unloading a
    model stuck in LOADING does not just kill the engine: the load runner
    (start_engine's wait_for_health + warmup task) used to keep running on
    its own schedule, with no generation check before writing a terminal
    status. Once it eventually gave up, it wrote 'failed' onto whatever row
    now owned model_id -- even a NEWER load that had since force-unloaded the
    first one, reloaded, and reached 'loaded' -- while the new engine kept
    running unaffected (the same end state as #209 Symptom B).

    Supervisor.unload() also cancels the registered load-runner task directly
    (tests/unit/runtime/test_supervisor_unload.py), which is the primary fix
    and would normally stop the first runner well before this race matters.
    This test deliberately no-ops that registration to isolate the SECOND,
    independent guard this fix adds: the load runner must re-check
    ``get_generation()`` before writing 'failed'/'loaded', so even a
    cancellation that did not land in time can never clobber a newer load.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="race-model")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # Mirrors exactly the generation bookkeeping the real Supervisor.load
        # does (app/runtime/supervisor.py), without spawning anything.
        self._generation[model.id] = self._generation.get(model.id, 0) + 1
        return self._generation[model.id]

    release_first = threading.Event()
    calls: list[int] = []

    async def fake_health(*, port, host, timeout_s, **_kwargs):
        idx = len(calls)
        calls.append(idx)
        if idx == 0:
            # The FIRST load's health wait: block (off the event loop) until
            # the second load has already reached 'loaded', then give up --
            # exactly the reported timing (previous load's start + timeout).
            await asyncio.to_thread(release_first.wait, 10.0)
            return False
        return True

    sup_mark_warming = AsyncMock()
    sup_mark_ready = AsyncMock()
    probe = AsyncMock(return_value=ProbeResult(ok=True, detail=None))

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=fake_load),
        patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming),
        patch("app.runtime.supervisor.Supervisor.mark_ready", new=sup_mark_ready),
        patch("app.runtime.supervisor.Supervisor.register_load_runner", new=lambda *a, **k: None),
        patch("app.models.routes_api.wait_for_health", new=fake_health),
        patch("app.models.routes_api.warmup_probe", new=probe),
    ):
        r = client.post("/api/models/race-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202
        wait_until(lambda: len(calls) >= 1, what="the first load to reach wait_for_health")

        r = client.post(
            "/api/models/race-model/unload?force=true",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202
        _wait_for_status(db_path, "race-model", ("pulled",))

        r = client.post("/api/models/race-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202
        _wait_for_status(db_path, "race-model", ("loaded",))

        # The superseded first load finally gives up.
        release_first.set()
        wait_until(lambda: len(calls) >= 2, what="the second load to reach wait_for_health")
        wait_until(lambda: probe.await_count >= 1, what="the second load's warmup probe to run")

        # Give the superseded runner a moment to reach (and be stopped by)
        # its generation check.
        time.sleep(0.3)
        with sqlite3.connect(db_path) as db:
            status, last_error = db.execute(
                "SELECT status, last_error FROM models WHERE id='race-model'"
            ).fetchone()
    assert status == "loaded", (
        f"a superseded load's stale write clobbered the newer load's row "
        f"(status={status!r}, last_error={last_error!r})"
    )


def test_spawn_failure_does_not_clobber_a_newer_load_that_already_loaded(tmp_data_dir, client):
    """#278 slice 2, writer (2): when ``sup.load`` itself raises because another
    writer already owns the model ("model m already running"), the 'failed'
    write must not land on the row a NEWER load has since driven to 'loaded'.
    The guard is the generation this load captured before spawning: if it has
    moved, a newer load owns model_id and the terminal write is dropped."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="m")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    ran = []

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # A concurrent load already won: it owns the generation and drove the
        # row to 'loaded' before our spawn failed to start.
        self._generation[model.id] = self._generation.get(model.id, 0) + 1
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE models SET status='loaded' WHERE id=?", (model.id,))
            db.commit()
        ran.append(1)
        raise RuntimeError("model m already running")

    with patch("app.runtime.supervisor.Supervisor.load", new=fake_load):
        r = client.post("/api/models/m/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202
        wait_until(lambda: len(ran) >= 1, what="the failing spawn to run")
        # Let the guarded except path settle before reading the row.
        time.sleep(0.3)

    with sqlite3.connect(db_path) as db:
        status, last_error = db.execute(
            "SELECT status, last_error FROM models WHERE id='m'"
        ).fetchone()
    assert status == "loaded", (
        f"a spawn that failed because a newer load already owns the model must not "
        f"write 'failed' over that load's row (status={status!r}, last_error={last_error!r})"
    )


# --- #278 slice 1: /load and the watchdog claim the model before touching it -


def _row_state(db_path, model_id):
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT status, prior_status FROM models WHERE id=?", (model_id,)
        ).fetchone()


def test_load_refuses_409_while_the_watchdog_holds_the_claim(tmp_data_dir, client):
    """#278 slice 1 — the watchdog is mid-restart and holds the claim. /load
    must refuse with 409 BEFORE writing anything: the row stays exactly as
    the restart left it, and no engine is spawned for the loser."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "clashed", "failed")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    sup = client.app.state.supervisor
    sup.claim("clashed", "watchdog")

    sup_load = AsyncMock()
    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load):
        r = client.post("/api/models/clashed/load", headers={**auth, **csrf_header(client)})

    assert r.status_code == 409
    assert "watchdog" in r.json()["detail"]
    assert _row_state(db_path, "clashed") == (
        "failed",
        None,
    ), "a refused /load must not touch the row the watchdog owns"
    sup_load.assert_not_awaited()


def test_load_admits_and_settles_the_claim_once_the_restart_releases(tmp_data_dir, client):
    """#278 slice 1 — refused while the watchdog holds the claim, admitted
    after it releases, and the claim is settled when the load runner
    completes (here the spawned load fails its warmup state check, but the
    settlement must not care WHY the runner ended)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "clashed", "failed")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    sup = client.app.state.supervisor
    sup.claim("clashed", "watchdog")

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post("/api/models/clashed/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 409, "still refused while the watchdog holds the claim"
        sup.release_claim("clashed", "watchdog")
        r = client.post("/api/models/clashed/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
    wait_until(lambda: sup_load.await_count >= 1, what="the admitted load to reach Supervisor.load")
    wait_until(
        lambda: "clashed" not in sup._claims,
        what="the claim to settle once the load runner completes",
    )


def test_load_settles_the_claim_when_the_health_check_times_out(tmp_data_dir, client):
    """#278 slice 1 — a load that fails its health check must still release
    the claim, so a stuck engine never wedges the model for the watchdog."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_model_in_status(db_path, "clashed", "failed")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    sup = client.app.state.supervisor

    sup_load = AsyncMock()
    health = AsyncMock(return_value=False)
    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.models.routes_api.wait_for_health", new=health),
    ):
        r = client.post("/api/models/clashed/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        wait_until(lambda: sup_load.await_count >= 1, what="the load to reach Supervisor.load")
    wait_until(
        lambda: "clashed" not in sup._claims,
        what="the claim to settle once the health-timeout runner completes",
    )


def test_load_releases_the_claim_when_preflight_rejects_the_model(tmp_data_dir, client):
    """#278 slice 1 — a synchronous 422 (wont_fit) between the claim and the
    spawn must release the claim, or the model would be wedged for the
    watchdog with nothing in flight."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="clashbig")
    _stub_probe(client, [0])
    _set_explicit_max_model_len(db_path, "clashbig", 262144)
    auth = _jwt_login(client)

    async def fake_inputs(settings, model, request):
        return _HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B

    sup = client.app.state.supervisor
    sup_load = AsyncMock()
    with (
        patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs),
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
    ):
        r = client.post("/api/models/clashbig/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "wont_fit"
    sup_load.assert_not_awaited()
    assert (
        "clashbig" not in sup._claims
    ), "a wont_fit 422 must not leave the claim held for a load that never spawned"
