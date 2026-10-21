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


def test_load_wires_stall_watcher_into_wait_for_health(tmp_data_dir, client):
    """2026-10-18 — start_engine must pass an ``alive`` probe, a ``stop``
    event, and the policy's ``max_s`` (the absolute cap) as the timeout to
    wait_for_health — NOT the old ``hard_timeout_s`` / ``on_soft_timeout``
    pair.  Without this wiring a slow-but-live load (e.g. NVFP4/Blackwell
    autotuning past 600s, #275) is judged by the soft deadline alone again,
    and the stall watcher cannot stop the health wait on a give-up."""
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
        # The new wiring: no hard_timeout_s, no on_soft_timeout.  The
        # timeout is the policy's max_s (absolute cap), and a stop event
        # lets the stall watcher end the wait on a give-up.
        assert "hard_timeout_s" not in kwargs or kwargs["hard_timeout_s"] is None
        assert "on_soft_timeout" not in kwargs or kwargs["on_soft_timeout"] is None
        assert kwargs["timeout_s"] == pytest.approx(client.app.state.settings.load_max_s)
        assert "stop" in kwargs
        # ``stop`` is an asyncio.Event — the watcher sets it on give-up so
        # the health wait returns promptly instead of running out max_s.
        assert hasattr(kwargs["stop"], "is_set")
        assert "path" in kwargs
        assert kwargs["path"] == "/health"


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
    moved, a newer load owns model_id and the terminal write is dropped.

    The stub runs on the app's event loop, so its own DB write goes through
    ``asyncio.to_thread``: a blocking ``sqlite3`` write there stalls the loop,
    and any lifespan writer (the stats sampler's execute-then-commit, the
    ledger, the audit log) holding the write lock across an ``await`` can then
    never reach its commit -- the stub sat in sqlite3's 5 s busy wait, raised
    "database is locked" before recording that it ran, and the test timed out
    under ``-n auto`` (gate-56). The sibling writer below holds that lock on
    purpose, so the interleaving is forced on every run instead of left to
    xdist scheduling."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="m")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    from app.db.database import open_db

    async def _sibling_writer_mid_transaction():
        """A lifespan-style writer that holds the write lock across an await:
        it commits on the loop's first free turn after the stub starts writing,
        and only this loop can run that commit."""
        began = asyncio.Event()
        stub_writing = asyncio.Event()

        async def hold():
            async with open_db(db_path) as db:
                await db.execute("BEGIN IMMEDIATE")
                began.set()
                await stub_writing.wait()
                await db.commit()

        task = asyncio.create_task(hold())
        await began.wait()
        return task, stub_writing

    async def _settle(task):
        done, _ = await asyncio.wait({task}, timeout=30)
        return task in done

    def _mark_loaded(model_id):
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE models SET status='loaded' WHERE id=?", (model_id,))
            db.commit()

    ran = []
    sibling: dict = {}

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # A concurrent load already won: it owns the generation and drove the
        # row to 'loaded' before our spawn failed to start.
        self._generation[model.id] = self._generation.get(model.id, 0) + 1
        sibling["task"], stub_writing = await _sibling_writer_mid_transaction()
        stub_writing.set()
        await asyncio.to_thread(_mark_loaded, model.id)
        ran.append(1)
        raise RuntimeError("model m already running")

    with patch("app.runtime.supervisor.Supervisor.load", new=fake_load):
        r = client.post("/api/models/m/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202
        # The route registered its runner before answering; wait for it (and the
        # sibling writer) to finish instead of sleeping past the guarded except.
        runner = client.app.state.supervisor._load_runners["m"]
        assert client.portal.call(_settle, runner), "the load runner did not finish within 30 s"
        assert client.portal.call(
            _settle, sibling["task"]
        ), "the sibling writer did not commit within 30 s"
    assert ran == [1], "the failing spawn never got past its own DB write"

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


# ---------------------------------------------------------------------------
# Model load-progress watcher (2026-10-18): stall-based give-up
# ---------------------------------------------------------------------------


def test_no_progress_note_written_to_last_error(tmp_data_dir, client):
    """2026-10-18 — a slow fake engine that passes 2× ``load_timeout_s`` must
    NOT write a "still loading after …" progress note into ``last_error``.
    The old ``_note_still_loading`` / ``note_progress`` path is gone; the
    stall watcher replaces it and only writes on give-up."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="slow-model")
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
            "/api/models/slow-model/load",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202
        wait_until(lambda: health.await_count >= 1, what="wait_for_health to be awaited")

    # The load succeeded (health=True).  last_error must be None — no
    # progress note was written during the (simulated) slow load.
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status, last_error FROM models WHERE id='slow-model'").fetchone()
    assert row[0] in ("loaded", "loading")
    assert row[1] is None, f"no progress note should be written to last_error; got {row[1]!r}"


def test_give_up_marks_failed_and_unloads(tmp_data_dir, client):
    """2026-10-18 — a give-up verdict from the stall watcher must write
    ``failed`` with the stall reason and call ``sup.unload(force=True)`` so
    the engine is stopped and the GPUs are released."""
    from app.runtime.load_progress.policy import Verdict

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="stuck-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    health_was_called = {"v": False}
    give_up_fired = {"v": False}

    async def fake_health(**kwargs):
        health_was_called["v"] = True
        # Wait for the watcher's on_give_up to fire before returning —
        # the give-up handler sets the stop event and writes ``failed``.
        # This simulates the real flow: the watcher fires on_give_up,
        # which sets stop, which makes wait_for_health return False.

        deadline = time.monotonic() + 10.0
        while not give_up_fired["v"] and time.monotonic() < deadline:  # noqa: ASYNC110 - polls a fake's state with a deadline
            await asyncio.sleep(0.05)
        return False

    async def fake_watch_load(*, on_give_up, **_kwargs):
        v = Verdict(
            state="give_up",
            stalled_for_s=1200.0,
            give_up_at_s=7200.0,
            reason="no progress for 20 min",
            hint=None,
        )
        await on_give_up(v)
        give_up_fired["v"] = True

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.models.routes_api.wait_for_health", new=fake_health),
        patch("app.models.routes_api.watch_load", new=fake_watch_load),
    ):
        r = client.post(
            "/api/models/stuck-model/load",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202
        # on_give_up runs as a background task: wait for the unload call while
        # the patch is still active, or the handler binds the real method.
        wait_until(lambda: sup_unload.call_count >= 1, what="sup.unload(force=True) to be called")

    def _failed_row():
        if not health_was_called["v"]:
            return None
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id='stuck-model'"
            ).fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(_failed_row, what="the stuck-model row to reach 'failed' on give-up")
    assert row[0] == "failed"
    assert "gave up; engine stopped and GPUs released" in row[1]
    assert "no progress for 20 min" in row[1]
    # unload(force=True) was scheduled by on_give_up to kill the engine and
    # release GPUs.  The unload is scheduled as a separate asyncio task
    # (``asyncio.create_task``) — the mock's coroutine was CREATED (call_count
    # increments when the mock is called, before the task runs), proving the
    # handler reached the unload-scheduling step.  Whether the task actually
    # runs is proven by ``test_give_up_unload_survives_runner_cancellation``.
    # The ``with`` block has exited by now, so ``_Sup.unload`` is the real
    # method again.  But ``sup_unload`` is the same mock object that was
    # patched onto the class during the ``client.post`` call, and its
    # ``call_count`` reflects calls made during that window.
    assert (
        sup_unload.call_count == 1
    ), f"expected sup.unload to be called once, got {sup_unload.call_count}"
    assert sup_unload.call_args.kwargs["force"] is True


def _run_give_up_load(
    tmp_data_dir,
    client,
    *,
    model_id,
    log_text: str | None,
    backend: str | None = None,
    verdict=None,
):
    """POST a load for ``model_id`` whose stall watcher immediately fires a
    give-up verdict, optionally writing ``log_text`` to the model's engine log
    and setting the row's ``backend`` first. Returns (db_path, sup_unload) once
    the row is ``failed``; the caller asserts on ``last_error``."""
    from app.runtime.load_progress.policy import Verdict

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id=model_id)
    if backend is not None:
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE models SET backend = ? WHERE id = ?", (backend, model_id))
            db.commit()
    if log_text is not None:
        logs_dir = tmp_data_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"{model_id}.log").write_text(log_text)
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_unload = AsyncMock()
    give_up_fired = {"v": False}

    async def fake_health(**kwargs):
        deadline = time.monotonic() + 10.0
        while not give_up_fired["v"] and time.monotonic() < deadline:  # noqa: ASYNC110 - polls a fake's state with a deadline
            await asyncio.sleep(0.05)
        return False

    async def fake_watch_load(*, on_give_up, **_kwargs):
        await on_give_up(
            verdict
            or Verdict(
                state="give_up",
                stalled_for_s=1200.0,
                give_up_at_s=7200.0,
                reason="no progress for 20 min",
                hint=None,
            )
        )
        give_up_fired["v"] = True

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=AsyncMock()),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.models.routes_api.wait_for_health", new=fake_health),
        patch("app.models.routes_api.watch_load", new=fake_watch_load),
    ):
        r = client.post(f"/api/models/{model_id}/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202
        wait_until(lambda: sup_unload.call_count >= 1, what="sup.unload(force=True) to be called")

    def _failed_row():
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id = ?", (model_id,)
            ).fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(_failed_row, what=f"the {model_id} row to reach 'failed' on give-up")
    assert row[0] == "failed"
    return row[1], sup_unload


def test_give_up_appends_engine_log_tail(tmp_data_dir, client):
    """2026-10-21 (P5 follow-up) — a stall give-up must write the engine-log
    diagnosis (or a bounded raw tail when no pattern is recognised) into
    last_error, like the health-timeout path, so one last_error read tells the
    operator WHY the engine stalled. The red-first leg: an unrecognised log
    with a distinctive last line -> that line appears in last_error after the
    give-up (RED today: last_error is only the stall reason). Bounding, the
    recognised-pattern arm, and the no-log arm each have their own test
    below; (e) the MLX traceback leg registers MlxBackend for its duration
    because the linux profile does not."""
    early = "EARLY-MARKER-XYZ-000"
    filler = "\n".join(
        f"INFO loading weights shard {i:03d} of 200 padding-padding-padding" for i in range(120)
    )
    distinct = "Capturing CUDA graph shapes:  45%|####---- | 450/1000 [00:42<00:51, 10.7it/s]"
    last_error, _ = _run_give_up_load(
        tmp_data_dir,
        client,
        model_id="tail-model",
        log_text=early + "\n" + filler + "\n" + distinct + "\n",
    )
    # The stall reason + give-up phrasing are still there...
    assert "no progress for 20 min" in last_error
    assert "gave up; engine stopped and GPUs released" in last_error
    # ...and the distinctive last line of the unrecognised log is appended,
    # so the stall is self-diagnosing from last_error alone.
    assert distinct in last_error


def test_give_up_appends_bounded_tail_only(tmp_data_dir, client):
    """Leg (a)+(b): unrecognised log -> the distinctive last line is in
    last_error AND the stall reason is kept, but an early marker 120+ filler
    lines back is cut off by the tail limit (same shape as the #274 test)."""
    early = "EARLY-MARKER-XYZ-000"
    filler = "\n".join(
        f"INFO loading weights shard {i:03d} of 200 padding-padding-padding" for i in range(120)
    )
    distinct = "Capturing CUDA graph shapes:  45%|####---- | 450/1000 [00:42<00:51, 10.7it/s]"
    last_error, _ = _run_give_up_load(
        tmp_data_dir,
        client,
        model_id="tail-model",
        log_text=early + "\n" + filler + "\n" + distinct + "\n",
    )
    # (a) the stall reason + give-up phrasing are still there...
    assert "no progress for 20 min" in last_error
    assert "gave up; engine stopped and GPUs released" in last_error
    # ...and the unrecognised tail is appended, so the stall is self-diagnosing.
    assert distinct in last_error
    assert "--- engine log tail (no recognised failure pattern) ---" in last_error
    # (b) bounded: the early part of the log is cut off by the tail limit.
    assert early not in last_error


def test_give_up_recognised_log_appends_diagnosis_only(tmp_data_dir, client):
    """Leg (c): the SAME give-up path with a log tail that DOES match a known
    pattern produces the pattern diagnosis appended, and the raw-tail header is
    NOT present (the tail is not appended when a pattern already explains it)."""
    log_text = (
        "RAW-TAIL-SENTINEL-999 must never appear in a pattern-matched diagnosis\n"
        "ValueError: To serve at least one request with the models's max seq "
        "len (262144), (16.0 GiB KV cache is needed, which is larger than the "
        "available KV cache memory (9.6 GiB). Based on the available memory, "
        "the estimated maximum model length is 157216. Try increasing "
        "`gpu_memory_utilization` or decreasing `max_model_len`.\n"
    )
    last_error, _ = _run_give_up_load(tmp_data_dir, client, model_id="kv-model", log_text=log_text)
    # The stall reason + give-up phrasing are still there...
    assert "no progress for 20 min" in last_error
    assert "gave up; engine stopped and GPUs released" in last_error
    # ...and the pattern diagnosis was appended, with NO raw tail.
    assert "157216" in last_error
    assert "max_model_len" in last_error
    assert "--- engine log tail (no recognised failure pattern) ---" not in last_error
    assert "RAW-TAIL-SENTINEL-999" not in last_error


def test_give_up_no_log_file_keeps_exact_string(tmp_data_dir, client):
    """Leg (d): no engine log file -> last_error is EXACTLY today's string,
    byte-for-byte (no appended tail, no diagnosis)."""
    last_error, _ = _run_give_up_load(tmp_data_dir, client, model_id="no-log-model", log_text=None)
    assert last_error == "no progress for 20 min — gave up; engine stopped and GPUs released"


def test_give_up_mlx_traceback_appends_mlx_diagnosis(tmp_data_dir, client, monkeypatch):
    """Leg (e): an MLX row's give-up routes its engine log through the MLX
    grammar (the row's backend, not the profile default). The linux profile
    does not register ``mlx``, so the test registers the real MlxBackend for
    its duration; the route itself is unchanged. A Python traceback ending in
    a ValueError -> last_error carries diagnose_mlx_log's message for that
    exact text, and no raw-tail header."""
    from app.runtime.backends import registry
    from app.runtime.backends.mlx import MlxBackend
    from app.runtime.backends.mlx.diagnostics import diagnose_mlx_log

    monkeypatch.setitem(registry._BACKENDS, "mlx", MlxBackend())
    log_text = (
        "INFO mlx_lm.server starting\n"
        "Traceback (most recent call last):\n"
        '  File "/opt/mlx_lm/server.py", line 42, in load\n'
        "    raise ValueError('MLX-LEG-E unsupported model type qwen9')\n"
        "ValueError: MLX-LEG-E unsupported model type qwen9\n"
    )
    expected = diagnose_mlx_log(log_text)
    assert expected is not None
    last_error, _ = _run_give_up_load(
        tmp_data_dir, client, model_id="mlx-model", log_text=log_text, backend="mlx"
    )
    assert last_error == (
        "no progress for 20 min — gave up; engine stopped and GPUs released\n" + expected.message
    )
    assert "--- engine log tail (no recognised failure pattern) ---" not in last_error


def test_fatal_give_up_appends_diagnosis_and_tail(tmp_data_dir, client, monkeypatch):
    """BACKLOG-macos P5 fail-fast: a ``fatal`` give-up (the watcher saw
    mlx_lm.server's own fatal-load line) writes the reason, the MLX diagnosis
    AND the bounded raw tail -- the engine's traceback is the evidence the
    early kill was warranted -- and still force-unloads. Captured M1 log."""
    from app.runtime.backends import registry
    from app.runtime.backends.mlx import MlxBackend
    from app.runtime.backends.mlx.diagnostics import diagnose_mlx_log, fatal_mlx_load_line
    from app.runtime.load_progress.policy import Verdict
    from tests.unit.runtime.backends.mlx_load_failure_logs import (
        CORRUPT_SAFETENSORS_FATAL,
        CORRUPT_SAFETENSORS_LOG,
    )

    monkeypatch.setitem(registry._BACKENDS, "mlx", MlxBackend())
    engine_words = fatal_mlx_load_line(CORRUPT_SAFETENSORS_FATAL)
    reason = f"engine load failed and the engine will not recover ({engine_words})"
    expected = diagnose_mlx_log(CORRUPT_SAFETENSORS_LOG)
    assert expected is not None
    last_error, sup_unload = _run_give_up_load(
        tmp_data_dir,
        client,
        model_id="mlx-corrupt",
        log_text=CORRUPT_SAFETENSORS_LOG,
        backend="mlx",
        verdict=Verdict(
            state="give_up",
            stalled_for_s=0.0,
            give_up_at_s=4.0,
            reason=reason,
            hint=None,
            fatal=True,
        ),
    )
    head = f"{reason} — gave up; engine stopped and GPUs released\n{expected.message}\n"
    assert last_error.startswith(head)
    tail = last_error[len(head) :]
    assert tail.startswith("--- engine log tail ---\n")
    assert "RuntimeError: [load_safetensors] Invalid json header length" in tail
    assert '"GET /health HTTP/1.1" 503' in tail
    assert len(tail) <= len("--- engine log tail ---\n") + 1 + 1500
    assert sup_unload.call_args.kwargs["force"] is True


def test_fatal_give_up_without_log_keeps_exact_string(tmp_data_dir, client):
    """No engine log: a fatal give-up adds nothing after the give-up string."""
    from app.runtime.load_progress.policy import Verdict

    last_error, _ = _run_give_up_load(
        tmp_data_dir,
        client,
        model_id="fatal-no-log",
        log_text=None,
        verdict=Verdict(
            state="give_up",
            stalled_for_s=0.0,
            give_up_at_s=1.0,
            reason="engine load failed and the engine will not recover (x)",
            hint=None,
            fatal=True,
        ),
    )
    assert last_error == (
        "engine load failed and the engine will not recover (x)"
        " — gave up; engine stopped and GPUs released"
    )


def test_health_timeout_path_still_leaves_engine_running(tmp_data_dir, client):
    """2026-10-18 — the existing health-timeout path (engine alive past the
    cap, no give-up verdict) keeps today's behaviour: ``failed`` with
    "force-unload to release", the engine is NOT killed.  The stall watcher
    is the ONLY path that kills the engine."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="ht-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

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
            "/api/models/ht-model/load",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202

    def _failed_row():
        if health.await_count < 1:
            return None
        with sqlite3.connect(db_path) as db:
            row = db.execute("SELECT status, last_error FROM models WHERE id='ht-model'").fetchone()
        return row if row and row[0] == "failed" else None

    row = wait_until(_failed_row, what="the ht-model row to reach 'failed' on health timeout")
    assert row[0] == "failed"
    assert "health timeout" in row[1] or "force-unload" in row[1]
    # The health-timeout path does NOT unload — the engine is left running.
    sup_unload.assert_not_awaited()


def test_force_unload_mid_load_stops_watcher_without_write(tmp_data_dir, client):
    """2026-10-18 — a force-unload mid-load bumps the generation; the
    watcher's ``still_current()`` returns False and the watcher stops
    WITHOUT writing ``failed`` — the unload route owns the terminal write."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="force-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    sup_unload = AsyncMock()

    gen_bumped = {"v": False}

    def fake_get_generation(self, model_id):
        return 2 if gen_bumped["v"] else 1

    watch_started = {"v": False}
    watch_returned = {"v": False}
    on_give_up_calls = {"n": 0}

    async def fake_watch_load(*, still_current, on_give_up, **_kwargs):
        watch_started["v"] = True
        # Wait until the generation is bumped (force-unload happened).

        deadline = time.monotonic() + 10.0
        while still_current() and time.monotonic() < deadline:  # noqa: ASYNC110 - polls a fake's state with a deadline
            await asyncio.sleep(0.05)
        if not still_current():
            watch_returned["v"] = True
            return
        # If still current, don't fire (shouldn't happen in this test).
        on_give_up_calls["n"] += 1

    async def fake_health(**kwargs):
        # Block until the watcher returns (which happens when the
        # generation is bumped).

        deadline = time.monotonic() + 10.0
        while not watch_returned["v"] and time.monotonic() < deadline:  # noqa: ASYNC110 - polls a fake's state with a deadline
            await asyncio.sleep(0.05)
        # The stop event should have been set by on_give_up — but in
        # this test on_give_up is NOT called, so we return False because
        # the generation changed (the runner's generation check below
        # catches it).
        return False

    with (
        patch("app.runtime.supervisor.Supervisor.load", new=sup_load),
        patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload),
        patch("app.runtime.supervisor.Supervisor.is_running", return_value=True),
        patch("app.runtime.supervisor.Supervisor.get_generation", new=fake_get_generation),
        patch("app.models.routes_api.wait_for_health", new=fake_health),
        patch("app.models.routes_api.watch_load", new=fake_watch_load),
    ):
        r = client.post(
            "/api/models/force-model/load",
            headers={**auth, **csrf_header(client)},
        )
        assert r.status_code == 202
        wait_until(lambda: watch_started["v"], what="the watcher to start")
        # Simulate a force-unload bumping the generation.
        gen_bumped["v"] = True

    wait_until(lambda: watch_returned["v"], what="the watcher to return without firing on_give_up")
    assert on_give_up_calls["n"] == 0, "a superseded load must not fire on_give_up"
    # The row must NOT be 'failed' — the watcher did not write anything.
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status FROM models WHERE id='force-model'").fetchone()
    assert row[0] != "failed", "a superseded load must not write 'failed'"


def test_newer_generation_stops_watcher_without_write(tmp_data_dir, client):
    """A newer load supersedes this one while its watcher is still running.

    A second load bumps the generation (the way a second spawn does) and writes
    its own row. The first load's watcher must stop without giving up, and a
    stale give-up that still reaches the handler must not write anything. Both
    guards are needed: the watcher's ``still_current`` check, and the handler's
    own generation check.

    Driven through the real route, real watcher and real stall policy, with only
    the health wait, the verdict, and the engine spawn faked.
    """
    import threading

    from app.runtime.load_progress import watcher as watcher_mod
    from app.runtime.load_progress.policy import Verdict
    from app.runtime.supervisor import Supervisor

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="gen-model")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)
    sup = client.app.state.supervisor

    health_started = threading.Event()
    health_release = threading.Event()
    give_up_enabled = {"v": False}
    watcher_returned = threading.Event()
    stale_done = threading.Event()
    give_up_calls: list[Verdict] = []
    real_watch = watcher_mod.watch_load
    real_evaluate = watcher_mod.evaluate

    def fake_evaluate(tracker, now, policy):
        verdict = real_evaluate(tracker, now, policy)
        if give_up_enabled["v"]:
            return Verdict(
                state="give_up",
                stalled_for_s=1200.0,
                give_up_at_s=1200.0,
                reason="no progress for 20 min",
                hint=None,
            )
        return verdict

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        self._generation[model.id] = self._generation.get(model.id, 0) + 1

    async def fake_health(**kwargs):
        # The first engine never answers; hold the runner until the test has
        # superseded it and the watcher has observed that.
        health_started.set()
        await asyncio.to_thread(health_release.wait, 30.0)
        return False

    async def spying_watch(**kwargs):
        real_on_give_up = kwargs["on_give_up"]

        async def counting_on_give_up(verdict):
            give_up_calls.append(verdict)
            await real_on_give_up(verdict)

        kwargs["on_give_up"] = counting_on_give_up
        try:
            await real_watch(**kwargs)  # must return on supersede, with no give-up
        finally:
            watcher_returned.set()
        # A stale give-up verdict still reaching the handler after supersede.
        await real_on_give_up(
            Verdict(
                state="give_up",
                stalled_for_s=1200.0,
                give_up_at_s=1200.0,
                reason="stale give-up from a superseded load",
                hint=None,
            )
        )
        stale_done.set()

    with (
        patch.object(Supervisor, "load", new=fake_load),
        patch.object(Supervisor, "register_load_runner", new=lambda *a, **k: None),
        patch("app.models.routes_api.wait_for_health", new=fake_health),
        patch("app.models.routes_api.watch_load", new=spying_watch),
        patch.object(watcher_mod, "evaluate", new=fake_evaluate),
    ):
        r = client.post("/api/models/gen-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        assert health_started.wait(30.0), "the first load never reached its health wait"

        # The second load: bump the generation and write its own row, as a
        # second spawn does, then let the first watcher see it and stale-fire.
        sup._generation["gen-model"] = sup._generation.get("gen-model", 0) + 1
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE models SET status='loaded', last_error=NULL WHERE id='gen-model'")
            db.commit()
        give_up_enabled["v"] = True
        assert watcher_returned.wait(30.0), "the superseded watcher did not stop"
        assert stale_done.wait(30.0)
        health_release.set()

    # The superseded watcher gave up on nothing and wrote nothing.
    assert give_up_calls == [], "a superseded watcher must not fire on_give_up"
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status, last_error FROM models WHERE id='gen-model'").fetchone()
    assert row == ("loaded", None), f"the newer load's row must stand alone, got {row!r}"


@pytest.mark.asyncio
async def test_unload_cancels_registered_runner_and_still_terminates(tmp_data_dir, client):
    """2026-10-18 HAZARD — ``sup.unload(force=True)`` cancels the registered
    load runner task.  The give-up handler runs as part of that same runner.
    If it awaited ``unload`` directly, the cancellation would abort the
    teardown at its next await.  The handler must schedule ``unload`` as a
    SEPARATE task so the runner's own cancellation cannot abort it.

    This test proves the engine is terminated and GPUs released even though
    ``unload`` cancels the runner, and that the row stays ``failed``
    (``Supervisor.unload`` does not write the DB; ``on_exit`` only
    overwrites ``loaded``/``loading`` rows).
    """
    from app.runtime.supervisor import ModelState, Supervisor

    client.get("/healthz")  # ensure migrations ran
    db_path = tmp_data_dir / "vllm-warden.db"
    # Seed a model so the DB has the schema and a row to check.
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="haz")

    # A real Supervisor with a fake driver that records terminate calls.
    terminated = {"v": False}

    class _FakeHandle:
        pid = 99999
        returncode = None

        async def wait(self):
            await asyncio.sleep(100)

    class _FakeDriver:
        supports_engine_image = True
        default_image = None

        async def spawn(self, spec):
            return _FakeHandle()

        def engine_host(self, model_id):
            return "127.0.0.1"

        async def terminate(self, handle, *, grace_s):
            terminated["v"] = True

    from app.config import Settings

    settings = Settings(
        data_dir=tmp_data_dir,
        hf_cache_dir=tmp_data_dir,
        cookie_secret="x" * 32,
        container_gpu_count=2,
    )
    sup = Supervisor(settings, driver=_FakeDriver())
    # Manually register an engine so unload has something to terminate.
    sup._handles["haz"] = _FakeHandle()
    sup._ports["haz"] = 10000
    sup._hosts["haz"] = "127.0.0.1"
    sup._state["haz"] = ModelState.LOADING
    sup._generation["haz"] = 1
    sup.gpus.claim("haz", [0, 1])

    # Register a fake "load runner" task that ``unload`` will cancel.
    async def fake_runner():
        await asyncio.sleep(100)

    runner_task = asyncio.create_task(fake_runner())
    sup.register_load_runner("haz", runner_task)

    # Write ``failed`` first (as on_give_up would).
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE models SET status='failed', "
            "last_error='no progress — gave up; engine stopped and GPUs released' "
            "WHERE id='haz'"
        )
        db.commit()

    # Schedule the unload as a separate task (as on_give_up does).
    unload_task = asyncio.create_task(sup.unload("haz", force=True))

    # Give the unload a moment to run — it cancels the runner inside.
    await asyncio.sleep(0.05)

    # The unload task must complete despite cancelling the runner.
    await asyncio.wait_for(unload_task, timeout=5.0)

    assert terminated["v"], "the engine must be terminated by the unload task"
    assert "haz" not in sup._handles, "the handle must be removed"
    # The runner was cancelled by unload (as the real load runner would be).
    assert (
        runner_task.cancelled()
    ), "unload must cancel the registered load runner — that is the hazard"

    # The row stays ``failed`` — Supervisor.unload does NOT write the DB.
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status, last_error FROM models WHERE id='haz'").fetchone()
    assert row[0] == "failed", f"the row must stay 'failed' after unload; got {row[0]!r}"
    assert "gave up" in row[1]


def test_give_up_unload_survives_runner_cancellation(tmp_data_dir, client):
    """2026-10-18 HAZARD, end to end through the real give-up path.

    The watcher's give-up handler schedules ``sup.unload(force=True)`` as a
    separate task. That unload cancels the registered load runner (the task
    running start_engine). The engine must still be terminated, its handle
    dropped (GPUs released), and the row must stay ``failed``.
    The supervisor is real; only the driver and the watcher/health seams are fakes.
    """
    from app.runtime.load_progress.policy import Verdict
    from app.runtime.supervisor import ModelState

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1, 2, 3], gpus=[0, 1], model_id="haz")
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)
    sup = client.app.state.supervisor

    terminated: list[object] = []

    class _Handle:
        pid = 99999
        returncode = None

        async def wait(self):
            await asyncio.sleep(3600)

    class _Driver:
        supports_engine_image = True
        default_image = None

        async def spawn(self, spec):  # pragma: no cover - load is faked
            return _Handle()

        def engine_host(self, model_id):
            return "127.0.0.1"

        async def terminate(self, handle, *, grace_s):
            terminated.append(handle)

    real_driver = sup._driver
    sup._driver = _Driver()

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # What a successful spawn leaves behind, without starting an engine.
        self._handles[model.id] = _Handle()
        self._ports[model.id] = port
        self._hosts[model.id] = "127.0.0.1"
        self._state[model.id] = ModelState.LOADING
        self._generation[model.id] = self._generation.get(model.id, 0) + 1
        self.gpus.claim(model.id, model.gpu_indices)

    async def fake_health(**kwargs):
        # The engine never comes up; the watcher's give-up sets ``stop``.
        stop = kwargs["stop"]
        await asyncio.wait_for(stop.wait(), timeout=10.0)
        return False

    async def fake_watch_load(*, on_give_up, **_kwargs):
        await on_give_up(
            Verdict(
                state="give_up",
                stalled_for_s=1200.0,
                give_up_at_s=1200.0,
                reason="no progress for 20 min",
                hint=None,
            )
        )

    try:
        with (
            patch("app.runtime.supervisor.Supervisor.load", new=fake_load),
            patch("app.models.routes_api.wait_for_health", new=fake_health),
            patch("app.models.routes_api.watch_load", new=fake_watch_load),
        ):
            r = client.post("/api/models/haz/load", headers={**auth, **csrf_header(client)})
            assert r.status_code == 202, r.text
            wait_until(lambda: len(terminated) >= 1, what="the engine to be terminated")
    finally:
        sup._driver = real_driver

    assert "haz" not in sup._handles, "the handle must be dropped (GPUs released)"
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status, last_error FROM models WHERE id='haz'").fetchone()
    assert row[0] == "failed", f"row must stay failed, got {row[0]!r}"
    assert "gave up; engine stopped and GPUs released" in row[1]


class _FakeEngineHandle:
    pid = None
    returncode = None

    async def wait(self):
        await asyncio.sleep(3600)


def _fake_engine_load():
    """A spawn that leaves the supervisor in the state a live engine would."""

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        from app.runtime.supervisor import ModelState

        self._handles[model.id] = _FakeEngineHandle()
        self._ports[model.id] = port
        self._hosts[model.id] = "127.0.0.1"
        self._state[model.id] = ModelState.LOADING
        self._generation[model.id] = self._generation.get(model.id, 0) + 1
        self.gpus.claim(model.id, model.gpu_indices)

    return fake_load


def test_watcher_runs_during_warmup(tmp_data_dir, client):
    """The stall watcher keeps running while the warm-up probe is in flight, and
    the tracker shows the WARMUP step running at that point."""
    import threading

    from app.runtime.load_progress import watcher as watcher_mod
    from app.runtime.load_progress.parse import Step
    from app.runtime.supervisor import Supervisor
    from app.runtime.warmup_probe import ProbeResult

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="warm-model")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    probe_started = threading.Event()
    probe_release = threading.Event()
    recorder = {"started": 0, "finished": 0}
    real_watch = watcher_mod.watch_load

    async def probe(**_kw):
        probe_started.set()
        await asyncio.to_thread(probe_release.wait, 30.0)
        return ProbeResult(ok=True, detail=None)

    async def recording_watch(**kwargs):
        recorder["started"] += 1
        try:
            await real_watch(**kwargs)
        finally:
            recorder["finished"] += 1

    with (
        patch.object(Supervisor, "load", new=_fake_engine_load()),
        patch("app.models.routes_api.wait_for_health", new=AsyncMock(return_value=True)),
        patch("app.models.routes_api.warmup_probe", new=probe),
        patch("app.models.routes_api.watch_load", new=recording_watch),
    ):
        r = client.post("/api/models/warm-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        assert probe_started.wait(30.0), "the warm-up probe never started"
        tracker = client.app.state.load_progress.get("warm-model")
        assert tracker is not None, "the load must be tracked while warming up"
        assert tracker.step_start_at(Step.WARMUP) is not None, "WARMUP must be running"
        assert tracker.step_end_at(Step.WARMUP) is None, "WARMUP must not be done yet"
        assert (
            recorder["started"] == 1 and recorder["finished"] == 0
        ), "the watcher must still be running during warm-up"
        probe_release.set()
        _wait_for_status(db_path, "warm-model", ("loaded",))
        wait_until(lambda: recorder["finished"] == 1, what="the watcher to finish after warm-up")
    assert client.app.state.load_progress.get("warm-model") is None


def test_give_up_during_warmup_unloads(tmp_data_dir, client):
    """A stall give-up during warm-up takes the same path as one during health
    wait: failed with the give-up text, the stop event, then a separately
    scheduled force-unload that cancels the runner and terminates the engine."""
    from app.runtime.load_progress.parse import Step
    from app.runtime.load_progress.policy import Verdict
    from app.runtime.supervisor import Supervisor

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="warm-stuck")
    _stub_probe(client, [0])
    auth = _jwt_login(client)
    sup = client.app.state.supervisor

    terminated: list[object] = []

    class _Driver:
        supports_engine_image = True
        default_image = None

        def engine_host(self, model_id):
            return "127.0.0.1"

        async def terminate(self, handle, *, grace_s):
            terminated.append(handle)

    real_driver = sup._driver
    sup._driver = _Driver()

    async def stuck_probe(**_kw):
        await asyncio.sleep(3600)  # warm-up never answers; the unload cancels it

    async def fake_watch(*, tracker, on_give_up, **_kw):
        while tracker.step_start_at(Step.WARMUP) is None:  # noqa: ASYNC110 - polls the tracker
            await asyncio.sleep(0.05)
        await on_give_up(
            Verdict(
                state="give_up",
                stalled_for_s=1200.0,
                give_up_at_s=1200.0,
                reason="no progress for 20 min",
                hint=None,
            )
        )

    try:
        with (
            patch.object(Supervisor, "load", new=_fake_engine_load()),
            patch("app.models.routes_api.wait_for_health", new=AsyncMock(return_value=True)),
            patch("app.models.routes_api.warmup_probe", new=stuck_probe),
            patch("app.models.routes_api.watch_load", new=fake_watch),
        ):
            r = client.post("/api/models/warm-stuck/load", headers={**auth, **csrf_header(client)})
            assert r.status_code == 202, r.text
            wait_until(lambda: len(terminated) >= 1, what="the warm-up engine to be terminated")
    finally:
        sup._driver = real_driver

    assert "warm-stuck" not in sup._handles, "the handle must be dropped (GPUs released)"
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status, last_error FROM models WHERE id='warm-stuck'").fetchone()
    assert row[0] == "failed", f"row must be failed, got {row[0]!r}"
    assert "gave up; engine stopped and GPUs released" in row[1]


def test_wait_for_health_raising_leaves_no_watcher_or_registry_entry(tmp_data_dir, client):
    """If the health wait raises unexpectedly, the watcher task is still
    cancelled and the registry entry is removed (try/finally in the runner)."""
    from app.runtime.load_progress import watcher as watcher_mod
    from app.runtime.supervisor import Supervisor

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="raise-model")
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    watch_tasks: list[asyncio.Task] = []
    real_watch = watcher_mod.watch_load

    async def recording_watch(**kwargs):
        watch_tasks.append(asyncio.current_task())
        await real_watch(**kwargs)

    async def boom(**_kw):
        await asyncio.sleep(0)  # let the watcher task start before the health wait raises
        raise RuntimeError("health wait exploded")

    with (
        patch.object(Supervisor, "load", new=_fake_engine_load()),
        patch("app.models.routes_api.wait_for_health", new=boom),
        patch("app.models.routes_api.watch_load", new=recording_watch),
    ):
        r = client.post("/api/models/raise-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        wait_until(
            lambda: client.app.state.load_progress.get("raise-model") is None
            and len(watch_tasks) == 1,
            what="the registry entry to be removed after the runner raised",
        )
    wait_until(lambda: watch_tasks[0].done(), what="the watcher task to be cancelled")


def _usual_history(db_path, model_id, seconds):
    """Record one prior successful load for ``model_id`` under its real fingerprint."""
    import asyncio as _aio

    from app.db.database import open_db
    from app.db.repos.load_history import LoadHistoryRepo, load_fingerprint
    from app.db.repos.models import ModelRepo
    from app.runtime.load_progress.parse import Step
    from app.runtime.variants import baked_engine_version

    async def _go():
        async with open_db(db_path) as db:
            row = await ModelRepo(db).get(model_id)
            fp = load_fingerprint(row, None, baked_engine_version=baked_engine_version("vllm"))
            await LoadHistoryRepo(db).record(model_id, fp, seconds, {Step.WEIGHTS: 10.0})

    _aio.run(_go())


def _blocked_warmup_probe():
    """A warm-up probe that blocks until released; returns (started, release, probe)."""
    import threading

    from app.runtime.warmup_probe import ProbeResult

    probe_started = threading.Event()
    probe_release = threading.Event()

    async def probe(**_kw):
        probe_started.set()
        await asyncio.to_thread(probe_release.wait, 30.0)
        return ProbeResult(ok=True, detail=None)

    return probe_started, probe_release, probe


def test_load_with_history_fills_in_usual_times(tmp_data_dir, client):
    """A load whose model has history gets usual values on its tracker."""
    from app.runtime.supervisor import Supervisor

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="usual-model")
    _usual_history(db_path, "usual-model", 60.0)
    _stub_probe(client, [0])
    auth = _jwt_login(client)
    started, release, probe = _blocked_warmup_probe()
    with (
        patch.object(Supervisor, "load", new=_fake_engine_load()),
        patch("app.models.routes_api.wait_for_health", new=AsyncMock(return_value=True)),
        patch("app.models.routes_api.warmup_probe", new=probe),
    ):
        r = client.post("/api/models/usual-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        assert started.wait(30.0), "the warm-up probe never started"
        tracker = client.app.state.load_progress.get("usual-model")
        assert tracker is not None
        wait_until(lambda: tracker.usual_total_s is not None, what="usual times to fill in")
        assert tracker.usual_total_s == pytest.approx(60.0)
        release.set()
        _wait_for_status(db_path, "usual-model", ("loaded",))


def test_failing_usual_read_leaves_usual_empty_and_load_succeeds(tmp_data_dir, client):
    """A failing history read leaves usual as None and the load still reaches loaded."""
    import sqlite3

    from app.runtime.supervisor import Supervisor

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(db_path, allowed=[0, 1], gpus=[0], model_id="locked-model")
    _stub_probe(client, [0])
    auth = _jwt_login(client)
    calls: list[tuple] = []

    def _locked(*args, **_kw):
        calls.append(args)
        raise sqlite3.OperationalError("database is locked")

    started, release, probe = _blocked_warmup_probe()
    with (
        patch.object(Supervisor, "load", new=_fake_engine_load()),
        patch("app.models.routes_api.wait_for_health", new=AsyncMock(return_value=True)),
        patch("app.models.routes_api.warmup_probe", new=probe),
        patch("app.models.routes_api._read_usual_sync", new=_locked),
    ):
        r = client.post("/api/models/locked-model/load", headers={**auth, **csrf_header(client)})
        assert r.status_code == 202, r.text
        assert started.wait(30.0), "the warm-up probe never started"
        tracker = client.app.state.load_progress.get("locked-model")
        assert tracker is not None
        wait_until(lambda: calls, what="the usual read to be attempted")
        assert tracker.usual_total_s is None
        assert tracker.usual == {}
        release.set()
        _wait_for_status(db_path, "locked-model", ("loaded",))
