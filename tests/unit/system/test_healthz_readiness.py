"""/healthz reports model serving state; /readyz says whether a completion can be served (#342).

lmwarden.com, 2026-11-22: for ~3.5 minutes after a deploy /healthz said
``{"ok": true}`` while every completion 404'd "model not loaded" -- the restored
model was still loading. /healthz is the liveness probe, so it must stay 200 in
every one of these states (failing it mid-load restarts the api and the load
never finishes). Readiness is a separate endpoint.

Both are public, so they carry counts and reason codes only. Model ids, names,
``since`` and error text are for an authenticated admin, on
``GET /api/admin/readiness``.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socket
import sqlite3
import threading
import time

import pytest

from app.proxy import model_cache
from app.system import readiness as rd
from tests.conftest import jwt_login, seed_admin_user

COUNT_KEYS = {"loading", "ready", "failed", "unloading", "unloaded"}


def _model(
    db_path, mid: str, status: str, *, prior_status: str | None = None, last_error=None
) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
            "prior_status) "
            "VALUES (?,?,'org/repo','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,?,?)",
            (mid, f"{mid}-served", json.dumps([0]), status, last_error, prior_status),
        )
        db.commit()
    # A sync write bypasses ModelRepo, which is what drops the models cache.
    model_cache.invalidate_all()


@pytest.fixture(autouse=True)
def _no_memo(monkeypatch):
    # The shared result is reused for SHARED_TTL_S; these tests change state
    # between requests, so each request computes afresh. The memo itself is
    # tested directly below.
    monkeypatch.setattr(rd, "SHARED_TTL_S", 0.0)


@pytest.fixture
def db(client, tmp_data_dir):
    client.get("/healthz")  # lifespan ran, migrations applied
    return tmp_data_dir / "vllm-warden.db"


@pytest.fixture
def admin(client, db):
    seed_admin_user(db)
    return jwt_login(client)


@pytest.fixture
def engine(monkeypatch):
    """The engine health probe /readyz runs, with a scripted answer per port."""
    answers: dict[str, bool] = {}
    calls: list[tuple[str, int, str, object]] = []

    async def fake_probe(host, port, path="/health", *, client=None):
        calls.append((host, port, path, client))
        ok = answers.get(str(port), True)
        return ok, "HTTP 200" if ok else "ConnectError: refused"

    monkeypatch.setattr("app.system.readiness.probe_health", fake_probe)
    return answers, calls


def _serving(client, mid: str, port: int) -> None:
    client.app.state.supervisor._ports[mid] = port


def _watchdog(monkeypatch, *, enabled: bool, restore_on_boot: bool = True, boot_done=True) -> None:
    # Patched where readiness reads it: the app's real watchdog loop keeps
    # republishing app.state.watchdog_cfg, so setting app.state races it.
    view = ({"enabled": enabled, "restore_on_boot": restore_on_boot}, boot_done)
    monkeypatch.setattr(rd, "watchdog_view", lambda _state: view)


def _items(client, admin) -> dict[str, dict]:
    r = client.get("/api/admin/readiness", headers=admin)
    return {m["id"]: m for m in r.json()["models"]["items"]}


def _assert_public(client) -> None:
    """Nothing identifying in either public body."""
    h = client.get("/healthz").json()
    assert set(h) == {"ok", "gpu", "models"}
    assert set(h["models"]) == {"serving", "counts"}
    r = client.get("/readyz").json()
    assert set(r) == {"ready", "reason"}
    text = json.dumps(h["models"]) + json.dumps(r)  # gpu has its own since
    assert "-served" not in text
    assert "since" not in text


# --- no model configured ---------------------------------------------------


def test_no_model_configured(client, db, engine):
    h = client.get("/healthz")
    assert h.status_code == 200
    body = h.json()
    assert body["ok"] is True
    assert "gpu" in body  # existing keys stay
    assert body["models"]["serving"] is False
    assert set(body["models"]["counts"]) == COUNT_KEYS

    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"ready": False, "reason": "no_models_configured"}


# --- loading ----------------------------------------------------------------


def test_loading_model_is_live_but_not_ready(client, db, engine, admin):
    _model(db, "m1", "loading")
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    assert h.json()["models"]["serving"] is False
    assert h.json()["models"]["counts"]["loading"] == 1

    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "loading"
    assert engine[1] == []  # nothing to probe while loading
    item = _items(client, admin)["m1"]
    assert item["state"] == "loading"
    assert item["name"] == "m1-served"
    assert item["since"]
    _assert_public(client)


def test_restore_pending_reads_as_loading_when_the_watchdog_will_restore(
    client, db, engine, monkeypatch
):
    # Boot reconciliation marks a was-serving model failed + prior_status; the
    # watchdog then restores it. That window is the incident's: report loading.
    _model(db, "m1", "failed", prior_status="loaded")
    _watchdog(monkeypatch, enabled=True)
    assert client.get("/healthz").json()["models"]["counts"]["loading"] == 1
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "loading"


def test_restore_pending_before_the_boot_restore_ran(client, db, engine, monkeypatch):
    _model(db, "m1", "failed", prior_status="loading")
    _watchdog(monkeypatch, enabled=False, restore_on_boot=True, boot_done=False)
    assert client.get("/readyz").json()["reason"] == "loading"


def test_restore_pending_but_watchdog_disabled_is_failed(client, db, engine, monkeypatch):
    # Nothing will restore it: saying "loading" would keep a deploy gate
    # waiting for its whole timeout with the wrong reason.
    _model(db, "m1", "failed", prior_status="loaded")
    _watchdog(monkeypatch, enabled=False, restore_on_boot=False)
    assert client.get("/healthz").json()["models"]["counts"]["failed"] == 1
    assert client.get("/readyz").json()["reason"] == "failed"


def test_restore_pending_after_the_boot_restore_with_probing_off_is_failed(
    client, db, engine, monkeypatch
):
    _model(db, "m1", "failed", prior_status="loaded")
    _watchdog(monkeypatch, enabled=False, restore_on_boot=True, boot_done=True)
    assert client.get("/readyz").json()["reason"] == "failed"


def test_restore_pending_under_a_stress_lease_is_failed(client, db, engine, monkeypatch):
    # wants_restart refuses a leased model: the stress run restarts it itself.
    _model(db, "m1", "failed", prior_status="loaded")
    _watchdog(monkeypatch, enabled=True)
    client.app.state.stress_leases.is_held = lambda mid: mid == "m1"
    assert client.get("/readyz").json()["reason"] == "failed"


# --- ready ------------------------------------------------------------------


def test_ready_model(client, db, engine, admin):
    _model(db, "m1", "loaded")
    _serving(client, "m1", 19301)
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["models"]["serving"] is True
    assert engine[1] == []  # /healthz never probes the engine

    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"ready": True, "reason": None}
    assert engine[1] and engine[1][0][1] == 19301
    assert _items(client, admin)["m1"]["engine"] == "ok"
    _assert_public(client)


def test_loaded_row_whose_engine_does_not_answer_is_not_ready(client, db, engine, admin):
    _model(db, "m1", "loaded")
    _serving(client, "m1", 19302)
    engine[0]["19302"] = False
    assert client.get("/healthz").status_code == 200
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "engine_unhealthy"
    assert _items(client, admin)["m1"]["engine"] == "unhealthy"


def test_loaded_row_without_a_running_engine_is_not_serving(client, db, engine):
    # The router 404s "not running" when the supervisor has no port.
    _model(db, "m1", "loaded")
    h = client.get("/healthz")
    assert h.json()["models"]["serving"] is False
    assert h.json()["models"]["counts"]["failed"] == 1
    assert client.get("/readyz").status_code == 503


# --- failed / unloading -----------------------------------------------------


def test_failed_model(client, db, engine, admin):
    _model(db, "m1", "failed", last_error="CUDA out of memory")
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    assert h.json()["models"]["counts"]["failed"] == 1
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "failed"
    assert "CUDA" not in client.get("/healthz").text + r.text  # error text is admin-only
    item = _items(client, admin)["m1"]
    assert item["state"] == "failed"
    assert item["error"] == "CUDA out of memory"


def test_unloading_model(client, db, engine):
    _model(db, "m1", "unloading")
    assert client.get("/healthz").json()["models"]["counts"]["unloading"] == 1
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "unloading"


# --- unloaded on purpose ----------------------------------------------------


@pytest.mark.parametrize("status", ["pulled", "registered", "pulling"])
def test_unloaded_on_purpose(client, db, engine, admin, status):
    _model(db, "m1", status)
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    models = h.json()["models"]
    assert models["serving"] is False
    assert models["counts"]["unloaded"] == 1
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "no_models_configured"
    # Even the admin list names only models meant to be serving.
    assert _items(client, admin) == {}


# --- multi-model ------------------------------------------------------------


def test_multi_model_one_ready_is_ready(client, db, engine, admin):
    _model(db, "a", "loading")
    _model(db, "b", "loaded")
    _model(db, "c", "failed")
    _model(db, "d", "pulled")
    _serving(client, "b", 19303)
    h = client.get("/healthz").json()["models"]
    assert h["serving"] is True
    assert h["counts"] == {"loading": 1, "ready": 1, "failed": 1, "unloading": 0, "unloaded": 1}
    states = {k: v["state"] for k, v in _items(client, admin).items()}
    assert states == {"a": "loading", "b": "ready", "c": "failed"}

    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_multi_model_ready_but_every_engine_down(client, db, engine):
    _model(db, "a", "loaded")
    _model(db, "b", "loaded")
    _serving(client, "a", 19304)
    _serving(client, "b", 19305)
    engine[0]["19304"] = False
    engine[0]["19305"] = False
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "engine_unhealthy"
    engine[0]["19305"] = True
    assert client.get("/readyz").status_code == 200


def test_admin_can_ask_about_one_model(client, db, engine, admin):
    _model(db, "a", "loading")
    _model(db, "b", "loaded")
    _serving(client, "b", 19307)
    assert client.get("/api/admin/readiness?model=b", headers=admin).status_code == 200
    r = client.get("/api/admin/readiness?model=a-served", headers=admin)  # served name too
    assert r.status_code == 503
    assert r.json()["reason"] == "loading"
    assert client.get("/api/admin/readiness?model=nope", headers=admin).status_code == 404


# --- liveness never fails over the models read ------------------------------


def test_healthz_stays_200_when_the_models_read_fails(client, db, engine, admin, monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("database is locked at /data/secret/path")

    monkeypatch.setattr("app.system.readiness.all_models", boom)
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["ok"] is True
    assert h.json()["models"] == {"serving": None, "counts": None}
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"ready": False, "reason": "unavailable"}
    assert "secret" not in h.text + r.text
    a = client.get("/api/admin/readiness", headers=admin)
    assert a.json()["reason"] == "unavailable"
    assert "database is locked" in a.json()["error"]


# --- auth -------------------------------------------------------------------


def test_readyz_is_public(client, db, engine):
    # No session, no bearer: probes and load balancers carry no credential.
    assert client.get("/readyz").status_code == 503


def test_admin_readiness_needs_a_credential(client, db, engine):
    assert client.get("/api/admin/readiness").status_code == 401


def test_ready_agrees_with_the_router(client, db, engine):
    """'ready' is the router's own predicate, not a second opinion."""
    from app.proxy.routes import _resolve_loaded

    class _Req:
        app = client.app

    _model(db, "m1", "loaded")
    _serving(client, "m1", 19306)
    assert client.get("/healthz").json()["models"]["serving"] is True
    rows = {r.id: r for r in client.portal.call(_all_rows, client)}
    _resolve_loaded(_Req(), rows["m1"], "m1-served")  # does not raise


async def _all_rows(client):
    return await model_cache.all_models(
        client.app.state.model_cache, client.app.state.settings.db_path
    )


# --- the real engine probe --------------------------------------------------


class _Status503(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(503)
        self.end_headers()

    def log_message(self, *_a):
        pass


def test_real_probe_non_200_engine_is_unhealthy(client, db):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Status503)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _model(db, "m1", "loaded")
        _serving(client, "m1", srv.server_address[1])
        r = client.get("/readyz")
        assert r.status_code == 503
        assert r.json()["reason"] == "engine_unhealthy"
    finally:
        srv.shutdown()


def test_real_probe_times_out(client, db, monkeypatch):
    # Accepts the TCP connection (the kernel does, via the backlog) and never
    # answers: the probe must give up at ENGINE_PROBE_TIMEOUT_S.
    monkeypatch.setattr(rd, "ENGINE_PROBE_TIMEOUT_S", 0.3)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    try:
        _model(db, "m1", "loaded")
        _serving(client, "m1", sock.getsockname()[1])
        t0 = time.monotonic()
        r = client.get("/readyz")
        elapsed = time.monotonic() - t0
        assert r.status_code == 503
        assert r.json()["reason"] == "engine_unhealthy"
        assert elapsed < 2.0
    finally:
        sock.close()


def test_real_probe_connection_refused(client, db):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens there now
    _model(db, "m1", "loaded")
    _serving(client, "m1", port)
    assert client.get("/readyz").json()["reason"] == "engine_unhealthy"


# --- the shared result ------------------------------------------------------


def test_concurrent_callers_share_one_check(monkeypatch):
    monkeypatch.setattr(rd, "SHARED_TTL_S", 10.0)
    runs = 0

    async def compute():
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.05)
        return runs

    async def main():
        shared = rd._Shared()
        got = await asyncio.gather(*(shared.get(compute) for _ in range(20)))
        again = await shared.get(compute)  # inside the TTL: reused
        model_cache.invalidate_all()  # a models write: recomputed
        fresh = await shared.get(compute)
        return got, again, fresh

    got, again, fresh = asyncio.run(main())
    assert got == [1] * 20
    assert again == 1
    assert fresh == 2


def test_a_cancelled_caller_does_not_cancel_the_shared_check(monkeypatch):
    monkeypatch.setattr(rd, "SHARED_TTL_S", 10.0)

    async def compute():
        await asyncio.sleep(0.05)
        return "done"

    async def main():
        shared = rd._Shared()
        first = asyncio.create_task(shared.get(compute))
        await asyncio.sleep(0)
        second = asyncio.create_task(shared.get(compute))
        await asyncio.sleep(0.01)
        first.cancel()
        return await second

    assert asyncio.run(main()) == "done"


def test_healthz_hits_share_one_models_read(client, db, monkeypatch):
    monkeypatch.setattr(rd, "SHARED_TTL_S", 10.0)
    reads = 0
    real = rd.all_models

    async def counting(*a, **k):
        nonlocal reads
        reads += 1
        return await real(*a, **k)

    monkeypatch.setattr("app.system.readiness.all_models", counting)
    model_cache.invalidate_all()  # drop the result the fixture's boot call left
    for _ in range(5):
        assert client.get("/healthz").status_code == 200
        client.get("/readyz")
    assert reads == 1


def test_engine_probes_share_one_http_client(client, db, engine):
    _model(db, "a", "loaded")
    _model(db, "b", "loaded")
    _serving(client, "a", 19308)
    _serving(client, "b", 19309)
    client.get("/readyz")
    client.get("/readyz")
    clients = {id(c) for *_x, c in engine[1]}
    assert len(engine[1]) == 4
    assert len(clients) == 1 and None not in {c for *_x, c in engine[1]}


# --- /ui/healthz (the macOS single-process alias) ---------------------------


def test_ui_healthz_alias_is_public_shape(tmp_data_dir, tmp_path, monkeypatch, engine):
    from fastapi.testclient import TestClient

    from app.main import build_app

    export = tmp_path / "export"
    export.mkdir()
    (export / "index.html").write_text("<!doctype html><title>ui</title>", encoding="utf-8")
    monkeypatch.setenv("VW_WEBUI_DIR", str(export))
    with TestClient(build_app()) as c:
        r = c.get("/ui/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert set(body["models"]) == {"serving", "counts"}


def test_probes_read_past_a_cold_router_cache(client, db, engine):
    # Filling it would hide rows seeded with direct sqlite writes (which do not
    # bump the generation) from the router for the cache TTL.
    from app.proxy.model_cache import MISS

    model_cache.invalidate_all()
    client.get("/healthz")
    client.get("/readyz")
    assert client.app.state.model_cache.rows() is MISS


def test_admin_readiness_unexpected_error_is_a_clean_503(client, db, engine, admin, monkeypatch):
    async def boom(self):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(rd.ReadinessChecker, "readiness", boom)
    r = client.get("/api/admin/readiness", headers=admin)
    assert r.status_code == 503
    assert r.json()["reason"] == "unavailable"


def test_watchdog_view_reads_what_the_loop_publishes():
    from types import SimpleNamespace

    st = SimpleNamespace(settings=SimpleNamespace(watchdog_enabled=False))
    assert rd.watchdog_view(st) == ({"enabled": False, "restore_on_boot": True}, False)
    st.watchdog_cfg = {"enabled": True, "restore_on_boot": False}
    st.watchdog_boot_restore_done = True
    assert rd.watchdog_view(st) == ({"enabled": True, "restore_on_boot": False}, True)
