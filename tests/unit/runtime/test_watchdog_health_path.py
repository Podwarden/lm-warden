"""Watchdog probes the backend's own health path, not a hardcoded /health.

Pre-existing bug #2 in the Metal design: every engine answers its health
question at a path that belongs to the engine, not to warden. vLLM happens to
choose ``/health`` today, which is why the hardcoded URL has worked -- and
why the moment a backend answers anywhere else (a Metal-era backend, or a
future llamacpp endpoint) the watchdog would probe a route that 404s and
"restart" a perfectly healthy engine forever.
"""

from __future__ import annotations

import pytest

# --- probe_health: the path is a parameter, defaulting to /health ----------


@pytest.mark.asyncio
async def test_probe_uses_the_given_path(httpx_mock):
    """A probe told where to look must look THERE, not at the old hardcoded /health."""
    httpx_mock.add_response(url="http://h:9/v1/ready", status_code=200, text="ready")
    healthy, detail = await _probe(path="/v1/ready")
    assert healthy, "a 200 on the given path is healthy"
    assert "200" in detail


@pytest.mark.asyncio
async def test_probe_defaults_to_slash_health(httpx_mock):
    """No path given -> the historical /health, so existing callers are unchanged."""
    httpx_mock.add_response(url="http://h:9/health", status_code=200, text="OK")
    healthy, _ = await _probe()
    assert healthy


@pytest.mark.asyncio
async def test_probe_given_path_404_is_unhealthy(httpx_mock):
    """A path that 404s (the symptom before the fix) is unhealthy, not silently /health."""
    httpx_mock.add_response(url="http://h:9/v1/ready", status_code=404, text="nope")
    healthy, detail = await _probe(path="/v1/ready")
    assert not healthy
    assert "404" in detail


def _probe(*, path: str | None = None):
    """Import inside the body so the base tree (no ``path`` param) fails HERE, not at import."""
    from app.runtime.watchdog import probe_health

    if path is None:
        return probe_health("h", 9)
    return probe_health("h", 9, path=path)


# --- check_once: resolve each row's backend and pass its health_path -------


@pytest.mark.asyncio
async def test_check_once_probes_backend_health_path(wd_probe_env, httpx_mock):
    """check_once must ask each engine at ITS health_path, not the hardcoded /health.

    The row's backend answers at /v1/ready. Register ONLY that URL: a probe
    to /health would miss (unmatched request), the row would count as a
    failure, and the assertion on the failure streak below would catch it.
    """
    watchdog, s, app_state, state, budget = wd_probe_env
    httpx_mock.add_response(url="http://h:9/v1/ready", status_code=200)
    await watchdog.check_once(s, app_state, state, budget)
    assert state.get("m1", 0) == 0, "a 200 on the backend's own path must read as healthy"


@pytest.mark.asyncio
async def test_check_once_default_backend_unchanged(wd_probe_env, httpx_mock, monkeypatch):
    """A row whose backend has health_path='/health' behaves exactly as before."""
    watchdog, s, app_state, state, budget = wd_probe_env
    # The fixture patches the registry to a /v1/ready backend; restore the real
    # one so the row's vllm backend resolves to its own /health (leg d).
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "get", registry._BACKENDS.get, raising=False)
    # Register ONLY /health. If check_once probed anything else, the request
    # would miss and the row would count as a failure.
    httpx_mock.add_response(url="http://h:9/health", status_code=200)
    await watchdog.check_once(s, app_state, state, budget)
    assert state.get("m1", 0) == 0, "health_path=/health is the historical default; no change"


@pytest.mark.asyncio
async def test_check_once_registry_failure_falls_back_to_slash_health(
    wd_probe_env, httpx_mock, caplog, monkeypatch
):
    """A registry failure degrades to /health with a debug log; it does not crash the pass."""
    watchdog, s, app_state, state, budget = wd_probe_env
    from app.runtime.backends import registry

    def _boom(name):
        raise RuntimeError("boom")

    # Make the row's registry lookup blow up so check_once must fall back.
    monkeypatch.setattr(registry, "get", _boom)
    httpx_mock.add_response(url="http://h:9/health", status_code=200)
    with caplog.at_level("DEBUG"):
        await watchdog.check_once(s, app_state, state, budget)
    assert state.get("m1", 0) == 0, "the pass must still complete and read the row healthy"
    assert any("health path" in r.message for r in caplog.records), "fallback must be logged"


# --- fixture ----------------------------------------------------------------


@pytest.fixture
def wd_probe_env(tmp_path, monkeypatch):
    """Drive check_once in isolation with a row whose backend answers at /v1/ready."""
    from app.runtime import watchdog

    class FakeSettings:
        db_path = tmp_path / "wd.db"
        logs_dir = tmp_path / "logs"
        watchdog_failure_threshold = 3
        watchdog_max_restarts = 3
        watchdog_interval_s = 30
        watchdog_enabled = True
        watchdog_restore_on_boot = True

    FakeSettings.logs_dir.mkdir(parents=True)
    s = FakeSettings()

    class FakeRow:
        id = "m1"
        status = "loaded"
        backend = "vllm"

    class FakeSup:
        _handles = {"m1": object()}

        def get_port(self, mid):
            return 9

        def get_host(self, mid):
            return "h"

        def get_pid(self, mid):
            return 1

        def get_overrides(self, mid):
            return {}

    class FakeRepo:
        def __init__(self, db):
            pass

        async def list_all(self):
            return [FakeRow()]

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(watchdog, "ModelRepo", FakeRepo)
    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _NullDB())

    # Resolve the row's backend to one whose health_path is NOT /health, so a
    # hardcoded /health probe would visibly miss. Patch the real registry module
    # (the module the implementation imports `registry` as), so on the base tree
    # -- which never looks it up -- this patch is inert and the test fails for
    # the right reason (the probe still hits /health, not the registered URL).
    from app.runtime.backends import registry

    class _Backend:
        capabilities = type("C", (), {"health_path": "/v1/ready"})()

    # Only the ROW's own backend name resolves to /v1/ready; any other lookup
    # (e.g. ``registry.get(None)`` -> the default vllm) gets the real backend
    # and its /health, so check_once must pass row.backend, not a constant.
    real_get = registry.get
    monkeypatch.setattr(
        registry, "get", lambda name: _Backend() if name == "vllm" else real_get(name)
    )

    state: dict = {}
    budget = type("B", (), {"reset": staticmethod(lambda mid: None)})()
    app_state = type("S", (), {"supervisor": FakeSup()})()
    return watchdog, s, app_state, state, budget
