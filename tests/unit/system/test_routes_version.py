"""GET /api/version — JWT required, returns build-time version + sha."""

import importlib

import pytest

from tests.conftest import jwt_login, seed_admin_user


# #55 fix — local shims route through the shared barrier+retry helpers.
def _seed_done(db_path):
    seed_admin_user(db_path, allowed_gpu_indices=[0])


def _jwt_auth(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def test_version_requires_jwt(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")

    r = client.get("/api/version")
    assert r.status_code == 401


def test_version_returns_env_values_when_set(tmp_path, monkeypatch):
    """When the build args are baked into the image, /api/version surfaces them
    verbatim. The values are read at module import time, so we monkeypatch
    before re-importing both the routes module and the main app."""
    monkeypatch.setenv("VW_BUILD_VERSION", "v2026.05.13.0")
    monkeypatch.setenv("VW_BUILD_SHA", "abc1234deadbeef")
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    # Force a fresh import so the module-level cache picks up the env vars
    # that this test just set. Without this the cache from a previous test's
    # module import would shadow our values.
    import app.system.routes_version as routes_version_mod

    importlib.reload(routes_version_mod)
    import app.main as main_mod

    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.build_app()) as c:
        c.get("/healthz")
        _seed_done(tmp_path / "vllm-warden.db")
        auth = _jwt_auth(c)
        r = c.get("/api/version", headers=auth)

    assert r.status_code == 200
    body = r.json()
    assert body == {"version": "v2026.05.13.0", "sha": "abc1234deadbeef"}


def test_version_falls_back_to_dev_unknown_when_env_missing(tmp_path, monkeypatch):
    """No build args = dev binary. Values must be the literal strings the
    frontend can render without special-casing empty fields."""
    monkeypatch.delenv("VW_BUILD_VERSION", raising=False)
    monkeypatch.delenv("VW_BUILD_SHA", raising=False)
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    import app.system.routes_version as routes_version_mod

    importlib.reload(routes_version_mod)
    import app.main as main_mod

    importlib.reload(main_mod)

    from fastapi.testclient import TestClient

    with TestClient(main_mod.build_app()) as c:
        c.get("/healthz")
        _seed_done(tmp_path / "vllm-warden.db")
        auth = _jwt_auth(c)
        r = c.get("/api/version", headers=auth)

    assert r.status_code == 200
    assert r.json() == {"version": "dev", "sha": "unknown"}


@pytest.fixture(autouse=True)
def _restore_modules():
    """The two env-driven tests reload `app.main` and `app.system.routes_version`
    so other tests in the session see the post-import cached values. Reload
    once more after each test in this file to reset to whatever the test
    runner's env had at first import."""
    yield
    import app.main as main_mod
    import app.system.routes_version as routes_version_mod

    importlib.reload(routes_version_mod)
    importlib.reload(main_mod)
