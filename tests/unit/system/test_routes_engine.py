"""GET /api/system/engine — JWT-protected. Reports the active engine driver
and whether it can honor an engine-version pin (#177).

The frontend Try-stack panel reads this to disable + explain the version
selector when the deployment runs the in-container subprocess engine (whose
vLLM version is fixed by the warden image and cannot be swapped)."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import jwt_login, seed_admin_user


def _seed_done(db_path: Path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0])


def _jwt_auth(client):
    return jwt_login(client)


def test_engine_requires_jwt(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/system/engine")
    assert r.status_code == 401


def test_engine_subprocess_driver_reports_no_version_select(tmp_data_dir, client):
    # The default app is built with the in-container subprocess driver, which
    # cannot swap the engine image.
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)

    r = client.get("/api/system/engine", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["driver"] == "subprocess"
    assert body["supports_version_select"] is False
    assert "vllm_version" in body
    # vllm is not installed in the test image; the field is present and null.
    assert body["vllm_version"] is None


def test_engine_docker_capable_driver_reports_version_select(tmp_data_dir, client):
    # Swap in a driver that CAN honor an engine-image pin.
    class _Capable:
        supports_engine_image = True

    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)
    client.app.state.supervisor._driver = _Capable()

    r = client.get("/api/system/engine", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["driver"] == "docker"
    assert body["supports_version_select"] is True
