"""Runtime settings of the KV store daemon: ``kv_store_nvme_marker`` and
``kv_store_nvme_allow_root_disk``."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.settings.routes_api import RUNTIME_KEYS
from tests.conftest import csrf_header, jwt_login, seed_admin_user

URL = "/api/settings/runtime"


@pytest.fixture
def auth(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    return {**jwt_login(client), **csrf_header(client)}


def test_the_marker_setting_is_a_warden_restart_key() -> None:
    assert RUNTIME_KEYS["kv_store_nvme_marker"] == "warden-restart"


@pytest.mark.parametrize("value", [True, False])
def test_the_marker_setting_accepts_a_boolean(client, auth, value):
    r = client.patch(URL, json={"kv_store_nvme_marker": value}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["requires_restart_kinds"] == ["warden-restart"]
    got = client.get(URL, headers=auth).json()
    assert str(got["kv_store_nvme_marker"]).lower() == str(value).lower()


@pytest.mark.parametrize("value", ["maybe", 2, [], None])
def test_the_marker_setting_refuses_anything_else(client, auth, value):
    r = client.patch(URL, json={"kv_store_nvme_marker": value}, headers=auth)
    assert r.status_code == 422, r.text
    assert "kv_store_nvme_marker" in r.text


# ---- kv_store_nvme_allow_root_disk (Ruling R17) ---------------------------------------


def test_the_root_disk_override_is_a_warden_restart_key() -> None:
    assert RUNTIME_KEYS["kv_store_nvme_allow_root_disk"] == "warden-restart"


@pytest.mark.parametrize("value", [True, False])
def test_the_root_disk_override_accepts_a_boolean(client, auth, value):
    r = client.patch(URL, json={"kv_store_nvme_allow_root_disk": value}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["requires_restart_kinds"] == ["warden-restart"]
    got = client.get(URL, headers=auth).json()
    assert str(got["kv_store_nvme_allow_root_disk"]).lower() == str(value).lower()


@pytest.mark.parametrize("value", ["maybe", 2, [], None])
def test_the_root_disk_override_refuses_anything_else(client, auth, value):
    r = client.patch(URL, json={"kv_store_nvme_allow_root_disk": value}, headers=auth)
    assert r.status_code == 422, r.text
    assert "kv_store_nvme_allow_root_disk" in r.text
