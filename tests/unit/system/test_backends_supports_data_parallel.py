"""#286: ``GET /api/system/backends`` publishes ``supports_data_parallel``."""

from __future__ import annotations

from pathlib import Path

from app.runtime.backends import registry
from tests.conftest import jwt_login, seed_admin_user


def _ready(tmp_data_dir: Path, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0])
    return jwt_login(client)


def test_vllm_supports_data_parallel_and_llamacpp_does_not(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    body = client.get("/api/system/backends", headers=auth).json()
    by_name = {b["name"]: b for b in body["backends"]}
    assert by_name["vllm"]["supports_data_parallel"] is True
    assert by_name["llamacpp"]["supports_data_parallel"] is False


def test_capability_flag_defaults_false_and_registry_matches():
    assert registry.get("vllm").capabilities.supports_data_parallel is True
    assert registry.get("llamacpp").capabilities.supports_data_parallel is False
