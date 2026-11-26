"""#343: an operator picks a model's bundled vLLM engine through the settings PATCH.

``engine_vllm_version`` with no ``engine_channel`` names an install baked into the
image (``app/runtime/backends/vllm/engines.py``). The write must name one that is
actually present -- a typo, or a build without the bundled engine, is a 422 now
rather than an opaque load failure later -- and ``null`` returns the model to the
default engine. The model's detail response carries the bare pin.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from tests.conftest import csrf_header, jwt_login, seed_admin_user

ENGINE = "0.34.0rc1.dev1202-pr69165"


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="x",
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="registered",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


@pytest.fixture
def engines_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "engines"
    bin_dir = root / ENGINE / "bin"
    bin_dir.mkdir(parents=True)
    vllm = bin_dir / "vllm"
    vllm.write_text("#!/bin/sh\n")
    vllm.chmod(0o755)
    monkeypatch.setenv("VW_ENGINES_ROOT", str(root))
    return root


@pytest.fixture
def auth(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)

    async def _seed_model():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(_seed_model())
    return {**jwt_login(client), **csrf_header(client)}


def _version(client: TestClient, auth, path: str):
    return client.get(path, headers=auth).json()["engine_vllm_version"]


def test_patch_accepts_an_installed_engine(client: TestClient, auth, engines_root):
    r = client.patch("/api/models/m1/settings", json={"engine_vllm_version": ENGINE}, headers=auth)
    assert r.status_code == 200, r.text
    assert _version(client, auth, "/api/models/m1/settings") == ENGINE
    assert _version(client, auth, "/api/models/m1") == ENGINE


def test_patch_null_returns_to_the_default_engine(client: TestClient, auth, engines_root):
    client.patch("/api/models/m1/settings", json={"engine_vllm_version": ENGINE}, headers=auth)
    r = client.patch("/api/models/m1/settings", json={"engine_vllm_version": None}, headers=auth)
    assert r.status_code == 200, r.text
    assert _version(client, auth, "/api/models/m1") is None


def test_patch_rejects_an_engine_that_is_not_installed(client: TestClient, auth, engines_root):
    r = client.patch("/api/models/m1/settings", json={"engine_vllm_version": "9.9.9"}, headers=auth)
    assert r.status_code == 422
    assert "9.9.9" in r.text
    assert _version(client, auth, "/api/models/m1") is None


def test_patch_rejects_a_path_shaped_version(client: TestClient, auth, engines_root):
    r = client.patch(
        "/api/models/m1/settings", json={"engine_vllm_version": "../etc"}, headers=auth
    )
    assert r.status_code == 422


def test_patch_leaves_a_channel_pinned_version_alone(client: TestClient, auth, engines_root):
    """With an ``engine_channel`` the version is an upstream image tag, not a bundled id."""
    r = client.patch(
        "/api/models/m1/settings",
        json={"engine_channel": "stable", "engine_vllm_version": "0.9.2"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
