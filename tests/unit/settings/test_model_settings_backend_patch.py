"""PATCH /api/models/{id}/settings must validate ``backend`` at write time.

The allowlist is read off ``app/models/policy.py`` (before that: ModelRow's
fields minus ``_NEVER_PATCH``), and ``backend`` is an operator-writable column --
with no value check of its own. Without one, an operator could
write ``"sglang"``, the write would succeed, and the failure would surface much
later as an ``UnknownBackendError`` raised from inside ``Supervisor.load``:
after the GPU claim, as an opaque ``last_error``, at load time rather than at
write time.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from tests.conftest import csrf_header, jwt_login, seed_admin_user


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
def auth(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)

    async def _seed_model():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(_seed_model())
    return {**jwt_login(client), **csrf_header(client)}


def test_patch_rejects_an_unknown_backend(client: TestClient, auth):
    r = client.patch("/api/models/m1/settings", json={"backend": "sglang"}, headers=auth)
    assert r.status_code == 400
    assert "sglang" in r.json()["detail"]


def test_patch_accepts_a_known_backend(client: TestClient, auth):
    """Every name ``registry.available()`` reports must be writable, and only
    those. Parametrised off the registry rather than spelled out, so Task 8's
    registration of ``llamacpp`` extends this test by existing."""
    from app.runtime.backends import registry

    for name in registry.available():
        r = client.patch(
            "/api/models/m1/settings", json={"backend": name}, headers=auth
        )
        assert r.status_code == 200, (name, r.text)


def test_patch_accepts_null_backend(client: TestClient, auth):
    """NULL is the D6 default and must stay writable -- it is how an operator
    puts a row back on vLLM without knowing the default's name."""
    r = client.patch("/api/models/m1/settings", json={"backend": None}, headers=auth)
    assert r.status_code == 200


def test_patch_rejects_a_non_string_backend(client: TestClient, auth):
    r = client.patch("/api/models/m1/settings", json={"backend": 7}, headers=auth)
    assert r.status_code == 400
