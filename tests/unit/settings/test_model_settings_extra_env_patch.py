"""PATCH /api/models/{id}/settings must validate ``extra_env`` at write time
(#261).

Before this fix the PATCH took ``body: dict[str, Any]`` and wrote
``extra_env`` straight into SQLite via ``json.dumps`` -- it never ran
``ModelCreate``'s allowlist validator the way ``POST /api/models`` does. A
signed-in session (or an admin token) could PATCH a hard-locked key like
``LD_PRELOAD`` or ``HF_TOKEN`` and get a 200 with the row persisted, even
though the exact same body at ``POST /api/models`` is a 422. The persisted
row is not remote code execution -- ``filter_extra_env`` fails closed at
launch when it sees a hard-locked key -- but it is a per-model denial of
service: the row can no longer be loaded until an operator edits it back out
by hand.

The fix reuses ``ModelCreate``'s shared allowlist check
(``app.models.schemas.validate_extra_env_for_backend``), which every way the
API can write ``extra_env`` now goes through -- all four of them, the fourth
being the engine-template prefill that this fix's first pass missed (see
``tests/unit/models/test_template_extra_env_allowlist.py``).
"""

import asyncio
import json
import sqlite3
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
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])

    async def _seed_model():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(_seed_model())
    return {**jwt_login(client), **csrf_header(client)}


def _stored_extra_env(db_path: Path, model_id: str = "m1") -> dict:
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT extra_env FROM models WHERE id = ?", (model_id,)
        ).fetchone()
    return json.loads(row[0])


def test_patch_rejects_hard_locked_extra_env_key(
    client: TestClient, auth, tmp_data_dir: Path
):
    """The reviewer's PoC: LD_PRELOAD + HF_TOKEN must 422, not 200, and the
    row must be left exactly as it was."""
    r = client.patch(
        "/api/models/m1/settings",
        json={"extra_env": {"LD_PRELOAD": "/tmp/evil.so", "HF_TOKEN": "x"}},
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert "hard-locked" in r.text.lower()
    assert _stored_extra_env(tmp_data_dir / "vllm-warden.db") == {}


def test_patch_rejects_cuda_visible_devices(client: TestClient, auth):
    r = client.patch(
        "/api/models/m1/settings",
        json={"extra_env": {"CUDA_VISIBLE_DEVICES": "9"}},
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert "hard-locked" in r.text.lower()


def test_patch_rejects_a_key_not_on_the_allowlist(client: TestClient, auth):
    r = client.patch(
        "/api/models/m1/settings",
        json={"extra_env": {"FOO_BAR": "1"}},
        headers=auth,
    )
    assert r.status_code == 422, r.text
    assert "allowlist" in r.text.lower()


def test_patch_accepts_an_allowed_vllm_prefixed_key(
    client: TestClient, auth, tmp_data_dir: Path
):
    r = client.patch(
        "/api/models/m1/settings",
        json={"extra_env": {"VLLM_USE_V1": "1"}},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    assert _stored_extra_env(tmp_data_dir / "vllm-warden.db") == {"VLLM_USE_V1": "1"}


@pytest.mark.parametrize(
    "extra_env",
    [
        {"LD_PRELOAD": "/tmp/evil.so", "HF_TOKEN": "x"},
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"PATH": "/evil"},
        {"FOO_BAR": "1"},
    ],
)
def test_post_and_patch_agree_on_refusal(
    client: TestClient, tmp_data_dir: Path, extra_env
):
    """The rule the PATCH allowlist relies on: a key is patchable only if
    register already accepts it. Both routes must refuse the same body."""
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1])
    auth = {**jwt_login(client), **csrf_header(client)}

    create = client.post(
        "/api/models",
        json={
            "served_model_name": "y",
            "hf_repo": "o/r",
            "gpu_indices": [0],
            "extra_env": extra_env,
        },
        headers=auth,
    )
    assert create.status_code == 422, create.text

    async def _seed_model():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(_seed_model())

    patch = client.patch(
        "/api/models/m1/settings", json={"extra_env": extra_env}, headers=auth
    )
    assert patch.status_code == 422, patch.text
