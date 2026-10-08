"""PATCH /api/models/{id}/settings for the data-parallel columns (#286)."""

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
def db_path(client: TestClient, tmp_data_dir: Path) -> Path:
    p = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(p, allowed_gpu_indices=[0, 1, 2, 3])
    return p


def _seed(db_path: Path, **over) -> None:
    async def _go():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row(**over))

    asyncio.run(_go())


def _get(db_path: Path) -> ModelRow:
    async def _go():
        async with open_db(str(db_path)) as db:
            row = await ModelRepo(db).get("m1")
            assert row is not None
            return row

    return asyncio.run(_go())


@pytest.fixture
def auth(client: TestClient) -> dict[str, str]:
    return {**jwt_login(client), **csrf_header(client)}


def test_patch_layout_together(client, db_path, auth):
    _seed(db_path)
    r = client.patch(
        "/api/models/m1/settings",
        json={"gpu_indices": [0, 1, 2, 3], "data_parallel_size": 2, "tensor_parallel_size": 2},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    row = _get(db_path)
    assert (row.gpu_indices, row.data_parallel_size, row.tensor_parallel_size) == (
        [0, 1, 2, 3],
        2,
        2,
    )


def test_patch_dp_alone_on_one_gpu_row_is_422(client, db_path, auth):
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json={"data_parallel_size": 2}, headers=auth)
    assert r.status_code == 422, r.text
    assert "len(gpu_indices)" in r.text


def test_loaded_row_accepts_affinity_toggle(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch("/api/models/m1/settings", json={"dp_affinity_enabled": False}, headers=auth)
    assert r.status_code == 200, r.text
    assert _get(db_path).dp_affinity_enabled == 0


def test_loaded_row_accepts_spill_threshold(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch("/api/models/m1/settings", json={"dp_spill_threshold": 12}, headers=auth)
    assert r.status_code == 200, r.text
    assert _get(db_path).dp_spill_threshold == 12


def test_loaded_row_refuses_mixed_patch(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch(
        "/api/models/m1/settings",
        json={"dp_spill_threshold": 12, "max_model_len": 4096},
        headers=auth,
    )
    assert r.status_code == 409, r.text


def test_loaded_row_refuses_dp_size_change(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch("/api/models/m1/settings", json={"data_parallel_size": 2}, headers=auth)
    assert r.status_code == 409, r.text


def test_affinity_null_is_400(client, db_path, auth):
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json={"dp_affinity_enabled": None}, headers=auth)
    assert r.status_code == 400, r.text


def test_affinity_string_spelling_accepted(client, db_path, auth):
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json={"dp_affinity_enabled": "false"}, headers=auth)
    assert r.status_code == 200, r.text
    assert _get(db_path).dp_affinity_enabled == 0


def test_get_settings_carries_the_three_keys(client, db_path, auth):
    _seed(db_path, dp_spill_threshold=9)
    body = client.get("/api/models/m1/settings", headers=auth).json()
    assert body["data_parallel_size"] == 1
    assert body["dp_affinity_enabled"] == 1
    assert body["dp_spill_threshold"] == 9
