"""PATCH /api/models/{id}/settings must validate ``gpu_indices`` at write time.

Both other write paths already check the setup allowlist: creation rejects an
out-of-range index with a 400 (``POST /api/models``) and the load runner
rejects one with a 422. PATCH does neither -- ``gpu_indices`` is an
operator-writable column (``app/models/policy.py``), and the handler writes the
list straight into SQLite.

The consequence is a model that is impossible to load and impossible to
diagnose from the model page: the write succeeds, and the failure surfaces
later as ``gpu_indices [0, 1, 2, 3] not subset of allowed [0, 1]`` at load
time. Observed in production on a two-GPU host carrying a row that named four.
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
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])

    async def _seed_model():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(_seed_model())
    return {**jwt_login(client), **csrf_header(client)}


# `tensor_parallel_size` travels with `gpu_indices` in every body below: the
# PATCH validates the MERGED row, and register's cross-field check ties the two
# (#262). Changing the placement without restating the parallel size used to be
# a 200 that produced a row which could not load.
def test_patch_rejects_gpu_indices_outside_the_allowlist(client: TestClient, auth):
    r = client.patch(
        "/api/models/m1/settings",
        json={"gpu_indices": [0, 1, 2, 3], "tensor_parallel_size": 4},
        headers=auth,
    )
    assert r.status_code == 400, r.text
    assert "not in allowed_gpu_indices" in r.text.lower()


def test_patch_leaves_the_stored_value_untouched_on_rejection(
    client: TestClient, auth, tmp_data_dir: Path
):
    client.patch(
        "/api/models/m1/settings",
        json={"gpu_indices": [2, 3], "tensor_parallel_size": 2},
        headers=auth,
    )

    async def _read():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            return (await ModelRepo(db).get("m1")).gpu_indices

    assert asyncio.run(_read()) == [0]


def test_patch_accepts_gpu_indices_inside_the_allowlist(client: TestClient, auth):
    r = client.patch(
        "/api/models/m1/settings",
        json={"gpu_indices": [0, 1], "tensor_parallel_size": 2},
        headers=auth,
    )
    assert r.status_code == 200, r.text
