"""#286 review #8: the boot reconcile's outcome is visible through the models API
and clears when the operator fixes the row."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.boot_reconcile import reconcile_legacy_data_parallel_rows
from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="m1",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0, 1],
        tensor_parallel_size=2,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=["--tensor-parallel-size", "1"],
        status="pulled",
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

    async def _seed():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())
            await ModelRepo(db).insert(
                _row(
                    id="m2",
                    served_model_name="m2",
                    gpu_indices=[0],
                    tensor_parallel_size=1,
                    extra_args=["--data-parallel-size", "1"],
                )
            )
        await reconcile_legacy_data_parallel_rows(SimpleNamespace(db_path=db_path))

    asyncio.run(_seed())
    return {**jwt_login(client), **csrf_header(client)}


def test_unfixable_row_reports_a_warning_and_stays_editable(client: TestClient, auth):
    body = client.get("/api/models/m1", headers=auth).json()
    assert body["layout_notice"]["level"] == "warning"
    assert "extra_args" in body["layout_notice"]["message"]
    listed = {m["id"]: m for m in client.get("/api/models", headers=auth).json()["models"]}
    assert listed["m1"]["layout_notice"]["level"] == "warning"
    # not locked out of unrelated edits
    r = client.patch("/api/models/m1/settings", json={"max_model_len": 4096}, headers=auth)
    assert r.status_code == 200, r.text
    assert client.get("/api/models/m1", headers=auth).json()["layout_notice"] is not None
    # fixing extra_args clears the warning
    r = client.patch("/api/models/m1/settings", json={"extra_args": []}, headers=auth)
    assert r.status_code == 200, r.text
    assert client.get("/api/models/m1", headers=auth).json()["layout_notice"] is None


def test_cleaned_row_reports_info_until_next_edit(client: TestClient, auth):
    body = client.get("/api/models/m2", headers=auth).json()
    assert body["layout_notice"]["level"] == "info"
    assert body["extra_args"] == []
    r = client.patch("/api/models/m2/settings", json={"max_model_len": 4096}, headers=auth)
    assert r.status_code == 200, r.text
    assert client.get("/api/models/m2", headers=auth).json()["layout_notice"] is None
