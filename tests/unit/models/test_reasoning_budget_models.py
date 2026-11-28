"""``models.reasoning_budget`` / ``reasoning_budget_mode`` / ``reasoning_effort_budgets`` (#348)."""

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
    seed_admin_user(p, allowed_gpu_indices=[0, 1])
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


def test_defaults_mean_no_limit(db_path):
    _seed(db_path)
    row = _get(db_path)
    assert row.reasoning_budget is None
    assert row.reasoning_budget_mode == "force_answer"
    assert row.reasoning_effort_budgets is None


def test_row_roundtrip(db_path):
    _seed(
        db_path,
        reasoning_budget=3000,
        reasoning_budget_mode="stop",
        reasoning_effort_budgets={"low": 500, "high": 9000},
    )
    row = _get(db_path)
    assert (row.reasoning_budget, row.reasoning_budget_mode) == (3000, "stop")
    assert row.reasoning_effort_budgets == {"low": 500, "high": 9000}


def test_patch_while_loaded_needs_no_reload(client, db_path, auth, monkeypatch):
    _seed(db_path, status="loaded")
    sup = client.app.state.supervisor
    calls: list[str] = []

    async def _record(name, *a, **k):
        calls.append(name)

    for name in ("load", "unload", "ensure_unloadable"):
        monkeypatch.setattr(sup, name, lambda *a, _n=name, **k: _record(_n, *a, **k))
    r = client.patch(
        "/api/models/m1/settings",
        json={
            "reasoning_budget": 2000,
            "reasoning_budget_mode": "stop",
            "reasoning_effort_budgets": {"medium": 1500},
        },
        headers=auth,
    )
    assert r.status_code == 200, r.text
    row = _get(db_path)
    assert (row.reasoning_budget, row.reasoning_budget_mode) == (2000, "stop")
    assert row.reasoning_effort_budgets == {"medium": 1500}
    assert row.status == "loaded" and calls == []
    got = client.get("/api/models/m1", headers=auth).json()
    assert got["reasoning_budget"] == 2000
    assert got["reasoning_budget_mode"] == "stop"
    assert got["reasoning_effort_budgets"] == {"medium": 1500}
    (listed,) = client.get("/api/models", headers=auth).json()["models"]
    assert listed["reasoning_budget"] == 2000


def test_patch_null_clears(client, db_path, auth):
    _seed(db_path, reasoning_budget=3000, reasoning_effort_budgets={"low": 5})
    r = client.patch(
        "/api/models/m1/settings",
        json={"reasoning_budget": None, "reasoning_effort_budgets": None},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    row = _get(db_path)
    assert row.reasoning_budget is None and row.reasoning_effort_budgets is None


@pytest.mark.parametrize(
    "patch",
    [
        {"reasoning_budget": 0},
        {"reasoning_budget": -4},
        {"reasoning_budget": 1.5},
        {"reasoning_budget": "9"},
        {"reasoning_budget_mode": "bogus"},
        {"reasoning_budget_mode": None},
        {"reasoning_effort_budgets": {"extreme": 5}},
        {"reasoning_effort_budgets": {"low": 0}},
        {"reasoning_effort_budgets": {"low": "5"}},
        {"reasoning_effort_budgets": [1]},
    ],
)
def test_bad_values_are_422(client, db_path, auth, patch):
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json=patch, headers=auth)
    assert r.status_code == 422, (patch, r.text)
    assert _get(db_path).reasoning_budget is None


def test_loaded_row_refuses_mixed_patch(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch(
        "/api/models/m1/settings",
        json={"reasoning_budget": 10, "max_model_len": 4096},
        headers=auth,
    )
    assert r.status_code == 409, r.text
