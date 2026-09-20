"""Typing GET /api/models and GET /api/models/{id} changed nothing on the wire:
the typed responses equal the untyped serialiser output, key for key and value
for value, and a key ModelOut does not declare still gets through."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.models import routes_api
from app.models.serialisation import model_detail
from tests.conftest import jwt_login, seed_admin_user


def _row() -> ModelRow:
    return ModelRow(
        id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
        gpu_indices=[0], tensor_parallel_size=1, dtype=None, max_model_len=8192,
        gpu_memory_utilization=0.9, trust_remote_code=False, extra_args=["--x"],
        status="registered", pulled_bytes=0, pulled_total=None, last_error=None,
        extra_env={"A": "1"}, engine_channel="stable", engine_vllm_version="0.9.2",
        engine_image=None, supports_vision=1, backend="vllm",
    )


@pytest.fixture
def auth(client: TestClient, tmp_data_dir: Path) -> dict[str, str]:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)

    async def seed() -> None:
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row())

    asyncio.run(seed())
    return jwt_login(client)


def _stored(client: TestClient, tmp_data_dir: Path) -> dict:
    async def load() -> ModelRow | None:
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            return await ModelRepo(db).get("m1")

    row = asyncio.run(load())
    assert row is not None
    return json.loads(json.dumps(model_detail(row)))


def test_the_typed_responses_equal_the_serialiser(client, auth, tmp_data_dir):
    expected = _stored(client, tmp_data_dir)
    assert client.get("/api/models/m1", headers=auth).json() == expected
    assert client.get("/api/models", headers=auth).json() == {"models": [expected]}


def test_an_undeclared_key_is_not_dropped(client, auth, monkeypatch):
    real = routes_api.model_detail
    monkeypatch.setattr(routes_api, "model_detail", lambda row: {**real(row), "future_column": 7})
    assert client.get("/api/models/m1", headers=auth).json()["future_column"] == 7
