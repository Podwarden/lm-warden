"""``models.thinking_policy`` (live-editable) and ``thinking_policy_supported``."""

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


def test_default_is_client(db_path):
    _seed(db_path)
    assert _get(db_path).thinking_policy == "client"


def test_model_thinking_policy_patch_while_loaded(client, db_path, auth, monkeypatch):
    _seed(db_path, status="loaded")
    sup = client.app.state.supervisor
    calls: list[str] = []

    async def _record(name, *a, **k):
        calls.append(name)

    for name in ("load", "unload", "ensure_unloadable"):
        monkeypatch.setattr(sup, name, lambda *a, _n=name, **k: _record(_n, *a, **k))
    r = client.patch("/api/models/m1/settings", json={"thinking_policy": "off"}, headers=auth)
    assert r.status_code == 200, r.text
    row = _get(db_path)
    assert row.thinking_policy == "off"
    assert row.status == "loaded"
    assert calls == []  # no unload or reload
    assert client.get("/api/models/m1", headers=auth).json()["thinking_policy"] == "off"


def test_bad_value_is_422(client, db_path, auth):
    _seed(db_path)
    for bad in ("on", "", None):
        r = client.patch("/api/models/m1/settings", json={"thinking_policy": bad}, headers=auth)
        assert r.status_code == 422, (bad, r.text)
    assert _get(db_path).thinking_policy == "client"


def test_loaded_row_refuses_mixed_patch(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch(
        "/api/models/m1/settings",
        json={"thinking_policy": "off", "max_model_len": 4096},
        headers=auth,
    )
    assert r.status_code == 409, r.text


@pytest.mark.parametrize(
    ("template", "expected"),
    [("{% if enable_thinking %}x{% endif %}", True), ("plain template", False), (None, None)],
)
def test_supported_flag_from_template(client, db_path, auth, monkeypatch, template, expected):
    import app.models.serialisation as ser

    seen = []

    def fake(_cache, repo):
        seen.append(repo)
        return template

    monkeypatch.setattr(ser, "read_hf_chat_template", fake)
    _seed(db_path)
    one = client.get("/api/models/m1", headers=auth).json()
    assert one["thinking_policy_supported"] is expected
    (listed,) = client.get("/api/models", headers=auth).json()["models"]
    assert listed["thinking_policy_supported"] is expected
    assert seen and set(seen) == {"a/b"}


@pytest.mark.parametrize(
    ("template", "expected"),
    [("{% if enable_thinking %}x{% endif %}", True), ("plain template", False), (None, None)],
)
def test_settings_route_carries_supported_flag(
    client, db_path, auth, monkeypatch, template, expected
):
    import app.models.serialisation as ser

    monkeypatch.setattr(ser, "read_hf_chat_template", lambda _cache, repo: template)
    _seed(db_path)
    body = client.get("/api/models/m1/settings", headers=auth).json()
    assert body["thinking_policy_supported"] is expected
