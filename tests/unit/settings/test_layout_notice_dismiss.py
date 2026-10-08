"""DELETE /api/models/{id}/layout-notice: dismiss an info notice (#286 re-review)."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from tests.conftest import csrf_header, jwt_login, seed_admin_user
from tests.unit.settings.test_model_settings_data_parallel_patch import _get, _seed


@pytest.fixture
def db_path(client: TestClient, tmp_data_dir: Path) -> Path:
    p = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(p, allowed_gpu_indices=[0, 1, 2, 3])
    return p


@pytest.fixture
def auth(client: TestClient) -> dict[str, str]:
    return {**jwt_login(client), **csrf_header(client)}


def _notice(db_path: Path, level: str | None, message: str | None) -> None:
    async def _go():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).set_layout_notice("m1", level, message)

    asyncio.run(_go())


def test_info_notice_is_dismissed(client, db_path, auth):
    _seed(db_path, extra_args=["--keep"])
    _notice(db_path, "info", "Parallel layout updated")
    before = _get(db_path)
    r = client.delete("/api/models/m1/layout-notice", headers=auth)
    assert r.status_code == 204, r.text
    row = _get(db_path)
    assert row.layout_notice_level is None and row.layout_notice is None
    assert row.extra_args == ["--keep"] and row.updated_at == before.updated_at
    assert client.get("/api/models/m1", headers=auth).json()["layout_notice"] is None


def test_warning_is_refused_and_kept(client, db_path, auth):
    _seed(db_path)
    _notice(db_path, "warning", "extra_args still carry a layout")
    r = client.delete("/api/models/m1/layout-notice", headers=auth)
    assert r.status_code == 409, r.text
    row = _get(db_path)
    assert row.layout_notice_level == "warning" and row.layout_notice


def test_no_notice_is_a_noop(client, db_path, auth):
    _seed(db_path)
    assert client.delete("/api/models/m1/layout-notice", headers=auth).status_code == 204


def test_unknown_model_is_404(client, db_path, auth):
    assert client.delete("/api/models/nope/layout-notice", headers=auth).status_code == 404


def test_requires_admin_login(client, db_path):
    _seed(db_path)
    _notice(db_path, "info", "x")
    assert client.delete("/api/models/m1/layout-notice").status_code in (401, 403)
    assert _get(db_path).layout_notice_level == "info"


def test_repo_clear_only_touches_info(client, db_path):
    _seed(db_path)
    _notice(db_path, "warning", "w")

    async def _go():
        async with open_db(str(db_path)) as db:
            return await ModelRepo(db).clear_info_layout_notice("m1")

    assert asyncio.run(_go()) is False
    assert _get(db_path).layout_notice_level == "warning"
