"""A ``kv_store`` engine's owner is released from the daemon when it goes away (I3).

Spec 2026-11-21 (Plan 2) section 7.1 and Review Focus 5: a ``ram`` row and a ``ram+nvme``
row on the same weights keep NVMe off while the ``ram`` row is registered, so the api
must drop a row's owner on unload, engine exit (crash or reload), delete and when
``kv_store`` goes ``off``. With the daemon down none of these fail.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

from app.kv import policy_sync
from app.runtime.supervisor import Supervisor
from tests.conftest import csrf_header
from tests.fakes.fake_engine import FakeDriver
from tests.unit.stats.test_forest_routes import _ready_key

SHA = "c" * 40


class _Client:
    def __init__(self, fail: bool = False) -> None:
        self.released: list[str] = []
        self.fail = fail

    def release_owner(self, owner: str) -> int | None:
        if self.fail:
            raise OSError("daemon down")
        self.released.append(owner)
        return 0


class _Kv:
    def __init__(self, client: _Client | None) -> None:
        self._client = client

    async def ensure_running(self, timeout_s: float | None = None) -> bool:
        return True

    def generation(self) -> int:
        return 1

    def healthy(self) -> bool:
        return self._client is not None

    def on_restart(self, cb: Any) -> None:
        pass

    def client(self) -> _Client | None:
        return self._client


class _Settings:
    load_max_s = 7200.0


class _M:
    id = "qwen"
    hf_repo = "Qwen/Qwen3.5-9B"
    hf_revision = "main"
    served_model_name = "qwen3.5-9b"
    gpu_indices = [1]
    tensor_parallel_size = 1
    max_model_len = 8192
    dtype = "auto"
    gpu_memory_utilization = 0.9
    extra_env: dict[str, str] = {}


async def _loaded(tmp_path, kv: _Kv, kv_store: str = "ram") -> tuple[Supervisor, FakeDriver]:
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    settings.db_path = tmp_path / "w.db"
    settings.hf_cache_dir = str(tmp_path / "hf")
    refs = tmp_path / "hf" / "models--Qwen--Qwen3.5-9B" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(SHA)
    driver = FakeDriver()
    sup = Supervisor(settings, driver=driver, kv_store=kv)
    model = _M()
    model.kv_store = kv_store
    await sup.load(model, port=10001)
    return sup, driver


async def _until(cond, limit_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + limit_s
    while not cond():
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_unload_releases_the_owner(tmp_path):
    client = _Client()
    sup, _driver = await _loaded(tmp_path, _Kv(client))
    await sup.unload("qwen", force=True)
    assert client.released == ["qwen"]


@pytest.mark.asyncio
async def test_engine_crash_releases_the_owner_before_on_exit(tmp_path):
    client = _Client()
    settings_kv = _Kv(client)
    sup, driver = await _loaded(tmp_path, settings_kv)
    seen: list[list[str]] = []

    async def on_exit(rc: int) -> None:
        seen.append(list(client.released))

    # Re-arm the watcher with an on_exit hook, as the load runner does.
    sup._watchers["qwen"].cancel()
    sup._watchers["qwen"] = asyncio.create_task(sup._watch_exit("qwen", on_exit))
    await asyncio.sleep(0)
    driver.handles["qwen"]._finish(1)
    await _until(lambda: bool(seen))
    assert seen == [["qwen"]]  # released before the crash handler may reload it
    assert client.released == ["qwen"]


@pytest.mark.asyncio
async def test_a_reload_releases_then_registers_again(tmp_path):
    client = _Client()
    kv = _Kv(client)
    sup, _driver = await _loaded(tmp_path, kv)
    await sup.unload("qwen", force=True)
    model = _M()
    model.kv_store = "ram"
    await sup.load(model, port=10002)
    assert client.released == ["qwen"]
    await sup.unload("qwen", force=True)
    assert client.released == ["qwen", "qwen"]


@pytest.mark.asyncio
async def test_a_model_without_a_tier_releases_nothing(tmp_path):
    client = _Client()
    sup, _driver = await _loaded(tmp_path, _Kv(client), kv_store="off")
    await sup.unload("qwen", force=True)
    assert client.released == []


@pytest.mark.asyncio
@pytest.mark.parametrize("client", [None, _Client(fail=True)])
async def test_with_the_daemon_down_unload_and_exit_do_not_fail(tmp_path, client):
    sup, driver = await _loaded(tmp_path, _Kv(client))
    await sup.unload("qwen", force=True)
    assert "qwen" not in sup._handles
    sup, driver = await _loaded(tmp_path, _Kv(client))
    driver.handles["qwen"]._finish(1)
    await _until(lambda: "qwen" not in sup._handles)


# ---- the routes ------------------------------------------------------------------------


@pytest.fixture
def released(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def fake(kv_store: Any, owner: str) -> None:
        calls.append(owner)

    monkeypatch.setattr(policy_sync, "release_owner", fake)
    monkeypatch.setattr(policy_sync, "request_sync", lambda app_state: None)
    return calls


def _add_model(db_path, mid: str, kv: str) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
            "kv_store) VALUES (?, ?, 'org/m', 'main', '[0]', 1, 'auto', 4096, 0.9, 0, '[]', "
            "'registered', 0, NULL, NULL, ?)",
            (mid, mid, kv),
        )
        db.commit()


def test_delete_releases_the_owner(tmp_data_dir, client, released):
    db_path, _plain, admin = _ready_key(client, tmp_data_dir)
    admin = {**admin, **csrf_header(client)}
    _add_model(db_path, "m2", "ram")
    assert client.delete("/api/models/m2", headers=admin).status_code == 204
    assert released == ["m2"]


def test_kv_store_off_releases_the_owner(tmp_data_dir, client, released):
    db_path, _plain, admin = _ready_key(client, tmp_data_dir)
    admin = {**admin, **csrf_header(client)}
    _add_model(db_path, "m2", "ram")
    with sqlite3.connect(db_path) as db:  # RAM + disk needs a disk folder
        db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('kv_store_nvme_dir', '/kv')"
        )
        db.commit()
    r = client.patch("/api/models/m2/settings", json={"kv_store": "ram+nvme"}, headers=admin)
    assert r.status_code == 200, r.text
    assert released == []
    r = client.patch("/api/models/m2/settings", json={"kv_store": "off"}, headers=admin)
    assert r.status_code == 200, r.text
    assert released == ["m2"]


def test_routes_do_not_fail_with_the_daemon_down(tmp_data_dir, client, monkeypatch):
    """The real ``release_owner`` against an app whose daemon never started."""
    monkeypatch.setattr(policy_sync, "request_sync", lambda app_state: None)
    db_path, _plain, admin = _ready_key(client, tmp_data_dir)
    admin = {**admin, **csrf_header(client)}
    _add_model(db_path, "m2", "ram")
    r = client.patch("/api/models/m2/settings", json={"kv_store": "off"}, headers=admin)
    assert r.status_code == 200, r.text
    assert client.delete("/api/models/m2", headers=admin).status_code == 204
