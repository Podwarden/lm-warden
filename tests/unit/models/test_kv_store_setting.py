"""``models.kv_store``: default, validation, reload-only, extra_args guard, effective argv."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cobs2.keys import KEY_FILENAME
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.backends.vllm.env import ENGINE_SHM_PATH
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


def test_default_is_off_and_round_trips(client, db_path, auth):
    _seed(db_path)
    assert _get(db_path).kv_store == "off"
    assert client.get("/api/models/m1", headers=auth).json()["kv_store"] == "off"
    _seed_id = _row(id="m2", served_model_name="y", kv_store="ram+nvme")

    async def _go():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_seed_id)

    asyncio.run(_go())
    assert client.get("/api/models/m2", headers=auth).json()["kv_store"] == "ram+nvme"


def test_create_passes_kv_store(client, db_path, auth):
    r = client.post(
        "/api/models",
        json={"served_model_name": "kv", "hf_repo": "o/m", "gpu_indices": [0], "kv_store": "ram"},
        headers=auth,
    )
    assert r.status_code == 201, r.text
    assert client.get(f"/api/models/{r.json()['id']}", headers=auth).json()["kv_store"] == "ram"


def test_patch_when_unloaded(client, db_path, auth):
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram"}, headers=auth)
    assert r.status_code == 200, r.text
    assert _get(db_path).kv_store == "ram"


def test_bad_value_is_422(client, db_path, auth):
    _seed(db_path)
    for bad in ("on", "nvme", "", None):
        r = client.patch("/api/models/m1/settings", json={"kv_store": bad}, headers=auth)
        assert r.status_code == 422, (bad, r.text)
    assert _get(db_path).kv_store == "off"


def test_loaded_row_refuses_change(client, db_path, auth):
    _seed(db_path, status="loaded")
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram"}, headers=auth)
    assert r.status_code == 409, r.text
    assert _get(db_path).kv_store == "off"


@pytest.mark.parametrize(
    "flag", ["--kv-transfer-config", "--kv-offloading-backend", "--kv-offloading-size"]
)
def test_extra_args_conflicting_flags_are_422_naming_setting(client, db_path, auth, flag):
    _seed(db_path, kv_store="ram")
    for args in ([flag, "x"], [f"{flag}=x"]):
        r = client.patch("/api/models/m1/settings", json={"extra_args": args}, headers=auth)
        assert r.status_code == 422, (args, r.text)
        assert "kv_store" in r.text
    assert _get(db_path).extra_args == []


@pytest.mark.parametrize(
    "args",
    [
        ["--kv_transfer_config", "{}"],  # underscores: FlexibleArgumentParser rewrites them
        ["--kv-transfer-config.kv_connector=X"],  # a dotted key sets the same flag
        ["--kv_transfer_config.kv_connector", "X"],
        ["--kv-transfer-c", "{}"],  # an argparse abbreviation
        ["--kv-t={}"],  # the shortest prefix vLLM resolves to it
        ["--kv_offloading_size", "8"],
        ["--kv-offloading-b=native"],
        ["--kv-offloading.x=1"],
    ],
)
def test_extra_args_kv_flags_in_every_vllm_spelling_are_422(client, db_path, auth, args):
    """F2: vLLM accepts underscores, dotted keys and abbreviations; each reaches the flag."""
    _seed(db_path, kv_store="ram")
    r = client.patch("/api/models/m1/settings", json={"extra_args": args}, headers=auth)
    assert r.status_code == 422, (args, r.text)
    assert "kv_store" in r.text
    assert _get(db_path).extra_args == []


@pytest.mark.parametrize(
    "args", [["--kv-cache-dtype", "fp8"], ["--kv_cache_dtype=fp8"], ["--kv-", "x"], ["--kv"]]
)
def test_other_kv_flags_stay_allowed_while_kv_store_is_on(client, db_path, auth, args):
    """Only the flags kv_store owns are refused; a prefix vLLM cannot resolve to them is not."""
    _seed(db_path, kv_store="ram")
    r = client.patch("/api/models/m1/settings", json={"extra_args": args}, headers=auth)
    assert r.status_code == 200, (args, r.text)


@pytest.mark.parametrize(
    "flag", ["--kv-transfer-config", "--kv-offloading-backend", "--kv-offloading-size"]
)
def test_kv_flags_in_extra_args_are_allowed_while_kv_store_is_off(client, db_path, auth, flag):
    """M5: with kv_store off, vLLM's own KV flags are the operator's to set."""
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json={"extra_args": [flag, "x"]}, headers=auth)
    assert r.status_code == 200, r.text
    assert _get(db_path).extra_args == [flag, "x"]
    # turning kv_store on while they are there is refused, naming the setting
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram"}, headers=auth)
    assert r.status_code == 422, r.text
    assert "kv_store" in r.text
    assert _get(db_path).kv_store == "off"


def test_kv_store_on_is_422_under_the_docker_driver(client, db_path, auth, monkeypatch):
    """F1: only the in-container engine can reach the KV store daemon."""
    _seed(db_path)
    monkeypatch.setenv("VW_ENGINE_DRIVER", "docker")
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram"}, headers=auth)
    assert r.status_code == 422, r.text
    assert "VW_ENGINE_DRIVER=docker" in r.text
    assert _get(db_path).kv_store == "off"
    r = client.post(
        "/api/models",
        json={"served_model_name": "kv", "hf_repo": "o/m", "gpu_indices": [0], "kv_store": "ram"},
        headers=auth,
    )
    assert r.status_code == 422, r.text
    # other edits and kv_store off stay allowed
    r = client.patch("/api/models/m1/settings", json={"kv_store": "off"}, headers=auth)
    assert r.status_code == 200, r.text
    monkeypatch.setenv("VW_ENGINE_DRIVER", "local")
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram"}, headers=auth)
    assert r.status_code == 200, r.text


def test_non_vllm_backend_refuses_kv_store(client, db_path, auth):
    _seed(db_path, backend="llamacpp", filename="m.gguf")
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram"}, headers=auth)
    assert r.status_code == 422, r.text


def test_effective_argv_shows_config_without_secret(client, db_path, auth, tmp_data_dir):
    _seed(db_path, kv_store="ram", hf_revision="d" * 40)  # a commit: resolves to itself
    r = client.get("/api/models/m1/effective-argv", headers=auth)
    assert r.status_code == 200, r.text
    argv = r.json()["argv"]
    cfg = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["socket"] == f"{ENGINE_SHM_PATH}/lmw-kv.sock"
    key_file = tmp_data_dir / KEY_FILENAME
    if key_file.exists():
        secret = key_file.read_text().strip().split(":", 1)[-1]
        assert secret not in r.text


def test_effective_argv_shows_the_tier_off_when_the_revision_has_no_commit(client, db_path, auth):
    """F6: 'main' with nothing in the model cache: the load would run without the connector."""
    _seed(db_path, kv_store="ram", hf_revision="main")
    r = client.get("/api/models/m1/effective-argv", headers=auth)
    assert r.status_code == 200, r.text
    assert "--kv-transfer-config" not in r.json()["argv"]


def _set_nvme_dir(db_path: Path, value: str) -> None:
    from app.db.repos.settings import SettingsRepo

    async def _go():
        async with open_db(str(db_path)) as db:
            await SettingsRepo(db).set("kv_store_nvme_dir", value)

    asyncio.run(_go())


def test_ram_and_disk_needs_a_disk_folder(client, db_path, auth):
    """RAM + disk with no disk folder would run as RAM only: refused, saying where to set it."""
    _seed(db_path)
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram+nvme"}, headers=auth)
    assert r.status_code == 400, r.text
    assert "kv_store_nvme_dir" in r.text and "disk folder" in r.text
    assert _get(db_path).kv_store == "off"
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "kv2",
            "hf_repo": "o/m",
            "gpu_indices": [0],
            "kv_store": "ram+nvme",
        },
        headers=auth,
    )
    assert r.status_code == 400, r.text
    _set_nvme_dir(db_path, "   ")
    assert (
        client.patch(
            "/api/models/m1/settings", json={"kv_store": "ram+nvme"}, headers=auth
        ).status_code
        == 400
    )


def test_ram_and_disk_with_a_disk_folder(client, db_path, auth):
    _seed(db_path)
    _set_nvme_dir(db_path, "/kv")
    r = client.patch("/api/models/m1/settings", json={"kv_store": "ram+nvme"}, headers=auth)
    assert r.status_code == 200, r.text
    assert _get(db_path).kv_store == "ram+nvme"
