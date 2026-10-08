"""#215 (slice a) — the gpt-oss-20b template's engine pin under each driver.

The builtin template pins ``EngineSpec(channel="cuda-stable",
vllm_version="0.20.0")`` and ``create_model`` used to persist that pin as the
row's ``engine_image`` regardless of the active driver. Under the in-container
subprocess driver (``supports_engine_image = False``, what production runs)
``Supervisor.load`` then refused every load of such a model with
``EnginePinUnsupported``: a model created from the template could never load.

A pin that comes ONLY from the template is not applied when the running
driver cannot swap the engine image. A pin the request body states explicitly
is still honoured, and the existing load-time refusal still reports it.
"""

import asyncio
from pathlib import Path

import pytest

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.supervisor import EnginePinUnsupported
from tests.conftest import csrf_header, jwt_login, seed_admin_user
from tests.fakes.fake_engine import FakeDriver


class _SubprocessStyleDriver(FakeDriver):
    """FakeDriver that, like the real in-container subprocess driver, cannot
    honor an engine-image pin."""

    supports_engine_image = False


class _ImageSwappingDriver(FakeDriver):
    """FakeDriver that, like the docker driver, can swap the engine image."""

    supports_engine_image = True


def _seed_done(db_path: Path, allowed: list[int]) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client):
    return jwt_login(client, username="admin", password="hunter2")


def _get_row(data_dir: Path, model_id: str) -> ModelRow:
    async def _get() -> ModelRow:
        async with open_db(data_dir / "vllm-warden.db") as db:
            return await ModelRepo(db).get(model_id)

    return asyncio.run(_get())


def test_template_pin_not_persisted_under_imageless_driver(tmp_data_dir, client):
    """#215: under a driver that cannot swap the engine image, a pin that
    comes only from the template is not persisted, and loading the created
    model does not raise ``EnginePinUnsupported``."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    driver = _SubprocessStyleDriver()
    client.app.state.supervisor._driver = driver
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "from-tpl",
            "gpu_indices": [0, 1],
            "template_id": "gpt-oss-20b",
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"] is None, (
        "a template-only pin the active driver cannot honor must not be "
        f"persisted, got {body['engine']!r}"
    )
    # The original symptom: the row's pin made Supervisor.load raise
    # EnginePinUnsupported. The created model must load, and the driver must
    # not be asked to run a pinned image.
    sup = client.app.state.supervisor
    row = _get_row(tmp_data_dir, mid)
    client.portal.call(lambda: sup.load(row, port=19987))
    try:
        assert len(driver.spawned) == 1
        assert driver.spawned[0].image is None
    finally:
        client.portal.call(lambda: sup.unload(mid, force=True))


def test_template_pin_persisted_under_image_swapping_driver(tmp_data_dir, client):
    """Control: with the docker driver — or a fake whose
    ``supports_engine_image`` is true — the template pin is still persisted."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    client.app.state.supervisor._driver = _ImageSwappingDriver()
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "from-tpl-capable",
            "gpu_indices": [0, 1],
            "template_id": "gpt-oss-20b",
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"] == {
        "channel": "cuda-stable",
        "vllm_version": "0.20.0",
        "image": "vllm/vllm-openai:v0.20.0",
    }


def test_explicit_body_pin_still_honoured_under_imageless_driver(tmp_data_dir, client):
    """Control: an engine axis stated in the request body is the operator's
    explicit call — it is still persisted under the subprocess driver, and
    the existing load-time refusal still reports it instead of silently
    discarding it (the #177 guard)."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    driver = _SubprocessStyleDriver()
    client.app.state.supervisor._driver = driver
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "explicit-pin",
            "gpu_indices": [0, 1],
            "template_id": "gpt-oss-20b",
            "engine_image": "vllm/vllm-openai:v0.21.0",
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"]["image"] == "vllm/vllm-openai:v0.21.0"

    sup = client.app.state.supervisor
    row = _get_row(tmp_data_dir, mid)
    with pytest.raises(EnginePinUnsupported):
        client.portal.call(lambda: sup.load(row, port=19988))
