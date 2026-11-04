import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.ext import Registry
from tests.conftest import (
    FakeEntryPoint,
    bearer,
    csrf_header,
    jwt_login,
    seed_admin_user,
)


def _admin(client: TestClient, tmp_data_dir) -> dict[str, str]:
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return jwt_login(client)


def test_no_extensions_means_community(ext_registry, client, tmp_data_dir):
    h = _admin(client, tmp_data_dir)
    assert client.get("/api/edition", headers=h).json() == {
        "edition": "community",
        "features": [],
        "load_errors": [],
    }


def test_edition_requires_admin(ext_registry, client, tmp_data_dir):
    assert client.get("/api/edition").status_code == 401
    # a non-admin credential (an inference key) is refused too
    h = _admin(client, tmp_data_dir)
    r = client.post("/api/tokens", json={"name": "t"}, headers={**h, **csrf_header(client)})
    assert r.status_code == 201, r.text
    key = r.json()["plaintext"]
    assert client.get("/api/edition", headers=bearer(key)).status_code in (401, 403)


def test_failing_extension_is_recorded_not_fatal(ext_registry, client_factory, tmp_data_dir):
    ext_registry(FakeEntryPoint("broken", RuntimeError("boom")))
    with client_factory() as client:
        assert client.get("/healthz").status_code == 200
        h = _admin(client, tmp_data_dir)
        assert client.get("/api/edition", headers=h).json()["load_errors"] == [
            "broken: RuntimeError('boom')"
        ]


def _ee(reg: Registry) -> None:
    r = APIRouter()

    @r.get("/api/ext-test/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    reg.routers.append(r)
    reg.set_edition("enterprise")
    reg.add_feature("home.custom")


def test_extension_router_and_edition(ext_registry, client_factory, tmp_data_dir):
    ext_registry(FakeEntryPoint("ee", _ee))
    with client_factory() as client:
        h = _admin(client, tmp_data_dir)
        body = client.get("/api/edition", headers=h).json()
        assert body["edition"] == "enterprise"
        assert body["features"] == ["home.custom"]
        assert client.get("/api/ext-test/ping").status_code == 200


def test_vendor_outranks_enterprise_regardless_of_load_order(monkeypatch):
    from app.ext.loader import load_extensions

    def ee(r: Registry) -> None:
        r.set_edition("enterprise")

    def ven(r: Registry) -> None:
        r.set_edition("vendor")

    for first, second in ((ee, ven), (ven, ee)):
        eps = [FakeEntryPoint("a", first), FakeEntryPoint("b", second)]
        monkeypatch.setattr("app.ext.loader._entry_points", lambda eps=eps: eps)
        reg = Registry()
        load_extensions(reg)
        assert reg.edition == "vendor"


def test_has_feature_and_unknown_edition():
    reg = Registry()
    reg.add_feature("x")
    assert reg.has_feature("x") and not reg.has_feature("y")
    with pytest.raises(ValueError):
        reg.set_edition("bogus")
    assert reg.edition == "community"


def test_failing_extension_leaves_nothing_behind(ext_registry, client_factory, tmp_data_dir):
    def half(reg: Registry) -> None:
        r = APIRouter()

        @r.get("/api/ext-test/half")
        async def half_route() -> dict[str, bool]:
            return {"ok": True}

        reg.routers.append(r)
        reg.add_feature("half.feature")
        reg.set_edition("enterprise")
        raise RuntimeError("late")

    ext_registry(FakeEntryPoint("half", half))
    with client_factory() as client:
        h = _admin(client, tmp_data_dir)
        assert client.get("/api/ext-test/half").status_code == 404
        body = client.get("/api/edition", headers=h).json()
        assert body == {
            "edition": "community",
            "features": [],
            "load_errors": ["half: RuntimeError('late')"],
        }
