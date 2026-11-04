"""Public routes, no-cookie paths and settings keys come from the registry,
and core's auth code never imports the website."""

import re
from pathlib import Path

from fastapi import APIRouter

from app.auth.public_routes import core_public_routes, mints_csrf
from app.ext import PublicRoute, Registry, SettingKey
from tests.conftest import FakeEntryPoint, csrf_header, jwt_login, seed_admin_user

ROOT = Path(__file__).parents[3]


def _ext(reg: Registry) -> None:
    r = APIRouter()

    @r.get("/x/page")
    async def page() -> dict[str, bool]:
        return {"ok": True}

    @r.get("/x/secret")
    async def secret() -> dict[str, bool]:
        return {"ok": True}

    reg.routers.append(r)
    reg.public_routes.append(PublicRoute("GET", "/x/page", mint_csrf=False))
    reg.settings_keys.append(SettingKey("x_key", "none", str))


def test_extension_public_route_is_open_and_cookieless(ext_registry, client_factory):
    ext_registry(FakeEntryPoint("x", _ext))
    with client_factory() as c:
        r = c.get("/x/page")
        assert r.status_code == 200
        assert "set-cookie" not in r.headers


def test_extension_public_route_is_marked_public_in_openapi(
    ext_registry, client_factory, tmp_data_dir
):
    ext_registry(FakeEntryPoint("x", _ext))
    with client_factory() as c:
        seed_admin_user(tmp_data_dir / "vllm-warden.db")
        h = jwt_login(c)
        spec = c.get("/api/openapi.json", headers=h).json()
        assert spec["paths"]["/x/page"]["get"]["security"] == []
        assert "security" not in spec["paths"]["/x/secret"]["get"]


def test_extension_setting_key_is_accepted_and_unknown_still_400(
    ext_registry, client_factory, tmp_data_dir
):
    ext_registry(FakeEntryPoint("x", _ext))
    with client_factory() as c:
        seed_admin_user(tmp_data_dir / "vllm-warden.db")
        h = {**jwt_login(c)}
        h.update(csrf_header(c))
        ok = c.patch("/api/settings/runtime", json={"x_key": "v"}, headers=h)
        assert ok.status_code == 200, ok.text
        assert c.get("/api/settings/runtime", headers=h).json()["x_key"] == "v"
        bad = c.patch("/api/settings/runtime", json={"nope": 1}, headers=h)
        assert bad.status_code == 400


def test_without_extension_the_key_is_unknown(ext_registry, client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = {**jwt_login(client)}
    h.update(csrf_header(client))
    assert client.patch("/api/settings/runtime", json={"x_key": "v"}, headers=h).status_code == 400


def test_core_mint_rules(client):
    app = client.app
    assert not mints_csrf(app, "/agent-guide.md")
    assert mints_csrf(app, "/api/csrf")
    assert mints_csrf(app, "/healthz")
    assert mints_csrf(app, "/anything/else")
    assert "/_landing" not in {r.path for r in core_public_routes()}


def test_auth_and_openapi_do_not_import_the_website():
    files = [*(ROOT / "app/auth").rglob("*.py"), ROOT / "app/openapi_spec.py"]
    pat = re.compile(r"^\s*(from|import)\s+app\.landing\b", re.M)
    offenders = [str(f.relative_to(ROOT)) for f in files if pat.search(f.read_text())]
    assert offenders == []
