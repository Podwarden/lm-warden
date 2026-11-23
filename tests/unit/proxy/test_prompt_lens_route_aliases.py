"""The routes Prompt Lens had as "god mode" stay for one release as deprecated aliases.

Every ``/api/admin/prompt-lens/...`` and ``/api/me/prompt-lens/...`` route has
its ``/godmode`` twin: the same handler and the same auth, marked
``deprecated`` in the OpenAPI document, and every answer (errors included)
carries ``Deprecation: true`` and a ``Link`` to the new path. The new routes
carry neither.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastapi.routing import APIRoute

from app.proxy.prompt_lens_legacy import LEGACY_STREAM_PATH, legacy_path
from tests.asgi_stream import open_stream
from tests.conftest import bearer, jwt_login
from tests.unit.tokens.test_me_prompt_lens_stream import A, Env, env

__all__ = ["env"]  # the fixture, re-used

NEW = ("/api/admin/prompt-lens", "/api/me/prompt-lens")


def _routes(app: Any) -> dict[tuple[str, str], APIRoute]:
    return {(m, r.path): r for r in app.routes if isinstance(r, APIRoute) for m in r.methods}


def test_legacy_path_maps_both_prefixes() -> None:
    assert legacy_path("/api/admin/prompt-lens/stream") == "/api/admin/godmode/stream"
    assert legacy_path("/api/me/prompt-lens/kept/{req_id}") == "/api/me/godmode/kept/{req_id}"
    assert LEGACY_STREAM_PATH == "/api/admin/godmode/stream"


def test_every_new_route_has_a_deprecated_twin(client: Any) -> None:
    routes = _routes(client.app)
    new = {k: r for k, r in routes.items() if k[1].startswith(NEW)}
    assert len(new) >= 11
    for (method, path), route in new.items():
        old = routes.get((method, legacy_path(path)))
        assert old is not None, (method, path)
        assert old.endpoint is route.endpoint, path  # same handler, so the same auth
        assert old.deprecated is True and not route.deprecated
    old_routes = {k for k in routes if "/godmode" in k[1]}
    assert old_routes == {(m, legacy_path(p)) for m, p in new}


def test_openapi_marks_only_the_old_paths_deprecated(client: Any) -> None:
    paths = client.app.openapi()["paths"]
    old = {p: ops for p, ops in paths.items() if "/godmode" in p}
    new = {p: ops for p, ops in paths.items() if "/prompt-lens" in p}
    assert old and len(old) == len(new)
    assert all(op.get("deprecated") is True for ops in old.values() for op in ops.values())
    assert not any(op.get("deprecated") for ops in new.values() for op in ops.values())


def test_old_admin_path_same_answer_with_deprecation_headers(client: Any, seeded_db: Any) -> None:
    auth = jwt_login(client)
    new = client.get("/api/admin/prompt-lens/status", headers=auth)
    old = client.get("/api/admin/godmode/status", headers=auth)
    assert new.status_code == old.status_code == 200
    assert old.json() == new.json()
    assert old.headers["deprecation"] == "true"
    assert old.headers["link"] == '</api/admin/prompt-lens/status>; rel="successor-version"'
    assert "deprecation" not in new.headers


def test_old_path_validation_error_is_marked(client: Any, seeded_db: Any) -> None:
    auth = jwt_login(client)
    r = client.get("/api/admin/godmode/kept?limit=0", headers=auth)
    assert r.status_code == 422
    assert r.headers["deprecation"] == "true"
    assert r.headers["link"] == '</api/admin/prompt-lens/kept>; rel="successor-version"'
    new = client.get("/api/admin/prompt-lens/kept?limit=0", headers=auth)
    assert new.status_code == 422 and "deprecation" not in new.headers


def test_old_admin_path_same_auth(client: Any, seeded_db: Any) -> None:
    r = client.get("/api/admin/godmode/status")
    assert r.status_code == 401
    assert r.headers["deprecation"] == "true"
    r = client.get("/api/admin/godmode/status", headers=bearer("vw_not-a-real-key-000000000000000"))
    assert r.status_code in (401, 403)


async def test_old_key_holder_routes_work_for_the_key_only(env: Env) -> None:
    r = await env.http.post("/api/me/godmode/ticket", headers=env.key[A])
    assert r.status_code == 200, r.text
    assert r.headers["deprecation"] == "true"
    # The admin session is refused there, as on the new path.
    r = await env.http.post("/api/me/godmode/ticket", headers=env.admin)
    assert r.status_code == 403
    r = await env.http.get("/api/me/godmode/kept", headers=env.key[A])
    assert r.status_code == 200 and r.json() == {"kept": []}


async def test_old_key_stream_takes_a_ticket_from_either_mint(env: Env) -> None:
    for mint in ("/api/me/godmode/ticket", "/api/me/prompt-lens/ticket"):
        t = (await env.http.post(mint, headers=env.key[A])).json()["ticket"]
        async with open_stream(env.app, f"/api/me/godmode/stream?ticket={t}", {}) as s:
            assert s.status == 200, mint


async def test_old_admin_stream_path(env: Env) -> None:
    r = await env.http.post(
        "/api/auth/sse-ticket", json={"path": LEGACY_STREAM_PATH}, headers=env.admin
    )
    assert r.status_code == 200, r.text
    async with open_stream(env.app, f"{LEGACY_STREAM_PATH}?ticket={r.json()['ticket']}", {}) as s:
        assert s.status == 200
    # Off: the mint for the old path is refused too.
    env.prompt_lens(False)
    r = await env.http.post(
        "/api/auth/sse-ticket", json={"path": LEGACY_STREAM_PATH}, headers=env.admin
    )
    assert r.status_code == 409


async def test_key_page_reports_both_field_names(env: Env) -> None:
    r = await env.http.get("/api/me/key", headers=env.key[A])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt_lens_view_available"] is True
    assert body["godmode_view_available"] is True  # deprecated alias
    env.app.state.settings = dataclasses.replace(env.app.state.settings, prompt_lens_enabled=False)
    body = (await env.http.get("/api/me/key", headers=env.key[A])).json()
    assert body["prompt_lens_view_available"] is body["godmode_view_available"] is False
