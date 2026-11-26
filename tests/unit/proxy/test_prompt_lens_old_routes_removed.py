"""The routes Prompt Lens had under its former name are gone (removed after the
one-release deprecation window): they answer 404, with no handler and no
OpenAPI entry, and the key page no longer reports the old field name."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.routing import APIRoute

from tests.conftest import jwt_login
from tests.unit.tokens.test_me_prompt_lens_stream import A, Env, env

__all__ = ["env"]  # the fixture, re-used

_OLD = "god" + "mode"  # the former path segment, assembled so a repo-wide search for the old name
# finds only the few places that must name it

OLD_ADMIN = [
    ("GET", f"/api/admin/{_OLD}/status"),
    ("GET", f"/api/admin/{_OLD}/stream"),
    ("GET", f"/api/admin/{_OLD}/kept"),
    ("GET", f"/api/admin/{_OLD}/kept/abc"),
    ("GET", f"/api/admin/{_OLD}/requests/abc/section/prompt"),
    ("GET", f"/api/admin/{_OLD}/media/abc"),
]
OLD_ME = [
    ("POST", f"/api/me/{_OLD}/ticket"),
    ("GET", f"/api/me/{_OLD}/stream"),
    ("GET", f"/api/me/{_OLD}/kept"),
    ("GET", f"/api/me/{_OLD}/kept/abc"),
    ("GET", f"/api/me/{_OLD}/requests/abc/section/prompt"),
]


@pytest.mark.parametrize(("method", "path"), OLD_ADMIN)
def test_old_admin_paths_are_404(client: Any, seeded_db: Any, method: str, path: str) -> None:
    r = client.request(method, path, headers=jwt_login(client))
    assert r.status_code == 404, path
    assert "deprecation" not in r.headers


@pytest.mark.parametrize(("method", "path"), OLD_ME)
async def test_old_key_holder_paths_are_404(env: Env, method: str, path: str) -> None:
    r = await env.http.request(method, path, headers=env.key[A])
    assert r.status_code == 404, path


def test_no_route_and_no_openapi_path_carries_the_old_name(client: Any) -> None:
    assert not [r.path for r in client.app.routes if _OLD in getattr(r, "path", "")]
    assert not [p for p in client.app.openapi()["paths"] if _OLD in p]
    assert not [
        r.path
        for r in client.app.routes
        if isinstance(r, APIRoute) and getattr(r, "deprecated", False) and "prompt-lens" in r.path
    ]


async def test_key_page_reports_only_the_new_field_name(env: Env) -> None:
    r = await env.http.get("/api/me/key", headers=env.key[A])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt_lens_view_available"] is True
    assert not [k for k in body if _OLD in k]
