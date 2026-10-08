"""The auth matrix (spec 2026-09-19, "Testing"): every mounted route, every
credential.

Static half -- app/auth/policy.py sorts every APIRoute into a bucket and the
route's dependency tree must carry exactly that bucket's guard:

  inference  /v1/*                 require_bearer
  session    SESSION_ONLY_ROUTES   require_session
             + SESSION_ONLY_PREFIXES
  public     PUBLIC_ROUTES         no guard
  admin      every other route     require_jwt | require_sse_ticket | require_operator

A new route that fits no bucket, a guard in the wrong bucket, or a constant
naming a route that no longer exists fails here. So does a route that is not
an APIRoute (an ``app.mount()``ed sub-app, a bare Starlette route): the
matrix cannot see inside it, so it must not appear silently. FastAPI's own
docs routes (/docs, /redoc, /openapi.json) are off (spec 2026-09-19,
decision 11), so there is no longer an exemption for them here.

Live half -- every non-public route is called with an admin token, a session
JWT, an inference token and no credential. ``_stub_endpoints`` first swaps
each handler for a stub, so a call runs the middleware (CSRF included) and
every dependency but never the handler: no side effects, no streams. A route
is "reached" when the stub answered or FastAPI returned 422 for the dummy path
value / missing body -- both only happen after every dependency has passed.
"""

import asyncio
import re
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.auth import deps
from app.auth.policy import (
    FOREST_ROUTES,
    PUBLIC_ROUTES,
    SESSION_ONLY_PREFIXES,
    SESSION_ONLY_ROUTES,
    is_inference,
    is_session_only,
)
from app.models.routes_logs import require_sse_ticket
from app.proxy.auth import require_bearer
from app.stats.routes_forest import forest_viewer
from app.stress.routes_api import require_operator
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token

# forest_viewer: an admin credential, or a key holder's forest JWT scoped to
# their own key (tests/unit/stats/test_forest_routes.py proves the JWT opens
# nothing else).
ADMIN_CAPABLE: set[Callable[..., Any]] = {
    deps.require_jwt,
    require_sse_ticket,
    require_operator,
    forest_viewer,
}
GUARDS: set[Callable[..., Any]] = ADMIN_CAPABLE | {deps.require_session, require_bearer}
STUB = "x-auth-matrix"
# origin_check_dep runs before require_session on POST /api/auth/logout; this
# is Settings.allowed_origins' default.
ORIGIN = {"Origin": "http://localhost:3000"}


def _endpoints(app: FastAPI) -> list[tuple[str, str, APIRoute]]:
    return [
        (method, route.path, route)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods - {"HEAD"})
    ]


def _guards(route: APIRoute) -> set[Callable[..., Any]]:
    found: set[Callable[..., Any]] = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call in GUARDS:
            found.add(dep.call)
        stack.extend(dep.dependencies)
    return found


def _bucket(method: str, path: str) -> str:
    if is_inference(method, path):
        return "inference"
    if is_session_only(method, path):
        return "session"
    if (method, path) in PUBLIC_ROUTES:
        return "public"
    return "admin"


def _stub_endpoints(app: FastAPI) -> None:
    async def astub(**_: Any) -> Response:
        return Response(status_code=200, headers={STUB: "1"})

    def sstub(**_: Any) -> Response:
        return Response(status_code=200, headers={STUB: "1"})

    for route in app.routes:
        if isinstance(route, APIRoute):
            original = route.dependant.call
            route.dependant.call = astub if asyncio.iscoroutinefunction(original) else sstub


def _call(client: TestClient, method: str, path: str, headers: dict[str, str]) -> Any:
    return client.request(method, re.sub(r"\{[^}]+\}", "x", path), headers=headers)


def _reached(r: Any) -> bool:
    return r.status_code == 422 or r.headers.get(STUB) == "1"


def _error_code(r: Any) -> str | None:
    try:
        detail = r.json().get("detail")
    except ValueError:
        return None
    return detail.get("error_code") if isinstance(detail, dict) else None


def _line(method: str, path: str, r: Any) -> str:
    return f"{method} {path}: {r.status_code} {r.text[:160]}"


@pytest.fixture
def creds(client: TestClient, seeded_db) -> dict[str, dict[str, str]]:
    session = {**jwt_login(client), **csrf_header(client), **ORIGIN}
    r = client.post("/api/tokens", json={"name": "matrix"}, headers=session)
    assert r.status_code == 201, r.text
    inference = r.json()["plaintext"]
    fr = client.post("/api/forest/login", headers=bearer(inference))
    assert fr.status_code == 200, fr.text
    _, admin = seed_admin_token(seeded_db)
    csrf = csrf_header(client)
    _stub_endpoints(client.app)  # after every real call above
    return {
        # No CSRF header on purpose: an admin bearer must not need one.
        "admin": {**bearer(admin), **ORIGIN},
        "session": session,
        "inference": {**bearer(inference), **csrf, **ORIGIN},
        "none": {**csrf, **ORIGIN},
        "forest": {**bearer(fr.json()["token"]), **csrf, **ORIGIN},
    }


def test_every_route_is_one_the_matrix_can_see(client: TestClient) -> None:
    unseen = [
        f"{type(route).__name__} {getattr(route, 'path', route)!r}"
        for route in client.app.routes
        if not isinstance(route, APIRoute)
    ]
    assert unseen == [], "routes outside the auth matrix -- give each a guard and a bucket"


def _misguarded(app: FastAPI) -> list[str]:
    """Every route whose dependencies do not carry its bucket's guard."""
    wrong: list[str] = []
    for method, path, route in _endpoints(app):
        guards = _guards(route)
        bucket = _bucket(method, path)
        if bucket == "inference":
            ok = guards == {require_bearer}
        elif bucket == "session":
            ok = guards == {deps.require_session}
        elif bucket == "public":
            ok = guards == set()
        else:
            ok = bool(guards & ADMIN_CAPABLE) and not guards & {
                deps.require_session,
                require_bearer,
            }
        if not ok:
            names = sorted(g.__name__ for g in guards)
            wrong.append(f"{method} {path}: bucket {bucket}, guards {names}")
    return wrong


def test_every_route_carries_the_guard_of_its_bucket(client: TestClient) -> None:
    assert _misguarded(client.app) == []
    mounted = {(method, path) for method, path, _ in _endpoints(client.app)}
    assert SESSION_ONLY_ROUTES - mounted == set(), "SESSION_ONLY_ROUTES names unmounted routes"
    assert PUBLIC_ROUTES - mounted == set(), "PUBLIC_ROUTES names unmounted routes"


@pytest.mark.parametrize("prefix", SESSION_ONLY_PREFIXES)
def test_a_route_under_a_session_only_prefix_must_be_session_only(
    client: TestClient, prefix: str
) -> None:
    """Forgetting to list a route in SESSION_ONLY_ROUTES does not put it in
    the admin bucket: under a session-only prefix, require_jwt is reported."""

    async def leaky(_user: str = Depends(deps.require_jwt)) -> None:
        return None

    async def fine(_user: str = Depends(deps.require_session)) -> None:
        return None

    client.app.add_api_route(f"{prefix}/{{token_id}}/leak", leaky, methods=["POST"])
    client.app.add_api_route(f"{prefix}/{{token_id}}/fine", fine, methods=["POST"])
    assert _misguarded(client.app) == [
        f"POST {prefix}/{{token_id}}/leak: bucket session, guards ['require_jwt']"
    ]


def test_the_stub_stands_in_for_every_handler(client: TestClient, creds) -> None:
    r = client.get("/healthz")
    assert (
        r.headers.get(STUB) == "1"
    ), "FastAPI no longer reads route.dependant.call per request -- rework _stub_endpoints"


def test_an_admin_token_opens_every_control_route_but_the_session_only_ones(
    client: TestClient, creds
) -> None:
    wrong: list[str] = []
    for method, path, _ in _endpoints(client.app):
        bucket = _bucket(method, path)
        if bucket == "public":
            continue
        r = _call(client, method, path, creds["admin"])
        if bucket == "admin":
            ok = _reached(r)
        elif bucket == "session":
            ok = r.status_code == 403 and _error_code(r) == "session_only"
        else:  # /v1
            ok = r.status_code == 401
        if not ok:
            wrong.append(_line(method, path, r))
    assert wrong == []


def test_a_session_opens_every_control_route(client: TestClient, creds) -> None:
    wrong: list[str] = []
    for method, path, route in _endpoints(client.app):
        bucket = _bucket(method, path)
        if bucket == "public":
            continue
        r = _call(client, method, path, creds["session"])
        # /v1 wants an inference key; a stream wants a ticket, not the header.
        refused = bucket == "inference" or require_sse_ticket in _guards(route)
        ok = r.status_code == 401 if refused else _reached(r)
        if not ok:
            wrong.append(_line(method, path, r))
    assert wrong == []


def test_an_inference_token_is_refused_on_every_control_route(client: TestClient, creds) -> None:
    """401 everywhere, except where require_operator guards the route (the
    stress runs): there the refusal is a 403 `operator_only` that says why."""
    wrong: list[str] = []
    operator_routes = 0
    for method, path, route in _endpoints(client.app):
        if _bucket(method, path) not in ("admin", "session"):
            continue
        r = _call(client, method, path, creds["inference"])
        if require_operator in _guards(route):
            operator_routes += 1
            ok = r.status_code == 403 and _error_code(r) == "operator_only"
        else:
            ok = r.status_code == 401
        if not ok:
            wrong.append(_line(method, path, r))
    assert wrong == []
    assert operator_routes > 0, "no require_operator route left -- drop the special case"


def test_no_credential_is_refused_everywhere_but_the_public_routes(
    client: TestClient, creds
) -> None:
    wrong = [
        _line(method, path, r)
        for method, path, _ in _endpoints(client.app)
        if _bucket(method, path) != "public"
        for r in [_call(client, method, path, creds["none"])]
        if r.status_code != 401
    ]
    assert wrong == []


def test_the_connect_catalogue_is_an_admin_route(client: TestClient) -> None:
    """GET /api/connect/clients (connect-clients plan §2): a session or an
    admin token, never an inference key and never anonymous -- it renders
    this warden's origin and rules, though never a key."""
    assert _bucket("GET", "/api/connect/clients") == "admin"
    route = next(
        r for m, p, r in _endpoints(client.app) if (m, p) == ("GET", "/api/connect/clients")
    )
    assert _guards(route) == {deps.require_jwt}


def test_a_forest_token_opens_only_the_forest_routes(client: TestClient, creds) -> None:
    """The forest JWT (typ="forest") is refused 401 on every non-public route
    but FOREST_ROUTES, where it is accepted."""
    wrong: list[str] = []
    forest_seen = set()
    for method, path, _ in _endpoints(client.app):
        if _bucket(method, path) == "public":
            continue
        r = _call(client, method, path, creds["forest"])
        if (method, path) in FOREST_ROUTES:
            forest_seen.add((method, path))
            ok = _reached(r)
        else:
            ok = r.status_code == 401
        if not ok:
            wrong.append(_line(method, path, r))
    assert wrong == []
    assert forest_seen == FOREST_ROUTES


def test_forest_viewer_guards_only_the_forest_routes(client: TestClient) -> None:
    carriers = {
        (method, path)
        for method, path, route in _endpoints(client.app)
        if forest_viewer in _guards(route)
    }
    assert carriers == FOREST_ROUTES
