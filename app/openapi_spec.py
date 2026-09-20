"""The control API's OpenAPI document, for credentialed clients only.

FastAPI's /docs, /redoc and /openapi.json are switched off in build_app
(docs/superpowers/specs/2026-09-19-admin-tokens-design.md, decision 11): a
browser page cannot send a bearer header when it fetches the spec, and an
anonymous spec is a map of the API for anyone who finds the port.
``GET /api/openapi.json`` serves the same document behind ``require_jwt`` --
a session or an admin token.

``install_openapi`` post-processes FastAPI's generated schema once:

* ``components.securitySchemes.bearerAuth`` (HTTP bearer) and a top-level
  ``security: [{bearerAuth: []}]``;
* ``security: []`` on every operation that takes no control-plane credential:
  app/auth/policy.py::PUBLIC_ROUTES and every /v1 route (an inference token is
  a different credential; documents/API.md describes it).

CI (typecheck:api-types) and ``make generate-api-types`` still build the
document in-process with ``app.openapi()`` -- the same object this serves.
"""

from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.auth.deps import require_jwt
from app.auth.policy import INFERENCE_PREFIX, PUBLIC_ROUTES

BEARER_SCHEME = "bearerAuth"

router = APIRouter(tags=["meta"])


def _is_public(method: str, path: str) -> bool:
    return path.startswith(INFERENCE_PREFIX) or (method, path) in PUBLIC_ROUTES


def add_security(schema: dict[str, Any], app: FastAPI) -> None:
    """Declare the bearer scheme globally and exempt the public operations."""
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})[BEARER_SCHEME] = {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "A session access token from POST /api/auth/login, or an admin "
            "token (vwa_...) issued in Settings -> Admin tokens."
        ),
    }
    schema["security"] = [{BEARER_SCHEME: []}]
    paths: dict[str, Any] = schema.get("paths", {})
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        for method in sorted(route.methods):
            if not _is_public(method, route.path):
                continue
            operation = paths.get(route.path_format, {}).get(method.lower())
            if operation is not None:
                operation["security"] = []


def install_openapi(app: FastAPI) -> None:
    """Make ``app.openapi()`` return the post-processed document (built once,
    cached on ``app.openapi_schema`` exactly like FastAPI's own)."""
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            add_security(generate(), app)
        assert app.openapi_schema is not None
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


@router.get("/api/openapi.json", include_in_schema=False)
async def openapi_json(request: Request, _user: str = Depends(require_jwt)) -> JSONResponse:
    """The whole API description. A session or an admin token."""
    return JSONResponse(request.app.openapi())
