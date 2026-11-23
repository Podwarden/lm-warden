"""The routes Prompt Lens had as "god mode", kept one release as deprecated aliases.

Prompt Lens was called god mode until 2026-11-22. Scripts and open browser tabs
may still call ``/api/admin/godmode/...`` and ``/api/me/godmode/...``, so each
new route gets a twin under the old prefix (``legacy_router``): the same
handler, hence the same auth and the same answers, marked ``deprecated`` in the
OpenAPI document. Every answer under an old prefix, an error included (auth,
422, 404), carries ``Deprecation: true`` and a ``Link`` to the new path
(``DeprecatedPathMiddleware``). The UI uses the new paths
only. Remove this module, its two ``include_router`` calls, the middleware and the old paths in
app/auth/policy.py in the release after next.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_PREFIXES = {
    "/api/admin/prompt-lens": "/api/admin/godmode",
    "/api/me/prompt-lens": "/api/me/godmode",
}

#: The admin live stream's old path: the SSE-ticket mint refuses a ticket for it
#: while Prompt Lens is off, as for the new one (app/auth/routes.py).
LEGACY_STREAM_PATH = "/api/admin/godmode/stream"


def legacy_path(path: str) -> str:
    """The old path of a new Prompt Lens route path (unchanged if it is not one)."""
    for new, old in _PREFIXES.items():
        if path == new or path.startswith(new + "/"):
            return old + path[len(new) :]
    return path


def _successor(path: str) -> str:
    for new, old in _PREFIXES.items():
        if path == old or path.startswith(old + "/"):
            return new + path[len(old) :]
    return path


def is_legacy_path(path: str) -> bool:
    return any(path == old or path.startswith(old + "/") for old in _PREFIXES.values())


class DeprecatedPathMiddleware:
    """Pure ASGI: every response under an old prefix -- a handler's answer, an
    auth refusal, a 422 validation error, a 404 -- carries ``Deprecation: true``
    and a ``Link`` to the new path. Streams pass through untouched."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if not is_legacy_path(path):
            await self.app(scope, receive, send)
            return
        link = f'<{_successor(path)}>; rel="successor-version"'

        async def marked(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Deprecation"] = "true"
                headers["Link"] = link
            await send(message)

        await self.app(scope, receive, marked)


def legacy_router(router: APIRouter) -> APIRouter:
    """A router with a deprecated twin, under the old prefix, of each route of
    ``router`` (whose routes carry their full ``/prompt-lens`` paths)."""
    old = APIRouter()
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        path = legacy_path(route.path)
        if path == route.path:
            continue
        old.add_api_route(
            path,
            route.endpoint,
            methods=sorted(route.methods),
            response_model=route.response_model,
            status_code=route.status_code,
            tags=list(route.tags),
            dependencies=list(route.dependencies),
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=dict(route.responses),
            deprecated=True,
            name=f"{route.name}_legacy",
            response_class=route.response_class,
            include_in_schema=route.include_in_schema,
        )
    return old
