"""Route some paths around an inner middleware stack (issue #279 stage 3).

The two CSRF layers are @app.middleware("http") (BaseHTTPMiddleware), which
re-wraps every streamed chunk. /v1 is bearer-authenticated and never needs a
CSRF id or check, so its requests skip that stack entirely; everything else
goes through it unchanged.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send


class PathBypass:
    def __init__(self, app: ASGIApp, *, inner: ASGIApp, prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.inner = inner
        self.prefixes = prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("path", "").startswith(self.prefixes):
            await self.app(scope, receive, send)
        else:
            await self.inner(scope, receive, send)
