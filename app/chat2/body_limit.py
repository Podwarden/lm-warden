"""Pure-ASGI request-size guard for the chat2 attachment upload route.

FastAPI resolves route dependencies (the ``UploadFile``/``Form`` parameters
on ``POST /api/chat2/attachments``) by awaiting ``Request.form()``, which
spools the ENTIRE multipart body to memory/disk before the route function
body ever runs. An in-route Content-Length check (see
``routes_attachments._reject_oversized_content_length``) is therefore too
late to stop a large declared upload from being read in the first place.

This middleware sits outside FastAPI's routing entirely -- it's a raw ASGI
callable (not a Starlette ``BaseHTTPMiddleware``/``@app.middleware("http")``
dispatch function), registered ahead of routing in ``app.main.build_app``
-- so it can reject an oversized declared Content-Length before a single
byte of the request body is read.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Receive, Scope, Send

from app.chat2.limits import MAX_IMAGE_BYTES, MULTIPART_OVERHEAD_BYTES

_GUARDED_METHOD = "POST"
_GUARDED_PATH = "/api/chat2/attachments"
_MAX_DECLARED_BYTES = MAX_IMAGE_BYTES + MULTIPART_OVERHEAD_BYTES


class Chat2BodyLimitMiddleware:
    """Reject ``POST /api/chat2/attachments`` with 413 if its declared
    ``Content-Length`` exceeds ``MAX_IMAGE_BYTES + MULTIPART_OVERHEAD_BYTES``,
    without ever reading the request body."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != _GUARDED_METHOD
            or scope.get("path") != _GUARDED_PATH
        ):
            await self.app(scope, receive, send)
            return

        declared: int | None = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                break

        if declared is not None and declared > _MAX_DECLARED_BYTES:
            body = json.dumps(
                {
                    "detail": {
                        "code": "too_large",
                        "message": f"request body larger than {MAX_IMAGE_BYTES} bytes",
                    }
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
