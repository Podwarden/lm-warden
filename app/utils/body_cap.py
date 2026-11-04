"""Pure-ASGI request-body cap (verified findings #5, #6). Rejects a declared Content-Length over
the path's cap at once, and counts the bytes of every body as it is read, so a chunked body is
stopped at the cap too. Nothing is buffered: an oversized request costs at most the cap."""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

MB = 1024 * 1024
#: Everything under /api except the chat attachment upload.
API_MAX_BYTES = 16 * MB
#: Anything else (the landing site, health checks).
OTHER_MAX_BYTES = 1 * MB


class _TooLarge(Exception):
    pass


def _cap_for(scope: Scope) -> int:
    path = scope.get("path", "")
    if path.startswith("/v1/"):
        settings = getattr(getattr(scope.get("app"), "state", None), "settings", None)
        return int(getattr(settings, "max_request_body_mb", 64)) * MB
    if path == "/api/chat2/attachments":
        from app.chat2.body_limit import _MAX_DECLARED_BYTES

        return _MAX_DECLARED_BYTES
    if path.startswith("/api/"):
        return API_MAX_BYTES
    return OTHER_MAX_BYTES


async def _too_large(send: Send, cap: int) -> None:
    body = json.dumps(
        {"detail": {"code": "too_large", "message": f"Request body over {cap // MB or 1} MB."}}
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodyCapMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        cap = _cap_for(scope)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > cap:
                        await _too_large(send, cap)
                        return
                except ValueError:
                    pass
                break
        seen = 0
        started = False
        over = False

        async def counting_receive() -> Message:
            nonlocal seen, over
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > cap:
                    over = True
                    raise _TooLarge
            return msg

        async def tracking_send(msg: Message) -> None:
            nonlocal started
            if over:
                # A parser that swallowed _TooLarge (multipart answers its own 400): the answer
                # is the 413, whatever the app tries to send instead.
                if msg["type"] == "http.response.start" and not started:
                    started = True
                    await _too_large(send, cap)
                return
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _TooLarge:
            if not started:
                await _too_large(send, cap)
