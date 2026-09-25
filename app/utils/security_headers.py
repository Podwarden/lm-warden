"""Security headers on every api response, set by the application itself.

The api is reached through whatever front door the deployment has -- our
Caddyfile, a Hub-installed stack's own Caddyfile, Traefik, or nothing at all
-- so these cannot be left to the proxy. The UI sets the same set in
frontend/next.config.ts (frontend/src/lib/security-headers.ts).

* ``X-Content-Type-Options: nosniff`` -- a response is only ever what its
  Content-Type says (god-mode media and chat2 attachments are user-supplied
  bytes).
* ``Referrer-Policy: strict-origin-when-cross-origin`` -- no paths or query
  strings (signed attachment URLs, SSE tickets) leak to other origins.
* ``X-Frame-Options: DENY`` plus CSP ``frame-ancestors 'none'`` -- nothing is
  meant to be framed; no clickjacking of the console or the landing page.
* ``Permissions-Policy`` -- none of the powerful browser features is used.
* ``Content-Security-Policy: default-src 'none'; frame-ancestors 'none';
  sandbox`` -- api responses are JSON, SSE, text or media, never a page that
  should run anything, so a response opened directly in a tab is inert. The
  landing page sets its own, real policy (app/landing/routes.py) and a route's
  own header always wins over these defaults.
* ``Strict-Transport-Security: max-age=31536000`` -- only when the request
  reached us over HTTPS as the browser saw it, derived exactly like the
  session cookie's Secure flag (app/auth/cookies.py::cookie_secure). A
  plain-HTTP install must never be told to insist on HTTPS: it has none.

Pure ASGI: it edits the response-start message and passes the body through
untouched, so SSE streams are not buffered and no caching header changes.
"""

from __future__ import annotations

import logging

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.auth.cookies import cookie_secure

log = logging.getLogger(__name__)

PERMISSIONS_POLICY = (
    "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
    "magnetometer=(), microphone=(), payment=(), usb=(), browsing-topics=()"
)
API_CSP = "default-src 'none'; frame-ancestors 'none'; sandbox"
HSTS = "max-age=31536000"

BASE_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ("X-Frame-Options", "DENY"),
    ("Permissions-Policy", PERMISSIONS_POLICY),
    ("Content-Security-Policy", API_CSP),
)


def _is_https(scope: Scope) -> bool:
    try:
        return cookie_secure(Request(scope))
    except Exception:  # noqa: BLE001 -- settings not loaded yet (pre-lifespan)
        return False


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in BASE_HEADERS:
                    headers.setdefault(name, value)
                if _is_https(scope):
                    headers.setdefault("Strict-Transport-Security", HSTS)
            await send(message)

        await self.app(scope, receive, send_with_headers)
