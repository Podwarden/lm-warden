"""Serve the static UI export at /ui (BACKLOG-macos P4 part a; Metal design §1).

The existing Next.js app is built as a static export (``output: 'export'``,
``trailingSlash: true``) and served by the same uvicorn process that serves
the api, so a Mac install holds one process (Metal design §1's one-process
topology). The export's index.html, per-directory index.html files and _next
assets are mounted read-only at /ui.

The UI carries its own security headers, set on every /ui response by the
pure-ASGI layer below. They are the same set the standalone UI sends in
frontend/src/lib/security-headers.ts; the two constants here port those
values verbatim (that file is the source of record). The UI's
Content-Security-Policy in particular must win over the api's
``default-src 'none'; sandbox`` default (SecurityHeadersMiddleware): that api
CSP makes any page inert, and a Next.js page needs to run its scripts. It
wins because SecurityHeadersMiddleware fills headers with ``setdefault`` and
this inner layer has already set the CSP.

The wiring task (queued p4c) registers this LAST in ``build_app()`` and only
when ``VW_WEBUI_DIR`` names an export directory; it is not registered here,
so a Linux/CI install without an export is byte-identical.

No GZipMiddleware: it would compress text/event-stream and reproduce issue
#52 (the sparse-SSE buffering bug ``compress: false`` documents in
frontend/next.config.ts).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: UI's Content-Security-Policy -- ports frontend/src/lib/security-headers.ts
#: (UI_CSP) verbatim. Directives only, nothing that breaks a page: no
#: framing, no <base> hijack, no plugins.
UI_CSP = "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"

#: UI's Permissions-Policy -- ports frontend/src/lib/security-headers.ts
#: (PERMISSIONS_POLICY) verbatim; none of the powerful features is used.
UI_PERMISSIONS_POLICY = (
    "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
    "magnetometer=(), microphone=(), payment=(), usb=(), browsing-topics=()"
)

#: The fixed set, in header-name order, set on every /ui response. The CSP
#: here is what makes the page usable in place of the api's default.
_UI_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ("X-Frame-Options", "DENY"),
    ("Permissions-Policy", UI_PERMISSIONS_POLICY),
    ("Content-Security-Policy", UI_CSP),
)


#: Export files whose name carries no extension, so ``mimetypes`` cannot type
#: them and ``FileResponse`` would fall back to text/plain. Next writes a
#: metadata image route into the export under its route name: the
#: apple-icon (frontend/src/app/apple-icon.tsx, ``contentType = "image/png"``)
#: lands as ``apple-icon``, and with nosniff a text/plain icon is unusable.
_EXTENSIONLESS_TYPES: dict[str, str] = {"apple-icon": "image/png"}


class _ExportFiles(StaticFiles):
    """StaticFiles that types the export's extensionless metadata images."""

    def file_response(
        self,
        full_path: os.PathLike[str] | str,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        media = _EXTENSIONLESS_TYPES.get(os.path.basename(full_path))
        if media is not None and isinstance(response, FileResponse):
            response.headers["content-type"] = media
        return response


def _ui_headers(app: ASGIApp) -> ASGIApp:
    """Set the UI's own security headers on every /ui response.

    Pure ASGI: it edits the response-start message and passes the body
    through untouched, so no caching header changes and nothing is buffered
    (a file asset streams straight through).
    """

    async def wrapper(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        async def send_with_ui_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _UI_HEADERS:
                    headers.setdefault(name, value)
            await send(message)

        await app(scope, receive, send_with_ui_headers)

    return wrapper


def mount_webui(app: FastAPI, export_dir: Path) -> None:
    """Mount the UI static export at /ui (Metal design §1).

    ``export_dir`` must exist and hold an ``index.html`` (the trailingSlash
    export's root page); a missing directory or a directory without one is a
    deployment mistake, so it is refused at mount time rather than answered
    with a bare 404 on /ui/ later.

    The ``StaticFiles`` app is wrapped in the pure-ASGI header layer above so
    the UI's headers are set by the innermost middleware the /ui path passes
    through -- ahead of the api's SecurityHeadersMiddleware, whose
    ``setdefault`` therefore leaves them alone.
    """
    if not export_dir.is_dir():
        raise ValueError(f"webui export directory does not exist: {export_dir}")
    if not (export_dir / "index.html").is_file():
        raise ValueError(f"webui export has no index.html: {export_dir}")

    static = _ExportFiles(directory=export_dir, html=True)
    app.mount("/ui", _ui_headers(static), name="webui")
