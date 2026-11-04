"""Core's built-in welcome page: the lowest-priority ``builtin`` home provider."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Request
from fastapi.responses import HTMLResponse, Response

_HTML = (Path(__file__).parent / "welcome.html").read_text(encoding="utf-8")

#: No scripts, no third-party requests, nothing but inline style and data: images.
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class WelcomePage:
    mode: Literal["builtin", "custom"] = "builtin"
    priority = 0

    async def home(self, request: Request) -> Response:
        return HTMLResponse(
            _HTML,
            headers={
                "Content-Security-Policy": CSP,
                "Cache-Control": "no-cache",
                "X-Robots-Tag": "noindex",
            },
        )

    async def not_found(self, request: Request) -> Response | None:
        return None

    async def robots_txt(self, request: Request) -> Response | None:
        return None
