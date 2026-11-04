"""``/``, ``/_landing`` and ``/robots.txt``: the dispatcher over home providers."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response

from app.home import active_provider

router = APIRouter()

DISALLOW_ALL = "User-agent: *\nDisallow: /\n"

#: Paths that are never a home page's 404: the API keeps its JSON body, the
#: console and its assets their own.
_NOT_HOME = ("/api/", "/v1/", "/ui", "/_next/", "/_landing/")


@router.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@router.api_route("/_landing", methods=["GET", "HEAD"], include_in_schema=False)
async def home(request: Request) -> Response:
    _mode, action, provider = await active_provider(request)
    if provider is not None:
        return await provider.home(request)
    if action == "404":
        return Response(status_code=404)
    # Temporary and uncached: it follows a runtime setting, and a browser must
    # not keep it once the home page is switched on.
    return RedirectResponse("/ui/", status_code=307, headers={"Cache-Control": "no-store"})


@router.api_route("/robots.txt", methods=["GET", "HEAD"], include_in_schema=False)
async def robots_txt(request: Request) -> Response:
    _mode, _action, provider = await active_provider(request)
    if provider is not None:
        response = await provider.robots_txt(request)
        if response is not None:
            return response
    return PlainTextResponse(DISALLOW_ALL)


async def not_found_response(request: Request) -> Response | None:
    """The active home provider's HTML 404 for an unknown GET/HEAD path; None
    means the default JSON 404."""
    if request.method not in ("GET", "HEAD") or request.url.path.startswith(_NOT_HOME):
        return None
    mode, _action, provider = await active_provider(request)
    if mode == "off" or provider is None:
        return None
    return await provider.not_found(request)
