"""HTTP for the public website (#155; multi-page since 2026-10-07).

Routes (all unauthenticated, all hidden from the OpenAPI schema):

  GET /_landing, /              Home. Caddy's `handle /` block rewrites the
                                bare root to /_landing.
  GET <page path>               every page in app/landing/pages.py::PAGES
                                (/coding-agents, /faq, ...), plus a 308 from
                                its trailing-slash spelling.
  GET /coding-agents/{client_id} one generated page per Connect client
                                (app/landing/clients.py).
  GET /_landing/assets/{name}   video, screenshots and fonts, from a filename
                                allowlist built at import.
  GET /robots.txt               crawl policy (see "Indexing" below).
  GET /sitemap.xml              every page; only with a canonical origin.
  GET /llms.txt                 llmstxt.org summary for language models.
  GET /llms-full.txt            the same, self-contained, in one markdown file.
  GET /agent-guide.md           the control API, written for a coding agent.

Unknown GET paths get the site's HTML 404 through `not_found_response`,
which app/main.py's 404 handler calls. Rendering lives in render.py.

Disabling. The `landing_page_enabled` setting (default 'false' on a new
install, seeded by migration 0020; a missing row also means off) switches ALL
of the above off. The root (`/` and `/_landing`) then 308-redirects to the
console at /ui/, so a private install opens the product rather than a bare
404. Every other route answers a plain 404 -- not 403, so it behaves like a
feature that is not there rather than a permission boundary -- with one
exception. /robots.txt keeps answering, with disallow-all: a private
deployment that turned its landing page off wants crawlers kept out of everything, and a 404 robots.txt tells a crawler the
opposite ("no rules, crawl what you like").

Indexing is opt-in per instance. Every self-hosted install is private by
default: the page carries `<meta name="robots" content="noindex">` plus an
`X-Robots-Tag: noindex` header, robots.txt disallows everything and there is
no sitemap. Only when the operator sets VW_LANDING_CANONICAL_URL to an
http(s) origin (validated by app.config.parse_canonical_origin, and again
here because Settings can be constructed directly) does the page emit a
canonical link, absolute Open Graph URLs and `index,follow`, and robots.txt
point at /sitemap.xml.

Tracking. When Settings holds a Google Analytics or Ads ID, every page loads
Google's tag behind a consent banner and its CSP allows Google's origins
(render.py::page_csp); with none set, nothing changes.
"""

from __future__ import annotations

import html
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)

from app.config import parse_canonical_origin
from app.landing.clients import CLIENT_PAGES
from app.landing.pages import NOT_FOUND, PAGES, Page
from app.landing.render import (  # noqa: F401  (re-exported for tests)
    _ASSET_CACHE,
    _ASSETS,
    DESCRIPTION,
    INSTALL_CMD,
    RAW_URL,
    REPO_URL,
    TITLE,
    all_site_pages,
    asset_url,
    inline_scripts,
    landing_csp,
    page_csp,
    render_agent_guide,
    render_landing,
    render_llms,
    render_page,
)
from app.landing.site import site_state

router = APIRouter()


# --- Request helpers ------------------------------------------------------------


async def _is_enabled(request: Request) -> bool:
    return (await site_state(request)).enabled


def _origin(request: Request) -> str:
    raw = getattr(request.app.state.settings, "landing_canonical_url", "") or ""
    return parse_canonical_origin(raw)


def _robots_headers(origin: str) -> dict[str, str]:
    return {} if origin else {"X-Robots-Tag": "noindex"}


#: The sitemap's <lastmod>: the newest file the website ships, which in an
#: image is the build. Every page shares it; nothing finer is tracked.
_LASTMOD = (
    datetime.fromtimestamp(
        max(f.stat().st_mtime for f in Path(__file__).parent.rglob("*") if f.is_file()), tz=UTC
    )
    .date()
    .isoformat()
)

_DISALLOW_ALL = "User-agent: *\nDisallow: /\n"


# --- Routes ------------------------------------------------------------------------


async def _serve(request: Request, page: Page) -> Response:
    state = await site_state(request)
    if not state.enabled:
        if page.path == "/":
            # Off: the unified-port root opens the console, not a bare 404.
            # Temporary and uncached: it follows a runtime setting, and a
            # browser must not keep it once the website is switched on.
            return RedirectResponse("/ui/", status_code=307, headers={"Cache-Control": "no-store"})
        return Response(status_code=404)
    if page.tag_only and not state.analytics.tag_on:
        raise HTTPException(status_code=404)  # the site's 404 page
    origin = _origin(request)
    body = render_page(page, origin, state.analytics)
    return HTMLResponse(
        body,
        headers={
            **_robots_headers(origin),
            "Content-Security-Policy": page_csp(body, state.analytics),
        },
    )


def _page_handler(page: Page) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        return await _serve(request, page)

    return handler


def _slash_redirect(path: str) -> Callable[[], Awaitable[Response]]:
    async def handler() -> Response:
        return RedirectResponse(path, status_code=308)

    return handler


# One route per registry page (the home page answers at Caddy's rewrite
# target /_landing and at / for a direct hit), plus a 308 from the
# trailing-slash spelling to the canonical path. A relative Location keeps
# the redirect on whatever host the visitor used.
for _p in PAGES:
    for _path in ("/_landing", "/") if _p.path == "/" else (_p.path,):
        router.add_api_route(
            _path, _page_handler(_p), methods=["GET", "HEAD"], include_in_schema=False
        )
    if _p.path != "/":
        router.add_api_route(
            _p.path + "/",
            _slash_redirect(_p.path),
            methods=["GET", "HEAD"],
            include_in_schema=False,
        )


#: Paths that are never the website's, whatever their status: the API keeps
#: its JSON 404, the console and its assets their own.
_NOT_SITE = ("/api/", "/v1/", "/ui", "/_next/", "/_landing/")


async def not_found_response(request: Request) -> Response | None:
    """The site's HTML 404 for an unknown GET/HEAD path while the website is
    on; None means "not a website 404", so the default handler answers."""
    if request.method not in ("GET", "HEAD") or request.url.path.startswith(_NOT_SITE):
        return None
    state = await site_state(request)
    if not state.enabled:
        return None
    body = render_page(NOT_FOUND, "", state.analytics)  # origin "": always noindex
    return HTMLResponse(
        body,
        status_code=404,
        headers={
            "X-Robots-Tag": "noindex",
            "Content-Security-Policy": page_csp(body, state.analytics),
        },
    )


@router.api_route("/coding-agents/{client_id}", methods=["GET", "HEAD"], include_in_schema=False)
async def client_page(client_id: str, request: Request) -> Response:
    page = CLIENT_PAGES.get(client_id)
    if page is None:
        raise HTTPException(status_code=404)  # the site's 404 page, when on
    return await _serve(request, page)


@router.api_route("/coding-agents/{client_id}/", methods=["GET", "HEAD"], include_in_schema=False)
async def client_page_slash(client_id: str) -> Response:
    # Relative, like the static pages' redirects: Starlette's own
    # redirect_slashes would answer 307 with an absolute http:// URL behind
    # a TLS-terminating proxy.
    return RedirectResponse(f"/coding-agents/{client_id}", status_code=308)


@router.api_route("/_landing/assets/{name}", methods=["GET", "HEAD"], include_in_schema=False)
async def landing_asset(name: str, request: Request) -> Response:
    entry = _ASSETS.get(name)
    if entry is None or not await _is_enabled(request):
        return Response(status_code=404)
    path, ctype, _ = entry
    # FileResponse answers Range requests with 206 (Starlette >= 0.39), which
    # Safari requires before it will play a <video>.
    return FileResponse(
        path,
        media_type=ctype,
        headers={"Cache-Control": _ASSET_CACHE, "X-Content-Type-Options": "nosniff"},
    )


@router.api_route("/robots.txt", methods=["GET", "HEAD"], include_in_schema=False)
async def robots_txt(request: Request) -> Response:
    origin = _origin(request)
    if not origin or not await _is_enabled(request):
        return PlainTextResponse(_DISALLOW_ALL)
    body = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /ui/\n"
        "Disallow: /api/\n"
        "Disallow: /v1/\n"
        f"\nSitemap: {origin}/sitemap.xml\n"
    )
    return PlainTextResponse(body)


@router.api_route("/sitemap.xml", methods=["GET", "HEAD"], include_in_schema=False)
async def sitemap_xml(request: Request) -> Response:
    origin = _origin(request)
    if not origin or not await _is_enabled(request):
        return Response(status_code=404)
    urls = "".join(
        f"  <url><loc>{html.escape(origin + ('/' if p.path == '/' else p.path))}</loc>"
        f"<lastmod>{_LASTMOD}</lastmod></url>\n"
        for p in all_site_pages()
    )
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}</urlset>\n"
    )
    return Response(body, media_type="application/xml")


@router.api_route("/agent-guide.md", methods=["GET", "HEAD"], include_in_schema=False)
async def agent_guide(request: Request) -> Response:
    if not await _is_enabled(request):
        return Response(status_code=404)
    return PlainTextResponse(
        render_agent_guide(_origin(request)),
        media_type="text/markdown; charset=utf-8",
        headers={"X-Robots-Tag": "noindex"},
    )


@router.api_route("/llms.txt", methods=["GET", "HEAD"], include_in_schema=False)
async def llms_txt(request: Request) -> Response:
    if not await _is_enabled(request):
        return Response(status_code=404)
    origin = _origin(request)
    return PlainTextResponse(render_llms(origin, False), headers=_robots_headers(origin))


@router.api_route("/llms-full.txt", methods=["GET", "HEAD"], include_in_schema=False)
async def llms_full_txt(request: Request) -> Response:
    if not await _is_enabled(request):
        return Response(status_code=404)
    origin = _origin(request)
    return PlainTextResponse(render_llms(origin, True), headers=_robots_headers(origin))
