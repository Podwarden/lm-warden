"""HTTP for the public website (#155; multi-page since 2026-10-07).

Routes (all unauthenticated, all hidden from the OpenAPI schema):

  GET /_landing, /              Home. Caddy's `handle /` block rewrites the
                                bare root to /_landing. Picks the visitor's
                                language (see "Languages" below).
  GET /<locale>/                Home translated (/de/, /pt-BR/, ...), for each
                                locale whose catalog ships; 404 otherwise.
                                /<locale> 308s to it.
  GET <page path>               every page in app/landing/pages.py::PAGES
                                (/coding-agents, /faq, ...), plus a 308 from
                                its trailing-slash spelling.
  GET /coding-agents/{client_id} one generated page per Connect client
                                (app/landing/clients.py).
  GET /<locale><page path>      each of those in another language (/de/faq,
  GET /<locale>/coding-agents/{client_id}   /de/coding-agents/aider), where
                                that page's catalogs are complete
                                (i18n.page_ships); the site's 404 otherwise.
  GET /_landing/assets/{name}   video and screenshots, and
  GET /_landing/assets/fonts/{name}  the self-hosted web fonts and their
                                licences, from a filename allowlist built at
                                import.
  GET /robots.txt               crawl policy (see "Indexing" below).
  GET /sitemap.xml              every page; only with a canonical origin.
  GET /llms.txt                 llmstxt.org summary for language models.
  GET /llms-full.txt            the same, self-contained, in one markdown file.
  GET /agent-guide.md           the control API, written for a coding agent.
  GET /download/mac[.sha256]    307 to the Mac beta DMG / its checksum on
                                GitHub (app/landing/downloads.py). Always
                                registered, whatever the canonical URL or the
                                website setting: it is a redirect, not a page.

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

Languages (app/landing/i18n). Every page is translated where its catalogs
are complete. `?lang=<tag>` on any page (the switcher's links, each to the
same page in that language) sets the `lw_lang` cookie for a year: when
<tag> is the page already shown it is served, otherwise the request 302s to
the same page in <tag>. Without ?lang, a GET by a browser (not a crawler, not
HEAD) of the English Home honours the cookie, then Accept-Language, and 302s
to the matching /<locale>/; a GET of any other English page honours the
cookie only, and 302s to that page in the cookie's language where it is
published. English or no match stays. Those responses vary, so they say
`Vary: Accept-Language, Cookie` (Home) or `Vary: Cookie` (other pages) and
`Cache-Control: private`. A translated URL is never redirected but by
?lang.

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

from app.auth.cookies import cookie_secure
from app.config import parse_canonical_origin
from app.landing import i18n
from app.landing.clients import CLIENT_PAGES
from app.landing.downloads import MAC_DMG_URL, MAC_SHA256_URL
from app.landing.pages import CLIENT_PATH, HOME, LOCALES, NOT_FOUND, PAGES, Page
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


async def _serve(request: Request, page: Page, locale: str = "en") -> Response:
    state = await site_state(request)
    if not state.enabled:
        if page.path == "/" and locale == "en":
            # Off: the unified-port root opens the console, not a bare 404.
            # Temporary and uncached: it follows a runtime setting, and a
            # browser must not keep it once the website is switched on.
            return RedirectResponse("/ui/", status_code=307, headers={"Cache-Control": "no-store"})
        return Response(status_code=404)
    if locale != "en" and not i18n.page_ships(page, locale):
        raise HTTPException(status_code=404)  # the site's 404 page, in that language
    if page.tag_only and not state.analytics.tag_on:
        raise HTTPException(status_code=404)  # the site's 404 page
    chosen = _language_redirect(request, page, locale)
    if chosen is not None:
        return chosen
    origin = _origin(request)
    body = render_page(page, origin, state.analytics, locale)
    headers = {
        **_robots_headers(origin),
        "Content-Security-Policy": page_csp(body, state.analytics),
    }
    if locale == "en":
        headers.update(_vary(page))
    response = HTMLResponse(body, headers=headers)
    if request.query_params.get("lang") == locale:
        _remember_language(response, request, locale)
    return response


def _remember_language(response: Response, request: Request, tag: str) -> None:
    """The switcher's choice, kept a year (lw_lang)."""
    response.set_cookie(
        i18n.COOKIE,
        tag,
        max_age=i18n.COOKIE_MAX_AGE,
        path="/",
        samesite="lax",
        secure=cookie_secure(request),
        httponly=True,
    )


#: On every answer of the English Home: it depends on the cookie and on
#: Accept-Language, so no shared cache may hand one visitor's to another.
_NEGOTIATED = {"Vary": "Accept-Language, Cookie", "Cache-Control": "private"}
#: On an English page other than Home that is published in another language
#: too: the cookie may redirect it.
_COOKIE_VARY = {"Vary": "Cookie", "Cache-Control": "private"}


def _vary(page: Page) -> dict[str, str]:
    """The caching headers of a page's English answer (and its redirects)."""
    if page.path == "/":
        return _NEGOTIATED
    return _COOKIE_VARY if len(i18n.page_locales(page)) > 1 else {}


def _language_redirect(request: Request, page: Page, locale: str) -> Response | None:
    """The redirect a page request gets, or None to serve the page.

    ?lang=<tag> where this page is published in <tag>: remember it (lw_lang,
    a year) and go to this page in that language. Then, on an English page,
    for a GET from a browser: on Home the cookie, else Accept-Language, may
    send it to /<locale>/; on any other page only the cookie may, to the
    same page in that language. HEAD and crawlers are always served the page
    they asked for."""
    asked = request.query_params.get("lang")
    if asked is not None and asked != locale and i18n.page_ships(page, asked):
        response = RedirectResponse(
            i18n.page_path(page, asked), status_code=302, headers=_vary(page) or _COOKIE_VARY
        )
        _remember_language(response, request, asked)
        return response
    if asked == locale:
        # Already on the asked language's page: serve it (the caller sets the
        # cookie). Redirecting to the clean URL would let Accept-Language send
        # a visitor whose browser drops the cookie straight back, so with
        # cookies blocked and a French browser, English would be unreachable.
        return None
    if locale != "en" or request.method != "GET":
        return None
    if i18n.is_bot(request.headers.get("user-agent")):
        return None
    cookie = request.cookies.get(i18n.COOKIE)
    if page.path == "/":
        tag = i18n.negotiate(request.headers.get("accept-language"), cookie)
    else:
        tag = i18n.cookie_locale(page, cookie)
    if tag == "en":
        return None
    return RedirectResponse(i18n.page_path(page, tag), status_code=302, headers=_vary(page))


def _page_handler(page: Page) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        return await _serve(request, page)

    return handler


def _locale_handler(tag: str, page: Page = HOME) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        return await _serve(request, page, tag)

    return handler


def _locale_slash_redirect(tag: str) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        query = request.url.query
        return RedirectResponse(f"/{tag}/" + (f"?{query}" if query else ""), status_code=308)

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


def _client_handler(tag: str) -> Callable[[str, Request], Awaitable[Response]]:
    async def handler(client_id: str, request: Request) -> Response:
        page = CLIENT_PAGES.get(client_id)
        if page is None:
            raise HTTPException(status_code=404)  # the site's 404 page, when on
        return await _serve(request, page, tag)

    return handler


def _client_slash_redirect(tag: str) -> Callable[[str], Awaitable[Response]]:
    async def handler(client_id: str) -> Response:
        return RedirectResponse(f"/{tag}/coding-agents/{client_id}", status_code=308)

    return handler


# Home in each other language: /de/ (the canonical spelling) and /de, which
# redirects; then every other page in it (/de/faq). Registered for every
# locale; a page whose catalogs are not complete there answers 404.
for _loc in LOCALES:
    if _loc.tag == "en":
        continue
    router.add_api_route(
        f"/{_loc.tag}/", _locale_handler(_loc.tag), methods=["GET", "HEAD"], include_in_schema=False
    )
    router.add_api_route(
        f"/{_loc.tag}",
        _locale_slash_redirect(_loc.tag),
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    # Every other page in that language, and its trailing-slash spelling.
    for _p in PAGES:
        if _p.path == "/":
            continue
        _path = f"/{_loc.tag}{_p.path}"
        router.add_api_route(
            _path, _locale_handler(_loc.tag, _p), methods=["GET", "HEAD"], include_in_schema=False
        )
        router.add_api_route(
            _path + "/", _slash_redirect(_path), methods=["GET", "HEAD"], include_in_schema=False
        )
    router.add_api_route(
        f"/{_loc.tag}{CLIENT_PATH}",
        _client_handler(_loc.tag),
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    router.add_api_route(
        f"/{_loc.tag}{CLIENT_PATH}/",
        _client_slash_redirect(_loc.tag),
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
    # Under a locale's prefix (/de/nope), in that language where it is published.
    tag, _ = i18n.split_locale(request.url.path)
    locale = tag if i18n.page_ships(NOT_FOUND, tag) else "en"
    body = render_page(NOT_FOUND, "", state.analytics, locale)  # origin "": always noindex
    return HTMLResponse(
        body,
        status_code=404,
        headers={
            "X-Robots-Tag": "noindex",
            "Content-Security-Policy": page_csp(body, state.analytics),
        },
    )


def _download_redirect(url: str) -> Callable[[], Awaitable[Response]]:
    async def handler() -> Response:
        # Temporary and uncached: the next beta changes the target.
        return RedirectResponse(url, status_code=307, headers={"Cache-Control": "no-store"})

    return handler


router.add_api_route(
    "/download/mac",
    _download_redirect(MAC_DMG_URL),
    methods=["GET", "HEAD"],
    include_in_schema=False,
)
router.add_api_route(
    "/download/mac.sha256",
    _download_redirect(MAC_SHA256_URL),
    methods=["GET", "HEAD"],
    include_in_schema=False,
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
    return await _send_asset(name, request)


@router.api_route("/_landing/assets/fonts/{name}", methods=["GET", "HEAD"], include_in_schema=False)
async def landing_font(name: str, request: Request) -> Response:
    # The index keys assets/fonts/ files as "fonts/<name>"; {name} never
    # holds a "/", so this names exactly one file of that folder or nothing.
    return await _send_asset("fonts/" + name, request)


async def _send_asset(name: str, request: Request) -> Response:
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
    # Every page, each followed by its published translations.
    paths = [i18n.page_path(p, loc.tag) for p in all_site_pages() for loc in i18n.page_locales(p)]
    urls = "".join(
        f"  <url><loc>{html.escape(origin + path)}</loc><lastmod>{_LASTMOD}</lastmod></url>\n"
        for path in paths
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
