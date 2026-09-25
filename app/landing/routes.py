"""The public landing page and its companions (#155, landing redesign).

Routes (all unauthenticated, all hidden from the OpenAPI schema):

  GET /_landing                 the landing page. Caddy's `handle /` block
                                rewrites the bare root to it.
  GET /_landing/assets/{name}   the page's video, screenshots and fonts, from
                                a filename allowlist built at import.
  GET /robots.txt               crawl policy (see "Indexing" below).
  GET /sitemap.xml              only when a canonical origin is configured.
  GET /llms.txt                 llmstxt.org summary for language models.
  GET /llms-full.txt            the same, self-contained, in one markdown file.

Disabling. The `landing_page_enabled` setting (default 'true', seeded by
migration 0020) switches ALL of the above off: each answers a plain 404 --
not 403, so the root behaves like a feature that is not there rather than a
permission boundary -- with one exception. /robots.txt keeps answering, with
disallow-all: a private deployment that turned its landing page off wants
crawlers kept out of everything, and a 404 robots.txt tells a crawler the
opposite ("no rules, crawl what you like").

Indexing is opt-in per instance. Every self-hosted install is private by
default: the page carries `<meta name="robots" content="noindex">` plus an
`X-Robots-Tag: noindex` header, robots.txt disallows everything and there is
no sitemap. Only when the operator sets VW_LANDING_CANONICAL_URL to an
http(s) origin (validated by app.config.parse_canonical_origin, and again
here because Settings can be constructed directly) does the page emit a
canonical link, absolute Open Graph URLs and `index,follow`, and robots.txt
point at /sitemap.xml.

Everything is read once at import time. landing.html, llms.txt,
llms-full.txt and faq.json ship next to this module; the page is rendered
once per canonical origin and cached. Hot-editing them in a running container
is deliberately unsupported -- rebuild the image.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

from app.config import parse_canonical_origin
from app.db.database import open_db
from app.db.repos.settings import SettingsRepo

router = APIRouter()

_HERE = Path(__file__).parent
_ASSET_DIR = _HERE / "assets"

#: The public repository every link on the page points at.
REPO_URL = "https://github.com/Podwarden/lm-warden"
#: Raw-file base for llms.txt links (llmstxt.org prefers markdown targets).
RAW_URL = "https://raw.githubusercontent.com/Podwarden/lm-warden/main"
#: The documented no-clone one-liner (documents/INSTALL.md, "A6. Without a
#: clone"), on the renamed repository. The install directory keeps its old
#: identity slug on purpose: install.sh defaults to /opt/vllm-warden and an
#: existing install must still be found there.
INSTALL_CMD = (
    "curl -fsSL https://raw.githubusercontent.com/Podwarden/lm-warden/main/install.sh"
    " | sh -s -- --dir /opt/vllm-warden"
)

# --- Assets ------------------------------------------------------------------

#: Content types by suffix. A file with any other suffix is never served,
#: whatever is in the directory.
_ASSET_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
_ASSET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
#: Asset URLs carry ?v=<content hash>, so a response can be cached for a year
#: and marked immutable: a changed file gets a new URL.
_ASSET_CACHE = "public, max-age=31536000, immutable"


def _build_asset_index() -> dict[str, tuple[Path, str, str]]:
    """name -> (path, content type, short content hash). Built once; a request
    can only ever name a key of this dict, so there is no path to traverse."""
    index: dict[str, tuple[Path, str, str]] = {}
    if not _ASSET_DIR.is_dir():
        return index
    for path in sorted(_ASSET_DIR.iterdir()):
        ctype = _ASSET_TYPES.get(path.suffix.lower())
        if ctype is None or not path.is_file() or path.is_symlink():
            continue
        if not _ASSET_NAME_RE.match(path.name) or ".." in path.name:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        index[path.name] = (path, ctype, digest)
    return index


_ASSETS = _build_asset_index()


def asset_url(name: str) -> str:
    """Versioned, root-relative URL of an asset. KeyError for an unknown name,
    so a typo in the template fails at import (and in the tests), not as a
    broken image in production."""
    _, _, digest = _ASSETS[name]
    return f"/_landing/assets/{name}?v={digest}"


# --- FAQ: one source for the visible HTML, the JSON-LD and llms-full.txt ------

_FAQ: list[dict[str, str]] = json.loads((_HERE / "faq.json").read_text(encoding="utf-8"))
_TAG_RE = re.compile(r"<[^>]+>")
_LINK_RE = re.compile(r'<a href="([^"]+)">(.*?)</a>')


def _faq_text(answer_html: str) -> str:
    """Plain text of an answer, as a reader of the page sees it."""
    return html.unescape(_TAG_RE.sub("", answer_html))


def _faq_markdown(answer_html: str) -> str:
    md = _LINK_RE.sub(lambda m: f"[{m.group(2)}]({m.group(1)})", answer_html)
    md = md.replace("<code>", "`").replace("</code>", "`")
    md = md.replace("<strong>", "**").replace("</strong>", "**")
    return html.unescape(_TAG_RE.sub("", md))


def _faq_html() -> str:
    """Every answer visible: no accordion hides the page's plainest text."""
    items = []
    for i, qa in enumerate(_FAQ):
        items.append(
            f'<div class="qa" id="faq-{i + 1}">\n'
            f"  <h3>{html.escape(qa['q'])}</h3>\n"
            f"  <p>{qa['a']}</p>\n"
            "</div>"
        )
    return "\n".join(items)


# --- Page rendering -----------------------------------------------------------

_TOKEN_RE = re.compile(r"\{\{([a-z_]+)(?::([a-z0-9._-]+))?\}\}")
_LANDING_SRC: str = (_HERE / "landing.html").read_text(encoding="utf-8")
_LLMS_SRC: str = (_HERE / "llms.txt").read_text(encoding="utf-8")
_LLMS_FULL_SRC: str = (_HERE / "llms-full.txt").read_text(encoding="utf-8")

#: "LLM" stays in the title and description for search; the product is
#: "LM Warden".
TITLE = "LM Warden: self-hosted LLM gateway for your own GPUs"
DESCRIPTION = (
    "Serve LLMs from your own NVIDIA GPUs with vLLM and llama.cpp: one "
    "OpenAI-compatible /v1 endpoint, a key per app, and readings per card."
)
_OG_IMAGE = "og-image.jpg"


def _jsonld(origin: str) -> str:
    def absolute(path: str) -> str:
        return origin + path

    app: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "LM Warden",
        "description": DESCRIPTION,
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "Linux",
        "softwareRequirements": (
            "Linux x86_64, Docker with Compose v2.24+, an NVIDIA GPU (Turing or newer) "
            "and the NVIDIA Container Toolkit"
        ),
        "license": "https://www.apache.org/licenses/LICENSE-2.0",
        "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        "downloadUrl": REPO_URL,
        "sameAs": [REPO_URL],
        "screenshot": absolute(asset_url(_OG_IMAGE)),
    }
    if origin:
        app["url"] = origin + "/"
    faq = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": qa["q"],
                "acceptedAnswer": {"@type": "Answer", "text": _faq_text(qa["a"])},
            }
            for qa in _FAQ
        ],
    }
    blocks = []
    for obj in (app, faq):
        # "<" escaped so no string inside can close the <script> element.
        payload = json.dumps(obj, ensure_ascii=False, indent=1).replace("<", "\\u003c")
        blocks.append(f'<script type="application/ld+json">\n{payload}\n</script>')
    return "\n".join(blocks)


def _head(origin: str) -> str:
    """The origin-dependent part of <head>: robots, canonical and OG URLs."""
    e = html.escape
    image = e(origin + asset_url(_OG_IMAGE))
    lines = []
    if origin:
        lines += [
            '<meta name="robots" content="index,follow">',
            f'<link rel="canonical" href="{e(origin)}/">',
            f'<meta property="og:url" content="{e(origin)}/">',
        ]
    else:
        lines.append('<meta name="robots" content="noindex">')
    lines += [
        f'<meta property="og:image" content="{image}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        '<meta property="og:image:type" content="image/jpeg">',
        '<meta property="og:image:alt" content="LM Warden: an OpenAI-compatible API on the '
        'GPUs under your desk or in your rack, beside the Stats panel of one RTX A4000 with its '
        'driver-reported throttle point labelled">',
        f'<meta name="twitter:image" content="{image}">',
    ]
    return "\n  ".join(lines)


def _fill(src: str, values: dict[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        key, arg = m.group(1), m.group(2)
        if key == "a" and arg:
            return asset_url(arg)
        return values[key]

    return _TOKEN_RE.sub(sub, src)


#: A "/" that ends a path segment ("com/", "main/") or the scheme ("https://"),
#: not the leading "/" of an absolute path.
_SEGMENT_SLASH_RE = re.compile(r"(?<=[\w.-])/(?!/)|(?<=://)")
#: A word with a hyphen in it ("lm-warden", "--dir", "-fsSL"): kept on one
#: line, since a browser may break after a hyphen and "--" / "dir" would
#: read as two arguments.
_HYPHENATED_RE = re.compile(r"(?<![\w-])(-+\w*(?:-\w+)*|\w+(?:-\w+)+)(?![\w-])")


def _install_html() -> str:
    """The install command for a <pre>: a <wbr> after each "/" of the URL and
    path, and hyphenated words held together, so a phone wraps it at a path
    boundary or a space and never mid-word. Neither <wbr> nor the <span>
    adds text, so the copy button (which copies textContent) still copies
    the command exactly."""
    wrapped = _SEGMENT_SLASH_RE.sub(lambda m: m.group(0) + "<wbr>", html.escape(INSTALL_CMD))
    return _HYPHENATED_RE.sub(r'<span class="nw">\1</span>', wrapped)


@lru_cache(maxsize=8)
def render_landing(origin: str) -> str:
    """The page for one canonical origin ("" = private instance)."""
    return _fill(
        _LANDING_SRC,
        {
            "title": html.escape(TITLE),
            "description": html.escape(DESCRIPTION),
            "head": _head(origin),
            "jsonld": _jsonld(origin),
            "faq": _faq_html(),
            "repo": REPO_URL,
            "install": _install_html(),
        },
    )


@lru_cache(maxsize=8)
def render_llms(origin: str, full: bool) -> str:
    faq_md = "\n\n".join(f"### {qa['q']}\n\n{_faq_markdown(qa['a'])}" for qa in _FAQ)
    return _fill(
        _LLMS_FULL_SRC if full else _LLMS_SRC,
        {
            "repo": REPO_URL,
            "raw": RAW_URL,
            "origin": origin,
            "install": INSTALL_CMD,
            "faq": faq_md,
            "description": DESCRIPTION,
        },
    )


# --- Content-Security-Policy ------------------------------------------------------
#
# The page is static: no framework, fonts and media from this instance, two
# inline scripts (in <head>, the stored colour theme applied before the first
# paint; at the end of <body>, the theme toggle, copy buttons and video
# autoplay) and two JSON-LD data blocks, which a browser never executes and
# CSP does not govern. So the policy can be strict: nothing loads from
# anywhere but this origin, and the only scripts that run are the ones whose
# hashes are in the header -- every executable inline <script> in the page
# served, whatever attributes it carries. A script injected into the page by
# any means -- a tampered template, a future templated value that escapes
# badly -- simply does not run.
#
# Styles keep 'unsafe-inline': the page has one <style> block and a few style
# attributes, and hashing attributes needs 'unsafe-hashes', which older
# Safari does not honour. CSS injection on a page with no secrets and no
# user input is not worth breaking the layout for.
#
# The UI (Next.js) has no CSP yet -- see frontend/src/lib/security-headers.ts.

_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)
_SCRIPT_TYPE_RE = re.compile(r"""\btype\s*=\s*["']?([^"'\s>]+)""", re.IGNORECASE)
#: `type` values a browser executes; anything else (application/ld+json) is data.
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}


def inline_scripts(page: str) -> list[str]:
    """The body of every inline <script> a browser would execute, in order.
    External scripts (src=) and data blocks are skipped."""
    bodies = []
    for attrs, body in _SCRIPT_RE.findall(page):
        if re.search(r"\bsrc\s*=", attrs, re.IGNORECASE):
            continue
        m = _SCRIPT_TYPE_RE.search(attrs)
        if (m.group(1).lower() if m else "") in _JS_TYPES:
            bodies.append(body)
    return bodies


@lru_cache(maxsize=8)
def landing_csp(page: str) -> str:
    hashes = " ".join(
        dict.fromkeys(
            "'sha256-" + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode()
            + "'"
            for body in inline_scripts(page)
        )
    )
    script_src = hashes or "'none'"
    return (
        "default-src 'none'; "
        f"script-src {script_src}; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self'; media-src 'self'; font-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


# Render the private variant at import: a broken template token or an asset
# the page names but the image does not ship fails the process at start-up.
render_landing("")
render_llms("", False)
render_llms("", True)


# --- Request helpers ------------------------------------------------------------


async def _is_enabled(request: Request) -> bool:
    """Return True iff the `landing_page_enabled` setting is truthy.

    Defaults to True if the row is missing (defensive — migration 0020
    seeds it, but a partially-bootstrapped DB shouldn't silently mask
    the landing page).
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        raw = await SettingsRepo(db).get("landing_page_enabled")
    if raw is None:
        return True
    # Stored as canonical 'true' / 'false' by the settings coercer, but
    # tolerate any common truthy spelling for forward-compat with manual
    # edits to the settings table.
    return raw.strip().lower() in {"true", "1", "yes", "on"}


def _origin(request: Request) -> str:
    raw = getattr(request.app.state.settings, "landing_canonical_url", "") or ""
    return parse_canonical_origin(raw)


def _robots_headers(origin: str) -> dict[str, str]:
    return {} if origin else {"X-Robots-Tag": "noindex"}


_DISALLOW_ALL = "User-agent: *\nDisallow: /\n"


# --- Routes ------------------------------------------------------------------------


@router.api_route("/_landing", methods=["GET", "HEAD"], include_in_schema=False)
async def landing(request: Request) -> Response:
    if not await _is_enabled(request):
        return Response(status_code=404)
    origin = _origin(request)
    page = render_landing(origin)
    return HTMLResponse(
        page,
        headers={**_robots_headers(origin), "Content-Security-Policy": landing_csp(page)},
    )


@router.api_route(
    "/_landing/assets/{name}", methods=["GET", "HEAD"], include_in_schema=False
)
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
    loc = html.escape(origin + "/")
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{loc}</loc></url>\n"
        "</urlset>\n"
    )
    return Response(body, media_type="application/xml")


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
