"""Rendering for the public website (spec 2026-10-07 §4).

Everything is read once at import time: layout.html, the page fragments under
pages/, llms.txt, llms-full.txt and faq.json ship next to this module, and
each page is rendered once per (canonical origin, tracking IDs) and cached.
Hot-editing them in a running container is deliberately unsupported --
rebuild the image. HTTP lives in routes.py.
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

from app.landing.downloads import MAC_DMG_NAME, MAC_DMG_URL, MAC_RELEASE_URL
from app.landing.pages import HOME, PAGES, Page
from app.landing.site import Analytics

_HERE = Path(__file__).parent
_ASSET_DIR = _HERE / "assets"
_FIGURE_DIR = _HERE / "figures"

#: The public repository every link on the page points at.
REPO_URL = "https://github.com/Podwarden/lm-warden"
#: Raw-file base for llms.txt links (llmstxt.org prefers markdown targets).
RAW_URL = "https://raw.githubusercontent.com/Podwarden/lm-warden/main"
#: The documented no-clone one-liner (documents/INSTALL.md, "A6. Without a
#: clone"), on the renamed repository. The install directory keeps its old
#: identity slug on purpose: install.sh defaults to /opt/vllm-warden and an
#: existing install must still be found there.
INSTALL_SCRIPT_URL = "https://raw.githubusercontent.com/Podwarden/lm-warden/main/install.sh"
INSTALL_CMD = f"curl -fsSL {INSTALL_SCRIPT_URL} | sh -s -- --dir /opt/vllm-warden"

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


#: The one subdirectory of assets/ that is served: the self-hosted web fonts
#: and their licences, keyed "fonts/<name>". Nothing deeper is ever indexed.
ASSET_SUBDIRS = ("fonts",)


def _build_asset_index() -> dict[str, tuple[Path, str, str]]:
    """name -> (path, content type, short content hash). Built once; a request
    can only ever name a key of this dict, so there is no path to traverse.
    Files in assets/fonts/ are keyed "fonts/<name>"."""
    index: dict[str, tuple[Path, str, str]] = {}
    if not _ASSET_DIR.is_dir():
        return index
    dirs = [("", _ASSET_DIR)] + [
        (sub + "/", _ASSET_DIR / sub)
        for sub in ASSET_SUBDIRS
        if (_ASSET_DIR / sub).is_dir() and not (_ASSET_DIR / sub).is_symlink()
    ]
    for prefix, folder in dirs:
        for path in sorted(folder.iterdir()):
            ctype = _ASSET_TYPES.get(path.suffix.lower())
            if ctype is None or not path.is_file() or path.is_symlink():
                continue
            if not _ASSET_NAME_RE.match(path.name) or ".." in path.name:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            index[prefix + path.name] = (path, ctype, digest)
    return index


_ASSETS = _build_asset_index()


def asset_url(name: str) -> str:
    """Versioned, root-relative URL of an asset. KeyError for an unknown name,
    so a typo in the template fails at import (and in the tests), not as a
    broken image in production."""
    _, _, digest = _ASSETS[name]
    return f"/_landing/assets/{name}?v={digest}"


# --- Figures: inline SVG and table fragments, read once --------------------------
#
# `{{fig:name}}` in landing.html is replaced by the text of figures/<name>.svg
# (or .html). They are inline, not assets: inline SVG follows the page's theme
# tokens (`var(--ink)`), and the asset allowlist above has no .svg on purpose.
# d1-flow.svg is drawn by hand; the others are written by
# scripts/bench-figures.py. A name with no file raises at import, like
# asset_url.


def _build_figure_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if _FIGURE_DIR.is_dir():
        for path in sorted(_FIGURE_DIR.iterdir()):
            if path.suffix in {".svg", ".html"} and path.is_file() and not path.is_symlink():
                index[path.stem] = path.read_text(encoding="utf-8").strip()
    return index


_FIGURES = _build_figure_index()


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

_TOKEN_RE = re.compile(r"\{\{([a-z_]+)(?::([a-z0-9._/-]+))?\}\}")
_LAYOUT_SRC: str = (_HERE / "layout.html").read_text(encoding="utf-8")
_FRAGMENTS: dict[str, str] = {
    p.name: p.read_text(encoding="utf-8") for p in sorted((_HERE / "pages").glob("*.html"))
}
_LLMS_SRC: str = (_HERE / "llms.txt").read_text(encoding="utf-8")
_LLMS_FULL_SRC: str = (_HERE / "llms-full.txt").read_text(encoding="utf-8")
_AGENT_GUIDE_SRC: str = (_HERE / "agent-guide.md").read_text(encoding="utf-8")
#: Page CSS, appended to the layout's shared <style> block: home.css for Home
#: (the console-first landing page), content.css for every other page (the
#: article styles the pages were written against, on the same dark tokens).
_STYLES: dict[str, str] = {
    p.stem: p.read_text(encoding="utf-8").strip() for p in sorted((_HERE / "styles").glob("*.css"))
}

TITLE = HOME.title
DESCRIPTION = HOME.description
_OG_IMAGE = "og-image.jpg"

#: What the home page shows LM Warden doing, one line each, for the
#: SoftwareApplication's featureList (its carousel and sections, in order).
FEATURES: tuple[str, ...] = (
    "Fit preview: every file of a Hugging Face repo checked against the GPUs you pick "
    "before anything downloads",
    "Coding agents (Codex CLI, OpenCode, Aider, Grok CLI, Continue, Cline, Cursor, the "
    "OpenAI and Anthropic SDKs, Claude Code) on your own models, configured for you",
    "Claude Code router: a local model answers for the Claude models you choose, the rest "
    "go to Anthropic on your own login",
    "Stats page: GPU utilisation, power, tokens per second, cache share, latency and "
    "preemptions, 1 hour to 7 days",
    "Cache-aware routing that keeps each conversation on the replica holding its prefix",
    "Agent tuning: a coding agent picks the model, engine and layout for your GPUs through "
    "the API, benchmarking as it goes",
    "Chat with streaming Markdown, images for vision models and saved history",
    "God mode: one key's prompts and replies streamed live for debugging, off by default",
    "API keys with strict priority 9 to 0 and per-key usage",
    "Per-GPU panels from a 16 GB M1 to an 8 x B300 NVSwitch server, naming narrow PCIe "
    "links, inactive NVLink and ECC differences",
)


#: The project's own public instance. The privacy page names PodWarden as
#: the contact only there; any other operator's site names its operator.
OFFICIAL_ORIGIN = "https://lmwarden.com"

#: Who makes LM Warden, for structured data (spec 2026-10-07: LM Warden
#: exists to bring people to PodWarden).
PODWARDEN: dict[str, Any] = {
    "@type": "Organization",
    "name": "PodWarden",
    "url": "https://podwarden.com",
    "sameAs": ["https://github.com/Podwarden"],
}


def _url(origin: str, page: Page) -> str:
    return origin + ("/" if page.path == "/" else page.path)


def _jsonld(origin: str, page: Page) -> str:
    def absolute(path: str) -> str:
        return origin + path

    app: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "LM Warden",
        "description": DESCRIPTION,
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "Linux, macOS (beta)",
        "softwareRequirements": (
            "Linux x86_64, Docker with Compose v2.24+, an NVIDIA GPU (Turing or newer) "
            "and the NVIDIA Container Toolkit; or, for the Mac beta, an Apple silicon "
            "Mac on macOS 14 or later"
        ),
        "license": "https://www.apache.org/licenses/LICENSE-2.0",
        "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        # One download per platform that ships today: the Linux installer
        # script and the Mac beta DMG. Windows (next) and the appliance
        # (coming soon) have nothing to download yet, so they are not here.
        "downloadUrl": [INSTALL_SCRIPT_URL, MAC_DMG_URL],
        "featureList": list(FEATURES),
        "sameAs": [REPO_URL],
        "screenshot": absolute(asset_url(_OG_IMAGE)),
        "publisher": PODWARDEN,
        "author": PODWARDEN,
    }
    if origin:
        app["url"] = origin + "/"
        app["installUrl"] = origin + "/#install"
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
    website: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "LM Warden",
        "publisher": PODWARDEN,
    }
    article: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": page.title.split(" | ")[0],
        "description": page.description,
        "publisher": PODWARDEN,
    }
    howto: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": "Let your coding agent pick and tune the model",
        "description": page.description,
        "step": [
            {"@type": "HowToStep", "position": i, "name": name}
            for i, name in enumerate(
                (
                    "Install LM Warden on the GPU host",
                    "Issue an admin token",
                    "Connect Claude Code or Codex",
                    "Paste a tuning prompt",
                ),
                start=1,
            )
        ],
    }
    if origin:
        website["url"] = origin + "/"
        article["url"] = _url(origin, page)
    chosen = {"app": app, "website": website, "faq": faq, "article": article, "howto": howto}
    objs = [chosen[k] for k in page.jsonld if k in chosen]
    if page.crumbs:
        trail = [("Home", "/"), *page.crumbs]
        items = []
        for i, (label, path) in enumerate(trail, start=1):
            item: dict[str, Any] = {"@type": "ListItem", "position": i, "name": label}
            if origin:  # a private instance names no URLs
                item["item"] = origin + path
            items.append(item)
        objs.append(
            {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": items}
        )
    blocks = []
    for obj in objs:
        # "<" escaped so no string inside can close the <script> element.
        payload = json.dumps(obj, ensure_ascii=False, indent=1).replace("<", "\\u003c")
        blocks.append(f'<script type="application/ld+json">\n{payload}\n</script>')
    return "\n".join(blocks)


def _head(origin: str, page: Page) -> str:
    """The origin-dependent part of <head>: robots, canonical and OG URLs."""
    e = html.escape
    image = e(origin + asset_url(_OG_IMAGE))
    lines = []
    if origin:
        lines += [
            '<meta name="robots" content="index,follow">',
            f'<link rel="canonical" href="{e(_url(origin, page))}">',
            f'<meta property="og:url" content="{e(_url(origin, page))}">',
        ]
    else:
        lines.append('<meta name="robots" content="noindex">')
    lines += [
        f'<meta property="og:image" content="{image}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        '<meta property="og:image:type" content="image/jpeg">',
        '<meta property="og:image:alt" content="LM Warden: an OpenAI-compatible API on the '
        "GPUs under your desk or in your rack, beside the Stats panel of one RTX A4000 with its "
        'driver-reported throttle point labelled">',
        f'<meta name="twitter:image" content="{image}">',
    ]
    return "\n  ".join(lines)


def _fill(src: str, values: dict[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        key, arg = m.group(1), m.group(2)
        if key == "a" and arg:
            return asset_url(arg)
        if key == "fig" and arg:
            return _FIGURES[arg]
        if key == "frag" and arg:
            return _fill(_FRAGMENTS[arg + ".html"], values)
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


def all_site_pages() -> list[Page]:
    """Every page a visitor reaches without tracking configured, in sitemap
    order (tag-only pages such as /privacy are left out)."""
    from app.landing.clients import CLIENT_PAGES

    return [p for p in PAGES if not p.tag_only] + list(CLIENT_PAGES.values())


#: The GitHub mark (16px, decorative) for the menu's GitHub item.
_GH_MARK = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 '
    "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53"
    "-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 "
    "1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15"
    "-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 "
    "1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 "
    '1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>'
)

#: The menu's Download & Install sub-items: each opens its panel of the
#: install chooser on Home through :target (nothing is pre-selected).
_NAV_PLATFORMS = (
    ("/#p-linux", "Linux + NVIDIA GPUs"),
    ("/#p-mac", "macOS, Apple silicon (beta)"),
    ("/#p-win", "Windows (next)"),
    ("/#p-appliance", "Appliance (coming soon)"),
    ("/#upgrade", "Upgrade an existing install"),
)


def nav_html(current: Page) -> str:
    """The primary menu, shared by every page. A <details>: on a phone it is
    the Menu button that opens a full-screen sheet; from 1280px the layout
    script keeps it open and the CSS lays the links out in the bar. Links to
    Home's sections are written "/#id" so they work from any page."""

    marked: list[str] = []

    def link(path: str, label: str) -> str:
        cur = ""
        if path == current.path and not marked:
            marked.append(path)
            cur = ' aria-current="page"'
        return f'<a href="{path}"{cur}>{label}</a>'

    plats = "".join(f'<li><a href="{p}">{label}</a></li>' for p, label in _NAV_PLATFORMS)
    return (
        '<nav aria-label="Primary">\n'
        '        <details class="menu">\n'
        "          <summary>Menu</summary>\n"
        '          <div class="panel">\n'
        "            <ul>\n"
        '              <li><a href="/#why">Use cases</a></li>\n'
        '              <li><a href="/#features">Features</a></li>\n'
        f"              <li>{link('/coding-agents', 'Coding agents')}</li>\n"
        f"              <li>{link('/faq', 'FAQ')}</li>\n"
        f'              <li class="gh"><a href="{REPO_URL}" rel="noopener" '
        f'title="Source on GitHub, Apache-2.0">{_GH_MARK}GitHub<span class="lic"> · '
        "Apache-2.0</span></a></li>\n"
        '              <li class="plats"><a href="/#install">Download &amp; Install</a>\n'
        f'                <ul style="list-style:none;padding:0;margin:0">{plats}</ul></li>\n'
        '              <li class="signin"><a href="/ui/login">Console sign-in</a></li>\n'
        "            </ul>\n"
        "          </div>\n"
        "        </details>\n"
        "      </nav>"
    )


def crumbs_html(page: Page) -> str:
    if not page.crumbs:
        return ""
    items = ['<li><a href="/">Home</a></li>']
    for i, (label, path) in enumerate(page.crumbs):
        cur = ' aria-current="page"' if i == len(page.crumbs) - 1 else ""
        items.append(f'<li><a href="{path}"{cur}>{html.escape(label)}</a></li>')
    return (
        '    <nav class="crumbs" aria-label="Breadcrumb"><div class="wrap"><ol>'
        + "".join(items)
        + "</ol></div></nav>\n"
    )


#: Links to the single page's sections (README, catalogue, bookmarks) still
#: land: on Home, a fragment that names a moved section goes to its new page,
#: and one that names a section the redesign (2026-10-21) folded into the
#: install chooser opens that panel instead.
_ANCHORS = """  <script>
    (function () {
      var here = { "#mac": "#p-mac", "#mac-open": "#p-mac", "#roadmap": "#p-linux", "#install-end": "#install" };
      var map = { "#faq": "/faq", "#faq-title": "/faq", "#router-title": "/coding-agents/claude-code",
        "#box-title": "/gpu-monitoring", "#cards-title": "/gpu-monitoring", "#inflight-title": "/gpu-monitoring",
        "#replica-title": "/cache-aware-routing", "#keys-title": "/api-keys", "#leave-title": "/api-keys",
        "#lessons-title": "/features", "#small-title": "/features" };
      var h = window.location.hash;
      if (here[h]) { window.location.replace(here[h]); return; }
      if (/^#faq-[0-9]+$/.test(h)) { window.location.replace("/faq" + h); return; }
      if (map[h]) window.location.replace(map[h] + h);
    })();
  </script>
"""
_POSTER_PRELOAD = (
    '  <link rel="preload" href="{{a:demo-stats-poster.webp}}" as="image" fetchpriority="high">\n'
)


_GENERIC_CAPTION = (
    "Where a request from a coding agent goes. Every agent sets the same two things, a base "
    "URL and a key; the warden checks the key, queues the request and sends it to the "
    "replica that already holds the session’s cache. Each agent’s page draws its own "
    "settings."
)


@lru_cache(maxsize=256)
def render_page(page: Page, origin: str, analytics: Analytics) -> str:
    """One page for one canonical origin ("" = private instance) and one set
    of tracking IDs. Body first, then the layout, so tokens inside the body
    (assets, figures, the FAQ) are filled by the same pass."""
    from app.landing.clients import client_body, client_def, clients_table_html
    from app.landing.diagrams import GENERIC, figure_html

    home = page.path == "/"
    if page.client_id:
        body = client_body(client_def(page.client_id), _FRAGMENTS["claude-code-router.html"])
    else:
        body = _FRAGMENTS[page.fragment]
    src = _LAYOUT_SRC.replace("{{body}}\n", body)
    src = src.replace("{{preload}}", _POSTER_PRELOAD if home else "", 1)
    return _fill(
        src,
        {
            "title": html.escape(page.title),
            "description": html.escape(page.description),
            "head": _head(origin, page),
            "jsonld": _jsonld(origin, page),
            "faq": _faq_html(),
            "repo": REPO_URL,
            "mac_release": MAC_RELEASE_URL,
            "mac_dmg": MAC_DMG_NAME,
            "install": _install_html(),
            "nav": nav_html(page),
            "page_css": _STYLES["home" if home else "content"],
            "crumbs": crumbs_html(page),
            "anchors": _ANCHORS if home else "",
            "clients_table": clients_table_html(),
            "flow_generic": figure_html(GENERIC, "flow-generic", _GENERIC_CAPTION),
            "tag": tag_html(analytics),
            "banner": banner_html(analytics),
            # Only with a Google tag: the footer then says what loads and
            # links the cookie settings (the banner's way back).
            "thirdparty": (
                f'      <p class="tag-note">{_WITH_TAG}</p>\n' if analytics.tag_on else ""
            ),
            "privacy_ads": (
                " Google Ads measures whether ads bring people here, and its "
                "advertising cookies follow the same consent choice."
                if analytics.ads_id
                else ""
            ),
            "privacy_contact": (
                ' Write to <a href="mailto:info@podwarden.com">info@podwarden.com</a> '
                "with any question about this site."
                if origin == OFFICIAL_ORIGIN
                else " Questions about this site go to the operator of this site."
            ),
            "gsc": (
                f'  <meta name="google-site-verification" '
                f'content="{html.escape(analytics.gsc_token)}">\n'
                if analytics.gsc_token and home
                else ""
            ),
        },
    )


def render_landing(origin: str) -> str:
    """The home page without tracking (kept for callers and tests)."""
    return render_page(HOME, origin, Analytics())


@lru_cache(maxsize=8)
def render_llms(origin: str, full: bool) -> str:
    faq_md = "\n\n".join(f"### {qa['q']}\n\n{_faq_markdown(qa['a'])}" for qa in _FAQ)
    return _fill(
        _LLMS_FULL_SRC if full else _LLMS_SRC,
        {
            "repo": REPO_URL,
            "raw": RAW_URL,
            "mac_release": MAC_RELEASE_URL,
            "mac_dmg": MAC_DMG_NAME,
            "origin": origin,
            "install": INSTALL_CMD,
            "faq": faq_md,
            "description": DESCRIPTION,
            "pages": "\n".join(
                f"- [{p.title.split(' | ')[0]}]({_url(origin, p)}): {p.description}"
                for p in all_site_pages()
            ),
        },
    )


@lru_cache(maxsize=8)
def render_agent_guide(origin: str) -> str:
    """The LLM-oriented control-API guide served at /agent-guide.md."""
    return _AGENT_GUIDE_SRC.replace("{{raw}}", RAW_URL)


# --- Content-Security-Policy ------------------------------------------------------
#
# The page is static: no framework, fonts and media from this instance, a few
# inline scripts (in <head>, the one line that marks the page as scripted; on
# Home, the use-case carousel and the screenshot lightbox; at the end of
# <body>, the menu, copy buttons and video autoplay) and JSON-LD data blocks,
# which a browser never executes and CSP does not govern. So the policy can be strict: nothing loads from
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
            "'sha256-"
            + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode()
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


#: Origins the Google tag needs, from Google's "Content Security Policy"
#: guide (developers.google.com/tag-platform/security/guides/csp, read
#: 2026-10-07): GA4 "with Ads features" (an Ads account will be linked) and
#: Google Ads conversion + remarketing. Google also lists the country domains
#: https://*.google.<TLD>, which CSP cannot wildcard; only .com is allowed, so
#: a visitor routed to a country domain may lose an ads ping. script-src-elem
#: in the guide is folded into script-src, which the base policy uses.
_GA_CSP: dict[str, tuple[str, ...]] = {
    "script-src": ("https://www.googletagmanager.com",),
    "img-src": (
        "https://www.googletagmanager.com",
        "https://*.google-analytics.com",
        "https://*.google.com",
        "https://*.g.doubleclick.net",
    ),
    "connect-src": (
        "https://www.googletagmanager.com",
        "https://*.google-analytics.com",
        "https://*.google.com",
        "https://*.g.doubleclick.net",
        "https://pagead2.googlesyndication.com",
    ),
    "frame-src": ("https://www.googletagmanager.com",),
}
_ADS_CSP: dict[str, tuple[str, ...]] = {
    "script-src": (
        "https://www.googletagmanager.com",
        "https://www.googleadservices.com",
        "https://www.google.com",
    ),
    "img-src": (
        "https://www.googletagmanager.com",
        "https://www.googleadservices.com",
        "https://googleads.g.doubleclick.net",
        "https://pagead2.googlesyndication.com",
        "https://www.google.com",
    ),
    "connect-src": (
        "https://www.googletagmanager.com",
        "https://www.googleadservices.com",
        "https://googleads.g.doubleclick.net",
        "https://pagead2.googlesyndication.com",
        "https://www.google.com",
        "https://ad.doubleclick.net",
    ),
    "frame-src": ("https://www.googletagmanager.com",),
}


def page_csp(page_html: str, analytics: Analytics) -> str:
    """The policy for one rendered page: the strict base, plus exactly the
    Google origins the configured IDs need. No ID, no change."""
    base = landing_csp(page_html)
    if not analytics.tag_on:
        return base
    extra: dict[str, list[str]] = {}
    tables = ([_GA_CSP] if analytics.ga4_id else []) + ([_ADS_CSP] if analytics.ads_id else [])
    for table in tables:
        for directive, origins in table.items():
            have = extra.setdefault(directive, [])
            have.extend(o for o in origins if o not in have)
    parts = []
    for directive in base.split("; "):
        name = directive.split(" ", 1)[0]
        if name in extra:
            directive += " " + " ".join(extra.pop(name))
        parts.append(directive)
    parts += [f"{name} {' '.join(origins)}" for name, origins in extra.items()]
    return "; ".join(parts)


#: Consent Mode v2, before gtag.js loads: everything denied until the visitor
#: accepts; a stored "granted" is applied at once. Static text, so its CSP
#: hash never changes.
_CONSENT_DEFAULT = """  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag() { window.dataLayer.push(arguments); }
    gtag('consent', 'default', { ad_storage: 'denied', ad_user_data: 'denied', ad_personalization: 'denied', analytics_storage: 'denied', wait_for_update: 500 });
    (function () {
      var c = null;
      try { c = window.localStorage.getItem("lmw-consent"); } catch (e) { /* storage blocked */ }
      if (c === "granted") gtag('consent', 'update', { ad_storage: 'granted', ad_user_data: 'granted', ad_personalization: 'granted', analytics_storage: 'granted' });
    })();
  </script>
"""

#: The tag's configuration, the banner and the conversion events. The IDs
#: come from the data-ids attribute, so this text (and its hash) is the same
#: for every ID.
_TAG_BOOT = """  <script>
    (function () {
      var el = document.getElementById("lmw-tag");
      var ids = (el.getAttribute("data-ids") || "").split(",");
      gtag('js', new Date());
      ids.forEach(function (id) { if (id) gtag('config', id); });
      // Google's script waits for the page: parsed with it, it held a
      // throttled phone's main thread for over a second. The commands above
      // queue in dataLayer until it arrives.
      function loadTag() {
        var s = document.createElement("script");
        s.async = true;
        s.src = "https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(ids[0]);
        document.head.appendChild(s);
      }
      if (document.readyState === "complete") loadTag();
      else window.addEventListener("load", loadTag);
      var banner = document.querySelector(".consent"), stored = null;
      try { stored = window.localStorage.getItem("lmw-consent"); } catch (e) { /* storage blocked */ }
      function choose(v) {
        try { window.localStorage.setItem("lmw-consent", v); } catch (e) { /* lasts this page */ }
        gtag('consent', 'update', { ad_storage: v, ad_user_data: v, ad_personalization: v, analytics_storage: v });
        banner.hidden = true;
      }
      if (stored !== "granted" && stored !== "denied") banner.hidden = false;
      banner.querySelectorAll("[data-consent]").forEach(function (b) {
        b.addEventListener("click", function () { choose(b.getAttribute("data-consent")); });
      });
      document.querySelectorAll("[data-consent-open]").forEach(function (a) {
        a.addEventListener("click", function (ev) { ev.preventDefault(); banner.hidden = false; });
      });
      document.addEventListener("click", function (ev) {
        var t = ev.target.closest ? ev.target.closest("a, button") : null;
        if (!t) return;
        var href = t.getAttribute("href") || "";
        if (t.matches("button[data-copy^='cmd-']")) gtag('event', 'copy_install');
        else if (href.indexOf("https://github.com/Podwarden/lm-warden") === 0) gtag('event', 'click_github');
        else if (href.indexOf("https://podwarden.com") === 0 || href === "https://github.com/Podwarden") gtag('event', 'click_podwarden');
      });
    })();
  </script>
"""


def _ids(analytics: Analytics) -> list[str]:
    return [i for i in (analytics.ga4_id, analytics.ads_id) if i]


def tag_html(analytics: Analytics) -> str:
    """The <head> part of the Google tag: the consent default, which must
    run before anything else touches dataLayer. gtag.js itself is injected
    after the load event by the boot script (banner_html). "" when no ID is
    set."""
    if not analytics.tag_on:
        return ""
    return _CONSENT_DEFAULT


def banner_html(analytics: Analytics) -> str:
    """The end-of-<body> part: the banner, the IDs and the boot script, which
    looks both up as it runs and so must come after them."""
    if not analytics.tag_on:
        return ""
    return (
        _BANNER
        + f'  <span id="lmw-tag" hidden data-ids="{html.escape(",".join(_ids(analytics)))}"></span>\n'
        + _TAG_BOOT
    )


_BANNER = (
    '  <div class="consent" role="region" aria-label="Cookie choice" hidden>'
    "<p>This site uses Google Analytics cookies to count visits and measure ads, but only if "
    'you allow it. <a href="/privacy">Privacy</a></p>'
    '<div class="consent-actions">'
    '<button class="btn" type="button" data-consent="denied"><span>Decline</span></button>'
    '<button class="btn" type="button" data-consent="granted"><span>Accept</span></button>'
    "</div></div>\n"
)
_WITH_TAG = (
    "This page is served by LM Warden itself and loads Google’s tag to count visits. "
    'Cookies are set only if you accept them (<a href="/privacy">privacy</a>, '
    '<a href="/privacy" data-consent-open>cookie settings</a>).'
)


# Render the private variant at import: a broken template token or an asset
# the page names but the image does not ship fails the process at start-up.
render_landing("")
render_llms("", False)
render_llms("", True)
