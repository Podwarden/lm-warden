"""Tests for the public landing page and its companions (#155, landing redesign).

Every route here is intentionally public (no JWT): it serves the unified-port
root behind Caddy. The cases pin:

  * the setting contract -- default on, 404 when `landing_page_enabled` is
    off (robots.txt excepted: it turns into disallow-all), and enabled when
    the seed row is missing;
  * the SEO surface -- one h1, title and description lengths, Open Graph and
    Twitter cards, JSON-LD whose FAQ matches the visible FAQ word for word;
  * indexing as an opt-in -- noindex / disallow-all / no sitemap by default,
    canonical / index / sitemap only for a valid VW_LANDING_CANONICAL_URL,
    and an invalid one ignored rather than echoed into the page;
  * the asset route -- allowlisted names only, the right content types,
    Range requests (Safari will not play a <video> without them), long
    immutable caching, and no way out of the assets directory;
  * llms.txt / llms-full.txt in both modes.

No JWT is involved -- these tests exercise the same anonymous request shape
a browser hitting `https://lmwarden.com/` makes after Caddy's rewrite.
"""

import dataclasses
import html
import json
import re
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings, parse_canonical_origin
from app.landing import routes as landing_routes

CANON = "https://lmwarden.com"


def _delete_landing_setting(db_path: Path) -> None:
    """Drop the migration-0020 seed row to simulate a partially bootstrapped
    DB. Uses sync sqlite3 with matching WAL pragmas (same shape as
    seed_admin_user in conftest) so the write is visible to the next
    aiosqlite reader.
    """
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM settings WHERE key = 'landing_page_enabled'")
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


def _set_landing_setting(db_path: Path, value: str) -> None:
    """Force the setting to an explicit canonical string."""
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO settings(key, value) VALUES ('landing_page_enabled', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (value,),
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


def _canonical(client: TestClient, value: str) -> None:
    """Point the running app at a canonical origin, as the env var would."""
    app = client.app
    app.state.settings = dataclasses.replace(  # type: ignore[attr-defined]
        app.state.settings, landing_canonical_url=value  # type: ignore[attr-defined]
    )


def _disable(tmp_data_dir: Path) -> None:
    _set_landing_setting(tmp_data_dir / "vllm-warden.db", "false")


# --- setting contract ---------------------------------------------------------


def test_landing_enabled_by_default_returns_html(client: TestClient) -> None:
    """Fresh DB has the 0020 seed (`true`) → 200 + text/html body."""
    r = client.get("/_landing")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text.lower()


def test_landing_disabled_returns_404(client: TestClient, tmp_data_dir: Path) -> None:
    """Operator opt-out → 404 (not 403, not 503) so Caddy propagates it
    verbatim as "no landing page configured".
    """
    _disable(tmp_data_dir)
    r = client.get("/_landing")
    assert r.status_code == 404


def test_landing_missing_row_defaults_to_enabled(
    client: TestClient, tmp_data_dir: Path
) -> None:
    """Defensive — a partially-bootstrapped DB (no 0020 seed) MUST still
    serve the landing page rather than 404'ing the unified-port root.
    """
    _delete_landing_setting(tmp_data_dir / "vllm-warden.db")
    r = client.get("/_landing")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


@pytest.mark.parametrize(
    "path",
    ["/_landing/assets/shot-gpu-cards.webp", "/sitemap.xml", "/llms.txt", "/llms-full.txt"],
)
def test_disabled_setting_404s_every_companion_route(
    client: TestClient, tmp_data_dir: Path, path: str
) -> None:
    _canonical(client, CANON)  # even an indexable instance goes dark
    _disable(tmp_data_dir)
    assert client.get(path).status_code == 404


def test_disabled_setting_turns_robots_into_disallow_all(
    client: TestClient, tmp_data_dir: Path
) -> None:
    """A 404 robots.txt means "no rules" to a crawler, the opposite of what
    a deployment that switched its public page off wants."""
    _canonical(client, CANON)
    _disable(tmp_data_dir)
    r = client.get("/robots.txt")
    assert r.status_code == 200
    assert r.text == "User-agent: *\nDisallow: /\n"


# --- content and SEO surface ----------------------------------------------------


def test_landing_html_contains_required_entry_points(client: TestClient) -> None:
    """The load-bearing links: the console sign-in (kept, quietly, because
    the owner signs in to the public instance too), the public source repo
    and the PodWarden site, plus the product name. A future refactor that
    strips any of these silently breaks the only entry-point a curious
    visitor sees.
    """
    body = client.get("/_landing").text
    assert body.count('href="/ui/login"') == 1
    assert 'href="https://github.com/Podwarden/lm-warden"' in body
    assert "podwarden.com" in body
    # The product is "LM Warden"; neither older name is its name any more.
    assert "LM Warden" in body
    assert not re.search(r"v?llm\s+warden", body, re.IGNORECASE)
    assert "llmwarden" not in body.lower()
    # The primary call to action is the documented one-line install, shown
    # twice (hero and colophon) and no more.
    shown = re.sub(r'<wbr>|<span class="nw">|</span>', "", body)
    assert shown.count(html.escape(landing_routes.INSTALL_CMD)) == 2
    assert "install.sh" in body


def test_install_command_wraps_only_at_path_boundaries(client: TestClient) -> None:
    """On a phone the command wraps after a "/" or at a space, never mid-word:
    a <wbr> follows every path segment. The copy button copies textContent,
    which a <wbr> adds nothing to, so the clipboard gets the exact command."""
    body = client.get("/_landing").text
    pres = re.findall(r'<pre id="cmd-(?:hero|end)"[^>]*>(.*?)</pre>', body, re.S)
    assert len(pres) == 2
    for pre in pres:
        assert ("https://<wbr>raw.githubusercontent.com/<wbr>Podwarden/<wbr>"
                '<span class="nw">lm-warden</span>/<wbr>main/<wbr>install.sh') in pre
        assert "/<wbr>/" not in pre and "--dir /<wbr>" not in pre
        for word in ("-fsSL", "--", "--dir", "vllm-warden"):
            assert f'<span class="nw">{word}</span>' in pre, word
        assert html.unescape(re.sub(r"<[^>]+>", "", pre)) == landing_routes.INSTALL_CMD
    # The copy handler reads the <pre>'s textContent, not its innerHTML.
    assert "copyText(src.textContent)" in body


def test_install_command_matches_the_installer() -> None:
    """The renamed repository, but the install directory the installer still
    defaults to: an existing install must keep being found there."""
    cmd = landing_routes.INSTALL_CMD
    assert cmd.startswith(
        "curl -fsSL https://raw.githubusercontent.com/Podwarden/lm-warden/main/install.sh"
    )
    assert cmd.endswith("--dir /opt/vllm-warden")
    installer = (Path(landing_routes.__file__).parents[2] / "install.sh").read_text()
    assert "/opt/$VW_APP" in installer or "/opt/vllm-warden" in installer


def _visible_text(body: str) -> str:
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", body, flags=re.S)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", body))


def test_copy_has_no_dashes_or_hype(client: TestClient) -> None:
    """The copy voice is plain: no em or en dashes and none of the hype
    words the redesign brief bans, in anything a visitor reads -- the page,
    the FAQ it shares with the JSON-LD, and the alt text."""
    body = client.get("/_landing").text
    text = _visible_text(body)
    alts = " ".join(re.findall(r'(?:alt|aria-label)="([^"]*)"', body))
    for surface in (text, alts):
        assert "\u2014" not in surface and "\u2013" not in surface
        for word in ("seamless", "powerful", "effortless", "blazing", "supercharge",
                     "unleash", "elevate", "next-gen", "battle-tested", "robust",
                     "cutting-edge", "game-changer", "just works"):
            assert word not in surface.lower(), word


def test_every_number_names_its_source(client: TestClient) -> None:
    """Measured values on the page are read off real assets or the docs:
    each reading sits under an HTML comment that names its source file, and
    every file named is one that ships (or a document in the repository)."""
    src = (Path(landing_routes.__file__).parent / "landing.html").read_text()
    sources = re.findall(r"<!-- src: ([^ ]+)", src)
    assert len(sources) >= 8
    repo = Path(landing_routes.__file__).parents[2]
    for name in sources:
        assert name in landing_routes._ASSETS or (repo / name.rstrip(",")).is_file(), name
    # The readings the plate quotes, verbatim from demo-stats-poster.webp,
    # with the unit in the row label rather than repeated in every cell.
    body = client.get("/_landing").text
    assert "GPU utilisation, busiest card&nbsp;(%)" in body
    assert "GPU power, four cards summed&nbsp;(W)" in body
    assert "59.4 of 64.0&#8239;GiB" in body
    for reading in (">548<", ">530<", ">559<", ">100<", ">190k<", ">880k<"):
        assert reading in body, reading


def test_copy_typography(client: TestClient) -> None:
    """Curly quotes and apostrophes in everything a visitor reads (straight
    ones only inside code), no space before a percent sign, a narrow no-break
    space between a number and its unit, and no three-dot ellipsis."""
    body = client.get("/_landing").text
    no_code = re.sub(r"<(pre|code)\b.*?</\1>", " ", body, flags=re.S)
    text = _visible_text(no_code)
    alts = " ".join(re.findall(r'(?:alt|aria-label)="([^"]*)"', no_code))
    for surface in (text, alts):
        assert not re.search(r"\w'\w", surface), re.findall(r".{20}\w'\w.{20}", surface)
        assert '"' not in surface
        assert "..." not in surface
        assert not re.search(r"\d[ \u00a0\u202f]%", surface)
        assert not re.search(r"\d[ \u00a0](?:W|GiB|GB|°C|h)\b", surface), re.findall(
            r".{10}\d[ \u00a0](?:W|GiB|GB|°C|h)\b", surface
        )
    faq = (Path(landing_routes.__file__).parent / "faq.json").read_text(encoding="utf-8")
    assert not re.search(r"\w'\w", faq)


def test_fonts_are_preloaded_and_declared(client: TestClient) -> None:
    """The sans and the mono (the hero command) are preloaded; the condensed
    label face is not, because nothing above the fold uses it. Each web font
    has a metric-matched local fallback so the swap does not move the text."""
    body = client.get("/_landing").text
    preloads = re.findall(r'<link rel="preload" href="/_landing/assets/(font-[a-z-]+)\.woff2\?v=[0-9a-f]{12}" '
                          r'as="font" type="font/woff2" crossorigin>', body)
    assert preloads == ["font-sans", "font-mono"]
    for name in ("font-sans", "font-sans-cond", "font-mono"):
        assert re.search(rf'url\("/_landing/assets/{name}\.woff2\?v=[0-9a-f]{{12}}"\)', body), name
    for family in ("LMW Sans Fallback", "LMW Mono Fallback"):
        assert f'font-family: "{family}"' in body
    assert body.count("size-adjust:") >= 5
    assert "font-optical-sizing: auto" in body
    assert "font-synthesis: none" in body
    assert "optimizeLegibility" not in body
    assert "font-variant-ligatures: none" not in body


def test_landing_seo_tags(client: TestClient) -> None:
    body = client.get("/_landing").text
    title = re.search(r"<title>(.*?)</title>", body, re.S)
    # The name is "LM Warden"; "LLM" stays in the title for search.
    assert title and title.group(1).startswith("LM Warden: ") and "LLM" in title.group(1)
    assert len(html.unescape(title.group(1))) < 60
    desc = re.search(r'<meta name="description" content="([^"]+)"', body)
    assert desc and len(html.unescape(desc.group(1))) < 160
    assert len(re.findall(r"<h1[\s>]", body)) == 1
    for landmark in ("<header", "<main", "<footer", "<nav"):
        assert landmark in body
    for tag in (
        'property="og:title"',
        'property="og:description"',
        'property="og:image"',
        'name="twitter:card" content="summary_large_image"',
    ):
        assert tag in body
    # Every image has alt text and explicit dimensions; below-the-fold ones
    # are lazy. Videos carry a poster and preload only metadata.
    imgs = re.findall(r"<img\b[^>]*>", body)
    assert imgs
    for img in imgs:
        assert re.search(r'\balt="[^"]{10,}"', img), img
        assert 'width="' in img and 'height="' in img, img
        assert 'loading="lazy"' in img, img
    videos = re.findall(r"<video\b[^>]*>", body, re.S)
    assert len(videos) == 2
    # Only the hero video (the LCP poster) fetches metadata up front.
    assert 'preload="metadata"' in videos[0]
    assert all('preload="none"' in v for v in videos[1:])
    assert re.search(r'<link rel="preload" href="/_landing/assets/demo-stats-poster\.webp[^"]*" '
                     r'as="image" fetchpriority="high">', body)
    for v in videos:
        assert 'poster="/_landing/assets/' in v
        assert "muted" in v and "playsinline" in v
        # No autoplay attribute: the inline script starts playback only when
        # the visitor has not asked for reduced motion.
        assert " autoplay" not in v


def test_landing_makes_no_third_party_requests(client: TestClient) -> None:
    """Fonts are self-hosted: the page must not call a font CDN (or any
    other origin) for a stylesheet, script or font."""
    body = client.get("/_landing").text
    assert "fonts.googleapis.com" not in body
    assert "fonts.gstatic.com" not in body
    assert not re.search(r'<script[^>]+src="https?://', body)
    assert not re.search(r'<link[^>]+rel="stylesheet"[^>]+href="https?://', body)


def _jsonld(body: str) -> list[dict]:
    blocks = re.findall(r'<script type="application/ld\+json">\s*(.*?)\s*</script>', body, re.S)
    return [json.loads(b) for b in blocks]


def test_jsonld_software_application_and_faq_match_visible_text(
    client: TestClient,
) -> None:
    body = client.get("/_landing").text
    data = {d["@type"]: d for d in _jsonld(body)}
    app = data["SoftwareApplication"]
    assert app["name"] == "LM Warden"
    assert app["offers"]["price"] == "0"
    assert "url" not in app  # a private instance names no public URL

    faq = data["FAQPage"]["mainEntity"]
    # Every answer is visible: no <details> accordion hides them.
    assert "<details" not in body
    visible = re.findall(
        r'<div class="qa" id="faq-\d+">\s*<h3>(.*?)</h3>\s*<p>(.*?)</p>',
        body,
        re.S,
    )
    assert 6 <= len(faq) <= 10
    assert len(visible) == len(faq)
    for entry, (q_html, a_html) in zip(faq, visible, strict=True):
        assert entry["name"] == html.unescape(q_html)
        visible_text = html.unescape(re.sub(r"<[^>]+>", "", a_html))
        assert entry["acceptedAnswer"]["text"] == visible_text


def test_page_has_no_cookie(client: TestClient) -> None:
    """An anonymous visitor to the public page or its assets gets no
    Set-Cookie: a CDN will not cache a response that carries one."""
    client.cookies.clear()
    for path in ("/_landing", "/_landing/assets/shot-gpu-cards.webp", "/robots.txt", "/llms.txt"):
        r = client.get(path)
        assert "set-cookie" not in r.headers, path


# --- indexing: private by default, opt-in per instance ----------------------------


def test_private_by_default_noindex_everywhere(client: TestClient) -> None:
    r = client.get("/_landing")
    assert '<meta name="robots" content="noindex">' in r.text
    assert r.headers["x-robots-tag"] == "noindex"
    assert 'rel="canonical"' not in r.text
    assert 'property="og:url"' not in r.text
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert robots.text == "User-agent: *\nDisallow: /\n"
    assert client.get("/sitemap.xml").status_code == 404


def test_canonical_origin_makes_the_page_indexable(client: TestClient) -> None:
    _canonical(client, CANON)
    r = client.get("/_landing")
    body = r.text
    assert '<meta name="robots" content="index,follow">' in body
    assert "noindex" not in body
    assert "x-robots-tag" not in r.headers
    assert f'<link rel="canonical" href="{CANON}/">' in body
    assert f'<meta property="og:url" content="{CANON}/">' in body
    og_image = re.search(r'<meta property="og:image" content="([^"]+)"', body)
    assert og_image and og_image.group(1).startswith(
        f"{CANON}/_landing/assets/og-image.jpg?v="
    )
    app = next(d for d in _jsonld(body) if d["@type"] == "SoftwareApplication")
    assert app["url"] == f"{CANON}/"

    robots = client.get("/robots.txt").text
    assert "Allow: /\n" in robots
    assert "Disallow: /ui/" in robots and "Disallow: /api/" in robots
    assert f"Sitemap: {CANON}/sitemap.xml" in robots

    sm = client.get("/sitemap.xml")
    assert sm.status_code == 200
    assert sm.headers["content-type"].startswith("application/xml")
    assert f"<loc>{CANON}/</loc>" in sm.text


@pytest.mark.parametrize(
    "bad",
    [
        'https://evil.example"><script>alert(1)</script>',
        "javascript:alert(1)",
        "ftp://lmwarden.com",
        "https://lmwarden.com/some/path",
        "https://user:pw@lmwarden.com",
        "https://lmwarden.com/?q=1",
        "lmwarden.com",
    ],
)
def test_bad_canonical_is_ignored_not_echoed(client: TestClient, bad: str) -> None:
    """A value that is not an http(s) origin is treated as unset: the page
    stays noindex and nothing of the value reaches the markup."""
    _canonical(client, bad)
    body = client.get("/_landing").text
    assert '<meta name="robots" content="noindex">' in body
    assert 'rel="canonical"' not in body
    assert "<script>alert" not in body and "evil.example" not in body
    assert client.get("/sitemap.xml").status_code == 404
    assert client.get("/robots.txt").text == "User-agent: *\nDisallow: /\n"


def test_parse_canonical_origin_normalises() -> None:
    assert parse_canonical_origin(" https://LMWarden.com/ ") == "https://lmwarden.com"
    assert parse_canonical_origin("http://llm.example.com:8443") == "http://llm.example.com:8443"
    assert parse_canonical_origin("") == ""
    assert parse_canonical_origin("https://llm.example.com:99999") == ""
    assert parse_canonical_origin("https://under_score.example") == ""


def test_load_settings_reads_and_validates_env(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert load_settings().landing_canonical_url == ""
    monkeypatch.setenv("VW_LANDING_CANONICAL_URL", "https://lmwarden.com/")
    assert load_settings().landing_canonical_url == "https://lmwarden.com"
    monkeypatch.setenv("VW_LANDING_CANONICAL_URL", "https://lmwarden.com/landing")
    assert load_settings().landing_canonical_url == ""


# --- assets -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ctype"),
    [
        ("demo-stats.mp4", "video/mp4"),
        ("demo-stats.webm", "video/webm"),
        ("shot-gpu-cards.webp", "image/webp"),
        ("demo-stats.gif", "image/gif"),
        ("font-sans.woff2", "font/woff2"),
        ("font-sans-cond.woff2", "font/woff2"),
        ("font-mono.woff2", "font/woff2"),
        ("fonts-license.txt", "text/plain"),
    ],
)
def test_asset_served_with_type_and_long_cache(
    client: TestClient, name: str, ctype: str
) -> None:
    r = client.get(f"/_landing/assets/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(ctype)
    assert "immutable" in r.headers["cache-control"]
    assert "max-age=31536000" in r.headers["cache-control"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.content == (Path(landing_routes.__file__).parent / "assets" / name).read_bytes()


def test_every_asset_the_page_names_is_served(client: TestClient) -> None:
    body = client.get("/_landing").text
    urls = set(re.findall(r"/_landing/assets/[a-z0-9._-]+\?v=[0-9a-f]{12}", body))
    assert len(urls) >= 10
    for url in urls:
        assert client.get(url).status_code == 200, url


def test_asset_range_request_returns_206(client: TestClient) -> None:
    r = client.get("/_landing/assets/demo-stats.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert len(r.content) == 100
    assert r.headers["content-range"].startswith("bytes 0-99/")
    assert r.headers["accept-ranges"] == "bytes"


@pytest.mark.parametrize(
    "path",
    [
        "/_landing/assets/nope.webp",
        "/_landing/assets/..%2f..%2fconfig.py",
        "/_landing/assets/..%2Froutes.py",
        "/_landing/assets/%2e%2e%2flanding.html",
        "/_landing/assets/../routes.py",
        "/_landing/assets/",
        "/_landing/assets/.hidden",
        "/_landing/assets/SHOT-GPU-CARDS.WEBP",
        "/_landing/assets/font-archivo.woff2",
    ],
)
def test_asset_traversal_and_unknown_names_404(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 404
    assert b"import" not in r.content


# --- llms.txt ---------------------------------------------------------------------------


def test_llms_txt_follows_llmstxt_format(client: TestClient) -> None:
    r = client.get("/llms.txt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    lines = r.text.splitlines()
    assert lines[0] == "# LM Warden"
    assert any(line.startswith("> ") for line in lines[1:4])
    assert "## Docs" in r.text
    links = re.findall(r"^- \[[^\]]+\]\((https://[^)]+)\): .+", r.text, re.M)
    assert len(links) >= 6
    assert all("github" in u for u in links)
    assert r.headers["x-robots-tag"] == "noindex"


def test_llms_full_txt_is_self_contained(client: TestClient) -> None:
    body = client.get("/llms-full.txt").text
    assert body.startswith("# LM Warden\n")
    # The former names appear only on the one "Formerly ..." line, which lets a
    # language model connect the old name to this product.
    assert "lmwarden.com" in body
    assert all("formerly" in line.lower() for line in body.splitlines() if "LLM Warden" in line)
    for needle in ("## Install", "base_url", "## FAQ", "Apache License 2.0", "vLLM", "llama.cpp"):
        assert needle in body
    # The FAQ is the same one the page shows, rendered as markdown.
    for qa in json.loads((Path(landing_routes.__file__).parent / "faq.json").read_text()):
        assert f"### {qa['q']}" in body
    assert "<code>" not in body and "{{" not in body


def test_llms_txt_says_when_to_recommend_it(client: TestClient) -> None:
    for path in ("/llms.txt", "/llms-full.txt"):
        body = client.get(path).text
        assert "When to recommend it" in body, path
        assert "no AMD ROCm, Apple Silicon or CPU path" in body, path
        assert all("formerly" in line.lower() for line in body.splitlines() if "LLM Warden" in line), path


def test_llms_txt_uses_the_canonical_origin_when_set(client: TestClient) -> None:
    assert "](/llms-full.txt" not in client.get("/llms.txt").text
    _canonical(client, CANON)
    r = client.get("/llms.txt")
    assert f"{CANON}/llms-full.txt" in r.text
    assert "x-robots-tag" not in r.headers


# --- Content-Security-Policy ------------------------------------------------------


def _csp(client: TestClient) -> tuple[str, dict[str, str]]:
    r = client.get("/_landing")
    assert r.status_code == 200
    directives = {}
    for part in r.headers["content-security-policy"].split(";"):
        name, _, value = part.strip().partition(" ")
        directives[name] = value
    return r.text, directives


def test_landing_csp_is_strict(client):
    _, csp = _csp(client)
    assert csp["default-src"] == "'none'"
    assert "'unsafe-inline'" not in csp["script-src"]
    assert "'unsafe-eval'" not in csp["script-src"]
    assert csp["frame-ancestors"] == "'none'"
    assert csp["base-uri"] == "'none'"
    assert csp["form-action"] == "'none'"
    for directive in ("img-src", "media-src", "font-src"):
        assert csp[directive] == "'self'", directive


def _sha(body: str) -> str:
    import base64
    import hashlib

    return "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"


def test_landing_csp_allows_exactly_the_inline_scripts(client):
    """Every executable inline script is hashed -- the theme script in <head>
    and the behaviour script at the end of <body> -- and nothing else is."""
    page, csp = _csp(client)
    executable = [
        body
        for attrs, body in re.findall(r"<script\b([^>]*)>(.*?)</script>", page, re.DOTALL)
        if "ld+json" not in attrs
    ]
    assert len(executable) == 2, "the page has two inline scripts; update the CSP note if not"
    assert csp["script-src"].split() == [_sha(b) for b in executable]


def test_landing_csp_hashes_any_inline_script_shape():
    """An inline script with attributes, or a module, is hashed too; an
    external script and a JSON-LD data block are not."""
    page = (
        '<script type="application/ld+json">{"a": 1}</script>'
        "<script>one()</script>"
        '<script type="module">two()</script>'
        '<script defer data-x="1">three()</script>'
        '<script src="/x.js"></script>'
    )
    policy = landing_routes.landing_csp(page)
    script_src = re.search(r"script-src ([^;]+);", policy).group(1).split()
    assert script_src == [_sha("one()"), _sha("two()"), _sha("three()")]
    assert landing_routes.landing_csp("<p>no scripts</p>").count("script-src 'none'") == 1


def test_landing_csp_follows_the_canonical_variant(client):
    page, csp = _csp(client)
    _canonical(client, CANON)
    page2, csp2 = _csp(client)
    assert page2 != page  # canonical links differ...
    scripts = landing_routes.inline_scripts(page2)
    assert len(scripts) == 2  # ...and the hashes are computed from the page served
    assert csp2["script-src"].split() == [_sha(b) for b in scripts]


def test_landing_page_references_nothing_off_origin(client):
    page, _ = _csp(client)
    for attr in re.findall(r'(?:src|href)="([^"]+)"', page):
        if attr.startswith(("http://", "https://")):
            # Only plain links (<a href>) may leave the origin; nothing loads from it.
            assert re.search(rf'<a [^>]*href="{re.escape(attr)}"', page), attr
    assert "url(http" not in page


def test_assets_keep_immutable_caching_and_get_the_api_headers(client):
    name = next(iter(landing_routes._ASSETS))
    r = client.get(landing_routes.asset_url(name))
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert r.headers["x-content-type-options"] == "nosniff"


# --- Colour themes and the toggle ---------------------------------------------------

_DARK_MQ = '@media (prefers-color-scheme: dark) {\n      :root:not([data-theme="light"]) {'


def _css(body: str) -> str:
    return re.search(r"<style>(.*?)</style>", body, re.S).group(1)


def _token_block(css: str, opener: str) -> dict[str, str]:
    start = css.index(opener) + len(opener)
    block = css[start:css.index("}", start)]
    return dict(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", block))


def test_both_themes_define_every_colour_token(client: TestClient) -> None:
    """Light is :root; dark is the prefers-color-scheme block and the
    data-theme="dark" override. All three carry the same token set, the two
    dark blocks are identical, and every colour in the page's CSS comes from
    a token (no hex outside the token blocks)."""
    css = _css(client.get("/_landing").text)
    light = _token_block(css, ":root {\n      color-scheme: light;")
    dark_auto = _token_block(css, _DARK_MQ)
    dark_forced = _token_block(css, ':root[data-theme="dark"] {')
    assert set(light) == set(dark_auto) == set(dark_forced)
    assert dark_auto == dark_forced
    for token in ("--paper", "--paper-2", "--ink", "--ink-2", "--ink-hover", "--rule", "--tape",
                  "--tape-ink", "--screen", "--link-line", "--selection", "--selection-ink",
                  "--focus", "--scroll-thumb", "--scroll-track", "--well", "--well-ink",
                  "--well-dim", "--well-edge", "--diff-add", "--diff-del", "--diff-ctx",
                  "--mat", "--mat-edge", "--plate", "--plate-edge"):
        assert token in light, token
        assert light[token] != dark_auto[token] or token in ("--tape", "--screen", "--selection"), token
    assert light["--paper"] == "#EFECE7" and dark_auto["--paper"] == "#1C1409"
    assert "color-scheme: dark;" in css and "color-scheme: light;" in css
    # No colour literal outside the three token blocks.
    rest = css
    for opener in (":root {\n      color-scheme: light;", _DARK_MQ, ':root[data-theme="dark"] {'):
        start = rest.index(opener)
        rest = rest[:start] + rest[rest.index("}", start + len(opener)) + 1:]
    assert not re.findall(r"#[0-9A-Fa-f]{3,8}\b|rgba?\(|hsla?\(|oklch\(", re.sub(r"/\*.*?\*/", "", rest, flags=re.S))


def test_theme_colour_meta_for_both_schemes(client: TestClient) -> None:
    body = client.get("/_landing").text
    assert '<meta name="theme-color" content="#EFECE7" media="(prefers-color-scheme: light)">' in body
    assert '<meta name="theme-color" content="#1C1409" media="(prefers-color-scheme: dark)">' in body
    assert '<meta name="color-scheme" content="light dark">' in body


def test_theme_script_runs_before_the_stylesheet(client: TestClient) -> None:
    """No flash of the wrong theme: the script that applies a stored choice
    sits in <head>, after the theme-color metas it rewrites and before the
    <style> block, so data-theme is set before anything is painted. It wraps
    storage in try/catch, since blocked storage throws."""
    body = client.get("/_landing").text
    head = body[: body.index("</head>")]
    scripts = landing_routes.inline_scripts(body)
    first = scripts[0]
    at = head.index(first)
    assert head.rindex('name="theme-color"', 0, at) < at < head.index("<style>")
    assert 'localStorage.getItem("lmw-theme")' in first and "try {" in first
    assert 'setAttribute("data-theme", pick)' in first
    assert 'root.classList.add("js")' in first
    # The toggle script also guards every storage write.
    tail = scripts[1]
    assert 'localStorage.setItem(KEY, next)' in tail
    assert tail.index("try {", tail.index("toggle.addEventListener")) < tail.index("localStorage.setItem(KEY")


def test_theme_toggle_markup(client: TestClient) -> None:
    """A single quiet button in the header, keyboard-operable because it is a
    <button>, named for screen readers, with a status region that announces
    the change; hidden until the head script marks the page as scripted."""
    body = client.get("/_landing").text
    header = body[body.index("<header"):body.index("</header>")]
    button = re.search(r'<button class="theme-toggle" type="button"[^>]*>', header)
    assert button, "the toggle lives in the header"
    assert 'aria-label="Colour theme: Auto"' in button.group(0)
    assert 'data-pref="auto"' in button.group(0)
    for icon in ("auto", "light", "dark"):
        assert f'data-icon="{icon}"' in header
    assert '<span class="label">Auto</span>' in header
    assert 'role="status"' in header and "data-theme-status" in header
    css = _css(body)
    assert ".theme-toggle { display: none; }" in css
    assert ".js .theme-toggle {" in css


def test_requests_figure(client: TestClient) -> None:
    """Fig. 4 is the 7-day Requests chart, beside the key section, with a
    source comment naming the asset and figures numbered 1 to 5 in order."""
    src = (Path(landing_routes.__file__).parent / "landing.html").read_text()
    assert "<!-- src: shot-requests-7d.webp" in src
    body = client.get("/_landing").text
    nums = re.findall(r"<b>Fig\.&nbsp;(\d)</b>", body)
    assert nums == ["1", "2", "3", "4", "5"]
    img = re.search(r'<img src="/_landing/assets/shot-requests-7d\.webp\?v=[0-9a-f]{12}"[^>]*>', body)
    assert img and 'width="2464" height="884"' in img.group(0)
    keys = body[body.index('aria-labelledby="keys-title"'):body.index('aria-labelledby="leave-title"')]
    assert "shot-requests-7d.webp" in keys
    caption = re.search(r"<b>Fig\.&nbsp;4</b>(.*?)</figcaption>", body, re.S).group(1)
    for fact in ("7 days", "28,906 requests", "23&nbsp;September 2026", "key names are replaced"):
        assert fact in caption, fact
