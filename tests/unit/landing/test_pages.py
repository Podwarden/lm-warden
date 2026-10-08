"""Registry-wide checks over every website page (spec 2026-10-07 §3, §6, §10)."""

import html
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth.csrf import _mints_cookie
from app.auth.policy import PUBLIC_ROUTES
from app.landing.pages import PAGES
from tests.unit.landing.conftest import all_pages, fetch

GOLDEN = Path(__file__).parents[2] / "fixtures/landing/landing-a4f06434.html"


def _disable(tmp_data_dir: Path) -> None:
    from tests.unit.landing.test_routes import _disable as off

    off(tmp_data_dir)


def test_expected_pages_exist() -> None:
    paths = {p.path for p in PAGES}
    assert {
        "/",
        "/coding-agents",
        "/agent-tuning",
        "/gpu-monitoring",
        "/cache-aware-routing",
        "/api-keys",
        "/features",
        "/faq",
    } <= paths


def test_titles_and_descriptions_unique_and_short(client: TestClient) -> None:
    titles, descs = set(), set()
    for path, body in all_pages(client).items():
        t = html.unescape(re.search(r"<title>(.*?)</title>", body, re.S).group(1))
        d = html.unescape(re.search(r'<meta name="description" content="([^"]+)"', body).group(1))
        assert len(t) < 60 and len(d) < 160, path
        assert t not in titles and d not in descs, path
        titles.add(t)
        descs.add(d)


def test_one_h1_per_page(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        assert len(re.findall(r"<h1[\s>]", body)) == 1, path


def test_every_internal_link_resolves(client: TestClient) -> None:
    known = {p.path for p in PAGES} | {
        "/ui/login",
        "/llms.txt",
        "/llms-full.txt",
        "/agent-guide.md",
    }
    for path, body in all_pages(client).items():
        for href in re.findall(r'href="(/[^"#?]*)', body):
            if href.startswith("/_landing/assets/") or href.startswith("/ui/icon"):
                continue
            assert href in known or href.startswith("/coding-agents/"), (path, href)


def test_every_page_route_is_public_and_cookieless() -> None:
    for p in PAGES:
        assert ("GET", p.path) in PUBLIC_ROUTES, p.path
        assert not _mints_cookie(p.path), p.path


def test_menu_marks_the_current_page(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        nav = re.search(r'<nav aria-label="Primary">(.*?)</nav>', body, re.S).group(1)
        assert nav.count('aria-current="page"') <= 1, path
        assert '<details class="menu"' in nav  # the phone menu


def test_non_home_pages_show_breadcrumbs(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        if path == "/":
            assert 'aria-label="Breadcrumb"' not in body
        else:
            assert '<nav class="crumbs" aria-label="Breadcrumb">' in body, path


def test_no_accordion_hides_page_content(client: TestClient) -> None:
    """The phone menu is a <details>; nothing inside <main> is."""
    for path, body in all_pages(client).items():
        main = re.search(r"<main\b.*?</main>", body, re.S).group(0)
        assert "<details" not in main, path


def test_trailing_slash_redirects_to_the_canonical_path(client: TestClient) -> None:
    r = client.get("/faq/", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/faq"


def test_disabled_site_404s_every_page(client: TestClient, tmp_data_dir: Path) -> None:
    _disable(tmp_data_dir)
    for p in PAGES:
        if p.path != "/":
            assert client.get(p.path).status_code == 404, p.path


def test_old_anchors_map_to_their_new_pages(client: TestClient) -> None:
    home = fetch(client, "/")
    for anchor, page in (
        ("#faq", "/faq"),
        ("#router-title", "/coding-agents/claude-code"),
        ("#cards-title", "/gpu-monitoring"),
        ("#replica-title", "/cache-aware-routing"),
        ("#keys-title", "/api-keys"),
        ("#small-title", "/features"),
    ):
        assert f'"{anchor}": "{page}"' in home, anchor
    assert '"#faq"' not in fetch(client, "/faq")  # the map runs on Home only


def _blocks(body: str) -> list[str]:
    main = re.search(r"<main\b.*?</main>", body, re.S).group(0)
    main = re.sub(r"<!--.*?-->", "", main, flags=re.S)
    text = re.sub(r"\s+", " ", main)
    # A section's first heading may now be its page's <h1>; a link to a
    # section now on Home reads "/#id" instead of "#id".
    text = re.sub(r"<h1\b", "<h2", text).replace("</h1>", "</h2>")
    text = text.replace('href="/#', 'href="#')
    tags = r"p|h2|h3|li|figcaption|td|th|dt|dd"
    return [m.strip() for m in re.findall(rf"<(?:{tags})\b[^>]*>(.*?)</(?:{tags})>", text)]


#: Blocks of the pre-split page that were deliberately rewritten, with why.
REWRITTEN = {
    # Hero lede named only Claude Code; now names the other agents too (spec §3).
    "LM Warden runs vLLM and llama.cpp on your NVIDIA cards behind one port. Each app gets "
    "its own key, and each card reports for itself. Point Claude Code at it, and a model on "
    "your cards answers for the Claude models you choose, under their names, while the rest "
    "go to Anthropic on your own login.": "multi-agent hero, spec §3",
    # Promoted to the GPU page's <h1> with the topic in front, for search.
    "The box behind these numbers.": "GPU monitoring: the box behind these numbers.",
}


def test_nothing_from_the_single_page_was_lost(client: TestClient) -> None:
    """Spec §1: reorganise, do not remove. Every heading, paragraph, list
    item, caption, table cell and definition of the pre-split page appears
    on some page of the site."""
    old = _blocks(GOLDEN.read_text())
    new = " ".join(" ".join(_blocks(b)) for b in all_pages(client).values())
    new = re.sub(r"\s+", " ", new)
    missing = [b for b in old if b and b not in new and b not in REWRITTEN]
    assert not missing, missing[:5]
