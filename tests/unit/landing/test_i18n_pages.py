"""Every page in other languages (owner, 2026-10-21: "I select Russian, click
Coding agents and it opened the English page; then I click Russian and it
opened main"). The pages, their links and switcher, the cookie redirect on
English sub-pages, SEO per page, and the per-part catalogs. Home's own
behaviour is in test_i18n.py."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.landing import i18n
from app.landing.clients import CLIENT_PAGES
from app.landing.pages import AGENTS, CACHE, FAQ, HOME, LOCALES, NOT_FOUND, PAGES, Page
from app.landing.render import (
    inline_scripts,
    marked_source,
    pages_of,
    render_page,
    source_catalog,
)
from tests.unit.landing.test_routes import CANON, _canonical

#: Every page a visitor can reach (the tag-only /privacy needs a Google ID).
SUB_PAGES: list[Page] = [p for p in PAGES if p.path != "/" and not p.tag_only] + list(
    CLIENT_PAGES.values()
)
#: (locale, page) for every non-English page that ships.
SHIPPED = [(loc.tag, p) for p in SUB_PAGES for loc in i18n.page_locales(p) if loc.tag != "en"]
#: The pilot: Russian ships every page.
RU = "ru"
#: A locale whose Home ships but whose other pages do not (None once one does).
HOME_ONLY = next(
    (
        loc.tag
        for loc in LOCALES[1:]
        if i18n.is_available(loc.tag) and not i18n.page_ships(FAQ, loc.tag)
    ),
    None,
)
_SITE_HREF = re.compile(r'href="(/[^"#?]*)([^"]*)"')


def _ids(pairs: list[tuple[str, Page]]) -> list[str]:
    return [f"{tag}{p.path}" for tag, p in pairs]


# --- Catalogs, across every part ----------------------------------------------------


@pytest.mark.parametrize("part", ["home", *i18n.PARTS])
def test_part_sources_are_current(part: str) -> None:
    on_disk = json.loads(i18n.source_path(part).read_text(encoding="utf-8"))
    assert on_disk == source_catalog(part), (
        f"{i18n.source_path(part).name} ({part}) is out of date: run "
        "scripts/landing_i18n.py write-source, then refresh the stale translations"
    )


@pytest.mark.parametrize("page", [*SUB_PAGES, NOT_FOUND, PAGES[-1]], ids=lambda p: p.path)
def test_every_visible_string_on_every_page_is_a_segment(page: Page) -> None:
    src = marked_source(page)
    found = [(u.node.tag, u.attr, u.text[:60]) for u in i18n.unmarked(src)]
    assert not found, f"{page.path}: unmarked text (scripts/landing_i18n.py unmarked): {found}"
    parts = i18n.page_parts(page)
    stray = [s.key for s in i18n.segments(src) if i18n.part_of(s.key) not in parts]
    assert not stray, f"{page.path}: keys outside its parts {parts}: {stray}"


def _part_catalogs() -> Iterator[tuple[str, str]]:
    for part in i18n.PARTS:
        for loc in LOCALES[1:]:
            if i18n.catalog_path(part, loc.tag).is_file():
                yield loc.tag, part


@pytest.mark.parametrize(("tag", "part"), list(_part_catalogs()))
def test_part_catalog_is_complete_fresh_and_faithful(tag: str, part: str) -> None:
    """A part file that ships must pass every check (a partial one would not be
    served, but must not be committed either: the README's hand-off)."""
    catalog = json.loads(i18n.catalog_path(part, tag).read_text(encoding="utf-8"))
    found = {
        k: v for k, v in i18n.catalog_problems(tag, catalog, source_catalog(part)).items() if v
    }
    assert not found, f"{tag}/{part}.json: {found}"


def test_part_files_are_only_known_parts_and_locales() -> None:
    tags = {loc.tag for loc in LOCALES}
    for d in i18n.DIR.iterdir():
        if d.is_dir() and not d.name.startswith("__"):
            assert d.name in tags, d
            assert {f.stem for f in d.glob("*.json")} <= set(i18n.PARTS), d


def test_russian_is_the_pilot_and_ships_every_page() -> None:
    for page in [*PAGES, NOT_FOUND, *CLIENT_PAGES.values()]:
        assert i18n.page_ships(page, RU), page.path


def test_keys_belong_to_their_part() -> None:
    assert i18n.part_of("chrome.nav.faq") == "home"
    assert i18n.part_of("faq.q-3") == "faq"
    assert i18n.part_of("clients.aider.summary") == "clients"
    with pytest.raises(ValueError):
        i18n.part_of("nosuchpart.x")


def test_a_partial_part_file_is_not_served(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "en").mkdir()
    (tmp_path / "xx").mkdir()
    src = {"faq.a": {"text": "A", "note": "", "hash": "1"}, "faq.b": {"text": "B", "hash": "2"}}
    (tmp_path / "en" / "faq.json").write_text(json.dumps(src))
    partial = {"faq.a": {"text": "Ä", "source_hash": "1"}}
    (tmp_path / "xx" / "faq.json").write_text(json.dumps(partial))
    monkeypatch.setattr(i18n, "DIR", tmp_path)
    monkeypatch.setattr(i18n, "PARTS", ("faq",))
    monkeypatch.setattr(i18n, "LOCALES", (LOCALES[0], i18n.Locale("xx", "Xx", "xx_XX")))
    assert i18n._load_parts() == {"faq": {}}
    full = {**partial, "faq.b": {"text": "Ḃ", "source_hash": "2"}}
    (tmp_path / "xx" / "faq.json").write_text(json.dumps(full))
    assert i18n._load_parts() == {"faq": {"xx": {"faq.a": "Ä", "faq.b": "Ḃ"}}}


# --- Rendered pages ----------------------------------------------------------------------


@pytest.mark.parametrize(("tag", "page"), SHIPPED, ids=_ids(SHIPPED))
def test_localized_page_head_and_body(client: TestClient, tag: str, page: Page) -> None:
    _canonical(client, CANON)
    r = client.get(i18n.page_path(page, tag))
    assert r.status_code == 200
    body = r.text
    assert f'<html lang="{tag}">' in body
    assert f'<link rel="canonical" href="{CANON}{i18n.page_path(page, tag)}">' in body
    alternates = dict(re.findall(r'<link rel="alternate" hreflang="([^"]+)" href="([^"]+)">', body))
    want = {loc.tag: CANON + i18n.page_path(page, loc.tag) for loc in i18n.page_locales(page)}
    assert alternates == {**want, "x-default": CANON + page.path}
    assert f'<meta property="og:locale" content="{i18n.locale(tag).og}">' in body
    assert not re.search(r' data-t(?:-[a-z-]+)?="', body)  # markers never ship
    assert not re.search(r"\{\{[a-z_]", body)  # every token filled
    for block in re.findall(r'<script type="application/ld\+json">\n(.*?)\n</script>', body, re.S):
        obj = json.loads(block)
        if obj["@type"] in {"TechArticle", "HowTo", "FAQPage"}:
            assert obj["inLanguage"] == tag, obj["@type"]
        if obj["@type"] == "TechArticle":
            assert obj["url"] == CANON + i18n.page_path(page, tag)
        if obj["@type"] == "BreadcrumbList":
            assert obj["itemListElement"][0]["item"] == f"{CANON}/{tag}/"


@pytest.mark.parametrize(("tag", "page"), SHIPPED, ids=_ids(SHIPPED))
def test_localized_pages_scripts_are_hashed(client: TestClient, tag: str, page: Page) -> None:
    r = client.get(i18n.page_path(page, tag))
    en = client.get(page.path)
    assert inline_scripts(r.text) == inline_scripts(en.text)
    for script in inline_scripts(r.text):
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in r.headers["content-security-policy"]


def test_russian_pages_are_russian(client: TestClient) -> None:
    body = client.get("/ru/coding-agents").text
    assert "<title>Агенты для программирования на ваших GPU | LM Warden</title>" in body
    assert "Ваш агент для программирования на ваших собственных GPU." in body
    assert '<th scope="col">Клиент</th>' in body
    body = client.get("/ru/coding-agents/claude-code").text
    assert "<title>Claude Code с локальной LLM на ваших GPU | LM Warden</title>" in body
    # Code stays code, verbatim, inside the Russian prose.
    assert "<code>ANTHROPIC_API_KEY</code>" in body and "<code>/v1</code>" in body
    assert "<span>Копировать</span>" in body
    body = client.get("/ru/faq").text
    faq = next(
        json.loads(b)
        for b in re.findall(r'<script type="application/ld\+json">\n(.*?)\n</script>', body, re.S)
        if '"FAQPage"' in b
    )
    assert faq["mainEntity"][0]["name"] == "Что такое LM Warden?"
    assert "<code>" not in faq["mainEntity"][0]["acceptedAnswer"]["text"]


# --- Links and the switcher --------------------------------------------------------------


def test_links_inside_a_localized_page_stay_in_its_locale(client: TestClient) -> None:
    site = {p.path for p in [*PAGES, *CLIENT_PAGES.values()]}
    for page in SUB_PAGES:
        body = client.get(i18n.page_path(page, RU)).text
        for path, rest in _SITE_HREF.findall(body):
            if "lang=" in rest:
                continue  # the switcher, below
            assert path not in site, (page.path, path + rest)
        main = re.search(r"<main\b.*?</main>", body, re.S).group(0)
        nav = re.search(r'<nav aria-label="[^"]*">(.*?)</nav>', body, re.S).group(1)
        assert 'href="/ru/coding-agents"' in nav and 'href="/ru/faq"' in nav, page.path
        assert 'href="/ru/#install"' in body  # the bar's install button
        # External and app links are untouched.
        assert 'href="/ui/login"' in body and 'href="/llms.txt"' in body
        assert "https://github.com/Podwarden/lm-warden" in body
        assert main


def test_client_tiles_crumbs_and_related_links_are_localized(client: TestClient) -> None:
    body = client.get("/ru/coding-agents").text
    for c in CLIENT_PAGES:
        assert f'href="/ru/coding-agents/{c}"' in body, c
    body = client.get("/ru/coding-agents/aider").text
    crumbs = re.search(r'<nav class="crumbs"[^>]*>.*?</nav>', body, re.S).group(0)
    assert 'aria-label="Навигационная цепочка"' in crumbs
    assert '<a href="/ru/">Главная</a>' in crumbs
    assert '<a href="/ru/coding-agents">Агенты для кода</a>' in crumbs
    assert 'href="/ru/agent-tuning"' in body and 'href="/ru/api-keys"' in body
    # Home's own links to the sub-pages follow too.
    home = client.get("/ru/").text
    assert 'href="/ru/coding-agents/claude-code"' in home or 'href="/ru/coding-agents"' in home
    assert 'href="/coding-agents"' not in home


@pytest.mark.parametrize(
    "page", SUB_PAGES[:8] + [CLIENT_PAGES["claude-code"]], ids=lambda p: p.path
)
def test_switcher_links_to_the_same_page(client: TestClient, page: Page) -> None:
    body = client.get(i18n.page_path(page, RU)).text
    bar = re.search(r'<details class="langsw">.*?</details>', body, re.S).group(0)
    links = re.findall(r'<a href="([^"]+)" hreflang="([^"]+)"', bar)
    assert links == [
        (f"{i18n.page_path(page, loc.tag)}?lang={loc.tag}", loc.tag)
        for loc in i18n.page_locales(page)
    ]
    assert (
        f'<a class="en-link" href="{page.path}?lang=en" hreflang="en" lang="en"' in body
    ), page.path
    sheet = re.search(r'<li class="langs-m">.*?</ul></li>', body, re.S).group(0)
    assert f'href="{page.path}?lang=en"' in sheet and f'href="/ru{page.path}?lang=ru"' in sheet
    en = client.get(page.path).text
    assert f'<a href="/ru{page.path}?lang=ru" hreflang="ru"' in en
    assert 'class="en-link"' not in en


def test_the_owners_click_path(client: TestClient) -> None:
    """/ru/ -> Coding agents -> English -> the same page in English -> Русский
    -> back to /ru/coding-agents."""
    home = client.get("/ru/", follow_redirects=False)
    assert home.status_code == 200
    href = re.search(r'<a href="([^"]+)"[^>]*>Агенты для кода</a>', home.text).group(1)
    assert href == "/ru/coding-agents"
    page = client.get(href, follow_redirects=False)
    assert page.status_code == 200 and '<html lang="ru">' in page.text
    english = re.search(r'<a class="en-link" href="([^"]+)"', page.text).group(1)
    assert english == "/coding-agents?lang=en"
    r = client.get(english, follow_redirects=False)
    assert r.status_code == 200 and '<html lang="en">' in r.text
    assert r.headers["set-cookie"].startswith(f"{i18n.COOKIE}=en;")
    russian = re.search(r'<a href="([^"]+)" hreflang="ru"', r.text).group(1)
    assert russian == "/ru/coding-agents?lang=ru"
    r = client.get(russian, follow_redirects=False)
    assert r.status_code == 200 and '<html lang="ru">' in r.text
    assert r.headers["set-cookie"].startswith(f"{i18n.COOKIE}=ru;")


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ("/faq?lang=ru", "/ru/faq"),
        ("/ru/faq?lang=en", "/faq"),
        ("/coding-agents/aider?lang=ru", "/ru/coding-agents/aider"),
        ("/ru/coding-agents/aider?lang=en", "/coding-agents/aider"),
    ],
)
def test_lang_query_on_a_sub_page_goes_to_the_same_page(
    client: TestClient, path: str, target: str
) -> None:
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == target
    assert r.headers["set-cookie"].startswith(f"{i18n.COOKIE}={path.rsplit('=', 1)[1]};")


# --- The cookie on English sub-pages -------------------------------------------------------


def test_cookie_sends_an_english_sub_page_to_its_language(client: TestClient) -> None:
    client.cookies.set(i18n.COOKIE, RU)
    for page in (AGENTS, FAQ, CACHE, CLIENT_PAGES["claude-code"]):
        r = client.get(page.path, follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == f"/ru{page.path}", page.path
        assert r.headers["vary"] == "Cookie" and r.headers["cache-control"] == "private"


def test_lang_en_keeps_a_russian_visitor_on_english(client: TestClient) -> None:
    client.cookies.set(i18n.COOKIE, RU)
    r = client.get("/faq?lang=en", follow_redirects=False)
    assert r.status_code == 200 and '<html lang="en">' in r.text
    assert r.headers["set-cookie"].startswith(f"{i18n.COOKIE}=en;")
    client.cookies.set(i18n.COOKIE, "en")
    assert client.get("/faq", follow_redirects=False).status_code == 200


@pytest.mark.parametrize(
    "agent",
    [
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; YandexBot/3.0)",
    ],
)
def test_crawlers_and_head_are_not_redirected_on_sub_pages(client: TestClient, agent: str) -> None:
    client.cookies.set(i18n.COOKIE, RU)
    r = client.get("/faq", headers={"User-Agent": agent}, follow_redirects=False)
    assert r.status_code == 200 and '<html lang="en">' in r.text
    assert client.head("/faq", follow_redirects=False).status_code == 200


def test_accept_language_alone_does_not_move_a_sub_page(client: TestClient) -> None:
    client.cookies.clear()
    r = client.get("/faq", headers={"Accept-Language": "ru-RU,ru"}, follow_redirects=False)
    assert r.status_code == 200 and '<html lang="en">' in r.text


def test_cookie_for_a_locale_without_that_page_stays_english(client: TestClient) -> None:
    if HOME_ONLY is None:
        pytest.skip("every locale with a Home ships its other pages too")
    client.cookies.set(i18n.COOKIE, HOME_ONLY)
    assert client.get("/faq", follow_redirects=False).status_code == 200
    assert client.get(f"/{HOME_ONLY}/faq").status_code == 404
    body = client.get(f"/{HOME_ONLY}/").text
    assert 'href="/faq"' in body  # an unshipped page keeps its English link


def test_localized_url_is_never_redirected_by_the_cookie(client: TestClient) -> None:
    client.cookies.set(i18n.COOKIE, "en")
    assert client.get("/ru/faq", follow_redirects=False).status_code == 200


# --- Partial coverage ---------------------------------------------------------------------


@pytest.fixture
def ru_faq_withdrawn(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """As if ru/faq.json were incomplete: the part does not ship in ru."""
    parts = {part: dict(cats) for part, cats in i18n._PARTS.items()}
    parts["faq"].pop(RU)
    monkeypatch.setattr(i18n, "_PARTS", parts)
    render_page.cache_clear()
    yield
    render_page.cache_clear()


@pytest.mark.usefixtures("ru_faq_withdrawn")
def test_an_incomplete_page_is_404_and_offered_nowhere(client: TestClient) -> None:
    _canonical(client, CANON)
    r = client.get("/ru/faq")
    assert r.status_code == 404 and '<html lang="ru">' in r.text  # the 404, in Russian
    faq = client.get("/faq").text
    # Russian is not offered for the FAQ; the switcher still lists the languages
    # that do publish it.
    assert 'hreflang="ru"' not in faq and "/ru/faq" not in faq
    other = client.get("/ru/coding-agents").text
    assert 'href="/faq"' in other and 'href="/ru/faq"' not in other
    assert "/ru/faq" not in client.get("/sitemap.xml").text
    client.cookies.set(i18n.COOKIE, RU)
    assert client.get("/faq", follow_redirects=False).status_code == 200
    assert client.get("/ru/", follow_redirects=False).status_code == 200  # Home unaffected


# --- Routes ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "target"),
    [("/ru/faq/", "/ru/faq"), ("/ru/coding-agents/aider/", "/ru/coding-agents/aider")],
)
def test_localized_trailing_slash_redirects(client: TestClient, path: str, target: str) -> None:
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == target


@pytest.mark.parametrize("path", ["/ru/nope", "/ru/coding-agents/nope", "/ru/faq/x/y"])
def test_404_localizes_under_a_locale_prefix(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 404 and '<html lang="ru">' in r.text
    assert "Такой страницы здесь нет." in r.text and 'href="/ru/coding-agents"' in r.text
    assert 'class="langsw"' not in r.text
    en = client.get("/nope")
    assert en.status_code == 404 and '<html lang="en">' in en.text


def test_localized_pages_are_off_with_the_site(client: TestClient, tmp_data_dir) -> None:
    from tests.unit.landing.test_routes import _disable

    _disable(tmp_data_dir)
    assert client.get("/ru/faq").status_code == 404
    assert client.get("/ru/coding-agents/aider").status_code == 404


def test_localized_sub_paths_are_public_and_cookieless() -> None:
    from app.auth.csrf import _mints_cookie
    from app.auth.policy import PUBLIC_ROUTES

    for loc in LOCALES[1:]:
        for path in (f"/{loc.tag}/faq", f"/{loc.tag}/coding-agents/{{client_id}}"):
            assert ("GET", path) in PUBLIC_ROUTES and not _mints_cookie(path), path


def test_sitemap_lists_every_page_in_every_language(client: TestClient) -> None:
    _canonical(client, CANON)
    body = client.get("/sitemap.xml").text
    for page in SUB_PAGES:
        for loc in i18n.page_locales(page):
            assert f"<loc>{CANON}{i18n.page_path(page, loc.tag)}</loc>" in body
    assert f"<loc>{CANON}/ru/coding-agents/claude-code</loc>" in body


def test_pages_of_each_part() -> None:
    assert pages_of("faq") == [FAQ]
    assert HOME not in pages_of("common") and NOT_FOUND in pages_of("common")
    assert set(pages_of("clients")) == set(CLIENT_PAGES.values())
