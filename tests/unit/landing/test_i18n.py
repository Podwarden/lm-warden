"""Home in other languages (app/landing/i18n, README.md beside it): the
catalogs against the English source, the rendered locale pages, the
language a visitor of / is sent to, and the switcher."""

from __future__ import annotations

import base64
import hashlib
import json
import re

import pytest
from fastapi.testclient import TestClient

from app.landing import i18n
from app.landing.pages import HOME, LOCALES
from app.landing.render import inline_scripts, page_source, source_catalog
from app.landing.site import Analytics
from tests.unit.landing.test_routes import CANON, _canonical

#: Every locale whose catalog ships (en excluded).
SHIPPED = [loc.tag for loc in i18n.available() if loc.tag != "en"]
#: A locale with no catalog yet, for the "not published" paths (None once
#: all ten ship, and those tests skip).
UNSHIPPED = next((loc.tag for loc in LOCALES if not i18n.is_available(loc.tag)), None)


# --- Catalogs ------------------------------------------------------------------


def test_en_json_is_the_current_source() -> None:
    on_disk = json.loads((i18n.DIR / "en.json").read_text(encoding="utf-8"))
    assert on_disk == source_catalog(), (
        "app/landing/i18n/en.json is out of date with the marked English: run "
        "scripts/landing_i18n.py write-source, then refresh the stale translations"
    )


def test_every_visible_english_string_is_a_segment() -> None:
    """No English a reader sees on Home or its chrome escapes translation."""
    src = page_source(HOME, Analytics(ga4_id="G-SOURCE"), "en", every_part=True)
    found = [(u.node.tag, u.attr, u.text[:60]) for u in i18n.unmarked(src)]
    assert (
        not found
    ), f"unmarked text, mark it with data-t (scripts/landing_i18n.py unmarked): {found}"


def test_at_least_one_translation_ships() -> None:
    assert "de" in SHIPPED  # the pilot


@pytest.mark.parametrize("tag", SHIPPED)
def test_catalog_is_complete_fresh_and_faithful(tag: str) -> None:
    catalog = json.loads((i18n.DIR / f"{tag}.json").read_text(encoding="utf-8"))
    found = {k: v for k, v in i18n.catalog_problems(tag, catalog, source_catalog()).items() if v}
    hints = {
        "missing": "keys with no translation",
        "stale": "the English changed since these were translated (source_hash differs): retranslate",
        "unknown": "keys the English no longer has: delete them",
        "placeholders": "a tag, {{token}} or {arg} was dropped, added or altered",
        "code": "a <code> span was not kept verbatim",
        "numbers": "digits differ from the English",
        "glossary": "a do-not-translate term is missing",
        "untranslated": "identical to the English (allowlist in glossary.json identical_ok if deliberate)",
        "attribute": 'an attribute value contains " or <',
    }
    assert not found, "\n".join(
        f"{tag} {kind} ({hints[kind]}): {keys}" for kind, keys in found.items()
    )


def test_catalog_files_are_only_known_locales() -> None:
    names = {p.stem for p in i18n.DIR.glob("*.json")} - {"en", "glossary"}
    assert names <= {loc.tag for loc in LOCALES}, names


def test_limits_are_kept() -> None:
    from app.landing.render import _LIMITS

    for tag in SHIPPED:
        catalog = json.loads((i18n.DIR / f"{tag}.json").read_text(encoding="utf-8"))
        for key, limit in _LIMITS.items():
            text = re.sub(r"<[^>]+>", "", catalog[key]["text"])
            assert len(text) <= limit, (tag, key, text)


def test_checks_catch_broken_translations() -> None:
    src = {"k": {"text": 'See <a href="/x">2.05 s</a> in <code>vLLM</code> on Linux', "hash": "h"}}
    bad = {"k": {"text": "Siehe 2,5 s in vllm", "source_hash": "old"}}
    found = i18n.catalog_problems("de", bad, src)
    assert found["stale"] == ["k"] and found["placeholders"] == ["k"] and found["code"] == ["k"]
    assert found["numbers"] == ["k"] and found["glossary"] == ["k: vLLM, Linux"]
    good = {
        "k": {
            "text": 'Siehe <a href="/x">2,05 s</a> in <code>vLLM</code> unter Linux',
            "source_hash": "h",
        }
    }
    assert not any(i18n.catalog_problems("de", good, src).values())


# --- Rendered pages ------------------------------------------------------------------


@pytest.mark.parametrize("tag", SHIPPED)
def test_locale_home_head(client: TestClient, tag: str) -> None:
    _canonical(client, CANON)
    r = client.get(f"/{tag}/")
    assert r.status_code == 200
    body = r.text
    assert f'<html lang="{tag}">' in body
    assert f'<link rel="canonical" href="{CANON}/{tag}/">' in body
    alternates = dict(re.findall(r'<link rel="alternate" hreflang="([^"]+)" href="([^"]+)">', body))
    expected = {loc.tag: CANON + i18n.home_path(loc.tag) for loc in i18n.available()}
    assert alternates == {**expected, "x-default": CANON + "/"}
    assert f'<meta property="og:locale" content="{i18n.locale(tag).og}">' in body
    assert '<meta property="og:locale:alternate" content="en_US">' in body
    app = next(
        json.loads(b)
        for b in re.findall(r'<script type="application/ld\+json">\n(.*?)\n</script>', body, re.S)
        if '"SoftwareApplication"' in b
    )
    assert app["inLanguage"] == tag and app["url"] == f"{CANON}/{tag}/"
    catalog = json.loads((i18n.DIR / f"{tag}.json").read_text(encoding="utf-8"))
    assert app["description"] == catalog["meta.description"]["text"]
    assert app["featureList"][0] == catalog["jsonld.feature-1"]["text"]
    assert "data-t" not in body  # markers never ship


def test_english_home_announces_its_translations(client: TestClient) -> None:
    _canonical(client, CANON)
    body = client.get("/_landing").text
    assert '<html lang="en">' in body
    assert f'<link rel="canonical" href="{CANON}/">' in body
    assert '<link rel="alternate" hreflang="x-default" href="https://lmwarden.com/">' in body
    for tag in SHIPPED:
        assert f'hreflang="{tag}" href="{CANON}/{tag}/"' in body
    assert '<meta property="og:locale" content="en_US">' in body


def test_other_pages_stay_english_and_name_no_alternates(client: TestClient) -> None:
    _canonical(client, CANON)
    body = client.get("/faq").text
    assert '<html lang="en">' in body and 'hreflang="x-default"' not in body


def test_german_home_is_german(client: TestClient) -> None:
    body = client.get("/de/").text
    assert '<h1 id="hero-title">Ihre GPUs, bereitgestellt wie eine gehostete API.</h1>' in body
    assert "<title>LM Warden: selbst gehostetes LLM-Gateway für Ihre eigenen GPUs</title>" in body
    # Links to Home's sections stay on the German Home; other pages are English.
    assert '<a class="btn go" href="/de/#install">Installieren</a>' in body
    assert '<a class="brand" href="/de/"' in body
    assert '<a href="/de/#why">Anwendungsfälle</a>' in body
    assert '<a href="/faq">FAQ</a>' in body
    # Script labels travel as attributes the shared scripts read.
    assert 'data-l-copied="Kopiert"' in body and 'data-l-dot="Anwendungsfall {n} von {m}"' in body


def test_scripts_are_identical_in_every_language_and_hashed(client: TestClient) -> None:
    en = client.get("/_landing")
    assert inline_scripts(en.text)
    for tag in SHIPPED:
        r = client.get(f"/{tag}/")
        assert inline_scripts(r.text) == inline_scripts(en.text)
        csp = r.headers["content-security-policy"]
        for body in inline_scripts(r.text):
            digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
            assert f"'sha256-{digest}'" in csp


def test_unpublished_locale_is_404_and_not_offered(client: TestClient) -> None:
    if UNSHIPPED is None:
        pytest.skip("every locale ships")
    assert client.get(f"/{UNSHIPPED}/").status_code == 404
    body = client.get("/_landing").text
    assert f'hreflang="{UNSHIPPED}"' not in body and f"?lang={UNSHIPPED}" not in body
    r = client.get(f"/?lang={UNSHIPPED}", follow_redirects=False)
    assert r.status_code == 200 and i18n.COOKIE not in r.headers.get("set-cookie", "")


def test_slashless_locale_redirects(client: TestClient) -> None:
    r = client.get("/de?lang=de", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/de/?lang=de"


def test_sitemap_lists_translations(client: TestClient) -> None:
    _canonical(client, CANON)
    body = client.get("/sitemap.xml").text
    for tag in SHIPPED:
        assert f"<loc>{CANON}/{tag}/</loc>" in body


# --- The switcher -----------------------------------------------------------------------


def test_switcher_lists_english_first_then_native_names(client: TestClient) -> None:
    body = client.get("/de/").text
    bar = re.search(r'<details class="langsw">.*?</details>', body, re.S).group(0)
    names = re.findall(r'hreflang="([^"]+)" lang="[^"]+"(?: aria-current="true")?>([^<]+)</a>', bar)
    assert names == [(loc.tag, loc.name) for loc in i18n.available()]
    assert names[0] == ("en", "English")
    assert '<a href="/de/?lang=de" hreflang="de" lang="de" aria-current="true">Deutsch</a>' in bar
    assert "<summary>" in bar and '<span class="ln" lang="de">Deutsch</span>' in bar
    # The visible way back, and the same list in the phone's menu sheet.
    assert (
        '<a class="en-link" href="/?lang=en" hreflang="en" lang="en" aria-label="English">' in body
    )
    assert '<li class="langs-m">' in body
    assert 'class="en-link"' not in client.get("/_landing").text


# --- Choosing the language --------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "cookie", "shipped", "want"),
    [
        (None, None, {"de"}, "en"),
        ("de-DE,de;q=0.9,en;q=0.8", None, {"de"}, "de"),
        ("en-US,en;q=0.9,de;q=0.8", None, {"de"}, "en"),
        ("en;q=0.5,de;q=0.9", None, {"de"}, "de"),  # q wins over order
        ("de;q=0", None, {"de"}, "en"),  # q=0: not acceptable
        ("ko-KR,de;q=0.5", None, {"de"}, "de"),  # unsupported first, then a match
        ("ko-KR,ja;q=0.5", None, {"de"}, "en"),
        ("fr-CA", None, {"de"}, "en"),  # supported language, catalog not shipped
        ("fr-CA", None, {"fr"}, "fr"),
        ("zh", None, {"zh-CN"}, "zh-CN"),
        ("zh-Hans", None, {"zh-CN"}, "zh-CN"),
        ("zh-SG", None, {"zh-CN"}, "zh-CN"),
        ("zh-Hans-HK", None, {"zh-CN"}, "zh-CN"),
        ("zh-TW,zh-Hant;q=0.9", None, {"zh-CN"}, "en"),  # never Simplified for Traditional
        ("pt-PT", None, {"pt-BR"}, "pt-BR"),
        ("pt", None, {"pt-BR"}, "pt-BR"),
        ("UK-ua", None, {"uk"}, "uk"),
        ("hi-IN", None, {"hi"}, "hi"),
        ("*", None, {"de"}, "en"),
        ("de;q=abc", None, {"de"}, "en"),
        ("de", "en", {"de"}, "en"),  # the switcher's choice beats the browser
        ("en", "de", {"de"}, "de"),
        ("de", "fr", {"de"}, "de"),  # a cookie naming an unshipped locale is ignored
        ("de", "garbage", {"de"}, "de"),
    ],
)
def test_negotiate(
    monkeypatch: pytest.MonkeyPatch,
    header: str | None,
    cookie: str | None,
    shipped: set[str],
    want: str,
) -> None:
    monkeypatch.setattr(i18n, "_CATALOGS", {tag: {} for tag in shipped})
    assert i18n.negotiate(header, cookie) == want


def test_browser_is_sent_to_its_language(client: TestClient) -> None:
    r = client.get("/", headers={"Accept-Language": "de-DE,de;q=0.9"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/de/"
    assert (
        r.headers["vary"] == "Accept-Language, Cookie" and r.headers["cache-control"] == "private"
    )
    r = client.get("/_landing", headers={"Accept-Language": "de"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/de/"


def test_english_home_varies_and_stays_english(client: TestClient) -> None:
    r = client.get("/_landing", headers={"Accept-Language": "en-GB,ko"}, follow_redirects=False)
    assert r.status_code == 200 and r.headers["vary"] == "Accept-Language, Cookie"
    assert r.headers["cache-control"] == "private"


@pytest.mark.parametrize(
    "agent",
    [
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
        "Mozilla/5.0 (compatible; YandexBot/3.0)",
        "facebookexternalhit/1.1",
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; GPTBot/1.2)",
    ],
)
def test_crawlers_are_never_redirected(client: TestClient, agent: str) -> None:
    r = client.get(
        "/", headers={"Accept-Language": "de", "User-Agent": agent}, follow_redirects=False
    )
    assert r.status_code == 200 and '<html lang="en">' in r.text


def test_head_and_localized_urls_are_never_redirected(client: TestClient) -> None:
    assert (
        client.head("/", headers={"Accept-Language": "de"}, follow_redirects=False).status_code
        == 200
    )
    client.cookies.set(i18n.COOKIE, "en")
    r = client.get("/de/", headers={"Accept-Language": "de"}, follow_redirects=False)
    assert r.status_code == 200 and '<html lang="de">' in r.text


def test_cookie_overrides_the_browser(client: TestClient) -> None:
    client.cookies.set(i18n.COOKIE, "en")
    r = client.get("/", headers={"Accept-Language": "de"}, follow_redirects=False)
    assert r.status_code == 200
    client.cookies.set(i18n.COOKIE, "de")
    r = client.get("/", headers={"Accept-Language": "en"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/de/"


@pytest.mark.parametrize(("path", "lang", "target"), [("/", "de", "/de/"), ("/de/", "en", "/")])
def test_lang_query_sets_the_cookie_and_cleans_the_url(
    client: TestClient, path: str, lang: str, target: str
) -> None:
    r = client.get(f"{path}?lang={lang}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == target
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{i18n.COOKIE}={lang};")
    assert "Max-Age=31536000" in cookie and "Path=/" in cookie and "SameSite=lax" in cookie
    assert "Secure" not in cookie  # plain http in the test client


@pytest.mark.parametrize(("path", "lang"), [("/", "en"), ("/de/", "de")])
def test_lang_query_for_the_page_already_shown_serves_it_and_sets_the_cookie(
    client: TestClient, path: str, lang: str
) -> None:
    # No redirect to the clean URL: with cookies blocked, Accept-Language would
    # send the visitor straight back, and English would be unreachable.
    r = client.get(f"{path}?lang={lang}", headers={"Accept-Language": "de"}, follow_redirects=False)
    assert r.status_code == 200
    assert r.headers["set-cookie"].startswith(f"{i18n.COOKIE}={lang};")


def test_english_is_reachable_with_cookies_blocked(client: TestClient) -> None:
    client.cookies.clear()
    r = client.get("/?lang=en", headers={"Accept-Language": "de"}, follow_redirects=False)
    assert r.status_code == 200 and '<html lang="en"' in r.text


@pytest.mark.parametrize(
    "agent",
    [
        "Mozilla/5.0 (compatible; Google-InspectionTool/1.0;)",
        "Mozilla/5.0 (compatible; GoogleOther)",
    ],
)
def test_google_tools_are_not_redirected(client: TestClient, agent: str) -> None:
    r = client.get(
        "/", headers={"Accept-Language": "de", "User-Agent": agent}, follow_redirects=False
    )
    assert r.status_code == 200


def test_lang_cookie_is_secure_behind_https(client: TestClient) -> None:
    r = client.get("https://testserver/?lang=de", follow_redirects=False)
    assert "Secure" in r.headers["set-cookie"]


def test_localized_paths_are_public_and_cookieless() -> None:
    from app.auth.csrf import _mints_cookie
    from app.auth.policy import PUBLIC_ROUTES

    for loc in LOCALES[1:]:
        for path in (f"/{loc.tag}/", f"/{loc.tag}"):
            assert ("GET", path) in PUBLIC_ROUTES and not _mints_cookie(path), path
