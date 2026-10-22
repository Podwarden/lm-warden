"""Google tag, consent banner, CSP and privacy page (spec 2026-10-07 §8)."""

import re
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.landing.pages import HOME
from app.landing.render import all_site_pages, inline_scripts, render_page
from app.landing.site import Analytics


def _set(tmp_data_dir: Path, **kv: str) -> None:
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db", isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        for k, v in kv.items():
            db.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (k, v),
            )
        db.execute("PRAGMA wal_checkpoint(FULL)")


def _get(client: TestClient, path: str):
    return client.get("/_landing" if path == "/" else path)


def test_no_id_means_no_google_anywhere(client: TestClient) -> None:
    for p in all_site_pages():
        r = _get(client, p.path)
        for needle in ("googletagmanager", "google-analytics", 'class="consent"', "gtag("):
            assert needle not in r.text, (p.path, needle)
        assert "google" not in r.headers["content-security-policy"], p.path
    assert client.get("/privacy").status_code == 404


def test_ga4_id_adds_tag_banner_and_csp(client: TestClient, tmp_data_dir: Path) -> None:
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    r = client.get("/faq")
    assert 'data-ids="G-AB12CD34EF"' in r.text
    banner = re.search(r'<div class="consent"[^>]*>', r.text)
    assert banner and "hidden" in banner.group(0)
    csp = r.headers["content-security-policy"]
    assert "https://www.googletagmanager.com" in csp
    assert "https://*.google-analytics.com" in csp
    assert "googleadservices" not in csp  # Ads origins only with an Ads ID
    assert client.get("/privacy").status_code == 200


def test_ads_id_adds_ads_origins(client: TestClient, tmp_data_dir: Path) -> None:
    _set(tmp_data_dir, analytics_google_ads_id="AW-123456789")
    r = client.get("/faq")
    assert 'data-ids="AW-123456789"' in r.text
    csp = r.headers["content-security-policy"]
    assert "https://www.googleadservices.com" in csp
    assert "https://googleads.g.doubleclick.net" in csp


def test_consent_defaults_to_denied_before_the_tag(client: TestClient, tmp_data_dir: Path) -> None:
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    body = client.get("/").text
    default = body.index("gtag('consent', 'default'")
    assert body.index("googletagmanager.com/gtag/js") > default  # the deferred loader
    block = body[default : default + 400]
    for k in ("ad_storage", "ad_user_data", "ad_personalization", "analytics_storage"):
        assert f"{k}: 'denied'" in block, k


def test_hostile_db_value_never_reaches_the_page(client: TestClient, tmp_data_dir: Path) -> None:
    _set(tmp_data_dir, analytics_ga4_id='G-1"><script>alert(1)</script>')
    body = client.get("/").text
    assert "alert(1)" not in body and "googletagmanager" not in body


def test_tag_scripts_do_not_vary_with_the_id() -> None:
    """IDs travel in a data attribute, so the inline scripts (and their CSP
    hashes) are the same whatever ID is configured."""
    a = render_page(HOME, "", Analytics(ga4_id="G-AAAA1111"))
    b = render_page(HOME, "", Analytics(ga4_id="G-BBBB2222"))
    assert inline_scripts(a) == inline_scripts(b)


def test_csp_hashes_cover_the_tag_scripts(client: TestClient, tmp_data_dir: Path) -> None:
    import base64
    import hashlib

    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    r = client.get("/faq")
    csp = r.headers["content-security-policy"]
    for body in inline_scripts(r.text):
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp


def test_search_console_meta_only_on_home(client: TestClient, tmp_data_dir: Path) -> None:
    token = "abcDEF123_-abcDEF123_-xyz"
    _set(tmp_data_dir, search_console_verification=token)
    home = client.get("/")
    assert f'<meta name="google-site-verification" content="{token}">' in home.text
    assert "google-site-verification" not in client.get("/faq").text
    assert "google" not in home.headers["content-security-policy"]


def test_footer_says_what_loads(client: TestClient, tmp_data_dir: Path) -> None:
    """With a tag ID the footer says Google's tag loads and links the privacy
    page and the cookie settings (the way back to the banner). Without one it
    is the approved footer exactly: the redesign (mockup B) dropped the old
    "makes no third-party requests" line, which has nothing to qualify when
    nothing third-party loads."""
    body = client.get("/faq").text
    assert 'class="tag-note"' not in body and "data-consent-open" not in body
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    for path in ("/", "/faq"):
        body = _get(client, path).text
        footer = body[body.index('<footer class="colophon"') : body.index("</footer>")]
        assert 'class="tag-note"' in footer, path
        assert 'href="/privacy"' in footer and "data-consent-open" in footer, path


def test_banner_on_every_page_only_with_an_id(client: TestClient, tmp_data_dir: Path) -> None:
    """Rule for the redesign: the consent banner renders on every page,
    Home included, whenever a GA4 or Ads ID is set, and on none when neither
    is (test_no_id_means_no_google_anywhere). It ships hidden and is shown by
    the boot script only while no choice is stored; Decline and Accept are
    equal buttons."""
    pages = [p.path for p in all_site_pages()] + ["/privacy"]
    for ids in ({"analytics_ga4_id": "G-AB12CD34EF"}, {"analytics_google_ads_id": "AW-123456789"}):
        _set(tmp_data_dir, **{"analytics_ga4_id": "", "analytics_google_ads_id": "", **ids})
        for path in pages:
            body = _get(client, path).text
            assert (
                body.count('<div class="consent" role="region" aria-label="Cookie choice" hidden>')
                == 1
            ), (ids, path)
            assert (
                '<button class="btn" type="button" data-consent="denied"><span>Decline</span></button>'
                in body
            )
            assert (
                '<button class="btn" type="button" data-consent="granted"><span>Accept</span></button>'
                in body
            )
            assert body.index('<div class="consent"') > body.index("</footer>"), path
    css = _get(client, "/").text
    for rule in (
        ".consent{position:fixed;",
        ".consent[hidden]{display:none}",
        ".consent .btn{min-height:44px;",
    ):
        assert rule in css, rule


def test_csp_adds_google_only_with_an_id(client: TestClient, tmp_data_dir: Path) -> None:
    """Every page's CSP names googletagmanager.com (script, img, connect,
    frame) only when an ID is set; Ads origins only with an Ads ID; nothing
    else changes (font-src and media-src stay 'self')."""
    pages = [p.path for p in all_site_pages()]
    for path in pages:
        assert "googletagmanager" not in _get(client, path).headers["content-security-policy"], path
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    for path in pages:
        csp = dict(
            part.strip().partition(" ")[::2]
            for part in _get(client, path).headers["content-security-policy"].split(";")
        )
        for d in ("script-src", "img-src", "connect-src", "frame-src"):
            assert "https://www.googletagmanager.com" in csp[d], (path, d)
        assert "googleadservices" not in " ".join(csp.values()), path
        assert csp["font-src"] == "'self'" and csp["media-src"] == "'self'", path
    _set(tmp_data_dir, analytics_ga4_id="", analytics_google_ads_id="AW-123456789")
    csp = _get(client, "/").headers["content-security-policy"]
    assert "https://www.googleadservices.com" in csp and "google-analytics" not in csp


def test_banner_records_the_choice_and_events(client: TestClient, tmp_data_dir: Path) -> None:
    """Script contract: the choice is stored under lmw-consent, a stored
    choice hides the banner, and the three conversion events exist."""
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    scripts = "\n".join(inline_scripts(client.get("/").text))
    assert 'localStorage.setItem("lmw-consent", v)' in scripts
    assert 'stored !== "granted" && stored !== "denied"' in scripts
    for event in ("copy_install", "click_github", "click_podwarden"):
        assert f"gtag('event', '{event}')" in scripts, event


def test_banner_script_runs_after_the_banner_exists(client: TestClient, tmp_data_dir: Path) -> None:
    """The boot script looks the banner up when it runs, so it must come
    after the banner element; the consent default must still run in <head>,
    before gtag.js."""
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    body = client.get("/faq").text
    head = body[: body.index("</head>")]
    assert "gtag('consent', 'default'" in head
    assert 'id="lmw-tag"' not in head
    boot = body.index('document.querySelector(".consent")')
    assert body.index('<div class="consent"') < boot
    assert body.index('id="lmw-tag"') < boot


def test_gtag_js_loads_after_the_page_does(client: TestClient, tmp_data_dir: Path) -> None:
    """Google's script is injected after the load event, not parsed with the
    page: on a throttled phone it cost Home about 30 Lighthouse points."""
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    body = client.get("/").text
    assert not re.search(r'<script[^>]+src="https://www\.googletagmanager\.com', body)
    boot = "\n".join(inline_scripts(body))
    assert 'window.addEventListener("load"' in boot
    assert '"https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(ids[0])' in boot
