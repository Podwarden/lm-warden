"""Per-page search metadata and PodWarden cross-links (spec 2026-10-07 §6)."""

import json
import re

from fastapi.testclient import TestClient

from app.landing.render import all_site_pages
from tests.unit.landing.conftest import fetch
from tests.unit.landing.test_routes import CANON, _canonical


def _url(path: str) -> str:
    return CANON + ("/" if path == "/" else path)


def _ld(body: str) -> dict[str, dict]:
    blocks = re.findall(r'<script type="application/ld\+json">\s*(.*?)\s*</script>', body, re.S)
    return {d["@type"]: d for d in map(json.loads, blocks)}


def test_each_page_has_its_own_canonical_and_og_url(client: TestClient) -> None:
    _canonical(client, CANON)
    for p in all_site_pages():
        body = fetch(client, p.path)
        assert f'<link rel="canonical" href="{_url(p.path)}">' in body, p.path
        assert f'<meta property="og:url" content="{_url(p.path)}">' in body, p.path


def test_private_instance_pages_are_noindex(client: TestClient) -> None:
    for p in all_site_pages():
        r = client.get("/_landing" if p.path == "/" else p.path)
        assert '<meta name="robots" content="noindex">' in r.text, p.path
        assert r.headers["x-robots-tag"] == "noindex", p.path


def test_structured_data_per_page_kind(client: TestClient) -> None:
    _canonical(client, CANON)
    home = _ld(fetch(client, "/"))
    assert {"SoftwareApplication", "WebSite"} <= set(home)
    pub = home["SoftwareApplication"]["publisher"]
    assert pub["name"] == "PodWarden" and pub["url"] == "https://podwarden.com"
    assert "FAQPage" in _ld(fetch(client, "/faq"))
    assert "HowTo" in _ld(fetch(client, "/agent-tuning"))
    codex = _ld(fetch(client, "/coding-agents/codex-cli"))
    assert "TechArticle" in codex
    crumbs = codex["BreadcrumbList"]["itemListElement"]
    assert [c["name"] for c in crumbs] == ["Home", "Coding agents", "Codex CLI"]
    assert crumbs[-1]["item"] == CANON + "/coding-agents/codex-cli"


def test_private_breadcrumbs_name_no_urls(client: TestClient) -> None:
    crumbs = _ld(fetch(client, "/faq"))["BreadcrumbList"]["itemListElement"]
    assert all("item" not in c for c in crumbs)


def test_sitemap_lists_every_page(client: TestClient) -> None:
    _canonical(client, CANON)
    body = client.get("/sitemap.xml").text
    locs = re.findall(r"<loc>(.*?)</loc>", body)
    assert sorted(locs) == sorted(_url(p.path) for p in all_site_pages())
    assert "/privacy" not in body  # tag-only page is not in the sitemap
    assert body.count("<lastmod>") == len(locs)


def test_llms_txt_links_every_page_and_names_podwarden(client: TestClient) -> None:
    _canonical(client, CANON)
    body = client.get("/llms.txt").text
    for p in all_site_pages():
        assert _url(p.path) in body, p.path
    assert "https://podwarden.com" in body and "https://github.com/Podwarden" in body
    assert "https://github.com/Podwarden" in client.get("/llms-full.txt").text


def test_podwarden_strip_and_footer(client: TestClient) -> None:
    assert 'aria-labelledby="family-title"' in fetch(client, "/")
    for p in all_site_pages():
        body = fetch(client, p.path)
        assert "LM Warden is a PodWarden project" in body, p.path
        assert 'href="https://github.com/Podwarden"' in body, p.path
