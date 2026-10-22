"""Coding-agent pages generated from the Connect catalogue (spec 2026-10-07 §3)."""

import html
import re

from fastapi.testclient import TestClient

from app.connect.catalog import CLIENTS
from app.landing.clients import CLIENT_PAGES, capability

# The site prints catalogue prose through prose_html, which also puts its
# code-like words in <code> (so a translation must keep them verbatim); it
# replaced plain _e here when the coding-agent pages were localized.
from app.landing.clients import prose_html as prose


def test_one_page_per_catalogue_client(client: TestClient) -> None:
    assert set(CLIENT_PAGES) == {c.id for c in CLIENTS}
    for c in CLIENTS:
        r = client.get(f"/coding-agents/{c.id}")
        assert r.status_code == 200, c.id
        body = r.text
        assert len(re.findall(r"<h1[\s>]", body)) == 1, c.id
        assert prose(c.summary) in body, c.id
        t = html.unescape(re.search(r"<title>(.*?)</title>", body).group(1))
        assert len(t) < 60, t


def test_unknown_client_is_404(client: TestClient) -> None:
    assert client.get("/coding-agents/not-a-client").status_code == 404


def test_snippets_hold_placeholders_never_a_key(client: TestClient) -> None:
    for c in CLIENTS:
        body = client.get(f"/coding-agents/{c.id}").text
        assert "{{" not in body, c.id
        assert not re.search(r"\bvw_(?!YOUR_KEY)[A-Za-z0-9]{6,}", body), c.id


def test_no_tested_versus_documented_status(client: TestClient) -> None:
    """The website does not set the clients we ran apart from the ones
    configured from their docs (owner, 2026-10-21; it replaced a guard that
    pinned each client's verification date on its page). The catalogue keeps
    the record for the console's Connect page; no page prints it, and the
    one capability sentence that stated it is dropped from the site copy."""
    pages = {f"/coding-agents/{c.id}" for c in CLIENTS} | {"/coding-agents", "/"}
    for path in sorted(pages):
        body = client.get("/_landing" if path == "/" else path).text
        for wording in (
            "Verified",
            "Run live",
            "docs only",
            "Configured from its docs",
            "Checked against its docs",
            "Run against a live warden",
            "Not yet run",
            "was not run",
            'class="badge',
            "<b>documented</b>",
        ):
            assert wording not in body, (path, wording)
    for c in CLIENTS:
        body = client.get(f"/coding-agents/{c.id}").text
        assert prose(capability(c)) in body, c.id


def test_hub_lists_and_links_every_client(client: TestClient) -> None:
    body = client.get("/coding-agents").text
    for c in CLIENTS:
        assert f'href="/coding-agents/{c.id}"' in body, c.id


def test_claude_code_page_carries_the_router_section(client: TestClient) -> None:
    body = client.get("/coding-agents/claude-code").text
    assert 'aria-labelledby="router-title"' in body
    assert body.count("<h1") == 1


def test_client_pages_are_disabled_with_the_site(client: TestClient, tmp_data_dir) -> None:
    from tests.unit.landing.test_routes import _disable

    _disable(tmp_data_dir)
    assert client.get("/coding-agents/codex-cli").status_code == 404
