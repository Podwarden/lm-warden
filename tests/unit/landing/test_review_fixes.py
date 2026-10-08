"""Findings from the whole-branch review (2026-10-07), each pinned."""

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from tests.unit.landing.test_routes import CANON, _canonical, _disable


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


def test_off_redirect_is_temporary_and_uncached(client: TestClient, tmp_data_dir: Path) -> None:
    """The root's redirect follows a runtime setting: a browser must not keep
    it once the website is switched on."""
    _disable(tmp_data_dir)
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"
    assert "no-store" in r.headers["cache-control"]


def test_unknown_client_gets_the_site_404(client: TestClient) -> None:
    r = client.get("/coding-agents/codex", headers={"accept": "text/html"})
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    assert 'href="/coding-agents"' in r.text


def test_privacy_without_an_id_gets_the_site_404(client: TestClient) -> None:
    r = client.get("/privacy", headers={"accept": "text/html"})
    assert r.status_code == 404 and "text/html" in r.headers["content-type"]


def test_client_trailing_slash_is_a_relative_308(client: TestClient) -> None:
    r = client.get("/coding-agents/codex-cli/", follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"] == "/coding-agents/codex-cli"


def test_privacy_page_is_neutral_for_other_operators(
    client: TestClient, tmp_data_dir: Path
) -> None:
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF")
    body = client.get("/privacy").text
    assert "info@podwarden.com" not in body
    assert "Google Ads" not in body  # no Ads ID set
    assert "operator of this site" in body


def test_privacy_page_names_podwarden_on_lmwarden_com(
    client: TestClient, tmp_data_dir: Path
) -> None:
    _canonical(client, "https://lmwarden.com")
    _set(tmp_data_dir, analytics_ga4_id="G-AB12CD34EF", analytics_google_ads_id="AW-123456789")
    body = client.get("/privacy").text
    assert "info@podwarden.com" in body
    assert "Google Ads" in body


def test_canon_constant_is_lmwarden() -> None:
    assert CANON == "https://lmwarden.com"
