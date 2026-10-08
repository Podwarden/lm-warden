"""Landing tests exercise the website, which new installs ship switched off
(spec 2026-10-07 §9). Every test that builds a client starts with it on; a
test that wants it off calls _disable / _delete_landing_setting from
test_routes as before."""

import sqlite3
from pathlib import Path

import pytest


def enable_site(tmp_data_dir: Path) -> None:
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db", isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO settings(key, value) VALUES ('landing_page_enabled', 'true') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


@pytest.fixture(autouse=True)
def _site_on(request: pytest.FixtureRequest) -> None:
    if "client" in request.fixturenames:
        request.getfixturevalue("client")
        enable_site(request.getfixturevalue("tmp_data_dir"))


#: The static website pages, in the order the pre-split single page read
#: (home, router, GPU, cache, keys, features, FAQ). `site_html` joins them so
#: tests about the page as one document (figure numbering, every reading
#: naming its source) still see one document.
SITE_ORDER = (
    "/",
    "/coding-agents/claude-code",
    "/gpu-monitoring",
    "/cache-aware-routing",
    "/api-keys",
    "/features",
    "/faq",
)


def fetch(client, path: str) -> str:
    r = client.get("/_landing" if path == "/" else path)
    assert r.status_code == 200, (path, r.status_code)
    return r.text


def all_pages(client) -> dict[str, str]:
    """path -> HTML for every page in the registry a visitor can reach
    without tracking configured (tag-only pages excluded)."""
    from app.landing.render import all_site_pages

    return {p.path: fetch(client, p.path) for p in all_site_pages()}


def site_html(client) -> str:
    return "\n".join(fetch(client, p) for p in SITE_ORDER)
