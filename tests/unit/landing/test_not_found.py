"""The website's own 404 page for unknown paths (spec 2026-10-07 §4)."""

from pathlib import Path

from fastapi.testclient import TestClient


def test_unknown_site_path_gets_the_site_404(client: TestClient) -> None:
    r = client.get("/no-such-page", headers={"accept": "text/html"})
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    assert '<meta name="robots" content="noindex">' in r.text
    assert r.headers["x-robots-tag"] == "noindex"
    assert 'href="/coding-agents"' in r.text


def test_api_paths_keep_json_404(client: TestClient) -> None:
    for path in ("/api/no-such-thing", "/v1/no-such-thing"):
        r = client.get(path, headers={"accept": "text/html"})
        assert r.status_code in (401, 404), path
        assert "text/html" not in r.headers.get("content-type", ""), path


def test_post_to_unknown_path_is_not_html(client: TestClient) -> None:
    r = client.post("/no-such-page")
    assert "text/html" not in r.headers.get("content-type", "")


def test_disabled_site_keeps_the_plain_404(client: TestClient, tmp_data_dir: Path) -> None:
    from tests.unit.landing.test_routes import _disable

    _disable(tmp_data_dir)
    r = client.get("/no-such-page", headers={"accept": "text/html"})
    assert r.status_code == 404 and "text/html" not in r.headers.get("content-type", "")
