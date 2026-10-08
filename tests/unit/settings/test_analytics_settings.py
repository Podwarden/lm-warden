"""Website tracking IDs: validated, clearable, session-only (spec 2026-10-07 §8)."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import csrf_header, jwt_login, seed_admin_user

KEYS = ("analytics_ga4_id", "analytics_google_ads_id", "search_console_verification")
GOOD = {
    "analytics_ga4_id": "G-AB12CD34EF",
    "analytics_google_ads_id": "AW-123456789",
    "search_console_verification": "abcDEF123_-abcDEF123_-xyz",
}


@pytest.fixture
def auth_headers(client: TestClient, tmp_data_dir) -> dict[str, str]:
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    headers = jwt_login(client)
    headers.update(csrf_header(client))
    return headers


def test_tracking_keys_are_blank_by_default(client, auth_headers) -> None:
    body = client.get("/api/settings/runtime", headers=auth_headers).json()
    for k in KEYS:
        assert body[k] in (None, "")


def test_valid_ids_round_trip_without_restart(client, auth_headers) -> None:
    r = client.patch("/api/settings/runtime", headers=auth_headers, json=GOOD)
    assert r.status_code == 200, r.text
    assert r.json()["requires_restart_kinds"] == []
    body = client.get("/api/settings/runtime", headers=auth_headers).json()
    for k, v in GOOD.items():
        assert body[k] == v


def test_ids_are_trimmed_and_upper_cased_where_google_does(client, auth_headers) -> None:
    r = client.patch(
        "/api/settings/runtime",
        headers=auth_headers,
        json={"analytics_ga4_id": "  g-ab12cd34ef ", "analytics_google_ads_id": "aw-123456789"},
    )
    assert r.status_code == 200, r.text
    body = client.get("/api/settings/runtime", headers=auth_headers).json()
    assert body["analytics_ga4_id"] == "G-AB12CD34EF"
    assert body["analytics_google_ads_id"] == "AW-123456789"


def test_empty_string_clears(client, auth_headers) -> None:
    client.patch("/api/settings/runtime", headers=auth_headers, json=GOOD)
    r = client.patch("/api/settings/runtime", headers=auth_headers, json={k: "" for k in KEYS})
    assert r.status_code == 200, r.text
    body = client.get("/api/settings/runtime", headers=auth_headers).json()
    for k in KEYS:
        assert body[k] == ""


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("analytics_ga4_id", 'G-1"><script>alert(1)</script>'),
        ("analytics_ga4_id", "UA-12345-1"),
        ("analytics_ga4_id", "G-" + "A" * 40),
        ("analytics_ga4_id", 123),
        ("analytics_google_ads_id", "AW-12ab"),
        ("analytics_google_ads_id", "G-AB12CD34EF"),
        ("search_console_verification", "short"),
        ("search_console_verification", '<meta name="x">'),
    ],
)
def test_bad_ids_are_refused(client, auth_headers, key, bad) -> None:
    r = client.patch("/api/settings/runtime", headers=auth_headers, json={key: bad})
    assert r.status_code == 422, r.text
    assert key in r.json()["detail"]


def test_admin_token_cannot_set_tracking(client, auth_headers) -> None:
    """A leaked vwa_ token must not be able to point the public site's
    analytics at a property its holder controls."""
    issued = client.post(
        "/api/admin-tokens",
        headers=auth_headers,
        json={"name": "t", "expires_in_days": 30},
    ).json()["plaintext"]
    r = client.patch(
        "/api/settings/runtime",
        headers={"Authorization": f"Bearer {issued}"},
        json={"analytics_ga4_id": "G-AB12CD34EF"},
    )
    assert r.status_code == 403, r.text
