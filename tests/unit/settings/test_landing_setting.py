"""GET/PATCH round-trip for the `landing_page_enabled` runtime setting (#155).

Pins that the new runtime key is wired through the same surface as
every other RUNTIME_KEYS entry: GET returns the seeded value, PATCH
coerces a JSON bool to the canonical lowercase string, and a follow-up
GET observes the new value.

`landing_page_enabled` is classified as `"none"` (takes effect on the
next request, no restart), so `requires_restart_kinds` MUST come back
empty on the PATCH response — the FE banner stays silent for this key.
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import csrf_header, jwt_login, seed_admin_user


@pytest.fixture
def auth_headers(client: TestClient, tmp_data_dir) -> dict[str, str]:
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    headers = jwt_login(client)
    headers.update(csrf_header(client))
    return headers


def test_landing_setting_get_returns_seeded_default(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Migration 0020 seeds `landing_page_enabled='false'` on a new install."""
    r = client.get("/api/settings/runtime", headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "landing_page_enabled" in body
    assert body["landing_page_enabled"] == "false"


def test_landing_setting_patch_round_trip_disable(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """PATCH with a JSON bool persists as the canonical 'false' string and
    the response classifies the key as no-restart (kinds list empty).
    """
    r = client.patch(
        "/api/settings/runtime",
        headers=auth_headers,
        json={"landing_page_enabled": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "ok": True,
        "requires_restart_kinds": [],
        "requires_restart": [],
    }

    # Round-trip — GET must observe the new value.
    r = client.get("/api/settings/runtime", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["landing_page_enabled"] == "false"


def test_landing_setting_patch_rejects_non_bool(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Free-form strings ("maybe", "1.5", etc.) must 422 — coercer
    accepts only bool-shaped values (real bools + canonical truthy/falsy
    strings) so we don't end up with junk values on disk that
    `_is_enabled()` then has to guess at.
    """
    r = client.patch(
        "/api/settings/runtime",
        headers=auth_headers,
        json={"landing_page_enabled": "maybe"},
    )
    assert r.status_code == 422, r.text
