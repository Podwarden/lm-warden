"""GET /api/system/update-status and its dismiss action."""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.updates import client as uc
from tests.conftest import csrf_header, jwt_login, seed_admin_token, seed_admin_user


def _put(db_path: Path, key: str, value: str) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


RESULT = {
    "latest": {"version": "v2026.12.01.1", "released_at": "2026-12-01", "url": "https://x/n"},
    "advisories": [
        {
            "id": "LMW-A",
            "severity": "high",
            "title": "applies",
            "fixed_in": "2026.12.01.1",
            "url": "https://x/a",
        },
        {
            "id": "LMW-B",
            "severity": "low",
            "title": "old",
            "fixed_in": "2026.01.01.1",
            "url": "https://x/b",
        },
    ],
}


@pytest.fixture
def authed(client: TestClient, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.updates.routes_api.current_version", lambda: "v2026.11.20.1")
    # The suite turns the check off (tests/conftest.py); these tests are about it on.
    monkeypatch.setenv("VW_UPDATE_CHECK", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return {**jwt_login(client), **csrf_header(client)}


def test_status_needs_a_login(client: TestClient) -> None:
    assert client.get("/api/system/update-status").status_code == 401


def test_status_before_any_check(client: TestClient, authed: dict) -> None:
    body = client.get("/api/system/update-status", headers=authed).json()
    assert body["current"] == "v2026.11.20.1"
    assert body["latest"] is None
    assert body["update_available"] is False
    assert body["advisories"] == []
    assert body["last_checked_at"] is None
    assert body["enabled"] in (True, False)


def test_status_lists_only_applying_advisories(
    client: TestClient, authed: dict, tmp_data_dir: Path
) -> None:
    db = tmp_data_dir / "vllm-warden.db"
    _put(db, uc.RESULT_KEY, json.dumps(RESULT))
    _put(db, uc.CHECKED_KEY, "2026-11-20T10:00:00Z")
    body = client.get("/api/system/update-status", headers=authed).json()
    assert body["update_available"] is True
    assert body["latest"]["version"] == "v2026.12.01.1"
    assert [a["id"] for a in body["advisories"]] == ["LMW-A"]
    assert body["last_checked_at"] == "2026-11-20T10:00:00Z"


def test_dismiss_hides_the_release_but_not_the_advisory(
    client: TestClient, authed: dict, tmp_data_dir: Path
) -> None:
    _put(tmp_data_dir / "vllm-warden.db", uc.RESULT_KEY, json.dumps(RESULT))
    r = client.post(
        "/api/system/update-status/dismiss", headers=authed, json={"version": "v2026.12.01.1"}
    )
    assert r.status_code == 200, r.text
    body = client.get("/api/system/update-status", headers=authed).json()
    assert body["update_available"] is False
    assert body["dismissed"] is True
    assert [a["id"] for a in body["advisories"]] == ["LMW-A"]


def test_dismiss_rejects_junk(client: TestClient, authed: dict) -> None:
    r = client.post("/api/system/update-status/dismiss", headers=authed, json={"version": "x"})
    assert r.status_code == 422


def test_admin_token_can_read_status(client: TestClient, tmp_data_dir: Path) -> None:
    _, secret = seed_admin_token(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/system/update-status", headers={"Authorization": f"Bearer {secret}"})
    assert r.status_code == 200


def test_runtime_setting_round_trips(client: TestClient, authed: dict) -> None:
    r = client.patch("/api/settings/runtime", headers=authed, json={"update_check_enabled": False})
    assert r.status_code == 200, r.text
    assert (
        client.get("/api/settings/runtime", headers=authed).json()["update_check_enabled"]
        == "false"
    )


def test_a_malformed_advisory_is_normalised(
    client: TestClient, authed: dict, tmp_data_dir: Path
) -> None:
    bad = {
        "latest": None,
        "advisories": [
            {
                "id": 7,
                "severity": {"cvss": 9.8},
                "title": {"en": "x"},
                "fixed_in": "2026.12.01.1",
                "url": "javascript:alert(1)",
            },
            {"id": "LMW-OK", "fixed_in": "2026.12.01.1", "url": "https://example.test/a"},
            "not-an-object",
        ],
    }
    _put(tmp_data_dir / "vllm-warden.db", uc.RESULT_KEY, json.dumps(bad))
    advs = client.get("/api/system/update-status", headers=authed).json()["advisories"]
    for a in advs:
        assert set(a) == {"id", "severity", "title", "fixed_in", "url"}
        assert all(v is None or isinstance(v, str) for v in a.values())
    assert advs[0]["url"] is None
    assert advs[1]["url"] == "https://example.test/a"


def test_turning_the_check_off_hides_the_release_banner(
    client: TestClient, authed: dict, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VW_UPDATE_CHECK", "0")
    _put(tmp_data_dir / "vllm-warden.db", uc.RESULT_KEY, json.dumps(RESULT))
    body = client.get("/api/system/update-status", headers=authed).json()
    assert body["enabled"] is False
    assert body["update_available"] is False
