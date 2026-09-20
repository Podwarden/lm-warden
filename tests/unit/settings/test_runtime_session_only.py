"""PATCH /api/settings/runtime: the operator's credentials and the session
lifetimes are session-only (spec 2026-09-19, decision 3). An admin token that
could change the password would lock the owner out of the session-only revoke
that ends it; it may still change every other runtime setting."""

import sqlite3

import pytest

from app.settings.routes_api import SESSION_ONLY_RUNTIME_KEYS
from tests.conftest import bearer, csrf_header, jwt_login, seed_admin_token

URL = "/api/settings/runtime"
CHANGES = {
    "admin_username": "intruder",
    "admin_password": "new-password",
    "session_access_ttl_minutes": 9999,
    "session_refresh_ttl_days": 9999,
    "sse_ticket_ttl_seconds": 9999,
    # I1 (final-review 2026-09-19): an admin token that could steer
    # `public_url` could point the next secret's curl example at a host it
    # controls. https://proxy.example is a documented placeholder, not a
    # real attacker-controlled host.
    "public_url": "https://proxy.example",
}


def _users(db_path) -> list[tuple[str, str]]:
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT username, password_hash FROM users").fetchall()


def _kv(db_path) -> dict[str, str]:
    with sqlite3.connect(db_path) as db:
        return dict(db.execute("SELECT key, value FROM settings").fetchall())


def test_every_session_only_key_is_exercised():
    assert set(CHANGES) == SESSION_ONLY_RUNTIME_KEYS


@pytest.mark.parametrize("key", sorted(CHANGES))
def test_an_admin_token_cannot_change_a_session_only_key(client, seeded_db, key):
    _, secret = seed_admin_token(seeded_db)
    users, kv = _users(seeded_db), _kv(seeded_db)
    # Mixed with an ordinary key: nothing at all may be applied.
    r = client.patch(URL, json={key: CHANGES[key], "log_retention_lines": 777},
                     headers=bearer(secret))
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["error_code"] == "session_only"
    assert r.json()["detail"]["keys"] == [key]
    assert (_users(seeded_db), _kv(seeded_db)) == (users, kv)
    # The session and its password still work.
    assert "Authorization" in jwt_login(client)


def test_an_admin_token_changes_ordinary_runtime_settings(client, seeded_db):
    _, secret = seed_admin_token(seeded_db)
    r = client.patch(URL, json={"log_retention_lines": 777}, headers=bearer(secret))
    assert r.status_code == 200, r.text
    assert _kv(seeded_db)["log_retention_lines"] == "777"


def test_a_session_still_changes_the_password(client, seeded_db):
    session = {**jwt_login(client), **csrf_header(client)}
    r = client.patch(URL, json={"admin_password": "new-password"}, headers=session)
    assert r.status_code == 200, r.text
    assert "Authorization" in jwt_login(client, password="new-password")
