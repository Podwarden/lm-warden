"""PATCH /api/settings/runtime admin_password: the new-password rule
(app/auth/passwords.py) -- at least 12 characters, at most 72 bytes -- and a
password stored under the old 6-character minimum still signs in."""

import sqlite3

import pytest

from tests.conftest import csrf_header, jwt_login

URL = "/api/settings/runtime"


def _hash(db_path) -> str:
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT password_hash FROM users").fetchone()[0]


def test_a_stored_short_password_still_signs_in(client, seeded_db):
    # seed_admin_user stores "hunter2" (7 chars), legal under the old rule.
    assert "Authorization" in jwt_login(client, password="hunter2")


@pytest.mark.parametrize(
    "password,detail",
    [
        ("a" * 11, "admin_password: password must be at least 12 characters"),
        ("x" * 73, "admin_password: password must be at most 72 bytes"),
    ],
)
def test_a_new_password_outside_the_bounds_is_refused(client, seeded_db, password, detail):
    session = {**jwt_login(client), **csrf_header(client)}
    before = _hash(seeded_db)
    r = client.patch(URL, json={"admin_password": password}, headers=session)
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == detail
    assert _hash(seeded_db) == before


def test_a_twelve_character_password_is_accepted(client, seeded_db):
    session = {**jwt_login(client), **csrf_header(client)}
    r = client.patch(URL, json={"admin_password": "twelve-chars"}, headers=session)
    assert r.status_code == 200, r.text
    assert "Authorization" in jwt_login(client, password="twelve-chars")
