"""The key holder's own area: /api/me/* acts only on the signed-in key."""

from __future__ import annotations

import sqlite3

from tests.unit.stats.test_forest_routes import _forest_login, _ready_key


def test_key_holder_reads_only_their_own_key(tmp_data_dir, client):
    _db, plaintext, _admin = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    d = client.get("/api/me/key", headers=k).json()
    assert d["id"] == "tok1" and "plaintext" not in d
    for path in ("/api/me/usage?range=24h", "/api/me/cache?range=24h", "/api/me/guard/events"):
        assert client.get(path, headers=k).status_code == 200, path


def test_admin_and_anonymous_are_refused(tmp_data_dir, client):
    _db, _plain, admin = _ready_key(client, tmp_data_dir)
    assert client.get("/api/me/key", headers=admin).status_code == 403
    assert client.get("/api/me/key").status_code == 401


def test_rename_but_nothing_else(tmp_data_dir, client):
    _db, plaintext, _ = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    r = client.patch("/api/me/key", json={"name": "my-laptop"}, headers=k)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "my-laptop"
    assert client.patch("/api/me/key", json={"priority": 9}, headers=k).status_code == 422
    assert (
        client.patch(
            "/api/me/key", json={"name": "x", "guard": {"off": True}}, headers=k
        ).status_code
        == 422
    )


def test_rotate_returns_the_new_key_once_and_a_session_for_it(tmp_data_dir, client):
    _db, plaintext, _ = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    r = client.post("/api/me/key/rotate", json={"grace_hours": 1}, headers=k)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["plaintext"].startswith("vw_") and body["token"]
    assert client.get("/api/me/key", headers=k).status_code == 401  # the old sign-in ended
    k2 = {"Authorization": f"Bearer {body['token']}"}
    assert client.get("/api/me/key", headers=k2).json()["id"] == body["id"]
    assert (
        client.post("/api/me/key/rotate", json={"grace_hours": 48}, headers=k2).status_code == 422
    )


def test_revoke_needs_the_name_and_ends_everything(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    name = client.get("/api/me/key", headers=k).json()["name"]
    assert (
        client.post("/api/me/key/revoke", json={"confirm_name": "wrong"}, headers=k).status_code
        == 409
    )
    r = client.post("/api/me/key/revoke", json={"confirm_name": name}, headers=k)
    assert r.status_code == 204, r.text
    assert client.get("/api/me/key", headers=k).status_code == 401
    assert (
        client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"}).status_code
        == 401
    )
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM api_tokens WHERE id = 'tok1'").fetchone()[0] == 0
