"""God-mode media endpoint — GET /api/admin/godmode/media/{media_id}.

* Unauthenticated -> 401.
* Disabled -> 409 (same disabled shape as the stream).
* Valid id -> decoded bytes, stored Content-Type, private cache header.
* Unknown/evicted/malformed id -> 404.
"""

import base64
import dataclasses

from tests.conftest import jwt_login, seed_admin_user

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)  # 1x1 png
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def _enable_godmode(client):
    client.app.state.settings = dataclasses.replace(client.app.state.settings, godmode_enabled=True)


def _auth(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return jwt_login(client)


def test_media_rejects_unauthenticated(tmp_data_dir, client):
    _auth(tmp_data_dir, client)
    _enable_godmode(client)
    r = client.get("/api/admin/godmode/media/" + "a" * 16)
    assert r.status_code == 401


def test_media_disabled_returns_409(tmp_data_dir, client):
    auth = _auth(tmp_data_dir, client)
    r = client.get("/api/admin/godmode/media/" + "a" * 16, headers=auth)
    assert r.status_code == 409
    assert "disabled" in r.text.lower()


def test_media_serves_stored_image(tmp_data_dir, client):
    auth = _auth(tmp_data_dir, client)
    _enable_godmode(client)
    client.app.state.godmode_media.put("a" * 16, "image/png", PNG_B64)
    r = client.get("/api/admin/godmode/media/" + "a" * 16, headers=auth)
    assert r.status_code == 200
    assert r.content == PNG_BYTES
    assert r.headers["content-type"].startswith("image/png")
    assert r.headers["cache-control"] == "private, max-age=3600"


def test_media_unknown_id_404(tmp_data_dir, client):
    auth = _auth(tmp_data_dir, client)
    _enable_godmode(client)
    r = client.get("/api/admin/godmode/media/" + "b" * 16, headers=auth)
    assert r.status_code == 404


def test_media_malformed_id_404(tmp_data_dir, client):
    auth = _auth(tmp_data_dir, client)
    _enable_godmode(client)
    for bad in ("short", "Z" * 16, "a" * 17, "../../etc/passwd"):
        r = client.get(f"/api/admin/godmode/media/{bad}", headers=auth)
        assert r.status_code == 404, bad


def test_media_undecodable_payload_404(tmp_data_dir, client):
    auth = _auth(tmp_data_dir, client)
    _enable_godmode(client)
    # "YWJj!!!!" is valid length (8 chars) but contains non-alphabet chars.
    # Without validate=True, it silently discards !!!!, decodes to b'abc'.
    # With validate=True (in the endpoint), it raises binascii.Error -> 404.
    # This test pins that we use validate=True and reject crafted payloads.
    client.app.state.godmode_media.put("c" * 16, "image/png", "YWJj!!!!")
    r = client.get("/api/admin/godmode/media/" + "c" * 16, headers=auth)
    assert r.status_code == 404
