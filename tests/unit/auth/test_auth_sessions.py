"""Sign-in sessions end when they should (verified finding #2): logout, a password change,
a key holder's sign-out, and rotating or revoking the key. Tokens without a session are refused."""

from __future__ import annotations

from app.auth.jwt import mint_access
from tests.conftest import csrf_header, jwt_login, seed_admin_user
from tests.unit.stats.test_forest_routes import _forest_login, _ready_key

ORIGIN = {"Origin": "http://localhost:3000"}


def _login(client) -> dict[str, str]:
    return jwt_login(client)


def test_logout_ends_the_session_and_its_refresh_cookie(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = _login(client)
    assert client.get("/api/tokens", headers=h).status_code == 200
    assert client.post("/api/auth/refresh", headers=ORIGIN).status_code == 200
    assert client.post("/api/auth/logout", headers={**h, **ORIGIN}).status_code == 204
    assert client.get("/api/tokens", headers=h).status_code == 401
    # the cookie is gone from this client; replaying the old one must fail too


def test_refresh_cookie_replayed_after_logout_is_refused(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = _login(client)
    cookie = client.cookies.get("vw_refresh")
    client.post("/api/auth/logout", headers={**h, **ORIGIN})
    client.cookies.set("vw_refresh", cookie)
    assert client.post("/api/auth/refresh", headers=ORIGIN).status_code == 401


def test_password_change_ends_other_sessions(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    other = _login(client)
    other_cookie = client.cookies.get("vw_refresh")
    mine = _login(client)
    r = client.patch(
        "/api/settings/runtime",
        json={"admin_password": "n3w-Passw0rd-long"},
        headers={**mine, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    assert client.get("/api/tokens", headers=other).status_code == 401
    client.cookies.set("vw_refresh", other_cookie)
    assert client.post("/api/auth/refresh", headers=ORIGIN).status_code == 401


def test_a_token_without_a_session_is_refused(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    legacy = mint_access("admin", client.app.state.jwt_secret, 15)
    assert (
        client.get("/api/tokens", headers={"Authorization": f"Bearer {legacy}"}).status_code == 401
    )


def test_key_holder_sign_out(tmp_data_dir, client):
    _db, plaintext, _ = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    assert client.get("/api/sessions?range=1h", headers=k).status_code == 200
    assert client.post("/api/forest/logout", headers=k).status_code == 204
    assert client.get("/api/sessions?range=1h", headers=k).status_code == 401


def test_rotating_or_revoking_a_key_ends_its_holders_sessions(tmp_data_dir, client):
    _db, plaintext, admin = _ready_key(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    r = client.post(
        "/api/tokens/tok1/rotate",
        json={"grace_hours": 24},
        headers={**admin, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    assert client.get("/api/sessions?range=1h", headers=k).status_code == 401
    new_plain = r.json()["plaintext"]
    k2 = _forest_login(client, new_plain)
    assert client.get("/api/sessions?range=1h", headers=k2).status_code == 200
    new_id = r.json()["id"]
    assert (
        client.delete(f"/api/tokens/{new_id}", headers={**admin, **csrf_header(client)}).status_code
        == 204
    )
    assert client.get("/api/sessions?range=1h", headers=k2).status_code == 401
