import re
import sqlite3
from datetime import UTC

import bcrypt


def _seed_admin(db_path, username="admin", pw="hunter2"):
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, h),
        )
        db.execute("UPDATE setup_state SET step = 'done' WHERE id = 1")
        db.commit()


def _extract_refresh_cookie(response) -> str:
    """vw_refresh is Secure=True so TestClient drops it from the cookie jar.
    Parse it out of the raw Set-Cookie header instead."""
    set_cookie = response.headers["set-cookie"]
    m = re.search(r"vw_refresh=([^;]+)", set_cookie)
    assert m, f"no vw_refresh in set-cookie: {set_cookie!r}"
    return m.group(1)


ORIGIN_HEADER = {"Origin": "http://localhost:3000"}


def test_refresh_success(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "hunter2"},
        headers=ORIGIN_HEADER,
    )
    assert login.status_code == 200, login.text
    refresh_jwt = _extract_refresh_cookie(login)
    r = client.post(
        "/api/auth/refresh",
        headers=ORIGIN_HEADER,
        cookies={"vw_refresh": refresh_jwt},
    )
    assert r.status_code == 200, r.text
    assert "access_token" in r.json()
    assert r.json()["expires_in"] == 15 * 60


def test_refresh_no_origin_rejected(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "hunter2"},
        headers=ORIGIN_HEADER,
    )
    refresh_jwt = _extract_refresh_cookie(login)
    r = client.post(
        "/api/auth/refresh",
        cookies={"vw_refresh": refresh_jwt},
    )
    assert r.status_code == 403


def test_refresh_wrong_origin_rejected(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "hunter2"},
        headers=ORIGIN_HEADER,
    )
    refresh_jwt = _extract_refresh_cookie(login)
    r = client.post(
        "/api/auth/refresh",
        headers={"Origin": "https://evil.example.com"},
        cookies={"vw_refresh": refresh_jwt},
    )
    assert r.status_code == 403


def test_refresh_no_cookie_401(tmp_data_dir, client):
    client.get("/healthz")
    r = client.post("/api/auth/refresh", headers=ORIGIN_HEADER)
    assert r.status_code == 401


def test_refresh_invalid_token_401(tmp_data_dir, client):
    client.get("/healthz")
    r = client.post(
        "/api/auth/refresh",
        headers=ORIGIN_HEADER,
        cookies={"vw_refresh": "not-a-jwt"},
    )
    assert r.status_code == 401


def test_refresh_rejects_access_token(tmp_data_dir, client):
    """A token of typ=access must NOT be accepted on /refresh."""
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "hunter2"},
        headers=ORIGIN_HEADER,
    )
    access_jwt = login.json()["access_token"]
    r = client.post(
        "/api/auth/refresh",
        headers=ORIGIN_HEADER,
        cookies={"vw_refresh": access_jwt},
    )
    assert r.status_code == 401


def test_refresh_missing_sub_claim_401(tmp_data_dir, client):
    """A validly-signed refresh JWT without a sub claim must yield 401, not 500."""
    from datetime import datetime, timedelta

    import jwt as pyjwt

    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")

    # Mint a refresh-typed JWT without `sub`.
    secret = client.app.state.jwt_secret
    now = datetime.now(UTC)
    token = pyjwt.encode(
        {
            "typ": "refresh",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=7)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )

    r = client.post(
        "/api/auth/refresh",
        cookies={"vw_refresh": token},
        headers=ORIGIN_HEADER,
    )
    assert r.status_code == 401, r.text
