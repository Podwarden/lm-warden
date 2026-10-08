import sqlite3

import bcrypt


def _seed_admin(db_path, username="admin", pw="hunter2"):
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, h))
        db.execute("UPDATE setup_state SET step = 'done' WHERE id = 1")
        db.commit()


def test_login_success_sets_cookie_returns_access(tmp_data_dir, client):
    client.get("/healthz")  # boot, run migrations
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "hunter2"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "access_token" in body
    assert body["expires_in"] == 15 * 60
    set_cookie = r.headers["set-cookie"]
    assert "vw_refresh=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=Strict" in set_cookie or "samesite=strict" in set_cookie.lower()
    assert "Path=/api/auth" in set_cookie or "path=/api/auth" in set_cookie.lower()


def test_login_wrong_password_401(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong"},
    )
    assert r.status_code == 401


def test_login_unknown_user_401(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/auth/login",
        json={"username": "ghost", "password": "hunter2"},
    )
    assert r.status_code == 401


def test_login_unknown_user_still_runs_bcrypt(tmp_data_dir, client, monkeypatch):
    import app.auth.routes as routes_mod

    call_count = {"n": 0}
    real_checkpw = routes_mod.bcrypt.checkpw

    def counting_checkpw(*args, **kwargs):
        call_count["n"] += 1
        return real_checkpw(*args, **kwargs)

    monkeypatch.setattr(routes_mod.bcrypt, "checkpw", counting_checkpw)

    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/auth/login", json={"username": "ghost", "password": "hunter2"})
    assert r.status_code == 401
    assert call_count["n"] >= 1, "bcrypt.checkpw must run even when user is unknown"
