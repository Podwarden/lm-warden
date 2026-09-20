import sqlite3

import bcrypt


def _seed_admin(db_path, username="admin", pw="hunter2"):
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, h))
        db.execute("UPDATE setup_state SET step='done' WHERE id=1")
        db.commit()


def _login_access(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


class _FakeTask:
    """Duck-typed stand-in for asyncio.Task.

    StreamRegistry.cancel_user() only calls .cancel() on whatever it holds,
    so a plain object with that method is sufficient — and avoids the
    cross-loop hazard of registering a real Task on asyncio.run()'s loop
    while the sync TestClient runs the handler on an anyio worker thread.
    """

    def __init__(self) -> None:
        self.cancelled_called = False

    def cancel(self) -> bool:
        self.cancelled_called = True
        return True


def test_logout_clears_cookie_and_cancels_streams(tmp_data_dir, client):
    client.get("/healthz")  # boot lifespan, run migrations
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    access = _login_access(client)

    # Pretend a stream is registered for this user.
    fake = _FakeTask()
    # A session's streams register under session:<username> (app/auth/deps.py).
    client.app.state.stream_registry.register("session:admin", fake)
    assert client.app.state.stream_registry.count("session:admin") == 1

    r = client.post(
        "/api/auth/logout",
        headers={
            "Origin": "http://localhost:3000",
            "Authorization": f"Bearer {access}",
        },
    )

    assert r.status_code == 204
    set_cookie = r.headers["set-cookie"].lower()
    assert 'vw_refresh=""' in set_cookie or "vw_refresh=;" in set_cookie or "max-age=0" in set_cookie
    assert fake.cancelled_called
    # User-visible contract: logout empties the user's stream bucket.
    assert client.app.state.stream_registry.count("session:admin") == 0
