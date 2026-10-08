"""End-to-end test: token rotation grace window honors `revoked_at` timing.

Spec semantics (docs/superpowers/specs/2026-05-11-vllm-warden-ui-redesign-design.md):

  * `POST /api/tokens/{id}/rotate` sets the predecessor's `revoked_at` to
    `now + grace_hours` (a FUTURE timestamp when grace_hours > 0).
  * Bearer auth in `app.proxy.auth.require_bearer` must therefore only reject
    a token whose `revoked_at` is **non-null AND <= now()** — otherwise the
    grace window does nothing and the old token dies the instant rotate runs.

These tests exercise the full HTTP path through the live FastAPI app on a
real httpx ASGI transport (mirroring tests/integration/test_logout_cancels_stream.py)
so the middleware stack + DB writes are all real.

Bearer-protected route under test: `GET /v1/models`. It depends solely on
`require_bearer`, returns 200 with an empty list when no models are loaded,
and is CSRF-bypassed (prefix `/v1/`) — ideal for isolating the auth check.
"""

import json
import sqlite3
from datetime import timedelta

import bcrypt
import httpx
import pytest
from httpx import ASGITransport

from app.db.repos.tokens import sqlite_utc_in, sqlite_utc_now
from app.proxy.token_cache import invalidate_all


def _seed_admin(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            ("admin", pw),
        )
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.commit()


async def _login_and_prime_csrf(client):
    """Log in as admin and return (jwt_auth_header, csrf_header)."""
    r = await client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    jwt = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {jwt}"}

    r = await client.get("/api/csrf")
    assert r.status_code == 200, r.text
    csrf = r.json()["csrf"]
    # /api/tokens is NOT in the origin_check_dep set — only X-CSRF-Token is
    # required. (Verified: app/tokens/routes_api.py uses require_jwt only;
    # app/auth/routes.py is the only module wiring origin_check_dep.)
    mut = {**auth, "X-CSRF-Token": csrf}
    return auth, mut


async def _create_token(client, mut_headers, name="grace-test"):
    r = await client.post(
        "/api/tokens",
        json={"name": name, "expires_in_days": 0},
        headers=mut_headers,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["id"], body["plaintext"]


async def _rotate_token(client, mut_headers, old_id, grace_hours):
    r = await client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": grace_hours, "expires_in_days": 0},
        headers=mut_headers,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["id"], body["plaintext"]


async def _hit_bearer_route(client, plaintext):
    """Call a bearer-only route and return the response.

    /v1/models is GET, depends only on require_bearer, and returns 200 with
    an empty `data` list when no models are loaded. CSRF middleware bypasses
    the /v1/ prefix.
    """
    return await client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {plaintext}"},
    )


@pytest.fixture
async def app_and_client(tmp_path, monkeypatch):
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    from app.main import build_app

    app = build_app()
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        _seed_admin(tmp_path / "vllm-warden.db")
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=10.0
        ) as client:
            yield app, client, tmp_path
    finally:
        await lifespan_cm.__aexit__(None, None, None)


@pytest.mark.integration
async def test_grace_zero_immediately_rejects_old_token(app_and_client):
    """grace_hours=0 → revoked_at = now → old token rejected on next call."""
    app, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    old_id, old_plaintext = await _create_token(client, mut, name="zero-grace")
    # Sanity: old token works before rotation.
    r = await _hit_bearer_route(client, old_plaintext)
    assert r.status_code == 200, f"pre-rotation auth failed: {r.status_code} {r.text}"

    await _rotate_token(client, mut, old_id, grace_hours=0)

    # NO fixup of revoked_at here — that is the whole point of the test (#185).
    # An earlier version pinned revoked_at one second into the past "for
    # determinism", which overwrote the value under assertion: the test then
    # passed even if grace_hours had been ignored entirely and revoked_at set
    # to now+24h, making it a duplicate of the grace-window test below.
    #
    # No fixup is needed. rotate(grace_hours=0) writes
    # `sqlite_utc_in(timedelta(hours=0))`, i.e. now, and require_bearer
    # compares `revoked_at <= sqlite_utc_now()` (app/proxy/auth.py) — `<=`,
    # over second-granularity "%Y-%m-%d %H:%M:%S" strings. A request landing
    # in the same second as the rotate therefore compares EQUAL and is
    # rejected. The same-clock-tick case is the safe one, not the racy one.
    db_path = tmp_path / "vllm-warden.db"
    with sqlite3.connect(db_path) as conn:
        (revoked_at,) = conn.execute(
            "SELECT revoked_at FROM api_tokens WHERE id = ?",
            (old_id,),
        ).fetchone()
    assert revoked_at is not None, "grace_hours=0 must set revoked_at"
    assert (
        revoked_at <= sqlite_utc_now()
    ), f"grace_hours=0 must revoke NOW, not schedule it; got {revoked_at!r}"

    # After rotate with grace=0, revoked_at <= now → require_bearer must reject.
    r = await _hit_bearer_route(client, old_plaintext)
    assert r.status_code == 401, (
        f"old token must be rejected immediately with grace_hours=0; "
        f"got {r.status_code} {r.text!r}"
    )
    assert "revoked" in r.text.lower(), f"detail should mention 'revoked'; got {r.text!r}"


@pytest.mark.integration
async def test_grace_window_keeps_old_token_alive_then_expires(app_and_client):
    """During grace: old + new both work. After grace elapses: old rejected."""
    app, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    old_id, old_plaintext = await _create_token(client, mut, name="grace-window")
    new_id, new_plaintext = await _rotate_token(client, mut, old_id, grace_hours=24)

    # During the grace window, BOTH tokens must authenticate.
    r_old = await _hit_bearer_route(client, old_plaintext)
    assert r_old.status_code == 200, (
        f"old token must still work during grace window; " f"got {r_old.status_code} {r_old.text!r}"
    )
    r_new = await _hit_bearer_route(client, new_plaintext)
    assert r_new.status_code == 200, (
        f"successor token must work immediately; " f"got {r_new.status_code} {r_new.text!r}"
    )

    # Now fast-forward by moving the predecessor's revoked_at into the past.
    # We use sqlite_utc_in with a negative delta so the column value matches
    # the same SQLite-native naive UTC format that require_bearer compares.
    db_path = tmp_path / "vllm-warden.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE api_tokens SET revoked_at = ? WHERE id = ?",
            (sqlite_utc_in(timedelta(seconds=-1)), old_id),
        )
        conn.commit()
    # The raw UPDATE bypasses TokenRepo, which invalidates the token cache (#279).
    invalidate_all()

    # Old token is now past its grace window → must be rejected.
    r_old = await _hit_bearer_route(client, old_plaintext)
    assert r_old.status_code == 401, (
        f"old token must be rejected after grace window elapses; "
        f"got {r_old.status_code} {r_old.text!r}"
    )
    assert "revoked" in r_old.text.lower(), f"detail should mention 'revoked'; got {r_old.text!r}"

    # Successor must still authenticate — it has no revoked_at set.
    r_new = await _hit_bearer_route(client, new_plaintext)
    assert r_new.status_code == 200, (
        f"successor must remain valid after predecessor's grace expires; "
        f"got {r_new.status_code} {r_new.text!r}"
    )
