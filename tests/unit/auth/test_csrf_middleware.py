"""CSRF middleware integration tests.

After Phase 4 the cookie-session flow is gone; the CSRF middleware now always
binds tokens to the anonymous `vw_csrf_id` cookie. Tests that asserted on
deleted routes (POST /login, POST /tokens form, session-bound CSRF) have been
removed — the new frontend will own those scenarios in MR-2.
"""

import sqlite3

from tests.conftest import csrf_header, jwt_login, seed_admin_user

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _seed_done(db_path, *, allowed=None, plaintext_token=None):
    """Seed setup-done state; optionally add an admin user and/or an API token.

    #55 fix — defer the admin user + setup_state writes to the shared
    barrier-aware helper. The optional API-token row stays inline since
    it's specific to this module's CSRF + bearer-token scenarios.
    """
    seed_admin_user(db_path, allowed_gpu_indices=allowed)
    if plaintext_token is not None:
        from app.db.repos.tokens import hash_token

        with sqlite3.connect(db_path) as db:
            db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, allowed_models) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "tok-csrf-test",
                    "csrf-test",
                    plaintext_token[:8],
                    hash_token(plaintext_token),
                    "inference",
                    None,
                ),
            )
            db.commit()


def _jwt_auth(client, username="admin", password="hunter2"):
    # #55 fix — delegate to the barrier+retry helper.
    return jwt_login(client, username=username, password=password)


# ---------------------------------------------------------------------------
# Test 1: GET passes without any CSRF token
# ---------------------------------------------------------------------------


def test_get_request_passes_without_csrf(tmp_data_dir, client):
    """GET to any path must not require a CSRF token."""
    r = client.get("/healthz")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test 2: POST without CSRF returns 403
# ---------------------------------------------------------------------------


def test_post_without_csrf_returns_403(tmp_data_dir, client):
    """POST /api/settings with no CSRF header and no form field → 403."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)
    r = client.post("/api/settings", json={"allowed_gpu_indices": [0]}, headers=auth)
    assert r.status_code == 403
    assert r.json()["detail"] == "csrf token invalid"


# ---------------------------------------------------------------------------
# Test 3: POST with invalid CSRF returns 403
# ---------------------------------------------------------------------------


def test_post_with_invalid_csrf_returns_403(tmp_data_dir, client):
    """POST with a garbage X-CSRF-Token header → 403."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)
    r = client.post(
        "/api/settings",
        json={"allowed_gpu_indices": [0]},
        headers={**auth, "X-CSRF-Token": "deadbeef"},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "csrf token invalid"


# ---------------------------------------------------------------------------
# Test 4: POST with valid CSRF passes (reaches route, NOT 403)
# ---------------------------------------------------------------------------


def test_post_with_valid_csrf_passes(tmp_data_dir, client):
    """GET to mint vw_csrf_id + token; subsequent POST is not 403."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)
    # /api/settings returns 401 if not authed, but we are logged in, so we
    # should get 200 or some domain error — just NOT 403 csrf.
    from unittest.mock import AsyncMock, patch

    fake = AsyncMock(return_value=None)
    with patch("app.settings.routes_api.validate_hf_token", fake):
        r = client.post(
            "/api/settings",
            json={"allowed_gpu_indices": [0, 1, 2, 3]},
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code != 403 or r.json().get("detail") != "csrf token invalid"


# ---------------------------------------------------------------------------
# Test 5: /v1/ bearer routes skip CSRF check
# ---------------------------------------------------------------------------


def test_v1_bearer_routes_skip_csrf(tmp_data_dir, client):
    """POST /v1/chat/completions without X-CSRF-Token must NOT return 403 (csrf)."""
    client.get("/healthz")
    plaintext = "vw_csrftesttoken1234567890abcdef"
    _seed_done(tmp_data_dir / "vllm-warden.db", plaintext_token=plaintext)

    # No CSRF header at all — should reach the proxy layer and 404/401, not 403 csrf.
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Anything except 403 "csrf token invalid" is acceptable.
    assert not (r.status_code == 403 and r.json().get("detail") == "csrf token invalid")


# ---------------------------------------------------------------------------
# Test 6: GET /api/csrf returns token and that token works in a POST
# ---------------------------------------------------------------------------


def test_csrf_endpoint_returns_token(tmp_data_dir, client):
    """GET /api/csrf → 200 with {csrf: <64-hex-char string>}.
    The token must also be valid for a subsequent POST."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)

    r = client.get("/api/csrf")
    assert r.status_code == 200
    body = r.json()
    assert "csrf" in body
    token = body["csrf"]
    # HMAC-SHA256 hex digest is 64 chars
    assert len(token) == 64
    assert all(c in "0123456789abcdef" for c in token)

    # Verify the token actually works on a protected POST.
    from unittest.mock import AsyncMock, patch

    fake = AsyncMock(return_value=None)
    with patch("app.settings.routes_api.validate_hf_token", fake):
        post_r = client.post(
            "/api/settings",
            json={"allowed_gpu_indices": [0, 1, 2, 3]},
            headers={**auth, "X-CSRF-Token": token},
        )
    assert not (post_r.status_code == 403 and post_r.json().get("detail") == "csrf token invalid")


# ---------------------------------------------------------------------------
# Test 7: vw_csrf_id cookie is auto-minted on first request
# ---------------------------------------------------------------------------


def test_csrf_id_cookie_auto_minted(tmp_data_dir, client):
    """First GET with no cookies sets Set-Cookie: vw_csrf_id=..."""
    # Use a fresh client that has no cookies yet.
    from fastapi.testclient import TestClient

    from app.main import build_app

    app = build_app()
    with TestClient(app, cookies={}) as fresh_client:
        r = fresh_client.get("/healthz")
    # TestClient follows cookies by default, but we can inspect Set-Cookie header.
    assert "vw_csrf_id" in r.cookies or "vw_csrf_id" in r.headers.get("set-cookie", "")
