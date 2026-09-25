"""Integration tests for GET/PATCH /api/settings/runtime.

Exercises the full ASGI stack (csrf middleware, gate_setup, attach_user) so the
test catches issues that a unit-level call wouldn't — in particular the CSRF
header requirement on PATCH. Pattern lifted from
`tests/integration/test_token_rotate_grace.py`.
"""
import json
import sqlite3

import bcrypt
import httpx
import pytest
from httpx import ASGITransport


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
    """Log in as admin and return (auth_only_headers, csrf_plus_auth_headers)."""
    r = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "hunter2"}
    )
    assert r.status_code == 200, r.text
    jwt = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {jwt}"}

    r = await client.get("/api/csrf")
    assert r.status_code == 200, r.text
    csrf = r.json()["csrf"]
    # /api/settings/* is NOT in origin_check_dep — only X-CSRF-Token needed.
    mut = {**auth, "X-CSRF-Token": csrf}
    return auth, mut


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
async def test_get_runtime_returns_full_surface(app_and_client):
    """GET returns all 11 runtime keys (8 seeded + 3 derived).

    `hf_cache_dir` was a ghost key — editable in the UI but never read; the
    cache path is env-driven (`VW_HF_CACHE_DIR`). It is removed from the
    runtime surface entirely, so it must NOT appear here.
    """
    _, client, _ = app_and_client
    auth, _ = await _login_and_prime_csrf(client)

    r = await client.get("/api/settings/runtime", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    expected = (
        "admin_username", "admin_password",
        "hf_token",
        "default_gpu_indices", "default_token_expiration_days",
        "rotation_grace_hours",
        "session_access_ttl_minutes", "session_refresh_ttl_days",
        "sse_ticket_ttl_seconds", "vllm_version", "log_retention_lines",
    )
    for key in expected:
        assert key in body, f"missing key: {key}"
    assert "hf_cache_dir" not in body, "ghost key must be gone from the surface"

    # Sanity: seeded defaults present.
    assert body["session_access_ttl_minutes"] == "15"
    assert body["rotation_grace_hours"] == "24"
    assert body["vllm_version"] == "0.9.2"
    # Admin row exists → password is sentinel, never plaintext.
    assert body["admin_username"] == "admin"
    assert body["admin_password"] == "***"
    # hf_token has never been written → None.
    assert body["hf_token"] is None


@pytest.mark.integration
async def test_patch_no_restart_takes_immediate_effect(app_and_client):
    """default_token_expiration_days is classified "none" → no restart kinds."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"default_token_expiration_days": 90},
        headers=mut,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["requires_restart_kinds"] == []
    assert body["requires_restart"] == []

    # Value was persisted.
    r = await client.get("/api/settings/runtime", headers=mut)
    assert r.json()["default_token_expiration_days"] == "90"


@pytest.mark.integration
async def test_patch_session_ttl_flags_warden_restart(app_and_client):
    """session_access_ttl_minutes is bound at import → warden-restart kind."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"session_access_ttl_minutes": 30},
        headers=mut,
    )
    assert r.status_code == 200, r.text
    assert "warden-restart" in r.json()["requires_restart_kinds"]


@pytest.mark.integration
async def test_patch_hf_token_flags_model_reload(app_and_client, monkeypatch):
    """hf_token is classified model-reload; validator is mocked to avoid HF API."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    async def _fake_validate(token: str):
        # Stand-in for app.system.hf.validate_hf_token. Returns truthy by default.
        class _Who:
            username = "test"
            account_type = "user"
        return _Who()

    monkeypatch.setattr(
        "app.settings.routes_api.validate_hf_token", _fake_validate
    )

    r = await client.patch(
        "/api/settings/runtime",
        json={"hf_token": "hf_test_token_xxx"},
        headers=mut,
    )
    assert r.status_code == 200, r.text
    assert "model-reload" in r.json()["requires_restart_kinds"]

    # Stored — and masked on read.
    r = await client.get("/api/settings/runtime", headers=mut)
    assert r.json()["hf_token"] == "***"


@pytest.mark.integration
async def test_patch_rejects_unknown_keys(app_and_client):
    """Unknown keys → 400 with the offending keys echoed."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"definitely_not_a_real_key": "x"},
        headers=mut,
    )
    assert r.status_code == 400, r.text
    assert "definitely_not_a_real_key" in r.text


@pytest.mark.integration
async def test_patch_hf_cache_dir_is_rejected_as_unknown(app_and_client):
    """`hf_cache_dir` was removed from RUNTIME_KEYS — a PATCH naming it must be
    rejected as an unknown key (400), not silently accepted. The cache path is
    env-driven via `VW_HF_CACHE_DIR` and is not runtime-editable."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"hf_cache_dir": "/somewhere/else"},
        headers=mut,
    )
    assert r.status_code == 400, r.text
    assert "hf_cache_dir" in r.text


@pytest.mark.integration
async def test_patch_hf_token_validator_rejection_is_422(app_and_client, monkeypatch):
    """ValueError from validate_hf_token surfaces as 422 with the message."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    async def _bad_validate(token: str):
        raise ValueError("HuggingFace rejected token")

    monkeypatch.setattr(
        "app.settings.routes_api.validate_hf_token", _bad_validate
    )

    r = await client.patch(
        "/api/settings/runtime",
        json={"hf_token": "hf_badxxx"},
        headers=mut,
    )
    assert r.status_code == 422, r.text
    assert "HuggingFace rejected token" in r.text


@pytest.mark.integration
async def test_get_runtime_requires_jwt(app_and_client):
    """Calling GET without bearer → 401."""
    _, client, _ = app_and_client
    r = await client.get("/api/settings/runtime")
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------------------
# Validation + admin-cred routing (Phase 3 review follow-ups).
# These cover the fixes that landed in the same commit as their tests:
#   - admin_password hashes into users.password_hash (not plaintext into KV)
#   - admin_username updates users.username
#   - hf_token="" returns 422 instead of silently bypassing the validator
#   - per-key coercion: rejects negatives, non-integers, malformed GPU lists
#   - PATCH is all-or-nothing — one bad key aborts before any write commits
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_patch_admin_password_updates_users_table_not_settings_kv(app_and_client):
    """admin_password must bcrypt-hash into users, NOT store plaintext in settings KV."""
    _, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"admin_password": "new-passphrase!"},
        headers=mut,
    )
    assert r.status_code == 200, r.text

    db_path = tmp_path / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        # No plaintext leak — the settings KV either has no row OR has the
        # masked sentinel (we never write the password sentinel today, so the
        # row simply doesn't exist).
        row = db.execute(
            "SELECT value FROM settings WHERE key = 'admin_password'"
        ).fetchone()
        if row is not None:
            assert row[0] != "new-passphrase!", "password leaked into settings KV"
            assert row[0] in ("***", "")

        # bcrypt hash on users.password_hash verifies the new password.
        pw_row = db.execute(
            "SELECT password_hash FROM users ORDER BY id LIMIT 1"
        ).fetchone()
    assert pw_row is not None
    assert bcrypt.checkpw(b"new-passphrase!", pw_row[0].encode())
    assert not bcrypt.checkpw(b"hunter2", pw_row[0].encode())

    # End-to-end: login with the new password succeeds.
    r = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "new-passphrase!"}
    )
    assert r.status_code == 200, r.text


@pytest.mark.integration
async def test_patch_admin_username_updates_users_table(app_and_client):
    """admin_username must update users.username (the source of truth for login)."""
    _, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"admin_username": "newadmin"},
        headers=mut,
    )
    assert r.status_code == 200, r.text

    with sqlite3.connect(tmp_path / "vllm-warden.db") as db:
        username_row = db.execute(
            "SELECT username FROM users ORDER BY id LIMIT 1"
        ).fetchone()
    assert username_row[0] == "newadmin"

    # Sanity: login with the new username (and the original seeded password)
    # succeeds — proves the admin row was mutated, not duplicated.
    r = await client.post(
        "/api/auth/login",
        json={"username": "newadmin", "password": "hunter2"},
    )
    assert r.status_code == 200, r.text


@pytest.mark.integration
async def test_patch_hf_token_empty_string_is_422(app_and_client):
    """hf_token="" must 422 (clearing the token isn't a supported affordance)."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"hf_token": ""},
        headers=mut,
    )
    assert r.status_code == 422, r.text
    assert "empty" in r.text.lower()


@pytest.mark.integration
async def test_patch_runtime_rejects_negative_session_ttl(app_and_client):
    """session_access_ttl_minutes is required positive; -1 must 422."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"session_access_ttl_minutes": -1},
        headers=mut,
    )
    assert r.status_code == 422, r.text
    assert "session_access_ttl_minutes" in r.text


@pytest.mark.integration
async def test_patch_runtime_rejects_non_integer_session_ttl(app_and_client):
    """session_access_ttl_minutes="abc" must 422 (coercer rejects non-numeric)."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"session_access_ttl_minutes": "abc"},
        headers=mut,
    )
    assert r.status_code == 422, r.text
    assert "session_access_ttl_minutes" in r.text


@pytest.mark.integration
async def test_patch_runtime_rejects_malformed_gpu_indices(app_and_client):
    """default_gpu_indices must be a JSON list of non-negative ints — exercise the bad cases + happy path."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    # Not a list (string that doesn't parse as JSON).
    r = await client.patch(
        "/api/settings/runtime",
        json={"default_gpu_indices": "not-a-list"},
        headers=mut,
    )
    assert r.status_code == 422, r.text

    # List with a negative int.
    r = await client.patch(
        "/api/settings/runtime",
        json={"default_gpu_indices": [0, -1]},
        headers=mut,
    )
    assert r.status_code == 422, r.text

    # Happy path: round-trips through GET as a JSON string.
    r = await client.patch(
        "/api/settings/runtime",
        json={"default_gpu_indices": [0, 1, 2]},
        headers=mut,
    )
    assert r.status_code == 200, r.text

    r = await client.get("/api/settings/runtime", headers=mut)
    persisted = r.json()["default_gpu_indices"]
    # Stored as JSON-encoded TEXT; callers parse on read.
    assert json.loads(persisted) == [0, 1, 2]


@pytest.mark.integration
async def test_patch_runtime_partial_failure_writes_nothing(app_and_client):
    """One bad key in a multi-key PATCH must abort the whole transaction — no partial writes."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    # Confirm the seeded baseline first.
    r = await client.get("/api/settings/runtime", headers=mut)
    assert r.json()["session_access_ttl_minutes"] == "15"

    # Send one valid key + one invalid key in the same PATCH.
    r = await client.patch(
        "/api/settings/runtime",
        json={
            "session_access_ttl_minutes": 30,
            "log_retention_lines": -5,  # invalid: must be > 0
        },
        headers=mut,
    )
    assert r.status_code == 422, r.text

    # Re-read: the "valid" key was NOT persisted because validation failed
    # before any DB write.
    r = await client.get("/api/settings/runtime", headers=mut)
    assert r.json()["session_access_ttl_minutes"] == "15"


# ---------------------------------------------------------------------------
# #154 settings redesign — public_url (subsumes #151).
# Pins that the new runtime key is wired through the same surface as every
# other RUNTIME_KEYS entry: GET returns None on a fresh DB (no seed),
# PATCH validates http(s) URLs + strips trailing slash, and the round-trip
# returns the canonical value. Classified `none` → empty kinds.
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_public_url_get_returns_none_on_fresh_db(app_and_client):
    """Migration 0021 does not seed `public_url` — GET reflects absence as None."""
    _, client, _ = app_and_client
    auth, _ = await _login_and_prime_csrf(client)

    r = await client.get("/api/settings/runtime", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "public_url" in body, "key must appear in GET surface even when unset"
    assert body["public_url"] is None


@pytest.mark.integration
async def test_public_url_patch_round_trip_strips_trailing_slash(app_and_client):
    """PATCH a valid URL with trailing slash → stored without slash, GET round-trips."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"public_url": "https://vllm.example.com/"},
        headers=mut,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "ok": True,
        "requires_restart_kinds": [],
        "requires_restart": [],
    }

    # GET observes the trailing-slash-stripped canonical value.
    r = await client.get("/api/settings/runtime", headers=mut)
    assert r.status_code == 200, r.text
    assert r.json()["public_url"] == "https://vllm.example.com"


@pytest.mark.integration
async def test_public_url_patch_422_on_bad_scheme(app_and_client):
    """Non-http(s) scheme → 422 with `public_url` in the detail."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    r = await client.patch(
        "/api/settings/runtime",
        json={"public_url": "ftp://example.com"},
        headers=mut,
    )
    assert r.status_code == 422, r.text
    assert "public_url" in r.text


@pytest.mark.integration
async def test_public_url_patch_422_on_garbage(app_and_client):
    """Unparseable / netloc-less URLs → 422 with `public_url` in the detail."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)

    for bad in ("not a url", "http://", ""):
        r = await client.patch(
            "/api/settings/runtime",
            json={"public_url": bad},
            headers=mut,
        )
        assert r.status_code == 422, f"input={bad!r}: {r.text}"
        assert "public_url" in r.text
