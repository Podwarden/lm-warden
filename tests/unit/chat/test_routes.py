"""JWT-gate smokes for ``app/chat/routes_api.py``.

Parity with ``tests/unit/cache/test_routes.py`` — every other ``/api/*``
slice carries a unit test that pins the 401 contract on each of its
routes, so a future refactor of the auth dependency wiring (removing
``Depends(require_jwt)`` from a single route, swapping the auth
dependency, etc.) is caught here instead of at integration time.

The S8 playground exposes three endpoints, all behind ``require_jwt``:

* ``POST /api/chat/playground/ensure``  — idempotent token-mint
* ``POST /api/chat/completions``        — SSE proxy
* ``GET  /api/admin/active-requests``   — diagnostic counter

For the POST routes we attach a valid CSRF header — without it the CSRF
middleware short-circuits with a 403 BEFORE the auth dependency runs,
which would mask the gate we actually want to assert. Same pattern
used by ``tests/unit/tokens/test_tokens_api.py::test_create_requires_session``.
"""

from __future__ import annotations

from tests.conftest import csrf_header, seed_admin_user


def _seed(db_path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0])


def test_ensure_playground_requires_jwt(tmp_data_dir, client) -> None:
    """``POST /api/chat/playground/ensure`` without a Bearer returns 401."""
    # Force lifespan startup (runs migrations + creates the users table)
    # before seeding via sqlite3.
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/chat/playground/ensure", headers=csrf_header(client))
    assert r.status_code == 401, r.text


def test_chat_completions_requires_jwt(tmp_data_dir, client) -> None:
    """``POST /api/chat/completions`` without a Bearer returns 401.

    Passing a real-looking body would normally trigger Pydantic
    validation, but the auth dependency runs before body parsing — an
    empty JSON object is enough to verify the 401 lands.
    """
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/chat/completions",
        json={},
        headers=csrf_header(client),
    )
    assert r.status_code == 401, r.text


def test_active_requests_requires_jwt(tmp_data_dir, client) -> None:
    """``GET /api/admin/active-requests`` without a Bearer returns 401.

    GET routes don't go through CSRF (``_SAFE_METHODS`` bypass), so the
    csrf_header is not strictly required here — but we send it for shape
    parity with the POST tests above.
    """
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/admin/active-requests")
    assert r.status_code == 401, r.text
