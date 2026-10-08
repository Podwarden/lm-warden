"""HTTP route tests for ``GET /api/presets``.

These tests run against the real FastAPI app and a temp DB seeded the
same way every other ``/api/*`` test does. Auth is JWT-only (no SSE
ticket / CSRF dance required for a GET).

We pin:

* 401 without a Bearer (the auth contract for ``/api/*``),
* 200 with the right shape (`presets: [{id, name, ...}]`),
* the four built-in IDs (so a JSON edit catches mismatch with FE tests),
* every preset's ``settings`` dict is non-empty.
"""

from __future__ import annotations

from tests.conftest import jwt_login, seed_admin_user


def _seed(db_path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])


def test_presets_requires_auth(tmp_data_dir, client) -> None:
    """Unauthenticated GET returns 401, matching the rest of ``/api/*``."""
    # Drive /healthz to force lifespan startup (runs migrations + creates
    # `users` and `setup_state`) before we seed via sqlite3.
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/presets")
    assert r.status_code == 401


def test_presets_response_shape(tmp_data_dir, client) -> None:
    """Authed GET returns an envelope ``{presets: [...]}`` with four entries.

    The four IDs are the contract surface the FE test in
    ``add-model-modal`` + ``settings/page`` rely on; renaming any of them
    is a coordinated change.
    """
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/presets", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "presets" in body
    assert isinstance(body["presets"], list)
    ids = {p["id"] for p in body["presets"]}
    assert ids == {
        "a4000-tight-awq",
        "h100-single-shot",
        "dev-tiny",
        "moe-balanced",
    }
    # Each entry must include the full surface — partial shapes would
    # silently break the FE diff popover.
    for entry in body["presets"]:
        assert set(entry.keys()) == {
            "id",
            "name",
            "description",
            "target_archetype",
            "settings",
        }
        assert isinstance(entry["settings"], dict)
        assert entry["settings"], f"preset {entry['id']!r} has empty settings — would be no-op"
