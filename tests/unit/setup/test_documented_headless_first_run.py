"""The headless first run, exactly as documents/API.md documents it.

documents/API.md's "First run without a browser" section (a shared fragment,
once part of the README) prints a `curl` sequence. A stranger following it
has no other
reference — there is no OpenAPI link in the public docs and the wizard is
otherwise browser-only — so if the endpoints,
their order, their bodies or their response KEYS drift, that section becomes
a set of instructions that cannot work, and the reader has no way to tell
which line is wrong.

This test walks the documented sequence and nothing else. It is a doc test as
much as a route test: each assertion corresponds to a literal claim in that
section, and the comment says which one.

The claims under test:
  1. Six calls, in this order, with these bodies.
  2. /api/setup/* needs NO CSRF token (it is in `_BYPASS_PREFIXES`), which is
     what makes an unattended first run possible at all.
  3. GET /api/setup/state reports the current step, so a partly-finished
     wizard can be resumed rather than restarted.
  4. Posting out of order returns 400 "not at <x> step (current: <y>)".
  5. The HF token accepts JSON null.
  6. The password must be >= 12 characters and <= 72 bytes.
  7. GET /api/csrf returns the key `csrf` -- NOT `csrf_token`.
  8. A minted API token's plaintext is returned once and never again.
"""
from __future__ import annotations

import sqlite3

import pytest

from app.system.gpu import GpuInfo


@pytest.fixture
def two_gpus(monkeypatch):
    from app.system import gpu as gpu_mod

    async def fake_query():
        return [GpuInfo(0, "NVIDIA RTX A4000", 16376, 0, 0),
                GpuInfo(1, "Quadro RTX 5000", 15360, 0, 0)]

    monkeypatch.setattr(gpu_mod, "query_gpus", fake_query)


def test_documented_setup_sequence_completes_without_csrf(
    tmp_data_dir, client, two_gpus
):
    """Claims 1, 2, 3, 5 -- the six calls in the README, verbatim."""
    client.get("/healthz")  # boot, run migrations

    # 1. GET /api/setup/state
    r = client.get("/api/setup/state")
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "welcome", "done": False}

    # 2. POST /api/setup/welcome -- no body, and deliberately NO CSRF header.
    r = client.post("/api/setup/welcome")
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "gpus"}

    # 3. GET /api/setup/gpus -- the indices the next call chooses from.
    r = client.get("/api/setup/gpus")
    assert r.status_code == 200, r.text
    assert [g["index"] for g in r.json()] == [0, 1]
    assert r.json()[0]["name"] == "NVIDIA RTX A4000"

    # Claim 3: the state endpoint tracks progress, so an interrupted run can
    # be resumed from wherever it stopped.
    assert client.get("/api/setup/state").json() == {"step": "gpus", "done": False}

    # 4. POST /api/setup/gpus
    r = client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0, 1]})
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "hf_token"}

    # 5. POST /api/setup/hf_token -- claim 5: JSON null is accepted, and is
    # the right answer unless you need gated models.
    r = client.post("/api/setup/hf_token", json={"hf_token": None})
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "admin", "whoami": None}

    # 6. POST /api/setup/admin
    r = client.post(
        "/api/setup/admin", json={"username": "admin", "password": "s3cret-passphrase"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "done"}

    # Final state.
    assert client.get("/api/setup/state").json() == {"step": "done", "done": True}


def test_out_of_order_step_reports_where_you_actually_are(tmp_data_dir, client):
    """Claim 4. The message names the current step, which is what makes the
    README's advice ("ask /api/setup/state") actionable."""
    client.get("/healthz")
    r = client.post("/api/setup/hf_token", json={"hf_token": None})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "not at hf_token step (current: welcome)"


@pytest.mark.parametrize(
    "password,detail",
    [
        ("short", "password must be at least 12 characters"),
        ("x" * 11, "password must be at least 12 characters"),
        ("x" * 73, "password must be at most 72 bytes"),
    ],
)
def test_documented_password_rules(tmp_data_dir, client, two_gpus, password, detail):
    """Claim 6. Undocumented before this section existed; a generated
    passphrase longer than 72 bytes fails at the LAST step of the wizard."""
    client.get("/healthz")
    client.post("/api/setup/welcome")
    client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0]})
    client.post("/api/setup/hf_token", json={"hf_token": None})

    r = client.post("/api/setup/admin", json={"username": "admin", "password": password})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == detail


def test_gpus_endpoint_is_hidden_once_setup_is_done(tmp_data_dir, client, two_gpus):
    """Documented so a reader who re-runs the script understands the 404."""
    from tests.conftest import seed_admin_user

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/setup/gpus").status_code == 404


def test_documented_csrf_key_is_csrf_not_csrf_token(tmp_data_dir, client):
    """Claim 7. Guessing `csrf_token` yields a flat 403 "csrf token invalid"
    with nothing pointing at the field name -- the single sharpest edge in
    driving this API by hand, and the reason the README prints the jq path."""
    client.get("/healthz")
    body = client.get("/api/csrf").json()
    assert "csrf" in body
    assert "csrf_token" not in body
    assert isinstance(body["csrf"], str) and body["csrf"]


def test_documented_token_mint_bootstrap(tmp_data_dir, client, two_gpus):
    """Claims 7 and 8 end to end: CSRF -> login -> mint, exactly as printed.

    The control API (unlike /api/setup) DOES enforce CSRF, so this is the
    two-header dance a reader has to get right: `X-CSRF-Token` from
    /api/csrf plus `Authorization: Bearer <access_token>` from the login.
    """
    client.get("/healthz")
    client.post("/api/setup/welcome")
    client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0]})
    client.post("/api/setup/hf_token", json={"hf_token": None})
    client.post("/api/setup/admin", json={"username": "admin", "password": "s3cret-passphrase"})

    csrf = client.get("/api/csrf").json()["csrf"]

    login = client.post(
        "/api/auth/login", json={"username": "admin", "password": "s3cret-passphrase"}
    )
    assert login.status_code == 200, login.text
    jwt = login.json()["access_token"]

    minted = client.post(
        "/api/tokens",
        json={"name": "first-key"},
        headers={"Authorization": f"Bearer {jwt}", "X-CSRF-Token": csrf},
    )
    assert minted.status_code == 201, minted.text
    body = minted.json()
    plaintext = body["plaintext"]
    assert plaintext.startswith("vw_")
    assert body["prefix"] == plaintext[:8]

    # Claim 8: shown exactly once. Only a SHA-256 hash is stored, so the
    # listing can never return it again -- copy it now or mint another.
    listed = client.get("/api/tokens", headers={"Authorization": f"Bearer {jwt}"})
    assert listed.status_code == 200, listed.text
    (row,) = listed.json()["items"]
    assert row["id"] == body["id"]
    assert "plaintext" not in row
    assert plaintext not in listed.text

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        (stored,) = db.execute(
            "SELECT hash FROM api_tokens WHERE id = ?", (body["id"],)
        ).fetchone()
    assert stored != plaintext


def test_minted_token_gates_the_openai_endpoint(tmp_data_dir, client, two_gpus):
    """The point of the whole sequence: a key that opens /v1."""
    client.get("/healthz")
    client.post("/api/setup/welcome")
    client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0]})
    client.post("/api/setup/hf_token", json={"hf_token": None})
    client.post("/api/setup/admin", json={"username": "admin", "password": "s3cret-passphrase"})

    csrf = client.get("/api/csrf").json()["csrf"]
    jwt = client.post(
        "/api/auth/login", json={"username": "admin", "password": "s3cret-passphrase"}
    ).json()["access_token"]
    key = client.post(
        "/api/tokens",
        json={"name": "first-key"},
        headers={"Authorization": f"Bearer {jwt}", "X-CSRF-Token": csrf},
    ).json()["plaintext"]

    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert ok.status_code == 200, ok.text
    assert ok.json() == {"object": "list", "data": []}
