import json
import sqlite3

import pytest

from tests.conftest import csrf_header


def _seed_to_step(db_path, step, draft):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step = ?, draft = ? WHERE id = 1",
            (step, json.dumps(draft)),
        )
        db.commit()


def test_post_hf_token_validates_and_advances(tmp_data_dir, client, monkeypatch):
    from app.system import hf as hf_mod

    async def fake_validate(tok):
        from app.system.hf import HfWhoAmI

        return HfWhoAmI(username="alice", account_type="user")

    monkeypatch.setattr(hf_mod, "validate_hf_token", fake_validate)

    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "hf_token", {"allowed_gpu_indices": [0]})

    r = client.post("/api/setup/hf_token", json={"hf_token": "hf_xxx"}, headers=csrf_header(client))
    assert r.status_code == 200
    assert r.json()["whoami"]["username"] == "alice"

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        step, draft_json = db.execute("SELECT step, draft FROM setup_state").fetchone()
    draft = json.loads(draft_json)
    assert step == "admin"
    assert draft["hf_token"] == "hf_xxx"


def test_post_hf_token_can_be_skipped(tmp_data_dir, client):
    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "hf_token", {"allowed_gpu_indices": [0]})

    r = client.post("/api/setup/hf_token", json={"hf_token": None}, headers=csrf_header(client))
    assert r.status_code == 200

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        step, draft_json = db.execute("SELECT step, draft FROM setup_state").fetchone()
    assert step == "admin"
    assert "hf_token" not in json.loads(draft_json)


def test_post_admin_creates_user_and_finalizes(tmp_data_dir, client):
    client.get("/healthz")
    _seed_to_step(
        tmp_data_dir / "vllm-warden.db",
        "admin",
        {"allowed_gpu_indices": [0, 1], "hf_token": "hf_xxx"},
    )
    r = client.post(
        "/api/setup/admin",
        json={"username": "admin", "password": "hunter2-hunter2"},
        headers=csrf_header(client),
    )
    assert r.status_code == 200

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        step, draft_json = db.execute("SELECT step, draft FROM setup_state").fetchone()
        (count,) = db.execute("SELECT COUNT(*) FROM users").fetchone()
    assert step == "done"
    assert count == 1
    draft = json.loads(draft_json)
    assert "hf_token" not in draft
    assert draft.get("hf_token_present") is True


def test_post_admin_rejects_short_password(tmp_data_dir, client):
    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "admin", {})
    r = client.post(
        "/api/setup/admin",
        json={"username": "admin", "password": "abc"},
        headers=csrf_header(client),
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    "password,ok",
    [
        ("a" * 11, False),
        ("a" * 12, True),
        ("é" * 12, True),  # 12 characters, 24 bytes
        ("é" * 36, True),  # 72 bytes exactly
        ("é" * 37, False),  # 74 bytes: past bcrypt's limit
    ],
)
def test_post_admin_password_bounds(tmp_data_dir, client, password, ok):
    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "admin", {})
    r = client.post(
        "/api/setup/admin",
        json={"username": "admin", "password": password},
        headers=csrf_header(client),
    )
    assert (r.status_code == 200) is ok, r.text
    if not ok:
        assert r.status_code == 400


def test_admin_step_refuses_when_any_admin_exists(tmp_data_dir, client):
    """Verified finding #1: the wizard must never add a second admin."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_to_step(db_path, "admin", {"allowed_gpu_indices": [0]})
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES ('first', 'x')")
        db.commit()
    r = client.post(
        "/api/setup/admin",
        json={"username": "intruder", "password": "a-long-password-1"},
        headers=csrf_header(client),
    )
    assert r.status_code == 409, r.text
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT username FROM users").fetchall() == [("first",)]


async def test_two_simultaneous_admin_submits_make_one_admin(live_app, tmp_data_dir):
    """Verified finding #1: two submits racing at the admin step create exactly one admin."""
    import asyncio

    import httpx

    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_to_step(db_path, "admin", {"allowed_gpu_indices": [0]})
    with sqlite3.connect(db_path) as db:  # a fresh install: no admin yet
        db.execute("DELETE FROM users")
        db.commit()
    transport = httpx.ASGITransport(app=live_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
        rs = await asyncio.gather(
            *[
                c.post(
                    "/api/setup/admin",
                    json={"username": f"admin{i}", "password": "a-long-password-1"},
                )
                for i in range(2)
            ]
        )
    codes = sorted(r.status_code for r in rs)
    assert codes[0] == 200 and codes[1] in (400, 409), [r.text for r in rs]
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
