"""Rotation keeps a key's policy and never revives a dead key (verified findings #14, #15)."""

from __future__ import annotations

import sqlite3
from dataclasses import fields

from app.db.repos.tokens import TokenRow
from tests.conftest import csrf_header, jwt_login, seed_admin_user

#: Columns that legitimately differ between a key and its successor.
NOT_POLICY = {
    "id",
    "name",
    "prefix",
    "revoked_at",
    "last_used_at",
    "created_at",
    "rotated_at",
    "rotated_from",
    "paused_at",
    # legacy, unenforced; tests/unit/db/test_repos_tokens_v2.py pins that they are not carried over
    "rate_limit_rpm",
    "rate_limit_tpm",
    "rate_limit_tps",
    # inert since 2026-11-22 (owner decision): nothing reads it, not carried
    "godmode_self_view",
}


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return {**jwt_login(client), **csrf_header(client)}


def _row(db_path, tid):
    cols = [f.name for f in fields(TokenRow)]
    with sqlite3.connect(db_path) as db:
        r = db.execute(f"SELECT {', '.join(cols)} FROM api_tokens WHERE id = ?", (tid,)).fetchone()
    return dict(zip(cols, r, strict=True))


def test_rotation_copies_every_policy_column(tmp_data_dir, client):
    h = _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    tid = client.post(
        "/api/tokens", json={"name": "k", "priority": 3, "thinking_policy": "off"}, headers=h
    ).json()["id"]
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE api_tokens SET allowed_models = 'qwen', rate_limit_rpm = 10, rate_limit_tpm = 1000, rate_limit_tps = 5,"
            ' anthropic_relay = 1, prompt_diet = \'{"strip_tools":["x"],"strip_reasoning":false}\','
            " capture_until = 9999999999, guard = '{\"off\": true}', keep_captures = 1"
            " WHERE id = ?",
            (tid,),
        )
        db.commit()
    r = client.post(f"/api/tokens/{tid}/rotate", json={"grace_hours": 1}, headers=h)
    assert r.status_code == 201, r.text
    old, new = _row(db_path, tid), _row(db_path, r.json()["id"])
    differ = {k for k in old if k not in NOT_POLICY and old[k] != new[k]}
    assert differ == set(), {k: (old[k], new[k]) for k in differ}
    assert new["allowed_models"] == "qwen" and new["guard"] == '{"off": true}'
    assert new["keep_captures"] == 1


def test_a_revoked_key_cannot_be_rotated_back_to_life(tmp_data_dir, client):
    h = _ready(client, tmp_data_dir)
    created = client.post("/api/tokens", json={"name": "k"}, headers=h).json()
    tid, plain = created["id"], created["plaintext"]
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:  # revoked, never rotated
        db.execute(
            "UPDATE api_tokens SET revoked_at = datetime('now', '-1 hour') WHERE id = ?", (tid,)
        )
        db.commit()
    r = client.post(f"/api/tokens/{tid}/rotate", json={"grace_hours": 24}, headers=h)
    assert r.status_code == 409, r.text
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {plain}"}).status_code == 401


def test_an_expired_key_cannot_be_rotated(tmp_data_dir, client):
    h = _ready(client, tmp_data_dir)
    tid = client.post("/api/tokens", json={"name": "k"}, headers=h).json()["id"]
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE api_tokens SET expires_at = datetime('now', '-1 day') WHERE id = ?", (tid,)
        )
        db.commit()
    assert client.post(f"/api/tokens/{tid}/rotate", json={}, headers=h).status_code == 409
