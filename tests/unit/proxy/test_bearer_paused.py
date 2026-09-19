"""require_bearer and a paused key (spec 2026-09-18 §3.1, D2).

  * paused -> 403 "token paused", and the refusal is not recorded as a use
  * paused AND expired / revoked -> the 401 for the dead state wins
  * paused inside a rotation grace window -> 403 (pausing cuts an old key off
    early)
  * resume -> the very next request passes: auth reads the row every time
"""

import json
import secrets
import sqlite3

from fastapi import Depends
from fastapi.testclient import TestClient

from app.db.repos.tokens import hash_token
from app.proxy.auth import require_bearer
from tests.conftest import csrf_header

PLAINTEXT = "vw_pausedtoken1234567890abcdef1234"


def _seed_token(db_path) -> str:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        tid = secrets.token_hex(16)
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            (tid, "test", PLAINTEXT[:8], hash_token(PLAINTEXT), "inference"),
        )
        db.commit()
    return tid


def _set(db_path, tid: str, assignment: str) -> None:
    # `assignment` is a trusted SQL fragment written in this file, never input.
    with sqlite3.connect(db_path) as db:
        db.execute(f"UPDATE api_tokens SET {assignment} WHERE id = ?", (tid,))
        db.commit()


def _build_test_app():
    from app.main import build_app

    app = build_app()

    @app.post("/protected")
    async def protected(token=Depends(require_bearer)):
        return {"token_id": token.id}

    return app


def _call(client):
    return client.post(
        "/protected",
        headers={**csrf_header(client), "Authorization": f"Bearer {PLAINTEXT}"},
    )


def test_a_paused_key_gets_403_token_paused(tmp_data_dir):
    with TestClient(_build_test_app()) as client:
        client.get("/healthz")
        db_path = tmp_data_dir / "vllm-warden.db"
        tid = _seed_token(db_path)
        _set(db_path, tid, "paused_at = datetime('now')")
        r = _call(client)
        assert r.status_code == 403
        assert r.json()["detail"] == "token paused"
        with sqlite3.connect(db_path) as db:
            (last_used,) = db.execute(
                "SELECT last_used_at FROM api_tokens WHERE id = ?", (tid,)
            ).fetchone()
        assert last_used is None


def test_paused_and_expired_gets_the_expired_401(tmp_data_dir):
    with TestClient(_build_test_app()) as client:
        client.get("/healthz")
        db_path = tmp_data_dir / "vllm-warden.db"
        tid = _seed_token(db_path)
        _set(db_path, tid, "paused_at = datetime('now'), expires_at = datetime('now', '-1 day')")
        r = _call(client)
        assert r.status_code == 401
        assert "expired" in r.json()["detail"]


def test_paused_and_revoked_gets_the_revoked_401(tmp_data_dir):
    with TestClient(_build_test_app()) as client:
        client.get("/healthz")
        db_path = tmp_data_dir / "vllm-warden.db"
        tid = _seed_token(db_path)
        _set(db_path, tid, "paused_at = datetime('now'), revoked_at = datetime('now', '-1 second')")
        r = _call(client)
        assert r.status_code == 401
        assert "revoked" in r.json()["detail"]


def test_paused_inside_a_grace_window_gets_403(tmp_data_dir):
    with TestClient(_build_test_app()) as client:
        client.get("/healthz")
        db_path = tmp_data_dir / "vllm-warden.db"
        tid = _seed_token(db_path)
        _set(db_path, tid, "paused_at = datetime('now'), revoked_at = datetime('now', '+1 day')")
        assert _call(client).status_code == 403


def test_resume_lets_the_next_request_through(tmp_data_dir):
    with TestClient(_build_test_app()) as client:
        client.get("/healthz")
        db_path = tmp_data_dir / "vllm-warden.db"
        tid = _seed_token(db_path)
        _set(db_path, tid, "paused_at = datetime('now')")
        assert _call(client).status_code == 403
        _set(db_path, tid, "paused_at = NULL")
        r = _call(client)
        assert r.status_code == 200
        assert r.json()["token_id"] == tid
