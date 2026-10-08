import json
import secrets
import sqlite3

from fastapi import Depends
from fastapi.testclient import TestClient

from app.proxy.auth import require_bearer
from tests.conftest import csrf_header


def _seed_done_with_token(db_path, plaintext, *, expires_sql: str | None = None):
    """Seed setup-done + insert a known token row.

    expires_sql, when given, is a SQL expression evaluated for expires_at
    (e.g. ``datetime('now', '-1 day')``). When None, leaves the column NULL.
    """
    from app.db.repos.tokens import hash_token

    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        tid = secrets.token_hex(16)
        if expires_sql is None:
            db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope) " "VALUES (?, ?, ?, ?, ?)",
                (tid, "test", plaintext[:8], hash_token(plaintext), "inference"),
            )
        else:
            # expires_sql is interpolated as a raw SQL fragment, not parameterized — pass
            # trusted SQLite expressions only (e.g. datetime('now', '+1 day')), never user input.
            db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, expires_at) "
                f"VALUES (?, ?, ?, ?, ?, {expires_sql})",
                (tid, "test", plaintext[:8], hash_token(plaintext), "inference"),
            )
        db.commit()
        return tid


def _build_test_app():
    """Build a fresh FastAPI with a /protected route guarded by require_bearer."""
    from app.main import build_app

    app = build_app()

    @app.post("/protected")
    async def protected(token=Depends(require_bearer)):
        return {"token_id": token.id}

    return app


def test_proxy_rejects_expired_bearer(tmp_data_dir):
    app = _build_test_app()
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_expiredtoken1234567890abcdef12"
        _seed_done_with_token(
            tmp_data_dir / "vllm-warden.db",
            plaintext,
            expires_sql="datetime('now', '-1 day')",
        )
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401
        detail = r.json().get("detail", "")
        assert "expired" in detail.lower(), f"expected 'expired' in detail, got: {detail!r}"


def test_proxy_accepts_token_with_null_expires_at(tmp_data_dir):
    app = _build_test_app()
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_nullexpiry1234567890abcdef12345"
        tid = _seed_done_with_token(
            tmp_data_dir / "vllm-warden.db",
            plaintext,
            expires_sql=None,
        )
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 200
        assert r.json()["token_id"] == tid


def test_proxy_accepts_token_with_future_expires_at(tmp_data_dir):
    app = _build_test_app()
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_futurexpiry1234567890abcdef1234"
        tid = _seed_done_with_token(
            tmp_data_dir / "vllm-warden.db",
            plaintext,
            expires_sql="datetime('now', '+1 day')",
        )
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 200
        assert r.json()["token_id"] == tid
