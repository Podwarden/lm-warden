import json
import secrets
import sqlite3

from fastapi import Depends, Request
from fastapi.testclient import TestClient

from app.proxy.auth import require_bearer, upstream_credential_headers
from tests.conftest import csrf_header
from tests.unit.proxy.ledger_helpers import flush_ledger


def _seed_done_with_token(db_path, plaintext):
    """Seed setup-done + insert a known token row."""
    from app.db.repos.tokens import hash_token

    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        tid = secrets.token_hex(16)
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            (tid, "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return tid


def _build_test_app(settings):
    """Build a fresh FastAPI with a /protected route guarded by require_bearer.
    Reuses real settings/middleware via the main app's lifespan."""
    from app.main import build_app

    app = build_app()

    @app.post("/protected")
    async def protected(token=Depends(require_bearer)):
        return {"token_id": token.id}

    return app


def test_proxy_rejects_missing_bearer(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", "vw_unused1234567890abcdef")
        h = csrf_header(client)
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


def test_proxy_rejects_unknown_bearer(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", "vw_unused1234567890abcdef")
        h = {**csrf_header(client), "Authorization": "Bearer vw_nonexistent99"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


def test_proxy_rejects_revoked_bearer(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_validtoken1234567890abcdef12345"
        tid = _seed_done_with_token(tmp_data_dir / "vllm-warden.db", plaintext)
        # revoke it
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            db.execute(
                "UPDATE api_tokens SET revoked_at = datetime('now', '-1 second') WHERE id = ?",
                (tid,),
            )
            db.commit()
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


def test_proxy_accepts_valid_bearer_and_records_last_used(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_validtoken1234567890abcdef12345"
        tid = _seed_done_with_token(tmp_data_dir / "vllm-warden.db", plaintext)
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 200
        assert r.json()["token_id"] == tid
        # last_used_at must now be populated (once the ledger has flushed)
        flush_ledger(client)
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            cur = db.execute("SELECT last_used_at FROM api_tokens WHERE id = ?", (tid,))
            (last_used,) = cur.fetchone()
        assert last_used is not None


def test_proxy_rejects_non_vw_token(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", "vw_unused1234567890abcdef")
        h = {**csrf_header(client), "Authorization": "Bearer sk-openai-style"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


# --- #287: X-LMWarden-Key ----------------------------------------------------


def test_lmwarden_header_is_the_credential_and_marks_the_request(tmp_data_dir):
    from app.main import build_app

    plaintext = "vw_" + secrets.token_hex(16)
    app = build_app()

    @app.post("/protected2")
    async def protected2(request: Request, token=Depends(require_bearer)):
        return {
            "token_id": token.id,
            "via_header": request.state.lmwarden_key_header,
            "upstream": upstream_credential_headers(request),
        }

    with TestClient(app) as c:
        c.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", plaintext)
        csrf = csrf_header(c)
        h = {**csrf, "X-LMWarden-Key": plaintext, "Authorization": "Bearer sk-ant-FAKE"}
        r = c.post("/protected2", headers=h)
        assert r.status_code == 200
        assert r.json()["via_header"] is True
        assert r.json()["upstream"] == {"authorization": "Bearer sk-ant-FAKE"}
        # without the header: today's behaviour, no upstream credential at all
        r = c.post("/protected2", headers={**csrf, "Authorization": f"Bearer {plaintext}"})
        assert r.json()["via_header"] is False and r.json()["upstream"] == {}
        # an empty header is absent
        r = c.post(
            "/protected2",
            headers={**csrf, "X-LMWarden-Key": "  ", "Authorization": f"Bearer {plaintext}"},
        )
        assert r.status_code == 200 and r.json()["via_header"] is False
        # a present header is the credential: nothing else is tried
        r = c.post(
            "/protected2",
            headers={**csrf, "X-LMWarden-Key": "vw_nope", "Authorization": f"Bearer {plaintext}"},
        )
        assert r.status_code == 401
        r = c.post(
            "/protected2",
            headers={**csrf, "X-LMWarden-Key": "notvw", "Authorization": f"Bearer {plaintext}"},
        )
        assert r.status_code == 401
