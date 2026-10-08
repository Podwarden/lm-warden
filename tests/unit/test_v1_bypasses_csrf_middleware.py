"""#279 stage 3: /v1 skips the BaseHTTPMiddleware CSRF pair; everything else keeps it."""

import json
import sqlite3

from app.db.repos.tokens import hash_token


def _seed_token(db_path) -> str:
    plaintext = "vw_validtoken1234567890abcdef12345"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
    return plaintext


def test_v1_skips_csrf_stack_but_api_csrf_keeps_it(tmp_data_dir, client, monkeypatch):
    import app.main as main_mod

    client.get("/healthz")
    plaintext = _seed_token(tmp_data_dir / "vllm-warden.db")

    calls = {"ensure": 0, "check": 0}
    real_ensure, real_check = main_mod.ensure_csrf_id, main_mod.csrf_check

    async def counting_ensure(request, call_next):
        calls["ensure"] += 1
        return await real_ensure(request, call_next)

    async def counting_check(request, call_next):
        calls["check"] += 1
        return await real_check(request, call_next)

    # The factory resolves these names at request time, so patching the module
    # attributes on the already-built app is enough.
    monkeypatch.setattr(main_mod, "ensure_csrf_id", counting_ensure)
    monkeypatch.setattr(main_mod, "csrf_check", counting_check)

    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    assert calls == {"ensure": 0, "check": 0}

    r = client.get("/api/csrf")
    assert r.status_code == 200 and r.json()["csrf"]
    assert calls["ensure"] == 1 and calls["check"] == 1
