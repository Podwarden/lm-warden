"""Spin up a fake-vllm subprocess, stitch it into the supervisor's state, then
send a real HTTP request through the proxy and verify routing + accounting."""
import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime

import bcrypt
import httpx
import pytest
from httpx import ASGITransport

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo
from app.db.repos.tokens import hash_token


def _seed_done_with_token(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    plaintext = "vw_integtoken1234567890abcdef12345"
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
            "extra_env) "
            "VALUES ('fake','fake-model','fake/repo','main',?,1,'auto',4096,0.9,0,'[]','pulled',0,NULL,NULL,'{}')",
            (json.dumps([0]),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "integ", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
    return plaintext


@pytest.mark.integration
async def test_proxy_routes_to_real_fake_vllm_subprocess(tmp_path, monkeypatch):
    """End-to-end: proxy → fake_vllm subprocess → response."""
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    from app.main import build_app
    app = build_app()

    # Manually run the lifespan so app.state is populated before making requests.
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        plaintext = _seed_done_with_token(tmp_path / "vllm-warden.db")

        # Mock the tokenizer cache — never actually download HF tokenizer.
        from unittest.mock import AsyncMock, MagicMock
        tok_double = MagicMock()
        tok_double.count = AsyncMock(
            side_effect=lambda repo, text, *, fallback_repo=None: len((text or "").split())
        )
        app.state.tokenizers = tok_double

        # Spawn fake_vllm on a fixed port
        port = 18099
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "tests.fakes.fake_vllm",
            "--port", str(port), "--served-model-name", "fake-model",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            # Wait for fake to come up
            async with httpx.AsyncClient(timeout=5.0) as probe:
                for _ in range(50):
                    try:
                        r = await probe.get(f"http://127.0.0.1:{port}/health")
                        if r.status_code == 200:
                            break
                    except Exception:
                        await asyncio.sleep(0.1)
                else:
                    pytest.fail("fake_vllm did not become healthy")

            # Stitch into supervisor + DB so /v1/* routes find the running model
            sup = app.state.supervisor
            from app.runtime.engine.local_subprocess import LocalHandle
            sup._handles["fake"] = LocalHandle(proc)
            sup._ports["fake"] = port
            sup.gpus.claim("fake", [0])
            now = datetime.now(UTC).isoformat()
            async with open_db(app.state.settings.db_path) as db:
                await ModelRepo(db).update_status("fake", "loaded")
                await RuntimeRepo(db).upsert("fake", proc.pid, port, now)

            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Verify /v1/models lists fake-model
                r = await client.get(
                    "/v1/models", headers={"Authorization": f"Bearer {plaintext}"},
                )
                assert r.status_code == 200
                assert any(m["id"] == "fake-model" for m in r.json()["data"])

                # Real request through the proxy
                r = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {plaintext}"},
                    json={"model": "fake-model",
                          "messages": [{"role": "user", "content": "hello"}]},
                )
                assert r.status_code == 200
                body = r.json()
                assert body["choices"][0]["message"]["content"].startswith("echo:")

            # Verify counters got recorded
            with sqlite3.connect(tmp_path / "vllm-warden.db") as db:
                cur = db.execute("SELECT requests, prompt_tokens FROM counters WHERE model_id='fake'")
                row = cur.fetchone()
            assert row is not None
            assert row[0] >= 1  # requests
        finally:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                proc.kill()
                await proc.wait()
    finally:
        await lifespan_cm.__aexit__(None, None, None)
