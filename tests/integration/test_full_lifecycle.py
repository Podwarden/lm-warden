"""Hermetic full-lifecycle integration test (#126 / S9).

Drives the entire overhauled happy-path through one process so a regression
in the proxy → counters → token_usage_minute → stats-v2 chain trips a single
test instead of leaking past unit coverage:

  seed admin user + inference token + a 'loaded' model row
  → JWT login (real /api/auth/login route)
  → POST /v1/chat/completions through the proxy → fake_vllm subprocess
  → assert counters row + token_usage_minute row landed
  → GET /api/stats/v2/overview with the JWT and assert series.tokens
    reflects the request that just ran

Mirrors the subprocess + lifespan-stitch pattern from
``test_proxy_real_subprocess.py``; uses its own free-port helper so two
copies of the integration suite can run in parallel without colliding on
port 18099 (Plan flagged the leak risk).
"""
import asyncio
import json
import socket
import sqlite3
import sys
import time
from datetime import UTC, datetime

import bcrypt
import httpx
import pytest
from httpx import ASGITransport

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo
from app.db.repos.tokens import hash_token


def _find_free_port() -> int:
    """Pick an ephemeral port the OS is happy to hand out *right now*.

    Releases the socket before returning so the subprocess can rebind. There
    is a tiny TOCTOU window between close and rebind but in practice it has
    never collided on CI; the alternative (SO_REUSEPORT + held socket) is
    more code than the risk warrants.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _seed_done_with_token(db_path) -> tuple[str, str]:
    """Seed an admin user, a fake-model row in 'loaded' state, and one
    inference api_token. Returns (username/password, token plaintext).

    Goes straight to sqlite3 — the wizard's POST endpoints have their own
    coverage and we want the test to fail on proxy/stats wiring, not on
    incidental setup-flow regressions.
    """
    username = "admin"
    password = "hunter2"
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    plaintext = "vw_fulllifecycletoken1234567890ab"
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)",
                   (username, pw_hash))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes, pulled_total, last_error, extra_env) VALUES "
            "('fake','fake-model','fake/repo','main',?,1,'auto',4096,0.9,0,"
            "'[]','pulled',0,NULL,NULL,'{}')",
            (json.dumps([0]),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES (?, ?, ?, ?, ?)",
            ("tok1", "integ", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
    return password, plaintext


@pytest.mark.integration
async def test_full_lifecycle_chat_then_stats_v2(tmp_path, monkeypatch):
    """seed → login → chat → counters/token_usage_minute → stats-v2 overview."""
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    from app.main import build_app
    app = build_app()

    # Manually drive the lifespan so app.state is populated before the test
    # speaks HTTP to it. Mirrors test_proxy_real_subprocess.py.
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        password, plaintext = _seed_done_with_token(tmp_path / "vllm-warden.db")

        # Stub the tokenizer cache so the proxy doesn't reach out to HF for
        # the prompt-token count; the real tokenizer path has its own tests
        # and would slow this case to the network-IO it's specifically
        # designed to avoid.
        from unittest.mock import AsyncMock, MagicMock
        tok_double = MagicMock()
        tok_double.count = AsyncMock(
            side_effect=lambda repo, text, *, fallback_repo=None:
                len((text or "").split())
        )
        app.state.tokenizers = tok_double

        port = _find_free_port()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "tests.fakes.fake_vllm",
            "--port", str(port), "--served-model-name", "fake-model",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        try:
            # Wait for fake_vllm to become healthy. 1s budget (10 × 100ms).
            async with httpx.AsyncClient(timeout=1.0) as probe:
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        r = await probe.get(f"http://127.0.0.1:{port}/health")
                        if r.status_code == 200:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("fake_vllm did not become healthy within 1s")

            # Stitch into supervisor + DB so /v1/* routes find the running model.
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
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver",
            ) as client:
                # Real login through the real route. Captures the JWT access
                # token in the body and the refresh cookie in the jar.
                r = await client.post(
                    "/api/auth/login",
                    json={"username": "admin", "password": password},
                )
                assert r.status_code == 200, r.text
                jwt = r.json()["access_token"]
                assert jwt

                # The chat call goes through the proxy with the inference
                # token (Bearer vw_...), NOT the JWT. The JWT is for
                # the admin/stats surface we hit next.
                r = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {plaintext}"},
                    json={
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["choices"][0]["message"]["content"].startswith("echo:")

                # Stats overview, JWT-authed. Use 1h (smallest valid range).
                r = await client.get(
                    "/api/stats/v2/overview",
                    params={"range": "1h"},
                    headers={"Authorization": f"Bearer {jwt}"},
                )
                assert r.status_code == 200, r.text
                overview = r.json()

            # Assert wire-shape AND the inference call we just made is
            # visible: token_usage_minute is written in _record_counters
            # only when token_id is set, so seeing > 0 in series.tokens
            # proves the proxy → CountersRepo → TokenUsageRepo path ran.
            assert "series" in overview and "tokens" in overview["series"]
            token_series = overview["series"]["tokens"]
            assert len(token_series) > 0, overview
            assert sum(b["prompt"] + b["completion"] for b in token_series) > 0

            # current.tps is the last-full-minute view; might be 0 if the
            # request landed in a fresh minute, so we don't require it. But
            # the snapshot block must exist and be shaped correctly.
            assert "current" in overview
            assert "vram_used_mib" in overview["current"]
            assert "tps" in overview["current"]

            # Raw DB sanity checks — belt and suspenders so a future
            # stats-v2 response shape change doesn't silently mask
            # a counters regression.
            with sqlite3.connect(tmp_path / "vllm-warden.db") as db:
                cur = db.execute(
                    "SELECT requests, prompt_tokens, completion_tokens "
                    "FROM counters WHERE model_id='fake' AND token_id='tok1'"
                )
                row = cur.fetchone()
                assert row is not None and row[0] >= 1, row
                cur = db.execute(
                    "SELECT requests FROM token_usage_minute "
                    "WHERE token_id='tok1'"
                )
                usage = cur.fetchall()
                assert len(usage) >= 1, usage
                assert sum(r[0] for r in usage) >= 1
        finally:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                proc.kill()
                await proc.wait()
            sup._handles.clear()
            sup._ports.clear()
    finally:
        await lifespan_cm.__aexit__(None, None, None)
