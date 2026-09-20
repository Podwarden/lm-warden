import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token


def _seed_loaded(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def _read_counters(db_path):
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "SELECT model_id, token_id, requests, prompt_tokens, completion_tokens FROM counters"
        )
        return [dict(zip([d[0] for d in cur.description], r, strict=False)) for r in cur.fetchall()]


def _read_samples(db_path):
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "SELECT model_id, minute, requests, prompt_tokens, completion_tokens FROM model_samples"
        )
        return [dict(zip([d[0] for d in cur.description], r, strict=False)) for r in cur.fetchall()]


def _make_fake_tokenizer(token_count_for):
    """Return a TokenizerCache double whose .count(repo, text) returns the dict lookup."""
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: token_count_for(text))
    return cache


def test_proxy_increments_counters_on_non_streaming(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(
        lambda text: 5 if "hi" in text else 0
    )

    body = {
        "id": "x", "model": "qwen",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200

    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    assert rows[0]["model_id"] == "qwen"
    assert rows[0]["token_id"] == "tok1"
    assert rows[0]["prompt_tokens"] == 5
    assert rows[0]["completion_tokens"] == 2
    assert rows[0]["requests"] == 1

    samples = _read_samples(tmp_data_dir / "vllm-warden.db")
    assert len(samples) == 1
    assert samples[0]["prompt_tokens"] == 5
    assert samples[0]["completion_tokens"] == 2


def test_proxy_streaming_counts_tokens_via_tokenizer(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    # Map text -> token count for the test
    client.app.state.tokenizers = _make_fake_tokenizer(
        lambda text: len(text.split()) if text else 0
    )

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}],"model":"qwen"}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}],"model":"qwen"}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen", "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            # Drain so _forward.gen()'s finally block runs accounting.
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    # accumulated == "hello world" → tokenizer counts split() words → 2
    assert rows[0]["completion_tokens"] == 2
    assert rows[0]["prompt_tokens"] == 1  # "hi" → 1 word


def test_proxy_anonymous_token_id_when_dependency_skipped(tmp_data_dir, client):
    """Sanity: when require_bearer rejects, no counter row is inserted."""
    client.get("/healthz")
    _seed_loaded(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": []},
    )
    assert r.status_code == 401
    assert _read_counters(tmp_data_dir / "vllm-warden.db") == []
