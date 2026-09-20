"""God-mode taps in ``_forward``.

Two contracts:
  * OFF by default (production safety): with ``godmode_enabled=false`` the hub
    is NEVER called and the streamed response bytes are byte-identical to today.
  * ON: a streaming request emits request_start → delta(content) +
    delta(reasoning) → request_end (with the tracked finish_reason); a
    non-stream request emits request_start → delta(s) → request_end.
"""

import dataclasses
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
            ("tok1", "my-token", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def _fake_tokenizer(count_for):
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: count_for(text))
    return cache


def _enable_godmode(client):
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, godmode_enabled=True
    )


# --------------------------------------------------------------------------
# Production-safety regression: disabled → zero hub calls + byte-identical body
# --------------------------------------------------------------------------
def test_disabled_makes_zero_hub_calls_and_byte_identical_stream(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)

    # Spy hub in place of the real one. godmode_enabled is false by default,
    # so the gate must never touch it.
    spy = MagicMock()
    client.app.state.godmode_hub = spy

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
            "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            body = b"".join(r.iter_bytes())
            assert r.status_code == 200

    # Byte-identical to the raw upstream stream.
    assert body == b"".join(sse_chunks)
    # And NOT ONE hub method was called on the disabled hot path.
    assert spy.publish.call_count == 0
    assert spy.subscribe.call_count == 0
    assert spy.method_calls == []


# --------------------------------------------------------------------------
# Enabled — streaming: request_start → delta(content) + delta(reasoning) → end
# --------------------------------------------------------------------------
def test_enabled_streaming_emits_start_deltas_end(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    _enable_godmode(client)

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
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
            "POST", "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    hub = client.app.state.godmode_hub
    _q, snap = hub.subscribe()
    types = [e["type"] for e in snap]
    assert types[0] == "request_start"
    assert types[-1] == "request_end"

    start = snap[0]
    assert start["prompt"] == "hi"
    assert start["token_label"] == "my-token"
    assert start["token_id"] == "tok1"
    assert start["model"] == "qwen"
    assert start["stream"] is True
    # The raw bearer token must never appear on any event.
    assert all(plaintext not in json.dumps(e) for e in snap)

    content = [e for e in snap if e["type"] == "delta" and e["channel"] == "content"]
    reasoning = [e for e in snap if e["type"] == "delta" and e["channel"] == "reasoning"]
    assert "".join(e["text"] for e in content) == "hello"
    assert [e["text"] for e in reasoning] == ["thinking"]

    end = snap[-1]
    assert end["finish_reason"] == "stop"
    assert end["prompt_tokens"] == 1  # "hi" → one word

    # Every event names the key, so a per-token stream can filter exactly.
    assert [e.get("token_id") for e in snap] == ["tok1"] * len(snap)


# --------------------------------------------------------------------------
# Enabled — non-stream: request_start → one delta per channel → request_end
# --------------------------------------------------------------------------
def test_enabled_nonstream_emits_start_delta_end(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 5 if "hi" in t else 0)
    _enable_godmode(client)

    body = {
        "id": "x", "model": "qwen",
        "choices": [{
            "message": {"content": "answer", "reasoning_content": "because"},
            "finish_reason": "stop",
        }],
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

    hub = client.app.state.godmode_hub
    _q, snap = hub.subscribe()
    types = [e["type"] for e in snap]
    assert types[0] == "request_start"
    assert snap[0]["stream"] is False
    assert types[-1] == "request_end"

    content = [e for e in snap if e["type"] == "delta" and e["channel"] == "content"]
    reasoning = [e for e in snap if e["type"] == "delta" and e["channel"] == "reasoning"]
    assert [e["text"] for e in content] == ["answer"]
    assert [e["text"] for e in reasoning] == ["because"]

    end = snap[-1]
    assert end["finish_reason"] == "stop"
    assert end["completion_tokens"] == 2

    # Every event names the key, so a per-token stream can filter exactly.
    assert [e.get("token_id") for e in snap] == ["tok1"] * len(snap)
