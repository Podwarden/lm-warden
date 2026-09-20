"""Integration tests for the runaway detector wired into the proxy forward path.

These drive real proxied requests through the FastAPI TestClient (same shape as
test_content_log.py: seed a loaded model + token, patch httpx.send) and assert
on the forward path's observable behaviour under each ``VW_RUNAWAY_MODE``:

* ``off`` — byte-identical passthrough AND no upstream stream-forcing (the hard
  regression guard).
* ``log`` / ``enforce`` — upstream stream is forced (so the detector sees every
  token); a non-streaming client gets a faithfully re-aggregated JSON response;
  a trip yields the runaway ``finish_reason`` on partial content at HTTP 200
  (never a 500); a streaming client that trips gets a terminal runaway chunk.
"""

import asyncio
import dataclasses
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.config import Settings
from app.db.repos.tokens import hash_token
from app.proxy import content_log

# ---------------------------------------------------------------------------
# write_entry(extra=...) — incident record carries the think-token count (§5)
# ---------------------------------------------------------------------------

def test_write_entry_merges_extra_fields(tmp_path):
    from pathlib import Path

    settings = Settings(
        data_dir=Path("/data"),
        hf_cache_dir=Path("/hf"),
        cookie_secret="x" * 32,
        container_gpu_count=0,
        content_log_path=tmp_path / "c.jsonl",
        content_log_max_chars=40000,
    )
    # write_entry is a coroutine since the write moved off the event loop; a
    # bare call would build a coroutine object and write nothing.
    asyncio.run(content_log.write_entry(
        settings,
        token_id="tok1",
        model_id="qwen",
        served_name="qwen",
        stream=True,
        max_tokens=None,
        prompt_tokens=3,
        completion_tokens=100,
        finish_reason="runaway_think",
        prompt="p",
        completion="c",
        extra={"think_tokens": 88, "signal": "runaway_think"},
    ))
    rec = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])
    assert rec["think_tokens"] == 88
    assert rec["signal"] == "runaway_think"
    assert rec["finish_reason"] == "runaway_think"


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------

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


def _prep(client, tmp_data_dir, **runaway):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    tok = MagicMock()
    tok.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text.split()) if text else 0)
    client.app.state.tokenizers = tok
    client.app.state.settings = dataclasses.replace(client.app.state.settings, **runaway)
    return plaintext


def _fake_nonstream_resp(body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def _fake_stream_resp(chunks: list[bytes]):
    async def aiter():
        for c in chunks:
            yield c

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _fake_error_resp(status: int, body: dict):
    """Upstream ERROR response: a plain JSON error envelope, NOT an SSE stream.

    vLLM returns ``content-type: application/json`` with a
    ``{"error": {...}}`` body on 4xx/5xx even when the request asked for
    ``stream:true``. ``aiter_bytes`` yields that body once so the (buggy)
    re-aggregation loop can run against it and produce its degenerate empty
    skeleton pre-fix; ``aread`` buffers the whole body on the forced-stream
    response the fall-through non-stream path relies on (as real httpx does).
    """
    payload = json.dumps(body).encode()

    async def aiter():
        yield payload

    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.aiter_bytes = aiter
    resp.aread = AsyncMock(return_value=payload)
    resp.aclose = AsyncMock()
    return resp


def _capturing_send(resp):
    """AsyncMock replacement for httpx.AsyncClient.send that records the
    outgoing (body, stream) so tests can assert on stream-forcing."""
    calls = []

    async def _send(request, *, stream=False, **kw):
        raw = request.read()
        calls.append({"body": json.loads(raw) if raw else {}, "stream": stream})
        return resp

    return AsyncMock(side_effect=_send), calls


def _chat_chunk(content=None, *, role=None, finish=None, reasoning=None):
    delta = {}
    if role is not None:
        delta["role"] = role
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if content is not None:
        delta["content"] = content
    ev = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1000,
        "model": "qwen",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(ev).encode() + b"\n\n"


def _usage_chunk(pt=1, ct=2):
    ev = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1000,
        "model": "qwen",
        "choices": [],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }
    return b"data: " + json.dumps(ev).encode() + b"\n\n"


# ---------------------------------------------------------------------------
# off mode: byte-identical + no stream forcing
# ---------------------------------------------------------------------------

def test_off_mode_nonstream_no_stream_forcing(tmp_data_dir, client):
    plaintext = _prep(client, tmp_data_dir, runaway_mode="off")
    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    send, calls = _capturing_send(_fake_nonstream_resp(body))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert r.json() == body
    # No stream-forcing: the request BODY is unmutated. Note the transport-level
    # send(stream=True) is unconditional in develop (f5a0a70 — it is what keeps
    # the non-stream wall-clock reaper live) and is therefore not a runaway
    # signal; the detector's forcing is visible only as body mutation below.
    assert calls[0]["stream"] is True
    assert "stream" not in calls[0]["body"]
    assert "stream_options" not in calls[0]["body"]


def test_off_mode_stream_passthrough_no_include_usage_injection(tmp_data_dir, client):
    plaintext = _prep(client, tmp_data_dir, runaway_mode="off")
    chunks = [_chat_chunk("hello", role="assistant"), _chat_chunk(" world", finish="stop"), b"data: [DONE]\n\n"]
    send, calls = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            out = b"".join(r.iter_bytes())
    assert r.status_code == 200
    # Passthrough is byte-for-byte the upstream SSE.
    assert out == b"".join(chunks)
    assert calls[0]["stream"] is True
    assert calls[0]["body"].get("stream") is True
    # off mode must NOT inject include_usage.
    assert "stream_options" not in calls[0]["body"]


# ---------------------------------------------------------------------------
# log/enforce: force upstream stream, re-aggregate for a non-streaming client
# ---------------------------------------------------------------------------

def test_log_mode_nonstream_client_gets_reaggregated_json(tmp_data_dir, client):
    plaintext = _prep(client, tmp_data_dir, runaway_mode="log")
    chunks = [
        _chat_chunk("the answer", role="assistant"),
        _chat_chunk(" is 42", finish="stop"),
        _usage_chunk(pt=5, ct=4),
        b"data: [DONE]\n\n",
    ]
    send, calls = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    # Upstream stream was forced with include_usage so we recovered usage.
    assert calls[0]["stream"] is True
    assert calls[0]["body"]["stream"] is True
    assert calls[0]["body"]["stream_options"]["include_usage"] is True
    out = r.json()
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "the answer is 42"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 9


def test_enforce_trip_nonstream_returns_partial_with_runaway_finish(tmp_data_dir, client):
    # hard_max=3 → trips after a few deltas, no natural finish arrives.
    plaintext = _prep(
        client, tmp_data_dir, runaway_mode="enforce",
        runaway_hard_max=3, runaway_think_budget=10**9, runaway_repeat_max=10**9,
    )
    chunks = [_chat_chunk("x", role="assistant")] + [_chat_chunk("x") for _ in range(20)]
    send, _ = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    # Never a 500 — the generation itself was fine, we tore it down deliberately.
    assert r.status_code == 200
    out = r.json()
    assert out["choices"][0]["finish_reason"] == "runaway_hard"
    # Partial content: fewer than the full 21 deltas made it in.
    assert 0 < len(out["choices"][0]["message"]["content"]) < 21


def test_enforce_stream_passthrough_ok_is_unchanged(tmp_data_dir, client):
    plaintext = _prep(
        client, tmp_data_dir, runaway_mode="enforce",
        runaway_hard_max=10**9, runaway_think_budget=10**9, runaway_repeat_max=10**9,
    )
    chunks = [
        _chat_chunk("hello", role="assistant"),
        _chat_chunk(" world", finish="stop"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    send, _ = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            out = b"".join(r.iter_bytes())
    assert r.status_code == 200
    # OK generation: client sees exactly the upstream SSE (passthrough), plus
    # the finish_reason:stop is preserved.
    assert b"hello" in out and b" world" in out
    assert b"stop" in out
    assert out.rstrip().endswith(b"[DONE]")


def test_enforce_trip_stream_emits_terminal_runaway_chunk(tmp_data_dir, client):
    plaintext = _prep(
        client, tmp_data_dir, runaway_mode="enforce",
        runaway_hard_max=3, runaway_think_budget=10**9, runaway_repeat_max=10**9,
    )
    chunks = [_chat_chunk("x", role="assistant")] + [_chat_chunk("x") for _ in range(20)]
    send, _ = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            out = b"".join(r.iter_bytes())
    assert r.status_code == 200
    assert b"runaway_hard" in out
    assert out.rstrip().endswith(b"[DONE]")


def test_enforce_think_runaway_via_reasoning_content(tmp_data_dir, client):
    # The prod reasoner streams reasoning as delta.reasoning_content (no <think>
    # tag in the text). The forward path must synthesize the think boundary so
    # the budget signal sees an unclosed reasoning block.
    plaintext = _prep(
        client, tmp_data_dir, runaway_mode="enforce",
        runaway_think_budget=5, runaway_hard_max=10**9, runaway_repeat_max=10**9,
    )
    # 30 reasoning deltas, no content ever → unclosed think block.
    chunks = [_chat_chunk(reasoning="step", role="assistant")]
    chunks += [_chat_chunk(reasoning="more") for _ in range(30)]
    send, _ = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "runaway_think"


# ---------------------------------------------------------------------------
# runaway armed + non-SSE upstream ERROR: real error body must pass through
# (regression — arming the detector must NOT re-aggregate a non-stream error
# response into an empty skeleton and swallow the genuine error envelope).
# ---------------------------------------------------------------------------

def test_runaway_nonstream_upstream_500_passes_through_error_body(tmp_data_dir, client):
    plaintext = _prep(client, tmp_data_dir, runaway_mode="log")
    err = {"error": {"message": "engine OOM", "type": "internal_server_error"}}
    send, calls = _capturing_send(_fake_error_resp(500, err))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    # We still forced upstream stream=true so the detector could watch tokens...
    assert calls[0]["stream"] is True
    assert calls[0]["body"]["stream"] is True
    # ...but the upstream errored with a JSON envelope, not an SSE stream, so the
    # re-aggregation branch must be bypassed and the genuine error body returned.
    assert r.status_code == 500
    out = r.json()
    assert out["error"]["message"] == "engine OOM"
    # NOT the degenerate re-aggregation skeleton.
    assert "choices" not in out
    assert out.get("object") != "chat.completion"


def test_runaway_nonstream_upstream_400_passes_through_error_body(tmp_data_dir, client):
    plaintext = _prep(client, tmp_data_dir, runaway_mode="enforce")
    err = {"error": {"message": "bad request: max_tokens too large", "type": "invalid_request_error"}}
    send, _ = _capturing_send(_fake_error_resp(400, err))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 400
    out = r.json()
    assert out["error"]["type"] == "invalid_request_error"
    assert "choices" not in out


def test_runaway_nonstream_upstream_5xx_gets_db_error_enrichment(tmp_data_dir, client):
    # Prove the fall-through path invokes enrich_5xx_from_db: seed a last_error on
    # the model row and assert the hint lands in the returned envelope. This only
    # runs on the non-stream fall-through (routes.py ~666-688), so its presence is
    # positive proof the re-aggregation branch was bypassed for the error response.
    plaintext = _prep(client, tmp_data_dir, runaway_mode="log")
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET last_error=? WHERE id='qwen'", ("vLLM warmup probe failed",))
        db.commit()
    err = {"error": {"message": "engine OOM", "type": "internal_server_error"}}
    send, _ = _capturing_send(_fake_error_resp(500, err))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 500
    out = r.json()
    assert out["error"]["message"] == "engine OOM"
    assert out["error"]["hint"]["last_error"] == "vLLM warmup probe failed"


def test_enforce_trip_writes_incident_when_allowlisted(tmp_data_dir, client):
    log_path = tmp_data_dir / "content.jsonl"
    plaintext = _prep(
        client, tmp_data_dir, runaway_mode="enforce",
        runaway_hard_max=3, runaway_think_budget=10**9, runaway_repeat_max=10**9,
        content_log_enabled=True, content_log_tokens=frozenset({"tok1"}),
        content_log_path=log_path, content_log_max_chars=40000,
    )
    chunks = [_chat_chunk("x", role="assistant")] + [_chat_chunk("x") for _ in range(20)]
    send, _ = _capturing_send(_fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["finish_reason"] == "runaway_hard"
    assert lines[0]["token_id"] == "tok1"
