"""R13: streamed ``/v1/completions`` completion-token accounting.

The legacy completions endpoint streams ``choices[0].text`` chunks, not
``choices[0].delta.content`` — the channel the streaming forward path
parses for chat. Without the ``text`` channel, ``accumulated`` stays
empty and the end-of-stream count records 0 completion tokens for every
streamed legacy-completions request. The non-streamed path reads
``usage.completion_tokens`` from the JSON body and is the control here,
as is the byte-identity of the forwarded stream.
"""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token
from tests.unit.proxy.ledger_helpers import flush_ledger


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


def _make_fake_tokenizer(token_count_for):
    cache = MagicMock()
    cache.count = AsyncMock(
        side_effect=lambda repo, text, *, fallback_repo=None: token_count_for(text)
    )
    return cache


def _run_stream(client, plaintext, chunks):
    """Drive the streaming forward path over /v1/completions; returns the
    bytes the client received."""

    async def aiter():
        for c in chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST",
            "/v1/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "prompt": "hi", "stream": True},
        ) as r:
            body = b"".join(r.iter_bytes())
            assert r.status_code == 200
    return body


def test_streamed_completions_text_deltas_are_counted(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    # vLLM's streamed legacy-completions shape: choices[0].text per frame,
    # the terminal frame carrying an empty text + finish_reason.
    chunks = [
        b'data: {"id":"cmpl-1","object":"text_completion","created":1,"model":"qwen","choices":[{"index":0,"text":"Hello","logprobs":null,"finish_reason":null,"stop_reason":null}]}\n\n',
        b'data: {"id":"cmpl-1","object":"text_completion","created":1,"model":"qwen","choices":[{"index":0,"text":" world","logprobs":null,"finish_reason":null,"stop_reason":null}]}\n\n',
        b'data: {"id":"cmpl-1","object":"text_completion","created":1,"model":"qwen","choices":[{"index":0,"text":"","logprobs":null,"finish_reason":"stop","stop_reason":null}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = _run_stream(client, plaintext, chunks)
    # Accounting must never touch the bytes forwarded to the client.
    assert body == b"".join(chunks)

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    # char count of "Hello" + " world" — the streamed text, not 0.
    assert rows[0]["completion_tokens"] == 11
    assert rows[0]["prompt_tokens"] == 2


def test_nonstreamed_completions_uses_usage_block(tmp_data_dir, client):
    """Control: the non-streamed /v1/completions path reads the JSON
    body's usage block and is unchanged by the streamed-text fix."""
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: 0)

    resp_body = {
        "id": "cmpl-1",
        "model": "qwen",
        "choices": [{"index": 0, "text": "Hello world", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 42, "total_tokens": 45},
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=json.dumps(resp_body).encode())
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        r = client.post(
            "/v1/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "prompt": "hi"},
        )
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == "Hello world"

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 42


def test_parse_sse_text_contract():
    from app.proxy.routes import _parse_sse_text

    # Non-data lines, [DONE], and malformed payloads parse to nothing.
    assert _parse_sse_text(b"event: x") is None
    assert _parse_sse_text(b"data: [DONE]") is None
    assert _parse_sse_text(b"data: {not json") is None
    # The legacy-completions text channel.
    assert _parse_sse_text(b'data: {"choices":[{"text":"hi"}]}') == "hi"
    # Chat frames carry no text channel.
    assert _parse_sse_text(b'data: {"choices":[{"delta":{"content":"hi"}}]}') is None
    # The empty text on the terminal frame is not a delta.
    assert _parse_sse_text(b'data: {"choices":[{"text":"","finish_reason":"stop"}]}') is None
