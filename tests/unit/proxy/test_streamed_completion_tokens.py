"""R11: streamed completion-token accounting.

The end-of-stream count must include the channels ``_parse_sse_delta`` does
not return — ``reasoning_content`` (the ``--reasoning-parser`` channel) and
tool-call ``function.name`` / ``function.arguments`` deltas — so an agent
that emits almost only those two is not metered as ~1 completion token per
request. When the stream itself carries ``usage.completion_tokens``
(``stream_options.include_usage``), that exact number must be recorded
instead of the tokenizer estimate. The forwarded bytes are untouched either
way.
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
    """Drive the streaming forward path; returns the bytes the client received."""

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
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            body = b"".join(r.iter_bytes())
            assert r.status_code == 200
    return body


def test_reasoning_and_tool_call_deltas_are_counted(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"step one"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"step two"}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"Paris\\"}"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"Sunny"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = _run_stream(client, plaintext, chunks)
    # Accounting must never touch the bytes forwarded to the client.
    assert body == b"".join(chunks)

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    # char count of "step one" + "step two" + "get_weather" + '{"city":"Paris"}' + "Sunny"
    assert rows[0]["completion_tokens"] == 48


def test_reasoning_only_stream_is_not_zero(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    # The prod shape: thinking, then an empty content delta, then stop.
    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking hard"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":""}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    _run_stream(client, plaintext, chunks)

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 13


def test_reasoning_field_deltas_are_counted(tmp_data_dir, client):
    """R22: newer vLLM names the channel ``reasoning`` in the delta (no
    ``reasoning_content`` key) — the prod shape found on
    qwen3.8-27b-fp8-model, 2026-09-27. It must count exactly like
    ``reasoning_content`` does."""
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    chunks = [
        b'data: {"choices":[{"delta":{"reasoning":"thinking hard"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":""}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = _run_stream(client, plaintext, chunks)
    assert body == b"".join(chunks)

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 13


def test_upstream_usage_frame_wins_over_estimate(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    # include_usage stream: vLLM carries "usage": null on the token frames and
    # the real usage block on the final frame with empty choices.
    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"step one"}}],"usage":null}\n\n',
        b'data: {"choices":[{"delta":{"content":"Sunny"}}],"usage":null}\n\n',
        b'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":777,"total_tokens":788}}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = _run_stream(client, plaintext, chunks)
    assert body == b"".join(chunks)

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 777


def test_parse_sse_accounting_contract():
    from app.proxy.routes import _parse_sse_accounting

    # Non-data lines, [DONE], and malformed payloads parse to nothing.
    assert _parse_sse_accounting(b"event: x") == (None, "")
    assert _parse_sse_accounting(b"data: [DONE]") == (None, "")
    assert _parse_sse_accounting(b"data: {not json") == (None, "")
    # The usage frame: empty choices, the engine's own count.
    assert _parse_sse_accounting(
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":777,"total_tokens":780}}'
    ) == (777, "")
    # vLLM's token frames carry "usage": null — that is not a count.
    assert _parse_sse_accounting(
        b'data: {"choices":[{"delta":{"content":"hi"}}],"usage":null}'
    ) == (None, "")
    # A null/absent usage on a reasoning frame, and the reasoning channel.
    assert _parse_sse_accounting(
        b'data: {"choices":[{"delta":{"reasoning_content":"step one"}}]}'
    ) == (None, "step one")
    # R22: newer vLLM names the same channel `reasoning`.
    assert _parse_sse_accounting(b'data: {"choices":[{"delta":{"reasoning":"The"}}]}') == (
        None,
        "The",
    )
    # Both names present: the traditional `reasoning_content` wins.
    assert _parse_sse_accounting(
        b'data: {"choices":[{"delta":{"reasoning_content":"a","reasoning":"b"}}]}'
    ) == (None, "a")
    # Tool-call deltas: function name + arguments, in order.
    assert _parse_sse_accounting(
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{\\"a\\":1}"}}]}}]}'
    ) == (None, 'f{"a":1}')
    # A non-chat frame (completions `text`) carries no delta — not counted here.
    assert _parse_sse_accounting(b'data: {"choices":[{"text":"hi"}]}') == (None, "")


def test_content_only_stream_counts_as_before(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}],"model":"qwen"}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}],"model":"qwen"}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = _run_stream(client, plaintext, chunks)
    assert body == b"".join(chunks)

    flush_ledger(client)
    rows = _read_counters(tmp_data_dir / "vllm-warden.db")
    assert len(rows) == 1
    # "hello world" -> 11 chars: identical to the pre-R11 behaviour
    # (content-only streams tokenize the same text as before).
    assert rows[0]["completion_tokens"] == 11
    assert rows[0]["prompt_tokens"] == 2
