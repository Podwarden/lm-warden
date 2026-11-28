"""Prompt Lens taps in ``_forward``.

Two contracts:
  * OFF by default (production safety): with ``prompt_lens_enabled=false`` the hub
    is NEVER called and the streamed response bytes are byte-identical to today.
  * ON: a streaming request emits request_start → delta(content) +
    delta(reasoning) → request_end (with the tracked finish_reason); a
    non-stream request emits request_start → delta(s) → request_end.
"""

import asyncio
import dataclasses
import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token
from app.proxy.routes import _gm_prompt_captures


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


def _enable_prompt_lens(client):
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )


# --------------------------------------------------------------------------
# Production-safety regression: disabled → zero hub calls + byte-identical body
# --------------------------------------------------------------------------
def test_disabled_makes_zero_hub_calls_and_byte_identical_stream(tmp_data_dir, client):
    client.get("/healthz")
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=False
    )
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)

    # Spy hub in place of the real one. prompt_lens_enabled is false by default,
    # so the gate must never touch it.
    spy = MagicMock()
    client.app.state.prompt_lens_hub = spy

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}],"model":"qwen"}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}],"model":"qwen"}\n\n',
        b"data: [DONE]\n\n",
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
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
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
    _enable_prompt_lens(client)

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
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
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    hub = client.app.state.prompt_lens_hub
    _q, snap = hub.subscribe()
    types = [e["type"] for e in snap]
    assert types[0] == "request_start"
    assert types[-1] == "request_end"

    start = snap[0]
    # Attribution (stats redesign B11): the capture names the request it belongs
    # to -- the same id as the live row and the history row.
    deadline = time.time() + 30
    ids: list = []
    while time.time() < deadline and not ids:
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            ids = [r[0] for r in db.execute("SELECT id FROM request_history")]
        time.sleep(0.05)
    assert ids == [start["request_id"]]
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


def test_enabled_streaming_emits_reasoning_field_deltas(tmp_data_dir, client):
    """R22: `delta.reasoning` (the newer vLLM field name) must surface on the
    reasoning channel exactly like `reasoning_content`."""
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    _enable_prompt_lens(client)

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
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
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    hub = client.app.state.prompt_lens_hub
    _q, snap = hub.subscribe()
    content = [e for e in snap if e["type"] == "delta" and e["channel"] == "content"]
    reasoning = [e for e in snap if e["type"] == "delta" and e["channel"] == "reasoning"]
    assert "".join(e["text"] for e in content) == "hello"
    assert [e["text"] for e in reasoning] == ["thinking"]


# --------------------------------------------------------------------------
# Enabled — non-stream: request_start → one delta per channel → request_end
# --------------------------------------------------------------------------
def test_enabled_nonstream_emits_start_delta_end(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 5 if "hi" in t else 0)
    _enable_prompt_lens(client)

    body = {
        "id": "x",
        "model": "qwen",
        "choices": [
            {
                "message": {"content": "answer", "reasoning_content": "because"},
                "finish_reason": "stop",
            }
        ],
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

    hub = client.app.state.prompt_lens_hub
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


def test_enabled_nonstream_emits_reasoning_field_delta(tmp_data_dir, client):
    """R22: a non-stream body that names the channel `reasoning` in the
    message (the newer vLLM field name) is published on the reasoning
    channel."""
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 5 if "hi" in t else 0)
    _enable_prompt_lens(client)

    body = {
        "id": "x",
        "model": "qwen",
        "choices": [
            {"message": {"content": "answer", "reasoning": "because"}, "finish_reason": "stop"}
        ],
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

    hub = client.app.state.prompt_lens_hub
    _q, snap = hub.subscribe()
    content = [e for e in snap if e["type"] == "delta" and e["channel"] == "content"]
    reasoning = [e for e in snap if e["type"] == "delta" and e["channel"] == "reasoning"]
    assert [e["text"] for e in content] == ["answer"]
    assert [e["text"] for e in reasoning] == ["because"]


# --------------------------------------------------------------------------
# G1: request_start carries the structured prompt, the session and the turn
# --------------------------------------------------------------------------
def _stream_once(client, plaintext, body, headers=None):
    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
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
            headers={"Authorization": f"Bearer {plaintext}", **(headers or {})},
            json=body,
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200


_SESSION = "5f0c2a9e-7d41-4b8e-9c3a-1e2f3a4b5c6d"


def test_request_start_carries_prompt_parts_session_and_turn(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    _enable_prompt_lens(client)

    body = {
        "model": "qwen",
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "the last user text"},
        ],
    }
    _stream_once(client, plaintext, body, {"X-Session-Id": _SESSION})
    _stream_once(client, plaintext, body, {"X-Session-Id": _SESSION})

    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    starts = [e for e in snap if e["type"] == "request_start"]
    assert len(starts) == 2
    first, second = starts
    parts = first["prompt_parts"]
    assert parts["messages"][-1]["text"] == "the last user text"
    assert parts["messages"][-1]["i"] == 2
    assert parts["sys"]["text"] == "be brief"
    assert parts["tools"]["names"] == ["read_file"]
    assert first["session_id"] == _SESSION
    assert (first["turn"], second["turn"]) == (0, 1)
    # The flat prompt stays for one release (scripts read it).
    assert "the last user text" in first["prompt"]
    assert first["prompt_elided"] is False


def test_request_start_without_session_has_null_session_and_turn(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    _enable_prompt_lens(client)

    _stream_once(
        client,
        plaintext,
        {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    start = next(e for e in snap if e["type"] == "request_start")
    assert "session_id" in start and "turn" in start
    assert start["prompt_parts"]["messages"] == [
        {"i": 0, "role": "user", "chars": 2, "text": "hi", "elided": False}
    ]


def test_disabled_never_builds_the_structured_prompt(tmp_data_dir, client):
    client.get("/healthz")
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=False
    )
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    spy_hub = MagicMock()
    client.app.state.prompt_lens_hub = spy_hub

    with patch("app.proxy.routes.structured_prompt") as spy:
        _stream_once(
            client,
            plaintext,
            {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert spy.call_count == 0
    assert spy_hub.method_calls == []


def test_big_body_is_captured_off_the_loop_and_windowed(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 1)
    _enable_prompt_lens(client)

    big = "q" * 300_000  # over the 256 KiB thread bound
    with patch("app.proxy.routes.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread:
        _stream_once(
            client,
            plaintext,
            {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": big}]},
        )
    # The capture ran on a worker thread (patching ``asyncio.to_thread`` is
    # global, so other workers show up too; only ours is looked for).
    assert any(
        getattr(c.args[0], "func", None) is _gm_prompt_captures
        for c in to_thread.call_args_list
        if c.args
    )
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    start = next(e for e in snap if e["type"] == "request_start")
    last = start["prompt_parts"]["messages"][-1]
    assert last["chars"] == 300_000 and last["elided"] is True
    assert len(last["text"]) < 6100


def test_the_section_store_gets_the_wide_prompt_and_the_reply(tmp_data_dir, client):
    """G9: the request's sections reach the store; tool results are whole up
    to the section budget, not the stream event's 6000-char window."""
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 1)
    _enable_prompt_lens(client)
    result = "r" * 20_000
    body = {
        "model": "qwen",
        "stream": True,
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": result},
        ],
    }
    _stream_once(client, plaintext, body)
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    start = next(e for e in snap if e["type"] == "request_start")
    narrow = start["prompt_parts"]["messages"][-1]
    assert narrow["elided"] is True and len(narrow["text"]) < 6100
    store = client.app.state.prompt_lens_sections
    tr = store.section(start["req_id"], "tool_results")
    assert tr is not None and tr.token_id == "tok1"
    assert [m["text"] for m in tr.messages] == [result]
    reply = store.section(start["req_id"], "reply")
    assert reply is not None and reply.text == "ok" and reply.live is False


def test_a_keeping_key_has_its_request_kept_on_disk(tmp_data_dir, client):
    """keep_captures (G-R4): the proxy marks the request from the key's row, so
    its record reaches prompt_lens_kept; a key without the switch keeps nothing."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 1)
    _enable_prompt_lens(client)
    body = {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "keep me"}]}
    _stream_once(client, plaintext, body)
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET keep_captures = 1 WHERE id = 'tok1'")
    from app.proxy.token_cache import invalidate_all

    invalidate_all()
    _stream_once(client, plaintext, body)
    client.portal.call(client.app.state.prompt_lens_kept.flush)
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT token_id, reply, prompt, events FROM prompt_lens_kept").fetchall()
    assert len(rows) == 1 and rows[0][0] == "tok1"
    assert rows[0][1] == "ok"
    assert json.loads(rows[0][2])["messages"][-1]["text"] == "keep me"
    events = json.loads(rows[0][3])
    assert events[0]["type"] == "request_start" and events[-1]["type"] == "request_end"
    assert "prompt_parts" not in events[0]


def test_disabled_puts_nothing_in_the_section_store(tmp_data_dir, client):
    client.get("/healthz")
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=False
    )
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: 1)
    spy = MagicMock()
    client.app.state.prompt_lens_sections = spy
    _stream_once(
        client,
        plaintext,
        {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert spy.method_calls == []


# --------------------------------------------------------------------------
# G2: tool-call deltas, streamed and non-streamed, on all three wire APIs
# --------------------------------------------------------------------------
def _ready_gm(client, tmp_data_dir, *, on=True):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer(lambda t: len(t.split()) if t else 0)
    if on:
        _enable_prompt_lens(client)
    else:
        client.app.state.settings = dataclasses.replace(
            client.app.state.settings, prompt_lens_enabled=False
        )
    return plaintext


def _sse(events):
    return [b"data: " + json.dumps(ev).encode() + b"\n\n" for ev in events] + [b"data: [DONE]\n\n"]


def _sse_resp(chunks):
    async def aiter():
        for c in chunks:
            yield c

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _json_resp(body):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def _tc(index, args=None, call_id=None, name=None):
    tc: dict = {"index": index, "function": {}}
    if call_id is not None:
        tc["id"] = call_id
        tc["type"] = "function"
    if name is not None:
        tc["function"]["name"] = name
    if args is not None:
        tc["function"]["arguments"] = args
    return tc


def _frame(delta=None, finish=None, usage=None):
    ev: dict = {"id": "c1", "object": "chat.completion.chunk", "model": "qwen", "choices": []}
    if delta is not None or finish is not None:
        ev["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        ev["usage"] = usage
    return ev


#: Two parallel calls whose arguments interleave: call 0 keeps getting
#: fragments after call 1 has started (G-F4: a later index is not "done").
_INTERLEAVED = [
    _frame({"role": "assistant", "content": ""}),
    _frame({"tool_calls": [_tc(0, "", "call_a", "read_file")]}),
    _frame({"tool_calls": [_tc(0, '{"pa')]}),
    _frame({"tool_calls": [_tc(1, "", "call_b", "grep")]}),
    _frame({"tool_calls": [_tc(1, '{"q": "x"')]}),
    _frame({"tool_calls": [_tc(0, 'th": "a.py"}'), _tc(1, "}")]}),
    _frame({}, finish="tool_calls"),
    _frame(usage={"prompt_tokens": 3, "completion_tokens": 9}),
]

#: What the interleaved stream must publish, in order: (index, call_id, name, text).
_INTERLEAVED_DELTAS = [
    (0, "call_a", "read_file", ""),
    (0, None, None, '{"pa'),
    (1, "call_b", "grep", ""),
    (1, None, None, '{"q": "x"'),
    (0, None, None, 'th": "a.py"}'),
    (1, None, None, "}"),
]


def _tool_deltas(snap):
    return [e for e in snap if e["type"] == "delta" and e["channel"] == "tool_call"]


def _as_tuples(deltas):
    return [(e["index"], e.get("call_id"), e.get("name"), e["text"]) for e in deltas]


def _stream_path(client, plaintext, path, body, chunks):
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_sse_resp(chunks))):
        with client.stream(
            "POST", path, headers={"Authorization": f"Bearer {plaintext}"}, json=body
        ) as r:
            raw = b"".join(r.iter_bytes())
            assert r.status_code == 200, raw
    return raw


def _post_path(client, plaintext, path, body, upstream_body):
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_json_resp(upstream_body))):
        r = client.post(path, headers={"Authorization": f"Bearer {plaintext}"}, json=body)
    assert r.status_code == 200, r.text
    return r


_CHAT = {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "go"}]}


def test_streaming_tool_calls_publish_tool_call_deltas(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    chunks = _sse(_INTERLEAVED)
    raw = _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, chunks)
    # No change to the bytes the client receives.
    assert raw == b"".join(chunks)

    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    deltas = _tool_deltas(snap)
    assert _as_tuples(deltas) == _INTERLEAVED_DELTAS
    # call_id/name only on the fragment that carries them; no key at all otherwise.
    for e in deltas[1:2] + deltas[3:]:
        assert "call_id" not in e and "name" not in e
    assert all(e["req_id"] == snap[0]["req_id"] and e["token_id"] == "tok1" for e in deltas)
    assert all(isinstance(e["ts"], float) for e in deltas)
    assert all("elided" not in e for e in deltas)
    # Reassembled per index, both calls' arguments are whole JSON.
    args = {0: "", 1: ""}
    for e in deltas:
        args[e["index"]] += e["text"]
    assert json.loads(args[0]) == {"path": "a.py"} and json.loads(args[1]) == {"q": "x"}
    # Every tool delta precedes request_end, which carries the raw finish reason
    # and no drop counters (nothing was dropped).
    end = snap[-1]
    assert end["type"] == "request_end" and end["finish_reason"] == "tool_calls"
    assert "tool_call_chars_dropped" not in end and "tool_calls_dropped" not in end
    # The empty content fragment is not published as a content delta.
    assert not [e for e in snap if e["type"] == "delta" and e["channel"] == "content"]


def test_streaming_tool_calls_with_content_and_reasoning(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    events = [
        _frame({"reasoning_content": "think"}),
        _frame({"content": "Let me look."}),
        _frame({"tool_calls": [_tc(0, '{"a":1}', "call_x", "ls")]}),
        _frame({}, finish="tool_calls"),
    ]
    _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, _sse(events))
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    seq = [(e["channel"], e["text"]) for e in snap if e["type"] == "delta"]
    assert seq == [("reasoning", "think"), ("content", "Let me look."), ("tool_call", '{"a":1}')]
    assert _as_tuples(_tool_deltas(snap)) == [(0, "call_x", "ls", '{"a":1}')]


def test_nonstream_tool_calls_publish_one_delta_per_call(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    upstream = {
        "id": "x",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "grep", "arguments": '{"q": "x"}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
    }
    r = _post_path(
        client,
        plaintext,
        "/v1/chat/completions",
        {"model": "qwen", "messages": [{"role": "user", "content": "go"}]},
        upstream,
    )
    assert r.json() == upstream  # client body untouched

    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == [
        (0, "call_a", "read_file", '{"path": "a.py"}'),
        (1, "call_b", "grep", '{"q": "x"}'),
    ]
    assert not [e for e in snap if e["type"] == "delta" and e["channel"] == "content"]
    types = [e["type"] for e in snap]
    assert types[-1] == "request_end" and snap[-1]["finish_reason"] == "tool_calls"
    assert types.index("request_end") > max(i for i, t in enumerate(types) if t == "delta")


def test_nonstream_tool_call_odd_shapes(tmp_data_dir, client):
    """Arguments as an object are dumped to JSON; a call without arguments has
    text ""; a non-dict entry is skipped but keeps its position as index."""
    plaintext = _ready_gm(client, tmp_data_dir)
    upstream = {
        "id": "x",
        "model": "qwen",
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [
                        {"id": "c0", "function": {"name": "a", "arguments": {"k": 1}}},
                        "junk",
                        {"id": "c2", "function": {"name": "b"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    _post_path(
        client,
        plaintext,
        "/v1/chat/completions",
        {"model": "qwen", "messages": [{"role": "user", "content": "go"}]},
        upstream,
    )
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == [(0, "c0", "a", '{"k":1}'), (2, "c2", "b", "")]
    assert [e["text"] for e in snap if e["type"] == "delta" and e["channel"] == "content"] == ["ok"]


def test_streaming_tool_call_capture_is_bounded_and_counts_the_drop(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_section_prompt_chars=20
    )
    events = [
        _frame({"tool_calls": [_tc(0, "", "c1", "ab")]}),  # 4 chars
        _frame({"tool_calls": [_tc(0, "0123456789")]}),  # 10 -> 14
        _frame({"tool_calls": [_tc(0, "abcdefghij")]}),  # cut to 6 -> 20
        _frame({"tool_calls": [_tc(0, "ZZZ"), _tc(1, "", "c2", "cd")]}),  # dropped
        _frame({"tool_calls": [_tc(1, "{}")]}),  # dropped
        _frame({}, finish="tool_calls"),
    ]
    chunks = _sse(events)
    raw = _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, chunks)
    assert raw == b"".join(chunks)  # the client still gets everything
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    deltas = _tool_deltas(snap)
    assert _as_tuples(deltas) == [
        (0, "c1", "ab", ""),
        (0, None, None, "0123456789"),
        (0, None, None, "abcdef"),
    ]
    assert deltas[-1]["elided"] is True
    end = snap[-1]
    assert end["type"] == "request_end"
    assert end["tool_call_chars_dropped"] == 4 + 3 + 2
    assert end["tool_calls_dropped"] == 1


def test_nonstream_tool_call_capture_is_bounded_and_counts_the_drop(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_section_prompt_chars=12
    )
    calls = [{"id": f"c{i}", "function": {"name": "n", "arguments": "x" * 8}} for i in range(3)]
    upstream = {
        "id": "x",
        "model": "qwen",
        "choices": [{"message": {"tool_calls": calls}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    r = _post_path(
        client,
        plaintext,
        "/v1/chat/completions",
        {"model": "qwen", "messages": [{"role": "user", "content": "go"}]},
        upstream,
    )
    assert r.json() == upstream
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    deltas = _tool_deltas(snap)
    # c0: 2 + 1 + 8 = 11; c1's opener (3) does not fit: dropped whole, as is c2.
    assert _as_tuples(deltas) == [(0, "c0", "n", "x" * 8)]
    end = snap[-1]
    assert end["tool_call_chars_dropped"] == 16 and end["tool_calls_dropped"] == 2


def test_disabled_tool_call_stream_makes_zero_hub_calls_and_identical_bytes(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir, on=False)
    spy = MagicMock()
    client.app.state.prompt_lens_hub = spy
    chunks = _sse(_INTERLEAVED)
    with patch("app.proxy.routes._parse_sse_prompt_lens") as parse:
        raw = _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, chunks)
    assert raw == b"".join(chunks)
    assert parse.call_count == 0
    assert spy.method_calls == []


def test_disabled_nonstream_tool_calls_make_zero_hub_calls(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir, on=False)
    spy = MagicMock()
    client.app.state.prompt_lens_hub = spy
    upstream = {
        "id": "x",
        "model": "qwen",
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "c", "function": {"name": "n", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    r = _post_path(
        client,
        plaintext,
        "/v1/chat/completions",
        {"model": "qwen", "messages": [{"role": "user", "content": "go"}]},
        upstream,
    )
    assert r.json() == upstream
    assert spy.method_calls == []


_TOOLS_ANTHROPIC = [
    {"name": "read_file", "input_schema": {"type": "object"}},
    {"name": "grep", "input_schema": {"type": "object"}},
]


def test_messages_stream_publishes_tool_call_deltas(tmp_data_dir, client):
    """Anthropic Messages: the route translates to chat, so the tap sees the
    engine's chat frames and publishes the same deltas."""
    plaintext = _ready_gm(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "stream": True,
        "tools": _TOOLS_ANTHROPIC,
        "messages": [{"role": "user", "content": "go"}],
    }
    raw = _stream_path(client, plaintext, "/v1/messages", body, _sse(_INTERLEAVED))
    assert b"tool_use" in raw  # the client got the Anthropic stream
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == _INTERLEAVED_DELTAS
    assert snap[-1]["type"] == "request_end" and snap[-1]["finish_reason"] == "tool_calls"


def test_messages_nonstream_publishes_one_delta_per_call(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    upstream = {
        "id": "x",
        "model": "qwen",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_a", "function": {"name": "read_file", "arguments": '{"p":1}'}},
                        {"id": "call_b", "function": {"name": "grep", "arguments": '{"q":2}'}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
    }
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "tools": _TOOLS_ANTHROPIC,
        "messages": [{"role": "user", "content": "go"}],
    }
    r = _post_path(client, plaintext, "/v1/messages", body, upstream)
    assert [b["type"] for b in r.json()["content"]] == ["tool_use", "tool_use"]
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == [
        (0, "call_a", "read_file", '{"p":1}'),
        (1, "call_b", "grep", '{"q":2}'),
    ]


_TOOLS_RESPONSES = [
    {"type": "function", "name": "read_file"},
    {"type": "function", "name": "grep"},
]


def test_responses_stream_publishes_tool_call_deltas(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    body = {"model": "qwen", "input": "go", "stream": True, "tools": _TOOLS_RESPONSES}
    raw = _stream_path(client, plaintext, "/v1/responses", body, _sse(_INTERLEAVED))
    assert b"function_call" in raw
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == _INTERLEAVED_DELTAS
    assert snap[-1]["type"] == "request_end" and snap[-1]["finish_reason"] == "tool_calls"


def test_responses_nonstream_publishes_one_delta_per_call(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    upstream = {
        "id": "x",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_a", "function": {"name": "read_file", "arguments": '{"p":1}'}},
                        {"id": "call_b", "function": {"name": "grep", "arguments": '{"q":2}'}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
    }
    body = {"model": "qwen", "input": "go", "tools": _TOOLS_RESPONSES}
    r = _post_path(client, plaintext, "/v1/responses", body, upstream)
    assert [o["type"] for o in r.json()["output"]] == ["function_call", "function_call"]
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == [
        (0, "call_a", "read_file", '{"p":1}'),
        (1, "call_b", "grep", '{"q":2}'),
    ]


def test_streaming_tool_call_frame_split_across_chunks(tmp_data_dir, client):
    """Review M3: one SSE frame arrives in two network chunks (split inside
    the JSON); the tap publishes it once, whole, after reassembly."""
    plaintext = _ready_gm(client, tmp_data_dir)
    frames = _sse(_INTERLEAVED)
    whole = b"".join(frames)
    target = frames[2]  # the '{"pa' fragment of call 0
    cut = whole.index(target) + len(target) // 2
    chunks = [whole[:cut], whole[cut:]]
    raw = _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, chunks)
    assert raw == whole
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert _as_tuples(_tool_deltas(snap)) == _INTERLEAVED_DELTAS


def test_streaming_dropped_opener_publishes_no_orphans(tmp_data_dir, client):
    """Review I1 end to end: the budget cannot hold call 1's opener, so neither
    call 1's arguments nor call 0's later fragments are published."""
    plaintext = _ready_gm(client, tmp_data_dir)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_section_prompt_chars=20
    )
    raw = _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, _sse(_INTERLEAVED))
    assert raw == b"".join(_sse(_INTERLEAVED))
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    # call_a + read_file = 15, '{"pa' = 4 -> 1 left; call_b + grep (10) does not
    # fit, so the capture ends there although 1 char of budget is left.
    deltas = _tool_deltas(snap)
    assert _as_tuples(deltas) == [(0, "call_a", "read_file", ""), (0, None, None, '{"pa')]
    assert all("elided" not in e for e in deltas)
    end = snap[-1]
    assert end["tool_calls_dropped"] == 1
    assert end["tool_call_chars_dropped"] == len('{"q": "x"') + len('th": "a.py"}') + 1


# --------------------------------------------------------------------------
# G3: the per-request kvsig parts event
# --------------------------------------------------------------------------
_KV_BODY = {
    "model": "qwen",
    "stream": True,
    "tools": [{"type": "function", "function": {"name": "read_file"}}],
    "messages": [
        {"role": "system", "content": "be brief " * 40},
        {"role": "user", "content": "first question " * 40},
        {"role": "assistant", "content": "first answer " * 40},
        {"role": "user", "content": "the last user text " * 40},
    ],
}


def _kvsigs(snap):
    return [e for e in snap if e["type"] == "kvsig"]


def _second_key(tmp_data_dir):
    plaintext = "vw_secondtoken1234567890abcdef1234"
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok2", "other-token", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
    return plaintext


def test_kvsig_event_published_after_admission(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _stream_once(client, plaintext, _KV_BODY, {"X-Session-Id": _SESSION})
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    kv = _kvsigs(snap)
    assert len(kv) == 1
    types = [e["type"] for e in snap]
    first_delta = types.index("delta")
    assert types.index("request_start") < types.index("kvsig") < first_delta
    ev = kv[0]
    start = snap[types.index("request_start")]
    assert ev["req_id"] == start["req_id"] and ev["token_id"] == "tok1"
    assert ev["session_id"] == _SESSION and ev["turn"] == 0
    assert ev["available"] is True and ev["profile"] == "loose/1"
    assert ev["n_nodes"] == 3 and ev["edge"] == "new"
    assert [(p["p"], p["from"], p["to"], p["state"]) for p in ev["parts"]] == [
        ("sys", -1, -1, "new"),
        ("tools", -1, -1, "new"),
        ("nodes", 0, 0, "new"),
        ("nodes", 1, 1, "new"),
        ("nodes", 2, 2, "new"),
    ]
    sizes = [m["chars"] for m in start["prompt_parts"]["messages"]]
    assert [p["chars"] for p in ev["parts"][2:]] == sizes
    assert ev["parts"][0]["chars"] == start["prompt_parts"]["sys"]["chars"]
    assert ev["parts"][1]["chars"] == start["prompt_parts"]["tools"]["chars"]
    assert ev["parts"][2]["roles"] == {"user": 1}
    # No content on the event: ids and sizes only.
    assert "first question" not in json.dumps(ev) and "be brief" not in json.dumps(ev)

    # The same body again, same key and session: the whole prefix is known.
    _stream_once(client, plaintext, _KV_BODY, {"X-Session-Id": _SESSION})
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    again = _kvsigs(snap)[-1]
    assert again["turn"] == 1 and again["exact_depth"] == 2
    assert [(p["p"], p["from"], p["to"], p["state"]) for p in again["parts"]] == [
        ("sys", -1, -1, "exact"),
        ("tools", -1, -1, "exact"),
        ("nodes", 0, 2, "exact"),
    ]
    assert again["reusable_own"] == again["reusable_any"] and again["reusable_own"] > 0


def test_kvsig_carries_the_admission_wait(tmp_data_dir, client):
    """G-R14 (c): the dock shows "queued" from admission on, not only at the end."""
    plaintext = _ready_gm(client, tmp_data_dir)
    _stream_once(client, plaintext, _KV_BODY, {"X-Session-Id": _SESSION})
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    kv = _kvsigs(snap)[-1]
    end = [e for e in snap if e["type"] == "request_end"][-1]
    assert isinstance(kv["queued_s"], float) and kv["queued_s"] >= 0
    assert kv["queued_s"] == end["queued_s"]
    # A key holder may see their own wait.
    from app.proxy.prompt_lens_scope import for_key_holder

    assert for_key_holder(kv)["queued_s"] == kv["queued_s"]


def test_kvsig_marks_another_keys_prefix_other(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    other = _second_key(tmp_data_dir)
    _stream_once(client, plaintext, _KV_BODY)
    _stream_once(client, other, _KV_BODY)
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    ev = _kvsigs(snap)[-1]
    assert ev["token_id"] == "tok2"
    # Its own lineage knows nothing (per key); key 1 sent every byte. Tools
    # end inside the first chunk, which this key might have sent itself.
    assert [p["state"] for p in ev["parts"]] == ["other", "new", "other", "other", "other"]
    assert ev["reusable_own"] == 0 and ev["reusable_any"] > 0
    # The other key is never named.
    assert "tok1" not in json.dumps(ev) and "my-token" not in json.dumps(ev)


def test_kvsig_unavailable_for_legacy_completions(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    chunks = [b'data: {"choices":[{"text":"ok","finish_reason":"stop"}]}\n\n', b"data: [DONE]\n\n"]
    _stream_path(
        client,
        plaintext,
        "/v1/completions",
        {"model": "qwen", "stream": True, "prompt": "hi"},
        chunks,
    )
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    kv = _kvsigs(snap)
    assert len(kv) == 1
    assert kv[0]["available"] is False and kv[0]["reason"] == "not_chat"
    assert kv[0]["parts"] == []


def test_kvsig_over_cap_and_no_key(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    monkeypatch.setattr("app.cache_obs.canonical.MAX_CHAIN_BYTES", 64)
    _stream_once(client, plaintext, _KV_BODY)
    monkeypatch.undo()
    client.app.state.cobs2_key = None
    _stream_once(client, plaintext, _KV_BODY)
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert [(e["available"], e["reason"]) for e in _kvsigs(snap)] == [
        (False, "over_cap"),
        (False, "no_key"),
    ]


def test_kvsig_layout_runs_in_the_cobs2_worker_for_a_big_body(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    from app.proxy import routes

    big = {
        "model": "qwen",
        "stream": True,
        "messages": [{"role": "user", "content": "q" * 300_000}],
    }
    with patch("app.proxy.routes.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread:
        _stream_once(client, plaintext, big)
    calls = [c for c in to_thread.call_args_list if c.args and c.args[0] is routes._cobs2_compute]
    assert len(calls) == 1 and calls[0].args[-1] is True
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    ev = _kvsigs(snap)[-1]
    assert ev["available"] is True and ev["parts"][-1]["chars"] == 300_000


def test_disabled_never_builds_kvsig_parts(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir, on=False)
    spy_hub = MagicMock()
    client.app.state.prompt_lens_hub = spy_hub
    with (
        patch("app.proxy.routes.body_layout") as lay,
        patch("app.proxy.routes.head_part_ids") as hp,
        patch("app.proxy.routes.build_kvsig") as bk,
    ):
        _stream_once(client, plaintext, _KV_BODY)
    assert (lay.call_count, hp.call_count, bk.call_count) == (0, 0, 0)
    assert spy_hub.method_calls == []


def test_refused_request_publishes_no_kvsig(tmp_data_dir, client):
    from fastapi import HTTPException

    plaintext = _ready_gm(client, tmp_data_dir)

    class Refused:
        async def __aenter__(self):
            raise HTTPException(status_code=503, detail="queue full")

        async def __aexit__(self, *a):
            return False

    client.app.state.scheduler.acquire = lambda **kw: Refused()
    r = client.post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {plaintext}"}, json=_KV_BODY
    )
    assert r.status_code == 503
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert [e["type"] for e in snap if e["type"] == "request_start"] == ["request_start"]
    assert _kvsigs(snap) == []


# --------------------------------------------------------------------------
# G6: live-only per-channel progress events
# --------------------------------------------------------------------------
_PROGRESS_KEYS = {
    "type",
    "req_id",
    "token_id",
    "ts",
    "phase",
    "frames",
    "completion_tokens",
    "seq",
}


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _clocked_stream(chunks, clock, step):
    """An upstream SSE body whose fake monotonic clock advances ``step`` per chunk."""

    async def aiter():
        for c in chunks:
            clock[0] += step
            yield c

    resp = _sse_resp(chunks)
    resp.aiter_bytes = aiter
    return resp


def _run_clocked(client, plaintext, chunks, *, step=0.4, body=_CHAT):
    """Stream ``chunks`` on a fake clock; return (live events, clock at each
    progress publish, ring snapshot)."""
    hub = client.app.state.prompt_lens_hub
    clock = [time.monotonic()]
    stamps: list[float] = []
    real_publish = hub.publish

    def publish(event, **kw):
        if event.get("type") == "progress":
            stamps.append(clock[0])
        return real_publish(event, **kw)

    q, _ = hub.subscribe()
    with (
        patch.object(hub, "publish", side_effect=publish),
        patch("app.proxy.routes._monotonic", new=lambda: clock[0]),
        patch("app.proxy.routes._LIVE_UPDATE_INTERVAL_S", 0),
        patch(
            "httpx.AsyncClient.send",
            new=AsyncMock(return_value=_clocked_stream(chunks, clock, step)),
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json=body,
        ) as r:
            raw = b"".join(r.iter_bytes())
            assert r.status_code == 200, raw
    live = _drain(q)
    hub.unsubscribe(q)
    _q2, snap = hub.subscribe()
    return live, stamps, snap


_THREE_TWO = _sse(
    [
        _frame({"reasoning_content": "a"}),
        _frame({"reasoning_content": "b"}),
        _frame({"reasoning_content": "c"}),
        _frame({"content": "d"}),
        _frame({"content": "e"}),
        _frame({}, finish="stop"),
    ]
)


def test_progress_events_count_frames_per_channel(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    live, stamps, _snap = _run_clocked(client, plaintext, _THREE_TWO)
    progress = [e for e in live if e["type"] == "progress"]
    assert len(progress) >= 2
    # G-F5: the last progress is the final one, right before request_end.
    types = [e["type"] for e in live]
    assert types[-2:] == ["progress", "request_end"]
    last = progress[-1]
    assert last["frames"] == {"reasoning": 3, "content": 2, "tool_call": 0}
    assert set(last) == _PROGRESS_KEYS
    start = next(e for e in live if e["type"] == "request_start")
    assert all(p["req_id"] == start["req_id"] and p["token_id"] == "tok1" for p in progress)
    assert all(isinstance(p["phase"], str) and isinstance(p["ts"], float) for p in progress)
    # The final carries the end's completion count.
    assert last["completion_tokens"] == live[-1]["completion_tokens"]
    # The throttled ones are at most one per second of the fake clock, though
    # _LIVE_UPDATE_INTERVAL_S is 0 and a chunk arrives every 0.4 s.
    ticks = stamps[:-1]
    assert ticks and all(b - a >= 1.0 for a, b in zip(ticks, ticks[1:], strict=False))
    # Frame counts only ever grow.
    seen = [p["frames"] for p in progress]
    for a, b in zip(seen, seen[1:], strict=False):
        assert all(b[k] >= a[k] for k in a)
    # The first tick reports the first frame at once.
    assert seen[0] == {"reasoning": 1, "content": 0, "tool_call": 0}


def test_progress_is_live_only_never_in_the_ring(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    live, _stamps, snap = _run_clocked(client, plaintext, _THREE_TWO)
    assert [e for e in live if e["type"] == "progress"]
    assert not [e for e in snap if e["type"] == "progress"]
    assert [e for e in snap if e["type"] == "request_end"]


def test_progress_exactly_one_final_per_request(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    # A slow clock: no throttled tick after the first, so every progress past
    # the first is the final one.
    live, _stamps, _snap = _run_clocked(client, plaintext, _THREE_TWO, step=0.01)
    progress = [e for e in live if e["type"] == "progress"]
    assert [p["frames"] for p in progress] == [
        {"reasoning": 1, "content": 0, "tool_call": 0},
        {"reasoning": 3, "content": 2, "tool_call": 0},
    ]
    assert [e["type"] for e in live].count("request_end") == 1


def test_progress_counts_tool_call_frames(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    live, _stamps, _snap = _run_clocked(client, plaintext, _sse(_INTERLEAVED))
    last = [e for e in live if e["type"] == "progress"][-1]
    # Five frames carry tool_calls; the empty content opener is not text.
    assert last["frames"] == {"reasoning": 0, "content": 0, "tool_call": 5}


def test_progress_final_on_a_stream_that_raises(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    chunks = _sse([_frame({"content": "x"}), _frame({"content": "y"})])[:2]

    async def aiter():
        for c in chunks:
            yield c
        raise RuntimeError("upstream broke")

    resp = _sse_resp(chunks)
    resp.aiter_bytes = aiter
    hub = client.app.state.prompt_lens_hub
    q, _ = hub.subscribe()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=resp)):
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json=_CHAT,
            ) as r:
                b"".join(r.iter_bytes())
        except Exception:  # noqa: BLE001 -- the body raised mid-stream
            pass
    live = _drain(q)
    ends = [e for e in live if e["type"] == "request_end"]
    assert len(ends) == 1 and ends[0]["finish_reason"] == "error"
    i = next(i for i, e in enumerate(live) if e["type"] == "request_end")
    assert live[i - 1]["type"] == "progress"
    assert live[i - 1]["frames"] == {"reasoning": 0, "content": 2, "tool_call": 0}
    assert [e["type"] for e in live].count("progress") == 2  # first tick + final


def test_nonstream_publishes_no_progress(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    hub = client.app.state.prompt_lens_hub
    q, _ = hub.subscribe()
    upstream = {
        "id": "x",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    _post_path(client, plaintext, "/v1/chat/completions", {**_CHAT, "stream": False}, upstream)
    live = _drain(q)
    assert [e["type"] for e in live].count("request_end") == 1
    assert not [e for e in live if e["type"] == "progress"]


def test_disabled_stream_publishes_no_progress_and_touches_no_hub(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir, on=False)
    spy = MagicMock()
    client.app.state.prompt_lens_hub = spy
    with patch("app.proxy.routes._LIVE_UPDATE_INTERVAL_S", 0):
        raw = _stream_path(client, plaintext, "/v1/chat/completions", _CHAT, _THREE_TWO)
    assert raw == b"".join(_THREE_TWO)
    assert spy.method_calls == []
