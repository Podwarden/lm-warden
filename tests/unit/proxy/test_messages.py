"""#281: the Anthropic Messages API passthrough (POST /v1/messages).

The route translates an Anthropic request into the OpenAI chat body, runs it
through the ordinary ``_forward`` (so the upstream engine is faked exactly as
the other proxy tests fake it, by patching ``httpx.AsyncClient.send``), and
translates the answer back. Each test asserts BOTH sides: what reached the
engine, and what the Anthropic client got.
"""

import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.db.repos.tokens import hash_token
from app.proxy.messages_schemas import MessagesRequest
from app.proxy.messages_translate import (
    AnthropicStreamTranslator,
    map_stop_reason,
    to_openai_request,
)
from tests.unit.proxy.ledger_helpers import flush_ledger

PLAINTEXT = "vw_validtoken1234567890abcdef12345"
AUTH = {"Authorization": f"Bearer {PLAINTEXT}"}


def _seed(db_path, *, allowed_models: str | None = None) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]',"
            "'loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, allowed_models) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("tok1", "test", PLAINTEXT[:8], hash_token(PLAINTEXT), "inference", allowed_models),
        )
        db.commit()


def _ready(client, tmp_data_dir, **kw) -> None:
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db", **kw)
    client.app.state.supervisor._ports["qwen"] = 19099
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text))
    client.app.state.tokenizers = cache


def _json_upstream(body: dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def _sse_upstream(events: list[dict[str, Any]]) -> MagicMock:
    frames = [b"data: " + json.dumps(ev).encode() + b"\n\n" for ev in events]
    frames.append(b"data: [DONE]\n\n")

    async def aiter():
        for f in frames:
            yield f

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _sent_body(send: AsyncMock) -> dict[str, Any]:
    upstream_request = send.call_args.args[0]
    assert str(upstream_request.url).endswith("/v1/chat/completions")
    return json.loads(upstream_request.content)


def _post(client, body, *, headers=AUTH, upstream=None, path="/v1/messages"):
    send = AsyncMock(return_value=upstream)
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(path, headers=headers, json=body)
    return r, send


def _stream(client, body, upstream, headers=AUTH):
    send = AsyncMock(return_value=upstream)
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream("POST", "/v1/messages", headers=headers, json=body) as r:
            raw = b"".join(r.iter_bytes())
            status, ctype = r.status_code, r.headers.get("content-type", "")
    return status, ctype, _parse_sse(raw), send


def _parse_sse(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for frame in raw.decode().split("\n\n"):
        if not frame.strip():
            continue
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        data = json.loads(lines["data"])
        assert data["type"] == lines["event"], frame
        events.append((lines["event"], data))
    return events


def _chunk(delta=None, finish=None, usage=None, **choice_extra) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [],
    }
    if delta is not None or finish is not None:
        ev["choices"] = [
            {"index": 0, "delta": delta or {}, "finish_reason": finish, **choice_extra}
        ]
    if usage is not None:
        ev["usage"] = usage
    return ev


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_text_round_trip_non_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    upstream = _json_upstream(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }
    )
    r, send = _post(
        client,
        {
            "model": "qwen",
            "max_tokens": 256,
            "temperature": 0.2,
            "stop_sequences": ["###"],
            "messages": [{"role": "user", "content": "hi"}],
        },
        upstream=upstream,
    )

    assert r.status_code == 200, r.text
    sent = _sent_body(send)
    # _forward's own #173 injection (token priority 5 -> vLLM -5): proof the
    # translated body went through the shared forward, not a side path.
    assert sent.pop("priority") == -5
    assert sent == {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "stop": ["###"],
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    msg = r.json()
    assert msg["id"].startswith("msg_")
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["model"] == "qwen"
    assert msg["content"] == [{"type": "text", "text": "Hello there"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["stop_sequence"] is None
    assert msg["usage"]["input_tokens"] == 12
    assert msg["usage"]["output_tokens"] == 3


def test_text_round_trip_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    usage = {"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9}
    upstream = _sse_upstream(
        [
            _chunk({"role": "assistant", "content": ""}, usage=usage),
            _chunk({"content": "Hel"}, usage={**usage, "completion_tokens": 1}),
            _chunk({"content": "lo"}, usage={**usage, "completion_tokens": 2}),
            _chunk({}, finish="stop", usage={**usage, "completion_tokens": 2}),
            _chunk(usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}),
        ]
    )
    status, ctype, events, send = _stream(
        client,
        {
            "model": "qwen",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        upstream,
    )

    assert status == 200
    assert ctype.startswith("text/event-stream")
    sent = _sent_body(send)
    assert sent["stream"] is True
    assert sent["stream_options"]["include_usage"] is True

    assert [e for e, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["role"] == "assistant" and start["content"] == []
    assert start["usage"]["input_tokens"] == 9
    assert events[1][1] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    text = "".join(d["delta"]["text"] for e, d in events if e == "content_block_delta")
    assert text == "Hello"
    delta = events[5][1]
    assert delta["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert delta["usage"]["output_tokens"] == 2

    # _forward's end-of-stream settle ran through the translating wrapper:
    # the request is billed, with the engine's own completion count.
    flush_ledger(client)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        rows = db.execute("SELECT requests, completion_tokens FROM counters").fetchall()
    assert rows == [(1, 2)]


# ---------------------------------------------------------------------------
# Tool use
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    # An Anthropic server tool: no input_schema, dropped rather than failing.
    {"type": "web_search_20250305", "name": "web_search"},
]


def test_tool_use_round_trip_non_stream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    upstream = _json_upstream(
        {
            "id": "chatcmpl-2",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Oslo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
    )
    r, send = _post(
        client,
        {
            "model": "qwen",
            "max_tokens": 128,
            "tools": _TOOLS,
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": "Weather in Oslo?"}],
        },
        upstream=upstream,
    )

    assert r.status_code == 200, r.text
    sent = _sent_body(send)
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": _TOOLS[0]["input_schema"],
            },
        }
    ]
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False

    msg = r.json()
    assert msg["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "call_abc", "name": "get_weather", "input": {"city": "Oslo"}},
    ]
    assert msg["stop_reason"] == "tool_use"


def test_tool_result_turn_translates_to_tool_messages(tmp_data_dir, client):
    """The follow-up turn: the assistant's tool_use becomes tool_calls, the
    user's tool_result becomes a role=tool message keyed by tool_call_id and
    placed BEFORE the user's text."""
    _ready(client, tmp_data_dir)
    upstream = _json_upstream(
        {
            "choices": [{"index": 0, "message": {"content": "Sunny."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 2},
        }
    )
    r, send = _post(
        client,
        {
            "model": "qwen",
            "max_tokens": 128,
            "tools": _TOOLS[:1],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [
                {"role": "user", "content": "Weather in Oslo?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "call_abc",
                            "name": "get_weather",
                            "input": {"city": "Oslo"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "and be brief"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_abc",
                            "content": [{"type": "text", "text": "18C, sunny"}],
                        },
                    ],
                },
            ],
        },
        upstream=upstream,
    )

    assert r.status_code == 200, r.text
    sent = _sent_body(send)
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert sent["messages"] == [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "18C, sunny"},
        {"role": "user", "content": "and be brief"},
    ]
    assert r.json()["content"] == [{"type": "text", "text": "Sunny."}]


def test_tool_use_round_trip_stream(tmp_data_dir, client):
    """Tool-call arguments arrive in fragments; the client gets one complete
    tool_use block after the text block, with arguments that parse."""
    _ready(client, tmp_data_dir)
    tc0 = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather"}}
    upstream = _sse_upstream(
        [
            _chunk({"role": "assistant", "content": "Checking"}),
            _chunk({"tool_calls": [tc0]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"ci'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Oslo"}'}}]}),
            _chunk({}, finish="tool_calls"),
            _chunk(usage={"prompt_tokens": 40, "completion_tokens": 12}),
        ]
    )
    status, _, events, _ = _stream(
        client,
        {
            "model": "qwen",
            "max_tokens": 64,
            "stream": True,
            "tools": _TOOLS[:1],
            "messages": [{"role": "user", "content": "Weather in Oslo?"}],
        },
        upstream,
    )

    assert status == 200
    assert [e for e, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    tool_start = events[4][1]
    assert tool_start["index"] == 1
    assert tool_start["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
    }
    tool_delta = events[5][1]
    assert tool_delta["index"] == 1
    assert tool_delta["delta"]["type"] == "input_json_delta"
    assert json.loads(tool_delta["delta"]["partial_json"]) == {"city": "Oslo"}
    assert events[7][1]["delta"]["stop_reason"] == "tool_use"
    assert events[7][1]["usage"]["input_tokens"] == 40
    assert events[7][1]["usage"]["output_tokens"] == 12


# ---------------------------------------------------------------------------
# System prompt, thinking, stop sequences
# ---------------------------------------------------------------------------


def test_system_prompt_becomes_leading_system_message():
    req = MessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 10,
            "system": [
                {"type": "text", "text": "You are Claude Code."},
                {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<reminder/>"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
        }
    )
    body = to_openai_request(req)
    assert body["messages"] == [
        {"role": "system", "content": "You are Claude Code.\n\nBe terse."},
        {"role": "user", "content": "<reminder/>\n\nhi"},
    ]


def test_system_string_reaches_the_engine(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    upstream = _json_upstream(
        {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    r, send = _post(
        client,
        {
            "model": "qwen",
            "max_tokens": 8,
            "system": "Be terse.",
            "messages": [{"role": "user", "content": "hi"}],
        },
        upstream=upstream,
    )
    assert r.status_code == 200, r.text
    assert _sent_body(send)["messages"][0] == {"role": "system", "content": "Be terse."}


def test_thinking_off_by_default_tells_the_engine():
    req = MessagesRequest.model_validate(
        {"model": "qwen", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    )
    body = to_openai_request(req)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_enabled_leaves_chat_template_kwargs_unset():
    req = MessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    body = to_openai_request(req)
    assert "chat_template_kwargs" not in body


def test_tool_result_image_moves_to_user_message():
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
    req = MessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [img]}],
                }
            ],
        }
    )
    msgs = to_openai_request(req)["messages"]
    assert msgs[0]["role"] == "tool" and msgs[0]["tool_call_id"] == "t1"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }


def test_reasoning_streams_as_thinking_only_when_requested():
    events = [
        _chunk({"reasoning_content": "let me think"}),
        _chunk({"content": "42"}),
        _chunk({}, finish="stop"),
    ]

    def run(emit: bool) -> list[str]:
        tr = AnthropicStreamTranslator(model="qwen", stop_sequences=None, emit_thinking=emit)
        raw = b"".join(
            b"".join(tr.feed_bytes(b"data: " + json.dumps(e).encode() + b"\n\n")) for e in events
        )
        raw += b"".join(tr.finish())
        return [
            d["content_block"]["type"] for e, d in _parse_sse(raw) if e == "content_block_start"
        ]

    assert run(True) == ["thinking", "text"]
    assert run(False) == ["text"]


def test_stop_reason_mapping():
    assert map_stop_reason("stop", None, None, False) == ("end_turn", None)
    assert map_stop_reason("length", None, None, False) == ("max_tokens", None)
    assert map_stop_reason("tool_calls", None, None, True) == ("tool_use", None)
    # An engine that ends a tool-calling turn with plain "stop".
    assert map_stop_reason("stop", None, None, True) == ("tool_use", None)
    # vLLM reports the matched stop string as the choice's stop_reason.
    assert map_stop_reason("stop", "###", ["###"], False) == ("stop_sequence", "###")
    assert map_stop_reason("runaway_repeat", None, None, False) == ("max_tokens", None)


def test_mid_stream_engine_error_becomes_error_event():
    tr = AnthropicStreamTranslator(model="qwen", stop_sequences=None, emit_thinking=False)
    raw = b"".join(
        tr.feed_bytes(b"data: " + json.dumps(_chunk({"content": "a"})).encode() + b"\n\n")
    )
    raw += b"".join(tr.feed_bytes(b'data: {"error": {"message": "engine died"}}\n\n'))
    raw += b"".join(tr.finish())
    events = _parse_sse(raw)
    assert events[-1][0] == "error"
    assert events[-1][1]["error"] == {"type": "api_error", "message": "engine died"}
    assert "message_stop" not in [e for e, _ in events]


# ---------------------------------------------------------------------------
# Auth reuse and error envelopes
# ---------------------------------------------------------------------------

_BODY = {"model": "qwen", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


def _assert_error(r, status: int, kind: str) -> None:
    assert r.status_code == status, r.text
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == kind
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_missing_token_is_401_in_anthropic_shape(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, send = _post(client, _BODY, headers={})
    _assert_error(r, 401, "authentication_error")
    send.assert_not_called()


def test_bad_token_is_401_in_anthropic_shape(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, send = _post(client, _BODY, headers={"Authorization": "Bearer vw_nope"})
    _assert_error(r, 401, "authentication_error")
    r, send = _post(client, _BODY, headers={"x-api-key": "vw_nope"})
    _assert_error(r, 401, "authentication_error")
    send.assert_not_called()


def test_x_api_key_is_accepted_and_never_forwarded(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    upstream = _json_upstream(
        {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    r, send = _post(
        client,
        _BODY,
        headers={"x-api-key": PLAINTEXT, "anthropic-version": "2023-06-01"},
        upstream=upstream,
    )
    assert r.status_code == 200, r.text
    forwarded = {k.lower() for k in send.call_args.args[0].headers}
    assert "x-api-key" not in forwarded
    assert "authorization" not in forwarded


def test_x_api_key_is_not_a_credential_on_openai_routes(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r = client.post("/v1/chat/completions", headers={"x-api-key": PLAINTEXT}, json=_BODY)
    assert r.status_code == 401


def test_unknown_model_is_404_not_found_error(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, send = _post(client, {**_BODY, "model": "claude-sonnet-4-5"})
    _assert_error(r, 404, "not_found_error")
    assert "claude-sonnet-4-5" in r.json()["error"]["message"]
    send.assert_not_called()


def test_model_outside_token_allow_list_is_403_permission_error(tmp_data_dir, client):
    _ready(client, tmp_data_dir, allowed_models="other")
    r, _ = _post(client, _BODY)
    _assert_error(r, 403, "permission_error")


def test_missing_max_tokens_is_400_invalid_request_error(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, send = _post(client, {"model": "qwen", "messages": []})
    _assert_error(r, 400, "invalid_request_error")
    assert "max_tokens" in r.json()["error"]["message"]
    send.assert_not_called()


def test_unsupported_block_is_400_invalid_request_error(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {
        **_BODY,
        "messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": "f"}]}],
    }
    r, _ = _post(client, body)
    _assert_error(r, 400, "invalid_request_error")


def test_engine_error_is_translated(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    upstream = _json_upstream(
        {"object": "error", "message": "max_tokens is too large", "code": 400}, status=400
    )
    r, _ = _post(client, _BODY, upstream=upstream)
    _assert_error(r, 400, "invalid_request_error")
    assert r.json()["error"]["message"] == "max_tokens is too large"


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_count_tokens_uses_the_model_tokenizer(tmp_data_dir, client):
    _ready(client, tmp_data_dir)  # fake tokenizer: one token per character
    r, send = _post(
        client,
        {"model": "qwen", "system": "sys", "messages": [{"role": "user", "content": "hello"}]},
        path="/v1/messages/count_tokens",
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"input_tokens": len("sys\nhello")}
    send.assert_not_called()


# ---------------------------------------------------------------------------
# #286: cache-affine replica routing keys on the ORIGINAL Anthropic body
# ---------------------------------------------------------------------------


def test_metadata_user_id_pins_the_replica(tmp_data_dir, client):
    from app.proxy.dp_affinity import RANK_HEADER

    _ready(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE models SET gpu_indices=?, data_parallel_size=7 WHERE id='qwen'",
            (json.dumps(list(range(7))),),
        )
    ok = {
        "id": "c1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def sent_rank(user_id):
        body = {**_BODY, "metadata": {"user_id": user_id}} if user_id else dict(_BODY)
        r, send = _post(client, body, upstream=_json_upstream(ok))
        assert r.status_code == 200, r.text
        return int(send.call_args.args[0].headers[RANK_HEADER]), send

    # A request with no metadata.user_id falls back to the prompt-hash key; a
    # metadata.user_id is a session of its own, whatever the prompt says.
    sent_rank(None)
    seen = {}
    for user_id in ("sess-C-0001", "sess-B-0001"):
        first, send = sent_rank(user_id)
        seen[user_id] = first
        again, send = sent_rank(user_id)
        assert again == first  # sticky
    state = client.app.state.dp_routing.snapshot("qwen", 7)["totals"]
    assert state["placed"] == 3 and state["sticky"] == 2  # fallback key + two sessions
    assert "user" not in _sent_body(send)  # the translated body carries no user


def test_stream_client_disconnect_before_first_chunk_releases_rank_and_slot(tmp_data_dir, client):
    """The Anthropic wrapper must carry the inner response's background task:
    a client gone before the first chunk leaves the generator unstarted, and
    only ``abandon`` (run as background) releases the rank and admission slot.
    Drives the REAL Starlette ``StreamingResponse.__call__``."""
    import asyncio

    _ready(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE models SET gpu_indices=?, data_parallel_size=7 WHERE id='qwen'",
            (json.dumps(list(range(7))),),
        )
    upstream = _sse_upstream([_chunk({"content": "a"}), _chunk({}, "stop")])
    raw = json.dumps({**_BODY, "stream": True}).encode()
    app = client.app

    async def run() -> None:
        from starlette.requests import Request

        from app.db.repos.tokens import TokenRow
        from app.proxy.routes_messages import messages

        async def body_receive() -> dict[str, Any]:
            return {"type": "http.request", "body": raw, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [(b"content-type", b"application/json")],
            "app": app,
            "query_string": b"",
            "client": ("127.0.0.1", 1),
        }
        token = TokenRow(
            id="tok1",
            name="test",
            prefix="vw_valid",
            scope="inference",
            allowed_models=None,
            rate_limit_rpm=None,
            rate_limit_tpm=None,
            revoked_at=None,
            last_used_at=None,
            created_at="2026-01-01",
            expires_at=None,
            rotated_at=None,
            rotated_from=None,
        )
        with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=upstream)):
            resp = await messages(Request(scope, body_receive), token)

        async def disconnected() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            # The client is gone before the response start lands, so the body
            # iterator is never read.
            await asyncio.sleep(3600)

        # The REAL Starlette StreamingResponse.__call__: disconnect cancels the
        # stream task, then the response's background task runs.
        await asyncio.wait_for(resp(scope, disconnected, send), timeout=10)

    client.portal.call(run)
    assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 0
    assert client.app.state.scheduler._inflight_for_test("qwen") == 0
