"""The proxy writes the forest columns end to end (spec §3)."""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import wait_until
from tests.unit.proxy.test_queue_wait_recorded import _fake_tokenizer, _seed_loaded

RAW_SID = "0b1c2d3e-1111-4222-8333-444455556666"


def _post(client, plaintext, messages, headers=None, reply=None):
    body = reply or {
        "id": "x",
        "model": "qwen",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "application/json"}
    fake.aread = AsyncMock(return_value=json.dumps(body).encode())
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)):
        return client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}", **(headers or {})},
            json={"model": "qwen", "messages": messages},
        )


def _rows(db_path, n):
    def _get():
        with sqlite3.connect(db_path) as db:
            r = db.execute(
                "SELECT session_key, turn_index, batch_id, tools_in, tools_out FROM request_history ORDER BY finished_at"
            ).fetchall()
        return r if len(r) >= n else None

    return wait_until(_get, what=f"{n} request_history rows")


def test_session_turn_batch_and_tools_are_recorded(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    hdr = {"X-Claude-Code-Session-Id": RAW_SID, "X-Batch-Id": "farm-A27"}
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 40},
    ]
    reply = {
        "id": "y",
        "model": "qwen",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "Edit", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }
    assert _post(client, plaintext, msgs, hdr, reply).status_code == 200
    assert _post(client, plaintext, msgs, hdr).status_code == 200
    rows = _rows(db_path, 2)
    k0, t0, b0, tin0, tout0 = rows[0]
    assert k0 and RAW_SID not in k0 and len(k0) == 16
    assert rows[1][0] == k0
    assert (t0, rows[1][1]) == (0, 1)
    assert b0 == "farm-A27"
    assert json.loads(tin0) == [["read", 10, False]]
    assert [x[0] for x in json.loads(tout0)] == ["edit"]


def test_no_session_signal_is_fine(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    assert _post(client, plaintext, [{"role": "user", "content": "hi"}]).status_code == 200
    ((k, t, b, tin, tout),) = _rows(db_path, 1)
    assert (k, t, b) == (None, None, None)
    assert json.loads(tin) == [] and json.loads(tout) == []


def _post_seeing_inflight(client, plaintext, messages, headers=None):
    """Post one request; while it is in flight, serialize the live registry's rows as
    ``GET /api/stats/requests`` does. Returns (response, in-flight rows)."""
    from app.stats.live_requests import _serialize

    seen: list[dict] = []
    body = {
        "id": "x",
        "model": "qwen",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "application/json"}
    fake.aread = AsyncMock(return_value=json.dumps(body).encode())
    fake.aclose = AsyncMock()

    async def send(*_a, **_k):
        reg = client.app.state.request_registry
        seen.extend(_serialize(r, r.started_monotonic) for r in reg.snapshot())
        return fake

    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}", **(headers or {})},
            json={"model": "qwen", "messages": messages},
        )
    return r, seen


def test_inflight_forest_session_is_the_history_session_key(tmp_data_dir, client):
    """C2: an in-flight row names its forest session by the same hashed key the forest uses."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    hdr = {"X-Claude-Code-Session-Id": RAW_SID}
    r, live = _post_seeing_inflight(client, plaintext, [{"role": "user", "content": "go"}], hdr)
    assert r.status_code == 200
    ((key, *_),) = _rows(db_path, 1)
    (row,) = live
    assert key and row["forest_session"] == key
    assert row["session_id"] == RAW_SID  # the raw id is unchanged


def test_inflight_forest_session_is_null_without_a_session(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    r, live = _post_seeing_inflight(client, plaintext, [{"role": "user", "content": "hi"}])
    assert r.status_code == 200
    (row,) = live
    assert "forest_session" in row and row["forest_session"] is None


# ---- streamed tool calls (the accumulator end to end, through _settle) -----------------------


def _sse(events):
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


def _chunk(delta=None, finish=None, usage=None):
    ev = {"id": "c", "object": "chat.completion.chunk", "model": "qwen", "choices": []}
    if delta is not None or finish is not None:
        ev["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        ev["usage"] = usage
    return ev


_TOOL_STREAM = [
    _chunk({"role": "assistant", "content": "Checking"}),
    _chunk(
        {"tool_calls": [{"index": 0, "id": "a", "type": "function", "function": {"name": "Bash"}}]}
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd": "ls"}'}}]}),
    _chunk(
        {
            "tool_calls": [
                {"index": 1, "id": "b", "type": "function", "function": {"name": "WebFetch"}}
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]}),
    _chunk({}, finish="tool_calls"),
    _chunk(usage={"prompt_tokens": 40, "completion_tokens": 12}),
]


def _stream_ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()
    return db_path, plaintext


def _stream(client, path, plaintext, body):
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_sse(_TOOL_STREAM))):
        with client.stream(
            "POST", path, headers={"Authorization": f"Bearer {plaintext}"}, json=body
        ) as r:
            raw = b"".join(r.iter_bytes())
            assert r.status_code == 200, raw
    return raw


def test_streamed_chat_tool_calls_land_in_tools_out(tmp_data_dir, client):
    db_path, plaintext = _stream_ready(client, tmp_data_dir)
    body = {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "go"}]}
    _stream(client, "/v1/chat/completions", plaintext, body)
    ((_, _, _, _, tout),) = _rows(db_path, 1)
    got = json.loads(tout)
    assert [x[0] for x in got] == ["shell", "web"]
    assert all(isinstance(x[1], str) and len(x[1]) == 8 for x in got)  # hashed, never the name
    assert "Bash" not in tout and "WebFetch" not in tout


def test_streamed_anthropic_messages_tool_calls_land_in_tools_out(tmp_data_dir, client):
    db_path, plaintext = _stream_ready(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "go"}],
    }
    raw = _stream(client, "/v1/messages", plaintext, body)
    assert b"tool_use" in raw  # the client got Anthropic frames, translated from the stream
    ((_, _, _, _, tout),) = _rows(db_path, 1)
    assert [x[0] for x in json.loads(tout)] == ["shell", "web"]
