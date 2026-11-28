"""Reasoning budget (#348) in the forward path: the three layers, both modes, the engine
field, and the client-visible stop reason, streaming and not."""

import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, patch

from tests.conftest import wait_until
from tests.unit.proxy.test_messages import (
    AUTH,
    _json_upstream,
    _parse_sse,
    _ready,
    _sse_upstream,
)

CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}


def _sql(client_or_path, sql: str, *params: Any) -> None:
    with sqlite3.connect(client_or_path) as db:
        db.execute(sql, params)
        db.commit()


def _setup(
    client,
    tmp_data_dir,
    *,
    model: str = "",
    key: str = "",
    backend: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Seed the fake model and key; ``model``/``key`` are SET clauses for the two rows."""
    _ready(client, tmp_data_dir)
    db = str(tmp_data_dir / "vllm-warden.db")
    if backend:
        _sql(db, "UPDATE models SET backend = ?", backend)
    if extra_args is not None:
        _sql(db, "UPDATE models SET extra_args = ?", json.dumps(extra_args))
    if model:
        _sql(db, f"UPDATE models SET {model}")
    if key:
        _sql(db, f"UPDATE api_tokens SET {key}")
    return db


def _r(text: str = "t") -> dict[str, Any]:
    return _delta({"reasoning_content": text})


def _c(text: str) -> dict[str, Any]:
    return _delta({"content": text})


def _delta(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "created": 1,
        "choices": [{"index": 0, "delta": d, "finish_reason": None}],
    }


def _fin(reason: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "created": 1,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }


USAGE = {
    "id": "c1",
    "object": "chat.completion.chunk",
    "model": "qwen",
    "choices": [],
    "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
}


def _looping(n: int = 20) -> list[dict[str, Any]]:
    """A model that never leaves its reasoning: n reasoning deltas, then (never reached) an answer."""
    return [_r() for _ in range(n)] + [_c("late answer"), _fin("stop"), USAGE]


def _answering(reasoning: int = 3) -> list[dict[str, Any]]:
    return [_r() for _ in range(reasoning)] + [_c("the answer"), _fin("stop"), USAGE]


def _post(client, body, upstream, *, path="/v1/chat/completions", stream=False):
    send = AsyncMock(return_value=upstream)
    with patch("httpx.AsyncClient.send", new=send):
        if stream:
            with client.stream("POST", path, headers=AUTH, json=body) as r:
                raw = b"".join(r.iter_bytes())
                return r.status_code, raw, send
        r = client.post(path, headers=AUTH, json=body)
        return r.status_code, r.json(), send


def _sse_events(raw: bytes) -> list[Any]:
    out: list[Any] = []
    for part in raw.decode().split("\n\n"):
        part = part.strip()
        if part.startswith("data:"):
            payload = part[5:].strip()
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


def _sent(send) -> dict[str, Any]:
    return json.loads(send.call_args.args[0].content)


def _history_finish(db: str) -> str | None:
    def _rows():
        with sqlite3.connect(db) as c:
            return c.execute("SELECT finish_reason FROM request_history").fetchall() or None

    rows = wait_until(_rows, what="request_history row")
    assert isinstance(rows, list)
    return rows[0][0]


# -- nothing set: no behaviour change ----------------------------------------


def test_nothing_set_forwards_byte_identical_and_adds_no_fields(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    ok = {
        "id": "c",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    status, body, send = _post(client, CHAT, _json_upstream(ok))
    assert status == 200 and body == ok
    assert send.call_args.args[0].content == json.dumps({**CHAT, "priority": -5}).encode()


# -- engine field --------------------------------------------------------------


def test_llamacpp_model_default_goes_to_the_engine_and_the_warden_field_is_stripped(
    tmp_data_dir, client
):
    _setup(client, tmp_data_dir, model="reasoning_budget = 800", backend="llamacpp")
    _, _, send = _post(client, {**CHAT, "lmw_reasoning_budget": 5000}, _sse_upstream(_answering(1)))
    sent = _sent(send)
    assert sent["reasoning_budget_tokens"] == 800  # the model default is the smaller
    assert "lmw_reasoning_budget" not in sent
    assert "thinking_token_budget" not in sent


def test_request_alone_is_enough_and_effort_maps_through_the_model_table(tmp_data_dir, client):
    _setup(
        client,
        tmp_data_dir,
        backend="llamacpp",
        model="reasoning_effort_budgets = '{\"low\": 123}'",
    )
    _, _, send = _post(client, {**CHAT, "reasoning_effort": "low"}, _sse_upstream(_answering(1)))
    sent = _sent(send)
    assert sent["reasoning_budget_tokens"] == 123
    assert sent["reasoning_effort"] == "low"  # the client's own field still reaches the engine


def test_vllm_with_a_reasoning_parser_gets_thinking_token_budget(tmp_data_dir, client):
    _setup(
        client,
        tmp_data_dir,
        model="reasoning_budget = 700",
        backend="vllm",
        extra_args=["--reasoning-parser", "qwen3"],
    )
    _, _, send = _post(client, CHAT, _sse_upstream(_answering(1)))
    assert _sent(send)["thinking_token_budget"] == 700


def test_vllm_without_a_parser_sends_no_engine_field_and_the_proxy_enforces(tmp_data_dir, client):
    _setup(client, tmp_data_dir, model="reasoning_budget = 3", backend="vllm", extra_args=[])
    status, body, send = _post(client, CHAT, _sse_upstream(_looping()))
    sent = _sent(send)
    assert "thinking_token_budget" not in sent and "reasoning_budget_tokens" not in sent
    choice = body["choices"][0]
    assert status == 200 and choice["finish_reason"] == "length"
    assert body["lmw_stop_reason"] == "reasoning_budget" and body["lmw_reasoning_budget"] == 3
    assert "late answer" not in json.dumps(body)


def test_a_non_streaming_client_is_streamed_upstream(tmp_data_dir, client):
    _setup(client, tmp_data_dir, model="reasoning_budget = 800", backend="llamacpp")
    _, _, send = _post(client, CHAT, _sse_upstream(_answering(1)))
    assert _sent(send)["stream"] is True


# -- precedence ---------------------------------------------------------------


def test_key_cap_beats_a_larger_request_and_a_smaller_request_beats_the_key(tmp_data_dir, client):
    _setup(client, tmp_data_dir, key="reasoning_budget = 1000", backend="llamacpp")
    _, _, send = _post(client, {**CHAT, "lmw_reasoning_budget": 9000}, _sse_upstream(_answering(1)))
    assert _sent(send)["reasoning_budget_tokens"] == 1000
    _, _, send = _post(client, {**CHAT, "lmw_reasoning_budget": 400}, _sse_upstream(_answering(1)))
    assert _sent(send)["reasoning_budget_tokens"] == 400


def test_key_mode_overrides_the_models(tmp_data_dir, client):
    _setup(
        client,
        tmp_data_dir,
        backend="llamacpp",
        model="reasoning_budget = 3, reasoning_budget_mode = 'force_answer'",
        key="reasoning_budget_mode = 'stop'",
    )
    status, body, send = _post(client, CHAT, _sse_upstream(_looping()))
    assert "reasoning_budget_tokens" not in _sent(send)  # stop: the proxy's job
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["lmw_stop_reason"] == "reasoning_budget"


# -- stop mode, streaming and not --------------------------------------------


def test_stop_mode_streaming_ends_with_length_and_the_stop_reason(tmp_data_dir, client):
    db = _setup(client, tmp_data_dir, model="reasoning_budget = 3, reasoning_budget_mode = 'stop'")
    upstream = _sse_upstream(_looping())
    status, raw, _ = _post(client, {**CHAT, "stream": True}, upstream, stream=True)
    events = _sse_events(raw)
    assert status == 200 and events[-1] == "[DONE]"
    last = events[-2]
    assert last["choices"][0]["finish_reason"] == "length"
    assert last["lmw_stop_reason"] == "reasoning_budget" and last["lmw_reasoning_budget"] == 3
    assert b"late answer" not in raw
    upstream.aclose.assert_awaited()  # the engine's generation is aborted
    assert _history_finish(db) == "reasoning_budget"


def test_stop_mode_non_streaming_body(tmp_data_dir, client):
    db = _setup(client, tmp_data_dir, model="reasoning_budget = 3, reasoning_budget_mode = 'stop'")
    status, body, _ = _post(client, CHAT, _sse_upstream(_looping()))
    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["content"] == ""
    assert body["lmw_stop_reason"] == "reasoning_budget" and body["lmw_reasoning_budget"] == 3
    assert _history_finish(db) == "reasoning_budget"


# -- force_answer (engine closes the block) ---------------------------------


def test_force_answer_keeps_the_reply_and_tells_the_client(tmp_data_dir, client):
    db = _setup(client, tmp_data_dir, model="reasoning_budget = 3", backend="llamacpp")
    status, body, _ = _post(client, CHAT, _sse_upstream(_answering(3)))
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "the answer"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["lmw_stop_reason"] == "reasoning_budget" and body["lmw_reasoning_budget"] == 3
    assert _history_finish(db) == "reasoning_budget"


def test_force_answer_streaming_final_chunk_carries_the_fields(tmp_data_dir, client):
    _setup(client, tmp_data_dir, model="reasoning_budget = 3", backend="llamacpp")
    _, raw, _ = _post(client, {**CHAT, "stream": True}, _sse_upstream(_answering(3)), stream=True)
    events = _sse_events(raw)
    fin = [e for e in events if e != "[DONE]" and e["choices"] and e["choices"][0]["finish_reason"]]
    assert fin[0]["lmw_stop_reason"] == "reasoning_budget"
    assert fin[0]["choices"][0]["finish_reason"] == "stop"
    assert b"the answer" in raw


def test_a_request_that_stayed_within_budget_is_unmarked(tmp_data_dir, client):
    db = _setup(client, tmp_data_dir, model="reasoning_budget = 50", backend="llamacpp")
    status, body, _ = _post(client, CHAT, _sse_upstream(_answering(3)))
    assert status == 200 and not any(k.startswith("lmw_") for k in body)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert _history_finish(db) == "stop"


def test_proxy_backstop_ends_an_engine_enforced_request_that_ignores_the_field(
    tmp_data_dir, client
):
    """A template without thinking tags: llama.cpp drops the field, the model loops."""
    _setup(client, tmp_data_dir, model="reasoning_budget = 1", backend="llamacpp")
    status, body, _ = _post(client, CHAT, _sse_upstream(_looping(400)))
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["lmw_stop_reason"] == "reasoning_budget"


# -- Anthropic ----------------------------------------------------------------


def test_anthropic_thinking_budget_stop_mode_streaming(tmp_data_dir, client):
    _setup(client, tmp_data_dir, key="reasoning_budget_mode = 'stop'")
    req = {
        "model": "qwen",
        "max_tokens": 100,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 2},
    }
    send = AsyncMock(return_value=_sse_upstream(_looping()))
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream("POST", "/v1/messages", headers=AUTH, json=req) as r:
            raw = b"".join(r.iter_bytes())
    events = _parse_sse(raw)
    delta = [d for name, d in events if name == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "max_tokens"
    assert delta["delta"]["lmw_stop_reason"] == "reasoning_budget"
    assert delta["delta"]["lmw_reasoning_budget"] == 2
    assert "late answer" not in raw.decode()


def test_anthropic_non_streaming_message(tmp_data_dir, client):
    _setup(client, tmp_data_dir, key="reasoning_budget_mode = 'stop'")
    req = {
        "model": "qwen",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 2},
    }
    status, body, _ = _post(client, req, _sse_upstream(_looping()), path="/v1/messages")
    assert status == 200
    assert body["stop_reason"] == "max_tokens"
    assert body["lmw_stop_reason"] == "reasoning_budget" and body["lmw_reasoning_budget"] == 2


# -- Responses ----------------------------------------------------------------


def test_responses_effort_maps_and_marks_the_response(tmp_data_dir, client):
    _setup(
        client,
        tmp_data_dir,
        key="reasoning_budget_mode = 'stop'",
        model="reasoning_effort_budgets = '{\"low\": 2}'",
    )
    req = {"model": "qwen", "input": "hi", "reasoning": {"effort": "low"}}
    status, body, send = _post(client, req, _sse_upstream(_looping()), path="/v1/responses")
    assert status == 200, body
    assert "reasoning_effort" not in _sent(send) and "lmw_reasoning_effort" not in _sent(send)
    assert body["status"] == "incomplete"
    assert body["lmw_stop_reason"] == "reasoning_budget" and body["lmw_reasoning_budget"] == 2
