"""Reasoning budget (#348): request mapping and the client-visible stop reason through the
Anthropic and Responses translators and the stream aggregator."""

import json
from typing import Any

from app.proxy.messages_schemas import MessagesRequest
from app.proxy.messages_translate import (
    AnthropicStreamTranslator,
    to_anthropic_message,
    to_openai_request,
)
from app.proxy.reaggregate import StreamAggregator
from app.proxy.reasoning_budget import effort_table, request_budget, strip_request_fields
from app.proxy.responses_schemas import ResponsesRequest
from app.proxy.responses_translate import (
    ResponsesStreamTranslator,
    to_response_object,
)
from app.proxy.responses_translate import (
    to_openai_request as responses_to_openai,
)


def _msg(**over: Any) -> MessagesRequest:
    base = {"model": "qwen", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    return MessagesRequest.model_validate({**base, **over})


# -- request mapping ----------------------------------------------------------


def test_anthropic_thinking_budget_becomes_the_explicit_field():
    oai = to_openai_request(_msg(thinking={"type": "enabled", "budget_tokens": 2000}))
    assert oai["lmw_reasoning_budget"] == 2000


def test_anthropic_without_thinking_or_budget_sets_nothing():
    assert "lmw_reasoning_budget" not in to_openai_request(_msg())
    assert "lmw_reasoning_budget" not in to_openai_request(_msg(thinking={"type": "disabled"}))
    assert "lmw_reasoning_budget" not in to_openai_request(_msg(thinking={"type": "enabled"}))


def test_anthropic_explicit_field_is_honoured_and_the_smaller_wins():
    req = _msg(thinking={"type": "enabled", "budget_tokens": 2000}, lmw_reasoning_budget=500)
    assert to_openai_request(req)["lmw_reasoning_budget"] == 500
    req = _msg(thinking={"type": "enabled", "budget_tokens": 300}, lmw_reasoning_budget=500)
    assert to_openai_request(req)["lmw_reasoning_budget"] == 300
    assert to_openai_request(_msg(lmw_reasoning_budget=700))["lmw_reasoning_budget"] == 700


def test_responses_effort_travels_as_an_internal_field_not_as_reasoning_effort():
    req = ResponsesRequest.model_validate(
        {"model": "qwen", "input": "hi", "reasoning": {"effort": "low"}}
    )
    oai = responses_to_openai(req).oai
    assert oai["lmw_reasoning_effort"] == "low"
    assert "reasoning_effort" not in oai  # llama.cpp refuses a level its template lacks
    assert request_budget(oai, effort_table(None)) == 1024
    none = ResponsesRequest.model_validate(
        {"model": "qwen", "input": "hi", "reasoning": {"effort": "none"}}
    )
    assert "lmw_reasoning_effort" not in responses_to_openai(none).oai


def test_strip_removes_every_internal_field():
    body = {"lmw_reasoning_budget": 1, "lmw_reasoning_effort": "low", "reasoning_effort": "high"}
    assert strip_request_fields(body) is True
    assert body == {"reasoning_effort": "high"}


def test_explicit_beats_openai_effort_beats_internal_effort():
    table = effort_table(None)
    assert (
        request_budget({"reasoning_effort": "low", "lmw_reasoning_effort": "high"}, table) == 1024
    )
    assert request_budget({"lmw_reasoning_effort": "high"}, table) == 16384


# -- aggregator ---------------------------------------------------------------

LMW = {"lmw_stop_reason": "reasoning_budget", "lmw_reasoning_budget": 50}


def _chunks(finish: str = "length") -> list[dict[str, Any]]:
    return [
        {
            "id": "c1",
            "model": "m",
            "created": 1,
            "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "hmm"}}],
        },
        {
            "id": "c1",
            "model": "m",
            "created": 1,
            "choices": [{"index": 0, "delta": {"content": "ans"}}],
        },
        {
            "id": "c1",
            "model": "m",
            "created": 1,
            **LMW,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        },
    ]


def test_aggregator_keeps_the_lmw_fields_and_adds_none_when_absent():
    agg = StreamAggregator(is_chat=True)
    for ev in _chunks():
        agg.feed_event(ev)
    out = agg.build()
    assert out["lmw_stop_reason"] == "reasoning_budget" and out["lmw_reasoning_budget"] == 50
    plain = StreamAggregator(is_chat=True)
    for ev in _chunks()[:2]:
        plain.feed_event(ev)
    assert not any(k.startswith("lmw_") for k in plain.build())


# -- Anthropic output ---------------------------------------------------------


def test_anthropic_message_carries_the_fields_and_max_tokens_on_length():
    oai = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        **LMW,
    }
    out = to_anthropic_message(oai, model="m", stop_sequences=None, emit_thinking=True)
    assert out["stop_reason"] == "max_tokens"
    assert out["lmw_stop_reason"] == "reasoning_budget"
    assert out["lmw_reasoning_budget"] == 50


def test_anthropic_message_unchanged_without_the_fields():
    oai = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}
        ]
    }
    out = to_anthropic_message(oai, model="m", stop_sequences=None, emit_thinking=True)
    assert not any(k.startswith("lmw_") for k in out)


def test_anthropic_stream_message_delta_carries_the_fields():
    tr = AnthropicStreamTranslator(model="m", stop_sequences=None, emit_thinking=True)
    frames: list[bytes] = []
    for ev in _chunks("stop"):
        frames += tr.feed_bytes(b"data: " + json.dumps(ev).encode() + b"\n\n")
    frames += tr.finish()
    delta = [f for f in frames if f.startswith(b"event: message_delta")][0]
    payload = json.loads(delta.split(b"data: ", 1)[1])
    assert payload["delta"]["stop_reason"] == "end_turn"  # force_answer keeps the engine's reason
    assert payload["delta"]["lmw_stop_reason"] == "reasoning_budget"
    assert payload["delta"]["lmw_reasoning_budget"] == 50


# -- Responses output ---------------------------------------------------------


def test_responses_object_carries_the_fields():
    req = ResponsesRequest.model_validate({"model": "qwen", "input": "hi"})
    tr = responses_to_openai(req)
    oai = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": "length",
            }
        ],
        **LMW,
    }
    out = to_response_object(oai, model="m", tr=tr)
    assert out["status"] == "incomplete"
    assert out["lmw_stop_reason"] == "reasoning_budget" and out["lmw_reasoning_budget"] == 50


def test_responses_stream_completed_event_carries_the_fields():
    req = ResponsesRequest.model_validate({"model": "qwen", "input": "hi", "stream": True})
    st = ResponsesStreamTranslator(model="m", tr=responses_to_openai(req))
    frames: list[bytes] = []
    for ev in _chunks():
        frames += st.feed_bytes(b"data: " + json.dumps(ev).encode() + b"\n\n")
    frames += st.finish()
    done = [f for f in frames if b"response.completed" in f][-1]
    payload = json.loads(done.split(b"data: ", 1)[1])
    assert payload["response"]["lmw_stop_reason"] == "reasoning_budget"
