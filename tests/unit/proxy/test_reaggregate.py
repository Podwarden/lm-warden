"""Unit tests for SSE -> non-stream re-aggregation (app/proxy/reaggregate.py).

When the runaway detector forces upstream ``stream=true`` for a client that
asked for ``stream=false``, we must rebuild vLLM's own non-stream JSON from the
chunk sequence. Fidelity is the risk, so these tests pin the reconstructed
schema against known chunk sequences: chat + text, usage tail, tool_calls by
index, ``n>1`` by choice index, reasoning_content, the trip finish_reason
override, and graceful handling of an unrecognized chunk shape.
"""

import json

from app.proxy.reaggregate import StreamAggregator, parse_sse_event


def _ev(**kw):
    return json.dumps(kw).encode()


# ---------------------------------------------------------------------------
# parse_sse_event
# ---------------------------------------------------------------------------


def test_parse_sse_event_returns_dict_for_data_line():
    line = b'data: {"id":"x","choices":[]}'
    assert parse_sse_event(line) == {"id": "x", "choices": []}


def test_parse_sse_event_done_returns_none():
    assert parse_sse_event(b"data: [DONE]") is None


def test_parse_sse_event_non_data_returns_none():
    assert parse_sse_event(b": keep-alive") is None


def test_parse_sse_event_malformed_returns_none():
    assert parse_sse_event(b"data: {not json") is None


# ---------------------------------------------------------------------------
# chat.completion reconstruction
# ---------------------------------------------------------------------------


def test_chat_basic_reconstruction_with_usage_tail():
    agg = StreamAggregator(is_chat=True)
    agg.feed_event(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1000,
            "model": "qwen",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
    )
    agg.feed_event(
        {"choices": [{"index": 0, "delta": {"content": "the answer"}, "finish_reason": None}]}
    )
    agg.feed_event(
        {"choices": [{"index": 0, "delta": {"content": " is 42"}, "finish_reason": None}]}
    )
    agg.feed_event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    # include_usage tail chunk: empty choices, populated usage.
    agg.feed_event(
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}}
    )

    out = agg.build()
    assert out["id"] == "chatcmpl-1"
    assert out["object"] == "chat.completion"
    assert out["created"] == 1000
    assert out["model"] == "qwen"
    assert out["usage"] == {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    assert len(out["choices"]) == 1
    ch = out["choices"][0]
    assert ch["index"] == 0
    assert ch["finish_reason"] == "stop"
    assert ch["message"]["role"] == "assistant"
    assert ch["message"]["content"] == "the answer is 42"


def test_chat_reasoning_content_accumulates_into_message():
    agg = StreamAggregator(is_chat=True)
    agg.feed_event(
        {
            "id": "chatcmpl-r",
            "model": "qwen",
            "created": 1,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "reasoning_content": "let me"}}
            ],
        }
    )
    agg.feed_event({"choices": [{"index": 0, "delta": {"reasoning_content": " think"}}]})
    agg.feed_event(
        {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
    )
    out = agg.build()
    ch = out["choices"][0]
    assert ch["message"]["reasoning_content"] == "let me think"
    assert ch["message"]["content"] == "done"


def test_chat_reasoning_field_accumulates_and_emits_the_engine_name():
    """R22: newer vLLM streams the channel as ``delta.reasoning``. The
    rebuilt body must carry it — under the same name the engine used, so the
    reconstruction matches what the engine's own non-stream endpoint would
    have produced."""
    agg = StreamAggregator(is_chat=True)
    agg.feed_event(
        {
            "id": "chatcmpl-r2",
            "model": "qwen",
            "created": 1,
            "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "let me"}}],
        }
    )
    agg.feed_event({"choices": [{"index": 0, "delta": {"reasoning": " think"}}]})
    agg.feed_event(
        {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
    )
    out = agg.build()
    ch = out["choices"][0]
    assert ch["message"].get("reasoning") == "let me think"
    assert "reasoning_content" not in ch["message"]
    assert ch["message"]["content"] == "done"


def test_chat_reasoning_field_name_first_seen_wins():
    """A stream that (pathologically) carries both names keeps the name it
    first used, so one channel is never split across two keys."""
    agg = StreamAggregator(is_chat=True)
    agg.feed_event({"choices": [{"index": 0, "delta": {"reasoning_content": "a"}}]})
    agg.feed_event({"choices": [{"index": 0, "delta": {"reasoning": "b"}}]})
    out = agg.build()
    ch = out["choices"][0]
    assert ch["message"]["reasoning_content"] == "ab"
    assert "reasoning" not in ch["message"]


def test_chat_tool_calls_reassembled_by_index():
    agg = StreamAggregator(is_chat=True)
    agg.feed_event(
        {
            "id": "chatcmpl-t",
            "model": "qwen",
            "created": 2,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                }
            ],
        }
    )
    agg.feed_event(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
                }
            ]
        }
    )
    agg.feed_event(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    out = agg.build()
    ch = out["choices"][0]
    assert ch["finish_reason"] == "tool_calls"
    tcs = ch["message"]["tool_calls"]
    assert len(tcs) == 1
    tc = tcs[0]
    assert tc["id"] == "call_abc"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == '{"city":"Paris"}'


def test_chat_n_gt_1_keyed_by_choice_index():
    agg = StreamAggregator(is_chat=True)
    agg.feed_event(
        {
            "id": "chatcmpl-n",
            "model": "qwen",
            "created": 3,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "A0"}},
                {"index": 1, "delta": {"role": "assistant", "content": "B0"}},
            ],
        }
    )
    agg.feed_event(
        {
            "choices": [
                {"index": 0, "delta": {"content": "A1"}, "finish_reason": "stop"},
                {"index": 1, "delta": {"content": "B1"}, "finish_reason": "length"},
            ]
        }
    )
    out = agg.build()
    assert len(out["choices"]) == 2
    by_index = {c["index"]: c for c in out["choices"]}
    assert by_index[0]["message"]["content"] == "A0A1"
    assert by_index[0]["finish_reason"] == "stop"
    assert by_index[1]["message"]["content"] == "B0B1"
    assert by_index[1]["finish_reason"] == "length"


def test_chat_finish_reason_override_on_trip():
    agg = StreamAggregator(is_chat=True)
    agg.feed_event(
        {
            "id": "chatcmpl-x",
            "model": "qwen",
            "created": 4,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}}],
        }
    )
    # No natural finish_reason arrived (runaway torn down mid-stream).
    out = agg.build(finish_reason_override="runaway_think")
    ch = out["choices"][0]
    assert ch["finish_reason"] == "runaway_think"
    assert ch["message"]["content"] == "partial"


# ---------------------------------------------------------------------------
# text_completion reconstruction
# ---------------------------------------------------------------------------


def test_text_completion_reconstruction():
    agg = StreamAggregator(is_chat=False)
    agg.feed_event(
        {
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 500,
            "model": "qwen",
            "choices": [{"index": 0, "text": "hello", "finish_reason": None}],
        }
    )
    agg.feed_event({"choices": [{"index": 0, "text": " world", "finish_reason": "stop"}]})
    agg.feed_event(
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}}
    )
    out = agg.build()
    assert out["id"] == "cmpl-1"
    assert out["object"] == "text_completion"
    assert out["created"] == 500
    assert out["model"] == "qwen"
    assert out["usage"]["total_tokens"] == 4
    ch = out["choices"][0]
    assert ch["index"] == 0
    assert ch["text"] == "hello world"
    assert ch["finish_reason"] == "stop"


def test_text_completion_finish_reason_override():
    agg = StreamAggregator(is_chat=False)
    agg.feed_event(
        {
            "id": "cmpl-2",
            "model": "qwen",
            "created": 6,
            "choices": [{"index": 0, "text": "loop loop", "finish_reason": None}],
        }
    )
    out = agg.build(finish_reason_override="runaway_repeat")
    assert out["choices"][0]["finish_reason"] == "runaway_repeat"
    assert out["choices"][0]["text"] == "loop loop"


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------


def test_unrecognized_chunk_shape_is_skipped_not_fatal():
    agg = StreamAggregator(is_chat=True)
    agg.feed_event({"garbage": True})  # no choices key
    agg.feed_event({"choices": "not-a-list"})  # wrong type
    agg.feed_event(
        {
            "id": "chatcmpl-ok",
            "model": "qwen",
            "created": 7,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    out = agg.build()
    # Despite the junk events, the good content still reconstructs.
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_build_with_no_events_returns_empty_choices():
    agg = StreamAggregator(is_chat=True)
    out = agg.build()
    assert out["object"] == "chat.completion"
    assert out["choices"] == []
