"""Per-call verdicts and reply counts (analysis spec 8.1, plan Task 1.4)."""

import json

import pytest

from app.lens.reply_verdicts import (
    CALL_CODES,
    MAX_VERDICTS,
    call_verdicts,
    judge_calls,
    offered_of,
    reply_counts,
)
from app.proxy.forest_fields import hkey

CANARY = "CANARY-1d4b7e-args"

OFFERED = {
    "search_docs": {
        "type": "object",
        "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["q", "limit"],
    },
    "read_file": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
        "required": ["path"],
    },
    "ping": {"type": "object", "properties": {}},
}


def call(name, args, cid="c"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def codes(v):
    return [e[-1] for e in v]


def test_codes_are_the_spec_codes():
    assert CALL_CODES == ("ok", "json", "unknown_tool", "missing_arg", "type", "leaked_markup")


def test_parallel_calls_each_get_a_verdict():  # Review Focus 2
    msg = {
        "tool_calls": [
            call("search_docs", '{"q":"x"}'),
            call("nope", "{}"),
            call("search_docs", "{bad"),
        ]
    }
    v = call_verdicts(msg, OFFERED, "s")
    assert codes(v) == ["missing_arg", "unknown_tool", "json"]
    # [k, hash, code]: the call's index in reply order, the tools_out name
    # hash (same secret, 8 hex).
    assert v == [
        [0, hkey("s", "search_docs", 8), "missing_arg"],
        [1, hkey("s", "nope", 8), "unknown_tool"],
        [2, hkey("s", "search_docs", 8), "json"],
    ]


def test_ok_calls():
    msg = {
        "tool_calls": [
            call("search_docs", '{"q":"x","limit":3}'),
            call("read_file", '{"path":"a"}'),
            call("ping", ""),
            call("ping", None),
        ]
    }
    assert codes(judge_calls(msg, OFFERED, "s")) == ["ok", "ok", "ok", "ok"]


def test_type_mismatch():
    msg = {"tool_calls": [call("search_docs", '{"q":"x","limit":"3"}')]}
    assert codes(judge_calls(msg, OFFERED, "s")) == ["type"]
    # A bool is not an integer; an integral float is.
    msg = {"tool_calls": [call("search_docs", '{"q":"x","limit":true}')]}
    assert codes(judge_calls(msg, OFFERED, "s")) == ["type"]
    msg = {"tool_calls": [call("search_docs", '{"q":"x","limit":3.0}')]}
    assert codes(judge_calls(msg, OFFERED, "s")) == ["ok"]


def test_type_union_and_number():
    off = {"f": {"type": "object", "properties": {"n": {"type": ["number", "null"]}}}}
    assert codes(judge_calls({"tool_calls": [call("f", '{"n":1.5}')]}, off, "s")) == ["ok"]
    assert codes(judge_calls({"tool_calls": [call("f", '{"n":null}')]}, off, "s")) == ["ok"]
    assert codes(judge_calls({"tool_calls": [call("f", '{"n":"x"}')]}, off, "s")) == ["type"]


def test_arguments_not_an_object_is_json():
    msg = {"tool_calls": [call("read_file", "[1,2]")]}
    assert codes(judge_calls(msg, OFFERED, "s")) == ["json"]


def test_arguments_already_parsed():
    msg = {"tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]}
    assert codes(judge_calls(msg, OFFERED, "s")) == ["ok"]


def test_no_offered_tools_makes_every_call_unknown():
    assert codes(judge_calls({"tool_calls": [call("read_file", "{}")]}, {}, "s")) == [
        "unknown_tool"
    ]


def test_missing_name_is_unknown_without_a_hash():
    v = call_verdicts({"tool_calls": [{"function": {"arguments": "{}"}}]}, OFFERED, "s")
    assert v == [[0, "", "unknown_tool"]]


def test_anthropic_tool_use_translated_reply_is_judged():
    """/v1/messages: Anthropic tools in the request become OpenAI tools in the
    chat body, the reply's tool calls OpenAI tool_calls (messages_translate)."""
    from app.proxy.messages_schemas import MessagesRequest
    from app.proxy.messages_translate import to_openai_request

    req = {
        "model": "qwen",
        "max_tokens": 64,
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": OFFERED["read_file"],
            }
        ],
        "messages": [{"role": "user", "content": "go"}],
    }
    body = to_openai_request(MessagesRequest.model_validate(req))
    off = offered_of(body.get("tools"))
    assert set(off) == {"read_file"}
    msg = {"tool_calls": [call("read_file", '{"lines":2}'), call("read_file", '{"path":"a"}')]}
    assert codes(judge_calls(msg, off, "s")) == ["missing_arg", "ok"]


def test_offered_of_reads_both_wire_shapes_and_garbage():
    tools = [
        {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
        {"name": "b", "input_schema": {"type": "object"}},
        {"type": "function", "function": {"name": "c"}},
        "junk",
        {"function": 5},
    ]
    off = offered_of(tools)
    assert off == {"a": {"type": "object"}, "b": {"type": "object"}, "c": {}}
    assert offered_of(None) == {} and offered_of("x") == {}


def test_leaked_markup_in_text_is_flagged():
    v = call_verdicts({"content": '<tool_call>{"name":"x"}</tool_call>'}, OFFERED, "s")
    # Not a call: no index.
    assert v == [[None, hkey("s", "x", 8), "leaked_markup"]]


@pytest.mark.parametrize(
    "text",
    [
        "<|tool_call|>",
        "[TOOL_CALLS] [{}]",
        "<|python_tag|>search()",
        "<function=read_file>{}</function>",
        "<|tool_calls_section_begin|>",
        "<tool▁calls▁begin>",
        '<invoke name="read_file">',
    ],
)
def test_leaked_markup_forms(text):
    assert codes(judge_calls({"content": f"ok then {text}"}, OFFERED, "s")) == ["leaked_markup"]


def test_plain_text_is_no_verdict():
    assert call_verdicts({"content": "All done, the tool call worked."}, OFFERED, "s") == []
    assert call_verdicts({"content": None}, OFFERED, "s") == []


def test_markup_in_reasoning_is_not_leaked():
    msg = {"content": "done", "reasoning_content": "<tool_call>{}</tool_call>"}
    assert call_verdicts(msg, OFFERED, "s") == []


def test_leaked_markup_after_the_calls():
    msg = {"content": [{"type": "text", "text": "<tool_call>"}], "tool_calls": [call("ping", "{}")]}
    assert codes(judge_calls(msg, OFFERED, "s")) == ["ok", "leaked_markup"]


def test_at_most_16_verdicts():
    msg = {"tool_calls": [call("nope", "{}") for _ in range(40)], "content": "<tool_call>"}
    v = call_verdicts(msg, OFFERED, "s")
    assert len(v) == MAX_VERDICTS == 16 and [e[0] for e in v] == list(range(16))


def test_ok_calls_are_never_stored():
    msg = {"tool_calls": [call("ping", "{}") for _ in range(4)]}
    assert call_verdicts(msg, OFFERED, "s") == []


def test_ok_verdicts_are_filtered_before_the_cap():
    """I4: 16 ok calls then 2 bad ones store the 2 bad, with their indices."""
    msg = {"tool_calls": [call("ping", "{}") for _ in range(16)] + [call("nope", "{}")] * 2}
    h = hkey("s", "nope", 8)
    assert call_verdicts(msg, OFFERED, "s") == [[16, h, "unknown_tool"], [17, h, "unknown_tool"]]


def test_parallel_same_tool_second_call_bad_keeps_its_index():
    msg = {
        "tool_calls": [
            call("search_docs", '{"q":"x","limit":1}'),
            call("search_docs", '{"q":"x"}'),
        ]
    }
    assert call_verdicts(msg, OFFERED, "s") == [[1, hkey("s", "search_docs", 8), "missing_arg"]]


def test_index_counts_every_tool_calls_entry():
    """k is the position in the reply's tool_calls (the request view's index),
    so an entry that is not a call still takes its place."""
    msg = {"tool_calls": [5, call("nope", "{}")]}
    assert call_verdicts(msg, OFFERED, "s") == [[1, hkey("s", "nope", 8), "unknown_tool"]]


def test_garbage_never_raises():
    for m in (None, {}, {"tool_calls": "x"}, {"tool_calls": [5, None, {"function": 3}]}, []):
        assert isinstance(call_verdicts(m, OFFERED, "s"), list)  # type: ignore[arg-type]
    assert call_verdicts({"tool_calls": [call("read_file", "{}")]}, None, "s") == [  # type: ignore[arg-type]
        [0, hkey("s", "read_file", 8), "unknown_tool"]
    ]


def test_no_argument_text_in_verdicts():
    msg = {
        "content": f"<tool_call>{CANARY}",
        "tool_calls": [
            call("read_file", json.dumps({"path": CANARY})),
            call(CANARY, "{}"),
            call("search_docs", "{" + CANARY),
        ],
    }
    v = call_verdicts(msg, OFFERED, "s")
    assert CANARY not in json.dumps(v)
    assert all(len(h) in (0, 8) and c in CALL_CODES for _k, h, c in v)


def test_reply_counts_reasoning_only_is_empty():
    assert reply_counts({"content": "", "reasoning_content": "hmm"}) == (0, 0)


def test_reply_counts():
    assert reply_counts({"content": "hello"}) == (5, 0)
    assert reply_counts({"content": None, "tool_calls": [call("a", "{}"), call("b", "{}")]}) == (
        0,
        2,
    )
    assert reply_counts(
        {"content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]}
    ) == (3, 0)
    assert reply_counts({}) == (0, 0)
    assert reply_counts(None) == (0, 0)  # type: ignore[arg-type]
