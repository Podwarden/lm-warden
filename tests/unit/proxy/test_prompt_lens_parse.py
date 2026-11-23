"""G2: ``_parse_sse_prompt_lens`` returns a ``GmFrame`` with the tool-call deltas.

The parse is Prompt Lens-local: content, reasoning and finish_reason as before,
plus one ``ToolCallDelta`` per ``choices[0].delta.tool_calls`` entry, in the
order of the frame. ``ToolCallBudget`` is the per-request cap on what of
them gets published.
"""

import json

from app.proxy.prompt_lens_capture import GmFrame, ToolCallBudget, ToolCallDelta
from app.proxy.routes import _parse_sse_prompt_lens

EMPTY = GmFrame(None, None, None, ())


def _f(delta, **choice):
    return b"data: " + json.dumps({"choices": [{"delta": delta, **choice}]}).encode()


def test_tool_call_fragments_parse():
    fr = _parse_sse_prompt_lens(
        _f(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": ""}}
                ]
            }
        )
    )
    assert fr.tool_calls == (ToolCallDelta(0, "call_1", "bash", ""),)
    fr = _parse_sse_prompt_lens(
        _f({"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd"'}}]})
    )
    assert fr.tool_calls == (ToolCallDelta(0, None, None, '{"cmd"'),)


def test_parallel_calls_in_one_frame_keep_their_index():
    fr = _parse_sse_prompt_lens(
        _f(
            {
                "tool_calls": [
                    {"index": 1, "function": {"arguments": "b"}},
                    {"index": 0, "function": {"arguments": "a"}},
                ]
            }
        )
    )
    assert [(t.index, t.text) for t in fr.tool_calls] == [(1, "b"), (0, "a")]


def test_content_reasoning_finish_unchanged():
    fr = _parse_sse_prompt_lens(
        b'data: {"choices":[{"delta":{"content":"hi","reasoning_content":"r"},"finish_reason":"stop"}]}'
    )
    assert (fr.content, fr.reasoning, fr.finish_reason, fr.tool_calls) == ("hi", "r", "stop", ())


def test_reasoning_field_name_is_read_too():
    fr = _parse_sse_prompt_lens(_f({"reasoning": "r2"}))
    assert fr == GmFrame(None, "r2", None, ())


def test_garbage_and_done():
    assert _parse_sse_prompt_lens(b"data: [DONE]") == EMPTY
    assert _parse_sse_prompt_lens(b"data: {bad") == EMPTY
    assert _parse_sse_prompt_lens(b": keepalive") == EMPTY
    assert _parse_sse_prompt_lens(b"event: x") == EMPTY
    # A usage-only frame (stream_options.include_usage) has no choices.
    assert _parse_sse_prompt_lens(b'data: {"choices":[],"usage":{"completion_tokens":3}}') == EMPTY
    assert _parse_sse_prompt_lens(b'data: {"usage":{"completion_tokens":3}}') == EMPTY
    assert _parse_sse_prompt_lens(b"data: [1, 2]") == EMPTY


def test_finish_tool_calls_frame():
    fr = _parse_sse_prompt_lens(_f({}, finish_reason="tool_calls"))
    assert fr == GmFrame(None, None, "tool_calls", ())


def test_non_int_index_falls_back_to_the_position_in_the_frame():
    fr = _parse_sse_prompt_lens(
        _f(
            {
                "tool_calls": [
                    {"index": 5, "function": {"arguments": "x"}},
                    {"index": "1", "function": {"arguments": "y"}},
                    {"function": {"arguments": "z"}},
                    {"index": True, "function": {"arguments": "w"}},
                ]
            }
        )
    )
    assert [(t.index, t.text) for t in fr.tool_calls] == [(5, "x"), (1, "y"), (2, "z"), (3, "w")]


def test_malformed_entries_are_skipped_and_odd_values_tolerated():
    fr = _parse_sse_prompt_lens(
        _f(
            {
                "content": "c",
                "tool_calls": [
                    "nope",
                    {"index": 1, "id": 7, "function": "nope"},
                    {"index": 2, "function": {"name": 3, "arguments": {"a": 1}}},
                    {"index": 3, "function": {"arguments": None}},
                ],
            }
        )
    )
    assert fr.content == "c"
    assert fr.tool_calls == (
        ToolCallDelta(1, None, None, ""),
        ToolCallDelta(2, None, None, '{"a":1}'),
        ToolCallDelta(3, None, None, ""),
    )


def test_tool_calls_not_a_list_gives_none():
    fr = _parse_sse_prompt_lens(_f({"content": "c", "tool_calls": {"index": 0}}))
    assert fr == GmFrame("c", None, None, ())


def test_bad_tool_calls_never_lose_the_other_channels():
    fr = _parse_sse_prompt_lens(
        _f(
            {"content": "c", "reasoning_content": "r", "tool_calls": [None, 1]},
            finish_reason="stop",
        )
    )
    assert fr == GmFrame("c", "r", "stop", ())


# --------------------------------------------------------------------------
# ToolCallBudget: per-request bound on published tool-call text
# --------------------------------------------------------------------------
def test_budget_passes_everything_that_fits():
    b = ToolCallBudget(100)
    assert b.take(ToolCallDelta(0, "call_1", "bash", "")) == {
        "index": 0,
        "call_id": "call_1",
        "name": "bash",
        "text": "",
    }
    assert b.take(ToolCallDelta(0, None, None, '{"a"')) == {"index": 0, "text": '{"a"'}
    assert (b.dropped_chars, b.dropped_calls) == (0, 0)


def test_budget_skips_empty_fragments():
    b = ToolCallBudget(100)
    assert b.take(ToolCallDelta(0, None, None, "")) is None
    assert (b.dropped_chars, b.dropped_calls) == (0, 0)


def test_budget_cuts_the_fragment_that_crosses_it_and_counts_the_rest():
    # Budget counts call_id + name + text: 6 + 4 = 10 for the opener.
    b = ToolCallBudget(15)
    assert b.take(ToolCallDelta(0, "call_1", "bash", "")) is not None
    cut = b.take(ToolCallDelta(0, None, None, "0123456789"))
    assert cut == {"index": 0, "text": "01234", "elided": True}
    assert b.take(ToolCallDelta(0, None, None, "abc")) is None
    assert b.take(ToolCallDelta(1, "call_2", "grep", "xy")) is None
    assert b.dropped_chars == 5 + 3 + 2
    assert b.dropped_calls == 1  # call_2 never published


def test_budget_an_opener_that_does_not_fit_is_dropped_whole():
    b = ToolCallBudget(5)
    assert b.take(ToolCallDelta(0, "call_1", "bash", "{}")) is None
    assert (b.dropped_chars, b.dropped_calls) == (2, 1)


def test_budget_fields():
    b = ToolCallBudget(4)
    assert b.end_fields() == {}
    b.take(ToolCallDelta(0, None, None, "abcdef"))
    assert b.end_fields() == {"tool_call_chars_dropped": 2}
    b.take(ToolCallDelta(1, "c", "n", ""))
    assert b.end_fields() == {"tool_call_chars_dropped": 2, "tool_calls_dropped": 1}


def test_budget_a_dropped_opener_ends_the_capture():
    """Review I1: once an opener is dropped, nothing after it is published:
    neither its own later fragments (orphans with no name) nor a later,
    shorter opener."""
    b = ToolCallBudget(5)
    assert b.take(ToolCallDelta(0, "call_1", "bash", "")) is None
    assert b.take(ToolCallDelta(0, None, None, '{"x":1}')) is None
    assert b.take(ToolCallDelta(1, "c", "n", "{}")) is None
    assert b.take(ToolCallDelta(1, None, None, "zz")) is None
    assert b.dropped_chars == 7 + 2 + 2
    assert b.dropped_calls == 2
    assert b.end_fields() == {"tool_call_chars_dropped": 11, "tool_calls_dropped": 2}


def test_budget_after_a_dropped_opener_earlier_calls_are_dropped_too():
    b = ToolCallBudget(12)
    assert b.take(ToolCallDelta(0, "c0", "n", "abc")) is not None  # 6 used, 6 left
    assert b.take(ToolCallDelta(1, "call_1", "long_name", "")) is None  # 15 > 6
    # Call 0 keeps streaming, but the capture has ended: one clean tail.
    assert b.take(ToolCallDelta(0, None, None, "d")) is None
    assert (b.dropped_chars, b.dropped_calls) == (1, 1)


def test_budget_an_opener_that_uses_the_budget_exactly_is_elided():
    """Review M2: an opener that leaves 0 marks itself elided, so the call shows
    as trimmed even though none of its arguments got through."""
    b = ToolCallBudget(10)
    assert b.take(ToolCallDelta(0, "call_1", "bash", "")) == {
        "index": 0,
        "call_id": "call_1",
        "name": "bash",
        "text": "",
        "elided": True,
    }
    assert b.take(ToolCallDelta(0, None, None, "{}")) is None
    assert (b.dropped_chars, b.dropped_calls) == (2, 0)


def test_budget_an_opener_that_fits_with_room_is_not_elided():
    b = ToolCallBudget(11)
    assert "elided" not in b.take(ToolCallDelta(0, "call_1", "bash", ""))
