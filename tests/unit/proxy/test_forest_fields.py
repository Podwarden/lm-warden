import json

from app.proxy import forest_fields as ff

SECRET = "s3cret"


def test_tool_family_maps_known_harness_names():
    assert ff.tool_family("Read") == "read"
    assert ff.tool_family("Grep") == "read"
    assert ff.tool_family("glob") == "read"
    assert ff.tool_family("Edit") == "edit"
    assert ff.tool_family("apply_patch") == "edit"
    assert ff.tool_family("Bash") == "shell"
    assert ff.tool_family("shell") == "shell"
    assert ff.tool_family("WebFetch") == "web"
    assert ff.tool_family("Task") == "agent"
    assert ff.tool_family("mcp__acme__thing") == "other"
    assert ff.tool_family(None) == "other"


def test_hkey_is_stable_hex_and_never_raw():
    a = ff.hkey(SECRET, "0b1c2d3e-1111-4222-8333-444455556666")
    assert a == ff.hkey(SECRET, "0b1c2d3e-1111-4222-8333-444455556666")
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)
    assert ff.hkey(SECRET, None) is None and ff.hkey(SECRET, "") is None
    assert ff.hkey("other", "x") != ff.hkey(SECRET, "x")


def test_clean_batch_id():
    assert ff.clean_batch_id(" farm-A27 ") == "farm-A27"
    assert ff.clean_batch_id("x" * 65) is None
    assert ff.clean_batch_id("bad\nid") is None
    assert ff.clean_batch_id("") is None and ff.clean_batch_id(None) is None


def _body():
    return {
        "messages": [
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 400},
            {"role": "tool", "tool_call_id": "c2", "content": "Error: exit 1"},
        ]
    }


def test_tools_in_reads_results_since_the_last_assistant_turn():
    got = ff.tools_in_from_body(_body())
    assert got == [["read", 100, False], ["shell", 4, True]]


def test_tools_in_tolerates_garbage():
    assert ff.tools_in_from_body(None) == []
    assert ff.tools_in_from_body({"messages": "nope"}) == []
    body = {
        "messages": [
            {"role": "tool", "tool_call_id": "zz", "content": [{"type": "text", "text": "abcd"}]}
        ]
    }
    assert ff.tools_in_from_body(body) == [["other", 1, False]]


def test_tools_out_from_message():
    msg = {"tool_calls": [{"function": {"name": "Edit"}}, {"function": {"name": "Grep"}}]}
    got = ff.tools_out_from_message(msg, SECRET)
    assert [g[0] for g in got] == ["edit", "read"]
    assert all(len(g[1]) == 8 for g in got)


def test_tools_out_tolerates_garbage():
    assert ff.tools_out_from_message(None, SECRET) == []
    assert ff.tools_out_from_message({"tool_calls": "x"}, SECRET) == []
    assert ff.tools_out_from_message({"tool_calls": [{"function": 5}]}, SECRET) == []


def _frame(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode()


def test_accumulator_collects_names_across_frames_and_ignores_others():
    acc = ff.ToolsOutAccumulator()
    acc.feed(_frame({"choices": [{"delta": {"content": "hi"}}]}))
    acc.feed(
        _frame(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"name": "Read", "arguments": ""}}
                            ]
                        }
                    }
                ]
            }
        )
    )
    acc.feed(
        _frame(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"p":1}'}}]}}
                ]
            }
        )
    )
    acc.feed(
        _frame(
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "Bash"}}]}}]}
        )
    )
    acc.feed(b"data: [DONE]")
    acc.feed(b'data: {not json "tool_calls"')
    assert [g[0] for g in acc.result(SECRET)] == ["read", "shell"]


def test_hkey_with_lone_surrogate_never_raises():
    # JSON "\ud800" is a lone surrogate that would raise without errors="surrogatepass"
    lone_surrogate_raw = "\ud800"
    result = ff.hkey(SECRET, lone_surrogate_raw)
    # Should return either a value or None, never raise
    assert result is None or (isinstance(result, str) and len(result) <= 16)


def test_accumulator_with_lone_surrogate_in_tool_name_never_raises():
    # Tool name with lone surrogate should not raise in result()
    acc = ff.ToolsOutAccumulator()
    # Build JSON with ensure_ascii=True to get the escape sequence
    obj = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "\ud800"}}]}}]}
    frame_bytes = b"data: " + json.dumps(obj, ensure_ascii=True).encode()
    acc.feed(frame_bytes)
    # result() should not raise, should return empty or valid result
    result = acc.result(SECRET)
    assert isinstance(result, list)


def test_accumulator_feed_ignores_non_bytes():
    acc = ff.ToolsOutAccumulator()
    # Feeding a string should not raise
    acc.feed("not bytes")  # type: ignore
    assert acc.result(SECRET) == []
    # Feeding None should not raise
    acc.feed(None)  # type: ignore
    assert acc.result(SECRET) == []


def test_tools_out_respects_max_out_limit():
    # Create a message with 20 tool calls, should cap at 16
    calls = [{"function": {"name": f"Tool{i}"}} for i in range(20)]
    msg = {"tool_calls": calls}
    got = ff.tools_out_from_message(msg, SECRET)
    assert len(got) <= ff._MAX_OUT
    assert len(got) == 16


def test_tools_in_respects_max_in_limit():
    # Create a body with 40 tool results, should cap at 32
    messages = [{"role": "user", "content": "fix it"}]
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
                for i in range(40)
            ],
        }
    )
    messages.extend([{"role": "tool", "tool_call_id": f"c{i}", "content": "x"} for i in range(40)])
    body = {"messages": messages}
    got = ff.tools_in_from_body(body)
    assert len(got) <= ff._MAX_IN
    assert len(got) == 32


def test_clean_batch_id_accepts_exactly_64_chars():
    batch_64 = "a" * 64
    result = ff.clean_batch_id(batch_64)
    assert result == batch_64
    # 65 chars should be rejected
    batch_65 = "a" * 65
    assert ff.clean_batch_id(batch_65) is None


def test_accumulator_result_never_contains_raw_tool_names():
    acc = ff.ToolsOutAccumulator()
    acc.feed(
        _frame(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "Read"}}]}}]}
        )
    )
    acc.feed(
        _frame(
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "Bash"}}]}}]}
        )
    )
    result = acc.result(SECRET)
    # result should be [[family, hashed_name], ...], not contain raw names
    assert result == [["read", ff.hkey(SECRET, "Read", 8)], ["shell", ff.hkey(SECRET, "Bash", 8)]]
    # Verify no raw name appears in result
    raw_names = {"Read", "Bash"}
    for item in result:
        assert item[1] not in raw_names


def test_accumulator_caps_names_at_max_out():
    acc = ff.ToolsOutAccumulator()
    # Feed 20 different tool names
    for i in range(20):
        frame = _frame(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": i, "function": {"name": f"Tool{i}"}}]}}
                ]
            }
        )
        acc.feed(frame)
    result = acc.result(SECRET)
    # Should cap at _MAX_OUT (16)
    assert len(result) <= ff._MAX_OUT
    assert len(result) == 16


def test_tools_out_from_message_skips_names_without_a_hash(monkeypatch):
    real = ff.hkey
    monkeypatch.setattr(ff, "hkey", lambda s, n, k=16: None if n == "Grep" else real(s, n, k))
    msg = {"tool_calls": [{"function": {"name": "Edit"}}, {"function": {"name": "Grep"}}]}
    assert [g[0] for g in ff.tools_out_from_message(msg, SECRET)] == ["edit"]
    acc = ff.ToolsOutAccumulator()
    for i, name in enumerate(("Edit", "Grep")):
        acc.feed(
            _frame(
                {"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"name": name}}]}}]}
            )
        )
    assert [g[0] for g in acc.result(SECRET)] == ["edit"]  # the same rule on both paths


def test_accumulator_accepts_a_data_line_with_leading_whitespace():
    acc = ff.ToolsOutAccumulator()
    frame = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "Bash"}}]}}]}
    acc.feed(b"  " + _frame(frame))
    assert [g[0] for g in acc.result(SECRET)] == ["shell"]
