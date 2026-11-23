"""Prompt Lens structured prompt capture (G1): roles, tools, tool calls and
results, images, windowed per message and bounded as a whole.

Display only: the capture is built from a tapped view of the client's body and
never touches the forwarded request.
"""

import copy
import json
import sys

from app.proxy.prompt_lens_capture import (
    COLLAPSED_LIST_MAX,
    ENTRY_CHARS,
    body_layout,
    split_head_body,
    structured_prompt,
)


def test_head_roles_form_sys_and_body_is_numbered():
    body = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "u0"},
            {"role": "developer", "content": "D"},
            {"role": "assistant", "content": "a1"},
        ]
    }
    p = structured_prompt(body, budget_chars=16000)
    assert p["sys"] == {"chars": 4, "text": "S\n\nD", "elided": False}
    assert [(m["i"], m["role"], m["text"]) for m in p["messages"]] == [
        (0, "user", "u0"),
        (1, "assistant", "a1"),
    ]
    assert p["collapsed"] == 0
    assert p["older"] == {"from": 0, "roles": "", "chars": []}
    assert p["total_chars"] == 4 + 2 + 2


def test_no_head_means_sys_none_and_no_tools():
    p = structured_prompt({"messages": [{"role": "user", "content": "hi"}]}, budget_chars=16000)
    assert p["sys"] is None
    assert p["tools"] == {"count": 0, "chars": 0, "names": []}
    assert p["messages"] == [{"i": 0, "role": "user", "chars": 2, "text": "hi", "elided": False}]


def test_tool_calls_and_results_keep_ids_and_names():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    body = {
        "tools": tools,
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_8f2a", "function": {"name": "read_file", "arguments": '{"p":1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_8f2a", "content": "file text"},
        ],
    }
    p = structured_prompt(body, budget_chars=16000)
    assert p["tools"] == {
        "count": 1,
        "chars": len(json.dumps(tools[0], separators=(",", ":"), ensure_ascii=False)),
        "names": ["read_file"],
    }
    call = p["messages"][1]
    assert call["tool_calls"] == [{"id": "call_8f2a", "name": "read_file", "args": '{"p":1}'}]
    assert call["text"] == "" and call["chars"] == len('{"p":1}')
    result = p["messages"][2]
    assert result["tool_call_id"] == "call_8f2a" and result["tool_name"] == "read_file"
    assert result["text"] == "file text"


def test_tool_names_fall_back_to_top_level_name():
    body = {
        "tools": [{"name": "grep"}, {"type": "function", "function": {"name": "ls"}}, "junk"],
        "messages": [{"role": "user", "content": "x"}],
    }
    assert structured_prompt(body, budget_chars=16000)["tools"]["names"] == ["grep", "ls"]
    assert structured_prompt(body, budget_chars=16000)["tools"]["count"] == 3


def test_tool_result_with_unknown_call_id_has_no_name():
    body = {"messages": [{"role": "tool", "tool_call_id": "nope", "content": "r"}]}
    m = structured_prompt(body, budget_chars=16000)["messages"][0]
    assert m["tool_call_id"] == "nope" and "tool_name" not in m


def test_budget_keeps_newest_whole_and_collapses_older():
    msgs = [{"role": "user", "content": "x" * 5000} for _ in range(10)]
    p = structured_prompt({"messages": msgs}, budget_chars=16000)
    kept = p["messages"]
    assert kept[-1]["i"] == 9 and len(kept) == 3 and p["collapsed"] == 7
    assert [m["i"] for m in kept] == [7, 8, 9]
    assert all(m["chars"] == 5000 and m["text"] == "x" * 5000 for m in kept)
    # Collapsed messages: only role and size, as compact parallel arrays.
    assert p["older"] == {"from": 0, "roles": "u" * 7, "chars": [5000] * 7}
    assert p["omitted"] == 0
    assert p["total_chars"] == 50_000


def test_newest_message_always_kept_and_windowed():
    p = structured_prompt(
        {"messages": [{"role": "user", "content": "y" * 50_000}]}, budget_chars=16000
    )
    m = p["messages"][0]
    assert m["elided"] is True and len(m["text"]) < 6100
    assert m["chars"] == 50_000
    # The tail (msg_chars // 4) survives the window.
    assert m["text"].endswith("y" * 1500)


def test_newest_kept_even_over_a_tiny_budget():
    p = structured_prompt(
        {
            "messages": [
                {"role": "user", "content": "a" * 100},
                {"role": "user", "content": "b" * 100},
            ]
        },
        budget_chars=10,
    )
    assert [m["i"] for m in p["messages"]] == [1]
    assert p["older"] == {"from": 0, "roles": "u", "chars": [100]}


def test_sys_is_windowed_with_its_own_cap():
    p = structured_prompt(
        {
            "messages": [
                {"role": "system", "content": "s" * 10_000},
                {"role": "user", "content": "q"},
            ]
        },
        budget_chars=16000,
    )
    assert p["sys"]["chars"] == 10_000 and p["sys"]["elided"] is True
    assert len(p["sys"]["text"]) < 2100 and p["sys"]["text"].endswith("s" * 500)


def test_images_are_counted_not_copied():
    content = [
        {"type": "text", "text": "see"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "input_image", "image_url": "data:image/png;base64,BBBB"},
    ]
    p = structured_prompt({"messages": [{"role": "user", "content": content}]}, budget_chars=16000)
    m = p["messages"][0]
    assert m["images"] == 2 and m["text"] == "see" and "base64" not in json.dumps(p)
    assert m["chars"] == 3


def test_chars_of_multi_part_content_match_the_joined_text():
    content = [
        {"type": "text", "text": "ab"},
        "cd",
        {"text": "ef"},
        {"type": "image", "text": "no"},
    ]
    m = structured_prompt({"messages": [{"role": "user", "content": content}]}, budget_chars=16000)[
        "messages"
    ][0]
    assert m["text"] == "ab\ncd\nef" and m["chars"] == len(m["text"])


def test_not_a_chat_body_returns_none():
    assert structured_prompt({"prompt": "legacy"}, budget_chars=16000) is None
    assert structured_prompt("not a dict", budget_chars=16000) is None  # type: ignore[arg-type]
    assert structured_prompt({"messages": "nope"}, budget_chars=16000) is None


def test_malformed_parts_never_raise():
    body = {
        "tools": [{"function": "x"}, None],
        "messages": [
            "not a dict",
            {"role": 7, "content": {"weird": True}},
            {
                "role": "assistant",
                "tool_calls": [
                    None,
                    {"function": None},
                    {"id": 3, "function": {"arguments": {"a": 1}}},
                ],
            },
            {"role": "tool", "tool_call_id": None, "content": [None, 5, "s"]},
        ],
    }
    p = structured_prompt(body, budget_chars=16000)
    assert p is not None
    # A non-dict message is skipped (as the canonical form does), so i stays the node index.
    assert [m["i"] for m in p["messages"]] == [0, 1, 2]
    assert p["messages"][0]["role"] == ""
    assert p["messages"][1]["tool_calls"] == [
        {"id": None, "name": "", "args": ""},
        {"id": None, "name": "", "args": '{"a":1}'},
    ]


def test_capture_never_mutates_the_body():
    body = {
        "tools": [{"type": "function", "function": {"name": "t"}}],
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 50_000}],
    }
    before = copy.deepcopy(body)
    structured_prompt(body, budget_chars=16000)
    assert body == before


def test_twenty_thousand_messages_stay_in_budget():
    msgs = [{"role": "user", "content": "z" * 100} for _ in range(20_000)]
    p = structured_prompt({"messages": msgs}, budget_chars=16000)
    assert sum(len(m["text"]) for m in p["messages"]) <= 16000 + 6000
    assert len(p["messages"]) == 160 and p["messages"][0]["i"] == 19_840
    assert p["collapsed"] == 20_000 - 160
    # The collapsed list is bounded: the newest COLLAPSED_LIST_MAX collapsed
    # messages are listed, the older rest only counted.
    older = p["older"]
    assert len(older["roles"]) == len(older["chars"]) == COLLAPSED_LIST_MAX
    assert older["from"] == p["omitted"] == 20_000 - 160 - COLLAPSED_LIST_MAX
    assert older["from"] + COLLAPSED_LIST_MAX == p["messages"][0]["i"]
    assert p["messages"][-1]["i"] == 19_999
    assert p["total_chars"] == 2_000_000


def test_split_head_body_keeps_raw_indices():
    msgs = [
        {"role": "system", "content": "S"},
        "junk",
        {"role": "user", "content": "u"},
        {"role": "developer", "content": "D"},
        {"role": "assistant", "content": "a"},
    ]
    head, body = split_head_body(msgs)
    assert [i for i, _ in head] == [0, 3]
    assert [i for i, _ in body] == [2, 4]
    assert [m["role"] for _, m in body] == ["user", "assistant"]


def test_section_prompt_chars_setting(monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.delenv("VW_PROMPT_LENS_SECTION_PROMPT_CHARS", raising=False)
    assert load_settings().prompt_lens_section_prompt_chars == 256_000
    monkeypatch.setenv("VW_PROMPT_LENS_SECTION_PROMPT_CHARS", "1234")
    assert load_settings().prompt_lens_section_prompt_chars == 1234


def _heap(obj):
    """A rough deep size of JSON-like data (dicts, lists, strings, ints)."""
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(_heap(k) + _heap(v) for k, v in obj.items())
    if isinstance(obj, list):
        return sys.getsizeof(obj) + sum(_heap(v) for v in obj)
    return sys.getsizeof(obj)


def test_collapsed_part_of_a_twenty_thousand_message_start_is_small():
    roles = ("user", "assistant", "tool")
    msgs = [{"role": roles[n % 3], "content": "w" * (n % 9000)} for n in range(20_000)]
    p = structured_prompt({"messages": msgs}, budget_chars=16000)
    collapsed_part = {k: p[k] for k in ("older", "collapsed", "omitted")}
    assert len(json.dumps(collapsed_part, separators=(",", ":"))) < 16_000
    assert _heap(collapsed_part) < 16_000
    assert set(p["older"]["roles"]) == {"u", "a", "t"}


def test_role_codes():
    msgs = [
        {"role": r, "content": "c" * 10} for r in ("user", "assistant", "tool", "function", "", "x")
    ]
    msgs.append({"role": "user", "content": "z" * 100})
    p = structured_prompt({"messages": msgs}, budget_chars=50)
    assert p["older"] == {"from": 0, "roles": "uatf??", "chars": [10] * 6}


def test_many_tool_calls_stay_within_the_entry_budget():
    calls = [
        {"id": f"call_{n}", "function": {"name": "w", "arguments": "a" * 6000}} for n in range(500)
    ]
    body = {"messages": [{"role": "assistant", "content": "t" * 6000, "tool_calls": calls}]}
    p = structured_prompt(body, budget_chars=16000)
    m = p["messages"][0]
    captured = len(m["text"]) + sum(len(c["args"]) for c in m["tool_calls"])
    assert captured <= ENTRY_CHARS
    assert m["calls_omitted"] == 500 - len(m["tool_calls"]) > 0
    # Calls are listed oldest first, up to the budget.
    assert [c["id"] for c in m["tool_calls"]] == [f"call_{n}" for n in range(len(m["tool_calls"]))]
    assert m["chars"] == 6000 + 500 * 6000 and m["elided"] is True
    assert len(json.dumps(p)) < ENTRY_CHARS + 4000


def test_a_few_tool_calls_have_no_calls_omitted():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "function": {"name": "n", "arguments": "{}"}}],
            }
        ]
    }
    assert "calls_omitted" not in structured_prompt(body, budget_chars=16000)["messages"][0]


def test_reasoning_is_captured_windowed_and_counted():
    body = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "ans", "reasoning_content": "r" * 10_000},
            {"role": "assistant", "content": "b", "reasoning": "why"},
            {"role": "user", "content": "q2", "reasoning": "ignored on a user message"},
        ]
    }
    p = structured_prompt(body, budget_chars=16000)
    a1, a2, u = p["messages"][1], p["messages"][2], p["messages"][3]
    assert a1["chars"] == 3 + 10_000 and a1["elided"] is True
    assert len(a1["reasoning"]) < 6100 and a1["reasoning"].endswith("r" * 1500)
    assert a1["text"] == "ans"
    assert a2["reasoning"] == "why" and a2["chars"] == 4 and a2["elided"] is False
    assert "reasoning" not in u and u["chars"] == 2
    assert p["total_chars"] == 1 + 10_003 + 4 + 2


def test_reasoning_counts_against_the_budget():
    msgs = [
        {"role": "assistant", "content": "", "reasoning_content": "r" * 5000} for _ in range(10)
    ]
    p = structured_prompt({"messages": msgs}, budget_chars=16000)
    assert len(p["messages"]) == 3 and p["collapsed"] == 7
    assert p["older"]["chars"] == [5000] * 7


def test_body_layout_matches_the_capture_sizes():
    body = {
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "messages": [
            {"role": "system", "content": "S"},
            "not a message",
            {
                "role": "user",
                "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}],
            },
            {"role": "developer", "content": "DD"},
            {
                "role": "assistant",
                "content": "x",
                "reasoning_content": "think",
                "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
    }
    lay = body_layout(body)
    assert lay is not None
    assert lay.raw_body_index == [2, 4, 5]
    assert lay.raw_head_index == [0, 3]
    assert lay.roles == ["user", "assistant", "tool"]
    assert lay.head_roles == ["system", "developer"]
    assert lay.parts_chars == [4, 1 + 5 + 2, 6]
    p = structured_prompt(body, budget_chars=16000)
    assert lay.parts_chars == [m["chars"] for m in p["messages"]]
    assert lay.sys_chars == p["sys"]["chars"] == len("S\n\nDD")
    assert lay.tools_chars == p["tools"]["chars"]


def test_body_layout_of_a_non_chat_body_is_none():
    assert body_layout({"prompt": "hi"}) is None
    assert body_layout([]) is None
    lay = body_layout({"messages": []})
    assert lay is not None and lay.raw_body_index == [] and lay.sys_chars == 0
