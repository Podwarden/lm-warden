"""The canonical message form (spec section 6.1-6.2): exact and loose."""

import base64
import hashlib
import json
import time
from typing import Any

from app.cobs2.canonical import LOOSE_TABLE_VERSION, Canon, canon_json, canonicalize, loose
from app.proxy.messages_schemas import MessagesRequest
from app.proxy.messages_translate import to_openai_request as messages_to_oai
from app.proxy.responses_schemas import ResponsesRequest
from app.proxy.responses_translate import to_openai_request as responses_to_oai

# key order as the Anthropic and Responses translators emit it
TOOL = {
    "type": "function",
    "function": {
        "name": "ls",
        "parameters": {"type": "object", "properties": {"p": {"type": "string"}}},
        "description": "list",
    },
}


def _canon(body: dict[str, Any]):
    c = canonicalize(body)
    assert c is not None
    return c


def _chat(call_id: str = "call_1") -> dict[str, Any]:
    return {
        "model": "m",
        "tools": [TOOL],
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "ls", "arguments": '{"p": "/", "a": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "out"},
            {"role": "user", "content": "thanks"},
        ],
    }


def test_parts_shape_text_media_tool_call_result_reasoning():
    png = base64.b64encode(b"PNGDATA").decode()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
                ],
            },
            {
                "role": "assistant",
                "content": "x",
                "reasoning_content": "thinking",
                "tool_calls": [
                    {"id": "c9", "function": {"name": "ls", "arguments": '{"b":2,"a":1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c9", "content": "res"},
        ]
    }
    c = _canon(body)
    assert c.messages[0] == {
        "role": "user",
        "parts": [
            {"t": "text", "v": "look"},
            {"t": "media", "sha256": hashlib.sha256(b"PNGDATA").hexdigest(), "mime": "image/png"},
        ],
    }
    assert c.messages[1]["parts"] == [
        {"t": "reasoning", "v": "thinking"},
        {"t": "text", "v": "x"},
        {"t": "tool_call", "name": "ls", "args": '{"b":2,"a":1}', "ref": 0},
    ]
    assert c.messages[2] == {
        "role": "tool",
        "parts": [{"t": "tool_result", "ref": 0, "parts": [{"t": "text", "v": "res"}]}],
    }


def test_unparseable_tool_args_keep_raw_string():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{oops"}}],
            }
        ]
    }
    assert _canon(body).messages[0]["parts"] == [
        {"t": "tool_call", "name": "f", "args": "{oops", "ref": 0}
    ]


def test_tool_call_ids_do_not_enter_hashes():
    a = _canon(_chat("call_AAA"))
    b = _canon(_chat("toolu_ZZZ"))
    assert canon_json(a.messages) == canon_json(b.messages)
    assert b"AAA" not in canon_json(a.messages)


def test_ref_is_local_so_a_message_alone_equals_it_in_context():
    reply = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "x1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "x2", "function": {"name": "b", "arguments": "{}"}},
                ],
            }
        ]
    }
    ctx = {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "first",
                "tool_calls": [{"id": "old", "function": {"name": "z", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "old", "content": "r"},
            reply["messages"][0],
            {"role": "tool", "tool_call_id": "x2", "content": "r2"},
            {"role": "tool", "tool_call_id": "nope", "content": "r3"},
        ]
    }
    alone = _canon(reply).messages[0]
    c = _canon(ctx)
    assert c.messages[3] == alone
    assert [p["ref"] for p in alone["parts"]] == [0, 1]
    assert c.messages[2]["parts"][0]["ref"] == 0
    assert c.messages[4]["parts"][0]["ref"] == 1
    assert c.messages[5]["parts"][0]["ref"] == -1


def test_same_conversation_three_apis_same_ids():
    chat = {
        "model": "m",
        "tools": [TOOL],
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": '{"p": "/"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"role": "user", "content": "thanks"},
        ],
    }
    anth = MessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 10,
            "system": "S",
            "tools": [
                {
                    "name": "ls",
                    "description": "list",
                    "input_schema": TOOL["function"]["parameters"],
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {"type": "tool_use", "id": "toolu_9", "name": "ls", "input": {"p": "/"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_9", "content": "out"}
                    ],
                },
                {"role": "user", "content": "thanks"},
            ],
        }
    )
    resp = ResponsesRequest.model_validate(
        {
            "model": "m",
            "instructions": "S",
            "tools": [
                {
                    "type": "function",
                    "name": "ls",
                    "description": "list",
                    "parameters": TOOL["function"]["parameters"],
                }
            ],
            "input": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {
                    "type": "function_call",
                    "call_id": "fc_7",
                    "name": "ls",
                    "arguments": '{"p":"/"}',
                },
                {"type": "function_call_output", "call_id": "fc_7", "output": "out"},
                {"role": "user", "content": "thanks"},
            ],
        }
    )
    a = _canon(chat)
    b = _canon(messages_to_oai(anth))
    c = _canon(responses_to_oai(resp).oai)
    assert a == b == c


def test_head_is_system_developer_and_tools():
    body = {
        "tools": [TOOL, {"type": "function", "function": {"name": "b"}}],
        "messages": [
            {"role": "system", "content": "S1"},
            {"role": "developer", "content": [{"type": "text", "text": "D1"}]},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "S2"},
        ],
    }
    c = _canon(body)
    assert [m["role"] for m in c.head] == ["system", "developer", "system"]
    assert c.head[1]["parts"] == [{"t": "text", "v": "D1"}]
    assert c.tools == [json.dumps(t, separators=(",", ":")) for t in body["tools"]]
    assert [m["role"] for m in c.messages] == ["user"]


def test_loose_drops_reasoning_reminders_dates_and_collapses_whitespace():
    assert LOOSE_TABLE_VERSION == 1
    body = {
        "messages": [
            {"role": "system", "content": "  be   nice \n"},
            {
                "role": "user",
                "content": "<system-reminder>\nnoise\n</system-reminder>hello\n\n   world\n"
                "2026-10-25T10:11:12Z\n<environment_context>cwd</environment_context>",
            },
            {"role": "user", "content": "Change the greeting to be more casual please"},
            {"role": "assistant", "content": "ans  wer", "reasoning_content": "hmm"},
            {"role": "tool", "tool_call_id": "q", "content": "  r \t x "},
        ]
    }
    exact = _canon(body)
    lo = loose(exact)
    assert lo.head[0]["parts"] == [{"t": "text", "v": "be nice"}]
    assert lo.messages[0]["parts"] == [{"t": "text", "v": "hello world"}]
    assert lo.messages[1]["parts"] == []
    assert lo.messages[2]["parts"] == [{"t": "text", "v": "ans wer"}]
    assert lo.messages[3]["parts"][0]["parts"] == [{"t": "text", "v": "r x"}]
    assert exact.messages[2]["parts"][0] == {"t": "reasoning", "v": "hmm"}  # input untouched
    assert loose(lo) == lo


def test_media_hashed_by_bytes_not_url_when_data_url():
    def media(url: str) -> dict[str, Any]:
        return _canon(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
                ]
            }
        ).messages[0]["parts"][0]

    raw = b"\x89PNG-bytes"
    a = media("data:image/png;base64," + base64.b64encode(raw).decode())
    b = media("data:image/png;base64," + base64.b64encode(raw).decode() + "\n")
    assert a["sha256"] == hashlib.sha256(raw).hexdigest() == b["sha256"]
    plain = media("https://x.test/a.png")
    assert plain["sha256"] == hashlib.sha256(b"https://x.test/a.png").hexdigest()
    assert plain["mime"] == "image/png"


def test_unreadable_body_returns_none():
    for bad in (None, 5, "x", {}, {"messages": "no"}, {"messages": None}, []):
        assert canonicalize(bad) is None  # type: ignore[arg-type]
    weird = canonicalize({"messages": [None, 3, {"role": 5, "content": object()}]})
    assert weird is not None  # never raises


def _asst(args: str, **extra: Any) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [{"id": "i", "function": {"name": "f", "arguments": args}}],
                **extra,
            }
        ]
    }


def test_key_order_is_kept_in_exact_and_ignored_in_loose():
    a = _canon(_asst('{"b":2,"a":1}'))
    b = _canon(_asst('{"a":1,"b":2}'))
    assert a != b
    assert loose(a) == loose(b)
    t1 = _canon({"messages": [], "tools": [{"y": 1, "x": 2}]})
    t2 = _canon({"messages": [], "tools": [{"x": 2, "y": 1}]})
    assert t1.tools != t2.tools
    assert loose(t1).tools == loose(t2).tools
    assert loose(loose(a)) == loose(a)


def test_different_inputs_hash_differently():
    base = _asst('{"a":1}', reasoning_content="r")
    assert _canon(base) != _canon(_asst('{"a":2}', reasoning_content="r"))
    assert _canon(base) != _canon(_asst('{"a":1}', reasoning_content="q"))
    other_role = {"messages": [{**base["messages"][0], "role": "user"}]}
    assert _canon(base) != _canon(other_role)

    def img(raw: bytes) -> Canon:
        url = "data:image/png;base64," + base64.b64encode(raw).decode()
        part = {"type": "image_url", "image_url": {"url": url}}
        return _canon({"messages": [{"role": "user", "content": [part]}]})

    assert img(b"one") != img(b"two")


def test_results_answered_out_of_order_stay_distinct():
    calls = [
        {"id": "p", "function": {"name": "a", "arguments": "{}"}},
        {"id": "q", "function": {"name": "b", "arguments": "{}"}},
    ]

    def conv(order: list[str]) -> Canon:
        msgs: list[dict[str, Any]] = [{"role": "assistant", "content": None, "tool_calls": calls}]
        msgs += [{"role": "tool", "tool_call_id": i, "content": "r"} for i in order]
        return _canon({"messages": msgs})

    assert conv(["p", "q"]) != conv(["q", "p"])
    assert [m["parts"][0]["ref"] for m in conv(["q", "p"]).messages[1:]] == [1, 0]


def test_string_equals_single_part_and_parts_are_not_merged():
    s = _canon({"messages": [{"role": "user", "content": "ab"}]})
    one = _canon({"messages": [{"role": "user", "content": [{"type": "text", "text": "ab"}]}]})
    two = _canon(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                }
            ]
        }
    )
    empty = _canon({"messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}]})
    assert s == one
    assert s != two
    assert empty.messages[0]["parts"] == []


def test_preamble_part_does_not_swallow_the_next_part():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Change the greeting to be more casual"},
                    {"type": "text", "text": "real question"},
                ],
            }
        ]
    }
    assert loose(_canon(body)).messages[0]["parts"] == [{"t": "text", "v": "real question"}]


def test_unclosed_tags_are_linear():
    text = "<system-reminder>x" * 16000 + " tail"
    body = {"messages": [{"role": "user", "content": text}]}
    t0 = time.perf_counter()
    lo = loose(_canon(body))
    assert time.perf_counter() - t0 < 0.5
    assert lo.messages[0]["parts"][0]["v"].endswith("tail")


def test_bad_base64_data_url_falls_back_to_url_hash():
    url = "data:image/png;base64,@@not-base64@@"
    part = _canon(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
            ]
        }
    )
    assert part.messages[0]["parts"][0]["sha256"] == hashlib.sha256(url.encode()).hexdigest()


def test_dict_content_is_an_unknown_part_not_dropped():
    c = _canon({"messages": [{"role": "user", "content": {"type": "weird", "k": 1}}]})
    assert c.messages[0]["parts"][0]["mime"] == "application/x-unknown"


def test_three_apis_two_tool_calls():
    chat = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"n":1}'},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "g", "arguments": '{"n":2}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "tool", "tool_call_id": "b", "content": "rb"},
        ],
    }
    anth = MessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 5,
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "f", "input": {"n": 1}},
                        {"type": "tool_use", "id": "t2", "name": "g", "input": {"n": 2}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ra"},
                        {"type": "tool_result", "tool_use_id": "t2", "content": "rb"},
                    ],
                },
            ],
        }
    )
    resp = ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": "go"},
                {"type": "function_call", "call_id": "x", "name": "f", "arguments": '{"n":1}'},
                {"type": "function_call", "call_id": "y", "name": "g", "arguments": '{"n":2}'},
                {"type": "function_call_output", "call_id": "x", "output": "ra"},
                {"type": "function_call_output", "call_id": "y", "output": "rb"},
            ],
        }
    )
    a = _canon(chat)
    assert a == _canon(messages_to_oai(anth)) == _canon(responses_to_oai(resp).oai)
    assert [p["ref"] for p in a.messages[1]["parts"]] == [0, 1]


# -- sessions redesign: boilerplate size ------------------------------------

from app.cobs2.canonical import boilerplate_chars  # noqa: E402


def test_boilerplate_chars_counts_blocks_and_date_lines():
    v = "a<system-reminder>xx</system-reminder>b\n2026-10-27\nc<ENVIRONMENT_CONTEXT>y</environment_context>"
    assert boilerplate_chars(v) == (
        len("<system-reminder>xx</system-reminder>")
        + len("2026-10-27")
        + len("<ENVIRONMENT_CONTEXT>y</environment_context>")
    )


def test_boilerplate_chars_zero_without_tags_or_dates():
    assert boilerplate_chars("plain text") == 0


def test_boilerplate_chars_unclosed_tag_is_linear():
    t0 = time.perf_counter()
    assert boilerplate_chars("<system-reminder>x" * 16000) == 0
    assert time.perf_counter() - t0 < 0.5
