"""Responses API -> chat request translation (pure, dicts only)."""

import json

import pytest

from app.proxy.responses_schemas import ResponsesRequest
from app.proxy.responses_translate import (
    Translation,
    TranslationError,
    thinking_requested,
    to_openai_request,
)


def tr(**body) -> Translation:
    body.setdefault("model", "qwen")
    return to_openai_request(ResponsesRequest.model_validate(body))


def msgs(**body):
    return tr(**body).oai["messages"]


def user(text, role="user"):
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def call(cid="c1", name="shell", args='{"cmd":"ls"}'):
    return {"type": "function_call", "call_id": cid, "name": name, "arguments": args}


def out(cid="c1", output="done"):
    return {"type": "function_call_output", "call_id": cid, "output": output}


# ---- basics -----------------------------------------------------------------------------------


def test_string_input_is_one_user_turn():
    assert msgs(input="hi") == [{"role": "user", "content": "hi"}]


def test_instructions_and_leading_developer_form_the_system_prompt():
    got = msgs(
        instructions="BASE",
        input=[user("skills", "developer"), user("ctx", "system"), user("question")],
    )
    assert got[0] == {"role": "system", "content": "BASE\n\nskills\n\nctx"}
    assert got[1:] == [{"role": "user", "content": "question"}]


def test_later_developer_item_is_folded_into_the_next_user_turn():
    got = msgs(
        input=[
            user("one"),
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "a"}],
            },
            user("rules", "developer"),
            user("two"),
        ]
    )
    assert [m["role"] for m in got] == ["user", "assistant", "user"]
    assert got[2]["content"] == "[system message]\nrules\n[end system message]\n\ntwo"


def test_trailing_developer_item_is_appended():
    got = msgs(input=[user("one"), user("late", "developer")])
    assert got == [
        {"role": "user", "content": "one\n\n[system message]\nlate\n[end system message]"}
    ]


def test_role_object_without_type_is_a_message():
    assert msgs(input=[{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]


def test_user_images_become_image_url_parts():
    got = msgs(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
                ],
            }
        ]
    )
    assert got[0]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]


def test_refusal_part_is_text():
    got = msgs(
        input=[
            user("q"),
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "no"}],
            },
        ]
    )
    assert got[1] == {"role": "assistant", "content": "no"}


# ---- tool calls ---------------------------------------------------------------------------------


def test_assistant_text_merges_with_following_calls_and_outputs_follow():
    got = msgs(
        input=[
            user("go"),
            {"type": "reasoning", "summary": [], "encrypted_content": None},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            },
            call("c1"),
            call("c2", args={"a": 1}),
            out("c1"),
            out("c2", [{"type": "input_text", "text": "x"}, {"type": "input_text", "text": "y"}]),
        ]
    )
    assert [m["role"] for m in got] == ["user", "assistant", "tool", "tool"]
    a = got[1]
    assert a["content"] == "ok"
    assert [c["id"] for c in a["tool_calls"]] == ["c1", "c2"]
    assert a["tool_calls"][1]["function"]["arguments"] == '{"a": 1}'
    assert got[2] == {"role": "tool", "tool_call_id": "c1", "content": "done"}
    assert got[3]["content"] == "x\n\ny"


def test_call_without_text_has_null_content_and_alternating_calls_stay_adjacent():
    got = msgs(input=[user("go"), call("c1"), out("c1"), call("c2"), out("c2")])
    assert [m["role"] for m in got] == ["user", "assistant", "tool", "assistant", "tool"]
    assert got[1]["content"] is None


def test_tool_output_images_move_to_a_following_user_message():
    got = msgs(
        input=[
            user("go"),
            call("c1"),
            out(
                "c1",
                [
                    {"type": "input_text", "text": "see"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
                ],
            ),
        ]
    )
    assert got[2]["content"] == "see\n\n[image attached in the next user message]"
    assert got[3]["role"] == "user"
    assert got[3]["content"][0]["text"] == "[image from tool output c1]"
    assert got[3]["content"][1]["type"] == "image_url"


def test_custom_tool_call_and_output_round_trip():
    got = msgs(
        input=[
            user("patch"),
            {
                "type": "custom_tool_call",
                "call_id": "k1",
                "name": "apply_patch",
                "input": "*** Begin",
            },
            {"type": "custom_tool_call_output", "call_id": "k1", "output": "ok"},
        ]
    )
    assert json.loads(got[1]["tool_calls"][0]["function"]["arguments"]) == {"input": "*** Begin"}
    assert got[2] == {"role": "tool", "tool_call_id": "k1", "content": "ok"}


def test_output_for_an_unknown_call_is_refused():
    with pytest.raises(TranslationError, match="unknown call_id"):
        msgs(input=[user("x"), out("nope")])


def test_reasoning_items_are_dropped():
    t = tr(
        input=[user("x"), {"type": "reasoning", "summary": [{"type": "summary_text", "text": "t"}]}]
    )
    assert len(t.oai["messages"]) == 1
    assert t.dropped == ["reasoning"]


@pytest.mark.parametrize(
    "item",
    [
        {"type": "item_reference", "id": "x"},
        {"type": "local_shell_call", "call_id": "x"},
        {"type": "compaction", "encrypted_content": "x"},
        {"type": "mcp_call", "id": "x"},
        {"id": "no-type"},
    ],
)
def test_unsupported_items_name_the_type_and_index(item):
    with pytest.raises(TranslationError, match=r"input\[1\]"):
        msgs(input=[user("x"), item])


def test_unsupported_content_parts():
    for kind in ("input_file", "input_audio"):
        with pytest.raises(TranslationError, match=kind):
            msgs(input=[{"type": "message", "role": "user", "content": [{"type": kind}]}])


def test_error_messages_never_echo_content():
    secret = "TOP-SECRET-PROMPT"
    with pytest.raises(TranslationError) as ei:
        msgs(input=[{"type": "mystery", "text": secret}])
    assert secret not in str(ei.value)


def test_empty_conversation_is_refused():
    with pytest.raises(TranslationError):
        msgs(input=[user("rules", "developer")])


# ---- refused request fields -----------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, word",
    [
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"conversation": "conv_1"}, "conversation"),
        ({"background": True}, "background"),
    ],
)
def test_stateful_fields_are_refused_by_name(extra, word):
    with pytest.raises(TranslationError, match=word):
        tr(input="hi", **extra)


def test_background_false_is_fine():
    tr(input="hi", background=False)


# ---- tools --------------------------------------------------------------------------------------


def test_function_and_custom_tools():
    t = tr(
        input="hi",
        tools=[
            {
                "type": "function",
                "name": "shell",
                "description": "run",
                "parameters": {"type": "object"},
                "strict": False,
            },
            {"type": "function", "name": "noparams"},
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "patch",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"},
            },
            {"type": "web_search"},
            {"type": "web_search_preview"},
        ],
    )
    tools = t.oai["tools"]
    assert [x["function"]["name"] for x in tools] == ["shell", "noparams", "apply_patch"]
    assert tools[0]["function"] == {
        "name": "shell",
        "description": "run",
        "parameters": {"type": "object"},
    }
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    custom = tools[2]["function"]
    assert custom["parameters"]["required"] == ["input"]
    assert "Freeform tool" in custom["description"] and "start: x" in custom["description"]
    assert t.custom_names == {"apply_patch"}
    assert t.dropped == ["web_search", "web_search_preview"]


@pytest.mark.parametrize(
    "tool",
    [
        "local_shell",
        "namespace",
        "tool_search",
        "mcp",
        "file_search",
        "code_interpreter",
        "image_generation",
        "computer_use_preview",
    ],
)
def test_unsupported_tool_types_are_refused(tool):
    with pytest.raises(TranslationError, match=rf"tools\[0\].*{tool}"):
        tr(input="hi", tools=[{"type": tool, "name": "x"}])


def test_duplicate_tool_names_are_refused():
    with pytest.raises(TranslationError, match="duplicate"):
        tr(
            input="hi",
            tools=[{"type": "function", "name": "a"}, {"type": "custom", "name": "a"}],
        )


def test_tool_choice_mapping():
    base = {
        "input": "hi",
        "tools": [{"type": "function", "name": "f"}, {"type": "custom", "name": "c"}],
    }
    assert tr(**base, tool_choice="required").oai["tool_choice"] == "required"
    assert tr(**base, tool_choice="none").oai["tool_choice"] == "none"
    want = {"type": "function", "function": {"name": "f"}}
    assert tr(**base, tool_choice={"type": "function", "name": "f"}).oai["tool_choice"] == want
    want_c = {"type": "function", "function": {"name": "c"}}
    assert tr(**base, tool_choice={"type": "custom", "name": "c"}).oai["tool_choice"] == want_c


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "allowed_tools", "tools": []},
        {"type": "mcp", "server_label": "s"},
        {"type": "web_search_preview"},
        {"type": "custom", "name": "missing"},
        "weird",
    ],
)
def test_unsupported_tool_choice_is_refused(choice):
    with pytest.raises(TranslationError, match="tool_choice"):
        tr(input="hi", tools=[{"type": "function", "name": "f"}], tool_choice=choice)


def test_parallel_tool_calls_only_with_tools():
    assert "parallel_tool_calls" not in tr(input="hi", parallel_tool_calls=True).oai
    t = tr(input="hi", tools=[{"type": "function", "name": "f"}], parallel_tool_calls=False)
    assert t.oai["parallel_tool_calls"] is False
    assert t.parallel_tool_calls is False


# ---- sampling, format, thinking, allow-list ---------------------------------------------------


def test_max_output_tokens_temperature_top_p():
    o = tr(input="hi", max_output_tokens=64, temperature=0.2, top_p=0.9).oai
    assert (o["max_tokens"], o["temperature"], o["top_p"]) == (64, 0.2, 0.9)
    assert "max_tokens" not in tr(input="hi").oai


def test_text_format():
    schema = {"type": "object"}
    o = tr(
        input="hi",
        text={"format": {"type": "json_schema", "name": "n", "schema": schema, "strict": True}},
    ).oai
    assert o["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "n", "schema": schema, "strict": True},
    }
    assert tr(input="hi", text={"format": {"type": "json_object"}}).oai["response_format"] == {
        "type": "json_object"
    }
    assert (
        "response_format"
        not in tr(input="hi", text={"format": {"type": "text"}, "verbosity": "low"}).oai
    )
    with pytest.raises(TranslationError, match="text.format"):
        tr(input="hi", text={"format": {"type": "weird"}})


def test_stream_options_only_when_streaming():
    assert "stream" not in tr(input="hi").oai
    o = tr(input="hi", stream=True).oai
    assert o["stream"] is True
    assert o["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}


@pytest.mark.parametrize(
    "reasoning, on",
    [
        (None, False),
        ({}, True),
        ({"effort": "high"}, True),
        ({"effort": "none"}, False),
        ({"summary": "auto"}, True),
    ],
)
def test_thinking_rule(reasoning, on):
    body = {"input": "hi"}
    if reasoning is not None:
        body["reasoning"] = reasoning
    t = tr(**body)
    assert t.emit_thinking is on
    assert thinking_requested(ResponsesRequest.model_validate({"model": "q", **body})) is on
    if on:
        assert "chat_template_kwargs" not in t.oai
    else:
        assert t.oai["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_body_is_an_allow_list():
    t = tr(
        input="hi",
        store=False,
        include=["reasoning.encrypted_content"],
        prompt_cache_key="abc",
        user="u",
        metadata={"a": "b"},
        client_metadata={"x": "y"},
        safety_identifier="s",
        service_tier="auto",
        truncation="auto",
        stream_options={"include_obfuscation": False},
        text={"verbosity": "low"},
    )
    assert set(t.oai) <= {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "max_tokens",
        "temperature",
        "top_p",
        "stream",
        "stream_options",
        "response_format",
        "chat_template_kwargs",
    }


# ---- review follow-ups ---------------------------------------------------------------------------


def test_namespace_member_colliding_with_another_tool_is_skipped_not_fatal():
    t = tr(
        input="hi",
        tools=[
            {"type": "function", "name": "a"},
            {
                "type": "namespace",
                "name": "n",
                "tools": [{"type": "function", "name": "a"}, {"type": "function", "name": "b"}],
            },
        ],
    )
    assert [x["function"]["name"] for x in t.oai["tools"]] == ["a", "b"]
    assert t.dropped == ["duplicate:n"]
    assert t.namespaces == {"b": "n"}  # the skipped one must not claim the top-level name


def test_top_level_duplicate_tool_name_is_still_a_400():
    with pytest.raises(TranslationError, match="duplicate"):
        tr(input="hi", tools=[{"type": "function", "name": "a"}, {"type": "function", "name": "a"}])


def test_web_search_call_items_are_dropped_like_reasoning():
    t = tr(input=[user("x"), {"type": "web_search_call", "id": "ws_1", "status": "completed"}])
    assert len(t.oai["messages"]) == 1
    assert t.dropped == ["web_search_call"]


def test_unknown_item_error_says_what_to_do():
    with pytest.raises(TranslationError, match=r"'compaction'.*new Codex thread"):
        msgs(input=[user("x"), {"type": "compaction", "encrypted_content": "x"}])
