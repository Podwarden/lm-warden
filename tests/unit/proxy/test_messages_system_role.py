"""Mid-conversation ``role: "system"`` messages (Claude Code, beta
``mid-conversation-system-2026-04-07``).

For a model name it does not recognise, Claude Code puts a system-role entry
inside ``messages``. The Messages API schema says user/assistant only, so the
warden answered 400 ``messages.1.role: Input should be 'user' or 'assistant'``
and Claude Code does not retry. The translator folds those entries into the
neighbouring user turn, because chat templates (Qwen) reject a system message
anywhere but first.
"""

from typing import Any

import pytest

from app.proxy.messages_schemas import CountTokensRequest, MessagesRequest
from app.proxy.messages_translate import count_text, to_openai_request
from tests.unit.proxy.test_messages import _json_upstream, _post, _ready

NOTE = "Plan mode is active."


def _req(messages: list[dict[str, Any]], system: Any = "You are Claude Code.") -> MessagesRequest:
    return MessagesRequest.model_validate(
        {"model": "qwen", "max_tokens": 64, "system": system, "messages": messages}
    )


def _roles(oai: dict[str, Any]) -> list[str]:
    return [m["role"] for m in oai["messages"]]


def _claude_code_body() -> dict[str, Any]:
    """The shape captured live from Claude Code 2.1.289 (content trimmed)."""
    return {
        "model": "qwen",
        "max_tokens": 32000,
        "stream": False,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.289"},
            {
                "type": "text",
                "text": "You are Claude Code.",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "metadata": {"user_id": "{}"},
        "thinking": {"type": "enabled", "budget_tokens": 31999, "display": "omitted"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "tools": [
            {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "read a.txt"}]},
            {"role": "system", "content": [{"type": "text", "text": NOTE}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body"},
                    {"type": "text", "text": "now summarise"},
                ],
            },
        ],
    }


def test_exact_400_reproduced_then_fixed(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    ok = {
        "id": "c1",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    r, send = _post(client, _claude_code_body(), upstream=_json_upstream(ok))
    assert r.status_code == 200, r.text
    import json

    sent = json.loads(send.call_args.args[0].content)
    # Only one system message, and it is first.
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert sent["messages"][3]["tool_call_id"] == "toolu_1"


def test_leading_system_merges_into_top_level_system():
    oai = to_openai_request(
        _req(
            [
                {"role": "system", "content": "lead one"},
                {"role": "system", "content": [{"type": "text", "text": "lead two"}]},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    assert _roles(oai) == ["system", "user"]
    assert oai["messages"][0]["content"] == "You are Claude Code.\n\nlead one\n\nlead two"
    assert oai["messages"][1]["content"] == "hi"


def test_leading_system_without_top_level_system():
    oai = to_openai_request(
        _req(
            [{"role": "system", "content": "lead"}, {"role": "user", "content": "hi"}], system=None
        )
    )
    assert oai["messages"][0] == {"role": "system", "content": "lead"}


def test_mid_conversation_prefixes_the_next_user_turn():
    oai = to_openai_request(
        _req(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "system", "content": NOTE},
                {"role": "user", "content": "three"},
            ]
        )
    )
    assert _roles(oai) == ["system", "user", "assistant", "user"]
    last = oai["messages"][-1]["content"]
    assert NOTE in last and last.index(NOTE) < last.index("three")
    assert "system" not in _roles(oai)[1:]


def test_after_the_last_user_turn_is_appended_to_it():
    oai = to_openai_request(
        _req([{"role": "user", "content": "one"}, {"role": "system", "content": NOTE}])
    )
    assert _roles(oai) == ["system", "user"]
    content = oai["messages"][1]["content"]
    assert content.index("one") < content.index(NOTE)


def test_after_an_assistant_turn_becomes_a_trailing_user_message():
    oai = to_openai_request(
        _req(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "system", "content": NOTE},
            ]
        )
    )
    assert _roles(oai) == ["system", "user", "assistant", "user"]
    assert NOTE in oai["messages"][-1]["content"]
    assert oai["messages"][1]["content"] == "one"


def test_adjacent_to_tool_result_keeps_tool_pairing_valid():
    oai = to_openai_request(
        _req(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                        {"type": "tool_use", "id": "t2", "name": "Read", "input": {}},
                    ],
                },
                {"role": "system", "content": NOTE},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                        {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
                    ],
                },
            ]
        )
    )
    # tool messages directly follow the assistant tool_calls; the note rides after them
    assert _roles(oai) == ["system", "user", "assistant", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in oai["messages"][3:5]] == ["t1", "t2"]
    assert NOTE in oai["messages"][-1]["content"]


def test_trailing_system_after_tool_results_does_not_touch_tool_messages():
    oai = to_openai_request(
        _req(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a"}],
                },
                {"role": "system", "content": NOTE},
            ]
        )
    )
    assert _roles(oai) == ["system", "user", "assistant", "tool", "user"]
    assert oai["messages"][3]["content"] == "a"
    assert oai["messages"][-1]["content"].count(NOTE) == 1


def test_multiple_system_messages_keep_their_order():
    oai = to_openai_request(
        _req(
            [
                {"role": "user", "content": "one"},
                {"role": "system", "content": "first note"},
                {"role": "system", "content": "second note"},
                {"role": "user", "content": "two"},
                {"role": "system", "content": "third note"},
            ]
        )
    )
    assert _roles(oai) == ["system", "user", "user"]
    mid = oai["messages"][2]["content"]
    assert mid.index("first note") < mid.index("second note") < mid.index("two")
    assert "third note" in oai["messages"][2]["content"]
    assert mid.index("two") < mid.index("third note")


def test_user_block_list_gets_a_text_block_prefix():
    oai = to_openai_request(
        _req(
            [
                {"role": "system", "content": "x"},
                {"role": "user", "content": "a"},
                {"role": "system", "content": NOTE},
                {"role": "user", "content": [{"type": "text", "text": "b"}]},
            ]
        )
    )
    assert _roles(oai) == ["system", "user", "user"]
    assert NOTE in oai["messages"][2]["content"]


@pytest.mark.parametrize("content", ["", [], [{"type": "text", "text": ""}], "   \n"])
def test_empty_system_content_is_dropped(content):
    base = to_openai_request(_req([{"role": "user", "content": "a"}]))
    oai = to_openai_request(
        _req(
            [
                {"role": "system", "content": content},
                {"role": "user", "content": "a"},
                {"role": "system", "content": content},
            ]
        )
    )
    assert oai["messages"] == base["messages"]


def test_non_text_system_block_is_a_translation_error():
    from app.proxy.messages_translate import TranslationError

    with pytest.raises(TranslationError):
        to_openai_request(
            _req([{"role": "system", "content": [{"type": "image", "source": {}}]}]),
        )


def test_user_assistant_only_requests_are_unchanged():
    oai = to_openai_request(
        _req([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    )
    assert oai["messages"] == [
        {"role": "system", "content": "You are Claude Code."},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_count_text_counts_the_system_note():
    msgs = [
        {"role": "user", "content": "one"},
        {"role": "system", "content": NOTE},
        {"role": "user", "content": "two"},
    ]
    n = CountTokensRequest.model_validate({"model": "qwen", "messages": msgs})
    m = MessagesRequest.model_validate({"model": "qwen", "max_tokens": 5, "messages": msgs})
    assert NOTE in count_text(to_openai_request(n))
    assert count_text(to_openai_request(n)) == count_text(to_openai_request(m))


def test_count_tokens_route_counts_the_same_text(tmp_data_dir, client):
    _ready(client, tmp_data_dir)  # fake tokenizer: one token per character
    body = {
        "model": "qwen",
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "system", "content": NOTE},
            {"role": "user", "content": "two"},
        ],
    }
    r, send = _post(client, body, path="/v1/messages/count_tokens")
    assert r.status_code == 200, r.text
    expected = count_text(to_openai_request(CountTokensRequest.model_validate(body)))
    assert r.json() == {"input_tokens": len(expected)}
    assert len(expected) > len("one\ntwo")
