from dataclasses import dataclass
from typing import Any

import pytest

from app.chat2.locks import TurnLocks
from app.chat2.titles import (
    LOWEST_PRIORITY,
    first_line_title,
    generate_llm_title,
    normalized_title,
    title_context,
)


@dataclass(frozen=True)
class _Row:
    """The bit of `MessageRow` `title_context` actually reads."""

    role: str
    parts: list[dict[str, Any]]


def _text(role: str, text: str) -> _Row:
    return _Row(role=role, parts=[{"type": "text", "text": text}])


def test_turn_locks_are_per_chat_and_non_reentrant() -> None:
    locks = TurnLocks()
    assert locks.try_acquire("a") and not locks.try_acquire("a") and locks.try_acquire("b")
    locks.release("a")
    assert not locks.held("a") and locks.try_acquire("a")


def test_first_line_title() -> None:
    assert first_line_title("  Hello world\nmore") == "Hello world"
    assert first_line_title("x" * 100, limit=10) == "x" * 9 + "…"
    assert first_line_title("   \n  ") == "New chat"


def test_normalized_title_strips_collapses_and_casefolds() -> None:
    assert normalized_title("  Deploy   Checklist \n") == "deploy checklist"
    assert normalized_title("Deploy checklist") == normalized_title("DEPLOY  CHECKLIST")


def test_title_context_keeps_the_last_six_turns_in_order() -> None:
    rows = [_text("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(10)]
    ctx = title_context(rows)
    lines = ctx.splitlines()
    assert len(lines) == 6
    # oldest -> newest, so the model reads the conversation the right way round
    assert lines[0] == "User: m4" and lines[-1] == "Assistant: m9"


def test_title_context_skips_non_text_and_non_chat_rows() -> None:
    rows = [
        _Row(role="tool", parts=[{"type": "text", "text": "tool noise"}]),
        _Row(role="user", parts=[{"type": "image", "attachmentId": "a1"}]),
        _text("user", "  real question  "),
    ]
    assert title_context(rows) == "User: real question"


def test_title_context_truncates_newest_first() -> None:
    rows = [_text("user", "old" * 100), _text("assistant", "new" * 100)]
    ctx = title_context(rows, limit_chars=60)
    assert len(ctx) <= 60
    # the budget is spent on the most recent turn; the older one is dropped
    assert ctx.startswith("Assistant: newnew") and "old" not in ctx


@pytest.mark.asyncio
async def test_generate_llm_title_uses_first_choice_and_trims() -> None:
    async def fake(payload):
        assert payload["max_tokens"] == 16 and payload["stream"] is False
        # spec §4.2 step 9: lowest proxy priority. vLLM orders its waiting
        # queue by priority ASCENDING and the proxy maps warden 0..9 onto
        # -9..0, so the title request must send a POSITIVE priority to sit
        # behind every user turn — a plain 0 would only tie with default
        # traffic. The proxy never overrides a client-set priority.
        assert payload["priority"] == LOWEST_PRIORITY == 9
        assert payload["messages"][-1]["content"] == "User: ship it"
        return {"choices": [{"message": {"content": '  "Deploy checklist"\n'}}]}

    got = await generate_llm_title(post_json=fake, model="m", context_text="User: ship it")
    assert got == "Deploy checklist"

    async def none(payload):
        return None

    assert await generate_llm_title(post_json=none, model="m", context_text="x") is None


@pytest.mark.asyncio
async def test_generate_llm_title_disables_thinking() -> None:
    """A reasoning model would spend the whole 16-token budget on its preamble
    and return an empty (or None) completion — so the title call turns thinking
    off explicitly, the same way a chat with `enable_thinking: false` does."""
    seen: dict[str, Any] = {}

    async def fake(payload):
        seen.update(payload)
        return {"choices": [{"message": {"content": "Deploy checklist"}}]}

    assert await generate_llm_title(post_json=fake, model="m", context_text="c")
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", "   \n ", '""', 123, {"text": "x"}])
async def test_generate_llm_title_rejects_a_non_string_or_empty_completion(content) -> None:
    """`str(None)` is the string "None" -- which used to sail through the
    truthiness check and rename the chat to the literal "None", displacing the
    perfectly good title into `title_prev`. An unusable completion must mean
    "no rename", not "rename to junk"."""

    async def fake(payload):
        return {"choices": [{"message": {"content": content}}]}

    assert await generate_llm_title(post_json=fake, model="m", context_text="c") is None
