"""Request models for the Anthropic Messages API (``POST /v1/messages``, #281).

Only the envelope is typed. Content blocks stay ``dict`` on purpose: the
Messages API grows block types faster than any gateway can follow
(``thinking``, ``redacted_thinking``, ``document``, ``search_result``,
server-tool results, ...), and the translator in ``messages_translate.py``
decides per block type what the OpenAI side can carry and answers a 400 for
what it cannot. ``extra="allow"`` everywhere for the same reason: Claude Code
sends fields a local engine has no use for (``metadata``, ``cache_control``,
``context_management``, ...), and refusing them would break a drop-in
base-URL swap for nothing.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class InputMessage(_Lenient):
    # "system" is not in the public Messages API schema, but Claude Code sends
    # it mid-conversation for model names it does not recognise (beta
    # ``mid-conversation-system-2026-04-07``); the translator folds it into
    # the neighbouring user turn.
    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]


class ToolDefinition(_Lenient):
    name: str
    description: str | None = None
    # Absent on Anthropic server tools (``web_search_20250305``, ``bash_...``),
    # which carry a versioned ``type`` instead; the translator skips those.
    input_schema: dict[str, Any] | None = None
    type: str | None = None


class ToolChoice(_Lenient):
    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


class ThinkingConfig(_Lenient):
    type: str
    budget_tokens: int | None = None


class MessagesRequest(_Lenient):
    model: str = Field(min_length=1)
    messages: list[InputMessage]
    # Required by the Messages API (unlike OpenAI's max_tokens): a request
    # without it is a 400 there, and is one here too.
    max_tokens: int = Field(ge=1)
    system: str | list[dict[str, Any]] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None


class CountTokensRequest(_Lenient):
    """``POST /v1/messages/count_tokens``: the same body minus ``max_tokens``."""

    model: str = Field(min_length=1)
    messages: list[InputMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None
