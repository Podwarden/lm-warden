"""Request model for the OpenAI Responses API (``POST /v1/responses``).

Only the envelope is typed, as in ``messages_schemas.py``: input items, tools
and content parts stay plain dicts because the Responses API grows item types
faster than a gateway can follow (``reasoning``, ``compaction``,
``web_search_call``, ``mcp_*``, ...). ``responses_translate.py`` decides per
item what a chat engine can carry and answers a 400 naming the type for what
it cannot. ``extra="allow"``: Codex sends ``store``, ``include``,
``prompt_cache_key``, ``client_metadata``, ``service_tier`` and the like, and
refusing them would break a drop-in base-URL swap for nothing.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class ResponsesRequest(_Lenient):
    model: str = Field(min_length=1)
    # A bare string is one user turn; otherwise a list of items.
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    # Refused by the translator (the warden stores no responses):
    previous_response_id: str | None = None
    conversation: str | dict[str, Any] | None = None
    background: bool | None = None
