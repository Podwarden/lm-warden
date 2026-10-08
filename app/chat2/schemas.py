from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# `scope` and `tool_policy` are stored opaque -- nothing downstream inspects or
# bounds them -- so without a cap a client can park arbitrarily large JSON in
# the settings blob of every chat it creates. A real binding is tens of bytes
# ({"instance_id": ...}); 4 KiB is orders of magnitude above that and still far
# below a storage problem.
_OPAQUE_MAX_BYTES = 4096


class SettingsPatch(BaseModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    top_p: float | None = Field(default=None, ge=0, le=1)
    system_prompt: str | None = Field(default=None, max_length=20000)
    enabled_tools: list[str] | None = None
    enabled_skills: list[str] | None = None
    enable_thinking: bool | None = None
    # A `chat_template_kwargs.reasoning_effort` value in the MODEL's vocabulary
    # (Qwen3.8: `low` / `medium` / `xhigh`), `""` to return to the engine
    # default. Deliberately a bounded string, not a Literal: the accepted set
    # belongs to the loaded model's chat template, which the catalog reads and
    # publishes as `ModelInfo.reasoning_efforts`, and the turn route only ever
    # forwards a value that set contains -- so a stored value can never reach a
    # template that would `raise_exception` on it (#241).
    reasoning_effort: str | None = Field(default=None, max_length=32, pattern=r"^[a-z0-9_-]*$")
    tool_policy: dict[str, Any] | None = (
        None  # Hub-only {max_iterations, tool_choice}; stored opaque (spec §2.4)
    )
    scope: dict[str, Any] | None = (
        None  # host-defined binding (Hub: {instance_id}); stored opaque (spec §5.2)
    )

    @field_validator("scope", "tool_policy")
    @classmethod
    def _bound_opaque(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        # A validator error surfaces as a 422 with FastAPI's list-shaped detail
        # rather than the chat2 {code, message} envelope -- the documented
        # boundary deviation, same as any other malformed-body rejection.
        if v is not None and len(json.dumps(v).encode()) > _OPAQUE_MAX_BYTES:
            raise ValueError("scope/tool_policy too large (max 4 KiB)")
        return v

    def merge_into(self, base: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in self.model_dump(exclude_none=True).items():
            out[k] = v
        return out


class ChatCreate(BaseModel):
    model: str | None = None
    settings: SettingsPatch | None = None


class ChatPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    model: str | None = None
    settings: SettingsPatch | None = None


class ForkBody(BaseModel):
    at_seq: int = Field(ge=1)
    edited_text: str | None = Field(default=None, max_length=100000)


class DefaultsPut(BaseModel):
    model: str | None = None
    settings: SettingsPatch


class ToolResultIn(BaseModel):
    call_id: str
    result: Any


class TextPartIn(BaseModel):
    """The only inbound user part shape (images travel as `attachment_ids`).

    Typed rather than `dict[str, Any]` so a malformed part -- `{"type": "text",
    "text": 123}` -- is a 422 at the FastAPI boundary instead of a TypeError
    deep inside the turn, after the upstream socket and the active-request
    counter have already been taken (T14 review, finding 2).
    """

    type: Literal["text"] = "text"
    text: str = Field(max_length=100000)


class TurnBody(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    user_parts: list[TextPartIn] | None = None
    attachment_ids: list[str] = Field(default_factory=list)
    tool_results: list[ToolResultIn] | None = None
    regenerate: bool = False
