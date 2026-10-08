from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.chat2.limits import RESERVE_CAP_TOKENS


@dataclass(frozen=True)
class ContextState:
    prompt_tokens: int
    window: int | None
    full: bool

    def as_event(self) -> dict[str, Any]:
        return {
            "type": "context",
            "promptTokens": self.prompt_tokens,
            "window": self.window,
            "full": self.full,
        }


def derive_context(
    messages: list[Any], settings: dict[str, Any], window: int | None
) -> ContextState:
    used = 0
    for m in reversed(messages):
        if m.role == "assistant" and m.usage:
            used = int(m.usage.get("prompt", 0)) + int(m.usage.get("completion", 0))
            break
    if window is None:
        return ContextState(used, None, False)
    reserve = min(int(settings.get("max_tokens", RESERVE_CAP_TOKENS)), RESERVE_CAP_TOKENS)
    return ContextState(used, window, used + reserve >= window)
