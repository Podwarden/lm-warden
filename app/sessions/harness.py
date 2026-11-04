"""Which harness a session came from, read from ``session_source`` (the header or
field that carried its session id). Lossy by design (R8): Hermes and Aider share
OpenCode's ``x_session_id``, pi shares Codex's ``session-id`` header."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

HARNESSES = ("claude_code", "codex", "opencode", "openai_sdk", "other")

HARNESS_OF: dict[str, str] = {
    "claude_code_header": "claude_code",
    "anthropic_metadata": "claude_code",
    "x_session_id": "opencode",
    "x_session_affinity": "opencode",
    "session_id_header": "codex",
    "prompt_cache_key": "codex",
    "openai_user": "openai_sdk",
}


def harness_of(source: str | None) -> str:
    return HARNESS_OF.get(source or "", "other")


def session_harness(sources: Iterable[str | None]) -> str:
    """The most common harness of a session's rows (ties in ``HARNESSES`` order)."""
    counts = Counter(harness_of(s) for s in sources)
    if not counts:
        return "other"
    return max(HARNESSES, key=lambda h: (counts.get(h, 0), -HARNESSES.index(h)))
