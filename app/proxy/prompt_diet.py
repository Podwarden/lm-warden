"""Prompt diet: a per-key, opt-in cut of dead weight from chat bodies on their way
to the engine (owner request 2026-10-28; experiment before any default).

- ``strip_tools``: tool definitions to drop, by function name. A tool named by
  ``tool_choice`` is never dropped; dropping every tool drops ``tools`` and a
  non-specific ``tool_choice`` too.
- ``strip_reasoning``: ``True`` drops ``reasoning_content`` / ``reasoning`` from
  every assistant message. The 2026-10-28 experiment showed this costs ~3x
  generation in agent loops: the model re-derives what it no longer sees.
  ``"before_last_user"`` drops it only before the last real user message (a user
  message that is nothing but ``<tool_response>`` does not count), so a tool
  chain keeps its reasoning -- DeepSeek V3.2's rule, and Qwen3.8's template rule
  under ``preserve_thinking=false``.

The cut depends only on the body and the key's diet, so it is the same on every
turn of a conversation and the engine's prefix cache keeps working. Lineage and
the recorded shape are computed on the client's body, before the cut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

ReasoningMode = bool | Literal["before_last_user"]

_REASONING_KEYS = ("reasoning_content", "reasoning")


@dataclass(frozen=True)
class Diet:
    strip_tools: frozenset[str]
    strip_reasoning: ReasoningMode


@dataclass(frozen=True)
class DietResult:
    tools: int
    tools_chars: int
    reason_chars: int

    def json(self) -> str:
        return json.dumps(
            {
                "tools": self.tools,
                "tools_chars": self.tools_chars,
                "reason_chars": self.reason_chars,
            },
            separators=(",", ":"),
        )


def parse_diet(raw: str | None) -> Diet | None:
    """The stored JSON (``{"strip_tools": [...], "strip_reasoning": bool}``), or
    None when unset, invalid, or a diet that cuts nothing."""
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    tools = d.get("strip_tools", [])
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        return None
    mode = d.get("strip_reasoning", False)
    if mode not in (True, False, "before_last_user"):
        return None
    diet = Diet(frozenset(tools), mode)
    return diet if diet.strip_tools or diet.strip_reasoning else None


def diet_json(d: Diet) -> str:
    return json.dumps(
        {"strip_tools": sorted(d.strip_tools), "strip_reasoning": d.strip_reasoning},
        separators=(",", ":"),
    )


def _tool_name(t: Any) -> str | None:
    if isinstance(t, dict):
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return str(fn["name"])
        if isinstance(t.get("name"), str):
            return str(t["name"])
    return None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _last_real_user(messages: list[Any]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            t = _text_of(m.get("content")).strip()
            if not (t.startswith("<tool_response>") and t.endswith("</tool_response>")):
                return i
    return None


def _chosen(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        return _tool_name(tool_choice)
    return None


def apply_diet(body: dict[str, Any], diet: Diet) -> DietResult | None:
    """Cut in place; the result says what was cut, or None when nothing was."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None
    n_tools = tools_chars = reason_chars = 0
    tools = body.get("tools")
    if diet.strip_tools and isinstance(tools, list):
        keep_name = _chosen(body.get("tool_choice"))
        kept: list[Any] = []
        for t in tools:
            name = _tool_name(t)
            if name is not None and name in diet.strip_tools and name != keep_name:
                n_tools += 1
                tools_chars += len(json.dumps(t, separators=(",", ":")))
            else:
                kept.append(t)
        if n_tools:
            if kept:
                body["tools"] = kept
            else:
                body.pop("tools", None)
                if not isinstance(body.get("tool_choice"), dict):
                    body.pop("tool_choice", None)
    if diet.strip_reasoning:
        msgs = body["messages"]
        if diet.strip_reasoning == "before_last_user":
            last = _last_real_user(msgs)
            upto = last if last is not None else 0
        else:
            upto = len(msgs)
        for m in msgs[:upto]:
            if isinstance(m, dict) and m.get("role") == "assistant":
                for k in _REASONING_KEYS:
                    v = m.pop(k, None)
                    if isinstance(v, str):
                        reason_chars += len(v)
    if not n_tools and not reason_chars:
        return None
    return DietResult(n_tools, tools_chars, reason_chars)
