"""OpenAI-compatible SSE stream -> ChatEvent normaliser (spec: docs/superpowers/specs/2026-08-23-chat2-design.md)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.chat2.ledger import Usage
from app.utils.sse import delta_reasoning

_GUARD_PREFIX = "runaway_"


def split_sse(buffer: bytes) -> tuple[list[str], bytes]:
    """Return complete `data:` payloads and the unconsumed tail.

    Some upstreams frame SSE with CRLF (`\\r\\n\\r\\n`) rather than the spec's
    bare `\\n\\n`; normalise by stripping `\\r` before splitting (mirrors
    frontend/src/lib/use-chat-stream.ts's splitSseEvents). This can discard a
    `\\r` embedded inside event data, an accepted trade-off since payloads are
    single-line JSON. Stripping is idempotent, so re-applying it to a leftover
    tail carried across chunk boundaries is safe even when a `\\r\\n\\r\\n`
    boundary itself is split across two chunks.
    """
    buffer = buffer.replace(b"\r", b"")
    events: list[str] = []
    while True:
        idx = buffer.find(b"\n\n")
        if idx < 0:
            return events, buffer
        block, buffer = buffer[:idx], buffer[idx + 2 :]
        for line in block.split(b"\n"):
            if line.startswith(b"data:"):
                events.append(line[5:].strip().decode("utf-8", errors="replace"))


def map_finish_reason(raw: str | None) -> str:
    if raw is None:
        return "stop"
    if raw.startswith(_GUARD_PREFIX):
        return "guard"
    return raw if raw in ("stop", "length", "tool_calls") else "stop"


def frame(event: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(event, separators=(", ", ": ")).encode() + b"\n\n"


@dataclass
class StreamState:
    text: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    usage: Usage | None = None
    finish_reason: str | None = None
    upstream_id: str | None = None

    def finished_tool_call(self, index: int, client_tool_names: set[str]) -> dict[str, Any]:
        tc = self.tool_calls[index]
        try:
            args = json.loads(tc["argsText"] or "{}")
        except ValueError:
            args = {"_raw": tc["argsText"]}
        return {
            "type": "tool-call",
            "id": tc["id"],
            "name": tc["name"],
            "args": args,
            "client": tc["name"] in client_tool_names,
        }

    def parts(self, client_tool_names: set[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.reasoning:
            out.append({"type": "reasoning", "text": self.reasoning})
        for idx in sorted(self.tool_calls):
            ev = self.finished_tool_call(idx, client_tool_names)
            out.append(
                {
                    "type": "tool_call",
                    "id": ev["id"],
                    "name": ev["name"],
                    "args": ev["args"],
                    "argsText": self.tool_calls[idx]["argsText"],
                    "client": ev["client"],
                }
            )
            if ev["name"] == "present_options" and isinstance(ev["args"], dict):
                opts = ev["args"].get("options")
                if isinstance(opts, list) and 2 <= len(opts) <= 8:
                    out.append(
                        {
                            "type": "options",
                            "callId": ev["id"],
                            "question": ev["args"].get("question"),
                            "options": [
                                {"label": str(o.get("label", "")), "value": str(o.get("value", ""))}
                                for o in opts
                                if isinstance(o, dict)
                            ],
                            "multi": bool(ev["args"].get("multi", False)),
                        }
                    )
        if self.text:
            out.append({"type": "text", "text": self.text})
        return out


async def normalize(
    chunks: AsyncIterator[bytes], state: StreamState, client_tool_names: set[str]
) -> AsyncIterator[dict[str, Any]]:
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        payloads, buffer = split_sse(buffer)
        for payload in payloads:
            if payload == "[DONE]":
                for idx in sorted(state.tool_calls):
                    if not state.tool_calls[idx].get("_emitted"):
                        state.tool_calls[idx]["_emitted"] = True
                        yield state.finished_tool_call(idx, client_tool_names)
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("id") and not state.upstream_id:
                state.upstream_id = str(obj["id"])
            if obj.get("usage"):
                u = obj["usage"]
                det = u.get("completion_tokens_details") or {}
                pdet = u.get("prompt_tokens_details") or {}
                state.usage = Usage(
                    prompt=int(u.get("prompt_tokens", 0)),
                    completion=int(u.get("completion_tokens", 0)),
                    reasoning=det.get("reasoning_tokens"),
                    cache_read=pdet.get("cached_tokens"),
                )
                yield {"type": "usage", **state.usage.as_dict()}
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                r = delta_reasoning(delta)
                if r:
                    state.reasoning += r
                    yield {"type": "reasoning-delta", "text": r}
                c = delta.get("content")
                if c:
                    state.text += c
                    yield {"type": "text-delta", "text": c}
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    slot = state.tool_calls.setdefault(idx, {"id": "", "name": "", "argsText": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["argsText"] += fn["arguments"]
                    tcd_event: dict[str, Any] = {
                        "type": "tool-call-delta",
                        "id": slot["id"],
                        "argsText": fn.get("arguments", ""),
                    }
                    if slot["name"]:
                        tcd_event["name"] = slot["name"]
                    yield tcd_event
                fr = choice.get("finish_reason")
                if fr:
                    state.finish_reason = map_finish_reason(fr)
                    for idx in sorted(state.tool_calls):
                        if not state.tool_calls[idx].get("_emitted"):
                            state.tool_calls[idx]["_emitted"] = True
                            yield state.finished_tool_call(idx, client_tool_names)
