"""Rebuild OpenAI-style chat completion messages from stored MessageRow parts."""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

# attachmentId -> (mime, bytes) or None (evicted/missing)
ImageLoader = Callable[[str], Awaitable[tuple[str, bytes] | None]]

PRESENT_OPTIONS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "present_options",
        "description": (
            "Offer the user 2-8 short choices as clickable buttons. Call this instead of "
            "writing a list when you want the user to pick."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["label", "value"],
                    },
                },
                "multi": {"type": "boolean"},
            },
            "required": ["options"],
        },
    },
}


async def build_messages(
    rows: list[Any], settings: dict[str, Any], *, supports_vision: bool, load_image: ImageLoader
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sp = (settings.get("system_prompt") or "").strip()
    if sp:
        out.append({"role": "system", "content": sp})
    for r in rows:
        if r.role == "user":
            content: list[dict[str, Any]] = []
            for p in r.parts:
                if p.get("type") == "text":
                    content.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image":
                    if not supports_vision:
                        content.append({"type": "text", "text": "[image omitted]"})
                        continue
                    loaded = await load_image(p.get("attachmentId", ""))
                    if loaded is None:
                        content.append({"type": "text", "text": "[image expired]"})
                    else:
                        mime, data = loaded
                        b64 = base64.b64encode(data).decode()
                        content.append(
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                        )
            out.append({"role": "user", "content": content})
        elif r.role == "assistant":
            text = "".join(p.get("text", "") for p in r.parts if p.get("type") == "text")
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            calls = [
                {
                    "id": p["id"],
                    "type": "function",
                    "function": {
                        "name": p["name"],
                        "arguments": p.get("argsText") or json.dumps(p.get("args", {})),
                    },
                }
                for p in r.parts
                if p.get("type") == "tool_call"
            ]
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif r.role == "tool":
            for p in r.parts:
                if p.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": p["callId"],
                            "content": json.dumps(p.get("result")),
                        }
                    )
    return out
