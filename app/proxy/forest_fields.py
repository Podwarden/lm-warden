"""Per-request fields for the session forest (spec 2026-10-05 §3).

Pure and fail-open: every function returns an empty/None value on input it does
not understand. The proxy calls these on its hot path, so nothing here
tokenizes, and SSE frames are parsed only behind a cheap byte gate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any

FAMILIES: tuple[str, ...] = ("read", "edit", "shell", "web", "agent", "other")

# Lower-cased tool names of the harnesses we know (Claude Code, OpenCode, Codex,
# Hermes, Pi). Unknown names, MCP tools included, are "other".
_FAMILY_OF: dict[str, str] = {}
for _fam, _names in {
    "read": (
        "read",
        "glob",
        "grep",
        "ls",
        "list",
        "view",
        "read_file",
        "search",
        "find",
        "notebookread",
    ),
    "edit": (
        "edit",
        "write",
        "multiedit",
        "notebookedit",
        "apply_patch",
        "patch",
        "write_file",
        "str_replace",
        "create_file",
    ),
    "shell": ("bash", "bashoutput", "killshell", "shell", "exec", "run", "terminal", "local_shell"),
    "web": ("webfetch", "websearch", "fetch", "web_search", "browser"),
    "agent": ("task", "agent", "spawn_agent", "subagent"),
}.items():
    for _n in _names:
        _FAMILY_OF[_n] = _fam

_MAX_IN, _MAX_OUT = 32, 16


def tool_family(name: str | None) -> str:
    if not isinstance(name, str) or not name:
        return "other"
    return _FAMILY_OF.get(name.strip().lower(), "other")


def hkey(secret: str, raw: str | None, n: int = 16) -> str | None:
    """HMAC-SHA256 hex prefix of ``raw``; the raw value is never stored."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        return hmac.new(
            secret.encode(errors="surrogatepass"),
            raw.encode(errors="surrogatepass"),
            hashlib.sha256,
        ).hexdigest()[:n]
    except Exception:  # noqa: BLE001
        return None


def clean_batch_id(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    v = raw.strip()
    if not v or len(v) > 64 or not all(32 < ord(c) < 127 or c == " " for c in v):
        return None
    return v


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                n += len(part["text"])
        return n
    return 0


def _failed(content: Any) -> bool:
    text = content if isinstance(content, str) else ""
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = str(content[0].get("text") or "")
    # messages_translate.py prefixes an Anthropic is_error result with "Error".
    return text.startswith("Error")


def tools_in_from_body(body: Any) -> list[list[Any]]:
    """Tool results this request carries since the last assistant turn."""
    try:
        msgs = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(msgs, list):
            return []
        names: dict[str, str] = {}
        last_asst = -1
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_asst = i
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        fname = fn.get("name") if fn else None
                        if isinstance(tc.get("id"), str):
                            names[tc["id"]] = fname if isinstance(fname, str) else ""
        out: list[list[Any]] = []
        for m in msgs[last_asst + 1 :]:
            if not (isinstance(m, dict) and m.get("role") == "tool"):
                continue
            content = m.get("content")
            fam = tool_family(names.get(str(m.get("tool_call_id") or "")))
            out.append([fam, math.ceil(_text_len(content) / 4), _failed(content)])
            if len(out) >= _MAX_IN:
                break
        return out
    except Exception:  # noqa: BLE001 -- display metadata only
        return []


def tools_out_from_message(message: Any, secret: str) -> list[list[Any]]:
    try:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list):
            return []
        out: list[list[Any]] = []
        for tc in calls:
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if not isinstance(name, str) or not name:
                continue
            h = hkey(secret, name, 8)
            if h is None:  # same rule as ToolsOutAccumulator.result: no hash, no entry
                continue
            out.append([tool_family(name), h])
            if len(out) >= _MAX_OUT:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


class ToolsOutAccumulator:
    """Tool names seen in streamed chat-completion frames, by tool index."""

    def __init__(self) -> None:
        self._names: dict[int, str] = {}

    def feed(self, line: bytes) -> None:
        try:
            if not isinstance(line, bytes) or b'"tool_calls"' not in line:
                return
            # lstrip first, as the proxy's TTFT check does: " data: {...}" is still a data line.
            line = line.lstrip()
            payload = line[5:].strip() if line.startswith(b"data:") else line.strip()
            ev = json.loads(payload)
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") if isinstance(ch, dict) else None
                for tc in (delta or {}).get("tool_calls") or []:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    name = fn.get("name") if isinstance(fn, dict) else None
                    idx = (
                        tc.get("index", len(self._names))
                        if isinstance(tc, dict)
                        else len(self._names)
                    )
                    if (
                        isinstance(name, str)
                        and name
                        and isinstance(idx, int)
                        and idx not in self._names
                        and len(self._names) < _MAX_OUT
                    ):
                        self._names[idx] = name
        except Exception:  # noqa: BLE001
            return

    def result(self, secret: str) -> list[list[Any]]:
        try:
            out: list[list[Any]] = []
            for _, n in sorted(self._names.items()):
                h = hkey(secret, n, 8)
                if h is not None:
                    out.append([tool_family(n), h])
            return out[:_MAX_OUT]
        except Exception:  # noqa: BLE001
            return []
