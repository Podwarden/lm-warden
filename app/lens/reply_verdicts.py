"""Per-call verdicts and reply counts (analysis spec 8.1, layer 3).

Run on the assistant message ``ReplyCapture`` assembled, after the response is
complete. Text-free: a verdict is ``[k, hash8, code]``, ``k`` the call's
0-based index in the reply's ``tool_calls`` (None for leaked markup, which is
no call), the hash the tool-name HMAC that ``tools_out`` stores (same
secret), and the counts are ints. No argument, reply or schema text is kept
or returned (A-3, A-10).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from app.proxy.forest_fields import hkey

CALL_CODES = ("ok", "json", "unknown_tool", "missing_arg", "type", "leaked_markup")
#: Verdicts kept per reply (``calls_bad`` holds at most this many non-ok ones).
MAX_VERDICTS = 16
#: Tool calls judged per reply, at most (the work bound; a reply is bounded too).
MAX_JUDGED = 1_024

#: Tool-call markup an engine's tool parser should have turned into a call
#: (Hermes/Qwen, Llama 3, Mistral, Kimi, DeepSeek, Granite, Claude-style XML).
_MARKUP = re.compile(
    r"<tool_call>|<\|tool_call\|>|<\|tool_calls_section_begin\|>|<\|tool_call_begin\|>"
    r"|<[｜|]?tool▁calls?▁begin[｜|]?>|\[TOOL_CALLS\]|<\|python_tag\|>|<function=|<tool_use>"
    r"|<function_calls>|<invoke name="
)
#: A tool name near the markup, only to key the verdict to its tool (hashed).
_MARKUP_NAME = re.compile(
    r'"name"\s*:\s*"([^"\\]{1,128})"|<function=([^>\s]{1,128})>|<invoke name="([^"]{1,128})"'
)
#: How far after the marker the name is looked for.
_NAME_WINDOW = 512

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def offered_of(tools: Any) -> dict[str, dict[str, Any]]:
    """Tool name -> its parameters schema, from a chat body's ``tools``
    (OpenAI ``function`` entries; Anthropic ``input_schema`` ones read too).
    References the body's schemas, never copies them."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        f = t.get("function")
        if isinstance(f, dict):
            name, params = f.get("name"), f.get("parameters")
        elif "function" in t:
            continue
        else:
            name, params = t.get("name"), t.get("input_schema")
        if isinstance(name, str) and name:
            out[name] = params if isinstance(params, dict) else {}
    return out


def _of_type(v: Any, t: Any) -> bool:
    """Whether ``v`` is of JSON schema type ``t`` (a name or a list of names);
    an unknown type name accepts anything."""
    if isinstance(t, list):
        return any(_of_type(v, x) for x in t) if t else True
    if not isinstance(t, str) or t not in _JSON_TYPES:
        return True
    if isinstance(v, bool) and t != "boolean":
        return False
    if t == "integer" and isinstance(v, float):
        return v.is_integer()
    return isinstance(v, _JSON_TYPES[t])


def _args_of(raw: Any) -> dict[str, Any] | None:
    """The arguments as an object; None when they are not a JSON object.
    Empty or missing arguments are the empty object (a call with no parameters)."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        v = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return v if isinstance(v, dict) else None


def _code(name: Any, raw_args: Any, offered: Mapping[str, dict[str, Any]]) -> str:
    if not isinstance(name, str) or not name or name not in offered:
        return "unknown_tool"
    args = _args_of(raw_args)
    if args is None:
        return "json"
    schema = offered[name]
    req = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(req, list) and any(isinstance(k, str) and k not in args for k in req):
        return "missing_arg"
    props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(props, dict):
        for k, v in args.items():
            p = props.get(k)
            if isinstance(p, dict) and "type" in p and not _of_type(v, p["type"]):
                return "type"
    return "ok"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _leaked(content: Any, secret: str) -> list[str] | None:
    """``[hash8, "leaked_markup"]`` when the reply text holds tool-call markup."""
    text = _text(content)
    if not text:
        return None
    m = _MARKUP.search(text)
    if m is None:
        return None
    n = _MARKUP_NAME.search(text, m.start(), m.end() + _NAME_WINDOW)
    name = next((g for g in n.groups() if g), None) if n is not None else None
    return [(hkey(secret, name, 8) or "") if name else "", "leaked_markup"]


def judge_calls(
    message: dict[str, Any], offered: Mapping[str, dict[str, Any]], secret: str
) -> list[list[Any]]:
    """``[[k, hash8, code], ...]``: every tool call's verdict in reply order
    (parallel calls each get theirs; ``k`` is the call's index in
    ``tool_calls``, an entry that is not a call keeping its place), then
    ``[None, hash8, "leaked_markup"]`` when the text holds tool-call markup.
    At most ``MAX_JUDGED`` calls. A call with no name hashes to ``""``."""
    if not isinstance(message, dict):
        return []
    if not isinstance(offered, Mapping):
        offered = {}
    out: list[list[Any]] = []
    calls = message.get("tool_calls")
    for k, tc in enumerate(calls[:MAX_JUDGED] if isinstance(calls, list) else ()):
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        h = (hkey(secret, name, 8) or "") if isinstance(name, str) and name else ""
        out.append([k, h, _code(name, fn.get("arguments"), offered)])
    leak = _leaked(message.get("content"), secret)
    if leak is not None:
        out.append([None, *leak])
    return out


def call_verdicts(
    message: dict[str, Any], offered: Mapping[str, dict[str, Any]], secret: str
) -> list[list[Any]]:
    """What ``calls_bad`` stores: the non-ok verdicts of ``judge_calls``
    (``[k, hash8, code]``), filtered first, then at most ``MAX_VERDICTS``
    (I4: 16 ok calls then 2 bad ones keep the 2 bad, at k = 16 and 17)."""
    return [v for v in judge_calls(message, offered, secret) if v[2] != "ok"][:MAX_VERDICTS]


def reply_counts(message: dict[str, Any]) -> tuple[int, int]:
    """``(reply_chars, calls_n)``: the visible text's length (reasoning not
    counted) and the number of tool calls."""
    if not isinstance(message, dict):
        return 0, 0
    calls = message.get("tool_calls")
    n = sum(1 for tc in calls if isinstance(tc, dict)) if isinstance(calls, list) else 0
    return len(_text(message.get("content"))), n
