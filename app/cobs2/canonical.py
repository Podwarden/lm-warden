"""The canonical message form (cobs2 spec sections 6.1-6.2).

Every API shape the proxy accepts reaches ``_forward`` as an OpenAI chat body
(``/v1/messages`` and ``/v1/responses`` are translated first). This module turns
that body into a client-independent canonical form that identity hashes are
taken over, in two strengths: exact, and loose (volatile noise removed).

Canonical message: ``{"role": str, "parts": [part, ...]}`` with parts

  ``{"t": "text", "v": str}``
  ``{"t": "media", "sha256": hex, "mime": str}``
  ``{"t": "tool_call", "name": str, "args": <JSON value>, "ref": int}``
  ``{"t": "tool_result", "ref": int, "parts": [...]}``
  ``{"t": "reasoning", "v": str}``

Raw tool-call ids never appear. ``ref`` is local: a call's ``ref`` is its index
within its own assistant message, a result's ``ref`` is the index of its call in
the nearest preceding assistant message holding that id (-1 when none does).
So a message canonicalised alone equals the same message inside a later request.

Pure, no I/O, never raises on bad input.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import json.encoder as _json_encoder
import mimetypes
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_to_bytes

from app.proxy.session_id import PREAMBLE_PREFIXES, text_of

#: Version of the volatile-span table ``loose`` applies. Bump on any change to it:
#: loose ids from different versions are not comparable.
LOOSE_TABLE_VERSION = 1

_HEAD_ROLES = ("system", "developer")

#: Tagged blocks the loose form removes, as (open, close), matched case-insensitively.
_BLOCK_TAGS = (
    ("<system-reminder>", "</system-reminder>"),
    ("<environment_context>", "</environment_context>"),
)
#: a line holding only an ISO date or date-time
_DATE_LINE_RE = re.compile(
    r"^[ \t]*\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?[ \t]*$",
    re.MULTILINE,
)
#: Necessary conditions for a ``_DATE_LINE_RE`` match (at the start, or after a
#: newline). Both are cheap literal-led scans, so text with no date line skips
#: the per-position MULTILINE scan.
_DATE_AT_START_RE = re.compile(r"[ \t]*\d{4}-\d{2}-\d{2}")
_DATE_AFTER_NL_RE = re.compile(r"\n[ \t]*\d{4}-\d{2}-\d{2}")
_PREAMBLE_SCAN = max((len(p) for p in PREAMBLE_PREFIXES), default=0)


@dataclass(frozen=True)
class Canon:
    head: list[dict[str, Any]]
    tools: list[Any]
    messages: list[dict[str, Any]]


#: One reusable encoder: ``json.dumps`` rebuilds it on every call, which is most
#: of the cost for the many small messages of a long conversation. Same output
#: as ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``.
_CANON_ENC = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _not_json(o: Any) -> Any:
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


#: The C encoder ``JSONEncoder.encode`` would build per call, built once. No
#: circular-reference markers: a canonical form is a tree (a cycle would end in
#: RecursionError, which every caller already fails open on).
_c_make_encoder = getattr(_json_encoder, "c_make_encoder", None)
_C_ENC = (
    _c_make_encoder(
        None, _not_json, _json_encoder.encode_basestring, None, ":", ",", True, False, True
    )
    if _c_make_encoder is not None
    else None
)


def canon_json(obj: Any) -> bytes:
    if _C_ENC is not None and not isinstance(obj, str):
        return "".join(_C_ENC(obj, 0)).encode("utf-8", "surrogatepass")
    return _CANON_ENC.encode(obj).encode("utf-8", "surrogatepass")


def _json_value(v: Any) -> Any:
    """``v`` as a plain JSON value (what a round trip through JSON gives)."""
    try:
        return json.loads(json.dumps(v))
    except (TypeError, ValueError):
        return str(v)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media(url: Any) -> dict[str, Any]:
    url = url if isinstance(url, str) else ""
    if url.startswith("data:") and "," in url:
        header, _, payload = url[5:].partition(",")
        meta = header.split(";")
        mime = meta[0].strip()
        try:
            if "base64" in (m.strip() for m in meta[1:]):
                data = base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
            else:
                data = unquote_to_bytes(payload)
            return {"t": "media", "sha256": _sha(data), "mime": mime}
        except (binascii.Error, ValueError):
            pass
    mime = ""
    if not url.startswith("data:"):
        mime = mimetypes.guess_type(url.split("?", 1)[0])[0] or ""
    return {"t": "media", "sha256": _sha(url.encode("utf-8", "surrogatepass")), "mime": mime}


def _unknown(p: Any) -> dict[str, Any]:
    return {
        "t": "media",
        "sha256": _sha(canon_json(_json_value(p))),
        "mime": "application/x-unknown",
    }


def _dump(v: Any) -> str:
    """Compact JSON in the CLIENT's key order: chat templates (``tojson``) keep it,
    so the exact form must tell two orders apart."""
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def _tool_text(t: Any) -> str:
    """``_dump(_json_value(t))`` without the round trip when ``t`` already dumps:
    a value parsed from JSON dumps to the same text either way."""
    try:
        return _dump(t)
    except (TypeError, ValueError, RecursionError):
        return _dump(_json_value(t))


def _resorted(s: str) -> str:
    """``s`` re-serialised with sorted keys when it is JSON, else unchanged."""
    try:
        return canon_json(json.loads(s)).decode("utf-8", "surrogatepass")
    except (ValueError, RecursionError):
        return s


def _content_parts(content: Any) -> list[dict[str, Any]]:
    """Chat ``content`` (string or part list) as canonical parts."""
    out: list[dict[str, Any]] = []

    def text(v: str) -> None:
        if not v:
            return
        out.append({"t": "text", "v": v})

    if isinstance(content, str):
        text(content)
        return out
    if isinstance(content, dict):
        out.append(_unknown(content))
        return out
    if not isinstance(content, list):
        return out
    for p in content:
        if isinstance(p, str):
            text(p)
        elif not isinstance(p, dict):
            continue
        elif p.get("type") in ("text", "input_text", "output_text") or (
            "type" not in p and isinstance(p.get("text"), str)
        ):
            text(text_of([p]))
        elif p.get("type") == "image_url":
            iu = p.get("image_url")
            out.append(_media(iu.get("url") if isinstance(iu, dict) else iu))
        elif p.get("type") == "input_audio" and isinstance(p.get("input_audio"), dict):
            a = p["input_audio"]
            raw = a.get("data") if isinstance(a.get("data"), str) else ""
            try:
                data = base64.b64decode(raw)
            except (binascii.Error, ValueError):
                data = raw.encode("utf-8", "surrogatepass")
            out.append({"t": "media", "sha256": _sha(data), "mime": f"audio/{a.get('format', '')}"})
        elif p.get("type") in ("video_url", "file", "input_image"):
            inner = p.get(p["type"])
            url = inner.get("url") if isinstance(inner, dict) else inner
            out.append(_media(url if isinstance(url, str) else canon_json(_json_value(p)).decode()))
        else:
            out.append(_unknown(p))
    return out


def _args(raw: Any) -> str:
    """Tool-call arguments as compact JSON in the client's key order; a string
    that does not parse is kept raw (it can never equal valid JSON text)."""
    if isinstance(raw, str):
        try:
            return _dump(json.loads(raw))
        except (ValueError, RecursionError):
            return raw
    return _dump(_json_value(raw))


def _canon_message(m: dict[str, Any], seen: dict[str, int]) -> dict[str, Any]:
    """One chat message. ``seen`` maps tool-call id -> index in the nearest
    preceding assistant message (updated here for assistant messages)."""
    role = m.get("role")
    role = role if isinstance(role, str) else ""
    if role == "tool":
        cid = m.get("tool_call_id")
        ref = seen.get(cid, -1) if isinstance(cid, str) else -1
        return {
            "role": role,
            "parts": [{"t": "tool_result", "ref": ref, "parts": _content_parts(m.get("content"))}],
        }
    parts: list[dict[str, Any]] = []
    if role == "assistant":
        for key in ("reasoning_content", "reasoning"):
            r = m.get(key)
            if isinstance(r, str) and r:
                parts.append({"t": "reasoning", "v": r})
                break
    parts.extend(_content_parts(m.get("content")))
    calls = m.get("tool_calls")
    if role == "assistant" and isinstance(calls, list):
        index = 0
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            fn = fn if isinstance(fn, dict) else {}
            name = fn.get("name")
            parts.append(
                {
                    "t": "tool_call",
                    "name": name if isinstance(name, str) else "",
                    "args": _args(fn.get("arguments")),
                    "ref": index,
                }
            )
            cid = call.get("id")
            if isinstance(cid, str):
                seen[cid] = index
            index += 1
    return {"role": role, "parts": parts}


def canonicalize(body: dict[str, Any]) -> Canon | None:
    """The canonical form of an OpenAI chat body, or None when it has no
    ``messages`` list or cannot be read."""
    try:
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return None
        tools = body.get("tools")
        head: list[dict[str, Any]] = []
        msgs: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for m in body["messages"]:
            if not isinstance(m, dict):
                continue
            c = _canon_message(m, seen)
            (head if c["role"] in _HEAD_ROLES else msgs).append(c)
        return Canon(
            head=head,
            tools=[_tool_text(t) for t in tools] if isinstance(tools, list) else [],
            messages=msgs,
        )
    except Exception:  # noqa: BLE001 -- identity input must never break a request
        return None


# ---- loose ---------------------------------------------------------------


def _strip_blocks(v: str, open_tag: str, close_tag: str) -> str:
    """``v`` without ``open_tag ... close_tag`` blocks, in linear time. An
    unclosed tag leaves the rest of the text as it is."""
    low = v.lower()
    if len(low) != len(v):  # case folding changed the length: no safe offsets
        low = v
    out: list[str] = []
    pos = 0
    while True:
        i = low.find(open_tag, pos)
        if i < 0:
            break
        j = low.find(close_tag, i + len(open_tag))
        if j < 0:
            break
        out.append(v[pos:i])
        pos = j + len(close_tag)
    out.append(v[pos:])
    return "".join(out)


def boilerplate_chars(v: str) -> int:
    """Chars of harness boilerplate in ``v`` (sessions pages, dead weight): the
    ``<system-reminder>`` / ``<environment_context>`` blocks and date-only lines
    the loose form drops. Same linear scans and gates as ``_loose_text``."""
    n = 0
    if "<" in v:
        for open_tag, close_tag in _BLOCK_TAGS:
            stripped = _strip_blocks(v, open_tag, close_tag)
            n += len(v) - len(stripped)
            v = stripped
    if _DATE_AT_START_RE.match(v) or _DATE_AFTER_NL_RE.search(v):
        n += sum(len(m.group(0).strip()) for m in _DATE_LINE_RE.finditer(v))
    return n


def _loose_text(v: str) -> str:
    # Hot path (every message of every request): each step is gated or done in
    # C so a 200 KB body stays inside the hashing budget. Output is identical to
    # the plain form: strip blocks, drop date lines, collapse whitespace.
    if "<" in v:
        for open_tag, close_tag in _BLOCK_TAGS:
            v = _strip_blocks(v, open_tag, close_tag)
    if _DATE_AT_START_RE.match(v) or _DATE_AFTER_NL_RE.search(v):
        v = _DATE_LINE_RE.sub("", v)
    # str.split() splits on exactly the characters re's \s matches (isspace).
    norm = " ".join(v.split())
    # The prefixes are ASCII, so lowering only the head decides the same thing.
    head = norm[: _PREAMBLE_SCAN + 1].lower()
    if any(head.startswith(p) for p in PREAMBLE_PREFIXES):
        return ""
    return norm


def _loose_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in parts:
        t = p.get("t")
        if t == "reasoning":
            continue
        if t == "text":
            v = _loose_text(p["v"])
            if v:
                out.append({"t": "text", "v": v})
        elif t == "tool_result":
            out.append({**p, "parts": _loose_parts(p["parts"])})
        elif t == "tool_call":
            out.append({**p, "args": _resorted(p["args"])})
        else:
            out.append(p)
    return out


def _loose_msgs(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": m["role"], "parts": _loose_parts(m["parts"])} for m in msgs]


def loose(canon: Canon) -> Canon:
    """``canon`` without reasoning, volatile spans (table version
    ``LOOSE_TABLE_VERSION``) and whitespace differences. Never mutates its input."""
    return Canon(
        head=_loose_msgs(canon.head),
        tools=[_resorted(t) if isinstance(t, str) else t for t in canon.tools],
        messages=_loose_msgs(canon.messages),
    )


def loose_message(m: dict[str, Any]) -> dict[str, Any]:
    """One canonical message in its loose form (what ``loose`` does to each)."""
    return {"role": m["role"], "parts": _loose_parts(m["parts"])}


def loose_head(canon: Canon) -> tuple[list[dict[str, Any]], list[Any]]:
    """The loose form of ``canon``'s head and tools only."""
    lc = loose(Canon(head=canon.head, tools=canon.tools, messages=[]))
    return lc.head, lc.tools


def loose_text(v: str) -> str:
    """One text in its loose form (what ``loose`` does to a text part)."""
    return _loose_text(v)
