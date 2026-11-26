"""Prompt Lens probes: text-free digests of messages, lines and tools.

Everything here is a pure function of its inputs and returns keyed hashes
(8 bytes, under the cobs2 network key) or codes. No prompt text is kept or
returned (A-3, A-10).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from app.cobs2.canonical import _canon_message, _content_parts, canon_json, loose_message
from app.cobs2.ids import _h
from app.cobs2.keys import NetworkKey
from app.lens.schema_lint import MAX_TOOLS
from app.proxy.forest_fields import hkey

PROBE_ORDER: tuple[str, ...] = ("noise", "no_reasoning", "text_only", "ws")

MAX_LINES = 1024
CHUNK_CHARS = 4096

SPAN_CLASSES = (
    "date",
    "time",
    "path",
    "uuid_or_hex_id",
    "version",
    "number",
    "url",
    "provider_or_model_id",
    "user_name_or_email",
    "other",
)
VOLATILE_CLASSES = frozenset(
    {"date", "time", "path", "uuid_or_hex_id", "version", "provider_or_model_id"}
)


def _d8(key: NetworkKey, label: bytes, salt: bytes, payload: bytes) -> bytes:
    """8 bytes of ``_h`` (after its epoch byte)."""
    return _h(key, label, salt, payload)[1:9]


def _map_parts(parts: list[dict[str, Any]], fn: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in parts:
        t = p.get("t")
        if t == "tool_result":
            out.append({**p, "parts": _map_parts(p["parts"], fn)})
            continue
        q = fn(p)
        if q is not None:
            out.append(q)
    return out


def _drop_reasoning(p: dict[str, Any]) -> dict[str, Any] | None:
    return None if p.get("t") == "reasoning" else p


def _drop_media(p: dict[str, Any]) -> dict[str, Any] | None:
    return None if p.get("t") == "media" else p


def _collapse_ws(p: dict[str, Any]) -> dict[str, Any] | None:
    if p.get("t") in ("text", "reasoning") and isinstance(p.get("v"), str):
        return {**p, "v": " ".join(p["v"].split())}
    return p


def message_probes(
    msg: dict[str, Any], key: NetworkKey, salt: bytes
) -> tuple[bytes, bytes, bytes, bytes]:
    """Four 8-byte digests of one chat message, in ``PROBE_ORDER``: the loose
    form, without reasoning, without media, and with whitespace collapsed."""
    c = _canon_message(msg, {})
    role, parts = c["role"], c["parts"]

    def dig(label: bytes, body: Any) -> bytes:
        return _d8(key, label, salt, canon_json(body))

    return (
        dig(b"lens-noise", loose_message(c)),
        dig(b"lens-norsn", {"role": role, "parts": _map_parts(parts, _drop_reasoning)}),
        dig(b"lens-text", {"role": role, "parts": _map_parts(parts, _drop_media)}),
        dig(b"lens-ws", {"role": role, "parts": _map_parts(parts, _collapse_ws)}),
    )


def sys_pieces(text: str) -> tuple[list[str], str]:
    """The first ``MAX_LINES`` lines of ``text`` without their endings, and
    the rest joined with ``"\n"`` (what ``sys_line_digests`` chunks)."""
    lines = text.splitlines()
    return lines[:MAX_LINES], "\n".join(lines[MAX_LINES:])


def sys_line_digests(text: str, key: NetworkKey, salt: bytes) -> tuple[bytes, ...]:
    """One 8-byte digest per line for the first 1,024 lines; the rest, joined
    with ``"\n"``, is cut into 4,096-char chunks, one digest each. Bounded: a
    1 MB single line is one digest. A line is digested without its line
    ending (M1): a CRLF/LF flip or a trailing newline changes no digest, so
    it locates no line (``sys_noise`` with no span line or class)."""
    pieces, rest = sys_pieces(text)
    pieces.extend(rest[i : i + CHUNK_CHARS] for i in range(0, len(rest), CHUNK_CHARS))
    return tuple(_d8(key, b"lens-line", salt, s.encode("utf-8", "surrogatepass")) for s in pieces)


def first_changed_line(a: Sequence[bytes], b: Sequence[bytes]) -> tuple[int, int] | None:
    """(1-based first differing line, number of differing lines), or None when
    equal. Lines past the shorter side count as differing."""
    n = max(len(a), len(b))
    first = 0
    count = 0
    for i in range(n):
        if i >= len(a) or i >= len(b) or a[i] != b[i]:
            if not count:
                first = i + 1
            count += 1
    return (first, count) if count else None


_RX: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b")),
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b")),
    ("url", re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.I)),
    (
        "path",
        re.compile(r"(?<![\w:/.])(?:~|\.{1,2})?/[\w.\-@+]+(?:/[\w.\-@+]+)+|\b[A-Za-z]:\\[\w.\\-]+"),
    ),
    (
        "uuid_or_hex_id",
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b(?=[0-9a-f]*\d)[0-9a-f]{16,}\b",
            re.I,
        ),
    ),
    ("version", re.compile(r"\bv?\d+\.\d+\.\d+(?:[.\-+][\w.]+)?\b")),
    ("number", re.compile(r"\b\d{2,}\b")),
    (
        "provider_or_model_id",
        re.compile(r"\b(?:provider|model)(?:[ _-]?(?:id|name))?\s*[:=]", re.I),
    ),
    (
        "user_name_or_email",
        re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b(?:user ?name|login)\s*[:=]", re.I),
    ),
)
# SPAN_CLASSES order, except url is tested ahead of path (a URL contains path
# lookalikes) and number after version; "other" is the fallback.
_RX_BY_CLASS = dict(_RX)
_RX_ORDER = (
    "date",
    "time",
    "url",
    "path",
    "uuid_or_hex_id",
    "version",
    "provider_or_model_id",
    "user_name_or_email",
    "number",
)


def span_class(line: str) -> str:
    """The class of the volatile-looking span in ``line`` (first match wins)."""
    for name in _RX_ORDER:
        if _RX_BY_CLASS[name].search(line):
            return name
    return "other"


def tool_def_hashes(
    canon_tools: Sequence[Any], key: NetworkKey, salt: bytes
) -> PackedDigests | None:
    """One 8-byte digest per canonical tool definition, in request order,
    packed into one bytes (``PackedDigests``). None above
    ``schema_lint.MAX_TOOLS`` tools: the hashes are unknown and the causes use
    the loose comparison (a ``{}`` tool is about 2 bytes, so only this count
    cap bounds the memory a body can pin)."""
    if len(canon_tools) > MAX_TOOLS:
        return None
    return PackedDigests(b"".join(_d8(key, b"lens-tool", salt, canon_json(t)) for t in canon_tools))


def content_hash(content: Any, key: NetworkKey, salt: bytes) -> bytes:
    """8-byte digest of a tool result's content (no ``tool_call_id``)."""
    return result_hash(_content_parts(content), key, salt)


def result_hash(parts: list[dict[str, Any]], key: NetworkKey, salt: bytes) -> bytes:
    """``content_hash`` from the canonical parts a ``tool_result`` part holds
    (``canonicalize`` already made them from the content)."""
    return _d8(key, b"lens-res", salt, canon_json(parts))


def args8(arguments: str | None, secret: str) -> str:
    """8 hex chars of the arguments as canonical JSON (sorted keys); text that
    is not JSON is hashed raw. Same secret as ``forest_fields.hkey``. Empty or
    missing arguments give ``""``."""
    if not arguments:
        return ""
    try:
        raw = json.dumps(
            json.loads(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except (ValueError, RecursionError):
        raw = arguments
    return hkey(secret, raw, 8) or ""


class PackedDigests(Sequence[bytes]):
    """8-byte digests packed into one ``bytes`` (one object, not one per
    digest: 2,048 digests are 16 KB plus one header, where a tuple of bytes
    objects is about 70 KB in CPython). Indexes like the tuple it replaces;
    equal to any sequence of the same digests."""

    __slots__ = ("raw",)

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    @classmethod
    def of(cls, digests: Sequence[bytes] | PackedDigests) -> PackedDigests:
        if isinstance(digests, PackedDigests):
            return digests
        return cls(b"".join(digests))

    def __len__(self) -> int:
        return len(self.raw) // 8

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return PackedDigests(b"".join(self[j] for j in range(*i.indices(len(self)))))
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        return self.raw[8 * i : 8 * i + 8]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PackedDigests):
            return self.raw == other.raw
        if isinstance(other, Sequence) and not isinstance(other, bytes | str):
            return len(other) == len(self) and all(a == b for a, b in zip(self, other, strict=True))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:
        return f"PackedDigests(n={len(self)})"
