# app/cache_obs/canonical.py
"""A request body as the engine sees its prefix: a chain of chunk digests.

Elements in engine order -- ``tools`` (one element), then each message (the
keys that reach the prompt: role, content, name, tool_calls, ...), or ``prompt``
for /v1/completions -- serialised deterministically and joined with \\x1e.
Large leaves (data: URLs, image/audio payloads, any string over 64 KiB) are
replaced by one digest of the whole string first: the same image still gives
the same chunk and a changed one still breaks the prefix, but a 3 MiB image
costs one hash instead of 12k chunk hashes and 12k index entries. The byte stream is cut into CHUNK_BYTES chunks and each
chunk's key chains the previous one (like vLLM's block hash), so a key means
"this exact prefix", never "this block anywhere". blake2b, never hash(): keys
must agree across processes and, later, nodes. Fields that do not reach the
prompt (model, temperature, ...) are ignored.

``cache_salt`` is the one non-prompt field that counts (#300). vLLM puts it in
the first block's hash (``kv_cache_utils.py`` L579-580 @v0.26.0), so requests
with different salts never share a block. A non-empty string salt becomes the
chain's root, the parent of the first chunk and a prefix of every message
digest, so a salted prefix matches only the same salt. Only its digest is kept.
No salt gives an empty root, and the keys are byte-identical to the unsalted
cobs1 chain. The caller passes ``honour_salt=False`` for an engine that ignores
the field (llama.cpp), which really does reuse a prefix across salts.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from typing import Any

SCHEMA = b"cobs1"
CHUNK_BYTES = 256
MAX_CHAIN_BYTES = 2 * 1024 * 1024
#: a string leaf longer than this is replaced by its digest
BLOB_CHARS = 64 * 1024
_SEP = b"\x1e"
#: message keys that reach the chat template (everything else is client metadata)
_MSG_KEYS = (
    "role",
    "content",
    "name",
    "tool_calls",
    "tool_call_id",
    "function_call",
    "reasoning_content",
)
#: (container key, leaf key) pairs that always hold a media payload
_BLOB_PATHS = {("image_url", "url"), ("input_audio", "data")}


class _TooBig(Exception):
    pass


@dataclass(frozen=True)
class Chain:
    chunks: tuple[bytes, ...]
    total_bytes: int
    #: byte offset where each element starts in the joined stream
    elem_starts: tuple[int, ...]
    #: 1 when element 0 is ``tools``, else 0: message k is element k + msg_offset
    msg_offset: int
    #: one digest per message, independent of position
    message_hashes: tuple[bytes, ...]


def _ser(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8", "surrogatepass"
    )


def _blob(s: str) -> dict[str, str]:
    d = hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()
    return {"__cobs_blob__": d}


class _Shrinker:
    """Replaces large string leaves by a digest and tallies the characters that
    remain, raising _TooBig as soon as the body is clearly over MAX_CHAIN_BYTES
    (UTF-8 bytes >= characters), so an oversize body is never serialised."""

    def __init__(self) -> None:
        self.chars = 0

    def _add(self, n: int) -> None:
        self.chars += n
        if self.chars > MAX_CHAIN_BYTES:
            raise _TooBig

    def walk(self, obj: Any, parent: str | None = None, key: str | None = None) -> Any:
        if isinstance(obj, str):
            if len(obj) > BLOB_CHARS or obj.startswith("data:") or (parent, key) in _BLOB_PATHS:
                self._add(50)
                return _blob(obj)
            self._add(len(obj))
            return obj
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    self._add(len(k))
                out[k] = self.walk(v, key, k)
            return out
        if isinstance(obj, list):
            return [self.walk(v, key, None) for v in obj]
        return obj


def _elements(body: dict[str, Any]) -> tuple[list[bytes], int, int] | None:
    """(elements, msg_offset, n_messages) or None when there is no prompt.
    Raises _TooBig for a body clearly over MAX_CHAIN_BYTES."""
    sh = _Shrinker()
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        parts: list[Any] = []
        offset = 0
        tools = body.get("tools")
        if tools:
            parts.append(sh.walk(tools))
            offset = 1
        for m in msgs:
            if isinstance(m, dict):
                parts.append(sh.walk({k: m[k] for k in _MSG_KEYS if k in m}))
            else:
                parts.append(sh.walk(m))
        return [_ser(p) for p in parts], offset, len(msgs)
    prompt = body.get("prompt")
    if isinstance(prompt, str | list) and prompt:
        return [_ser(sh.walk(prompt))], 0, 0
    return None


def _root(body: dict[str, Any], honour_salt: bool) -> bytes:
    """b"" without a salt; otherwise a digest of it. vLLM 400s an empty or
    non-string salt (chat_completion/protocol.py L927-935), so that is no salt."""
    salt = body.get("cache_salt") if honour_salt else None
    if not isinstance(salt, str) or not salt:
        return b""
    return hashlib.blake2b(
        SCHEMA + b"salt" + salt.encode("utf-8", "surrogatepass"), digest_size=16
    ).digest()


def chain_of(body: object, *, honour_salt: bool = True) -> Chain | None:
    if not isinstance(body, dict):
        return None
    try:
        got = _elements(body)
    except _TooBig:
        return None
    if got is None:
        return None
    elems, offset, n_msgs = got
    starts: list[int] = []
    pos = 0
    for i, e in enumerate(elems):
        if i:
            pos += len(_SEP)
        starts.append(pos)
        pos += len(e)
    if pos > MAX_CHAIN_BYTES:
        return None
    stream = _SEP.join(elems)
    root = _root(body, honour_salt)
    keys: list[bytes] = []
    prev = root
    for i in range(0, len(stream), CHUNK_BYTES):
        prev = hashlib.blake2b(SCHEMA + prev + stream[i : i + CHUNK_BYTES], digest_size=16).digest()
        keys.append(prev)
    msg_hashes = tuple(
        hashlib.blake2b(SCHEMA + root + b"m" + e, digest_size=16).digest()
        for e in elems[offset : offset + n_msgs]
    )
    return Chain(tuple(keys), len(stream), tuple(starts), offset, msg_hashes)


def element_of_byte(chain: Chain, pos: int) -> int:
    """Index of the element containing byte ``pos`` (a separator belongs to the preceding element)."""
    return max(0, bisect.bisect_right(chain.elem_starts, pos) - 1)
