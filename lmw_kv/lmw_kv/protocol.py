"""Wire protocol of the KV store daemon, version 1, minor 1.

One JSON object per line (UTF-8, ``\\n`` terminated) in each direction over a
Unix stream socket. A request is ``{"op": str, "id": int, ...}``; the reply is
``{"id": int, "ok": bool, ...}`` and, when ``ok`` is false, an ``error`` string
``"<code>: <message>"``. Byte fields (keys, ids) are lower-case hex. The
protocol is engine-neutral: it speaks of namespaces, block keys, groups and byte
slots, nothing else. ``PROTOCOL.md`` is the full reference.

Minor 1 adds optional fields (each with a stated default, so a minor 0 request means
what it always meant) and three ops. The daemon ignores fields it does not know.
"""

from __future__ import annotations

import json
from typing import Any

VERSION = 1
MINOR = 1
MAX_LINE = 8 << 20  # bytes per request or reply line

# op -> the request fields it takes (besides "op" and "id"); every field added in
# minor 1 is optional (the comments give the default).
OPS: dict[str, tuple[str, ...]] = {
    "hello": (),
    # nvme: true; owner: ""; model: "" (16 bytes); role: "attention"; rank: 0
    "register_ns": ("namespace", "groups", "nvme", "owner", "model", "role", "rank"),
    "lookup": ("ns", "keys"),
    # tenant, scope_tag: "" (16 bytes); pos: the key (32 bytes)
    "begin_save": ("ns", "key", "parent", "scope", "tenant", "scope_tag", "pos"),
    "commit_save": ("ns", "key"),
    "abort_save": ("ns", "key"),
    "begin_load": ("ns", "keys", "tenant"),  # tenant: ""
    "end_load": ("handle",),
    "record_hits": ("tag", "ram_tokens", "nvme_tokens", "load_s", "tenant"),  # tenant: ""
    "pop_hits": ("tag",),
    "stats": (),
    # minor 1
    "release_owner": ("owner",),
    "retire_scopes": ("scope_tags",),
    "set_policy": ("policy",),
}

# fields that appear only in replies
REPLY_FIELDS = (
    "protocol",
    "minor",
    "ram_path",
    "ram_bytes",
    "ns",
    "n",
    "offsets",
    "handle",
    "slots",
    "nvme",
    "hits",
    "stats",
    "released",
    "added",
    "error",
)

ENVELOPE_FIELDS = ("op", "id", "ok")

# error codes
BAD_REQUEST = "bad_request"
UNKNOWN_OP = "unknown_op"
INTERNAL = "internal"


class ProtocolError(Exception):
    """A request the daemon cannot accept; ``code`` goes into the reply."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def encode(msg: dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    try:
        msg = json.loads(line)
    except ValueError as exc:
        raise ProtocolError(BAD_REQUEST, f"not JSON: {exc}") from None
    if not isinstance(msg, dict):
        raise ProtocolError(BAD_REQUEST, "a message is a JSON object")
    return msg
