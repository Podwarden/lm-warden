"""Wire protocol of the KV store daemon, version 1, minor 2.

One JSON object per line (UTF-8, ``\\n`` terminated) in each direction over a
Unix stream socket. A request is ``{"op": str, "id": int, ...}``; the reply is
``{"id": int, "ok": bool, ...}`` and, when ``ok`` is false, an ``error`` string
``"<code>: <message>"``. Byte fields (keys, ids) are lower-case hex. The
protocol is engine-neutral: it speaks of namespaces, block keys, groups and byte
slots, nothing else. ``PROTOCOL.md`` is the full reference.

Minor 1 adds optional fields (each with a stated default, so a minor 0 request means
what it always meant) and three ops; minor 2 adds more optional fields and the op
``top_reused``. The daemon ignores fields it does not know.
"""

from __future__ import annotations

import json
from typing import Any

VERSION = 1
MINOR = 2
MAX_LINE = 8 << 20  # bytes per request or reply line

# op -> the request fields it takes (besides "op" and "id"); every field added in
# minor 1 is optional (the comments give the default).
OPS: dict[str, tuple[str, ...]] = {
    "hello": (),
    # nvme: true; owner: ""; model: "" (16 bytes); role: "attention"; rank: 0;
    # staging (minor 2): false
    "register_ns": ("namespace", "groups", "nvme", "owner", "model", "role", "rank", "staging"),
    "lookup": ("ns", "keys"),
    # tenant, scope_tag: "" (16 bytes); pos: the key (32 bytes); depth (minor 2): unknown
    "begin_save": ("ns", "key", "parent", "scope", "tenant", "scope_tag", "pos", "depth"),
    "commit_save": ("ns", "key"),
    "abort_save": ("ns", "key"),
    # tenant: ""; tag (minor 2): none, a load that never counts a reload
    "begin_load": ("ns", "keys", "tenant", "tag"),
    "end_load": ("handle", "used"),  # used (minor 2): unknown, no reload
    "record_hits": ("tag", "ram_tokens", "nvme_tokens", "load_s", "tenant"),  # tenant: ""
    "pop_hits": ("tag",),
    "stats": (),
    # minor 1
    "release_owner": ("owner",),
    "retire_scopes": ("scope_tags",),
    "set_policy": ("policy",),
    # minor 2: model, tenant (16 bytes, default any); view "admin" or "holder"; n 1 to 50
    "top_reused": ("model", "tenant", "view", "n"),
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
    "rows",
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
