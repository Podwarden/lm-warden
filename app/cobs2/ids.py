"""Content ids (cobs2 spec section 6.3).

``H(key, label, salt, payload) = epoch byte + HMAC-SHA-256(key, ...)[:16]``,
rendered as 34 hex chars. Message ids depend only on the message, so a reply
hashed alone matches the same message inside a later request. Node ids chain
over the body messages, seeded by the head id.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.cobs2.canonical import Canon, canon_json, loose_head, loose_message
from app.cobs2.keys import NetworkKey


@dataclass(frozen=True)
class Ids:
    head: str
    msgs: tuple[str, ...]
    nodes: tuple[str, ...]
    loose_head: str
    loose_msgs: tuple[str, ...]
    loose_nodes: tuple[str, ...]


def scope_salt(body: dict[str, Any]) -> bytes:
    s = body.get("cache_salt")
    if isinstance(s, str) and s:
        return hashlib.sha256(s.encode("utf-8", "surrogatepass")).digest()
    return b""


def _hasher(key: NetworkKey, label: bytes, salt: bytes) -> Callable[[bytes], bytes]:
    """``payload -> H(key, label, salt, payload)``, with the constant prefix built once."""
    pre = b"cobs2|" + label + b"|" + bytes([len(salt)]) + salt + b"|"
    epoch = bytes([key.epoch])
    k = key.key
    digest = hmac.digest  # the one-shot C path: no HMAC object per call

    def h(payload: bytes) -> bytes:
        return epoch + digest(k, pre + payload, "sha256")[:16]

    return h


def _h(key: NetworkKey, label: bytes, salt: bytes, payload: bytes) -> bytes:
    return _hasher(key, label, salt)(payload)


#: exact head / msg / node id bytes -> the matching loose id bytes. A loose id
#: is a pure function of what the exact id was keyed over (an exact node fixes
#: its whole prefix), so a hit is exact, and the loose transform -- the dearest
#: step -- runs only for what this process has not hashed yet (in a
#: conversation: the new tail). Ids only, never text.
_LOOSE_MEMO: OrderedDict[bytes, bytes] = OrderedDict()
#: About 200 bytes an entry (13 MB full); a 250-message conversation uses ~500.
_LOOSE_MEMO_MAX = 65_536


#: ids_for runs on the event loop and, for a big body, in a worker thread
#: (routes._cobs2_compute): every memo read-modify-write holds this lock.
_LOOSE_MEMO_LOCK = threading.Lock()


def _memo_get(exact: bytes) -> bytes | None:
    with _LOOSE_MEMO_LOCK:
        v = _LOOSE_MEMO.get(exact)
        if v is not None:
            _LOOSE_MEMO.move_to_end(exact)
        return v


def _memo_put(exact: bytes, lid: bytes) -> None:
    with _LOOSE_MEMO_LOCK:
        _LOOSE_MEMO[exact] = lid
        while len(_LOOSE_MEMO) > _LOOSE_MEMO_MAX:
            _LOOSE_MEMO.popitem(last=False)


def ids_for(canon: Canon, key: NetworkKey, salt: bytes) -> Ids:
    h_msg = _hasher(key, b"msg", salt)
    h_node = _hasher(key, b"node", salt)
    head = _h(key, b"head", salt, canon_json({"head": canon.head, "tools": canon.tools}))
    msgs = [h_msg(canon_json(m)) for m in canon.messages]
    lh = _memo_get(head)
    if lh is None:
        l_head, l_tools = loose_head(canon)
        lh = _h(key, b"loose-head", salt, canon_json({"head": l_head, "tools": l_tools}))
        _memo_put(head, lh)
    h_lmsg = _hasher(key, b"loose-msg", salt)
    h_lnode = _hasher(key, b"loose-node", salt)
    nodes: list[bytes] = []
    lms: list[bytes] = []
    lnodes: list[bytes] = []
    prev, lprev = head, lh
    for m, mid in zip(canon.messages, msgs, strict=True):
        prev = h_node(prev + mid)
        nodes.append(prev)
        lm = _memo_get(mid)
        if lm is None:
            lm = h_lmsg(canon_json(loose_message(m)))
            _memo_put(mid, lm)
        lms.append(lm)
        # An exact node fixes its whole prefix, so it fixes the loose node too:
        # only the new tail is chained.
        ln = _memo_get(prev)
        if ln is None:
            ln = h_lnode(lprev + lm)
            _memo_put(prev, ln)
        lnodes.append(ln)
        lprev = ln
    return Ids(
        head.hex(),
        tuple(m.hex() for m in msgs),
        tuple(n.hex() for n in nodes),
        lh.hex(),
        tuple(m.hex() for m in lms),
        tuple(n.hex() for n in lnodes),
    )


def ids_for_epochs(canon: Canon, keys: Sequence[NetworkKey], salt: bytes) -> list[Ids]:
    return [ids_for(canon, k, salt) for k in keys]


def conv_id(ids: Ids, key: NetworkKey, salt: bytes) -> str | None:
    """A conversation's id: the loose head and the first loose body node, so it
    holds across turns, reminder and date changes. None without body messages."""
    if not ids.loose_nodes:
        return None
    payload = bytes.fromhex(ids.loose_head) + bytes.fromhex(ids.loose_nodes[0])
    return _h(key, b"conv", salt, payload).hex()
