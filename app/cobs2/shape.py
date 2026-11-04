"""A request's shape and toolset for the sessions pages: sizes, counts and keyed
tool hashes -- never text. ``Toolset.names`` holds plain tool names in memory
only, for the per-token ``session_tools`` table (owner decision 2026-10-25)."""

from __future__ import annotations

import functools
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.cobs2 import canonical as _canonical
from app.cobs2.canonical import Canon, canon_json
from app.cobs2.ids import Ids
from app.cobs2.lineage import Observation
from app.proxy.forest_fields import hkey, tool_family

SHAPE_MAX_BYTES = 256
DW_MEMO_MAX = 65_536
MAX_TOOLS = 128
_NAME_MAX = 128


def _parts_chars(parts: list[dict[str, Any]]) -> int:
    n = 0
    for p in parts:
        t = p.get("t")
        if t in ("text", "reasoning"):
            v = p.get("v")
            n += len(v) if isinstance(v, str) else 0
        elif t == "tool_call":
            a = p.get("args")
            n += len(a) if isinstance(a, str) else 0
        elif t == "tool_result":
            sub = p.get("parts")
            n += _parts_chars(sub) if isinstance(sub, list) else 0
    return n


def shape_of(canon: Canon) -> dict[str, Any]:
    """Sizes (chars) and counts of a canonical request."""
    sys_chars = sum(_parts_chars(m["parts"]) for m in canon.head)
    tools_chars = sum(len(t) for t in canon.tools)
    msgs = {"user": 0, "assistant": 0, "tool": 0}
    body = 0
    last_user: int | None = None
    for m in canon.messages:
        c = _parts_chars(m["parts"])
        body += c
        role = m["role"]
        if role in msgs:
            msgs[role] += 1
        if role == "user":
            last_user = c
    return {
        "chars": sys_chars + body + tools_chars,
        "sys": sys_chars,
        "tools": len(canon.tools),
        "tools_chars": tools_chars,
        "msgs": msgs,
        "last_user": last_user,
    }


def break_index(obs: Observation | None) -> int | None:
    """The first body message (0-based) whose exact node the session had not
    seen -- only on a fork or a history rewrite (R6); None otherwise."""
    if obs is None:
        return None
    if obs.edge == "fork" or obs.loose_match_beyond_exact:
        return obs.exact_depth + 1
    return None


def shape_json(shape: dict[str, Any], brk: int | None) -> str | None:
    """Compact JSON of the shape with ``brk``; over the cap, ``dw`` goes first."""
    out = canon_json({**shape, "brk": brk}).decode()
    if len(out.encode()) <= SHAPE_MAX_BYTES:
        return out
    if "dw" in shape:
        out = canon_json({**{k: v for k, v in shape.items() if k != "dw"}, "brk": brk}).decode()
        if len(out.encode()) <= SHAPE_MAX_BYTES:
            return out
    return None


#: message id hex -> (boilerplate chars, reasoning chars, message chars). Ids
#: and ints only, never text; a conversation's turn scans only its new tail.
_DW_MEMO: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
_DW_MEMO_LOCK = threading.Lock()


def _text_boil(parts: list[dict[str, Any]]) -> tuple[int, int]:
    boil = reason = 0
    for p in parts:
        t = p.get("t")
        v = p.get("v")
        if t == "text" and isinstance(v, str):
            boil += _canonical.boilerplate_chars(v)
        elif t == "reasoning" and isinstance(v, str):
            reason += len(v)
        elif t == "tool_result" and isinstance(p.get("parts"), list):
            b, r = _text_boil(p["parts"])
            boil += b
            reason += r
    return boil, reason


def _measure(key: str, parts: list[dict[str, Any]]) -> tuple[int, int, int]:
    with _DW_MEMO_LOCK:
        hit = _DW_MEMO.get(key)
        if hit is not None:
            _DW_MEMO.move_to_end(key)
            return hit
    boil, reason = _text_boil(parts)
    val = (boil, reason, _parts_chars(parts))
    with _DW_MEMO_LOCK:
        _DW_MEMO[key] = val
        while len(_DW_MEMO) > DW_MEMO_MAX:
            _DW_MEMO.popitem(last=False)
    return val


def dead_weight_of(canon: Canon, ids: Ids) -> dict[str, int]:
    """Dead weight of one prompt, in chars (R1): ``dup`` is every message whose
    exact id already appeared earlier in this prompt, whole; ``boil`` (harness
    boilerplate) and ``reason`` (carried reasoning) count in the head and in
    first occurrences only, so nothing is counted twice."""
    head_parts = [p for m in canon.head for p in m["parts"]]
    boil, reason, _ = _measure("h" + ids.head, head_parts)
    dup = 0
    seen: set[str] = set()
    for m, mid in zip(canon.messages, ids.msgs, strict=False):
        b, r, chars = _measure(mid, m["parts"])
        if mid in seen:
            dup += chars
            continue
        seen.add(mid)
        boil += b
        reason += r
    return {"dup": dup, "boil": boil, "reason": reason}


@dataclass(frozen=True)
class Toolset:
    digest: str
    #: (hash8, family, chars) per offered tool, in offer order.
    entries: tuple[tuple[str, str, int], ...]
    #: Plain names, parallel to ``entries``; memory only (``session_tools``).
    names: tuple[str, ...]
    truncated: bool


@functools.lru_cache(maxsize=4096)
def _hash8(secret: str, name: str) -> str:
    return hkey(secret, name, 8) or "0" * 8


def _name(t: Any) -> str:
    if isinstance(t, dict):
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return str(fn["name"])
        if isinstance(t.get("name"), str):
            return str(t["name"])
    return ""


def toolset_of(tools: Any, canon_tools: list[Any], secret: str) -> Toolset | None:
    """The offered tools as keyed hashes, families and definition sizes, capped
    at ``MAX_TOOLS``; None when no tools were offered."""
    if not isinstance(tools, list) or not tools:
        return None
    entries: list[tuple[str, str, int]] = []
    names: list[str] = []
    for i, t in enumerate(tools[:MAX_TOOLS]):
        name = _name(t)
        chars = len(canon_tools[i]) if i < len(canon_tools) else 0
        entries.append((_hash8(secret, name), tool_family(name), chars))
        names.append(name[:_NAME_MAX])
    digest = hashlib.sha256(canon_json([list(e) for e in entries])).hexdigest()[:32]
    return Toolset(digest, tuple(entries), tuple(names), len(tools) > MAX_TOOLS)
