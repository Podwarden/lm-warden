"""Where the scannable text is in a request (OpenAI chat and Anthropic Messages), the inspection
(pure: mutates nothing), and applying the placeholders (copy-on-write). Per-message memo keyed by
the message's strings' (len, hash), verified by the value's hash before use."""

from __future__ import annotations

import copy
import re
import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.guard.config import Guard
from app.guard.placeholders import RestoreMap, placeholder_hex, render
from app.guard.scanner import scan_text

MEMO_MAX = 65_536
NOTE = (
    "[LM Warden] Some values in this conversation were replaced by placeholders of the form "
    "⟦LMW-SECRET-…⟧ before you saw them. Treat each placeholder as an opaque value: copy it exactly "
    "where the original would go; do not guess, decode or ask for it."
)

Path = tuple[Any, ...]


class Unscannable(Exception):
    """A part of the request the guard must check but cannot read (e.g. token ids)."""


@dataclass(frozen=True)
class Where:
    role: str
    #: 1-based message index; 0 = the head (system prompt outside messages, tool definitions).
    index: int
    kind: str


@dataclass(frozen=True)
class Hit:
    path: Path
    start: int
    end: int
    rule_id: str
    value: str
    hex: str
    where: Where


@dataclass
class Inspection:
    findings: list[Hit] = field(default_factory=list)
    restore: RestoreMap = field(default_factory=RestoreMap)


_MEMO: OrderedDict[tuple[Any, ...], tuple[tuple[int, int, int, str, int], ...]] = OrderedDict()
_LOCK = threading.Lock()


def _strings(obj: Any, path: Path) -> Iterator[tuple[Path, str]]:
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, (*path, i))


def _content(c: Any, path: Path) -> Iterator[tuple[Path, str]]:
    if isinstance(c, str):
        yield path, c
    elif isinstance(c, list):
        for i, p in enumerate(c):
            if (
                isinstance(p, dict)
                and p.get("type") in ("text", "input_text", "output_text")
                and isinstance(p.get("text"), str)
            ):
                yield (*path, i, "text"), p["text"]


def _openai(body: dict[str, Any]) -> Iterator[tuple[Where, list[tuple[Path, str]]]]:
    for ti, t in enumerate(body.get("tools") or []):
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict):
            strs = []
            if isinstance(fn.get("description"), str):
                strs.append((("tools", ti, "function", "description"), fn["description"]))
            strs += list(_strings(fn.get("parameters"), ("tools", ti, "function", "parameters")))
            yield Where("tools", 0, "tool definition"), strs
    for i, m in enumerate(body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role"))
        base = ("messages", i)
        kind = {
            "system": "system",
            "developer": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, role)
        strs = list(_content(m.get("content"), (*base, "content")))
        if strs:
            yield Where(role, i + 1, kind), strs
        for k in ("reasoning_content", "reasoning"):
            if isinstance(m.get(k), str):
                yield Where(role, i + 1, "reasoning"), [((*base, k), m[k])]
        for j, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                yield (
                    Where(role, i + 1, "tool call"),
                    [((*base, "tool_calls", j, "function", "arguments"), fn["arguments"])],
                )
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        yield Where("user", 1, "user"), [(("prompt",), prompt)]
    elif isinstance(prompt, list):
        # Legacy completions take a list of prompts, or token ids the guard cannot read.
        if not all(isinstance(p, str) for p in prompt):
            raise Unscannable("prompt holds token ids")
        yield Where("user", 1, "user"), [(("prompt", i), p) for i, p in enumerate(prompt)]
    elif prompt is not None:
        raise Unscannable("prompt is neither text nor a list of texts")


def _anthropic(body: dict[str, Any]) -> Iterator[tuple[Where, list[tuple[Path, str]]]]:
    sys_ = body.get("system")
    strs = list(_content(sys_, ("system",)))
    if strs:
        yield Where("system", 0, "system"), strs
    for ti, t in enumerate(body.get("tools") or []):
        if isinstance(t, dict):
            s = []
            if isinstance(t.get("description"), str):
                s.append((("tools", ti, "description"), t["description"]))
            s += list(_strings(t.get("input_schema"), ("tools", ti, "input_schema")))
            yield Where("tools", 0, "tool definition"), s
    for i, m in enumerate(body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role"))
        base = ("messages", i, "content")
        c = m.get("content")
        if isinstance(c, str):
            yield Where(role, i + 1, role), [(base, c)]
            continue
        for j, b in enumerate(c or []):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                yield Where(role, i + 1, role), [((*base, j, "text"), b["text"])]
            elif t == "tool_use":
                yield (
                    Where(role, i + 1, "tool call"),
                    list(_strings(b.get("input"), (*base, j, "input"))),
                )
            elif t == "tool_result":
                rc = b.get("content")
                yield Where(role, i + 1, "tool result"), list(_content(rc, (*base, j, "content")))
            elif t == "thinking" and isinstance(b.get("thinking"), str):
                # Signed by Anthropic: read, never rewritten (see run_guard).
                yield Where(role, i + 1, "thinking"), [((*base, j, "thinking"), b["thinking"])]


def _memo_get(key: tuple[Any, ...]) -> tuple[tuple[int, int, int, str, int], ...] | None:
    with _LOCK:
        v = _MEMO.get(key)
        if v is not None:
            _MEMO.move_to_end(key)
        return v


def _memo_put(key: tuple[Any, ...], v: tuple[tuple[int, int, int, str, int], ...]) -> None:
    with _LOCK:
        _MEMO[key] = v
        while len(_MEMO) > MEMO_MAX:
            _MEMO.popitem(last=False)


#: Where a value can end inside text: whitespace and the delimiters of configs, URLs, shells and
#: code. A value that itself contains one of these is not caught by literal matching.
#: A whitespace-free run of text, and the characters a value can sit next to inside one
#: (`app/<pw>@ORCL`, `AccountKey=<key>;`). A value that itself holds one of them still matches:
#: candidates are cut at every delimiter edge, not split on them.
_RUN = re.compile(r"\S+")
_DELIMS = frozenset("\"'`=:,;&()[]{}<>|@/")
_EDGE_TRIM = "\"'`.,!?;:)(]["


_LIT_MEMO: OrderedDict[tuple[Any, ...], tuple[tuple[int, int, str], ...]] = OrderedDict()


def _literal_spans(
    s: str, known: dict[str, tuple[Any, ...]], secret: bytes, token_id: str
) -> list[tuple[int, int, str]]:
    """Occurrences of values this session already found, matched by keyed hash, never by value.
    Memoised per (string, known set): a session's unchanged history is not re-hashed each turn."""
    key = (len(s), hash(s), token_id, frozenset(known))
    with _LOCK:
        cached = _LIT_MEMO.get(key)
        if cached is not None:
            _LIT_MEMO.move_to_end(key)
    if cached is not None:
        return list(cached)
    out = _literal_scan(s, known, secret, token_id)
    with _LOCK:
        _LIT_MEMO[key] = tuple(out)
        while len(_LIT_MEMO) > MEMO_MAX:
            _LIT_MEMO.popitem(last=False)
    return out


_DELIM_RE = re.compile(r"[\"'`=:,;&()\[\]{}<>|@/]|[.,!?](?=\s|$)")
_WS = re.compile(r"\s")


def _literal_scan(
    s: str, known: dict[str, tuple[Any, ...]], secret: bytes, token_id: str
) -> list[tuple[int, int, str]]:
    lengths = {v[0] for v in known.values()}
    # In-process prefilter: Python's (per-process, keyed) hash of each known value, so the keyed
    # HMAC runs only on real candidates. Integers in memory, like the rest of the session state.
    quick = {v[3] for v in known.values() if len(v) > 3}
    word_counts = {v[2] for v in known.values() if len(v) > 2 and v[2] > 1}
    out: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []

    def check(a: int, b: int) -> None:
        if b - a not in lengths:
            return
        sub = s[a:b]
        if quick and hash(sub) not in quick:
            return
        if any(a < y and b > x for x, y in taken):
            return
        hit = known.get(placeholder_hex(secret, token_id, sub))
        if hit is not None:
            out.append((a, b, str(hit[1])))
            taken.append((a, b))

    runs = list(_RUN.finditer(s))
    # Multi-word values (passphrases) first: runs of consecutive words, quotes and punctuation trimmed.
    for n in sorted(word_counts, reverse=True):
        for i in range(len(runs) - n + 1):
            a, b = runs[i].start(), runs[i + n - 1].end()
            while a < b and s[a] in _EDGE_TRIM:
                a += 1
            while b > a and s[b - 1] in _EDGE_TRIM:
                b -= 1
            check(a, b)
    # Single-word values: spans between two edges (run ends and delimiter edges), exactly one known
    # length apart and holding no whitespace. Linear in the edges, so a megabyte of minified code on
    # one line is fine.
    edges: set[int] = set()
    for m in runs:
        edges.add(m.start())
        edges.add(m.end())
    for m in _DELIM_RE.finditer(s):
        edges.add(m.start())
        edges.add(m.end())
    for e in edges:
        for n in lengths:
            f = e + n
            if f in edges and not _WS.search(s, e, f):
                check(e, f)
    out.sort()
    return out


def inspect_body(
    body: dict[str, Any],
    dialect: str,
    guard: Guard,
    *,
    token_id: str,
    secret: bytes,
    known: dict[str, tuple[Any, ...]] | None = None,
) -> Inspection:
    out = Inspection()
    groups = _anthropic(body) if dialect == "anthropic" else _openai(body)
    for where, strs in groups:
        if not strs:
            continue
        key = (guard.version, where.kind, tuple((len(s), hash(s)) for _p, s in strs))
        cached = _memo_get(key)
        spans: list[tuple[int, int, int, str]] | None = None
        if cached is not None:
            ok = all(hash(strs[si][1][a:b]) == vh for si, a, b, _r, vh in cached)
            spans = [(si, a, b, r) for si, a, b, r, _vh in cached] if ok else None
        if spans is None:
            spans = []
            for si, (_p, s) in enumerate(strs):
                spans += [(si, f.start, f.end, f.rule_id) for f in scan_text(s, guard.rules)]
            _memo_put(key, tuple((si, a, b, r, hash(strs[si][1][a:b])) for si, a, b, r in spans))
        if known:
            # Values already found in this session, wherever they turn up without their context.
            for si, (_p, s) in enumerate(strs):
                taken = [(a, b) for sj, a, b, _r in spans if sj == si]
                for a, b, r in _literal_spans(s, known, secret, token_id):
                    if all(b <= x or a >= y for x, y in taken):
                        spans.append((si, a, b, r))
                        taken.append((a, b))
        for si, a, b, r in spans:
            path, s = strs[si]
            value = s[a:b]
            hx = placeholder_hex(secret, token_id, value)
            out.findings.append(Hit(path, a, b, r, value, hx, where))
            out.restore.add(hx, value)
    return out


def _get(obj: Any, path: Path) -> Any:
    for k in path:
        obj = obj[k]
    return obj


def _set(obj: Any, path: Path, value: Any) -> None:
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value


def _append_note(body: dict[str, Any], dialect: str) -> None:
    msgs = body.get("messages") or []
    if not msgs or not isinstance(msgs[-1], dict) or msgs[-1].get("role") not in ("user", "tool"):
        return
    last = msgs[-1]
    c = last.get("content")
    if isinstance(c, str):
        last["content"] = f"{c}\n\n{NOTE}"
    elif isinstance(c, list):
        c.append({"type": "text", "text": NOTE})


def apply_findings(body: dict[str, Any], ins: Inspection, *, note: bool) -> dict[str, Any]:
    """A copy of the body with every finding replaced by its placeholder (and the note on the last
    turn); the body itself when there is nothing to replace."""
    if not ins.findings:
        return body
    new = copy.deepcopy(body)
    by_path: dict[Path, list[Hit]] = {}
    for h in ins.findings:
        by_path.setdefault(h.path, []).append(h)
    for path, hits in by_path.items():
        s = _get(new, path)
        for h in sorted(hits, key=lambda x: x.start, reverse=True):
            s = s[: h.start] + render(h.hex) + s[h.end :]
        _set(new, path, s)
    if note:
        _append_note(new, "openai")
    return new
