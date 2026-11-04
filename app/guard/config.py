"""A key's guard settings: parse, validate and compile (cached per distinct JSON)."""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from app.guard.rules import BUILTIN, DEFAULT_PACKS, PACKS, Rule

MODES = ("sanitize", "block")
RESTORE = ("all", "text_only", "off")
TRAILER = ("new_only", "always", "off")
FAIL = ("closed", "open")
MAX_CUSTOM_REGEX = 512
_BACKREF = re.compile(r"\\[1-9]|\(\?P=")
_LOOK = re.compile(r"\(\?(?:=|!|<=|<!)")


@dataclass(frozen=True)
class Guard:
    mode: str
    packs: tuple[str, ...]
    rules: tuple[Rule, ...]
    restore: str
    trailer: str
    fail: str
    version: str
    overrides: tuple[tuple[str, str], ...] = ()

    def action_for(self, rule: Rule) -> str:
        o = dict(self.overrides)
        return (
            o.get(rule.id) or o.get(rule.pack) or ("block" if self.mode == "block" else "sanitize")
        )


def lint_regex(pattern: str) -> None:
    """Reject patterns Python's backtracking engine could take exponential time on."""
    if len(pattern) > MAX_CUSTOM_REGEX:
        raise ValueError(f"pattern longer than {MAX_CUSTOM_REGEX} characters")
    if _BACKREF.search(pattern):
        raise ValueError("back-references can take exponential time")
    if _LOOK.search(pattern):
        raise ValueError("look-arounds are not allowed")
    try:
        re.compile(pattern)
        tree = _sre_parse(pattern)
    except re.error as e:
        raise ValueError(f"not a valid pattern: {e}") from None
    if _risky(tree, inside_repeat=False):
        raise ValueError(
            "a repeated group that holds another repeat or alternatives can take exponential "
            "time; pin each pass with a literal separator (e.g. (?:[a-z]+\\.)+) or write a flat "
            "pattern"
        )
    _probe(pattern, tree)


def _sre_parse(pattern: str) -> Any:
    try:
        from re import _parser as sre_parse  # type: ignore[attr-defined]  # Python 3.11+
    except ImportError:  # pragma: no cover
        import sre_parse
    return sre_parse.parse(pattern)


_REPEATS = ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")


def _required_literal(tree: Any) -> str:
    """The longest run of literal characters at the top level of a pattern: text every match must
    contain (`internal` in `(?:[a-z]+\\.)+internal`). Case-insensitive patterns give nothing."""
    best, run = "", ""
    for op, av in tree:
        if str(op) == "LITERAL":
            run += chr(av)
            continue
        best = max(best, run, key=len)
        run = ""
    best = max(best, run, key=len)
    return best


def _many(av: Any) -> bool:
    _lo, hi, _sub = av
    return bool(hi is None or hi > 1)


def _set_matches(op: str, av: Any, ch: str) -> bool:
    """Could this one-character node match `ch`? (Conservative: unknown means yes.)"""
    if op == "LITERAL":
        return bool(av == ord(ch))
    if op == "NOT_LITERAL":
        return bool(av != ord(ch))
    if op == "ANY":
        return ch != "\n"
    if op == "IN":
        hit, neg = False, False
        for iop, iav in av:
            n = str(iop)
            if n == "NEGATE":
                neg = True
            elif n == "LITERAL" and iav == ord(ch):
                hit = True
            elif n == "RANGE" and iav[0] <= ord(ch) <= iav[1]:
                hit = True
            elif n == "CATEGORY":
                hit = hit or _category(str(iav), ch)
        return hit != neg
    if op == "CATEGORY":
        return _category(str(av), ch)
    return True


def _category(cat: str, ch: str) -> bool:
    table = {
        "CATEGORY_DIGIT": ch.isdigit(),
        "CATEGORY_NOT_DIGIT": not ch.isdigit(),
        "CATEGORY_SPACE": ch.isspace(),
        "CATEGORY_NOT_SPACE": not ch.isspace(),
        "CATEGORY_WORD": ch.isalnum() or ch == "_",
        "CATEGORY_NOT_WORD": not (ch.isalnum() or ch == "_"),
    }
    return table.get(cat, True)


def _repeated_nodes(items: Any) -> list[tuple[str, Any]]:
    """The one-character nodes inside every repeat within `items` (what the repeats can eat)."""
    out: list[tuple[str, Any]] = []
    for op, av in items:
        name = str(op)
        if name in _REPEATS:
            for sop, sav in av[2]:
                if str(sop) in ("LITERAL", "NOT_LITERAL", "ANY", "IN", "CATEGORY"):
                    out.append((str(sop), sav))
            out += _repeated_nodes(av[2])
        elif name == "SUBPATTERN":
            out += _repeated_nodes(av[-1])
        elif name == "BRANCH":
            for b in av[1]:
                out += _repeated_nodes(b)
    return out


def _separated(items: Any) -> bool:
    """A repeated group whose first or last element is a literal no inner repeat can eat
    (`(?:[a-z]+\\.)+`, `(?:-[a-z0-9]+)*`): each pass is pinned to that character, so the
    nesting cannot backtrack exponentially."""
    seq = list(items)
    if len(seq) == 1 and str(seq[0][0]) == "SUBPATTERN":
        seq = list(seq[0][1][-1])
    inner = _repeated_nodes(seq)
    for op, av in (seq[0], seq[-1]) if seq else ():
        if str(op) == "LITERAL":
            ch = chr(av)
            if not any(_set_matches(o, a, ch) for o, a in inner):
                return True
    return False


def _risky(items: Any, *, inside_repeat: bool) -> bool:
    """True when a repeat (more than once) contains another repeat or an alternation that is not
    pinned by a separator: the shapes that make a backtracking engine take exponential time
    ((a+)+, (a|aa)+, (a{1,30}){1,30})."""
    for op, av in items:
        name = str(op)
        if name in _REPEATS:
            many = _many(av)
            if many and inside_repeat:
                return True
            sub = av[2]
            if many and _separated(sub):
                # Pinned by a separator: check the body as if it were not repeated.
                if _risky(sub, inside_repeat=False):
                    return True
            elif _risky(sub, inside_repeat=inside_repeat or many):
                return True
        elif name == "SUBPATTERN":
            if _risky(av[-1], inside_repeat=inside_repeat):
                return True
        elif name == "BRANCH":
            if inside_repeat:
                return True
            if any(_risky(b, inside_repeat=False) for b in av[1]):
                return True
        elif name in ("ATOMIC_GROUP",):
            if _risky(av, inside_repeat=inside_repeat):
                return True
    return False


#: The timed probe: adversarial inputs built from the pattern's own characters, run in a child
#: process with a deadline. Catches the polynomial blow-ups no structural rule describes
#: (\s*\s*\s*$, a*a*a*b). Only at save time; a request never waits on it.
_PROBE = """
import json, re, sys
d = json.load(sys.stdin)
rx = re.compile(d["p"])
for s in d["probes"]:
    rx.search(s)
"""
_PROBE_SECONDS = 3.0
#: Probe length: long enough that cubic and exponential patterns blow far past the deadline, short
#: enough that quadratic ones (any `[a-z]+\\.`) finish well inside it. Quadratic is accepted:
#: custom patterns are scanned in bounded chunks (scanner.py), so it stays bounded per megabyte.
_PROBE_LEN = 1500


def _probe_chars(tree: Any) -> list[str]:
    chars: list[str] = []
    for op, av in tree:
        name = str(op)
        if name == "LITERAL":
            chars.append(chr(av))
        elif name == "IN":
            for iop, iav in av:
                n = str(iop)
                if n == "LITERAL":
                    chars.append(chr(iav))
                elif n == "RANGE":
                    chars.append(chr(iav[0]))
                elif n == "CATEGORY":
                    chars += {
                        "CATEGORY_DIGIT": ["0"],
                        "CATEGORY_SPACE": [" "],
                        "CATEGORY_WORD": ["a"],
                    }.get(str(iav), [])
        elif name == "CATEGORY":
            chars += {"CATEGORY_DIGIT": ["0"], "CATEGORY_SPACE": [" "], "CATEGORY_WORD": ["a"]}.get(
                str(av), []
            )
        elif name in _REPEATS:
            chars += _probe_chars(av[2])
        elif name == "SUBPATTERN":
            chars += _probe_chars(av[-1])
        elif name == "BRANCH":
            for b in av[1]:
                chars += _probe_chars(b)
    return chars


def _probe(pattern: str, tree: Any) -> None:
    import subprocess
    import sys

    chars = list(dict.fromkeys(_probe_chars(tree) + [" ", "a", "0", "-", "."]))[:24]
    probes = [c * _PROBE_LEN + "\x00" for c in chars]
    probes += [(a + b) * (_PROBE_LEN // 2) + "\x00" for a in chars[:6] for b in chars[:6] if a != b]
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", _PROBE],
            input=json.dumps({"p": pattern, "probes": probes}),
            capture_output=True,
            text=True,
            timeout=_PROBE_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(
            "this pattern takes too long on some inputs (likely repeats that can match the same "
            "characters back to back, such as \\s*\\s*); write it with one repeat per character class"
        ) from None
    if r.returncode != 0:
        raise ValueError("this pattern could not be checked") from None


def _custom(r: dict[str, Any]) -> Rule:
    rid = str(r.get("id") or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{0,47}", rid):
        raise ValueError("rule id: 1-48 lowercase letters, digits or dashes")
    kind = r.get("type")
    if kind == "regex":
        pat = str(r.get("regex") or "")
        lint_regex(pat)
        kws = tuple(str(k).lower() for k in (r.get("keywords") or ()) if str(k))
        if not kws:
            # No keywords given: use the longest fixed text every match must contain, so the
            # pattern only runs near it (a megabyte without it costs one substring search).
            derived = _required_literal(_sre_parse(pat))
            kws = (derived.lower(),) if len(derived) >= 3 else ()
        return Rule(
            rid,
            "custom",
            str(r.get("label") or rid),
            re.compile(f"({pat})"),
            kws,
            back=2048,
            window=2048,
        )
    if kind == "fixed":
        vals = [str(v) for v in (r.get("values") or ()) if str(v).strip()]
        if not vals or len(vals) > 1000:
            raise ValueError("fixed words: 1 to 1000 values")
        alt = "|".join(re.escape(v) for v in sorted(vals, key=len, reverse=True))
        return Rule(
            rid,
            "custom",
            str(r.get("label") or rid),
            re.compile(f"(?i)({alt})"),
            tuple(v.lower() for v in vals),
            back=0,
            window=max(map(len, vals)) + 1,
        )
    raise ValueError("rule type must be regex or fixed")


def validate(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalised settings, or ValueError with a message for the admin."""
    if not isinstance(raw, dict):
        raise ValueError("guard must be an object")
    out: dict[str, Any] = {
        "mode": raw.get("mode", "sanitize"),
        "packs": list(raw.get("packs", list(DEFAULT_PACKS))),
        "rules": list(raw.get("rules", [])),
        "overrides": dict(raw.get("overrides", {})),
        "restore": raw.get("restore", "all"),
        "trailer": raw.get("trailer", "new_only"),
        "fail": raw.get("fail", "closed"),
    }
    for k, allowed in (("mode", MODES), ("restore", RESTORE), ("trailer", TRAILER), ("fail", FAIL)):
        if out[k] not in allowed:
            raise ValueError(f"{k} must be one of {list(allowed)}")
    for p in out["packs"]:
        if p not in PACKS:
            raise ValueError(f"unknown pack {p!r}")
    for k, v in out["overrides"].items():
        if v not in MODES:
            raise ValueError(f"override for {k!r} must be one of {list(MODES)}")
    if len(out["rules"]) > 100:
        raise ValueError("at most 100 custom rules")
    custom_ids = {_custom(r).id for r in out["rules"]}
    known = set(PACKS) | {r.id for r in BUILTIN} | custom_ids
    for k in out["overrides"]:
        if k not in known:
            raise ValueError(f"override for unknown pack or rule {k!r}")
    return out


@functools.lru_cache(maxsize=256)
def _compile_json(s: str) -> Guard:
    d = validate(json.loads(s))
    rules = tuple(r for r in BUILTIN if r.pack in d["packs"]) + tuple(
        _custom(r) for r in d["rules"]
    )
    return Guard(
        mode=d["mode"],
        packs=tuple(d["packs"]),
        rules=rules,
        restore=d["restore"],
        trailer=d["trailer"],
        fail=d["fail"],
        version=hashlib.sha256(s.encode()).hexdigest()[:12],
        overrides=tuple(sorted(d["overrides"].items())),
    )


def compile_guard(raw: dict[str, Any] | str | None) -> Guard | None:
    if raw is None or raw == "":
        return None
    s = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return _compile_json(s)
