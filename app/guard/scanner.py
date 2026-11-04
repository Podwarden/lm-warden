"""The scan: keyword pre-filter → the rule's regex in bounded windows → validator / entropy →
exclusions → leftmost-longest, non-overlapping. Pure; values stay in memory."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.guard.rules import Rule
from app.guard.validators import SHAPES, VALIDATORS, entropy

_PLACEHOLDER = re.compile(r"(?i)LMW-SECRET-[0-9a-f]{10}")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HEX_DIGEST = re.compile(r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|[0-9a-fA-F]{128})$")
_DOTTED_IDENT = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
#: Letters, "_" and "-" only: a name or an identifier, not a credential (real ones mix in digits).
_WORDY = re.compile(r"^[A-Za-z_\-]+$")
_GENERIC_REFS = (
    "$",
    "%",
    "process.env",
    "os.environ",
    "getenv",
    "sha512-",
    "sha256-",
    "sha384-",
    "sha1-",
)


@dataclass(frozen=True)
class Finding:
    start: int
    end: int
    rule_id: str
    value: str


def _excluded(value: str, rule: Rule) -> bool:
    if _PLACEHOLDER.search(value):
        return True
    if len(set(value)) <= 2:
        return True
    if rule.generic:
        if value.startswith(_GENERIC_REFS) or _UUID.match(value) or _HEX_DIGEST.match(value):
            return True
        if _DOTTED_IDENT.match(value) or _ENV_NAME.match(value):
            return True
        if "?." in value or "=>" in value or value.startswith(("this.", "self.", "props.")):
            return True
        if _WORDY.match(value) and not rule.allow_words:
            return True
        if value.lower() in (
            "password",
            "changeme",
            "true",
            "false",
            "null",
            "none",
            "undefined",
            "pass",
            "secret",
            "xxxx",
            "xxxxxxxx",
            "redacted",
            "example",
        ):
            return True
    # Vendors' published example keys (AKIAIOSFODNN7EXAMPLE) and docs placeholders.
    if "EXAMPLE" in value:
        return True
    return False


#: Custom (admin-written) patterns run over chunks of at most this many characters, overlapping so a
#: match up to the overlap is never cut at a chunk edge. A pattern that is quadratic in its input
#: (any `[a-z]+\\.`; worse ones are refused at save time) then costs chunk^2 per chunk, which stays
#: around a second per megabyte of the worst input. Built-in rules are linear and unaffected.
_CUSTOM_CHUNK = 1024
_CUSTOM_OVERLAP = 256


def _chunked(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in spans:
        while b - a > _CUSTOM_CHUNK:
            out.append((a, a + _CUSTOM_CHUNK))
            a += _CUSTOM_CHUNK - _CUSTOM_OVERLAP
        out.append((a, b))
    return out


def _windows(text: str, low: str, rule: Rule) -> list[tuple[int, int]]:
    spans = _keyword_windows(text, low, rule)
    return _chunked(spans) if rule.pack == "custom" else spans


def _keyword_windows(text: str, low: str, rule: Rule) -> list[tuple[int, int]]:
    if not rule.keywords:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    n = len(text)
    for kw in rule.keywords:
        i = low.find(kw)
        while i >= 0:
            spans.append((max(0, i - rule.back), min(n, i + rule.window)))
            i = low.find(kw, i + len(kw))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


#: Base64 runs worth decoding: long enough to hold a key, delimited, standard alphabet.
_B64_RUN = re.compile(r"(?<![A-Za-z0-9+/_\-])[A-Za-z0-9+/]{24,65536}={0,2}(?![A-Za-z0-9+/=])")
_B64_MAX_RUNS = 256


def _decoded(run: str) -> str | None:
    """The run as text when it is base64 of mostly printable UTF-8 (a kubeconfig key, a .env)."""
    try:
        raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=True)
        txt = raw.decode("utf-8")
    except (binascii.Error, ValueError):
        return None
    if len(txt) < 16:
        return None
    printable = sum(1 for c in txt if c.isprintable() or c in "\n\r\t")
    return txt if printable >= 0.95 * len(txt) else None


def _encoded_findings(
    text: str, rules: Sequence[Rule], taken: list[tuple[int, int]]
) -> list[Finding]:
    """Secrets inside base64 runs: the whole run is the value (it is what the model would see)."""
    out: list[Finding] = []
    for k, m in enumerate(_B64_RUN.finditer(text)):
        if k >= _B64_MAX_RUNS:
            break
        a, b = m.span()
        if any(a < y and b > x for x, y in taken):
            continue
        inner = _decoded(m.group(0))
        if inner is None:
            continue
        found = scan_text(inner, rules, _depth=1)
        if found:
            out.append(Finding(a, b, found[0].rule_id, m.group(0)))
    return out


def scan_text(text: str, rules: Sequence[Rule], *, _depth: int = 0) -> list[Finding]:
    if not text:
        return []
    low = text.lower()
    cands: list[tuple[int, int, int, Finding]] = []
    for order, rule in enumerate(rules):
        for a, b in _windows(text, low, rule):
            for m in rule.regex.finditer(text, a, b):
                s, e = m.span(rule.group)
                if s < 0 or e <= s:
                    continue
                value = text[s:e]
                if rule.generic and text[e : e + 1] == "(":
                    continue  # a function call, not a value
                if rule.key_group is not None:
                    key = (m.group(rule.key_group) or "").lower()
                    if any(x in key for x in rule.exclude_keys):
                        continue
                    sep = (m.group(rule.sep_group) or "=") if rule.sep_group else "="
                    if rule.shape and not SHAPES[rule.shape](
                        m.group(rule.key_group) or "", value, sep
                    ):
                        continue
                if rule.validator and not VALIDATORS[rule.validator](value):
                    continue
                if rule.entropy_min is not None and entropy(value) < rule.entropy_min:
                    continue
                if _excluded(value, rule):
                    continue
                cands.append((s, -(e - s), order, Finding(s, e, rule.id, value)))
    if _depth == 0:
        plain = [(f.start, f.end) for _s, _n, _o, f in cands]
        for f in _encoded_findings(text, rules, plain):
            cands.append((f.start, -(f.end - f.start), len(rules), f))
    # Leftmost-longest, non-overlapping -- but a specific rule (a GitHub token) is placed before
    # any generic one (a token= assignment), so a generic match never swallows it and the text
    # around it (a URL's host and path) stays readable.
    generic = {r.id for r in rules if r.generic}
    cands.sort()
    taken: list[tuple[int, int]] = []
    out: list[Finding] = []
    for want_generic in (False, True):
        for _s, _neg, _o, f in cands:
            if (f.rule_id in generic) != want_generic:
                continue
            if all(f.end <= a or f.start >= b for a, b in taken):
                out.append(f)
                taken.append((f.start, f.end))
    out.sort(key=lambda f: f.start)
    return out
