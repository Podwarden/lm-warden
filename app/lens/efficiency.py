"""Prompt Lens efficiency rules E-01..E-09 (analysis spec 11.1).

Pure functions over a session's ``LensTurn`` list, read from the layer 1
fields stored with each request (``lens_cause``, ``lens_lost_tokens``, ...).
Every rule fires only above its threshold and when its false-positive guard
does not hold (A-12); server and provider-minimum misses are E-08, shown but
never scored (A-11). Findings carry codes and numbers only (A-10).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime

from app.lens.catalogue import RULES, Confidence, Finding, Saving
from app.lens.hot import provider_min
from app.lens.probes import SPAN_CLASSES, VOLATILE_CLASSES
from app.lens.turns import LensTurn
from app.sessions.pauses import PAUSE_S

#: E-01, E-02: the least a head change must lose to be worth a finding.
MIN_LOST = 1_024
#: E-06: no-marks turns needed before advising breakpoints.
MIN_NO_MARKS_TURNS = 3

_SYS = frozenset({"sys_noise", "sys_changed"})
_TOOLS_ORDER = frozenset({"tools_reordered", "tools_key_order"})
_HISTORY = frozenset({"history_rewritten", "history_noise"})
_REASONING = frozenset({"reasoning_dropped", "reasoning_kept"})
#: Classes E-01 accepts on a ``sys_changed`` line: the volatile ones, plus an
#: unclassified line (low confidence). A changed number, URL or name reads as
#: an edit of the content, not as a value that moves every turn.
_E01_CLASSES = VOLATILE_CLASSES | {"other", None}


def _lost(t: LensTurn) -> int:
    return max(t.lost_tokens or 0, 0)


def _sum_lost(ts: Iterable[LensTurn]) -> int:
    return sum(_lost(t) for t in ts)


def _node(brk: str | None) -> int | None:
    if brk is None or not brk.startswith("node:"):
        return None
    try:
        return int(brk[5:])
    except ValueError:
        return None


def _most_common(vals: Iterable[object]) -> object:
    c = Counter(v for v in vals if v is not None)
    return c.most_common(1)[0][0] if c else None


class _Ctx:
    def __init__(self, turns: Sequence[LensTurn], prefill_rate: float | None) -> None:
        self.turns = turns
        self.rate = prefill_rate if prefill_rate and prefill_rate > 0 else None

    def saving(self, tokens: int) -> Saving:
        return Saving(tokens, (tokens / self.rate) if self.rate else None, None, None)

    def finding(
        self,
        rule: str,
        ts: Sequence[LensTurn],
        counts: dict[str, int],
        *,
        cause: str | None,
        confidence: Confidence,
        tokens: int | None = None,
    ) -> Finding:
        assert rule in RULES
        return Finding(
            rule=rule,
            turns=tuple(t.n for t in ts),
            counts=counts,
            tool=None,
            cause=cause,
            saving=self.saving(_sum_lost(ts) if tokens is None else tokens),
            confidence=confidence,
        )


def _of(turns: Sequence[LensTurn], causes: frozenset[str]) -> list[LensTurn]:
    # A fork is the user's own branch (an edit or a retry): never a fault.
    return [t for t in turns if t.cause in causes and t.edge != "fork"]


# -- E-01 ----------------------------------------------------------------------


def _utc_day(ts: float) -> int:
    return datetime.fromtimestamp(ts, tz=UTC).toordinal()


def _once_a_day(turns: Sequence[LensTurn], hits: Sequence[LensTurn]) -> bool:
    """Every change of the date line came with a new UTC day since the lane's
    previous turn: a date that changes once a day is expected (v1 9.3)."""
    prev_start: dict[int, float] = {}
    last: dict[object, float] = {}
    for t in turns:
        key = t.lane if t.lane is not None else ("solo", t.n)
        if key in last:
            prev_start[t.n] = last[key]
        last[key] = t.started_at
    for h in hits:
        p = prev_start.get(h.n)
        if p is None or _utc_day(p) == _utc_day(h.started_at):
            return False
    return True


#: The key's own earlier sessions with the same E-01 line: one count for
#: every line, or per span line (``service`` counts them per line; a line it
#: has no count for, or no line, is 0).
CrossSessions = int | Mapping[int, int]


def _cross(cross_sessions: CrossSessions, line: int | None) -> int:
    if isinstance(cross_sessions, Mapping):
        n = cross_sessions.get(line, 0) if line is not None else 0
    else:
        n = cross_sessions
    return max(n, 0) if isinstance(n, int) and not isinstance(n, bool) else 0


def _e01(c: _Ctx, cross_sessions: CrossSessions) -> list[Finding]:
    hits = [t for t in _of(c.turns, _SYS) if t.cause == "sys_noise" or t.span_class in _E01_CLASSES]
    by_line: dict[int | None, list[LensTurn]] = {}
    for t in hits:
        by_line.setdefault(t.span_line, []).append(t)
    out: list[Finding] = []
    for line, ts in by_line.items():
        sessions = _cross(cross_sessions, line) + 1
        lost = _sum_lost(ts)
        if (len(ts) < 2 and sessions < 2) or lost < MIN_LOST:
            continue
        cls = _most_common(t.span_class for t in ts)
        if cls == "date" and _once_a_day(c.turns, ts):
            continue
        counts = {"turns": len(ts), "lost": lost, "sessions": sessions}
        if line is not None:
            counts["line"] = line
        if isinstance(cls, str) and cls != "other" and cls in SPAN_CLASSES:
            # The class as its index in probes.SPAN_CLASSES (a code, no text).
            counts["class"] = SPAN_CLASSES.index(cls)
        unclassified = all(t.cause == "sys_changed" and t.span_class in ("other", None) for t in ts)
        confidence: Confidence = "low" if unclassified else "high" if len(ts) >= 2 else "medium"
        cause = "sys_changed" if any(t.cause == "sys_changed" for t in ts) else "sys_noise"
        out.append(c.finding("E-01", ts, counts, cause=cause, confidence=confidence))
    return out


# -- E-02 .. E-09 --------------------------------------------------------------


def _e02(c: _Ctx) -> list[Finding]:
    ts = _of(c.turns, _TOOLS_ORDER)
    lost = _sum_lost(ts)
    if not ts or lost < MIN_LOST:
        return []
    cause = str(_most_common(t.cause for t in ts))
    return [c.finding("E-02", ts, {"turns": len(ts), "lost": lost}, cause=cause, confidence="high")]


def _e03(c: _Ctx) -> list[Finding]:
    # The version guard (13.2) needs prompt versions, which arrive in S3:
    # until then no toolset change is explained by a version.
    ts = _of(c.turns, frozenset({"tools_changed"}))
    if len(ts) < 2:
        return []
    counts = {"turns": len(ts), "lost": _sum_lost(ts)}
    return [c.finding("E-03", ts, counts, cause="tools_changed", confidence="high")]


def _per_message(
    rule: str, causes: frozenset[str], minimum: int
) -> Callable[[_Ctx], list[Finding]]:
    def run(c: _Ctx) -> list[Finding]:
        ts = _of(c.turns, causes)
        if len(ts) < minimum:
            return []
        counts = {"turns": len(ts), "lost": _sum_lost(ts)}
        msg = _most_common(_node(t.brk) for t in ts)
        if isinstance(msg, int):
            counts["message"] = msg
        cause = str(_most_common(t.cause for t in ts))
        return [c.finding(rule, ts, counts, cause=cause, confidence="high")]

    return run


_e04 = _per_message("E-04", _HISTORY, 2)
_e09 = _per_message("E-09", frozenset({"media_changed"}), 2)


def _e05(c: _Ctx) -> list[Finding]:
    # Each reasoning_* turn flips the carry-over relative to its parent: a switch.
    ts = _of(c.turns, _REASONING)
    if len(ts) < 2:
        return []
    counts = {
        "switches": len(ts),
        "dropped": sum(1 for t in ts if t.cause == "reasoning_dropped"),
        "kept": sum(1 for t in ts if t.cause == "reasoning_kept"),
        "lost": _sum_lost(ts),
    }
    cause = str(_most_common(t.cause for t in ts))
    return [c.finding("E-05", ts, counts, cause=cause, confidence="high")]


def _no_marks_split(c: _Ctx) -> tuple[list[LensTurn], list[LensTurn]]:
    """``provider_no_marks`` relay turns at or above the provider minimum, and
    the provider-minimum misses (stored as such, or a no-marks turn whose
    prefix is under the minimum: marks would not have helped)."""
    marks: list[LensTurn] = []
    below: list[LensTurn] = []
    for t in c.turns:
        if t.cause == "provider_below_min":
            below.append(t)
        elif t.cause == "provider_no_marks" and t.relayed and not t.cache_marks:
            prefix = t.cf_tokens
            if prefix is not None and prefix >= provider_min(t.model):
                marks.append(t)
            else:
                below.append(t)
    return marks, below


def _e06(c: _Ctx, marks: list[LensTurn]) -> list[Finding]:
    if len(marks) < MIN_NO_MARKS_TURNS:
        return []
    counts = {"turns": len(marks), "lost": _sum_lost(marks), "marks": 0}
    # Predicted from kvsig: an upper bound the breakpoints would cap (spec 5.2).
    return [c.finding("E-06", marks, counts, cause="provider_no_marks", confidence="medium")]


def _e07(c: _Ctx) -> list[Finding]:
    ts = [
        t
        for t in c.turns
        if t.cause == "compacted" and t.brk in ("sys", "tools") and t.edge != "fork"
    ]
    if not ts:
        return []
    tokens = 0
    after = 0
    for t in ts:
        later = sum(1 for u in c.turns if u.n > t.n and u.lane == t.lane)
        after += later
        # The head is recomputed on the compaction turn and on every turn after it.
        tokens += max(t.cf_tokens or 0, 0) * (1 + later)
    counts = {"compactions": len(ts), "after": after, "lost": _sum_lost(ts)}
    return [c.finding("E-07", ts, counts, cause="compacted", confidence="high", tokens=tokens)]


def _e08(c: _Ctx, below: list[LensTurn]) -> list[Finding]:
    out: list[Finding] = []
    # pauses.py: relayed turns are the upstream's cache, never counted.
    evicted = [
        t
        for t in c.turns
        if t.cause == "server_evicted"
        and not t.relayed
        and t.gap_s is not None
        and t.gap_s >= PAUSE_S
    ]
    if len(evicted) >= 2:
        counts = {
            "turns": len(evicted),
            "lost": _sum_lost(evicted),
            "pause_s": round(statistics.median(t.gap_s or 0.0 for t in evicted)),
        }
        out.append(c.finding("E-08", evicted, counts, cause="server_evicted", confidence="high"))
    if len(below) >= 2:
        counts = {"turns": len(below), "lost": _sum_lost(below)}
        out.append(
            c.finding("E-08", below, counts, cause="provider_below_min", confidence="medium")
        )
    return out


def efficiency_findings(
    turns: Sequence[LensTurn],
    *,
    cross_sessions: CrossSessions,
    prefill_rate: float | None = None,
) -> list[Finding]:
    """The session's E findings, in catalogue order (largest saving first
    within a rule). ``cross_sessions`` is the number of the key's own earlier
    sessions showing the same E-01 line (the caller counts them), one for
    every line or per span line;
    ``prefill_rate`` (tokens/s) turns the saving into seconds when known."""
    c = _Ctx(turns, prefill_rate)
    marks, below = _no_marks_split(c)
    found = (
        _e01(c, cross_sessions)
        + _e02(c)
        + _e03(c)
        + _e04(c)
        + _e05(c)
        + _e06(c, marks)
        + _e07(c)
        + _e08(c, below)
        + _e09(c)
    )
    order = {rid: i for i, rid in enumerate(RULES)}
    return sorted(found, key=lambda f: (order[f.rule], -f.saving.tokens))
