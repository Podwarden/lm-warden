"""Resumes after a pause and the evictions they hit (sessions pages, planning
data for the SSD and cluster tiers). A pause is per conversation lane, never per
session: parallel agents interleave. Relayed turns are not counted (their cache
is the upstream's). Unknown cache data is "unassessed", never an eviction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.cache_obs.classify import MIN_REUSABLE

PAUSE_S = 60.0
EVICT_SHARE = 0.5
#: (label, min_s, max_s); a pause lands where min_s <= pause < max_s.
PAUSE_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("1-5m", 60, 300),
    ("5-15m", 300, 900),
    ("15-60m", 900, 3600),
    ("1-4h", 3600, 14400),
    ("4h+", 14400, None),
)


def lens(
    *, reusable: int | None, reusable_own: int | None, outcome: str | None, outcome_own: str | None
) -> tuple[int | None, str | None]:
    """The own lens when the row has it, else the fleet lens (R9)."""
    if reusable_own is not None:
        return reusable_own, outcome_own
    return reusable, outcome


@dataclass(frozen=True)
class PauseTurn:
    lane: str | None
    started_at: float
    finished_at: float
    relayed: bool
    cached: int | None
    reusable: int | None
    outcome: str | None
    rate: float | None


@dataclass(frozen=True)
class PauseMark:
    pause_s: float | None
    resume: bool
    evicted: bool
    unassessed: bool
    lost_tokens: int | None
    lost_s: float | None


_NONE = PauseMark(None, False, False, False, None, None)


def mark_pauses(turns: Sequence[PauseTurn]) -> list[PauseMark]:
    """One mark per turn, parallel to ``turns``."""
    out: list[PauseMark] = [_NONE] * len(turns)
    last_end: dict[str, float] = {}
    for i in sorted(range(len(turns)), key=lambda k: turns[k].started_at):
        t = turns[i]
        if t.lane is None:
            continue
        prev = last_end.get(t.lane)
        last_end[t.lane] = max(prev or t.finished_at, t.finished_at)
        if prev is None:
            continue
        pause = max(0.0, t.started_at - prev)
        if t.relayed or pause < PAUSE_S:
            out[i] = PauseMark(pause, False, False, False, None, None)
            continue
        if t.cached is None or t.reusable is None:
            out[i] = PauseMark(pause, True, False, True, None, None)
            continue
        lost = max(0, t.reusable - t.cached)
        evicted = t.reusable >= MIN_REUSABLE and (
            t.outcome == "lost" or lost >= EVICT_SHARE * t.reusable
        )
        if evicted:
            out[i] = PauseMark(pause, True, True, False, lost, (lost / t.rate) if t.rate else None)
        else:
            out[i] = PauseMark(pause, True, False, False, 0, None)
    return out


def _bucket(pause: float) -> int:
    for k, (_label, lo, hi) in enumerate(PAUSE_BUCKETS):
        if pause >= lo and (hi is None or pause < hi):
            return k
    return -1


def summarize_pauses(marks: Iterable[PauseMark]) -> dict[str, Any]:
    buckets: list[dict[str, Any]] = [
        {
            "label": lbl,
            "min_s": lo,
            "max_s": hi,
            "resumes": 0,
            "evicted": 0,
            "lost_tokens": 0,
            "lost_s": None,
        }
        for lbl, lo, hi in PAUSE_BUCKETS
    ]
    resumes = evicted = unassessed = lost_tokens = 0
    lost_s: float | None = None
    ev: list[PauseMark] = []
    for m in marks:
        if not m.resume or m.pause_s is None:
            continue
        resumes += 1
        k = _bucket(m.pause_s)
        b = buckets[k] if k >= 0 else None
        if b is not None:
            b["resumes"] += 1
        if m.unassessed:
            unassessed += 1
            continue
        if not m.evicted:
            continue
        evicted += 1
        ev.append(m)
        lost_tokens += m.lost_tokens or 0
        if b is not None:
            b["evicted"] += 1
            b["lost_tokens"] += m.lost_tokens or 0
        if m.lost_s is not None:
            lost_s = (lost_s or 0.0) + m.lost_s
            if b is not None:
                b["lost_s"] = (b["lost_s"] or 0.0) + m.lost_s
    hint = None
    if ev:
        top = max(_bucket(m.pause_s or 0.0) for m in ev)
        hi = PAUSE_BUCKETS[top][2]
        keep = float(hi) if hi is not None else max(m.pause_s or 0.0 for m in ev)
        recovered = sum(m.lost_s or 0.0 for m in ev if (m.pause_s or 0.0) <= keep)
        hint = {"keep_s": keep, "recovered_s": recovered}
    return {
        "resumes": resumes,
        "evicted": evicted,
        "unassessed": unassessed,
        "lost_tokens": lost_tokens,
        "lost_s": lost_s,
        "buckets": buckets,
        "disk_tier_hint": hint,
    }
