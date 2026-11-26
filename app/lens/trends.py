"""Trends and the version comparison (analysis Task 3.2, spec 13).

Pure functions over ``SessionScoreRow`` (one row per scored session, with
the won't-fix marks already applied by ``wontfix.apply_marks`` where the
caller wants them). Nothing here reads a prompt: scores, rule ids, labels.

* ``trend_series``: the median per day, week or prompt version, with the 25th
  to 75th percentile band of the overall. A bucket with fewer than
  ``MIN_BUCKET`` sessions has no number at all (A-7), and a layer with fewer
  than ``MIN_BUCKET`` scored values in a bucket has none for that layer.
* ``rank_test``: Mann-Whitney U, normal approximation with tie correction,
  two-sided; no scipy.
* ``compare_versions``: version B against version A (A the earlier) on the
  per-session scores, "clear" at p < 0.05, at least ``MIN_SIDE`` scored
  sessions on each side.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from app.lens.store import SessionScoreRow

#: Scored sessions a trend point needs (spec 10.3).
MIN_BUCKET = 3
#: Scored sessions each side of a version comparison needs (spec 10.3).
MIN_SIDE = 5
#: Significance level of a "clear" verdict.
ALPHA = 0.05
#: Median points under which an unclear difference reads "no change".
NO_CHANGE_POINTS = 2
#: Median turns may differ this much before the comparison is not like for like.
TURNS_RATIO = 0.5

METRICS = ("overall", "E", "C", "R")

By = Literal["day", "week", "version"]
Verdict = Literal[
    "better_clear",
    "better_unclear",
    "no_change",
    "worse_clear",
    "worse_unclear",
    "not_enough",
]


def value_of(row: SessionScoreRow, metric: str) -> int | None:
    return {"overall": row.overall, "E": row.e, "C": row.c, "R": row.r}[metric]


def _quantile(vals: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks; ``vals`` sorted, non-empty."""
    if len(vals) == 1:
        return float(vals[0])
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def median(vals: Iterable[float]) -> int | None:
    s = sorted(vals)
    return round(_quantile(s, 0.5)) if s else None


@dataclass(frozen=True)
class TrendPoint:
    bucket: str
    n: int
    medians: dict[str, int | None]
    #: 25th-75th percentile of the overall, None when it is not shown.
    band: tuple[int, int] | None
    #: First session start in the bucket (epoch s), to order and mark versions.
    start: float = 0.0


def bucket_of(started: float, by: Literal["day", "week"], tz_offset_min: int) -> str:
    """The local day (``YYYY-MM-DD``) or the local week's Monday."""
    local = datetime.fromtimestamp(started, UTC) + timedelta(minutes=tz_offset_min)
    day = local.date()
    if by == "week":
        day = day - timedelta(days=day.weekday())
    return day.isoformat()


def trend_series(
    rows: Sequence[SessionScoreRow], *, by: By, tz_offset_min: int = 0
) -> list[TrendPoint]:
    """One point per bucket, oldest first. ``by="version"`` buckets by the
    session's prompt version (a session with none is left out)."""
    groups: dict[str, list[SessionScoreRow]] = {}
    for r in rows:
        if by == "version":
            if r.version is None:
                continue
            key = r.version
        else:
            key = bucket_of(r.started, by, tz_offset_min)
        groups.setdefault(key, []).append(r)
    out: list[TrendPoint] = []
    for key, grp in groups.items():
        meds: dict[str, int | None] = {}
        for m in METRICS:
            vals = [v for r in grp if (v := value_of(r, m)) is not None]
            meds[m] = median(vals) if len(grp) >= MIN_BUCKET and len(vals) >= MIN_BUCKET else None
        overall = sorted(v for r in grp if (v := r.overall) is not None)
        band = (
            (round(_quantile(overall, 0.25)), round(_quantile(overall, 0.75)))
            if meds["overall"] is not None
            else None
        )
        out.append(TrendPoint(key, len(grp), meds, band, start=min(r.started for r in grp)))
    out.sort(key=lambda p: (p.start, p.bucket))
    return out


def rank_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided p of the Mann-Whitney U test (normal approximation with tie
    correction, no continuity correction). 1.0 for an empty side or when every
    value is equal."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = sorted([(float(v), 0) for v in a] + [(float(v), 1) for v in b])
    n = n1 + n2
    ranks = [0.0] * n
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    r1 = sum(ranks[k] for k in range(n) if pooled[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u1 - mu) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    n_a: int
    n_b: int
    diffs: dict[str, int | None]
    verdict: Verdict
    like_for_like: bool
    note: str | None = None
    resolved: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    #: The rank test's p on the overall, None when there was no test.
    p: float | None = None


def fired_rules(rows: Iterable[SessionScoreRow]) -> set[str]:
    out: set[str] = set()
    for r in rows:
        for ids in r.rules.values():
            if isinstance(ids, list):
                out.update(str(i) for i in ids)
    return out


def _common(vals: Iterable[str]) -> str | None:
    c = Counter(vals)
    return c.most_common(1)[0][0] if c else None


def compare_versions(
    a_rows: Sequence[SessionScoreRow],
    b_rows: Sequence[SessionScoreRow],
    *,
    a: str = "",
    b: str = "",
    scored: Callable[[SessionScoreRow], int | None] = lambda r: r.overall,
) -> Comparison:
    """Version ``b`` against version ``a`` (the earlier one). Each side's
    sessions are the scored ones (``scored`` is the overall by default)."""
    sa = [v for r in a_rows if (v := scored(r)) is not None]
    sb = [v for r in b_rows if (v := scored(r)) is not None]
    diffs: dict[str, int | None] = {}
    for m in METRICS:
        va = [v for r in a_rows if (v := value_of(r, m)) is not None]
        vb = [v for r in b_rows if (v := value_of(r, m)) is not None]
        ma, mb = median(va), median(vb)
        diffs[m] = (
            mb - ma
            if ma is not None and mb is not None and min(len(va), len(vb)) >= MIN_SIDE
            else None
        )
    notes: list[str] = []
    like = True
    ha, hb = _common(r.harness for r in a_rows), _common(r.harness for r in b_rows)
    if ha != hb:
        like = False
        notes.append("the sessions come from different harnesses")
    if {r.route for r in a_rows} != {r.route for r in b_rows}:
        like = False
        notes.append("the sessions took different routes")
    ta, tb = median(r.turns for r in a_rows), median(r.turns for r in b_rows)
    if ta and tb and abs(tb - ta) / ta > TURNS_RATIO:
        like = False
        notes.append(f"typical session length differs ({ta} vs {tb} turns)")
    note = "; ".join(notes) or None
    fa, fb = fired_rules(a_rows), fired_rules(b_rows)
    resolved, new = sorted(fa - fb), sorted(fb - fa)
    if len(sa) < MIN_SIDE or len(sb) < MIN_SIDE:
        short = f"{min(len(sa), len(sb))} of {MIN_SIDE} scored sessions on one side"
        return Comparison(
            a,
            b,
            len(sa),
            len(sb),
            diffs,
            "not_enough",
            like,
            "; ".join([short, *notes]),
            resolved,
            new,
            None,
        )
    p = rank_test(sa, sb)
    delta = (median(sb) or 0) - (median(sa) or 0)
    if delta == 0:
        delta = (sum(sb) / len(sb) > sum(sa) / len(sa)) - (sum(sb) / len(sb) < sum(sa) / len(sa))
    verdict: Verdict
    if p < ALPHA and delta != 0:
        verdict = "better_clear" if delta > 0 else "worse_clear"
    elif abs(delta) < NO_CHANGE_POINTS:
        verdict = "no_change"
    else:
        verdict = "better_unclear" if delta > 0 else "worse_unclear"
    return Comparison(a, b, len(sa), len(sb), diffs, verdict, like, note, resolved, new, p)
