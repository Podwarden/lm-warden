"""Trends and the version comparison (analysis Task 3.2, spec 13)."""

from __future__ import annotations

from app.lens.store import SessionScoreRow
from app.lens.trends import (
    Comparison,
    compare_versions,
    rank_test,
    trend_series,
)

DAY = 86_400.0
#: A Monday 00:00 UTC.
MON = 1_700_438_400.0 - (1_700_438_400.0 % 604_800) + 4 * DAY


def row(
    n: int,
    *,
    at: float = MON,
    overall: int | None = 60,
    e: int | None = None,
    c: int | None = None,
    r: int | None = None,
    version: str | None = "aaaa1111",
    harness: str = "opencode",
    route: str = "local",
    turns: int = 20,
    rules: dict | None = None,
) -> SessionScoreRow:
    return SessionScoreRow(
        key_id="k1",
        sess_ref=f"s_{n:032x}",
        started=at,
        last_at=at + 60,
        model="m",
        route=route,
        harness=harness,
        version=version,
        turns=turns,
        e=e if e is not None else overall,
        c=c if c is not None else overall,
        r=r if r is not None else overall,
        overall=overall,
        rules=rules or {},
    )


def test_rank_test_matches_the_hand_computed_value():
    # U = 0, n1 = n2 = 5: z = -12.5 / sqrt(25 * 11 / 12), two-sided p = 0.0090
    assert abs(rank_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) - 0.0090) < 5e-4
    assert rank_test([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]) == rank_test(
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    )


def test_rank_test_with_ties_and_degenerate_inputs():
    assert rank_test([5, 5, 5], [5, 5, 5]) == 1.0
    assert rank_test([], [1, 2]) == 1.0
    same = rank_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert same > 0.99
    # tied values lower the variance a little, never raise p above the untied p
    assert (
        rank_test([1, 1, 2, 2, 3], [4, 4, 5, 5, 6])
        <= rank_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) + 0.01
    )


def test_weekly_medians_and_band():
    rows = [row(i, at=MON + i * 3600, overall=v) for i, v in enumerate([40, 50, 60, 70, 80])]
    rows += [row(10 + i, at=MON + 7 * DAY + i * 3600, overall=90) for i in range(3)]
    pts = trend_series(rows, by="week")
    assert [p.bucket for p in pts] == ["2023-11-20", "2023-11-27"]  # the weeks' Mondays
    assert pts[0].n == 5 and pts[0].medians["overall"] == 60 and pts[0].band == (50, 70)
    assert pts[1].medians["overall"] == 90 and pts[1].band == (90, 90)
    assert pts[0].start < pts[1].start


def test_a_bucket_under_three_sessions_has_no_number():  # A-7
    rows = [row(1, overall=70), row(2, overall=80)]
    (p,) = trend_series(rows, by="week")
    assert p.n == 2 and p.band is None
    assert p.medians == {"overall": None, "E": None, "C": None, "R": None}


def test_a_layer_with_under_three_scored_values_has_no_median():
    rows = [row(i, overall=70, e=None if i < 2 else 70) for i in range(4)]
    rows = [
        SessionScoreRow(**{**r.__dict__, "e": None if i < 2 else 70}) for i, r in enumerate(rows)
    ]
    (p,) = trend_series(rows, by="week")
    assert p.medians["overall"] == 70 and p.medians["E"] is None


def test_days_use_the_callers_utc_offset():
    late = MON + 23 * 3600  # 23:00 UTC Monday
    rows = [row(i, at=late + i) for i in range(3)]
    assert (
        trend_series(rows, by="day", tz_offset_min=0)[0].bucket
        != trend_series(rows, by="day", tz_offset_min=120)[0].bucket
    )


def test_by_version_leaves_out_sessions_without_one():
    rows = [row(i, version="aaaa1111") for i in range(3)] + [row(9, version=None)]
    pts = trend_series(rows, by="version")
    assert [p.bucket for p in pts] == ["aaaa1111"] and pts[0].n == 3


def _side(n, base, version, **kw):
    return [
        row(i + hash(version) % 1000 * 100, overall=base + i, version=version, **kw)
        for i in range(n)
    ]


def test_verdicts_at_the_five_session_edge():
    a, b = _side(5, 30, "a"), _side(5, 70, "b")
    c = compare_versions(a, b, a="a", b="b")
    assert isinstance(c, Comparison) and c.verdict == "better_clear" and c.diffs["overall"] == 40
    assert compare_versions(b, a).verdict == "worse_clear"
    short = compare_versions(a[:4], b)
    assert short.verdict == "not_enough" and short.p is None and "4 of 5" in (short.note or "")
    assert compare_versions(a, b[:4]).verdict == "not_enough"


def test_unclear_and_no_change():
    a = [row(i, overall=v, version="a") for i, v in enumerate([50, 60, 70, 80, 90])]
    b = [row(10 + i, overall=v, version="b") for i, v in enumerate([60, 68, 74, 88, 100])]
    c = compare_versions(a, b)
    assert c.verdict == "better_unclear" and c.p is not None and c.p >= 0.05
    flat = [row(20 + i, overall=v, version="b") for i, v in enumerate([51, 60, 71, 79, 90])]
    assert compare_versions(a, flat).verdict == "no_change"
    worse = [row(30 + i, overall=v, version="b") for i, v in enumerate([40, 55, 58, 80, 85])]
    assert compare_versions(a, worse).verdict == "worse_unclear"


def test_like_for_like_notes():
    a, b = _side(5, 30, "a"), _side(5, 70, "b")
    assert compare_versions(a, b).like_for_like is True
    other = _side(5, 70, "b", harness="aider")
    c = compare_versions(a, other)
    assert c.like_for_like is False and "harness" in (c.note or "")
    long = _side(5, 70, "b", turns=60)
    c = compare_versions(a, long)
    assert c.like_for_like is False and "length" in (c.note or "")
    relay = _side(5, 70, "b", route="relay")
    assert compare_versions(a, relay).like_for_like is False


def test_resolved_and_new_rules():
    a = [row(i, version="a", rules={"C": ["C-01"], "E": ["E-01"]}) for i in range(5)]
    b = [row(10 + i, version="b", rules={"E": ["E-01"], "R": ["R-04"]}) for i in range(5)]
    c = compare_versions(a, b)
    assert c.resolved == ["C-01"] and c.new == ["R-04"]


def test_a_layer_diff_needs_five_values_each_side():
    a = [row(i, version="a", overall=50) for i in range(5)]
    b = [
        SessionScoreRow(**{**row(10 + i, version="b", overall=70).__dict__, "c": None})
        for i in range(5)
    ]
    c = compare_versions(a, b)
    assert c.diffs["overall"] == 20 and c.diffs["C"] is None and c.diffs["E"] == 20
