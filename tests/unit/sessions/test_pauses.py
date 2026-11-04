"""Resumes after a pause and the evictions they hit (sessions redesign, task 5)."""

import pytest

from app.sessions.pauses import PAUSE_BUCKETS, PauseTurn, lens, mark_pauses, summarize_pauses


def P(
    lane, start, dur=1.0, *, relayed=False, cached=0, reusable=10_000, outcome="lost", rate=1000.0
):
    return PauseTurn(lane, start, start + dur, relayed, cached, reusable, outcome, rate)


def test_pause_is_per_lane_not_per_session():
    turns = (
        [P("a", 0)]
        + [P("b", 5 + 10 * i, cached=9000, outcome="hit") for i in range(12)]
        + [P("a", 121)]
    )
    m = mark_pauses(turns)
    assert m[-1].pause_s == pytest.approx(120) and m[-1].resume and m[-1].evicted
    assert not any(x.resume for x in m[1:-1])
    assert m[0].pause_s is None


def test_relayed_resume_is_not_counted():
    assert mark_pauses([P("a", 0, relayed=True), P("a", 600, relayed=True)])[1].resume is False


def test_resume_that_hit_its_cache_is_not_an_eviction():
    m = mark_pauses([P("a", 0), P("a", 600, cached=9500, outcome="hit")])[1]
    assert m.resume and m.evicted is False and m.lost_tokens == 0


def test_half_lost_without_lost_outcome_is_an_eviction():
    m = mark_pauses([P("a", 0), P("a", 600, cached=4000, outcome="partial")])[1]
    assert m.evicted and m.lost_tokens == 6000 and m.lost_s == pytest.approx(6.0)


def test_small_reusable_is_never_an_eviction():
    assert mark_pauses([P("a", 0), P("a", 600, reusable=200)])[1].evicted is False


def test_unknown_cache_is_unassessed():
    m = mark_pauses([P("a", 0), P("a", 600, cached=None)])[1]
    assert m.resume and m.unassessed and not m.evicted
    assert summarize_pauses([m])["unassessed"] == 1


def test_null_lane_has_no_pause():
    assert mark_pauses([P(None, 0), P(None, 600)])[1].pause_s is None


def test_own_lens_wins_when_present():
    assert lens(reusable=10_000, reusable_own=300, outcome="lost", outcome_own="hit") == (
        300,
        "hit",
    )
    assert lens(reusable=10_000, reusable_own=None, outcome="lost", outcome_own=None) == (
        10_000,
        "lost",
    )


def test_buckets_and_disk_tier_hint():
    turns = [P("a", 0), P("a", 121), P("b", 0), P("b", 1001), P("c", 0), P("c", 7201, rate=None)]
    s = summarize_pauses(mark_pauses(turns))
    assert [b["resumes"] for b in s["buckets"]] == [1, 0, 1, 1, 0] and s["evicted"] == 3
    assert [b["label"] for b in s["buckets"]] == [b[0] for b in PAUSE_BUCKETS]
    assert s["disk_tier_hint"] == {"keep_s": 14400, "recovered_s": pytest.approx(20.0)}
    assert s["lost_s"] == pytest.approx(20.0) and s["lost_tokens"] == 30_000


def test_open_bucket_hint_uses_longest_evicted_pause():
    s = summarize_pauses(mark_pauses([P("a", 0), P("a", 18001)]))
    assert s["disk_tier_hint"]["keep_s"] == pytest.approx(18000)


def test_no_evictions_no_hint():
    assert summarize_pauses([])["disk_tier_hint"] is None
    assert len(summarize_pauses([])["buckets"]) == 5
