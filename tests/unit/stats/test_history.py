"""Server-side history for the stats page.

Three defects reported by the operator, 2026-09-05, all one architecture:
"after a request ends it seems that all charts and graphs reset". Nothing kept
history server-side, sparklines were ~80 s of client-side React state that
unmounted on navigation, and a completed request was erased from the registry
in the streaming `finally` -- so its duration, TTFT and finish reason were
discarded at the exact moment they became knowable.

Everything here is deliberately free of engine, DB and HTTP imports, so the
retention rules can be tested without any of them.
"""

from __future__ import annotations

from app.stats.history import FrameRing, busy_median, peak

# ---- the sample ring -----------------------------------------------------


def test_history_outlives_the_traffic_that_produced_it():
    """The whole point. A run that ended is still on the chart."""
    r = FrameRing(span_s=60, step_s=1)
    r.add(at=100.0, value=42.0)
    r.add(at=101.0, value=0.0)  # traffic stopped
    assert [s.value for s in r.samples()] == [42.0, 0.0]


def test_idle_is_a_sample_not_an_absence():
    """Zero-with-a-timestamp and no-data-at-all are different states, and the
    chart draws them differently: a quiet stretch versus a gap."""
    r = FrameRing(span_s=60, step_s=1)
    r.add(at=100.0, value=0.0)
    assert r.samples()[0].value == 0.0
    assert len(r.samples()) == 1


def test_samples_older_than_the_span_are_dropped():
    r = FrameRing(span_s=10, step_s=1)
    r.add(at=100.0, value=1.0)
    r.add(at=115.0, value=2.0)
    assert [s.value for s in r.samples()] == [2.0]


def test_the_ring_is_bounded_regardless_of_rate():
    """A caller ticking faster than `step_s` must not grow it without bound:
    this lives in memory for the life of the process."""
    r = FrameRing(span_s=10, step_s=1)
    for i in range(10_000):
        r.add(at=1000.0 + i * 0.001, value=1.0)
    assert len(r.samples()) <= r.capacity


def test_since_returns_only_what_the_caller_has_not_seen():
    """Backfill on mount, then the SSE takes over -- without re-sending the
    whole window on every reconnect."""
    r = FrameRing(span_s=60, step_s=1)
    for i in range(5):
        r.add(at=100.0 + i, value=float(i))
    assert [s.value for s in r.since(102.0)] == [3.0, 4.0]


def test_since_before_the_window_returns_everything_held():
    assert FrameRing(span_s=60, step_s=1).since(0.0) == []


# ---- the 24h reference ---------------------------------------------------


def test_the_baseline_is_the_median_of_busy_samples():
    """Mostly-idle series with bursts: a mean sits between the two states and
    describes neither."""
    values = [0, 0, 0, 0, 500, 510, 520]
    busy = [False, False, False, False, True, True, True]
    assert busy_median(values, busy) == 510


def test_the_baseline_ignores_idle_even_when_idle_dominates():
    values = [0] * 100 + [400, 500, 600]
    busy = [False] * 100 + [True, True, True]
    assert busy_median(values, busy) == 500


def test_a_series_with_no_busy_samples_has_no_baseline():
    """None, not zero: no line is drawn rather than one invented from idle."""
    assert busy_median([0, 0, 0], [False, False, False]) is None
    assert busy_median([], []) is None


def test_an_even_number_of_busy_samples_averages_the_middle_pair():
    assert busy_median([10, 20, 30, 40], [True] * 4) == 25


def test_the_peak_spans_every_sample_not_just_busy_ones():
    """A spike during an otherwise-quiet minute is still the peak, and for VRAM
    -- which does not fall when the box goes idle -- restricting it would hide
    exactly the event worth seeing."""
    assert peak([5, 900, 5], [False, False, False]) == 900


def test_the_peak_of_nothing_is_none():
    assert peak([], []) is None
