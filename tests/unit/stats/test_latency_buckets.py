"""Latency as a measured distribution, not five quantiles.

The operator asked for "gaussian diagrams for latency". A gaussian is the wrong
model and the project's own rule forbids it: TTFT here is bimodal by
construction — a prefix-cache hit lands near 0.3 s and a cold prefill near 6 s —
so a fitted curve, or the p50 of 2.2 s that currently sits in the valley
between the two humps, describes a request that never happened.

The raw material was already being thrown away. `Metrics.histogram()` returns
the engine's full cumulative bucket array and `build_frame` reduced it to five
numbers. Emitting the buckets lets the client draw windowed deltas — a real
measured distribution over the last N minutes — with the percentiles as markers
on it rather than as a substitute for it.
"""

from __future__ import annotations

from app.stats.live_engine import bucket_deltas, buckets_payload


def test_buckets_are_emitted_as_edge_and_count_pairs():
    hist = ([(0.1, 3.0), (1.0, 12.0), (5.0, 30.0)], 44.0, 30.0)
    assert buckets_payload(hist) == {
        "le": [0.1, 1.0, 5.0],
        "counts": [3.0, 12.0, 30.0],
        "count": 30.0,
        "sum": 44.0,
    }


def test_a_missing_histogram_emits_nothing_rather_than_zeros():
    """llama.cpp reports no histograms at all. Zeros would render as a
    distribution with every request instantaneous."""
    assert buckets_payload(None) is None


def test_the_infinity_bucket_survives_serialisation():
    """`+Inf` is the overflow bucket and carries the total; JSON has no
    infinity, so it must not become null and silently drop the tail."""
    hist = ([(0.5, 4.0), (float("inf"), 9.0)], 6.0, 9.0)
    out = buckets_payload(hist)
    assert out["le"][-1] is None  # the agreed encoding for +Inf
    assert out["counts"][-1] == 9.0


# ---- windowed deltas: cumulative counters into "the last N minutes" -------


def test_a_window_is_the_difference_between_two_cumulative_reads():
    older = {"le": [1.0, 5.0], "counts": [10.0, 20.0], "count": 20.0, "sum": 30.0}
    newer = {"le": [1.0, 5.0], "counts": [14.0, 31.0], "count": 31.0, "sum": 55.0}
    d = bucket_deltas(newer, older)
    assert d["counts"] == [4.0, 11.0]
    assert d["count"] == 11.0


def test_an_engine_restart_zeroes_the_counters_and_yields_nothing():
    """Cumulative counters restart at zero on a model reload. A naive
    subtraction gives negative bars; the honest answer is no window yet."""
    older = {"le": [1.0], "counts": [900.0], "count": 900.0, "sum": 100.0}
    newer = {"le": [1.0], "counts": [3.0], "count": 3.0, "sum": 1.0}
    assert bucket_deltas(newer, older) is None


def test_mismatched_bucket_boundaries_yield_nothing():
    """An engine upgrade can change the boundaries. Subtracting across them
    would silently compare different bins."""
    older = {"le": [1.0, 5.0], "counts": [1.0, 2.0], "count": 2.0, "sum": 1.0}
    newer = {"le": [1.0, 10.0], "counts": [2.0, 4.0], "count": 4.0, "sum": 2.0}
    assert bucket_deltas(newer, older) is None


def test_no_earlier_read_yields_nothing_rather_than_the_lifetime_total():
    """On the first frame the only honest answer is "not yet" — showing the
    cumulative total labelled 'last 5 min' is exactly the lifetime-as-live
    mislabelling this page is being rebuilt to remove."""
    assert bucket_deltas({"le": [1.0], "counts": [5.0], "count": 5.0, "sum": 1.0}, None) is None


def test_an_idle_window_is_an_empty_distribution_not_a_missing_one():
    """Nothing happened is a fact worth drawing: the panel says "no requests in
    this window" rather than going blank."""
    same = {"le": [1.0, 5.0], "counts": [7.0, 9.0], "count": 9.0, "sum": 3.0}
    d = bucket_deltas(same, same)
    assert d is not None and d["count"] == 0.0 and d["counts"] == [0.0, 0.0]
