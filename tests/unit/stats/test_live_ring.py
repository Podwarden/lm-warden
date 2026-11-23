"""B5: server-side smoothed live rates and the 5-minute ring.

Generation now = delta generation_tokens / delta t over about 10 s.
Prefill now (computed) = (delta prompt - delta prefix hits - delta external
hits) / delta t over about 10 s, clamped at 0. The ring survives page loads,
so a new viewer gets the last 5 minutes at once instead of an empty chart.
"""

from __future__ import annotations

from app.runtime.backends.metrics import EngineReading
from app.stats.live_ring import LiveRing


def _r(gen, prompt, hits, ext=0.0):
    return EngineReading(
        generation_tokens_total=gen,
        prompt_tokens_total=prompt,
        prefix_cache_hits=hits,
        external_prefix_cache_hits=ext,
    )


def test_ten_second_rates_and_computed_prefill():
    ring = LiveRing()
    # 2 s scrapes: 50 tok/s generation; 3000 prompt tok/s of which 1000 cached
    for i in range(8):
        ring.observe("m", 100.0 + 2 * i, _r(100.0 * i, 6000.0 * i, 2000.0 * i))
    now = ring.now("m")
    assert now["generation_tps_10s"] == 50.0
    assert now["prefill_computed_tps_10s"] == 2000.0
    assert now["window_s"] == 10.0


def test_counter_reset_starts_over_and_duplicates_are_ignored():
    ring = LiveRing()
    ring.observe("m", 100.0, _r(1000.0, 0.0, 0.0))
    ring.observe("m", 100.0, _r(1000.0, 0.0, 0.0))  # same scrape (shared cache)
    ring.observe("m", 102.0, _r(1100.0, 0.0, 0.0))
    assert len(ring.history("m", now=102.0)) == 1
    ring.observe("m", 104.0, _r(5.0, 0.0, 0.0))  # engine restarted
    assert ring.now("m")["generation_tps_10s"] is None
    assert ring.history("m", now=104.0) == []


def test_history_is_five_minutes_of_interval_rates():
    ring = LiveRing()
    for i in range(200):  # 400 s of scrapes
        ring.observe("m", 2.0 * i, _r(20.0 * i, 0.0, 0.0))
    h = ring.history("m", now=398.0)
    assert h and all(p["age_s"] <= 300.0 for p in h)
    assert h[-1]["gen"] == 10.0 and h[-1]["prefill"] == 0.0


def test_prefill_needs_the_hit_counter():
    ring = LiveRing()
    ring.observe("m", 0.0, _r(0.0, 0.0, None))
    ring.observe("m", 10.0, _r(10.0, 1000.0, None))
    assert ring.now("m")["prefill_computed_tps_10s"] is None
    assert ring.now("m")["generation_tps_10s"] == 1.0


def test_no_rate_across_an_unwatched_gap():
    """A viewer connecting after nobody watched must not get a multi-minute
    average dressed up as "now" (review finding 8): no reference older than
    2 x the smoothing window, so the first frames say unknown, not 78 tok/s."""
    ring = LiveRing()
    for i in range(5):
        ring.observe("m", 2.0 * i, _r(100.0 * i, 0.0, 0.0))  # busy until t=8
    ring.observe("m", 300.0, _r(5000.0, 0.0, 0.0))  # first scrape after the gap
    now = ring.now("m")
    assert now["generation_tps_10s"] is None and now["prefill_computed_tps_10s"] is None
    ring.observe("m", 302.0, _r(5000.0, 0.0, 0.0))
    assert ring.now("m")["generation_tps_10s"] == 0.0


def test_history_shows_an_unwatched_gap_as_a_gap():
    ring = LiveRing()
    for t in (0.0, 2.0, 4.0, 200.0, 202.0):
        ring.observe("m", t, _r(t * 10, 0.0, 0.0))
    h = ring.history("m", now=202.0)
    gens = [p["gen"] for p in h]
    assert None in gens  # the gap, not one interpolated segment
    assert gens[-1] == 10.0
