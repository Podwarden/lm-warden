"""Per-probe deadlines (design §5.1).

The harness must supply its own or "timeout" is not an observable outcome at
all: httpx.AsyncClient is constructed with timeout=None (app/proxy/routes.py:569),
the scheduler's queue wait is deliberately unbounded, and request_max_wall_s
defaults to 0.0.

v1 of the design used `base + L/prefill_rate + G/decode_rate` with the rates
fitted from the first successful probe and x3 slack. Prefill attention is
quadratic in L, so a rate fitted at 1k and extrapolated linearly to 128k
under-predicts badly -- manufacturing false TIMEOUTs that silently truncate the
published context. v2 drops the model entirely and tracks observed behaviour.
"""

from __future__ import annotations

from app.stress.deadline import DeadlinePolicy


def test_the_first_probe_gets_the_floor():
    """Nothing has been observed yet, so there is nothing to scale from."""
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    assert p.next_deadline() == 30.0


def test_the_deadline_scales_off_the_slowest_success_so_far():
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    p.record_success(20.0)
    assert p.next_deadline() == 80.0


def test_only_the_slowest_success_matters():
    """Probes climb, so the last one is not necessarily the slowest."""
    p = DeadlinePolicy(floor_s=10.0, cap_s=300.0, factor=4.0)
    p.record_success(40.0)
    p.record_success(5.0)
    assert p.next_deadline() == 160.0


def test_the_deadline_never_drops_below_the_floor():
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    p.record_success(0.5)
    assert p.next_deadline() == 30.0


def test_the_deadline_is_capped():
    """An unbounded deadline turns a wedged engine into an unbounded run."""
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    p.record_success(200.0)
    assert p.next_deadline() == 300.0


def test_it_grows_as_the_search_climbs():
    """Long-context probes are prefill-dominated and legitimately slow.

    A fixed deadline would fail them and record a capacity limit that is really
    a harness artefact.
    """
    p = DeadlinePolicy(floor_s=30.0, cap_s=900.0, factor=4.0)
    first = p.next_deadline()
    p.record_success(12.0)
    mid = p.next_deadline()
    p.record_success(60.0)
    late = p.next_deadline()
    assert first < mid < late


def test_a_timeout_does_not_raise_the_deadline():
    """Otherwise one slow probe ratchets the budget up for the whole run.

    Only a SUCCESS is evidence of how long the work legitimately takes.
    """
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    p.record_success(10.0)
    before = p.next_deadline()
    p.record_timeout(before)
    assert p.next_deadline() == before


def test_consecutive_timeouts_are_counted():
    """Three in a row means the engine is wedged, not that probes are big.

    A wedged engine still answers /health, so every probe times out -- and a
    timeout consumes no crash budget and does not stop the run. Without this
    counter the run burns the deadline cap on every remaining rung.
    """
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    assert p.wedged() is False
    p.record_timeout(30.0)
    p.record_timeout(30.0)
    assert p.wedged() is False
    p.record_timeout(30.0)
    assert p.wedged() is True


def test_a_success_clears_the_wedged_counter():
    p = DeadlinePolicy(floor_s=30.0, cap_s=300.0, factor=4.0)
    p.record_timeout(30.0)
    p.record_timeout(30.0)
    p.record_success(5.0)
    p.record_timeout(30.0)
    assert p.wedged() is False
