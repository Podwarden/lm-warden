"""Unit tests for the stateful runaway-generation detector (app/proxy/runaway.py).

Pure, synchronous tests: feed text deltas, assert the verdict. Thresholds are
shrunk to small values so a test needs only a handful of feeds — the detector's
logic is threshold-parametric, so a budget of 10 exercises exactly the same code
path as the production 24000.
"""

from app.proxy.runaway import RunawayDetector, Verdict, finish_reason_for


def test_think_never_closes_trips_at_budget():
    d = RunawayDetector(think_budget=10, repeat_max=10**9, hard_max=10**9)
    assert d.feed("<think>") == Verdict.OK
    out = [d.feed("a") for _ in range(20)]
    # First 10 deltas inside the open block stay OK; the 11th exceeds the budget.
    assert all(v == Verdict.OK for v in out[:10])
    assert out[10] == Verdict.TRIP_THINK
    assert d.tripped is True
    assert d.finish_reason == "runaway_think"


def test_think_opens_and_closes_normally_passes_even_when_long():
    d = RunawayDetector(think_budget=10, repeat_max=10**9, hard_max=10**9)
    d.feed("<think>")
    for _ in range(8):  # under budget while open
        assert d.feed("x") == Verdict.OK
    assert d.feed("</think>") == Verdict.OK
    # Think block is closed — a long, varied non-repeating tail must NOT trip
    # on the think budget even though it far exceeds it in length.
    out = [d.feed(f"answer chunk number {i} is unique\n") for i in range(200)]
    assert all(v == Verdict.OK for v in out)
    assert d.tripped is False


def test_repetition_loop_trips():
    d = RunawayDetector(think_budget=10**9, repeat_max=5, hard_max=10**9, shingle_size=8)
    # The same 8-char shingle recurs every feed; the 6th occurrence (>5) trips.
    out = [d.feed("ABCDEFGH") for _ in range(20)]
    assert Verdict.TRIP_REPEAT in out
    assert d.finish_reason == "runaway_repeat"


def test_long_non_repeating_output_does_not_false_trip():
    d = RunawayDetector(think_budget=10**9, repeat_max=6, hard_max=10**9)
    # Realistic-looking, non-repeating code output. Every 48-char window carries
    # a unique integer, so no shingle recurs anywhere near the threshold.
    for i in range(600):
        line = f"    result_{i} = compute_value({i}, seed={i * 7 + 3}, tag='r{i}')\n"
        assert d.feed(line) == Verdict.OK
    assert d.tripped is False


def test_hard_backstop_trips():
    d = RunawayDetector(think_budget=10**9, repeat_max=10**9, hard_max=10)
    out = [d.feed("x") for _ in range(15)]
    assert all(v == Verdict.OK for v in out[:10])
    assert out[10] == Verdict.TRIP_HARD
    assert d.finish_reason == "runaway_hard"


def test_open_tag_split_across_deltas_is_detected():
    d = RunawayDetector(think_budget=5, repeat_max=10**9, hard_max=10**9)
    # "<think>" arrives split across two SSE deltas.
    assert d.feed("<thi") == Verdict.OK
    assert d.feed("nk>") == Verdict.OK
    out = [d.feed("a") for _ in range(10)]
    assert Verdict.TRIP_THINK in out


def test_close_tag_split_across_deltas_closes_block():
    d = RunawayDetector(think_budget=5, repeat_max=10**9, hard_max=10**9)
    d.feed("<think>")
    d.feed("a")
    d.feed("b")
    # "</think>" split across deltas — must close the block so the budget stops.
    assert d.feed("</thi") == Verdict.OK
    assert d.feed("nk>") == Verdict.OK
    # Well past the budget of 5, but the block is closed → no think trip.
    out = [d.feed(f"tail-{i}-unique\n") for i in range(30)]
    assert all(v == Verdict.OK for v in out)


def test_tag_fully_inside_tail_is_not_double_counted():
    # Regression: the ≤16-char tail buffer is re-scanned each feed. A tag whose
    # bytes land entirely inside the retained tail must not be re-detected.
    d = RunawayDetector(think_budget=3, repeat_max=10**9, hard_max=10**9)
    # Feed the open tag, then a delta short enough that the tag stays in the tail.
    d.feed("<think>")
    out = [d.feed("z") for _ in range(10)]
    # Exactly one open was counted: the block trips once, after ~3 deltas, not
    # sooner from a phantom re-open resetting the counter.
    assert Verdict.TRIP_THINK in out


def test_empty_delta_does_not_count_toward_budget():
    d = RunawayDetector(think_budget=10**9, repeat_max=10**9, hard_max=3)
    for _ in range(10):
        assert d.feed("") == Verdict.OK
    assert d.total == 0
    assert d.tripped is False


def test_verdict_is_sticky_after_trip():
    d = RunawayDetector(think_budget=10**9, repeat_max=10**9, hard_max=2)
    d.feed("a")
    d.feed("a")
    assert d.feed("a") == Verdict.TRIP_HARD
    # Further feeds keep returning the same terminal verdict, no re-evaluation.
    assert d.feed("b") == Verdict.TRIP_HARD


def test_think_tokens_counts_deltas_inside_block():
    d = RunawayDetector(think_budget=10**9, repeat_max=10**9, hard_max=10**9)
    d.feed("<think>")
    d.feed("a")
    d.feed("b")
    d.feed("c")
    d.feed("</think>")
    d.feed("outside")
    assert d.think_tokens == 3


def test_finish_reason_for_ok_is_none():
    assert finish_reason_for(Verdict.OK) is None
    assert finish_reason_for(Verdict.TRIP_THINK) == "runaway_think"
    assert finish_reason_for(Verdict.TRIP_REPEAT) == "runaway_repeat"
    assert finish_reason_for(Verdict.TRIP_HARD) == "runaway_hard"
