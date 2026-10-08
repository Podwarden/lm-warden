"""The replicated-oracle search (design §3).

Two properties are load-bearing and both were wrong in v1 of the design:

1. Plain bisection is INVALID here. Measured on pw-bonus 2026-09-03: the same
   prompt size crashed once and passed once. A single query at the midpoint is
   wrong half the time in the transition band, the error is irreversible, and
   the algorithm cannot detect it. `PASS_N` fixes this by replicating the
   oracle, converging on a lower, quantifiable failure contour.

2. Bisection needs p(x) monotone in *x*, which is asserted nowhere and argued
   against in three places (sliding-window attention, the 65k->131k VRAM
   plateau, and the open question of whether the crash depends on length at
   all). Phase D exists to catch exactly that.

The oracle is injected, so the whole search is testable against a configurable
p(fail) with no GPU.
"""

from __future__ import annotations

import random

from app.stress.search import (
    CONTOUR_BY_N,
    Bracket,
    SearchOutcome,
    bisect,
    confirm,
    ladder,
    pass_n,
)

# ---- PASS_N: the replicated oracle ---------------------------------------


def test_pass_n_requires_every_trial_to_pass():
    assert pass_n(lambda: True, n=3) is True
    assert pass_n(lambda: False, n=3) is False


def test_pass_n_aborts_on_the_first_failure():
    """Early abort must not change the verdict, only the cost.

    An aborted rung yields the identical PASS_N result, because one failure
    already decides it. This is what makes the expected cost 1.75 trials per
    failing rung at p=0.5 rather than 3.
    """
    calls = []

    def oracle():
        calls.append(1)
        return False

    assert pass_n(oracle, n=5) is False
    assert len(calls) == 1, "must stop at the first failure, not run all 5"


def test_pass_n_runs_every_trial_when_they_all_pass():
    calls = []

    def oracle():
        calls.append(1)
        return True

    assert pass_n(oracle, n=5) is True
    assert len(calls) == 5


def test_the_published_contour_table_is_exact():
    """Pr[PASS_N] = (1-p)^N = 0.5  =>  p = 1 - 0.5**(1/N).

    These numbers are published in the contract (design §7.5, `contour_p`), so
    a client can tell a 21%-contour number from a 9%-contour one. They are
    asserted here rather than trusted.
    """
    assert abs(CONTOUR_BY_N[1] - 0.5000) < 5e-4
    assert abs(CONTOUR_BY_N[3] - 0.2063) < 5e-4
    assert abs(CONTOUR_BY_N[5] - 0.1294) < 5e-4
    assert abs(CONTOUR_BY_N[7] - 0.0943) < 5e-4


def test_the_contour_is_monotone_decreasing_in_n():
    """More replication means a safer contour. The bias is always downward."""
    vals = [CONTOUR_BY_N[n] for n in sorted(CONTOUR_BY_N)]
    assert vals == sorted(vals, reverse=True)


# ---- Phase A: bracketing -------------------------------------------------


def test_ladder_is_geometric_and_respects_the_cap():
    assert ladder(start=1024, cap=16384) == [1024, 2048, 4096, 8192, 16384]


def test_ladder_never_exceeds_the_cap_even_when_it_is_not_a_power_of_two():
    """The cap is a real limit (max_model_len / max_num_seqs), not a hint."""
    assert ladder(start=1000, cap=5000)[-1] <= 5000


def test_ladder_includes_the_cap_itself_when_reachable():
    """Phase D probes the maximum, so the maximum must be a real rung."""
    assert 5000 in ladder(start=1000, cap=5000)


# ---- Phase B/C: bisection and confirmation -------------------------------


def _step_oracle(threshold, p_band=0.0, rng=None):
    """A model whose failure probability is a step at `threshold`.

    With p_band > 0 the rung AT the threshold is flaky, which is the measured
    behaviour this whole design exists to survive.
    """
    rng = rng or random.Random(0)

    def probe(x):
        if x < threshold:
            return True
        if p_band and x == threshold:
            return rng.random() > p_band
        return False

    return probe


def test_bisect_converges_inside_the_bracket_on_a_clean_step():
    probe = _step_oracle(threshold=8000)
    got = bisect(Bracket(low=4096, high=16384), probe, n=3, steps=8)
    assert 4096 <= got < 8000
    assert got > 6000, "should get close to the threshold, not stop at the low bound"


def test_bisect_never_returns_a_value_that_failed():
    """The returned value must be one the oracle actually accepted.

    Publishing an unverified midpoint is how a search reports a limit the model
    does not have.
    """
    probe = _step_oracle(threshold=5000)
    got = bisect(Bracket(low=1024, high=16384), probe, n=3, steps=10)
    assert probe(got) is True


def test_bisect_is_conservative_under_a_flaky_band():
    """With a 50% flaky rung, PASS_3 should land BELOW it, not on it."""
    probe = _step_oracle(threshold=8192, p_band=0.5, rng=random.Random(7))
    got = bisect(Bracket(low=4096, high=16384), probe, n=3, steps=8)
    assert got < 8192


def test_confirm_requires_m_consecutive_passes():
    assert confirm(lambda: True, m=5) is True
    assert confirm(lambda: False, m=5) is False


def test_confirm_is_stricter_than_the_bisection_oracle():
    """M (5 or 7) exceeds N (3) deliberately: the final value is published."""
    calls = []

    def flaky():
        calls.append(1)
        return len(calls) != 4  # fails on the 4th

    assert confirm(flaky, m=5) is False


# ---- Phase D: the monotonicity check ------------------------------------


def test_phase_d_detects_a_non_monotone_failure_region():
    """A model that fails at 32k but works at 128k breaks bisection entirely.

    This is not hypothetical: a CUDA alignment fault is non-monotone in size,
    and the design's own evidence has a crash at ~2.2k on a card that ran
    clean to 24k. Without this check, one local dip locks the bracket and
    Phase C converges — with 5/5 confidence — on a value 4x too low, which is
    then published and fed to the chat context warning.
    """

    def probe(x):
        return not (30000 <= x <= 40000)  # an island of failure

    outcome = SearchOutcome(confirmed=28000, first_failure=32000, axis_max=131072)
    assert outcome.is_non_monotone(probe) is True


def test_phase_d_passes_a_genuinely_monotone_model():
    probe = _step_oracle(threshold=40000)
    outcome = SearchOutcome(confirmed=32000, first_failure=40000, axis_max=131072)
    assert outcome.is_non_monotone(probe) is False


def test_a_non_monotone_result_publishes_nothing_for_that_axis():
    """Refusing to publish is the point: a visible gap beats a silent 4x error."""
    outcome = SearchOutcome(
        confirmed=28000, first_failure=32000, axis_max=131072, non_monotone=True
    )
    assert outcome.publishable() is False
    assert (
        SearchOutcome(confirmed=28000, first_failure=32000, axis_max=131072).publishable() is True
    )


# ---- the recorded triple -------------------------------------------------


def test_the_recommended_value_applies_the_safety_factor():
    outcome = SearchOutcome(confirmed=100000, first_failure=131072, axis_max=131072)
    assert outcome.recommended(safety_factor=0.9) == 90000


def test_all_three_numbers_are_kept():
    """A probabilistic edge cannot be expressed as one number (design §3.4)."""
    outcome = SearchOutcome(confirmed=136533, first_failure=147456, axis_max=262144)
    assert outcome.confirmed == 136533
    assert outcome.first_failure == 147456
    assert outcome.recommended(0.9) < outcome.confirmed < outcome.first_failure
