"""The replicated-oracle search (design §3).

Plain binary search is INVALID for this problem. Each bound move is
irreversible, and near the threshold the failure probability is around 0.5 —
measured on pw-bonus 2026-09-03, where the same prompt size crashed on one
attempt and passed on the next. A single query at the midpoint is then wrong
half the time, the error compounds, and the algorithm cannot detect it.

The fix is to replicate the oracle rather than abandon bisection. Define
``PASS_N(x)`` = "N consecutive trials at x all passed". Since

    Pr[PASS_N(x)] = (1 - p(x))**N

is monotone decreasing in ``p``, ``PASS_N`` remains a valid stochastic
threshold oracle — it simply sits on a lower contour. Bisecting on it converges
where ``(1-p)**N = 0.5``, i.e. ``p = 1 - 0.5**(1/N)``. The bias is downward,
which is the safe direction, and it is quantifiable, which is what lets the
contract publish it as a number rather than a hand-wave.

That argument establishes monotonicity in ``p``. Bisection additionally needs
``p`` monotone in ``x``, which nothing guarantees here — see
``SearchOutcome.is_non_monotone`` for why that matters and what we do about it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Pr[PASS_N] = 0.5  =>  p = 1 - 0.5**(1/N). Published as `contour_p` so a
# client can distinguish a quick run's contour from a thorough run's.
CONTOUR_BY_N: dict[int, float] = {n: 1.0 - 0.5 ** (1.0 / n) for n in (1, 3, 5, 7)}

Oracle = Callable[[], bool]
Probe = Callable[[int], bool]


def pass_n(oracle: Oracle, *, n: int) -> bool:
    """True when ``n`` consecutive trials all pass.

    Aborts on the first failure. That does not bias the verdict — one failure
    already decides it — and it is what keeps the expected cost at 1.75 trials
    per failing rung at p=0.5 rather than a flat 3.
    """
    for _ in range(n):
        if not oracle():
            return False
    return True


def confirm(oracle: Oracle, *, m: int) -> bool:
    """Phase C. Stricter than the bisection oracle, deliberately.

    ``m`` (5 quick / 7 thorough) exceeds ``n`` (3) because this value is the
    one that gets published. At p=0.5 a 5/5 false pass has probability 3.1%.
    """
    return pass_n(oracle, n=m)


def ladder(*, start: int, cap: int) -> list[int]:
    """Phase A's geometric rungs, always including the cap itself.

    The cap is a hard limit (``max_model_len``, ``max_num_seqs``), not a hint.
    It is included as a real rung because Phase D probes the axis maximum and
    needs it to be a value the search actually visited.
    """
    if cap < start:
        return [cap]
    rungs = []
    x = start
    while x < cap:
        rungs.append(x)
        x *= 2
    rungs.append(cap)
    return rungs


@dataclass(frozen=True)
class Bracket:
    """Phase A's output: ``low`` passed, ``high`` did not."""

    low: int
    high: int


def bisect(bracket: Bracket, probe: Probe, *, n: int, steps: int) -> int:
    """Phase B. Bisect on PASS_N inside a known bracket.

    Returns the largest value observed to pass. Never returns an unverified
    midpoint: publishing a value the oracle did not actually accept is how a
    search reports a limit the model does not have.
    """
    low, high = bracket.low, bracket.high
    best = low
    for _ in range(steps):
        if high - low <= 1:
            break
        mid = (low + high) // 2
        # Bound as a default arg, not captured: a late-binding closure over a
        # loop variable is correct here only by accident of call ordering.
        if pass_n(lambda m=mid: probe(m), n=n):
            best = mid
            low = mid
        else:
            high = mid
    return best


@dataclass(frozen=True)
class SearchOutcome:
    """The three numbers one axis produces, plus its monotonicity verdict.

    Three, not one, because a probabilistic edge is not a scalar: the value we
    confirmed, the smallest value ever seen to fail, and a recommendation that
    carries a safety factor.
    """

    confirmed: int
    first_failure: int | None
    axis_max: int
    non_monotone: bool = False

    def recommended(self, safety_factor: float) -> int:
        return int(self.confirmed * safety_factor)

    def is_non_monotone(self, probe: Probe) -> bool:
        """Phase D. Probe the axis maximum; if it PASSES, the search was unsound.

        Bisection assumes the failure region is upward-closed. Nothing here
        guarantees that: sliding-window attention makes long contexts cost
        sub-linearly, the measured 65k->131k VRAM plateau shows capacity is not
        smooth, and a CUDA alignment fault is notoriously non-monotone in size.

        If the maximum passes while something below it failed, there is an
        island of failure, the bracket was locked by a local dip, and Phase C
        has confidently converged on a value that may be several times too low.
        One probe converts a silent error into a visible one.
        """
        if self.first_failure is None:
            return False
        if self.first_failure >= self.axis_max:
            return False
        return probe(self.axis_max)

    def publishable(self) -> bool:
        """A non-monotone axis publishes nothing.

        A visible gap is better than a number that is quietly wrong — the
        chat context warning consumes this value, so a 4x underestimate would
        actively degrade a working product.
        """
        return not self.non_monotone
