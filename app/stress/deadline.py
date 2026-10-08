"""Per-probe deadlines (design §5.1).

Nothing else in this stack imposes one. ``httpx.AsyncClient`` is constructed
with ``timeout=None`` (``app/proxy/routes.py:569``), the scheduler's queue wait
is deliberately unbounded, and ``request_max_wall_s`` defaults to ``0.0``. So
without a deadline here, TIMEOUT is not an observable outcome at all and a
wedged engine would stall a run indefinitely.

The v1 design computed ``base + L/prefill_rate + G/decode_rate`` with the rates
fitted from the first successful probe and ×3 slack. Prefill attention is
quadratic in ``L``, so a rate fitted at 1k tokens and extrapolated linearly to
128k under-predicts badly — and a deadline that is too tight manufactures false
TIMEOUTs, each of which silently truncates the published context.

This drops the analytic model in favour of observed behaviour: scale off the
slowest probe that actually succeeded. It cannot be wrong about the shape of
the curve because it does not assume one.
"""

from __future__ import annotations

#: Consecutive timeouts after which the engine is considered wedged rather
#: than merely slow. A wedged engine still answers /health, so every probe
#: times out — and TIMEOUT consumes no crash budget and does not stop the
#: search, so without this the run burns the full deadline on every remaining
#: rung of the ladder.
WEDGED_AFTER = 3


class DeadlinePolicy:
    """Tracks how long probes legitimately take, and rules on wedging."""

    def __init__(self, *, floor_s: float, cap_s: float, factor: float) -> None:
        self.floor_s = floor_s
        self.cap_s = cap_s
        self.factor = factor
        self._slowest_success = 0.0
        self._consecutive_timeouts = 0

    def next_deadline(self) -> float:
        """Scaled off the slowest SUCCESS, clamped to [floor, cap].

        Probes climb through the run, and a long-context probe is
        prefill-dominated and legitimately slow, so the budget has to grow with
        them — but only on evidence.
        """
        scaled = self._slowest_success * self.factor
        return max(self.floor_s, min(self.cap_s, scaled))

    def record_success(self, elapsed_s: float) -> None:
        """Only a success is evidence of how long the work honestly takes."""
        self._slowest_success = max(self._slowest_success, elapsed_s)
        self._consecutive_timeouts = 0

    def record_timeout(self, _deadline_s: float) -> None:
        """Deliberately does NOT raise the budget.

        Letting a timeout extend the deadline would let one stall ratchet the
        budget up for the rest of the run, which is how a wedged engine turns
        into an unbounded one.
        """
        self._consecutive_timeouts += 1

    def wedged(self) -> bool:
        return self._consecutive_timeouts >= WEDGED_AFTER
