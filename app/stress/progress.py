"""What a run reports about itself while it is still running.

Observations are written once, at finish. A thorough run takes hours, so until
this existed the modal showed "0 probe(s)", "No probes recorded yet", and a bar
rendered FULL — which reads as "finished" rather than "unknown", the worst of
the three possible wrong answers.

The ETA is derived, never extrapolated. Each phase declares how many steps it
plans, and the estimate applies THAT PHASE's measured pace to its own
remainder. Carrying a pace across phases would be meaningless: a length rung is
one HTTP probe, a sweep candidate is an engine reload that can take ten
minutes. A phase that cannot know its own total reports no ETA rather than a
guess — the same rule the published limits follow.
"""

from __future__ import annotations

import time


class Progress:
    """Mutable, single-writer, read by a snapshot. Never raises."""

    def __init__(self) -> None:
        self.phase: str | None = None
        self.note: str | None = None
        self.total: int | None = None
        self.step_n: int = 0
        self._phase_started: float = time.monotonic()

    def enter(
        self,
        phase: str,
        *,
        total: int | None = None,
        note: str | None = None,
        at: float | None = None,
    ) -> None:
        """Begin a phase, resetting the pace measurement with it."""
        self.phase = phase
        self.total = total
        self.note = note
        self.step_n = 0
        self._phase_started = time.monotonic() if at is None else at

    def step(self, *, note: str | None = None, at: float | None = None) -> None:
        self.step_n += 1
        if note is not None:
            self.note = note
        self._last = time.monotonic() if at is None else at

    def snapshot(self, *, now: float | None = None) -> dict:
        """A complete, JSON-safe view. Valid before any phase has begun."""
        now = time.monotonic() if now is None else now
        return {
            "phase": self.phase,
            "note": self.note,
            # `completed`/`total`, not `step`/`steps_total`: this is the
            # vocabulary StressRunProgress already reads. Inventing a second
            # one is how the results panel came to render nothing for a week.
            "completed": self.step_n,
            "total": self.total,
            "phase_elapsed_s": round(max(0.0, now - self._phase_started), 1),
            "eta_s": self._eta(now),
        }

    def _eta(self, now: float) -> float | None:
        # No total, no steps yet, or already past the plan: say nothing. An ETA
        # that counts backwards is worse than no ETA.
        if not self.total or self.step_n <= 0 or self.step_n >= self.total:
            return None
        elapsed = now - self._phase_started
        if elapsed <= 0:
            return None
        return round(elapsed / self.step_n * (self.total - self.step_n), 1)
