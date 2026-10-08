"""Server-side history for the stats page.

The stats page kept no history anywhere: sparklines were roughly 80 seconds of
client-side React state that unmounted on navigation, rates were per-SSE-
connection deltas so every reconnect punched a hole, and a completed request was
erased from the registry in the streaming ``finally`` — discarding its duration,
TTFT and finish reason at the exact moment they became knowable. The result was
a dashboard that went blank precisely when something interesting had just
happened.

Everything in this module is free of engine, DB and HTTP imports. Retention is
the part that has to be right, and it is much easier to be sure of when it can
be tested without any of them.

Memory is bounded by construction. The warden runs a single uvicorn worker (see
``app/proxy/scheduler.py``), so process-local rings are sufficient; if that ever
changes these become the first thing that needs a shared store.

Completed requests used to live here too, in a ``FinishedRing`` (200 rows, 15
minutes, gone with the process). They are persisted now -- see
``app/stats/request_history.py`` -- because a per-request history that dies
with the process could answer neither "what happened today" nor "what is this
model's TTFT distribution".
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """One reading, with the wall-clock time it was taken.

    Keyed on time rather than on position, because the point of this ring is to
    survive gaps: a run that stops leaves real zero-valued samples behind, and
    "zero at 14:03" is a different statement from "no data for 14:03".
    """

    at: float
    value: float


class FrameRing:
    """A fixed-span, time-keyed ring of samples for one series.

    ``span_s`` is what the chart shows; ``step_s`` is the expected cadence and
    only sets the capacity. A caller ticking faster than ``step_s`` cannot grow
    this without bound — it lives for the life of the process.
    """

    def __init__(self, *, span_s: float, step_s: float) -> None:
        self.span_s = float(span_s)
        self.step_s = max(1e-6, float(step_s))
        # One spare slot so a caller running slightly fast does not evict a
        # sample that is still inside the span.
        self.capacity = max(1, int(self.span_s / self.step_s) + 2)
        self._q: deque[Sample] = deque(maxlen=self.capacity)

    def add(self, *, at: float, value: float) -> None:
        self._q.append(Sample(at=float(at), value=float(value)))
        self._evict(now=at)

    def _evict(self, *, now: float) -> None:
        cutoff = now - self.span_s
        while self._q and self._q[0].at < cutoff:
            self._q.popleft()

    def samples(self) -> list[Sample]:
        return list(self._q)

    def since(self, at: float) -> list[Sample]:
        """Samples strictly newer than ``at``.

        Lets a client backfill on mount and then take the live stream, without
        re-sending the whole window on every reconnect.
        """
        return [s for s in self._q if s.at > at]


def busy_median(values: Iterable[float], busy: Iterable[bool]) -> float | None:
    """Median over the samples where the host was working, or None.

    A *mean over everything* is the obvious choice and the wrong one: these
    series are mostly idle with bursts, so it lands between the two states,
    describes neither, and drifts with however much idle time the window
    happens to contain.

    ``busy`` is decided once at host level and passed in for every series,
    rather than derived per series. VRAM is why: it does not fall when the box
    goes quiet, because the weights stay resident, so a per-series "non-zero"
    test would call every idle minute busy and hand back the mean this exists
    to avoid.

    None when nothing was busy — no baseline is better than one computed from
    idle samples.
    """
    # strict=False deliberately: both sequences come from the same sampler
    # pass, but a short one should narrow the median, not raise inside a
    # stats read.
    picked = sorted(v for v, b in zip(values, busy, strict=False) if b)
    if not picked:
        return None
    mid = len(picked) // 2
    if len(picked) % 2:
        return float(picked[mid])
    return (float(picked[mid - 1]) + float(picked[mid])) / 2


def peak(values: Iterable[float], busy: Iterable[bool]) -> float | None:
    """Highest sample over the WHOLE period, busy or not.

    Deliberately not masked. A spike during an otherwise-quiet minute is still
    the peak, and for a series like VRAM that does not track business at all,
    masking would hide the one event worth seeing. ``busy`` is accepted so the
    two reference functions share a signature at the call site.
    """
    vals = list(values)
    return max(vals) if vals else None
