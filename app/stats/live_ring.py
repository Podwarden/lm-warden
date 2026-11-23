"""Server-side smoothing and history of the live engine rates (stats redesign B5).

The SSE frame's ``generation_tokens_per_s`` is a delta between two scrapes of
ONE connection, about 2 s apart: noisy, null on the first frame, and the Now
card's 5-minute chart lived in browser memory, empty on every page load.

``LiveRing`` keeps, per model and for the whole process, the last few minutes
of the engine's cumulative counters, appended whenever a fresh scrape arrives
(the shared scrape cache already collapses every viewer to one scrape). From it:

* ``now(model)``: generation tok/s and COMPUTED prefill tok/s over about 10 s.
  Computed prefill is ``prompt - prefix_cache_hits - external_prefix_cache_hits``
  (the engine counts prefix hits in tokens), clamped at 0: cache hits are not
  compute, so they must not inflate a "prefill now" figure.
* ``history(model)``: the interval rates of the last 5 minutes, sent once when
  a viewer connects so the chart starts full.

There is no background scraper: the ring fills while anyone watches the page
(or a second tab does). An unwatched gap shows as a gap (null points), never
as zeros, and no "now" rate is derived across it.
A counter going backwards (engine restart) clears that model's ring.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

SMOOTH_S = 10.0
HISTORY_S = 300.0
#: Two samples further apart than this were not both watched: no rate is
#: derived across them ("now" says unknown, the history shows a gap).
GAP_S = 2 * SMOOTH_S
_KEEP_S = HISTORY_S + 30.0


@dataclass(frozen=True)
class _Sample:
    t: float
    gen: float | None
    prompt: float | None
    hits: float | None
    ext: float | None


def _computed(s: _Sample) -> float | None:
    if s.prompt is None or s.hits is None:
        return None
    return s.prompt - s.hits - (s.ext or 0.0)


def _rate(a: float | None, b: float | None, dt: float) -> float | None:
    if a is None or b is None or dt <= 0:
        return None
    return max(0.0, (b - a) / dt)


class LiveRing:
    """Per-model ring of cumulative engine counters. One per process."""

    def __init__(self) -> None:
        self._rings: dict[str, deque[_Sample]] = {}

    def observe(self, model_id: str, t: float, r: Any) -> None:
        """Append one scrape (``r`` an EngineReading) taken at monotonic ``t``."""
        s = _Sample(
            t=t,
            gen=r.generation_tokens_total,
            prompt=r.prompt_tokens_total,
            hits=r.prefix_cache_hits,
            ext=r.external_prefix_cache_hits,
        )
        ring = self._rings.setdefault(model_id, deque())
        if ring:
            last = ring[-1]
            if t <= last.t:
                return  # the same cached scrape, seen by another viewer
            if (s.gen is not None and last.gen is not None and s.gen < last.gen) or (
                s.prompt is not None and last.prompt is not None and s.prompt < last.prompt
            ):
                ring.clear()  # engine restart: counters reset
        ring.append(s)
        while ring and t - ring[0].t > _KEEP_S:
            ring.popleft()

    def forget(self, model_id: str) -> None:
        self._rings.pop(model_id, None)

    def now(self, model_id: str) -> dict[str, Any]:
        """Rates over the last ~10 s: ``generation_tps_10s``,
        ``prefill_computed_tps_10s`` (None when unknown) and ``window_s``."""
        ring = self._rings.get(model_id)
        out: dict[str, Any] = {
            "generation_tps_10s": None,
            "prefill_computed_tps_10s": None,
            "window_s": None,
        }
        if not ring or len(ring) < 2:
            return out
        last = ring[-1]
        # Only the samples watched without a gap up to now count: a reference
        # from before an unwatched gap would turn a multi-minute average into
        # "now" (an Idle chip next to "78 tok/s").
        seg = [last]
        for s in reversed(list(ring)[:-1]):
            if seg[-1].t - s.t > GAP_S:
                break
            seg.append(s)
        seg.reverse()
        if len(seg) < 2:
            return out
        ref = seg[0]
        for s in seg:
            if s.t <= last.t - SMOOTH_S:
                ref = s
            else:
                break
        dt = last.t - ref.t
        out["window_s"] = round(dt, 1)
        g = _rate(ref.gen, last.gen, dt)
        p = _rate(_computed(ref), _computed(last), dt)
        out["generation_tps_10s"] = round(g, 1) if g is not None else None
        out["prefill_computed_tps_10s"] = round(p, 1) if p is not None else None
        return out

    def history(self, model_id: str, *, now: float) -> list[dict[str, Any]]:
        """Interval rates between consecutive samples of the last 5 minutes,
        oldest first: ``{age_s, gen, prefill}`` (age of the interval's end)."""
        ring = self._rings.get(model_id)
        if not ring:
            return []
        out = []
        prev: _Sample | None = None
        for s in ring:
            if prev is not None and now - s.t <= HISTORY_S:
                dt = s.t - prev.t
                if dt > GAP_S:
                    # Nobody watched: a gap, never one interpolated segment.
                    out.append({"age_s": round(now - s.t, 1), "gen": None, "prefill": None})
                    prev = s
                    continue
                g = _rate(prev.gen, s.gen, dt)
                p = _rate(_computed(prev), _computed(s), dt)
                out.append(
                    {
                        "age_s": round(now - s.t, 1),
                        "gen": round(g, 1) if g is not None else None,
                        "prefill": round(p, 1) if p is not None else None,
                    }
                )
            prev = s
        return out
