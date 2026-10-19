"""Stall policy for the model-load-progress feature.

Replaces a plain wall-clock deadline with a progress-aware policy: from a
per-load :class:`~app.runtime.load_progress.tracker.LoadTracker` it decides
whether a model load is *progressing*, *stalled* (amber warning), or should be
*given up*.

The policy is pure — it holds no I/O, no timers, no callbacks.  Callers feed
it a tracker, a ``now`` (monotonic seconds), and a :class:`StallPolicy`, and
read the :class:`Verdict`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.config import Settings
    from app.runtime.load_progress.tracker import LoadTracker

__all__ = ["StallPolicy", "Verdict", "evaluate"]

# Case-insensitive match for NCCL startup strings that indicate the GPUs are
# still negotiating their interconnect — the most common real stall.
_NCCL_RE = re.compile(r"NCCL|init_process_group|distributed", re.IGNORECASE)


@dataclass(frozen=True)
class StallPolicy:
    """Thresholds for the stall policy, derived from :class:`Settings`.

    ``warn_s`` — no progress (excluding CPU) for this long -> amber warning.
    ``give_up_s`` — no progress (including CPU) for this long -> give up.
    ``max_s`` — absolute wall-clock cap since load start -> give up.
    """

    warn_s: float
    give_up_s: float
    max_s: float

    @classmethod
    def from_settings(cls, s: Settings) -> StallPolicy:
        return cls(
            warn_s=s.load_stall_warn_s,
            give_up_s=s.load_stall_timeout_s,
            max_s=s.load_max_s,
        )


@dataclass(frozen=True)
class Verdict:
    """The outcome of :func:`evaluate`.

    ``state`` — ``"progressing"``, ``"stalled"``, or ``"give_up"``.
    ``stalled_for_s`` — how long (seconds) the load has been stalled by the
        warn rule (``now - last_progress_at(include_cpu=False)``).  Always
        non-negative; 0.0 on a fresh tracker.
    ``give_up_at_s`` — seconds since load start at which the load WILL be given
        up if nothing moves: ``min(last_progress_at(include_cpu=True) -
        started_at + give_up_s, max_s)``.  Moves forward with progress.
    ``reason`` — human-readable string when stalled or give_up, else ``None``.
    ``hint`` — operator-facing hint for the most likely cause, else ``None``.
    """

    state: Literal["progressing", "stalled", "give_up"]
    stalled_for_s: float
    give_up_at_s: float
    reason: str | None
    hint: str | None


def evaluate(t: LoadTracker, now: float, policy: StallPolicy) -> Verdict:
    """Decide whether a model load is progressing, stalled, or should be given up.

    Never raises — this runs every few seconds inside the load path.  A fresh
    tracker with zero events and zero vitals returns ``"progressing"``.
    """
    # Guard against negative clocks / time going backwards.  The tracker is
    # monotonic, but a test or a clock skew could produce a negative delta; we
    # clamp to 0 so the thresholds are never accidentally satisfied by a
    # negative value.
    last_progress_no_cpu = t.last_progress_at(include_cpu=False)
    last_progress_with_cpu = t.last_progress_at(include_cpu=True)
    started_at = t.started_at

    stalled_for_s = max(0.0, now - last_progress_no_cpu)
    since_start = max(0.0, now - started_at)

    # give_up_at_s: when the load WILL be given up if nothing moves.
    # last_progress_with_cpu - started_at + give_up_s, capped by max_s.
    give_up_at_s = min(
        max(0.0, last_progress_with_cpu - started_at) + policy.give_up_s,
        policy.max_s,
    )

    # --- Determine state -----------------------------------------------------
    # Give-up: absolute cap, or no CPU progress for give_up_s.
    cap_exceeded = since_start >= policy.max_s
    cpu_stalled = max(0.0, now - last_progress_with_cpu) >= policy.give_up_s

    if cap_exceeded or cpu_stalled:
        reason: str | None
        if cap_exceeded:
            hours = policy.max_s / 3600.0
            # Trim to a reasonable precision for the message.
            if hours == int(hours):
                reason = f"still not ready after {int(hours)} h (limit)"
            else:
                reason = f"still not ready after {hours:.1f} h (limit)"
        else:
            minutes = int((now - last_progress_with_cpu) // 60)
            reason = f"no progress for {minutes} min"
        return Verdict(
            state="give_up",
            stalled_for_s=stalled_for_s,
            give_up_at_s=give_up_at_s,
            reason=reason,
            hint=_hint(t),
        )

    # Stalled (amber): no non-CPU progress for warn_s.
    if stalled_for_s >= policy.warn_s:
        minutes = int(stalled_for_s // 60)
        return Verdict(
            state="stalled",
            stalled_for_s=stalled_for_s,
            give_up_at_s=give_up_at_s,
            reason=f"no progress for {minutes} min",
            hint=_hint(t),
        )

    # Progressing.
    return Verdict(
        state="progressing",
        stalled_for_s=stalled_for_s,
        give_up_at_s=give_up_at_s,
        reason=None,
        hint=None,
    )


def _hint(t: LoadTracker) -> str | None:
    """Return the operator-facing hint, or ``None``.

    - NCCL/init_process_group/distributed in the last log line -> GPU-connect
      hint.
    - CPU is the only thing keeping the load alive -> CPU-busy hint.
    - Otherwise ``None``.
    """
    line = t.last_log_line
    if line and _NCCL_RE.search(line):
        return "Looks stuck while the GPUs connect to each other"

    # "CPU is the only thing keeping it alive": last_progress_at with CPU is
    # strictly later than without it — i.e. CPU vitals moved but nothing else.
    with_cpu = t.last_progress_at(include_cpu=True)
    without_cpu = t.last_progress_at(include_cpu=False)
    if with_cpu > without_cpu:
        return "CPU busy but nothing else moving"

    return None
