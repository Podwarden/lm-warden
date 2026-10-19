"""In-memory registry of active load-progress trackers.

Holds the :class:`~app.runtime.load_progress.tracker.LoadTracker` and the
latest :class:`~app.runtime.load_progress.policy.Verdict` for every model
load currently in flight, so the UI / API layer can read live progress
without reaching into the load runner.

Lives on ``app.state.load_progress`` (created once in ``app/main.py``) and
is mutated only by the load runner's watcher loop.  ``finish`` only removes
the entry when the generation matches, so a superseded load cannot evict a
newer load's tracker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.runtime.load_progress.policy import Verdict
    from app.runtime.load_progress.tracker import LoadTracker

__all__ = ["LoadProgressRegistry"]


class LoadProgressRegistry:
    """Per-model live load progress, keyed by ``model_id``.

    The registry holds at most one entry per model id: a newer load for the
    same model replaces the previous tracker the moment ``start`` is called.
    ``finish`` is generation-guarded so a superseded load cannot evict a
    newer load's entry by reporting its own completion.
    """

    def __init__(self) -> None:
        self._trackers: dict[str, LoadTracker] = {}
        self._verdicts: dict[str, Verdict] = {}

    def start(self, tracker: LoadTracker) -> None:
        """Register ``tracker`` for its ``model_id``, replacing any prior
        entry.  A newer load supersedes an older one by construction: the
        caller creates a fresh tracker (with a fresh generation) and calls
        ``start`` before the watcher loop begins."""
        self._trackers[tracker.model_id] = tracker
        self._verdicts.pop(tracker.model_id, None)

    def get(self, model_id: str) -> LoadTracker | None:
        return self._trackers.get(model_id)

    def verdict(self, model_id: str) -> Verdict | None:
        return self._verdicts.get(model_id)

    def set_verdict(self, model_id: str, verdict: Verdict) -> None:
        """Record the latest verdict for ``model_id``.

        Separated from ``start`` so the watcher loop can update the verdict
        every tick without re-creating the tracker.
        """
        # Only record a verdict for a model we are actively tracking.  A
        # verdict arriving after ``finish`` (an async watcher tick that
        # landed after the load completed) is dropped: the load is over and
        # the verdict is stale.
        if model_id in self._trackers:
            self._verdicts[model_id] = verdict

    def finish(self, model_id: str, generation: int) -> None:
        """Remove the entry for ``model_id`` only if ``generation`` matches
        the tracker's generation.  A superseded load finishing must not
        evict a newer load's tracker."""
        tracker = self._trackers.get(model_id)
        if tracker is None or tracker.generation != generation:
            return
        self._trackers.pop(model_id, None)
        self._verdicts.pop(model_id, None)

    def active(self) -> list[str]:
        """Model ids with a live load in progress."""
        return list(self._trackers)
