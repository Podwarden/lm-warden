"""The load-progress watcher loop.

``watch_load`` runs alongside the engine health wait and feeds engine log
lines and vital-sign samples into a :class:`LoadTracker`, then asks the
:class:`StallPolicy` whether the load is progressing, stalled, or should be
given up.  The loop is total: any exception inside one tick is logged and
the loop continues — it must never fail a load by raising.

The loop returns when:

* ``still_current()`` is false (a newer load has superseded this one), or
* the tracker is done (the load reached ready), or
* the policy returned a ``give_up`` verdict and ``on_give_up`` has been
  called once.

It is the caller's responsibility to translate ``give_up`` into a status
write, a stop-event signal, and an engine unload — see
``app/models/routes_api.py``'s ``_run_load`` for the wiring and the
``test_give_up_unload_survives_runner_cancellation`` test for the
cancellation hazard that wiring exists to avoid.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from app.runtime.load_progress.follow import LogFollower
from app.runtime.load_progress.parse import (
    LogEvent,
    parse_llamacpp_line,
    parse_vllm_line,
    split_log_chunk,
)
from app.runtime.load_progress.policy import StallPolicy, Verdict, evaluate
from app.runtime.load_progress.registry import LoadProgressRegistry
from app.runtime.load_progress.tracker import LoadTracker, VitalsSample

logger = logging.getLogger(__name__)

__all__ = ["watch_load"]


async def watch_load(
    *,
    tracker: LoadTracker,
    follower: LogFollower,
    sample: Callable[[], Awaitable[VitalsSample]],
    policy: StallPolicy,
    still_current: Callable[[], bool],
    on_give_up: Callable[[Verdict], Awaitable[None]],
    registry: LoadProgressRegistry,
    interval_s: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll the engine log and vitals, feed the tracker, and evaluate the
    stall policy every ``interval_s`` until the load is done, superseded,
    or given up.

    Never raises: any exception inside one tick is logged and the loop
    continues.  ``on_give_up`` is called at most once; the loop returns
    immediately after.
    """
    # Pick the parser once.  The tracker's backend is the source of truth —
    # a llama.cpp row never produces vLLM-shaped events and vice versa.
    parse_line = parse_vllm_line if tracker.backend != "llamacpp" else parse_llamacpp_line

    gave_up = False
    while True:
        try:
            now = clock()

            # --- read log → feed tracker ------------------------------------
            # ``read_new`` returns text since the last call ("" when there is
            # nothing new).  We split it into lines (tqdm ``\r``-separated and
            # ``(Worker…``-prefixed) and parse each line into events.
            chunk = follower.read_new()
            last_line = ""
            events: list[LogEvent] = []
            if chunk:
                for line in split_log_chunk(chunk):
                    events.extend(parse_line(line))
                    last_line = line
                tracker.feed(events, now, last_line)

            # --- sample vitals → feed tracker -------------------------------
            vitals = await sample()
            tracker.add_vitals(vitals)

            # --- evaluate policy → store verdict ----------------------------
            verdict = evaluate(tracker, clock(), policy)
            registry.set_verdict(tracker.model_id, verdict)

            # --- dispatch on verdict ----------------------------------------
            if verdict.state == "give_up" and not gave_up:
                gave_up = True
                # Re-check the generation BEFORE firing on_give_up: a newer
                # load that superseded this one owns the row now, and a
                # give-up write would clobber its state.  The watcher's
                # caller owns the generation check too (it stops the loop
                # below), but a give-up verdict can land on the same tick as
                # the supersession and the order matters.
                if not still_current():
                    return
                try:
                    await on_give_up(verdict)
                except Exception:  # noqa: BLE001 — total loop contract
                    logger.exception(
                        "load-progress on_give_up failed for %s",
                        tracker.model_id,
                    )
                return

            # --- terminal conditions ----------------------------------------
            # ``still_current`` false: a newer load has bumped the generation
            # or the engine was force-unloaded.  The watcher's verdict is
            # stale; drop it and return without writing anything.
            if not still_current():
                return

            # ``tracker.is_done``: the warmup step ended (``mark_done`` was
            # called).  The load succeeded; return so the caller can finish.
            # We rely on ``mark_done`` being called externally — the watcher
            # itself never marks the load done because it does not know
            # whether the warmup probe succeeded.
            if tracker.is_done():
                return
        except Exception:  # noqa: BLE001 — total loop contract
            logger.exception(
                "load-progress tick failed for %s; continuing",
                tracker.model_id,
            )

        await asyncio.sleep(interval_s)
