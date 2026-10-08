"""In-process registry of live (detached) chat2 turns.

A turn no longer lives inside its HTTP response: `routes_turn` runs the whole
generation pipeline on a detached task (the *runner*) that appends every SSE
`data:` frame — the exact bytes `frame(...)` produced — to a `LiveTurn` entry
here. The HTTP response is merely one *subscriber* over those frames, so a
client disconnect kills only the subscriber while the runner streams on to
completion and persists the full assistant row. Reattaching (a reload, a
second tab) is just another subscriber: `subscribe()` replays `frames[0:]`
byte-identically and then follows new appends until the turn is done, so the
frontend reducer runs the exact same path it would have on the original
response.

Single-process limitation is ACCEPTED: the registry is a plain in-memory dict
keyed by chat id, exactly like `TurnLocks` and the other `app.state`
singletons — the warden runs one uvicorn worker. If the app ever scales
workers this (and the turn lock) must move to shared storage.

Lifecycle of an entry:

* created by `start()` before the runner is spawned — a newer turn for the
  same chat replaces the old entry immediately (the per-chat turn lock
  guarantees the old one is already done);
* `finish()` marks it done and schedules its removal `done_ttl_s` (60s)
  later, leaving a grace window in which a late `GET .../turn/live` still
  gets a full, instantly-terminating replay instead of a 404;
* `shutdown()` (app lifespan teardown) cancels the reaper tasks and any
  still-running runners, whose CancelledError path persists an 'aborted'
  tail before the loop dies.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from app.chat2.clock import now_iso

# How long a finished turn's frames stay replayable. Long enough to cover the
# reattach race (detail says live, the turn finishes before the live GET
# lands), short enough that frames of a chatty turn don't pile up.
DONE_TTL_S = 60.0


def _caught_up(live: LiveTurn, pos: int) -> bool:
    """Subscriber wake predicate: a frame beyond `pos` exists, or the turn is
    done. Module-level (bound via `partial`) so each iteration's cursor is
    captured by value, not by closure over a loop variable."""
    return live.done or len(live.frames) > pos


@dataclass
class LiveTurn:
    """One detached turn: its identity, its frames, and its runner task."""

    chat_id: str
    request_id: str
    message_id: str
    frames: list[bytes] = field(default_factory=list)
    new_frame: asyncio.Condition = field(default_factory=asyncio.Condition)
    done: bool = False
    started_at: str = field(default_factory=now_iso)
    # The runner task, set right after it is spawned; the abort endpoint and
    # the lifespan shutdown cancel the turn through it.
    task: asyncio.Task[Any] | None = None


class LiveTurns:
    """Registry of live turns keyed by chat id (one live turn per chat)."""

    def __init__(self, done_ttl_s: float = DONE_TTL_S) -> None:
        self._turns: dict[str, LiveTurn] = {}
        self._reapers: set[asyncio.Task[None]] = set()
        self._done_ttl_s = done_ttl_s

    def start(self, chat_id: str, *, request_id: str, message_id: str) -> LiveTurn:
        """Register a new live turn, displacing any previous entry for the chat.

        The per-chat turn lock is already held when this runs, so a displaced
        entry is always a *finished* turn merely waiting out its replay TTL.
        """
        live = LiveTurn(chat_id=chat_id, request_id=request_id, message_id=message_id)
        self._turns[chat_id] = live
        return live

    def get(self, chat_id: str) -> LiveTurn | None:
        return self._turns.get(chat_id)

    async def append(self, live: LiveTurn, data: bytes) -> None:
        """Append one already-encoded SSE frame and wake every subscriber."""
        live.frames.append(data)
        async with live.new_frame:
            live.new_frame.notify_all()

    async def finish(self, live: LiveTurn) -> None:
        """Mark the turn done, release subscribers, schedule the reap."""
        live.done = True
        async with live.new_frame:
            live.new_frame.notify_all()
        # asyncio holds only weak refs to scheduled tasks — keep a strong one.
        task = asyncio.ensure_future(self._reap(live))
        self._reapers.add(task)
        task.add_done_callback(self._reapers.discard)

    async def _reap(self, live: LiveTurn) -> None:
        await asyncio.sleep(self._done_ttl_s)
        if self._turns.get(live.chat_id) is live:
            del self._turns[live.chat_id]

    async def subscribe(self, live: LiveTurn) -> AsyncIterator[bytes]:
        """Replay `frames[0:]`, then follow appends until the turn is done.

        Multiple concurrent subscribers are fine — each keeps its own cursor
        and the replay is byte-identical to what the original response saw.
        Cancelling a subscriber (client disconnect) cancels only this
        generator; the runner and the other subscribers are untouched.
        """
        i = 0
        while True:
            while i < len(live.frames):
                yield live.frames[i]
                i += 1
            if live.done:
                return
            async with live.new_frame:
                # the predicate binds THIS iteration's cursor (B023) and is
                # re-evaluated under the lock, closing the check-then-wait race
                await live.new_frame.wait_for(partial(_caught_up, live, i))

    async def shutdown(self) -> None:
        """Lifespan teardown: stop reapers, cancel runners, wait them out.

        Cancelling a runner drives its CancelledError path, which persists the
        partial tail as 'aborted' (shielded, so this gather really waits for
        the persist) and releases the chat's turn lock.
        """
        for reaper in list(self._reapers):
            reaper.cancel()
        runners = [
            lv.task for lv in self._turns.values() if lv.task is not None and not lv.task.done()
        ]
        for runner in runners:
            runner.cancel()
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)
