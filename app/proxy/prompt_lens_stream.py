"""The Prompt Lens SSE generator, shared by the admin stream
(``GET /api/admin/prompt-lens/stream``, app/proxy/routes_prompt_lens.py) and the key
holder's own stream (``GET /api/me/prompt-lens/stream``,
app/tokens/routes_me_prompt_lens.py).

One connection:

1. ``: connected`` (#251), so EventSource's ``open`` fires at once;
2. with ``ring_meta`` (admin only): a ``{"type": "ring", ...}`` frame, what the
   replay ring holds (``PromptLensHub.ring_span``) as the replay starts, then
   again every ``RING_INTERVAL_S``. A key holder never gets it: its counts are
   every key's;
3. the replay, passed through ``transform``. It opens with the start (and
   kvsig) of every request still live whose own start already left the
   replay, as copies marked ``restored: true`` (``PromptLensHub.restore`` /
   ``key_restore``: for a key holder, its own live requests from its own
   window, in its view). Then the ring snapshot filtered by
   ``wanted`` -- or, with ``key_id`` (the key holder), the key's OWN replay
   window (``PromptLensHub.subscribe_key``), whose depth depends on the key's own
   traffic only (review I1);
4. live events from the hub queue, the same filter and transform. The queue is
   subscribed with ``wanted``, so another key's traffic never reaches it.

With ``key_id`` every data frame carries ``seq``, numbered per connection
from 1 (review M1: no number shared across keys or connections, so it says
nothing about other keys and never repeats within a connection). A client
that reconnects gets the replay renumbered from 1: it rebuilds its view from
the replay, and tells requests apart by ``req_id``.

* ``: keepalive`` after ``routes_prompt_lens.KEEPALIVE_INTERVAL_S`` without a frame
  this client received. The wait is the time remaining, so a filtered stream
  keeps its cadence (#251). The interval is read from ``routes_prompt_lens`` at
  call time: tests patch it there.
* ``{"type": "lagged"}`` once per gap, before the first event after it, when
  the client fell behind its bounded queue and the hub dropped its oldest
  events (ruling G-F15). Memory per connection stays the queue's bound.
* ``recheck`` (the key holder's stream): run every
  ``min(KEEPALIVE_INTERVAL_S, RECHECK_INTERVAL_S)`` on its own timer task,
  independent of the keepalive, so a busy stream is re-checked too (ruling
  G-F10). False -- or a raise: it fails closed, the client mints a fresh
  ticket and the mint checks again -- ends the stream.
* Registered under ``user`` in the stream registry, so ``cancel_user(user)``
  ends it at once. Without ``recheck`` (the admin stream) that is
  ``guard_stream``, as before (it also re-checks an admin token). With it, the
  task registered is a stand-in whose cancellation, like a failed re-check,
  puts a stop mark on the stream's own queue: the generator then returns by
  itself, so the response completes. Cancelling the streaming task itself can
  land in the middle of a ``send`` -- likely on a busy stream -- outside the
  generator, and then the response never completes. The stand-in is
  registered when ``prompt_lens_sse`` is called, with no await before it, so a
  caller's count of open streams (the key holder's cap) cannot race.
* A response whose generator never starts (the client left first) gives back
  its subscription and registration after ``START_GRACE_S``.

Nothing here logs an event, a ticket or any content.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from app.auth.stream_guard import guard_stream
from app.proxy.prompt_lens import STOP, PromptLensHub, event_matches
from app.proxy.prompt_lens_scope import admin_view

__all__ = ["admin_view", "prompt_lens_sse"]
from app.utils.sse import sse_headers

#: How often an admin stream repeats its ``ring`` frame.
RING_INTERVAL_S: float = 30.0
#: The longest a re-checked stream goes between re-checks (the keepalive
#: interval, when shorter, is used instead). Read at every wait.
RECHECK_INTERVAL_S: float = 5.0
#: A response whose generator has not started by then is abandoned.
START_GRACE_S: float = 30.0

LAGGED_FRAME = f"data: {json.dumps({'type': 'lagged'})}\n\n"


def _keepalive_s() -> float:
    from app.proxy import routes_prompt_lens

    return float(routes_prompt_lens.KEEPALIVE_INTERVAL_S)


def _recheck_s() -> float:
    return min(_keepalive_s(), RECHECK_INTERVAL_S)


#: Put on a stream's queue to end it (``_KeyStop``; ``PromptLensHub.drop_all``).
_STOP: Any = STOP


class _KeyStop:
    """Ends a re-checked stream from outside it without cancelling its task
    (module docstring): a stop mark on its queue, and a flag for the replay."""

    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self.queue = queue
        self.stopped = False
        self._registry: Any = None
        self._user = ""
        self._standin: asyncio.Task[Any] | None = None
        self._timer: asyncio.Task[None] | None = None

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            self.queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            # Make room: the stream ends, so the event dropped is never sent.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(_STOP)
            except asyncio.QueueFull:  # pragma: no cover -- single-threaded loop
                pass

    async def _recheck(self, recheck: Callable[[], Awaitable[bool]]) -> None:
        while not self.stopped:
            await asyncio.sleep(_recheck_s())
            try:
                ok = await recheck()
            except Exception:  # noqa: BLE001 -- fail closed
                ok = False
            if not ok:
                self.stop()
                return

    def start(self, registry: Any, user: str, recheck: Callable[[], Awaitable[bool]]) -> None:
        """Register the stand-in under ``user`` and start the re-check timer.
        Synchronous: the caller's stream count sees this stream at once."""
        self._registry, self._user = registry, user
        # The registry's stand-in: cancel_user(user) cancels it, which stops us.
        self._standin = asyncio.create_task(asyncio.Event().wait())
        self._standin.add_done_callback(lambda _t: self.stop())
        registry.register(user, self._standin)
        self._timer = asyncio.create_task(self._recheck(recheck))

    def close(self) -> None:
        """Unregister and stop the timer (idempotent)."""
        if self._timer is not None:
            self._timer.cancel()
        if self._standin is not None:
            self._registry.unregister(self._user, self._standin)
            self._standin.cancel()

    @asynccontextmanager
    async def closing(self) -> AsyncIterator[None]:
        try:
            yield
        finally:
            self.close()


def _data(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _ring(span: dict[str, Any]) -> str:
    return _data({"type": "ring", **span})


async def prompt_lens_sse(
    request: Request,
    hub: PromptLensHub,
    *,
    user: str,
    wanted: frozenset[str] | None,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
    ring_meta: bool,
    recheck: Callable[[], Awaitable[bool]] | None,
    key_id: str | None = None,
) -> StreamingResponse:
    """The stream (module docstring). Subscribes (and, with ``recheck``,
    registers) now, so an event published between this call and the first
    read is not lost."""
    from app.proxy.routes_prompt_lens import CONNECTED_FRAME

    if key_id is not None:
        queue, snapshot = hub.subscribe_key(key_id)
        wanted = frozenset({key_id})
        restored = hub.key_restore(key_id, snapshot)
    else:
        queue, snapshot = hub.subscribe(token_ids=wanted)
        restored = hub.restore(snapshot, wanted)
    # Turned off after this point (``drop_all`` bumps the epoch): the snapshot
    # taken here must not go out.
    epoch = getattr(hub, "epoch", 0)

    def dropped() -> bool:
        return getattr(hub, "epoch", 0) != epoch

    # Live requests whose start already left the replay: the start (and
    # kvsig) go first, marked restored, numbered like any replayed frame.
    snapshot = [*restored, *snapshot]
    span = hub.ring_span() if ring_meta else None
    stopper = _KeyStop(queue) if recheck is not None else None
    if stopper is not None and recheck is not None:
        stopper.start(request.app.state.stream_registry, user, recheck)
    started = False
    given_up = False

    def abandoned() -> None:
        nonlocal given_up
        if not started:
            given_up = True
            hub.unsubscribe(queue)
            if stopper is not None:
                stopper.close()

    grace = asyncio.get_running_loop().call_later(START_GRACE_S, abandoned)
    numbered = 0

    def frame(event: dict[str, Any]) -> str:
        nonlocal numbered
        out = transform(event)
        if key_id is not None:
            numbered += 1
            out = {**out, "seq": numbered}
        return _data(out)

    async def gen() -> AsyncIterator[str]:
        nonlocal started
        if given_up:
            # Too late (review N3): the slot, the registration and the
            # re-check are gone, so a stream now would escape the cap and a
            # revoke. Send nothing.
            return
        started = True
        grace.cancel()
        try:
            guard = stopper.closing() if stopper is not None else guard_stream(request, user)
            async with guard:
                yield CONNECTED_FRAME
                if span is not None:
                    yield _ring(span)
                ring_due = time.monotonic() + RING_INTERVAL_S
                for event in snapshot:
                    if (stopper is not None and stopper.stopped) or dropped():
                        return
                    if event_matches(event, wanted):
                        yield frame(event)
                keepalive_due = time.monotonic() + _keepalive_s()
                while True:
                    if stopper is not None and stopper.stopped:
                        return
                    if await request.is_disconnected():
                        return
                    now = time.monotonic()
                    if ring_meta and now >= ring_due:
                        yield _ring(hub.ring_span())
                        ring_due = time.monotonic() + RING_INTERVAL_S
                        keepalive_due = time.monotonic() + _keepalive_s()
                        continue
                    due = min(keepalive_due, ring_due) if ring_meta else keepalive_due
                    remaining = due - now
                    try:
                        if remaining <= 0:
                            raise TimeoutError
                        event = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except TimeoutError:
                        if time.monotonic() >= keepalive_due:
                            yield ": keepalive\n\n"
                            keepalive_due = time.monotonic() + _keepalive_s()
                        continue
                    if event is _STOP:
                        return
                    if hub.take_lagged(queue):
                        yield LAGGED_FRAME
                    if event_matches(event, wanted):
                        yield frame(event)
                        keepalive_due = time.monotonic() + _keepalive_s()
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers())
