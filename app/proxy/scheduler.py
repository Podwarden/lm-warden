"""STRICT priority scheduler.

This module sits between the proxy `require_bearer` step and the upstream
vLLM forward.

``PriorityScheduler`` is a *per-engine, multi-slot* admission gate in
front of each vLLM engine. Each engine (keyed by an opaque string the
proxy supplies — the model id) admits up to ``VW_PROXY_MAX_INFLIGHT``
requests concurrently (default 16); vLLM's own continuous batching
does the real work once admitted, so the cap exists only to bound the
queue depth we hand the engine, not to serialise. (#173: the original
design was a single global slot — ONE request talked to vLLM at a
time, held for the whole SSE stream — which throttled the product path
to ~40 tok/s while the engines could sustain >1000 tok/s aggregate.)

Admission is STRICT-priority *ordered*: when an engine is at capacity,
waiters queue in a per-engine heap ordered first by priority (high →
low), then by enqueue time. Priority-9 is admitted before any waiting
priority-0..8; heavy priority-9 traffic CAN starve lower priorities
indefinitely. That trade-off is intentional and locked by CTO decision
#3 of the 2026-05 overhaul plan — operators surface the risk in the UI
(see tooltip on the Priority column). Priority is *also* pushed down
into vLLM itself via ``--scheduling-policy priority`` + a per-request
``priority`` field (#173 part B, see app/proxy/routes.py), so ordering
is honoured by the engine's batch scheduler once many requests are
admitted concurrently — not just at our admission boundary.

Engines are independent (separate GPUs): a saturated engine A never
blocks a request bound for an idle engine B — they have separate
queues and in-flight counters.

Priority is the only per-token fairness control. (A per-token
sliding-window rate limiter lived here until 2026-09; it charged only the
prompt-token estimate and permanently rejected any prompt larger than its
window budget, so it was removed.)

The scheduler is async-safe but NOT process-safe — the warden runs a
single uvicorn worker per pod (app/main.py), so a single in-memory
instance per app is sufficient. If we ever scale to multiple workers
we'd need a Redis-backed implementation; that's deferred.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Priority scheduler
# ---------------------------------------------------------------------------


def _default_max_inflight() -> int:
    """Per-engine concurrency cap for the priority scheduler (#173).

    Defaults to 16 — comfortably inside the per-engine sweet spot we measured
    on 4×A4000 (throughput keeps climbing through concurrency 16 with no error
    rate), while still bounding the queue depth handed to any one engine.
    ``VW_PROXY_MAX_INFLIGHT`` overrides it; values <= 0 fall back to the
    default rather than dead-locking every engine at zero admissions.
    """
    try:
        v = int(os.environ.get("VW_PROXY_MAX_INFLIGHT", "16"))
        return v if v > 0 else 16
    except ValueError:
        return 16


@dataclass(order=True)
class _Waiter:
    """One queued request waiting for the scheduler slot.

    The ``sort_index`` tuple gives the strict ordering:
      - ``-priority`` so HIGHER priority sorts FIRST,
      - ``enqueue_seq`` (a monotonically increasing counter, ints only)
        breaks ties so same-priority requests are FIFO inside their tier.
    asyncio.PriorityQueue is heap-backed; the heap invariant only needs
    a < comparison on the tuple, so the ``Event`` is excluded via
    ``compare=False`` (otherwise Event has no __lt__).

    ``cancelled`` is the tombstone flag: a waiter whose client disconnected
    while it was still queued cannot be removed from a PriorityQueue without
    rebuilding the heap, so it stays in place and ``_release`` drains it.
    It is a real field (rather than an attribute stuck on from the outside)
    so both the setter in ``acquire`` and the reader in ``_release`` are
    type-checked — #217 was a bug in exactly that handshake.
    """

    sort_index: tuple[int, int]
    event: asyncio.Event = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


_DEFAULT_ENGINE_KEY = "__default__"


class PriorityScheduler:
    """Per-engine, multi-slot priority admission gate in front of vLLM.

    Each engine (keyed by an opaque ``engine_key`` the caller supplies —
    in production the model id) admits up to ``max_inflight`` requests
    concurrently. When an engine is at capacity, further acquirers wait in
    that engine's heap ordered by priority (9 first), FIFO inside a tier.

    Priority 9 is highest. There is NO aging or anti-starvation — priority-0
    traffic CAN wait indefinitely behind a hot priority-9 client. Document
    this in the UI (Priority column tooltip) and in docs/operating.md.

    Usage:

        async with scheduler.acquire(priority=token.priority, engine_key=model.id):
            return await forward_to_vllm(...)

    ``engine_key`` defaults to a shared key so callers (and tests) that don't
    distinguish engines still get a single shared pool. A cancelled waiter
    (client disconnect) is tombstoned and skipped on the next release — see
    acquire()'s CancelledError handler and ``_release``.

    The invariant the whole class rests on: **``_inflight[engine_key]``
    returns to zero once every request against that engine has finished or
    been cancelled, under any interleaving.** Every admission has exactly one
    matching release — the holder's `finally`, or, for a request cancelled in
    the window between being handed a slot and taking it, the CancelledError
    handler in ``acquire`` (#217). BOTH of those release sites run
    ``_release`` under ``asyncio.shield``, because both can be entered while
    the task is already unwinding a cancellation and ``_release`` awaits a
    process-wide lock: unshielded, a second cancellation delivered while it
    waited on that lock would abandon the release half-done. "Under any
    interleaving" is only true because of those two shields; drop either and
    the sentence becomes a lie. Break the invariant and the engine silently
    stops admitting: waiters block forever in an untimed wait and the model
    looks healthy from every angle except the proxy path.
    """

    def __init__(self, max_inflight: int | None = None) -> None:
        self._max_inflight = (
            max_inflight if max_inflight is not None else _default_max_inflight()
        )
        # Per-engine admission heap. asyncio.PriorityQueue's heap-pop is
        # O(log n) and async-safe. Created lazily per engine_key.
        self._queues: dict[str, asyncio.PriorityQueue[_Waiter]] = {}
        # Per-engine count of requests currently admitted ("in flight").
        self._inflight: dict[str, int] = {}
        # One lock guards all the bookkeeping dicts. The critical sections
        # are tiny dict ops; contention between engines is negligible and a
        # single lock keeps the admit/release invariants trivially correct.
        self._lock = asyncio.Lock()
        # Monotonically increasing — used for FIFO tie-break inside a tier.
        self._seq = 0
        # Toggle for tests: when True, acquire() does NOT release the slot
        # automatically on context-manager exit; the test must call
        # ``_release_for_test`` to advance the queue. Operations code never
        # touches this.
        self._manual_release = False

    @property
    def max_inflight(self) -> int:
        return self._max_inflight

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _queue_for(self, engine_key: str) -> asyncio.PriorityQueue[_Waiter]:
        q = self._queues.get(engine_key)
        if q is None:
            q = asyncio.PriorityQueue()
            self._queues[engine_key] = q
        return q

    @asynccontextmanager
    async def acquire(
        self, priority: int, engine_key: str = _DEFAULT_ENGINE_KEY
    ) -> AsyncIterator[None]:
        """Acquire an admission slot for ``engine_key``, ordered by priority.

        Fast path — if the engine is below ``max_inflight`` and nobody is
        already queued for it, admit immediately without touching the heap
        (most production traffic IS idle most of the time). Otherwise enqueue
        in the engine's heap and wait for a holder to release.
        """
        # Sanity-clamp priority to 0..9 to match the DB CHECK. We trust
        # the DB but defending in depth costs nothing here.
        prio = max(0, min(9, int(priority)))

        # Decide fast-path vs enqueue atomically under the lock so two
        # acquirers can't both observe spare capacity and overshoot it, and
        # so we never jump the queue ahead of an already-waiting request.
        async with self._lock:
            inflight = self._inflight.get(engine_key, 0)
            q = self._queues.get(engine_key)
            queue_empty = q is None or q.empty()
            if inflight < self._max_inflight and queue_empty:
                self._inflight[engine_key] = inflight + 1
                fast_path = True
            else:
                fast_path = False
                waiter = _Waiter(
                    sort_index=(-prio, self._next_seq()),
                    event=asyncio.Event(),
                )
                await self._queue_for(engine_key).put(waiter)

        if not fast_path:
            try:
                # No timeout on this wait, deliberately: a queued request
                # waits exactly as long as its client is willing to, and the
                # client (or uvicorn, on disconnect) cancels us when it gives
                # up. That is only safe because of the invariant stated in
                # the class docstring — that _inflight always comes back
                # down, under any interleaving. If a slot could leak,
                # this wait would be the place it manifests: the engine sits
                # permanently at max_inflight, nobody ever releases, and every
                # later request for that engine_key blocks here forever
                # instead of getting an error. Do not add a timeout here
                # without also deciding what the client should be told; do not
                # break the invariant and expect this wait to save you.
                await waiter.event.wait()
            except asyncio.CancelledError:
                # Client disconnected while queued (routine under load —
                # app/proxy/routes.py drives this context manager by hand
                # around the whole upstream call, so a disconnect lands
                # right here). Which cleanup we owe depends on whether a
                # releaser had already handed us the slot, and the event is
                # the record of that: _release pops a waiter and calls
                # event.set() under the bookkeeping lock, *without*
                # decrementing _inflight, on the assumption that the woken
                # waiter will release in its turn.
                #
                # #217: that assumption breaks in one window. If the
                # cancellation is delivered after event.set() but before this
                # coroutine is rescheduled, event.wait() raises CancelledError
                # even though the event is set — so we were handed a slot we
                # will never enter the `yield` for. Nobody else holds it and
                # nobody else can see it, so it leaks: _inflight stays one too
                # high forever. VW_PROXY_MAX_INFLIGHT (default 16) such
                # disconnects on one engine_key and that engine's proxy path
                # is dead while the model still looks perfectly healthy.
                #
                # The waiter is the only party with the information needed to
                # tell the two cases apart at the moment of cancellation, so
                # the fix lives here rather than in _release: a releaser
                # cannot observe from inside its own critical section whether
                # the coroutine it just woke ever resumes without waiting for
                # it, which would serialise release behind waiter scheduling —
                # exactly the one-at-a-time behaviour #173 removed.
                #
                # There must be no await between the is_set() check and the
                # tombstone write: the loop is single-threaded, so with no
                # suspension point in between no releaser can slip in and set
                # the event after we decided it was unset.
                if waiter.event.is_set():
                    # We own a slot. Hand it on exactly as a normal holder's
                    # `finally` would (which is shielded for the same reason
                    # — see below). shield() because we are already unwinding
                    # a cancellation: if a second cancel were delivered while
                    # _release waited on a contended lock, an unshielded await
                    # would abandon the release half-done and leak the very
                    # slot we are here to save.
                    #
                    # Note this deliberately ignores _manual_release, unlike
                    # the `finally`: that flag is a test knob for stepping the
                    # heap by hand and is always False in production, and a
                    # slot handed to a waiter that then died is owed back
                    # regardless of who is driving the releases.
                    await asyncio.shield(self._release(engine_key))
                else:
                    # Still in the heap; no slot was ever ours, so there is
                    # nothing to give back. Tombstone ourselves so the next
                    # _release skips us (we cannot delete from a
                    # PriorityQueue without rebuilding the heap) and set the
                    # event so any releaser that reaches us finds a settled
                    # waiter rather than one that might still wake up.
                    waiter.cancelled = True
                    waiter.event.set()
                raise

        try:
            yield
        finally:
            if not self._manual_release:
                # shield() for exactly the reason spelled out in the
                # CancelledError handler above (#217) — this is acquire()'s
                # other release site and it needs the same protection. This
                # `finally` very often runs *because* the task was cancelled
                # (app/proxy/routes.py drives __aexit__ by hand once the
                # upstream call ends or blows up), and self._lock is a single
                # process-wide lock taken by every acquire and every release
                # for every engine, so `await self._release(...)` genuinely
                # suspends under the load this scheduler exists to handle.
                # Unshielded, a second cancellation landing in that suspension
                # abandons the release half-done: one admission slot lost
                # permanently, VW_PROXY_MAX_INFLIGHT (default 16) of them and
                # the engine's proxy path is dead while the model looks
                # healthy — the identical failure mode to #217 itself. Double
                # cancellation is not exotic: uvicorn cancels the request task
                # on client disconnect and shutdown cancels the survivors
                # again.
                await asyncio.shield(self._release(engine_key))

    async def _release(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> None:
        """Hand this engine's freed slot to its next waiter, or free it.

        Skips tombstoned (cancelled) waiters at the head. If a real waiter
        takes the slot, the in-flight count is unchanged (one holder swapped
        for another); otherwise the count is decremented and empty
        bookkeeping is dropped so a long-lived warden doesn't accumulate one
        dict entry per ever-seen engine.

        The handoff is optimistic: we set the woken waiter's event and return,
        trusting it to release in its turn. A waiter cancelled inside that
        window owes us the slot back and pays it in acquire()'s
        CancelledError handler (#217) — the two halves have to be read
        together or the in-flight accounting doesn't add up.
        """
        async with self._lock:
            q = self._queues.get(engine_key)
            while q is not None and not q.empty():
                nxt = q.get_nowait()
                if nxt.cancelled:
                    continue
                # Hand the slot off; in-flight count stays the same until
                # the new holder exits its `async with` and releases in turn.
                nxt.event.set()
                return
            cur = self._inflight.get(engine_key, 0)
            if cur > 0:
                cur -= 1
            if cur <= 0:
                self._inflight.pop(engine_key, None)
                if q is not None and q.empty():
                    self._queues.pop(engine_key, None)
            else:
                self._inflight[engine_key] = cur

    # ----- Test helpers (NOT for production code) ----------------------

    async def _release_for_test(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> None:
        """Public-for-tests release. The test sets _manual_release=True,
        runs N acquirers (each of which sleeps in their `async with`),
        then calls this once per acquirer to step the heap. Lets the
        scheduler ordering test pin down "9 fires before 0" without
        having to control coroutine scheduling via asyncio.sleep.
        """
        await self._release(engine_key)

    def _queue_size_for_test(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> int:
        q = self._queues.get(engine_key)
        return q.qsize() if q is not None else 0

    def _inflight_for_test(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> int:
        return self._inflight.get(engine_key, 0)
