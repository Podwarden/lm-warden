"""Tests for the STRICT priority scheduler (``app.proxy.scheduler``,
introduced in S5 / closes #104).
"""

import asyncio

import pytest

from app.proxy.scheduler import PriorityScheduler

# These tests deliberately busy-wait on an internal queue-size invariant
# (sched._queue_size_for_test) so we can deterministically observe that a
# previous task has reached its enqueue point before launching the next.
# An asyncio.Event won't work here — we're waiting for the *internal*
# heap state of asyncio.PriorityQueue.put(), not for a signal the
# waiter coroutine can hand us. ASYNC110 is suppressed at the line of
# each loop with a short pointer back to this paragraph.


# ---------------------------------------------------------------------------
# Priority scheduler
# ---------------------------------------------------------------------------


async def test_scheduler_fast_path_when_idle():
    """First acquirer when nobody's holding goes through the fast path
    without touching the heap."""
    sched = PriorityScheduler()
    async with sched.acquire(priority=5):
        assert sched._queue_size_for_test() == 0


async def test_scheduler_strict_priority_ordering():
    """When multiple waiters are queued, the highest priority MUST be
    served first — even if a lower-priority waiter was enqueued earlier.

    Forced to a single slot (max_inflight=1) so the engine saturates after
    the holder and every other acquirer must queue — admission ordering
    only manifests under contention.
    """
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True  # tests drive the release manually

    # Holder enters first to force everyone else to queue.
    holder_done = asyncio.Event()
    holder_release_signal = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5):
            holder_done.set()
            await holder_release_signal.wait()

    holder_task = asyncio.create_task(holder())
    await holder_done.wait()  # holder is inside the CM, others will queue

    # Now enqueue 10x priority-0 BEFORE the single priority-9, so FIFO
    # would serve them first. STRICT priority must defeat FIFO here.
    served: list[int] = []

    async def waiter(prio: int):
        async with sched.acquire(priority=prio):
            served.append(prio)

    waiters = [asyncio.create_task(waiter(0)) for _ in range(10)]
    # Wait until they're all inside the queue (sched._queue_size becomes 10).
    while sched._queue_size_for_test() < 10:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)
    # Enqueue the priority-9 waiter LAST in time.
    high = asyncio.create_task(waiter(9))
    while sched._queue_size_for_test() < 11:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    # Release holder → next-highest-priority waiter (the priority-9) fires.
    holder_release_signal.set()
    await holder_task
    # Manually step the heap one slot.
    await sched._release_for_test()
    # Allow the woken priority-9 waiter to enter its CM and append.
    await high
    # Now drain the rest of the heap so the test cleans up.
    for _ in range(10):
        await sched._release_for_test()
    await asyncio.gather(*waiters)

    # The first served (after the seed holder) must be the priority-9.
    assert served[0] == 9, f"strict priority violated: served order = {served}"
    # The remaining 10 are all priority-0 (FIFO inside the tier — order
    # of enqueue is preserved, but we don't assert that here because the
    # tier-internal ordering is covered by the FIFO test below).
    assert served[1:] == [0] * 10


async def test_scheduler_fifo_within_same_priority_tier():
    """Same-priority waiters must be served in enqueue order."""
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True
    holder_done = asyncio.Event()
    holder_release_signal = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5):
            holder_done.set()
            await holder_release_signal.wait()

    holder_task = asyncio.create_task(holder())
    await holder_done.wait()

    served: list[str] = []

    async def waiter(label: str):
        async with sched.acquire(priority=3):
            served.append(label)

    labels = list("ABCDE")
    tasks = []
    for label in labels:
        t = asyncio.create_task(waiter(label))
        tasks.append(t)
        # Spin until this waiter has actually enqueued before launching
        # the next — otherwise the enqueue order is non-deterministic.
        while sched._queue_size_for_test() < len(tasks):  # noqa: ASYNC110 — see module docstring
            await asyncio.sleep(0)

    holder_release_signal.set()
    await holder_task
    for _ in range(len(labels)):
        await sched._release_for_test()
    await asyncio.gather(*tasks)

    assert served == labels


async def test_scheduler_cancelled_waiter_skipped():
    """A waiter that's cancelled mid-wait must not block the next live
    waiter from being woken — the release() loop drains tombstones at
    the queue head before signalling a real event."""
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True
    holder_done = asyncio.Event()
    holder_release_signal = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5):
            holder_done.set()
            await holder_release_signal.wait()

    holder_task = asyncio.create_task(holder())
    await holder_done.wait()

    served: list[str] = []

    async def cancellable_waiter():
        async with sched.acquire(priority=8):  # higher than other
            served.append("cancelled-should-never-append")

    async def live_waiter():
        async with sched.acquire(priority=2):
            served.append("live")

    c = asyncio.create_task(cancellable_waiter())
    while sched._queue_size_for_test() < 1:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)
    live = asyncio.create_task(live_waiter())
    while sched._queue_size_for_test() < 2:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    # Cancel the high-priority waiter before holder releases.
    c.cancel()
    with pytest.raises(asyncio.CancelledError):
        await c

    holder_release_signal.set()
    await holder_task
    # Step the scheduler — first release must skip the tombstone for the
    # cancelled priority-8 and hand the slot to the priority-2 live waiter.
    await sched._release_for_test()
    await live

    assert served == ["live"]


async def test_scheduler_clamps_priority_out_of_range():
    """A buggy caller passing priority=99 must not crash the scheduler —
    we clamp to 0..9 as defence-in-depth alongside the DB CHECK."""
    sched = PriorityScheduler()
    async with sched.acquire(priority=99):
        pass
    async with sched.acquire(priority=-5):
        pass


# ---------------------------------------------------------------------------
# Per-engine multi-slot admission (#173 part A)
# ---------------------------------------------------------------------------


async def test_scheduler_admits_up_to_max_inflight_concurrently():
    """With max_inflight=N, the first N acquirers proceed concurrently
    without ever touching the heap; only the (N+1)th queues. This is the
    core of #173: the proxy must let vLLM batch many requests at once
    instead of serialising them one-at-a-time."""
    sched = PriorityScheduler(max_inflight=3)
    sched._manual_release = True

    inside = asyncio.Event()
    release = asyncio.Event()
    entered = 0

    async def holder():
        nonlocal entered
        async with sched.acquire(priority=5):
            entered += 1
            if entered == 3:
                inside.set()
            await release.wait()

    holders = [asyncio.create_task(holder()) for _ in range(3)]
    await inside.wait()  # all 3 admitted simultaneously
    assert sched._inflight_for_test() == 3
    assert sched._queue_size_for_test() == 0  # none had to queue

    # A 4th acquirer must now queue — capacity is full.
    async def fourth():
        async with sched.acquire(priority=5):
            pass

    q = asyncio.create_task(fourth())
    while sched._queue_size_for_test() < 1:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)
    assert sched._queue_size_for_test() == 1

    # Release the holders; step the heap once to admit the queued 4th.
    release.set()
    await asyncio.gather(*holders)
    await sched._release_for_test()
    await q


async def test_scheduler_engines_are_isolated():
    """A saturated engine A must NOT block an acquirer bound for engine B —
    they are different GPUs with independent queues and counters."""
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True

    a_inside = asyncio.Event()
    a_release = asyncio.Event()

    async def hold_a():
        async with sched.acquire(priority=5, engine_key="engine-A"):
            a_inside.set()
            await a_release.wait()

    a_task = asyncio.create_task(hold_a())
    await a_inside.wait()  # engine-A is now at capacity (1/1)

    # engine-B must admit immediately via its own fast path despite A being full.
    b_served = asyncio.Event()

    async def hit_b():
        async with sched.acquire(priority=0, engine_key="engine-B"):
            b_served.set()

    b_task = asyncio.create_task(hit_b())
    await asyncio.wait_for(b_served.wait(), timeout=1.0)
    assert sched._inflight_for_test("engine-A") == 1
    await b_task

    a_release.set()
    await a_task


async def test_scheduler_inflight_bookkeeping_drops_idle_engines():
    """After every holder releases, the engine's bookkeeping is removed so a
    long-lived warden doesn't accumulate one dict entry per ever-seen
    engine."""
    sched = PriorityScheduler(max_inflight=2)
    async with sched.acquire(priority=5, engine_key="ephemeral"):
        assert sched._inflight_for_test("ephemeral") == 1
    assert sched._inflight_for_test("ephemeral") == 0
    assert sched._queue_size_for_test("ephemeral") == 0


def test_scheduler_max_inflight_from_env(monkeypatch):
    """VW_PROXY_MAX_INFLIGHT sets the per-engine cap; non-positive / garbage
    values fall back to the default (16) rather than dead-locking at zero."""
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "32")
    assert PriorityScheduler().max_inflight == 32
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "0")
    assert PriorityScheduler().max_inflight == 16
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "not-a-number")
    assert PriorityScheduler().max_inflight == 16
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert PriorityScheduler().max_inflight == 16
    # An explicit constructor arg always wins over the env.
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "32")
    assert PriorityScheduler(max_inflight=4).max_inflight == 4


# ---------------------------------------------------------------------------
# Admission-slot accounting under cancellation (#217)
# ---------------------------------------------------------------------------


async def test_scheduler_cancel_after_handoff_does_not_leak_slot():
    """#217 — a waiter cancelled AFTER a release handed it the slot but
    BEFORE it ran must give the slot back.

    The window is real and tiny: ``_release`` pops the waiter, calls
    ``event.set()`` and returns *without* decrementing ``_inflight``,
    trusting the woken waiter to release in its turn. If the client
    disconnects in between, ``event.wait()`` raises CancelledError even
    though the event is set, and the old code just tombstoned the waiter —
    the slot was gone for good.

    The test pins the window open deterministically rather than racing for
    it: ``_release`` never suspends (uncontended lock, ``get_nowait``,
    unbounded queue), so ``await sched._release_for_test()`` returns without
    ever yielding to the event loop. The woken waiter has therefore not been
    scheduled yet when we cancel it — exactly the interleaving above.
    """
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True  # tests drive the release manually

    holder_done = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5, engine_key="E"):
            holder_done.set()

    await asyncio.create_task(holder())
    await holder_done.wait()
    assert sched._inflight_for_test("E") == 1  # manual mode: still held

    async def waiter():
        async with sched.acquire(priority=5, engine_key="E"):
            pass  # must never be reached — we cancel before it wakes

    w = asyncio.create_task(waiter())
    while sched._queue_size_for_test("E") < 1:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    # Hand the slot to w. This does not yield, so w has NOT resumed yet.
    await sched._release_for_test("E")
    assert sched._inflight_for_test("E") == 1  # slot passed on, not freed

    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w

    # The cancelled waiter must have released the slot it was handed.
    assert sched._inflight_for_test("E") == 0, "slot leaked on cancel-after-handoff"
    assert sched._queue_size_for_test("E") == 0

    # And the engine must still be usable — the failure mode this guards
    # against is an engine that looks healthy but admits nobody.
    sched._manual_release = False
    admitted = False
    async with sched.acquire(priority=5, engine_key="E"):
        admitted = True
    assert admitted
    assert sched._inflight_for_test("E") == 0


async def test_scheduler_cancel_while_still_queued_does_not_disturb_accounting():
    """The other half of #217: a waiter cancelled while it is *still in the
    heap* was never handed anything, so it must NOT release a slot it does
    not own — doing so would over-decrement and let the engine admit more
    than max_inflight. The holder's own release is the only one that counts.
    """
    sched = PriorityScheduler(max_inflight=1)
    holder_inside = asyncio.Event()
    holder_release = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5, engine_key="E"):
            holder_inside.set()
            await holder_release.wait()

    holder_task = asyncio.create_task(holder())
    await holder_inside.wait()

    async def waiter():
        async with sched.acquire(priority=5, engine_key="E"):
            pass

    w = asyncio.create_task(waiter())
    while sched._queue_size_for_test("E") < 1:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w
    # Nothing was handed out, so the holder's slot is untouched.
    assert sched._inflight_for_test("E") == 1

    holder_release.set()
    await holder_task
    # The holder's release drains the tombstone and frees the last slot.
    assert sched._inflight_for_test("E") == 0
    assert sched._queue_size_for_test("E") == 0


async def test_scheduler_queue_drains_normally_without_cancellation():
    """No-regression guard for #217: with nobody cancelling, a full queue
    still drains through the automatic (non-manual) release path — every
    waiter is served exactly once, in strict priority order, and the
    in-flight counter lands back on zero.
    """
    sched = PriorityScheduler(max_inflight=1)
    holder_inside = asyncio.Event()
    holder_release = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5, engine_key="E"):
            holder_inside.set()
            await holder_release.wait()

    holder_task = asyncio.create_task(holder())
    await holder_inside.wait()

    served: list[int] = []

    async def waiter(prio: int):
        async with sched.acquire(priority=prio, engine_key="E"):
            served.append(prio)

    tasks = []
    for prio in (0, 1, 2, 3):
        tasks.append(asyncio.create_task(waiter(prio)))
        while sched._queue_size_for_test("E") < len(tasks):  # noqa: ASYNC110 — see module docstring
            await asyncio.sleep(0)

    holder_release.set()
    await holder_task
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)

    assert served == [3, 2, 1, 0]  # strict priority, high first
    assert sched._inflight_for_test("E") == 0
    assert sched._queue_size_for_test("E") == 0


@pytest.mark.parametrize("ticks", list(range(8)))
async def test_scheduler_inflight_returns_to_zero_for_any_cancel_tick(ticks):
    """The #217 invariant stated directly: whichever event-loop tick a
    request is cancelled on — still queued, mid-handoff, or already
    admitted — ``_inflight`` must come back to zero once everything settles.

    Sweeping ``ticks`` across the drain walks the cancellation through every
    stage of the handoff, including the one-tick window between
    ``event.set()`` and the waiter resuming. Before the fix at least one
    tick in this sweep leaves the counter stuck above zero.
    """
    sched = PriorityScheduler(max_inflight=1)
    hold = asyncio.Event()

    async def worker():
        async with sched.acquire(priority=3, engine_key="E"):
            await hold.wait()

    tasks = [asyncio.create_task(worker()) for _ in range(4)]
    # One takes the fast path; the rest pile into the heap.
    while sched._queue_size_for_test("E") < 3:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    victim = tasks[2]
    hold.set()  # the queue starts draining...
    for _ in range(ticks):  # ...and the victim dies `ticks` ticks into it
        await asyncio.sleep(0)
    victim.cancel()

    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True), timeout=1.0
    )
    # Let any shielded release (the cancel-after-handoff branch runs the
    # release in its own task) finish before we read the counter.
    for _ in range(4):
        await asyncio.sleep(0)

    assert sched._inflight_for_test("E") == 0, (
        f"slot leaked when cancelling {ticks} tick(s) into the drain; "
        f"task outcomes = {results}"
    )
    assert sched._queue_size_for_test("E") == 0


@pytest.mark.parametrize("release_site", ["holder-finally", "cancel-handler"])
async def test_scheduler_second_cancel_during_release_does_not_leak_slot(release_site):
    """#217, both release sites: a *second* cancellation delivered while
    ``_release`` is suspended must not abandon the release half-done.

    ``acquire`` gives a slot back in two places — the holder's ``finally``
    and the CancelledError handler's handoff branch — and both are entered
    while the task is already unwinding a cancel. ``_release`` awaits
    ``self._lock``, a single process-wide lock taken by every acquire and
    every release for every engine, so under the load this scheduler exists
    to handle that await really does suspend. Unshielded there, the second
    cancel loses the admission slot for good — #217 again, by the other path.

    The contention is forced rather than raced for: the test holds
    ``sched._lock`` itself, so ``_release`` is guaranteed to be parked on
    ``lock.acquire()`` when the second ``cancel()`` lands. Both parameters
    must pass; ``holder-finally`` is the one that regressed.
    """
    sched = PriorityScheduler(max_inflight=1)
    inside = asyncio.Event()
    hold = asyncio.Event()

    if release_site == "holder-finally":
        # The victim is an admitted holder, cancelled inside its `async
        # with` — the ordinary client-disconnect shape, cleaned up by the
        # `finally`.
        async def holder():
            async with sched.acquire(priority=5, engine_key="E"):
                inside.set()
                await hold.wait()

        victim = asyncio.create_task(holder())
        await inside.wait()
    else:
        # The victim is a queued waiter that _release just handed the slot
        # to and that dies before it can resume — cleaned up by acquire()'s
        # CancelledError handler. Manual-release mode keeps the first
        # holder's slot outstanding so the waiter has to queue.
        sched._manual_release = True

        async def holder():
            async with sched.acquire(priority=5, engine_key="E"):
                inside.set()

        await asyncio.create_task(holder())
        await inside.wait()

        async def waiter():
            async with sched.acquire(priority=5, engine_key="E"):
                pass  # never reached — we cancel before it wakes

        victim = asyncio.create_task(waiter())
        while sched._queue_size_for_test("E") < 1:  # noqa: ASYNC110 — see module docstring
            await asyncio.sleep(0)
        # Hands the slot over without yielding, so the victim owns a slot it
        # has not resumed to take.
        await sched._release_for_test("E")

    assert sched._inflight_for_test("E") == 1

    # Pin the lock so the victim's _release must park on lock.acquire().
    await sched._lock.acquire()
    victim.cancel()
    for _ in range(3):  # let the cancel reach _release and block on the lock
        await asyncio.sleep(0)
    victim.cancel()  # the second cancel, landing mid-release
    for _ in range(3):
        await asyncio.sleep(0)
    sched._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await victim
    for _ in range(6):  # let the shielded release task finish
        await asyncio.sleep(0)

    assert sched._inflight_for_test("E") == 0, (
        f"slot leaked: the second cancel abandoned the release at {release_site}"
    )
