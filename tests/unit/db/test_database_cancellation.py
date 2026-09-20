"""#260: ``open_db`` must never abandon an aiosqlite connection thread.

``aiosqlite.Connection`` is a ``threading.Thread`` created with
``daemon=False`` that parks on a queue until ``close()`` enqueues its stop
sentinel. Drop one without closing it and CPython's interpreter shutdown joins
a thread that never finishes: the process hangs *after* all its work is done.
The symptom was the unit suite hanging after pytest had already printed
``N passed`` (and, in #43, a CI container wedging until the runner killed it).

The leak needs a cancellation delivered *inside* the connect, because that is
the one case where ``aiosqlite``'s own ``async with`` cleanup cannot run:
``__aenter__`` never returned, so ``__aexit__`` is never called, and the
connection object the caller would have had to close was never handed over.

"Never" has two deliberate exceptions, both bounded and both tested here: a
caller that cancels the same open over and over, and a connect that has not
returned by ``_CONNECT_CLEANUP_TIMEOUT`` after being cancelled. Each abandons
one *daemon* thread and logs it, because the alternative -- an uncancellable
await -- hangs interpreter shutdown, which is the bug itself.
"""

import asyncio
import logging
import sqlite3
import threading

import aiosqlite
import pytest

from app.db import database
from app.db.database import open_db


def _connection_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if isinstance(t, aiosqlite.Connection)}


def _assert_nothing_leaked(before: set[threading.Thread]) -> None:
    """Every connection thread this test started must have finished."""
    new = _connection_threads() - before
    for thread in new:
        # A closed connection breaks out of its queue loop as soon as it is
        # scheduled, so this join returns immediately. The timeout only bounds
        # the failure case: a leaked thread is parked forever and never joins.
        thread.join(10)
    alive = [t for t in new if t.is_alive()]
    assert not alive, f"open_db leaked {len(alive)} aiosqlite connection thread(s): {alive}"


def _install_gated_connect(
    monkeypatch: pytest.MonkeyPatch,
    gate: threading.Event,
    *,
    error: BaseException | None = None,
) -> None:
    """Hold the connect open in its worker thread until the test releases it.

    This is how a cancellation is landed *inside* the connect deterministically:
    the connection's thread is started and blocked in ``sqlite3.connect``, so a
    ``Task.cancel()`` at that moment lands on the ``await`` that ``open_db``
    does for the connect, exactly as the watchdog's background passes see it
    when the lifespan tears them down. With ``error``, the connect fails once
    released instead of succeeding.
    """

    def connect(path: object, **kwargs: object) -> aiosqlite.Connection:
        def connector() -> sqlite3.Connection:
            # Bounded, so a mistake in the test fails it instead of hanging it.
            gate.wait(30)
            if error is not None:
                raise error
            return sqlite3.connect(str(path), **kwargs)  # type: ignore[arg-type]

        return aiosqlite.Connection(connector, iter_chunk_size=64)

    monkeypatch.setattr(database.aiosqlite, "connect", connect)


@pytest.fixture
def gated_connect(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    gate = threading.Event()
    _install_gated_connect(monkeypatch, gate)
    return gate


async def _parked_in_connect(path, captured: dict | None = None) -> asyncio.Task:
    """A task suspended on ``open_db``'s await for a gated connect.

    ``entered`` is set immediately before ``open_db`` and nothing between it and
    the connect's ``await`` yields, so the task is parked on that await by the
    time this returns.
    """
    entered = asyncio.Event()

    async def use_db() -> None:
        entered.set()
        try:
            async with open_db(path):
                pytest.fail("the gated connect must not have completed for this task")
        except BaseException as exc:  # noqa: B036 - re-raised; the test wants the instance
            if captured is not None:
                captured["exc"] = exc
            raise

    task = asyncio.create_task(use_db())
    await entered.wait()
    return task


def _leaked_daemon(before: set[threading.Thread]) -> threading.Thread:
    """The one connection thread ``open_db`` gave up on, asserted to be a daemon.

    Abandoning it is the documented lesser evil (see ``app/db/database.py``); the
    point of the assertion is that it cannot block interpreter shutdown. Stopped
    here so the test does not leave it parked for the rest of the session.
    """
    new = _connection_threads() - before
    assert len(new) == 1, f"expected exactly one abandoned thread, got {new}"
    leaked = next(iter(new))
    assert leaked.daemon is True, "an abandoned connection thread must be a daemon"
    leaked._stop_running()  # type: ignore[attr-defined]
    leaked.join(10)
    return leaked


async def test_cancel_during_connect_leaves_no_thread_behind(tmp_data_dir, gated_connect):
    """The #260 leak: cancelled inside the connect, ``__aexit__`` never runs."""
    before = _connection_threads()
    entered = asyncio.Event()

    async def use_db() -> None:
        entered.set()
        async with open_db(tmp_data_dir / "vllm-warden.db"):
            pytest.fail("the gated connect must not have completed for this task")

    task = asyncio.create_task(use_db())
    # ``entered`` is set immediately before ``open_db``, and nothing between it
    # and the connect's ``await`` yields, so the task is parked on that await by
    # the time this resumes.
    await entered.wait()
    task.cancel()
    gated_connect.set()  # let the connect finish, with the task already cancelled

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_nothing_leaked(before)


async def test_cancel_inside_the_body_leaves_no_thread_behind(tmp_data_dir):
    before = _connection_threads()
    in_body = asyncio.Event()

    async def use_db() -> None:
        async with open_db(tmp_data_dir / "vllm-warden.db") as db:
            await db.execute("SELECT 1")
            in_body.set()
            await asyncio.Event().wait()  # park until cancelled

    task = asyncio.create_task(use_db())
    await in_body.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_nothing_leaked(before)


async def test_body_error_closes_the_connection_and_propagates(tmp_data_dir):
    before = _connection_threads()

    with pytest.raises(RuntimeError, match="boom"):
        async with open_db(tmp_data_dir / "vllm-warden.db") as db:
            await db.execute("SELECT 1")
            raise RuntimeError("boom")

    _assert_nothing_leaked(before)


async def test_failing_close_does_not_mask_the_body_error(tmp_data_dir):
    """A close that raises must not replace what we were unwinding from."""
    before = _connection_threads()

    with pytest.raises(RuntimeError, match="boom"):
        async with open_db(tmp_data_dir / "vllm-warden.db") as db:
            real_close = db.close

            async def close_then_fail() -> None:
                await real_close()
                raise sqlite3.OperationalError("close failed")

            db.close = close_then_fail  # type: ignore[method-assign]
            raise RuntimeError("boom")

    _assert_nothing_leaked(before)


async def test_cancel_during_the_close_still_closes_and_stops_the_thread(tmp_data_dir):
    """The close is deliberately unshielded, so this is the state that justifies
    it: a second cancellation landing while the close is in flight.

    ``Connection.close()`` queues ``sqlite3.Connection.close`` on the worker
    before it awaits, and stops the worker from a ``finally``, so the connection
    is closed and the thread stopped even though the await never resumes. That
    is why shielding it bought nothing but an uncancellable wait behind whatever
    statement was already in flight.
    """
    before = _connection_threads()
    in_body = asyncio.Event()
    blocked = threading.Event()
    conns: list[aiosqlite.Connection] = []

    async def use_db() -> None:
        async with open_db(tmp_data_dir / "vllm-warden.db") as db:
            conns.append(db)
            # A statement the worker thread is stuck on, so the close that
            # follows has to queue behind it.
            asyncio.ensure_future(db._execute(lambda: blocked.wait(30)))
            await asyncio.sleep(0)
            in_body.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(use_db())
    await in_body.wait()

    task.cancel()
    # One yield is enough for the task to unwind into open_db's `finally` and
    # park on the close; assert that it really did, so this test cannot pass by
    # accident if the shape ever changes.
    await asyncio.sleep(0)
    db = conns[0]
    assert not task.done()
    assert db._connection is not None, "the close should still be in flight here"

    task.cancel()  # lands on the close's own await
    blocked.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert db._connection is None, "aiosqlite's finally must have released it"
    _assert_nothing_leaked(before)


async def test_repeated_cancellation_abandons_one_daemon_thread_and_says_so(
    tmp_data_dir, gated_connect, monkeypatch, caplog
):
    """The attempt bound: a caller cancelling in a loop cannot pin us in an
    uncancellable wait, at the documented price of one daemon thread.

    The deadline is pushed out of reach so this test can only be satisfied by
    the attempt bound. How many event-loop turns each delivery costs is an
    asyncio implementation detail (``wait_for`` awaits the future it cancels),
    so keep cancelling until the task dies, with a cap so a regression fails
    the test instead of hanging it.
    """
    before = _connection_threads()
    monkeypatch.setattr(database, "_CONNECT_CLEANUP_TIMEOUT", 60.0)
    task = await _parked_in_connect(tmp_data_dir / "vllm-warden.db")

    with caplog.at_level(logging.ERROR, logger="app.db.database"):
        for _ in range(40):
            if task.done():
                break
            task.cancel()
            await asyncio.sleep(0)
        done, _pending = await asyncio.wait({task}, timeout=10)

    assert task in done, "open_db never gave up: the cancellation bound is gone"
    assert task.cancelled()
    logged = [r.getMessage() for r in caplog.records]
    assert any(
        "worker thread is leaked" in m and "cancelled 4 times" in m for m in logged
    ), logged
    # Release the connector first: the worker cannot see a stop sentinel while
    # it is still inside sqlite3.connect, so joining it before this would just
    # burn the join timeout.
    gated_connect.set()
    _leaked_daemon(before)


async def test_a_connect_that_never_returns_is_abandoned_on_a_deadline(
    tmp_data_dir, gated_connect, monkeypatch, caplog
):
    """The wall-clock bound. Without it a SINGLE ``Task.cancel()`` against a
    connect that never completes parks the task in an uncancellable await for
    good -- and a task that cannot be cancelled wedges ``asyncio.run``'s
    ``_cancel_all_tasks`` forever, which is #260's symptom in a new place.
    """
    before = _connection_threads()
    monkeypatch.setattr(database, "_CONNECT_CLEANUP_TIMEOUT", 0.05)
    task = await _parked_in_connect(tmp_data_dir / "vllm-warden.db")

    with caplog.at_level(logging.ERROR, logger="app.db.database"):
        task.cancel()  # exactly one, the ordinary shutdown shape
        done, _pending = await asyncio.wait({task}, timeout=10)

    assert task in done, "the cleanup wait has no deadline: the task is unkillable"
    assert task.cancelled()
    logged = [r.getMessage() for r in caplog.records]
    assert any("still connecting" in m and "worker thread is leaked" in m for m in logged), logged
    # Release the connector first: the worker cannot see a stop sentinel while
    # it is still inside sqlite3.connect, so joining it before this would just
    # burn the join timeout.
    gated_connect.set()
    _leaked_daemon(before)


async def test_a_failed_connect_propagates_unchanged(tmp_data_dir, monkeypatch):
    """No cancellation involved: aiosqlite stops the worker itself on this path,
    so the caller sees the sqlite error and nothing leaks."""
    before = _connection_threads()
    gate = threading.Event()
    gate.set()
    boom = sqlite3.OperationalError("unable to open database file")
    _install_gated_connect(monkeypatch, gate, error=boom)

    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        async with open_db(tmp_data_dir / "vllm-warden.db"):
            pytest.fail("the connect failed; the body must not run")

    _assert_nothing_leaked(before)


async def test_a_failed_connect_with_a_cancellation_pending_raises_the_cancellation(
    tmp_data_dir, monkeypatch
):
    """The subtlest branch in the module: the connect fails *while* we are
    holding a cancellation we owe the caller. The cancellation wins -- a task
    that was cancelled must end cancelled -- and the connect's failure is kept
    as its ``__cause__`` rather than thrown away.
    """
    before = _connection_threads()
    gate = threading.Event()
    boom = sqlite3.OperationalError("connect boom")
    _install_gated_connect(monkeypatch, gate, error=boom)
    captured: dict = {}
    task = await _parked_in_connect(tmp_data_dir / "vllm-warden.db", captured)

    task.cancel()
    gate.set()  # the connect now fails, with the cancellation already deferred
    done, _pending = await asyncio.wait({task}, timeout=10)

    assert task in done
    assert task.cancelled()
    # ``await task`` would raise a fresh CancelledError, so inspect the instance
    # the task actually saw.
    inner = captured["exc"]
    assert isinstance(inner, asyncio.CancelledError)
    assert inner.__cause__ is boom
    _assert_nothing_leaked(before)


async def test_a_timeout_around_open_db_raises_timeouterror(
    tmp_data_dir, gated_connect, monkeypatch
):
    """``asyncio.timeout`` bookkeeping survives the deferral: the caller gets a
    TimeoutError, not a stray CancelledError escaping the scope."""
    before = _connection_threads()
    monkeypatch.setattr(database, "_CONNECT_CLEANUP_TIMEOUT", 0.05)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            async with open_db(tmp_data_dir / "vllm-warden.db"):
                pytest.fail("the gated connect must not have completed")

    # Release the connector first: the worker cannot see a stop sentinel while
    # it is still inside sqlite3.connect, so joining it before this would just
    # burn the join timeout.
    gated_connect.set()
    _leaked_daemon(before)


async def test_a_deadline_that_races_the_connect_finishing_still_closes_it(
    tmp_data_dir, gated_connect, monkeypatch
):
    """The narrow window where the cleanup deadline expires and the connect
    completes anyway, between the timeout's cancellation being scheduled and
    the ``shield`` being unwound.

    ``shield`` drops that result on the floor, so ``connecting`` is the only
    thing still holding the live connection. Handing the cancellation on from
    there would abandon it -- one parked daemon thread and an open sqlite
    handle, with no log, because the "abandoned" error sits on the other exit.

    The window is sub-millisecond and needs a connect that has been wedged for
    exactly the cleanup budget, so it is injected rather than waited for: the
    first cleanup timeout is replaced by one that lets the connect land and
    then expires before the body runs -- which is the state that matters, a
    finished connect whose result never reached us. Everything downstream of
    that -- the branch, the loop, the close -- is the real code.
    """
    before = _connection_threads()
    injected = False
    real_timeout = asyncio.timeout

    class _TimeoutRacingTheConnect:
        async def __aenter__(self):
            gated_connect.set()
            # Wait for the connect to be genuinely finished before expiring, so
            # this reproduces the race rather than the ordinary deadline.
            # Bounded: if it never lands, the test fails on the leak check
            # below instead of hanging.
            for _ in range(500):
                conns = _connection_threads() - before
                if conns and next(iter(conns))._connection is not None:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0)  # let `connecting` observe its own result
            raise TimeoutError

        async def __aexit__(self, *exc):
            return False

    def timeout_factory(delay):
        nonlocal injected
        if injected:
            return real_timeout(delay)
        injected = True
        return _TimeoutRacingTheConnect()

    monkeypatch.setattr(database.asyncio, "timeout", timeout_factory)
    task = await _parked_in_connect(tmp_data_dir / "vllm-warden.db")

    task.cancel()
    done, _pending = await asyncio.wait({task}, timeout=10)

    assert task in done
    assert task.cancelled()
    assert injected, "the injected timeout was never reached"
    # The point of the test: the connection that landed in the window was
    # closed, not dropped.
    _assert_nothing_leaked(before)


async def test_a_close_that_is_never_acknowledged_is_not_waited_on_forever(
    tmp_data_dir, monkeypatch, caplog
):
    """A `finally` that parks forever is a process that will not exit.

    With the worker thread dead, nothing will ever resolve the close's future,
    and a task that has already taken its one cancellation has nothing left to
    break it out: it sits in ``open_db``'s ``finally`` indefinitely. (That is
    develop's behaviour too -- a plain ``async with aiosqlite.connect(...)``
    parks in exactly the same place -- so this is a bound being added, not a
    regression being fixed.) Giving up on the acknowledgement is safe because
    the close callable and the stop sentinel are already on the worker's
    queue.

    The one way a worker dies in the wild is ``call_soon_threadsafe`` raising
    "Event loop is closed" inside ``run()``, i.e. a connection outliving its
    loop -- #43's shape. Here it is done directly, with aiosqlite's own
    sentinel.
    """
    from aiosqlite.core import _STOP_RUNNING_SENTINEL

    monkeypatch.setattr(database, "_CLOSE_ACK_TIMEOUT", 0.05)
    in_body = asyncio.Event()

    async def use_db() -> None:
        async with open_db(tmp_data_dir / "vllm-warden.db") as db:
            await db.execute("SELECT 1")
            db._tx.put_nowait(_STOP_RUNNING_SENTINEL)
            for _ in range(500):
                if not db.is_alive():
                    break
                await asyncio.sleep(0.01)
            assert not db.is_alive(), "the worker thread should be gone"
            in_body.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(use_db())
    await in_body.wait()

    with caplog.at_level(logging.WARNING, logger="app.db.database"):
        task.cancel()  # exactly one, the ordinary shutdown shape
        done, _pending = await asyncio.wait({task}, timeout=10)

    assert task in done, "open_db parked forever in its finally on a dead worker"
    assert task.cancelled()
    assert any("not acknowledged" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


async def test_a_cancellation_during_the_close_is_never_lost():
    """A cancellation that arrives while the close is in flight must still end
    the task, bound or no bound.

    This is the invariant ``asyncio.wait_for`` broke here. On 3.11 it returns
    the result instead of re-raising when a cancellation arrives and the
    awaited future is already done (``asyncio/tasks.py:476``) -- and "already
    done" is the common case for a close. The consequence was not academic: a
    background loop cancelled by the lifespan while inside ``open_db``'s
    teardown lost its cancellation, went back round to
    ``await asyncio.sleep(GC_INTERVAL_S)`` -- six hours -- and the lifespan's
    ``gather`` waited for it, hanging ``tests/unit/auth`` outright about one
    run in three. ``asyncio.timeout`` uncancels only the cancellation it
    raised itself, so a foreign one still propagates.

    A fake connection, so the interleaving is exact rather than thread-timed.
    """
    gate = asyncio.Event()

    class FakeConnection:
        async def close(self) -> None:
            await gate.wait()

    task = asyncio.create_task(database._close(FakeConnection()))
    await asyncio.sleep(0)  # park it inside the close
    assert not task.done()

    task.cancel()
    gate.set()  # the close can finish; the cancellation still has to count

    done, _pending = await asyncio.wait({task}, timeout=5)
    assert task in done
    assert task.cancelled(), (
        "the cancellation was swallowed: a task cancelled during open_db's "
        "teardown would carry on running"
    )


def test_the_module_does_not_use_wait_for():
    """Static guard for the above: nothing in this module may use
    ``asyncio.wait_for``, whose 3.11 cancellation semantics are unsafe in a
    cleanup path. ``asyncio.timeout`` is the cancellation-correct equivalent.
    """
    from pathlib import Path

    source = Path(database.__file__).read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "wait_for(" in line and not line.lstrip().startswith(("#", "*"))
    ]
    assert not offenders, offenders
