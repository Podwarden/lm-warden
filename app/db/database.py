"""The single place a SQLite connection is opened.

Why this module is more than one ``async with`` (#43, #260)
----------------------------------------------------------
``aiosqlite`` drives every connection from a worker thread of its own:
``aiosqlite.Connection`` *is* a ``threading.Thread``, created with the default
``daemon=False``, and its ``run()`` loop parks on a queue until ``close()``
enqueues the stop sentinel. A connection that is dropped without being closed
therefore leaks a **non-daemon** thread, and CPython joins those during
interpreter shutdown before ``main`` returns — so the process hangs forever
*after* all of its work is finished, printing nothing at all.

That has bitten this repo twice: #43 (the CI integration job reporting
``47 passed`` and then wedging 25+ minutes until the runner killed it) and #260
(the unit suite hanging after pytest's summary line, roughly one run in three).
Both times the connection was abandoned inside ``open_db`` because the task that
opened it was cancelled — at shutdown the lifespan cancels the watchdog's
background passes, and those sit on ``async with open_db(...)``.

The one path that actually leaks is a cancellation delivered *inside* the
connect, and it leaks for two compounding reasons:

* ``Connection.__await__`` starts the thread and then awaits the connect, so a
  cancellation there leaves the caller holding no connection object at all —
  ``aiosqlite``'s ``__aexit__`` never runs, because ``__aenter__`` never
  returned.
* ``Connection._connect()`` only catches ``Exception``, and ``CancelledError``
  is a ``BaseException``, so the thread is not stopped on the way out either.
  Worse, ``close()`` is a no-op while ``_connection`` is ``None``, so even
  getting hold of that object would not be enough: there would be nothing to
  close and the thread would stay parked.

So ``open_db`` runs the connect as a shielded task and, if a cancellation
arrives, waits for that task to finish before closing what it produced — then
re-raises the cancellation, so callers still see the one they asked for.

Only the *connect* needs that treatment. A cancelled ``close()`` is safe on its
own: ``Connection._execute`` puts the callable on the worker's queue **before**
it awaits, and ``close()`` stops the thread from a ``finally``, so the sqlite
connection is closed and the thread stopped even when the await never resumes
(verified against aiosqlite 0.20.0: a cancelled close returns in 0.000s with
``_running`` False, ``_connection`` None and the thread joined). Shielding it
would only buy an uncancellable wait behind whatever statement is in flight —
up to ``busy_timeout`` — on every cancelled request and every shutdown. It is
still bounded by ``_CLOSE_ACK_TIMEOUT``, because a task that has already taken
its one cancellation has nothing left to break it out of a close that will
never be acknowledged, and a ``finally`` that parks forever is the same
process-that-will-not-exit in a different place. When ``open_db`` itself is
unwinding a cancellation the bound is far shorter
(``_CANCELLED_CLOSE_ACK_TIMEOUT``): the cancel may have left a statement in
flight that holds the close up for its whole busy wait.

The one deliberate concession: a cleanup wait cannot be cancelled, by
construction, so it is bounded by attempts *and* by the clock
(``_CONNECT_CLEANUP_TIMEOUT``). On either bound the connect is abandoned, which
leaks one parked thread — a daemon one, logged at ``error``. That is the lesser
evil: an unkillable task wedges ``asyncio.run``'s ``_cancel_all_tasks`` forever,
which is the very symptom #260 is about.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# How many times the shielded connect is re-awaited after a cancellation has
# been delivered to us. One retry covers the ordinary single ``Task.cancel()``;
# the bound keeps a caller that cancels in a loop from pinning us here.
_CANCEL_SHIELD_ATTEMPTS = 4

# Wall-clock budget for those cleanup re-awaits, measured from the first
# deferred cancellation. It bounds the case the attempt count cannot: one
# ``Task.cancel()`` against a connect that never completes, which would
# otherwise park us in an uncancellable await forever.
#
# It cannot fire on a healthy connect, because it is not a connect timeout --
# the first await is an ordinary cancellable one and has no deadline at all;
# this clock only starts once we are waiting purely to clean up. And 10s is
# twice sqlite3's own connect-time lock timeout (the `timeout` argument,
# default 5s), which is what bounds the slowest legitimate connect: one that
# has to wait out a lock while recovering a hot journal. A connect still
# outstanding after 10s is wedged, not slow -- its worker thread is gone or
# stuck -- so waiting longer cannot help. 10s also keeps the worst added
# shutdown latency inside Docker's 10s stop grace, where the alternative
# (waiting forever) guaranteed a SIGKILL.
_CONNECT_CLEANUP_TIMEOUT = 10.0

# How long a close waits to be acknowledged before it is left to finish on its
# own. This is not a shield -- the await stays cancellable throughout -- it only
# stops an *uncancelled* wait from parking forever on a worker thread that is
# gone. Expiring early is cheap: the close callable and the stop sentinel are
# both already on the worker's queue, so the connection closes and the thread
# stops either way, and all we give up is the acknowledgement. So this is
# deliberately shorter than the 30s `busy_timeout` a contended close can
# legitimately wait, and matches the connect budget: both exist to keep a
# shutdown bounded, and 10s is Docker's default stop grace.
_CLOSE_ACK_TIMEOUT = 10.0

# The same wait when ``open_db`` is unwinding a cancellation. A cancel that
# lands while a statement is in flight leaves that statement running on the
# worker thread -- under a write lock held elsewhere, inside SQLite's busy
# handler for up to the 30 s ``busy_timeout`` -- and the close is queued behind
# it. Waiting the full ``_CLOSE_ACK_TIMEOUT`` there added a fixed 10 s to every
# shutdown that caught a loop mid-write on a locked database (the stats pruner's
# boot-time prune, seen as a 2 s + 10.0 s shutdown in the ledger tests), which
# alone is Docker's whole 10 s stop grace. Nothing is gained by that wait: the
# cancelled caller has abandoned the statement, and the close and the stop
# sentinel are already queued, so the connection still closes and the thread
# still stops once the statement gives up. A healthy close is acknowledged in
# milliseconds, so this only ever cuts short a close stuck behind a statement.
# sqlite3's interrupt() cannot shorten that statement instead: it does not
# break a busy-handler wait.
_CANCELLED_CLOSE_ACK_TIMEOUT = 0.5


@asynccontextmanager
async def open_db(path: Path) -> AsyncIterator[aiosqlite.Connection]:
    db = await _connect(path)
    ack_timeout = _CLOSE_ACK_TIMEOUT
    try:
        # busy_timeout FIRST, before the two pragmas that can themselves hit the
        # lock: `journal_mode = WAL` needs an exclusive lock, so setting it
        # ahead of the timeout left the statement most likely to contend with
        # only sqlite3's 5s default. Background writers (stats sampler, pull
        # poller) run their own short-lived connections and can collide while a
        # model download saturates CPU/IO; a generous busy_timeout makes them
        # wait out the contention window instead of failing fast with
        # "database is locked".
        await db.execute("PRAGMA busy_timeout = 30000")
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        yield db
    except asyncio.CancelledError:
        ack_timeout = _CANCELLED_CLOSE_ACK_TIMEOUT
        raise
    finally:
        await _close(db, ack_timeout)


def _keep_draining_after_loop_close(conn: aiosqlite.Connection) -> None:
    """Keep *conn*'s worker thread draining its queue once its loop is closed.

    A statement abandoned by a cancelled ``open_db`` (see
    ``_CANCELLED_CLOSE_ACK_TIMEOUT``) can outlive the event loop -- at shutdown
    it is still in its busy wait when the loop closes. When it finally returns,
    aiosqlite 0.20.0's ``run()`` reports the result with
    ``future.get_loop().call_soon_threadsafe(...)``, which raises
    ``RuntimeError('Event loop is closed')`` from both of its branches and
    kills the thread: an unhandled thread exception on stderr, and the queued
    ``close`` and stop sentinel never run, so the sqlite connection (and any
    write lock its abandoned statement took) stays open until it is garbage
    collected. Nobody is waiting for that result any more, so drop it and go
    back to the queue: the close runs, the stop sentinel ends the thread.
    """
    drain = conn.run

    def run() -> None:
        while True:
            try:
                drain()
            except RuntimeError as exc:
                if "Event loop is closed" not in str(exc):
                    raise
                continue
            return

    # The thread calls ``self.run()``; an instance attribute shadows the method.
    conn.run = run  # type: ignore[method-assign]


async def _open(conn: aiosqlite.Connection) -> aiosqlite.Connection:
    """Start the connection's thread and connect, as its own awaitable."""
    return await conn


async def _connect(path: Path) -> aiosqlite.Connection:
    """Connect, without ever abandoning a started connection thread."""
    conn = aiosqlite.connect(path)
    # Belt and braces for the hang this module exists to prevent: should a
    # connection ever still escape the cleanup here, a daemon thread cannot
    # block interpreter shutdown the way a non-daemon one does. The flag has to
    # be set before the thread starts, which is what awaiting `conn` does.
    conn.daemon = True
    _keep_draining_after_loop_close(conn)
    connecting = asyncio.ensure_future(_open(conn))
    cancelled: asyncio.CancelledError | None = None
    deadline: float | None = None
    reason = f"cancelled {_CANCEL_SHIELD_ATTEMPTS} times"
    for _ in range(_CANCEL_SHIELD_ATTEMPTS):
        try:
            if deadline is None:
                # Nothing deferred yet: an ordinary, cancellable await.
                db = await asyncio.shield(connecting)
            else:
                # We owe the caller a cancellation, so this wait cannot be
                # interrupted by one -- hence the clock. `shield` keeps the
                # timeout's own cancel off `connecting`; cancelling the connect
                # here is the abandonment this whole module prevents.
                #
                # `asyncio.timeout`, never `asyncio.wait_for`: see _close.
                async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                    db = await asyncio.shield(connecting)
        except asyncio.CancelledError as exc:
            if connecting.cancelled():
                raise
            # The cancellation was aimed at us, not at the shielded connect. We
            # cannot hand it on yet: until the connect returns there is no
            # connection object to close, and its thread is already parked.
            cancelled = exc
            if deadline is None:
                deadline = time.monotonic() + _CONNECT_CLEANUP_TIMEOUT
            continue
        except TimeoutError:
            if not connecting.done():
                reason = f"still connecting {_CONNECT_CLEANUP_TIMEOUT}s after the cancellation"
                break
            # `connecting` is done, so this TimeoutError is one of two things.
            #
            # Either our deadline fired and the connect completed inside the
            # window `wait_for` spends cancelling and awaiting the shield --
            # which `shield` then drops on the floor. A live connection nobody
            # will ever close is exactly the leak this module exists to stop,
            # so go round again: the next `wait_for` sees a finished shield,
            # hands the connection back, and the `await _close(db)` below does
            # the right thing with it.
            if connecting.cancelled():
                continue
            err = connecting.exception()
            if err is None:
                continue
            # Or the connect itself raised TimeoutError, which takes the same
            # precedence as any other connect failure (see below).
            if cancelled is not None:
                raise cancelled from err
            raise err from None
        except BaseException as exc:
            # The connect genuinely failed; aiosqlite stops the thread itself on
            # that path. A cancellation delivered while we waited still wins,
            # and keeps the failure as its context.
            if cancelled is not None:
                raise cancelled from exc
            raise
        if cancelled is None:
            return db
        await _close(db)
        raise cancelled
    # Out of attempts, or out of time.
    if connecting.done() and not connecting.cancelled() and connecting.exception() is None:
        # It landed after all (the race described in the TimeoutError branch,
        # on the last iteration). Close it rather than drop it: nothing else
        # holds a reference, so this is the last chance anyone has.
        await _close(connecting.result())
        raise cancelled if cancelled is not None else asyncio.CancelledError()
    # Give up on the connect and say so: this is the one path that still leaks a
    # parked thread, and the only reason that is survivable is `conn.daemon`.
    connecting.cancel()
    logger.error("SQLite connect abandoned (%s); its worker thread is leaked", reason)
    # `cancelled` is always set here -- the two `continue`/`break` paths both
    # set it -- but the raise does not lean on that.
    raise cancelled if cancelled is not None else asyncio.CancelledError()


async def _close(db: aiosqlite.Connection, ack_timeout: float | None = None) -> None:
    """Close *db* without ever masking the exception we are unwinding from.

    Deliberately NOT shielded. ``Connection.close()`` queues ``sqlite3.
    Connection.close`` on the worker thread before it awaits, and stops that
    thread from a ``finally``, so a cancellation delivered here still leaves the
    connection closed and the thread joined -- nothing to protect. Shielding it
    instead bought an uncancellable wait behind whatever statement was in
    flight, up to ``busy_timeout``, on every client disconnect and every
    shutdown.

    The wait is bounded all the same. If the worker thread is dead -- the one
    way it can die is ``call_soon_threadsafe`` raising "Event loop is closed"
    inside ``run()``, i.e. a connection outliving its loop, which is what #43
    described -- nothing will ever resolve this await, and a `finally` that
    parks forever is a process that will not exit, the very thing #260 is
    about. Giving up on the wait costs nothing by the same argument as above:
    the close is already queued, so the connection still gets closed and the
    thread still stops, whether or not anyone is left waiting for the answer.

    ``asyncio.timeout``, NOT ``asyncio.wait_for``, and the difference is not
    cosmetic. On Python 3.11 ``wait_for`` **swallows a cancellation aimed at
    its caller** if the awaited future happens to be done in the same tick::

        except exceptions.CancelledError:
            if fut.done():
                return fut.result()      # asyncio/tasks.py:476 -- no re-raise

    which here is the common case, because the close usually completes at
    once. Measured: with ``wait_for``, a background loop cancelled by the
    lifespan lost its cancellation inside this very line, went back round to
    ``await asyncio.sleep(GC_INTERVAL_S)`` -- six hours -- and the lifespan's
    ``gather`` waited for it; ``tests/unit/auth`` hung outright, roughly one
    run in three. ``asyncio.timeout`` exists for exactly this: it uncancels
    only the cancellation it raised itself, so a foreign one still propagates.
    """
    # Read at call time (not as a default argument) so a test can lower it.
    if ack_timeout is None:
        ack_timeout = _CLOSE_ACK_TIMEOUT
    try:
        async with asyncio.timeout(ack_timeout):
            await db.close()
    except TimeoutError:
        logger.warning(
            "SQLite close was not acknowledged in %ss; it is queued on the "
            "connection's worker thread and no longer waited on",
            ack_timeout,
        )
    except Exception:
        # close() re-raises whatever the worker thread hit, having already
        # stopped it and released the connection in its own `finally`.
        logger.warning("closing the SQLite connection failed", exc_info=True)
