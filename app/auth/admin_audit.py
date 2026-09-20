"""The admin-token audit trail (spec 2026-09-19, decision 7).

Every request an admin token makes gets one ``admin_audit`` row. What marks a
request as one is an ``AdminAuditContext`` on ``request.state`` under
``STATE_KEY``, set by app/auth/deps.py::identify_admin_token as soon as the
secret matches an admin row -- BEFORE the expiry, revocation and pause checks
and before a session-only route's 403, so a refused attempt with a known token
is recorded too: a revoked token still being tried, or a leaked one probing
token management, is exactly what the trail is for. A secret matching no admin
row sets nothing (there is no token to file it under), and session requests
never carry a context.

Boundary: a request is audited once an auth dependency has identified its
token; requests refused before routing (unmatched 404, 405) or by an outer
middleware (e.g. the chat2 body-limit 413) are not -- no auth dependency ever
runs for them, so there is no token to attribute the row to. Identifying the
token in this middleware itself would add a DB lookup to every request just
to cover probes that never reach a handler, which is not worth it.

WRITE PATH (#258). The middleware does NOT write. It hands the finished row to
``AdminAuditWriter.record`` -- synchronous, non-blocking, never raises -- and
one background flusher (started by app/main.py's lifespan alongside the
pruner, sampler, watchdog, history writer and chat2 gc) drains the queue in
batches, one transaction per batch. Three reasons:

* A connection opened, a row inserted and an fsync paid per request put a
  SQLite commit on the response path of every admin-token call. A CI script
  polling the API paid for one commit per poll; a batch pays for one per
  flush.
* That write also had to be shielded so a client disconnecting mid-stream
  still recorded, which meant a task per request outliving the request.
* Rows pending at shutdown were lost, because nothing waited for them.
  Cancelling the flusher flushes, so the lifespan's existing stop sequence
  covers them -- including a batch already taken out of the queue when the
  cancel arrives, which goes back in and is written by that final flush.

What the queue costs: a row can be up to ``flush_interval_s`` old before it is
durable, and a hard kill -9 loses what is queued. The trail is a forensic
record, not a transaction log, and it was never synchronous with the response
anyway (the old write happened after the last byte). Ordering is by ``ts``,
which is stamped when the request ARRIVES, so batching cannot reorder the
trail even though insertion order within a flush is not guaranteed.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.db.database import open_db
from app.db.repos.admin_audit import AdminAuditRepo, AuditParams

logger = logging.getLogger(__name__)

#: The ``request.state`` attribute that carries the AdminAuditContext.
STATE_KEY = "admin_audit"

#: Rows in one transaction. 200 short rows is a single ``executemany`` and a
#: single fsync, which is the point; it also caps how long the flusher holds a
#: write transaction, which matters because every other DB user in the process
#: opens its own short-lived connection (see AdminAuditRepo.record_many).
BATCH_MAX_ROWS = 200
#: How long a row may wait for company. One second bounds the gap between a
#: response and its durable row, and matches the request-history writer
#: (app/stats/request_history.py) so the two background writers behave alike.
FLUSH_INTERVAL_S = 1.0
#: Rows the queue holds before it starts dropping. At ~100 bytes a row that is
#: ~1 MB, and it is 50 seconds of a sustained 200 requests/second burst -- far
#: past the point where the flusher (which drains the WHOLE queue, batch after
#: batch) would have caught up. Bounded on purpose: the audit must not be able
#: to grow the process's memory without limit because the data volume stalled.
QUEUE_MAX_ROWS = 10_000
#: How often the ENQUEUE side reports drops. Drops are also reported once per
#: flush, but the flush side alone was not enough: the state in which the queue
#: reliably fills is a flusher that is no longer running, and then no flush ever
#: comes. Matches the request-history writer's 60 s throttle
#: (app/stats/request_history.py::RequestHistoryStore.record), for the same
#: reason -- a dropped row must not cost a log line each.
DROP_LOG_INTERVAL_S = 60.0


@dataclass(frozen=True)
class AdminAuditContext:
    token_id: str
    #: The user the token acts as (``api_tokens.created_by``).
    username: str
    #: As reported through the proxy (X-Forwarded-For / X-Real-Ip), like
    #: request history -- forgeable by whoever holds the token.
    client_ip: str | None
    #: The socket peer, which the client cannot choose.
    peer_ip: str | None


def route_template(scope: Scope) -> str:
    """The matched route's template -- FastAPI puts the route on the scope --
    else the raw path."""
    template = getattr(scope.get("route"), "path_format", None)
    return template if isinstance(template, str) else str(scope.get("path", ""))


class AdminAuditWriter:
    """Bounded queue in front of the ``admin_audit`` table (module docstring).

    ``record`` is the only method the middleware calls: synchronous,
    non-blocking and it never raises. ``run_forever`` is the flusher the
    lifespan starts; ``flush`` is the same drain, awaitable directly by
    shutdown and by tests (tests/conftest.py::flush_audit) so nothing has to
    sleep past an interval to see a row.

    ``batch_max`` and ``flush_interval_s`` are plain attributes, read at every
    use, so a test can retune them on a live writer.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        queue_max: int = QUEUE_MAX_ROWS,
        batch_max: int = BATCH_MAX_ROWS,
        flush_interval_s: float = FLUSH_INTERVAL_S,
    ) -> None:
        self.db_path = db_path
        self.batch_max = max(1, int(batch_max))
        self.flush_interval_s = float(flush_interval_s)
        self._q: asyncio.Queue[AuditParams] = asyncio.Queue(maxsize=max(1, int(queue_max)))
        #: Rows refused because the queue was full. Counted, never silent: the
        #: log says how many, so a gap in a token's trail can be told from a
        #: token that made no calls.
        self.dropped = 0
        #: Rows written since start, for the same reason.
        self.written = 0
        #: Seconds between two enqueue-side drop reports. A plain attribute, so
        #: a test can retune it on a live writer.
        self.drop_log_interval_s = DROP_LOG_INTERVAL_S
        self._logged_drops = 0
        self._last_drop_log = 0.0
        # One flush at a time. The periodic flusher and an explicit flush
        # (shutdown, a test) share it, so two connections never hold write
        # transactions on the same file at once.
        self._flushing = asyncio.Lock()
        # Set by ``record``: something is queued. Cleared by the flusher.
        self._queued = asyncio.Event()
        # Set by ``record`` when the queue already holds a full batch, so the
        # flusher stops waiting out the interval.
        self._full_batch = asyncio.Event()

    # -- write side ---------------------------------------------------------

    def record(
        self,
        *,
        ts: float,
        token_id: str,
        username: str,
        method: str,
        path: str,
        status: int,
        duration_ms: int,
        client_ip: str | None,
        peer_ip: str | None,
    ) -> bool:
        """Enqueue one finished request. Never raises, never blocks.

        Returns False when the row was dropped (a full queue, or -- defence in
        depth -- anything unexpected). The audit must never turn a served
        request into an error, which is why the middleware ignores the answer.
        """
        try:
            row: AuditParams = (
                float(ts), str(token_id), str(username), str(method), str(path),
                int(status), int(duration_ms), client_ip, peer_ip,
            )
            self._q.put_nowait(row)
        except asyncio.QueueFull:
            self.dropped += 1
            self._log_drops_throttled()
            return False
        except Exception:  # noqa: BLE001 -- the audit must never fail a request
            self.dropped += 1
            logger.debug("admin audit: could not enqueue a row", exc_info=True)
            self._log_drops_throttled()
            return False
        self._queued.set()
        if self._q.qsize() >= self.batch_max:
            self._full_batch.set()
        return True

    def pending(self) -> int:
        return self._q.qsize()

    def _take_batch(self) -> list[AuditParams]:
        """Up to ``batch_max`` rows, taken from the queue right now."""
        batch: list[AuditParams] = []
        while len(batch) < self.batch_max:
            try:
                batch.append(self._q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    def _log_drops(self) -> None:
        """Report every drop not reported yet. Called at the end of each flush,
        and -- throttled -- from the enqueue side."""
        missed = self.dropped - self._logged_drops
        if missed:
            self._logged_drops = self.dropped
            self._last_drop_log = time.monotonic()
            logger.warning(
                "admin audit: dropped %d row(s) since the last report "
                "(%d in total); those requests are missing from their token's trail",
                missed,
                self.dropped,
            )

    def _log_drops_throttled(self) -> None:
        """``_log_drops`` from ``record``, at most once per
        ``drop_log_interval_s``.

        The reason the enqueue side reports at all: ``_log_drops`` used to be
        reachable only from ``flush``, so the one situation that reliably fills
        the queue -- a flusher that is no longer running -- was also the one
        situation in which nothing was ever logged. Never raises: ``record`` is
        on the response path.
        """
        try:
            now = time.monotonic()
            if now - self._last_drop_log >= self.drop_log_interval_s:
                self._log_drops()
        except Exception:  # noqa: BLE001 -- the audit must never fail a request
            pass

    def _requeue(self, batch: list[AuditParams]) -> None:
        """Put a batch back after a cancellation, counting whatever no longer
        fits.

        Ordering does not matter: the trail is read by ``ts``, stamped when the
        request arrived, so rows going back in at the tail are still in order.
        """
        for row in batch:
            try:
                self._q.put_nowait(row)
            except Exception:  # noqa: BLE001 -- full, or anything else
                self.dropped += 1
        self._log_drops_throttled()

    async def flush(self) -> int:
        """Write everything queued right now, one transaction per batch.

        Returns the rows written. A failed batch is logged, COUNTED as dropped
        and not re-queued: a persistently broken volume would otherwise grow the
        queue until it overflowed anyway, with the log noise multiplied -- but
        it is still a gap in a token's trail, so it belongs in the same counter
        as a full queue (two causes, one number). Drops are reported once per
        flush, here, so the warning is never silent and never per-row.

        A CANCELLATION is different from a failure: the batch is already out of
        the queue but has not been written, so it goes back in before the
        cancellation propagates. Otherwise a shutdown landing inside the write
        lost up to ``batch_max`` rows while claiming to lose nothing -- the
        final flush in ``run_forever`` only ever sees what is still queued.
        """
        written = 0
        async with self._flushing:
            while (batch := self._take_batch()):
                try:
                    async with open_db(self.db_path) as db:
                        await AdminAuditRepo(db).record_many(batch)
                except asyncio.CancelledError:
                    self._requeue(batch)
                    raise
                except Exception:
                    self.dropped += len(batch)
                    logger.exception(
                        "admin audit: dropped a batch of %d row(s)", len(batch)
                    )
                    continue
                self.written += len(batch)
                written += len(batch)
        self._log_drops()
        return written

    async def run_forever(self) -> None:
        """Flush on the interval, or as soon as a whole batch is queued.

        Waits for the first row, then gives a burst up to ``flush_interval_s``
        to gather behind it -- cut short once the queue holds a full batch --
        so a poll loop costs one transaction per interval rather than one per
        request. Rows stay IN the queue while it waits, so an explicit
        ``flush`` from elsewhere can never miss one the flusher is holding.

        Cancellation (the lifespan's shutdown) performs a final flush, so a
        request served a moment before shutdown still lands.

        Anything else is logged and the loop continues, like every other
        background loop in app/main.py (the watchdog, the pruner, the sampler,
        the chat2 gc). It used to catch only ``CancelledError``, so one
        unexpected exception ended the task -- and nothing observed it: the only
        await on it is a ``gather(..., return_exceptions=True)`` at shutdown,
        which discards it. Auditing then stopped for the life of the process,
        silently, and the queue filled to its cap without a log line. The
        interval-long pause on that path keeps a persistent failure (a volume
        that has gone away) from becoming a busy loop.
        """
        try:
            while True:
                try:
                    await self._queued.wait()
                    self._queued.clear()
                    self._full_batch.clear()
                    if self._q.qsize() < self.batch_max:
                        try:
                            await asyncio.wait_for(
                                self._full_batch.wait(), self.flush_interval_s
                            )
                        except TimeoutError:
                            pass
                    await self.flush()
                except Exception:
                    logger.exception(
                        "admin audit: the flusher hit an unexpected error; "
                        "retrying in %.1fs", self.flush_interval_s
                    )
                    # The wakeup was consumed above but the rows are still
                    # queued, so re-arm it or the loop would park on a queue
                    # that is not empty.
                    if not self._q.empty():
                        self._queued.set()
                    await asyncio.sleep(self.flush_interval_s)
        except asyncio.CancelledError:
            try:
                await self.flush()
            except Exception:  # noqa: BLE001 -- shutdown must not hang on the volume
                logger.debug("admin audit: final flush failed", exc_info=True)
            raise


class AdminAuditMiddleware:
    """Pure ASGI -- never ``BaseHTTPMiddleware`` -- so a streamed response
    passes through untouched. Notes the status from ``http.response.start``
    and, once the app has returned (for a stream: after the last byte or the
    client's disconnect), ENQUEUES the row if the request carried a context
    (``AdminAuditWriter``, module docstring). Session requests carry none and
    are not audited.

    The enqueue cannot block, cannot fail the request and does not outlive it,
    so a request cancelled mid-stream still records without a shielded write
    task per request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        ts = time.time()
        started = time.monotonic()
        status = 500  # what the client saw if the app raised before responding

        async def send_noting_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_noting_status)
        finally:
            state = scope.get("state")
            ctx = state.get(STATE_KEY) if isinstance(state, dict) else None
            if isinstance(ctx, AdminAuditContext):
                writer = getattr(scope["app"].state, "admin_audit", None)
                if isinstance(writer, AdminAuditWriter):
                    writer.record(
                        ts=ts,
                        token_id=ctx.token_id,
                        username=ctx.username,
                        method=str(scope.get("method", "")),
                        path=route_template(scope),
                        status=status,
                        duration_ms=int((time.monotonic() - started) * 1000),
                        client_ip=ctx.client_ip,
                        peer_ip=ctx.peer_ip,
                    )
                else:  # pragma: no cover -- the lifespan always sets one
                    logger.warning(
                        "admin audit: no writer on app.state; dropping the row for %s",
                        ctx.token_id,
                    )
