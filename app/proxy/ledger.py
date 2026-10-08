"""The proxy's request-path accounting, accumulated in memory (issue #279 stage 3).

Before this, every proxied request opened a connection and committed the
counters, the model's minute sample and the key's minute usage -- the writes
that, at a few req/s, made SQLite answer "database is locked" and failed 2% of
paid requests. Now a request only adds to dictionaries (synchronous, never
raises) and one background task writes everything once a second in ONE
transaction. Totals stay exact: a batch is swapped out atomically before the
await, records arriving meanwhile land in the next batch, and a failed write
merges its batch back so the next flush carries it.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.db.database import open_db
from app.db.repos.tokens import TokenRepo
from app.proxy import bookkeeping
from app.runtime.variants import ModelVariantRepo, is_recorded, mark_recorded

logger = logging.getLogger(__name__)

LEDGER_FLUSH_S = 1.0
MAX_FAIL_STREAK = 3
#: The ledger retries every second, so it never needs open_db's 30 s lock wait;
#: a short one keeps its worker thread (and shutdown) from hanging on a held lock.
LEDGER_BUSY_TIMEOUT_MS = 2000
#: A lock/busy retry loses nothing, so it is logged (at most this often), not
#: counted as a drop.
RETRY_LOG_EVERY_S = 60.0
#: How long a CANCELLED flush keeps waiting for its in-flight COMMIT (#295).
#: Matches database.py's _CLOSE_ACK_TIMEOUT: both exist to keep a shutdown
#: bounded when a worker thread has wedged. A healthy commit takes
#: milliseconds, so one still running after this long is stuck, not slow.
LEDGER_COMMIT_WAIT_S = 10.0


@dataclass
class _Batch:
    # (model_id, token_id) -> [requests, prompt, completion]
    counters: dict[tuple[Any, Any], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0])
    )
    # (model_id, minute) -> [requests, prompt, completion, cached, cached_measured]
    samples: dict[tuple[Any, Any], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0, 0, 0])
    )
    # (token_id, minute) -> [requests, prompt, completion]
    usage: dict[tuple[Any, Any], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0])
    )
    # (token_id, variant_id, model_id, minute) -> [requests, prompt, completion]
    variant_usage: dict[tuple[Any, Any, Any, Any], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0])
    )
    variants: dict[str, Any] = field(default_factory=dict)  # variant_id -> Variant
    touched: set[Any] = field(default_factory=set)
    committed: bool = False  # set right after COMMIT; a cancelled flush must not re-add it
    # The commit outlived LEDGER_COMMIT_WAIT_S after a cancel: it may yet land,
    # so a cancelled flush must not re-add this batch either.
    outcome_unknown: bool = False

    def requests(self) -> int:
        """How many proxied requests this batch carries (each adds one counter)."""
        return sum(v[0] for v in self.counters.values())

    def empty(self) -> bool:
        return not (
            self.counters or self.samples or self.usage or self.variant_usage or self.touched
        )

    def merge(self, other: _Batch) -> None:
        for name in ("counters", "samples", "usage", "variant_usage"):
            mine, theirs = getattr(self, name), getattr(other, name)
            for k, v in theirs.items():
                acc = mine[k]
                for i, x in enumerate(v):
                    acc[i] += x
        for k, v in other.variants.items():
            self.variants.setdefault(k, v)
        self.touched |= other.touched


def _is_lock_contention(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


class Ledger:
    def __init__(self, db_path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self._db_path = db_path
        self._clock = clock
        self._batch = _Batch()
        self._lock = asyncio.Lock()  # one flush at a time
        self._fail_streak = 0  # consecutive whole-batch non-Operational failures
        self._retry_warned_at = float("-inf")  # monotonic time of the last busy warning

    def pending(self) -> bool:
        return not self._batch.empty()

    def touched(self) -> frozenset[str]:
        """Token ids with a last-used touch waiting for the next flush."""
        return frozenset(self._batch.touched)

    def record_request(
        self,
        *,
        model_id: str,
        token_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int | None,
        variant_id: str | None,
        variant: Any = None,
    ) -> None:
        try:
            # Validate and coerce EVERYTHING before touching a dict, so a bad
            # input records nothing rather than half a request.
            prompt = int(prompt_tokens)
            completion = int(completion_tokens)
            cached = None if cached_tokens is None else int(cached_tokens)
            if prompt < 0 or completion < 0 or (cached is not None and cached < 0):
                raise ValueError("negative token count")
            if not isinstance(model_id, str) or not (token_id is None or isinstance(token_id, str)):
                raise TypeError("model_id/token_id must be str")
            if variant_id is not None and not isinstance(variant_id, str):
                raise TypeError("variant_id must be str")
            minute = int(self._clock() // 60)
            b = self._batch
            c = b.counters[(model_id, token_id)]
            c[0] += 1
            c[1] += prompt
            c[2] += completion
            s = b.samples[(model_id, minute)]
            s[0] += 1
            s[1] += prompt
            s[2] += completion
            if cached is not None:
                s[3] += cached
                s[4] += 1
            if token_id is not None:
                u = b.usage[(token_id, minute)]
                u[0] += 1
                u[1] += prompt
                u[2] += completion
                if variant_id is not None:
                    vu = b.variant_usage[(token_id, variant_id, model_id, minute)]
                    vu[0] += 1
                    vu[1] += prompt
                    vu[2] += completion
                    if variant is not None:
                        b.variants.setdefault(variant_id, variant)
        except Exception:  # noqa: BLE001 -- accounting must never fail a request
            logger.debug("ledger: could not record request", exc_info=True)

    def touch(self, token_id: str) -> None:
        try:
            self._batch.touched.add(token_id)
        except Exception:  # noqa: BLE001
            logger.debug("ledger: could not record touch", exc_info=True)

    def _restore(self, batch: _Batch) -> None:
        batch.merge(self._batch)
        self._batch = batch

    async def flush(self) -> int:
        async with self._lock:
            if self._batch.empty():
                return 0
            batch, self._batch = self._batch, _Batch()
            try:
                rows = await self._write(batch)
                self._fail_streak = 0
                return rows
            except asyncio.CancelledError:
                # Cancelled mid-write: the transaction rolls back unless it had
                # already committed, in which case re-adding would double count
                # (or may have: see LEDGER_COMMIT_WAIT_S in _write).
                if not (batch.committed or batch.outcome_unknown):
                    self._restore(batch)
                raise
            except sqlite3.OperationalError as exc:
                if _is_lock_contention(exc):  # locked / busy: retry whole batch, unbounded
                    self._restore(batch)
                    self._note_busy()
                    return 0
                return self._failed(batch, exc)
            except Exception as exc:  # noqa: BLE001 -- keep the data, keep the loop
                return self._failed(batch, exc)

    def _note_busy(self) -> None:
        """A lock/busy retry: nothing is lost, so log (rate-limited), never count."""
        now = time.monotonic()
        if now - self._retry_warned_at >= RETRY_LOG_EVERY_S:
            self._retry_warned_at = now
            logger.warning("ledger: flush deferred, database busy; retrying")

    def _failed(self, batch: _Batch, exc: Exception) -> int:
        """A whole-batch failure that is not lock contention: bounded retries.

        A retry loses nothing and is only logged; the drop counter moves when
        the batch is given up on (``ledger_batch_dropped``)."""
        self._fail_streak += 1
        if self._fail_streak >= MAX_FAIL_STREAK:
            # Bound memory: a batch that keeps failing for a non-lock
            # reason is poison. Newer records stay for the next flush.
            self._fail_streak = 0
            bookkeeping.note_dropped(
                "ledger_batch_dropped", exc, requests=batch.requests(), log=False
            )
            logger.warning(
                "ledger: dropping a batch of %d request(s) after %d consecutive failures",
                batch.requests(),
                MAX_FAIL_STREAK,
                exc_info=exc,
            )
        else:
            self._restore(batch)
            logger.warning("ledger: flush failed; will retry", exc_info=exc)
        return 0

    async def _write(self, b: _Batch) -> int:
        """One transaction; every key in its own SAVEPOINT.

        A key the database refuses for good (IntegrityError: its model or token
        was deleted while the record was pending) is rolled back to its
        savepoint, dropped and counted (once per flush, under
        ``ledger_key_dropped``); the rest commits. Locked/busy
        (OperationalError) aborts the whole transaction and the caller retries.
        """
        rows = 0
        recorded: list[str] = []
        # (kind, key, requests, exc) for every key the database refused
        dropped: list[tuple[str, tuple[Any, ...], int, Exception]] = []

        async def unit(
            db: Any,
            fn: Callable[[], Awaitable[None]],
            kind: str = "other",
            key: tuple[Any, ...] = (),
            n: int = 0,
        ) -> bool:
            await db.execute("SAVEPOINT ledger_unit")
            try:
                await fn()
            except sqlite3.OperationalError:
                raise
            except Exception as exc:  # noqa: BLE001 -- any per-key failure drops only that key
                await db.execute("ROLLBACK TO ledger_unit")
                await db.execute("RELEASE ledger_unit")
                dropped.append((kind, key, n, exc))
                return False
            await db.execute("RELEASE ledger_unit")
            return True

        async with open_db(Path(self._db_path)) as db:
            await db.execute(f"PRAGMA busy_timeout = {LEDGER_BUSY_TIMEOUT_MS}")
            await db.execute("BEGIN IMMEDIATE")
            for (model_id, token_id), (n, p, c) in b.counters.items():

                async def counter(
                    model_id: Any = model_id,
                    token_id: Any = token_id,
                    n: Any = n,
                    p: Any = p,
                    c: Any = c,
                ) -> None:
                    cur = await db.execute(
                        "UPDATE counters SET requests = requests + ?, prompt_tokens = prompt_tokens + ?, "
                        "completion_tokens = completion_tokens + ? WHERE model_id = ? "
                        "AND (token_id = ? OR (token_id IS NULL AND ? IS NULL))",
                        (n, p, c, model_id, token_id, token_id),
                    )
                    if cur.rowcount == 0:
                        await db.execute(
                            "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (model_id, token_id, n, p, c),
                        )

                rows += await unit(db, counter, "counter", (model_id, token_id), n)
            for (model_id, minute), (n, p, c, cached, measured) in b.samples.items():

                async def sample(
                    model_id: Any = model_id,
                    minute: Any = minute,
                    n: Any = n,
                    p: Any = p,
                    c: Any = c,
                    cached: Any = cached,
                    measured: Any = measured,
                ) -> None:
                    await db.execute(
                        "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, "
                        "completion_tokens, cached_tokens, cached_measured_requests) VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(model_id, minute) DO UPDATE SET requests = requests + excluded.requests, "
                        "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                        "completion_tokens = completion_tokens + excluded.completion_tokens, "
                        "cached_tokens = cached_tokens + excluded.cached_tokens, "
                        "cached_measured_requests = cached_measured_requests + excluded.cached_measured_requests",
                        (model_id, minute, n, p, c, cached, measured),
                    )

                rows += await unit(db, sample)
            for (token_id, minute), (n, p, c) in b.usage.items():

                async def usage(
                    token_id: Any = token_id,
                    minute: Any = minute,
                    n: Any = n,
                    p: Any = p,
                    c: Any = c,
                ) -> None:
                    await db.execute(
                        "INSERT INTO token_usage_minute"
                        "(token_id, minute, requests, prompt_tokens, completion_tokens) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(token_id, minute) DO UPDATE SET "
                        "  requests = requests + excluded.requests, "
                        "  prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                        "  completion_tokens = completion_tokens + excluded.completion_tokens",
                        (token_id, minute, n, p, c),
                    )

                rows += await unit(db, usage, "usage", (token_id, minute), n)
            # The descriptor row goes in before the usage rows that name it, in
            # its own savepoint so a failure here never sinks the batch.
            for vid, variant in b.variants.items():
                if not is_recorded(self._db_path, vid):

                    async def ensure(variant: Any = variant) -> None:
                        await ModelVariantRepo(db).ensure(variant, commit=False)

                    if await unit(db, ensure):
                        recorded.append(vid)
            for (token_id, variant_id, model_id, minute), (n, p, c) in b.variant_usage.items():

                async def variant_row(
                    token_id: Any = token_id,
                    variant_id: Any = variant_id,
                    model_id: Any = model_id,
                    minute: Any = minute,
                    n: Any = n,
                    p: Any = p,
                    c: Any = c,
                ) -> None:
                    await db.execute(
                        "INSERT INTO token_model_usage_minute"
                        "(token_id, variant_id, model_id, minute, requests, prompt_tokens, "
                        " completion_tokens) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(token_id, variant_id, minute) DO UPDATE SET "
                        "  requests = requests + excluded.requests, "
                        "  prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                        "  completion_tokens = completion_tokens + excluded.completion_tokens",
                        (token_id, variant_id, model_id, minute, n, p, c),
                    )

                rows += await unit(db, variant_row)
            for token_id in b.touched:

                async def touch(token_id: Any = token_id) -> None:
                    await db.execute(
                        "UPDATE api_tokens SET last_used_at = datetime('now') WHERE id = ?"
                        " AND (last_used_at IS NULL OR last_used_at < datetime('now', ?))",
                        (token_id, f"-{TokenRepo.LAST_USED_GRANULARITY_S} seconds"),
                    )

                rows += await unit(db, touch)
            # aiosqlite runs the commit on its worker thread; cancelling the
            # awaiter does not stop it. Shield it so a cancel cannot leave a
            # committed batch marked uncommitted (and so double counted).
            commit = asyncio.ensure_future(db.commit())
            try:
                await asyncio.shield(commit)
            except asyncio.CancelledError as cancel:
                # Let the commit finish, through any further cancels: stopping
                # early would leave a batch that DID commit marked uncommitted.
                # But not forever (#295): a commit stuck on a wedged worker
                # thread would hold shutdown until Docker's SIGKILL, past every
                # caller's own timeout. `asyncio.timeout`, never `wait_for`
                # (see database.py: on 3.11 wait_for swallows a cancel).
                deadline = time.monotonic() + LEDGER_COMMIT_WAIT_S
                while not commit.done():
                    try:
                        async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                            await asyncio.shield(commit)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 -- inspected just below (TimeoutError too)
                        break
                if not commit.done():
                    # The trade-off: the commit may still land, or never. Merging
                    # the batch back would count a commit that DID land twice,
                    # so it is not merged back; if it never lands, those
                    # requests go uncounted. A rare, logged undercount is
                    # preferred to a silent double charge.
                    b.outcome_unknown = True
                    logger.warning(
                        "ledger: commit outcome unknown: still running %.0fs after a "
                        "cancel; a batch of %d request(s) is not kept (it may or may "
                        "not have been written)",
                        LEDGER_COMMIT_WAIT_S,
                        b.requests(),
                    )
                elif not commit.cancelled() and commit.exception() is None:
                    b.committed = True
                else:
                    # Nothing committed: flush() merges the batch back. Never
                    # let this error replace the cancel (shutdown waits on it).
                    logger.warning(
                        "ledger: commit failed during a cancel; batch kept",
                        exc_info=None if commit.cancelled() else commit.exception(),
                    )
                raise cancel
            b.committed = True
        for vid in recorded:
            mark_recorded(self._db_path, vid)
        if dropped:
            self._note_key_drops(dropped)
        return rows

    @staticmethod
    def _note_key_drops(dropped: list[tuple[str, tuple[Any, ...], int, Exception]]) -> None:
        """One count and one warning per flush, however many keys were refused.

        Requests lost: the refused counter keys' requests, plus those of a
        refused usage key whose token has no refused counter key (a usage row
        and its counter row describe the same requests; count them once)."""
        counter_tokens = {key[1] for kind, key, _, _ in dropped if kind == "counter"}
        lost = sum(n for kind, _, n, _ in dropped if kind == "counter")
        lost += sum(
            n for kind, key, n, _ in dropped if kind == "usage" and key[0] not in counter_tokens
        )
        first = dropped[0][3]
        bookkeeping.note_dropped("ledger_key_dropped", first, requests=lost, log=False)
        logger.warning(
            "ledger: dropped %d key(s) the database refused, %d request(s) lost: %s",
            len(dropped),
            lost,
            first,
        )

    async def run_forever(self, interval: float = LEDGER_FLUSH_S) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
                await self.flush()
            except asyncio.CancelledError:
                # Records settled after the lifespan's explicit final flush
                # (or a flush cancelled mid-write) still get one bounded attempt.
                try:
                    async with asyncio.timeout(2):
                        await self.flush()
                except BaseException:  # noqa: BLE001 -- best effort, incl. timeout
                    logger.debug("ledger: final flush on cancel failed", exc_info=True)
                raise
            except Exception:  # noqa: BLE001
                logger.debug("ledger loop", exc_info=True)
