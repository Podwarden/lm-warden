"""Per-request history: the SQLite store behind the requests chart and the
latency distributions on the stats page.

WHY THIS EXISTS. Two operator complaints about the stats page had one cause:

  * "recently finished makes little sense" -- ten identical-looking rows,
    kept 15 minutes, nothing aggregated, so there was no question it answered.
  * "why are the latency charts only 5 minutes" -- the engine publishes only
    CUMULATIVE histograms, so the page kept bucket snapshots in React state and
    the "last 5 minutes" it drew was really "since you opened the tab, capped
    at 5 minutes".

Nothing persisted per-request history. This module does, and both panels read
from it. The columns are what ``app.proxy.request_registry.finished_record``
produces: the PROXY's own TTFT (first streamed frame) and duration, which exist
identically for every backend -- llama.cpp publishes no latency histogram at
all, and this is what gives GGUF models a latency panel for the first time.

WRITE PATH. ``RequestHistoryStore.record`` is called from the proxy's
``_deregister`` inside the streaming ``finally``, before the scheduler slot is
released. It therefore does NOT touch the database: it enqueues, and a
background writer drains the queue in batches. The proxy's existing guarantee
-- stats bookkeeping can never fail a request -- holds by construction here:
``record`` is synchronous, never raises, and a full queue drops the record
(counted, logged) rather than blocking a request on a busy /data volume. The
per-request accounting write (``_record_counters``) already opens its own
connection per request; this deliberately does not add a second one to the
slot-release path.

READ PATH. Module-level query functions take an open connection so the routes
can validate a selection and read on one connection, and so the queries can be
tested against a file with rows written by hand.

Everything statistical (quantiles, histograms) is pure and lives at the bottom.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from app.db.database import open_db

logger = logging.getLogger(__name__)

_COLUMNS = (
    "id",
    "finished_at",
    "model_id",
    "model",
    "token_name",
    "client_ip",
    "prompt_tokens",
    "completion_tokens",
    "duration_s",
    "ttft_s",
    "finish_reason",
    "orphan",
    "started_iso",
    "queued_s",
    "token_id",
    "variant_id",
)
_SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM request_history"
# INSERT OR IGNORE: the id is the registry's uuid, and a record that somehow
# reaches the writer twice must not turn into an IntegrityError that costs the
# whole batch.
_INSERT = (
    "INSERT OR IGNORE INTO request_history(" + ", ".join(_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" for _ in _COLUMNS) + ")"
)


def _row_params(record: dict[str, Any], finished_at: float) -> tuple[Any, ...]:
    """Coerce one ``finished_record`` dict into the INSERT's parameter tuple.

    Defensive on every field: the record is built on the proxy's hot path from
    values the engine and the client controlled, and a bad value here must
    cost this one row, not the batch it travels in.
    """
    def _int(v: Any) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    def _float(v: Any) -> float | None:
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    return (
        str(record.get("id") or ""),
        float(finished_at),
        str(record.get("model_id") or ""),
        str(record.get("model") or ""),
        record.get("token_name"),
        record.get("client_ip"),
        _int(record.get("prompt_tokens")),
        _int(record.get("completion_tokens")),
        _float(record.get("duration_s")) or 0.0,
        _float(record.get("ttft_s")),
        record.get("finish_reason"),
        1 if record.get("orphan") else 0,
        str(record.get("started_iso") or ""),
        _float(record.get("queued_s")),
        record.get("token_id"),
        record.get("variant_id"),
    )


def _row_dict(row: Iterable[Any]) -> dict[str, Any]:
    d = dict(zip(_COLUMNS, row, strict=True))
    d["orphan"] = bool(d["orphan"])
    return d


class RequestHistoryStore:
    """Queue in front of the ``request_history`` table.

    ``record`` is the only method the proxy calls. It is synchronous and
    fail-open. ``run_forever`` is the writer the app starts at boot;
    ``flush`` is the same drain, callable directly by tests and by shutdown.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        queue_max: int = 10_000,
        flush_interval_s: float = 1.0,
    ) -> None:
        self.db_path = db_path
        self.flush_interval_s = float(flush_interval_s)
        self._q: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue(maxsize=max(1, int(queue_max)))
        #: Records refused because the queue was full. Exposed so a test -- or
        #: an operator reading the log -- can tell "nothing happened" from
        #: "the volume was too slow to keep up".
        self.dropped = 0
        #: Rows written since start, for the same reason.
        self.written = 0
        self._last_drop_log = 0.0

    # -- write side ---------------------------------------------------------

    def record(self, record: dict[str, Any], *, finished_at: float | None = None) -> bool:
        """Enqueue one finished-request record. Never raises, never blocks.

        Returns False when the record was dropped. ``finished_at`` defaults to
        now (wall clock) -- the moment ``_deregister`` runs is the moment the
        request ended.
        """
        try:
            params = _row_params(record, time.time() if finished_at is None else finished_at)
            if not params[0]:
                return False
            self._q.put_nowait(params)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 60.0:
                self._last_drop_log = now
                logger.warning(
                    "request history: queue full, dropped %d record(s) so far; "
                    "the data volume is not keeping up",
                    self.dropped,
                )
            return False
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail a request
            logger.debug("request history: could not enqueue record", exc_info=True)
            return False

    def pending(self) -> int:
        return self._q.qsize()

    def _drain(self) -> list[tuple[Any, ...]]:
        batch: list[tuple[Any, ...]] = []
        while True:
            try:
                batch.append(self._q.get_nowait())
            except asyncio.QueueEmpty:
                return batch

    async def flush(self) -> int:
        """Write everything queued right now. Returns the number of rows sent.

        A failed write logs and drops the batch. Re-queueing would turn a
        persistently broken volume into an ever-growing queue and, once full,
        into the same drop -- with the log noise multiplied.
        """
        batch = self._drain()
        if not batch:
            return 0
        try:
            async with open_db(self.db_path) as db:
                await db.executemany(_INSERT, batch)
                await db.commit()
        except Exception:
            logger.exception("request history: dropped a batch of %d record(s)", len(batch))
            return 0
        self.written += len(batch)
        return len(batch)

    async def run_forever(self) -> None:
        """Drain the queue in batches, at most once per ``flush_interval_s``.

        Waits for the first record, then sleeps one interval so a burst lands
        as one transaction rather than one fsync per request. Cancellation
        performs a final flush so a clean shutdown loses nothing queued.
        """
        try:
            while True:
                first = await self._q.get()
                await asyncio.sleep(self.flush_interval_s)
                batch = [first, *self._drain()]
                try:
                    async with open_db(self.db_path) as db:
                        await db.executemany(_INSERT, batch)
                        await db.commit()
                    self.written += len(batch)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "request history: dropped a batch of %d record(s)", len(batch)
                    )
        except asyncio.CancelledError:
            try:
                await self.flush()
            except Exception:  # noqa: BLE001 -- shutdown must not hang on the volume
                logger.debug("request history: final flush failed", exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _model_clause(model_ids: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    """`` AND model_id IN (...)`` for a selection, or nothing when unfiltered.

    Empty when unfiltered so the unfiltered path executes exactly the SQL it
    always would rather than a filter that happens to match everything.
    """
    if model_ids is None:
        return "", ()
    if not model_ids:
        # `IN ()` is a syntax error in SQLite and a bare 1=1 would silently
        # widen to every model -- the substitution every stats route exists to
        # prevent.
        return " AND 0", ()
    return f" AND model_id IN ({','.join('?' for _ in model_ids)})", tuple(model_ids)


@dataclass(frozen=True)
class WindowRows:
    rows: list[dict[str, Any]]
    #: How many rows the window holds for the selection, before sampling.
    total: int
    #: 1 when every row is returned; k when every k-th row (newest first) is.
    stride: int


async def query_window(
    db: aiosqlite.Connection,
    *,
    since: float,
    model_ids: Sequence[str] | None,
    limit: int,
) -> WindowRows:
    """Rows that finished at or after ``since``, newest first, at most ``limit``.

    When the window holds more than ``limit`` rows, every k-th row is returned
    rather than the newest ``limit``: this feeds a chart with a TIME axis, and
    a "newest N" cut would leave the left of the chart empty while claiming to
    show the whole window. Uniform striding keeps the span covered; the
    response says the stride so the panel can state it.
    """
    where, args = _model_clause(model_ids)
    cur = await db.execute(
        "SELECT COUNT(*) FROM request_history WHERE finished_at >= ?" + where,
        (since, *args),
    )
    total = int((await cur.fetchone() or (0,))[0] or 0)
    limit = max(1, int(limit))
    if total <= limit:
        cur = await db.execute(
            _SELECT + " WHERE finished_at >= ?" + where + " ORDER BY finished_at DESC, id DESC",
            (since, *args),
        )
        rows = await cur.fetchall()
        return WindowRows(rows=[_row_dict(r) for r in rows], total=total, stride=1)
    stride = math.ceil(total / limit)
    cur = await db.execute(
        "SELECT " + ", ".join(_COLUMNS) + " FROM ("
        "  SELECT " + ", ".join(_COLUMNS) + ", "
        "         ROW_NUMBER() OVER (ORDER BY finished_at DESC, id DESC) AS rn"
        "  FROM request_history WHERE finished_at >= ?" + where +
        ") WHERE (rn - 1) % ? = 0 ORDER BY rn ASC LIMIT ?",
        (since, *args, stride, limit),
    )
    rows = await cur.fetchall()
    return WindowRows(rows=[_row_dict(r) for r in rows], total=total, stride=stride)


async def query_window_stats(
    db: aiosqlite.Connection,
    *,
    since: float,
    model_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Only the columns the summaries need, for EVERY row in the window.

    Feeds both ``latency_summary`` (TTFT / ITL / duration) and the prefill and
    generation rates -- one query rather than two nearly identical ones, since
    the rate functions need the same row plus ``prompt_tokens``.

    Deliberately uncapped: a distribution over a sample of the window is not
    the distribution of the window. Five columns a row keeps a full 7d at the
    retention cap well under a second.
    """
    where, args = _model_clause(model_ids)
    cur = await db.execute(
        "SELECT finished_at, ttft_s, duration_s, completion_tokens, prompt_tokens "
        "FROM request_history WHERE finished_at >= ?" + where,
        (since, *args),
    )
    return [
        {
            "finished_at": r[0],
            "ttft_s": r[1],
            "duration_s": r[2],
            "completion_tokens": r[3],
            "prompt_tokens": r[4],
        }
        for r in await cur.fetchall()
    ]


async def query_last(
    db: aiosqlite.Connection,
    *,
    n: int,
    model_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """The newest ``n`` rows for the selection regardless of time, newest first.

    The operator's alternative basis for the latency distributions: a fixed
    number of requests does not go silent when traffic is thin, which a time
    window does.
    """
    where, args = _model_clause(model_ids)
    cur = await db.execute(
        _SELECT + " WHERE 1" + where + " ORDER BY finished_at DESC, id DESC LIMIT ?",
        (*args, max(1, int(n))),
    )
    return [_row_dict(r) for r in await cur.fetchall()]


async def earliest_finished_at(db: aiosqlite.Connection) -> float | None:
    """When history begins, across the whole store. None when empty.

    Store-wide rather than per selection on purpose: "history begins 3 h ago"
    is a statement about the store (it was added in a release, or pruned), and
    a selection with no rows is a different fact that the row count carries.
    """
    cur = await db.execute("SELECT MIN(finished_at) FROM request_history")
    row = await cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


#: Most raw rows one series query reads for its percentiles, which are
#: computed in Python (spec 2026-09-18 §3.4). A busy key over a long window
#: could otherwise pull the whole retention cap into memory per poll.
TIMING_ROW_CAP = 50_000


@dataclass(frozen=True)
class TokenTimings:
    #: ``(finished_at, queued_s, ttft_s, duration_s)``, unordered.
    rows: list[tuple[float, float | None, float | None, float | None]]
    #: How many rows the window holds for these keys, before sampling.
    total: int
    #: 1 when every row is returned; k when every k-th row (oldest first) is.
    stride: int


async def query_token_timings(
    db: aiosqlite.Connection,
    *,
    token_ids: Sequence[str],
    since: float,
    until: float,
    cap: int = TIMING_ROW_CAP,
) -> TokenTimings:
    """Timing columns for every request by one of ``token_ids`` that finished
    in ``[since, until)`` -- or, past ``cap`` rows, every k-th of them.

    Feeds the token page's queue-wait and latency charts. Sampling is the
    same uniform stride ``query_window`` uses, ordered OLDEST first, so the
    whole period stays covered rather than the oldest part being cut off;
    ``k = ceil(total / cap)`` keeps the rows read at or under ``cap``.
    Served by idx_request_history_token (0033). Rows written before 0033
    have no token_id and are never matched.
    """
    if not token_ids:
        return TokenTimings(rows=[], total=0, stride=1)
    marks = ",".join("?" for _ in token_ids)
    where = f" WHERE token_id IN ({marks}) AND finished_at >= ? AND finished_at < ?"
    args = (*token_ids, since, until)
    cur = await db.execute("SELECT COUNT(*) FROM request_history" + where, args)
    total = int((await cur.fetchone() or (0,))[0] or 0)
    cap = max(1, int(cap))
    if total <= cap:
        stride = 1
        cur = await db.execute(
            "SELECT finished_at, queued_s, ttft_s, duration_s FROM request_history" + where,
            args,
        )
    else:
        stride = math.ceil(total / cap)
        cur = await db.execute(
            "SELECT finished_at, queued_s, ttft_s, duration_s FROM ("
            "  SELECT finished_at, queued_s, ttft_s, duration_s, "
            "         ROW_NUMBER() OVER (ORDER BY finished_at, id) AS rn"
            "  FROM request_history" + where +
            ") WHERE (rn - 1) % ? = 0",
            (*args, stride),
        )
    rows = [(float(r[0]), r[1], r[2], r[3]) for r in await cur.fetchall()]
    return TokenTimings(rows=rows, total=total, stride=stride)


#: ``earliest_token_finished_at``'s query. Served by the partial index
#: idx_request_history_token_finished (0034), whose ``WHERE token_id IS NOT
#: NULL`` this must keep implying; the migration's test EXPLAINs this string.
EARLIEST_TOKEN_FINISHED_SQL = (
    "SELECT MIN(finished_at) FROM request_history WHERE token_id IS NOT NULL"
)


async def earliest_token_finished_at(db: aiosqlite.Connection) -> float | None:
    """When per-TOKEN history begins: the oldest row that carries a token_id.

    Store-wide, like ``earliest_finished_at``, and for the same reason: it is
    a statement about the store -- token ids were first recorded at the 0033
    deploy, and retention prunes from the old end -- not about one key. The
    token page hatches its timing charts before this instant.
    """
    cur = await db.execute(EARLIEST_TOKEN_FINISHED_SQL)
    row = await cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


async def prune(db: aiosqlite.Connection, *, cutoff: float, max_rows: int) -> dict[str, int]:
    """Delete rows older than ``cutoff`` and, after that, any beyond ``max_rows``.

    Age first, then count, so the count cap trims the OLDEST survivors. Does
    not commit -- the caller's transaction owns that, as with the other
    pruners.
    """
    cur = await db.execute("DELETE FROM request_history WHERE finished_at < ?", (cutoff,))
    by_age = cur.rowcount or 0
    by_count = 0
    if max_rows > 0:
        cur = await db.execute(
            "DELETE FROM request_history WHERE id IN ("
            "  SELECT id FROM request_history ORDER BY finished_at DESC, id DESC"
            "  LIMIT -1 OFFSET ?"
            ")",
            (int(max_rows),),
        )
        by_count = cur.rowcount or 0
    return {"by_age": by_age, "by_count": by_count}


# ---------------------------------------------------------------------------
# Statistics -- pure
# ---------------------------------------------------------------------------

# Bucket edges in seconds. TTFT and ITL reuse vLLM's own histogram boundaries
# so the bars line up with what vLLM's dashboards show for the same model;
# duration reuses its e2e set. The trailing +Inf bucket is encoded as ``None``
# on the wire, exactly as the engine histograms are (JSON has no infinity).
TTFT_EDGES: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
)
ITL_EDGES: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
)
DURATION_EDGES: tuple[float, ...] = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0,
    50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
)


def quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile of an ascending sequence, or None if empty.

    Exact, from the samples themselves -- unlike ``hist_quantile`` over engine
    buckets, which can only interpolate inside a bucket. This is the reason to
    compute latency from per-request rows at all.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, q)) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return float(sorted_values[lo]) + frac * (float(sorted_values[hi]) - float(sorted_values[lo]))


def histogram(values: Iterable[float], edges: Sequence[float]) -> dict[str, Any]:
    """Cumulative buckets in the engine-histogram wire shape.

    ``le`` carries the edges plus ``None`` for +Inf; ``counts`` are cumulative
    along the boundaries, so the frontend's existing bucket renderer and
    quantile mirror consume this unchanged.
    """
    per_bucket = [0] * (len(edges) + 1)
    count = 0
    total = 0.0
    for v in values:
        count += 1
        total += v
        for i, edge in enumerate(edges):
            if v <= edge:
                per_bucket[i] += 1
                break
        else:
            per_bucket[len(edges)] += 1
    cumulative: list[int] = []
    running = 0
    for c in per_bucket:
        running += c
        cumulative.append(running)
    return {
        "le": [*edges, None],
        "counts": cumulative,
        "count": count,
        "sum": total,
    }


def distribution(values: Iterable[float], edges: Sequence[float]) -> dict[str, Any]:
    """Count, exact quantiles, mean and buckets for one latency series."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    return {
        "count": n,
        "p50": quantile(vals, 0.5),
        "p90": quantile(vals, 0.9),
        "p99": quantile(vals, 0.99),
        "mean": (sum(vals) / n) if n else None,
        "buckets": histogram(vals, edges),
    }


def itl_mean_of(row: dict[str, Any]) -> float | None:
    """Mean inter-token latency of ONE request, from the proxy's measurements.

    ``(duration - ttft) / (completion_tokens - 1)``: the decode phase divided by
    the gaps in it. This is a per-request MEAN, not the per-token distribution
    the engine's ITL histogram is -- contention inside a request averages out
    here. Every panel that shows it says "mean per request" for that reason.
    None when it cannot be computed: no first token, fewer than two tokens, or
    a decode phase of zero length (a non-streaming request has no first frame
    to time, so it lands here too).
    """
    ttft = row.get("ttft_s")
    dur = row.get("duration_s")
    n = row.get("completion_tokens") or 0
    if ttft is None or dur is None or n < 2:
        return None
    decode = float(dur) - float(ttft)
    if decode <= 0:
        return None
    return decode / (n - 1)


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The three distributions over a set of rows, plus their time span."""
    ttft = [float(r["ttft_s"]) for r in rows if r.get("ttft_s") is not None]
    durations = [float(r["duration_s"]) for r in rows if r.get("duration_s") is not None]
    itl = [v for v in (itl_mean_of(r) for r in rows) if v is not None]
    finished = [float(r["finished_at"]) for r in rows if r.get("finished_at") is not None]
    return {
        "count": len(rows),
        "oldest_epoch": min(finished) if finished else None,
        "newest_epoch": max(finished) if finished else None,
        "span_s": (max(finished) - min(finished)) if finished else None,
        "ttft": distribution(ttft, TTFT_EDGES),
        "itl": distribution(itl, ITL_EDGES),
        "duration": distribution(durations, DURATION_EDGES),
    }


# ---- token-rate statistics --------------------------------------------------
#
# The stats page asks how FAST the rig runs, which is not the question the
# tokens-per-minute rollup answers. These derive per-request prefill and
# generation speeds from the columns the proxy already records, and summarise
# a set of rates as average / max / mode.

# Geometric bin width for the mode. 1.05 = bins ~5% wide, so one width serves
# a 900 tok/s prefill and an 8 tok/s generation without per-model tuning.
MODE_BIN_RATIO = 1.05
# Below this many samples a "most common value" is noise wearing a statistic's
# clothes, so it is reported as absent rather than as a number.
MODE_MIN_SAMPLES = 5


def prefill_tps_of(row: dict[str, Any]) -> float | None:
    """Prefill speed of ONE request: ``prompt_tokens / ttft_s``.

    What this does NOT include, contrary to an earlier reading of it: the
    warden's admission queue. ``started_monotonic`` in app/proxy/routes.py is
    read once the scheduler slot is held, so waiting for that slot was never
    inside ``ttft_s`` -- it is recorded separately in ``queued_s`` (migration
    0032) and must not be subtracted here.

    What it DOES still include, and what no column can remove: the ENGINE's own
    waiting queue. vLLM's continuous batching keeps its own queue behind our
    admission gate, and time spent there lands in ``ttft_s`` with nothing at
    the proxy able to see it.

    Prefix caching does not merely skew this -- on a long-context workload it
    DOMINATES it, and the panel no longer calls the result a prefill rate.
    Measured on a production deployment: median prompt 46,363 tokens, median TTFT
    1.9s, i.e. ~26,000 tok/s. Prefilling 46k tokens through a 27B is ~2.5e15
    FLOPs; four A4000s at a generous 100 TFLOPS would need ~25 SECONDS. The
    tokens were served from the prefix cache, not computed. So this is a
    measure of CACHE HITS, not of compute, and every statistic over it --
    average, max and mode alike -- inherits that.

    None when it cannot be computed: no first token (an abort before the first
    frame, or a non-streaming request, which has no first frame to time), or an
    empty prompt.
    """
    raw_ttft = row.get("ttft_s")
    n = int(row.get("prompt_tokens") or 0)
    if raw_ttft is None or n <= 0:
        return None
    ttft = float(raw_ttft)
    if ttft <= 0:
        return None
    return n / ttft


def gen_tps_of(row: dict[str, Any]) -> float | None:
    """Generation speed of ONE request, in tokens per second.

    The reciprocal of ``itl_mean_of``: decode-phase tokens over decode-phase
    seconds is exactly one over the mean gap between them. Sharing that
    function rather than restating the arithmetic keeps the two panels'
    refusals identical -- a request with no ITL has no generation rate.
    """
    itl = itl_mean_of(row)
    if itl is None or itl <= 0:
        return None
    return 1.0 / itl


def mode_relative(
    values: Iterable[float],
    *,
    ratio: float = MODE_BIN_RATIO,
    min_samples: int = MODE_MIN_SAMPLES,
) -> float | None:
    """The most common rate, over bins whose width scales with magnitude.

    Bins are geometric (``floor(log(v) / log(ratio))``) so one setting covers
    every model on the box: 5% of 900 tok/s and 5% of 8 tok/s are both "near
    enough to be the same reading". The bin's geometric centre is reported --
    within 2.5% of every sample in it.

    Zero is its own bin rather than a dropped value: on the wall-clock basis an
    idle deployment HAS a modal throughput and it is zero, and dropping the
    idle minutes would report the rate of the busy ones while claiming to
    describe all of them.

    None when the sample is too small to have a mode, or when no bin holds
    more than one value -- a flat spread has no modal reading, and taking the
    lowest bin of size 1 would dress noise up as a typical one.
    """
    vals = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v)) and float(v) >= 0
    ]
    if len(vals) < min_samples:
        return None
    log_ratio = math.log(ratio)
    # None keys the zero bin; ints key the geometric ones.
    counts: dict[int | None, int] = {}
    for v in vals:
        key = None if v == 0 else math.floor(math.log(v) / log_ratio)
        counts[key] = counts.get(key, 0) + 1
    # Ties go to the SLOWER bin: two equally common readings and reporting the
    # faster one would flatter the deployment. The zero bin sorts below every
    # geometric one, which is what "slowest" means when it is in play.
    best_key, best_count = max(
        counts.items(),
        key=lambda kv: (kv[1], -(kv[0] if kv[0] is not None else -math.inf)),
    )
    if best_count < 2:
        return None
    return 0.0 if best_key is None else float(ratio ** (best_key + 0.5))


def rate_summary(values: Sequence[float]) -> dict[str, Any]:
    """Average, max and mode of one rate series, plus how many samples it had.

    Empty is reported as absent rather than as zero: no requests in the window
    is not the same claim as a deployment that ran at 0 tok/s.
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "avg": None, "max": None, "mode": None}
    return {
        "count": len(vals),
        "avg": sum(vals) / len(vals),
        "max": max(vals),
        "mode": mode_relative(vals),
    }
