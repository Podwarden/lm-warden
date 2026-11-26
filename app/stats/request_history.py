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
import json
import logging
import math
import re
import sqlite3
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from app.cache_obs.classify import BLOCK_AWARE_ABOVE, DIVERGENCE_CAUSES, OUTCOMES
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.lens.causes import CAUSES
from app.lens.probes import SPAN_CLASSES

logger = logging.getLogger(__name__)

#: The writer's own lock wait, instead of open_db's 30 s (#295). A statement
#: waiting on a held write lock cannot be cancelled -- it runs on the
#: connection's worker thread -- and the connection's close waits behind it, so
#: this is what bounds a write that shutdown cancels. A write that hits it is
#: retried (the rows go back on the queue), never dropped, like the ledger's.
HISTORY_BUSY_TIMEOUT_MS = 2000
#: The bound on the final flush at shutdown, like the ledger loop's last flush.
FINAL_FLUSH_TIMEOUT_S = 2.0
#: A busy retry loses nothing, so it is logged at most this often.
_BUSY_LOG_EVERY_S = 60.0


def _is_lock_contention(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


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
    "cached_tokens",
    "cache_est_tokens",
    "dp_rank",
    "inflight_same_rank_at_start",
    "gap_s",
    "session_source",
    "parent_session_id",
    "reusable_tokens",
    "reusable_tokens_fleet",
    "reusable_tokens_own",
    "cache_outcome",
    "cache_outcome_own",
    "diverged_at",
    "diverged_at_own",
    "cached_source",
    "cached_ttft_est_tokens",
    "thinking_forced_off",
    "session_key",
    "parent_session_key",
    "turn_index",
    "batch_id",
    "tools_out",
    "tools_in",
    "cobs2_head",
    "cobs2_node",
    "cobs2_parent_node",
    "cobs2_edge",
    "cobs2_reply_msg",
    "cobs2_compaction_request",
    "cobs2_divergence_cause",
    "cobs2_divergence_cause_own",
    "cobs2_session",
    "cobs2_conv",
    "cobs2_shape",
    "cobs2_parent_req",
    "cobs2_sess_ref",
    "cobs2_toolset",
    "relayed",
    "upstream",
    "served_gpu_tokens",
    "served_ram_tokens",
    "served_nvme_tokens",
    "served_remote_tokens",
    "load_s",
    "node_id",
    "prompt_diet",
    "guard",
    "kv_isolated",
    "kv_iso_cost_tokens",
    "lens_break",
    "lens_cause",
    "lens_span_class",
    "lens_span_line",
    "lens_lost_tokens",
    "lens_cf_tokens",
    "lens_other_breaks",
    "cache_write_tokens",
    "cache_marks",
    "lens_repeat_results",
    "max_tokens_req",
    "reasoning_tokens",
    "reply_chars",
    "calls_n",
    "calls_bad",
    "tools_lint",
    "lens_version",
)
# The forest's six columns (session_key .. tools_in, just before the cobs2 tail of _COLUMNS) are
# written with every row but read only by the forest route, which selects them itself: the v2
# readers below would otherwise drag two JSON blobs per row through every stats query.
_FOREST_COLUMNS = (
    "session_key",
    "parent_session_key",
    "turn_index",
    "batch_id",
    "tools_out",
    "tools_in",
)
# cobs2 L1 ids (migration 0052): written with every row, internal, never read by
# the v2 readers (no /api change). Hex ids only, never text.
_COBS2_COLUMNS = (
    "cobs2_head",
    "cobs2_node",
    "cobs2_parent_node",
    "cobs2_edge",
    "cobs2_reply_msg",
    "cobs2_compaction_request",
    "cobs2_divergence_cause",
    "cobs2_divergence_cause_own",
)
# Sessions pages (migration 0053): read only by app/sessions, never by the v2
# readers. Ids, sizes and counts only.
_SESSIONS_COLUMNS = (
    "cobs2_session",
    "cobs2_conv",
    "cobs2_shape",
    "cobs2_parent_req",
    "cobs2_sess_ref",
    "cobs2_toolset",
    "relayed",
    "upstream",
    "served_gpu_tokens",
    "served_ram_tokens",
    "served_nvme_tokens",
    "served_remote_tokens",
    "load_s",
    "node_id",
)
# Prompt diet (migration 0054): what a key's diet cut; written with every row,
# read only by ad hoc analysis.
_DIET_COLUMNS = ("prompt_diet", "guard")
# Per-key KV isolation (migration 0060): whether the turn ran isolated and what isolation cost
# it in recomputed tokens. Read by the sessions and key cache pages, never by the v2 readers.
_KV_ISO_COLUMNS = ("kv_isolated", "kv_iso_cost_tokens")
# Prompt Lens analysis (migration 0063): layer 1 codes and counts, relay cache
# writes and marks, layer 3 counts. Read by app/sessions and the lens only,
# never by the v2 readers. Codes, ints and JSON ints only, never text.
_LENS_COLUMNS = (
    "lens_break",
    "lens_cause",
    "lens_span_class",
    "lens_span_line",
    "lens_lost_tokens",
    "lens_cf_tokens",
    "lens_other_breaks",
    "cache_write_tokens",
    "cache_marks",
    "lens_repeat_results",
    "max_tokens_req",
    "reasoning_tokens",
    "reply_chars",
    "calls_n",
    "calls_bad",
    "tools_lint",
    "lens_version",
)
_COBS2_EDGES = frozenset({"new", "continue", "fork", "compacted_from", "subagent"})
_READ_COLUMNS = _COLUMNS[
    : -len(_FOREST_COLUMNS)
    - len(_COBS2_COLUMNS)
    - len(_SESSIONS_COLUMNS)
    - len(_DIET_COLUMNS)
    - len(_KV_ISO_COLUMNS)
    - len(_LENS_COLUMNS)
]
_SELECT = "SELECT " + ", ".join(_READ_COLUMNS) + " FROM request_history"
# INSERT OR IGNORE: the id is the registry's uuid, and a record that somehow
# reaches the writer twice must not turn into an IntegrityError that costs the
# whole batch.
_INSERT = (
    "INSERT OR IGNORE INTO request_history(" + ", ".join(_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" for _ in _COLUMNS) + ")"
)


def _json_text(v: Any) -> str | None:
    """A JSON column value: a str passes through, a list/dict is encoded."""
    if v is None or isinstance(v, str):
        return v
    try:
        return json.dumps(v, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _opt_str(v: Any) -> str | None:
    """An optional string column value: str passes through, anything else → None."""
    if v is None or isinstance(v, str):
        return v
    return None


def _version_text(v: Any) -> str | None:
    """A prompt version (0065): ``h:<name>`` or 8 hex, printable ASCII, at most 66 chars."""
    if isinstance(v, str) and 0 < len(v) <= 66 and all(32 <= ord(c) < 127 for c in v):
        return v
    return None


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

    def _opt_bool(v: Any) -> int | None:
        return None if v is None else (1 if v else 0)

    def _opt_int(v: Any) -> int | None:
        try:
            return None if v is None else int(v)
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
        _opt_int(record.get("cached_tokens")),
        _opt_int(record.get("cache_est_tokens")),
        _opt_int(record.get("dp_rank")),
        _opt_int(record.get("inflight_same_rank_at_start")),
        _float(record.get("gap_s")),
        record.get("session_source"),
        record.get("parent_session_id"),
        _opt_int(record.get("reusable_tokens")),
        _opt_int(record.get("reusable_tokens_fleet")),
        _opt_int(record.get("reusable_tokens_own")),
        record.get("cache_outcome"),
        record.get("cache_outcome_own"),
        _opt_int(record.get("diverged_at")),
        _opt_int(record.get("diverged_at_own")),
        record.get("cached_source"),
        _opt_int(record.get("cached_ttft_est_tokens")),
        1 if record.get("thinking_forced_off") else 0,
        _opt_str(record.get("session_key")),
        _opt_str(record.get("parent_session_key")),
        _opt_int(record.get("turn_index")),
        _opt_str(record.get("batch_id")),
        _json_text(record.get("tools_out")),
        _json_text(record.get("tools_in")),
        _opt_str(record.get("cobs2_head")),
        _opt_str(record.get("cobs2_node")),
        _opt_str(record.get("cobs2_parent_node")),
        edge
        if isinstance(edge := record.get("cobs2_edge"), str) and edge in _COBS2_EDGES
        else None,
        _opt_str(record.get("cobs2_reply_msg")),
        _opt_bool(record.get("cobs2_compaction_request")),
        _cause(record.get("cobs2_divergence_cause")),
        _cause(record.get("cobs2_divergence_cause_own")),
        _opt_str(record.get("cobs2_session")),
        _opt_str(record.get("cobs2_conv")),
        _opt_str(record.get("cobs2_shape")),
        _opt_str(record.get("cobs2_parent_req")),
        _opt_str(record.get("cobs2_sess_ref")),
        _opt_str(record.get("cobs2_toolset")),
        1 if record.get("relayed") else None,
        _opt_str(record.get("upstream")),
        _opt_int(record.get("served_gpu_tokens")),
        _opt_int(record.get("served_ram_tokens")),
        _opt_int(record.get("served_nvme_tokens")),
        _opt_int(record.get("served_remote_tokens")),
        _float(record.get("load_s")),
        _opt_str(record.get("node_id")),
        _opt_str(record.get("prompt_diet")),
        _opt_str(record.get("guard")),
        _opt_bool(record.get("kv_isolated")),
        _opt_int(record.get("kv_iso_cost_tokens")),
        _lens_break(record.get("lens_break")),
        _code(record.get("lens_cause"), _LENS_CAUSES),
        _code(record.get("lens_span_class"), _SPAN_CLASSES),
        _opt_int(record.get("lens_span_line")),
        _opt_int(record.get("lens_lost_tokens")),
        _opt_int(record.get("lens_cf_tokens")),
        _opt_int(record.get("lens_other_breaks")),
        _opt_int(record.get("cache_write_tokens")),
        _opt_int(record.get("cache_marks")),
        _opt_int(record.get("lens_repeat_results")),
        _opt_int(record.get("max_tokens_req")),
        _opt_int(record.get("reasoning_tokens")),
        _opt_int(record.get("reply_chars")),
        _opt_int(record.get("calls_n")),
        _json_text(record.get("calls_bad")),
        _json_text(record.get("tools_lint")),
        _version_text(record.get("lens_version")),
    )


#: A key's tool name is re-upserted at most this often (keeps ``last_seen``
#: within a day of the truth for retention at ~one write per tool per day).
TOOL_NAME_REFRESH_S = 86_400.0
_TOOL_SEEN_MAX = 65_536
_PENDING_TOOLSETS_MAX = 4096
_COMMITTED_MAX = 20_000
_UPSERT_TOOLSET = (
    "INSERT INTO cobs2_toolsets(digest, tools, last_seen) VALUES (?,?,?) "
    "ON CONFLICT(digest) DO UPDATE SET last_seen=max(last_seen, excluded.last_seen)"
)
_UPSERT_TOOL = (
    "INSERT INTO session_tools(token_id, tool_hmac, name, family, size, first_seen, last_seen) "
    "VALUES (?,?,?,?,?,?,?) ON CONFLICT(token_id, tool_hmac) DO UPDATE SET name=excluded.name, "
    "family=excluded.family, size=excluded.size, last_seen=max(last_seen, excluded.last_seen)"
)


_LENS_CAUSES = frozenset(CAUSES)
_SPAN_CLASSES = frozenset(SPAN_CLASSES)
_LENS_BREAK_RE = re.compile(r"sys|tools|node:\d{1,9}")


def _code(v: Any, known: frozenset[str]) -> str | None:
    """A code column: only a known code is stored, anything else is NULL."""
    return v if isinstance(v, str) and v in known else None


def _lens_break(v: Any) -> str | None:
    return v if isinstance(v, str) and _LENS_BREAK_RE.fullmatch(v) else None


def _cause(v: Any) -> str | None:
    return v if isinstance(v, str) and v in DIVERGENCE_CAUSES else None


def _row_dict(row: Iterable[Any]) -> dict[str, Any]:
    d = dict(zip(_READ_COLUMNS, row, strict=True))
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
        self._last_busy_log = float("-inf")
        #: digest -> (tools JSON, last_seen): upserted with the next batch.
        self._pending_toolsets: dict[str, tuple[str, float]] = {}
        #: (token_id, tool_hmac) -> (name, family, size, first_seen, last_seen).
        self._pending_tools: dict[tuple[str, str], tuple[str, str, int, float, float]] = {}
        #: (token_id, tool_hmac) -> (size, refreshed_at): what was last queued.
        self._tool_seen: OrderedDict[tuple[str, str], tuple[int, float]] = OrderedDict()
        self._committed: deque[str] = deque()
        self._committed_set: set[str] = set()
        self._commit_cond = asyncio.Condition()

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
            self._note_toolset(record, params[1])
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

    def _note_toolset(self, record: dict[str, Any], finished_at: float) -> None:
        """Queue the toolset and any due tool-name upserts; a bad value costs
        only these side rows, never the history row."""
        try:
            digest = record.get("cobs2_toolset")
            tools = record.get("cobs2_toolset_tools")
            if not isinstance(digest, str) or not isinstance(tools, list):
                return
            entries = [
                (str(e[0]), str(e[1]), int(e[2]))
                for e in tools
                if isinstance(e, list | tuple) and len(e) == 3
            ]
            if (
                digest in self._pending_toolsets
                or len(self._pending_toolsets) < _PENDING_TOOLSETS_MAX
            ):
                self._pending_toolsets[digest] = (
                    json.dumps([list(e) for e in entries], separators=(",", ":")),
                    finished_at,
                )
            token_id = record.get("token_id")
            names = record.get("cobs2_toolset_names")
            if not isinstance(token_id, str) or not token_id or not isinstance(names, list):
                return
            for row in self._tool_upserts_due(token_id, entries, names, finished_at):
                key = (row[0], row[1])
                old = self._pending_tools.get(key)
                first = min(old[3], row[5]) if old is not None else row[5]
                self._pending_tools[key] = (row[2], row[3], row[4], first, row[6])
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail a request
            logger.debug("request history: could not queue a toolset", exc_info=True)

    def _tool_upserts_due(
        self,
        token_id: str,
        entries: list[tuple[str, str, int]],
        names: list[Any],
        finished_at: float,
    ) -> list[tuple[str, str, str, str, int, float, float]]:
        """The (token, tool) name rows to write: unseen, resized, or not
        refreshed for ``TOOL_NAME_REFRESH_S``."""
        out: list[tuple[str, str, str, str, int, float, float]] = []
        seen = self._tool_seen
        for (h, fam, size), name in zip(entries, names, strict=False):
            if not isinstance(name, str) or not name:
                continue
            key = (token_id, h)
            prev = seen.get(key)
            if prev is not None and prev[0] == size and finished_at - prev[1] < TOOL_NAME_REFRESH_S:
                seen.move_to_end(key)
                continue
            seen[key] = (size, finished_at)
            seen.move_to_end(key)
            while len(seen) > _TOOL_SEEN_MAX:
                seen.popitem(last=False)
            out.append((token_id, h, name, fam, size, finished_at, finished_at))
        return out

    async def wait_for(self, request_id: str, timeout_s: float) -> bool:
        """True once a batch holding ``request_id`` has committed (in this
        process), waiting up to ``timeout_s``."""
        if request_id in self._committed_set:
            return True
        try:
            async with asyncio.timeout(max(0.0, timeout_s)):
                async with self._commit_cond:
                    await self._commit_cond.wait_for(lambda: request_id in self._committed_set)
        except TimeoutError:
            pass
        return request_id in self._committed_set

    async def wait_commit(self, timeout_s: float) -> None:
        """Return when a batch of rows has committed (in this process), or after ``timeout_s``.
        A caller loops over its own check: it may wake for a batch that is not the one it
        waits for."""
        try:
            async with asyncio.timeout(max(0.0, timeout_s)):
                async with self._commit_cond:
                    await self._commit_cond.wait()
        except TimeoutError:
            pass

    async def _mark_committed(self, batch: list[tuple[Any, ...]]) -> None:
        for params in batch:
            rid = params[0]
            if rid in self._committed_set:
                continue
            self._committed.append(rid)
            self._committed_set.add(rid)
            while len(self._committed) > _COMMITTED_MAX:
                self._committed_set.discard(self._committed.popleft())
        async with self._commit_cond:
            self._commit_cond.notify_all()

    def _drain(self) -> list[tuple[Any, ...]]:
        batch: list[tuple[Any, ...]] = []
        while True:
            try:
                batch.append(self._q.get_nowait())
            except asyncio.QueueEmpty:
                return batch

    def _requeue(self, batch: list[tuple[Any, ...]]) -> None:
        """Put rows back for the next write. INSERT OR IGNORE makes a row that
        did land after all harmless; a full queue drops (and counts) the rest."""
        for i, params in enumerate(batch):
            try:
                self._q.put_nowait(params)
            except asyncio.QueueFull:
                self.dropped += len(batch) - i
                return

    def _note_busy(self) -> None:
        now = time.monotonic()
        if now - self._last_busy_log >= _BUSY_LOG_EVERY_S:
            self._last_busy_log = now
            logger.warning("request history: write deferred, database busy; retrying")

    async def _write(self, batch: list[tuple[Any, ...]]) -> None:
        toolsets = dict(self._pending_toolsets)
        tools = dict(self._pending_tools)
        async with open_db(self.db_path) as db:
            await db.execute(f"PRAGMA busy_timeout = {int(HISTORY_BUSY_TIMEOUT_MS)}")
            await db.executemany(_INSERT, batch)
            if toolsets:
                await db.executemany(
                    _UPSERT_TOOLSET, [(d, t, ls) for d, (t, ls) in toolsets.items()]
                )
            if tools:
                await db.executemany(_UPSERT_TOOL, [(k[0], k[1], *v) for k, v in tools.items()])
            await db.commit()
        self.written += len(batch)
        for d, v in toolsets.items():
            if self._pending_toolsets.get(d) == v:
                del self._pending_toolsets[d]
        for k, tv in tools.items():
            if self._pending_tools.get(k) == tv:
                del self._pending_tools[k]
        await self._mark_committed(batch)

    async def _write_or_requeue(self, batch: list[tuple[Any, ...]]) -> int:
        """One write. Lock contention re-queues the batch; any other failure
        logs and drops it. A cancel re-queues it, so the final flush still
        sees it, and propagates."""
        try:
            await self._write(batch)
        except asyncio.CancelledError:
            self._requeue(batch)
            raise
        except sqlite3.OperationalError as exc:
            if not _is_lock_contention(exc):
                logger.exception("request history: dropped a batch of %d record(s)", len(batch))
                return 0
            self._requeue(batch)
            self._note_busy()
            return 0
        except Exception:
            logger.exception("request history: dropped a batch of %d record(s)", len(batch))
            return 0
        return len(batch)

    async def flush(self) -> int:
        """Write everything queued right now. Returns the number of rows sent.

        A failed write logs and drops the batch. Re-queueing would turn a
        persistently broken volume into an ever-growing queue and, once full,
        into the same drop -- with the log noise multiplied. A BUSY database is
        the exception: the rows go back on the queue for the next write.
        """
        batch = self._drain()
        if not batch:
            return 0
        return await self._write_or_requeue(batch)

    async def run_forever(self) -> None:
        """Drain the queue in batches, at most once per ``flush_interval_s``.

        Waits for the first record, then sleeps one interval so a burst lands
        as one transaction rather than one fsync per request. Cancellation
        performs a final flush so a clean shutdown loses nothing queued;
        bounded (#295), since under a held write lock it would otherwise add
        its whole lock wait to a shutdown that must fit Docker's stop grace.
        """
        try:
            while True:
                batch = [await self._q.get()]
                try:
                    await asyncio.sleep(self.flush_interval_s)
                except asyncio.CancelledError:
                    # Taken off the queue but not written: the final flush
                    # below only sees what is queued.
                    self._requeue(batch)
                    raise
                batch.extend(self._drain())
                await self._write_or_requeue(batch)
        except asyncio.CancelledError:
            try:
                # `asyncio.timeout`, not `wait_for`: see app/db/database.py.
                async with asyncio.timeout(FINAL_FLUSH_TIMEOUT_S):
                    await self.flush()
            except BaseException:  # noqa: BLE001 -- shutdown must not hang on the volume
                logger.debug("request history: final flush failed", exc_info=True)
            if self.pending():
                logger.warning(
                    "request history: %d record(s) not written at shutdown "
                    "(the database stayed busy)",
                    self.pending(),
                )
            raise


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


#: Every local stats reader sees local rows only: a relayed (router
#: passthrough) row's timings and cache are the upstream's (sessions pages, R4).
LOCAL_ONLY = " AND relayed IS NULL"


def _model_clause(model_ids: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    """``LOCAL_ONLY`` plus `` AND model_id IN (...)`` for a selection.

    No model filter when unfiltered, rather than one that happens to match
    every model.
    """
    if model_ids is None:
        return LOCAL_ONLY, ()
    if not model_ids:
        # `IN ()` is a syntax error in SQLite and a bare 1=1 would silently
        # widen to every model -- the substitution every stats route exists to
        # prevent.
        return LOCAL_ONLY + " AND 0", ()
    return (
        LOCAL_ONLY + f" AND model_id IN ({','.join('?' for _ in model_ids)})",
        tuple(model_ids),
    )


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
        "SELECT " + ", ".join(_READ_COLUMNS) + " FROM ("
        "  SELECT " + ", ".join(_READ_COLUMNS) + ", "
        "         ROW_NUMBER() OVER (ORDER BY finished_at DESC, id DESC) AS rn"
        "  FROM request_history WHERE finished_at >= ?"
        + where
        + ") WHERE (rn - 1) % ? = 0 ORDER BY rn ASC LIMIT ?",
        (since, *args, stride, limit),
    )
    rows = await cur.fetchall()
    return WindowRows(rows=[_row_dict(r) for r in rows], total=total, stride=stride)


_OUTCOMES = OUTCOMES


async def query_cache_summary(
    db: aiosqlite.Connection,
    *,
    since: float,
    model_ids: Sequence[str] | None,
    token_id: str | None = None,
    until: float | None = None,
) -> list[dict[str, Any]]:
    """Cache observation per model over a window (spec §5). ``until`` ends the
    window (default: now). ``token_id`` set =
    the OWN lens: own-scope columns, that token's rows only, no cross-token
    fields. Unknown values stay out of sums (NULL is not 0)."""
    reuse_col = "reusable_tokens_own" if token_id else "reusable_tokens"
    outcome_col = "cache_outcome_own" if token_id else "cache_outcome"
    # The fleet-scope break point can rest on other keys' messages: own lens
    # reads its own.
    div_col = "diverged_at_own" if token_id else "diverged_at"
    clause, params = _model_clause(model_ids)
    where = " WHERE finished_at >= ?" + clause
    args: tuple[Any, ...] = (since, *params)
    if until is not None:
        where += " AND finished_at <= ?"
        args += (until,)
    if token_id:
        where += " AND token_id = ?"
        args += (token_id,)
    fact = "COALESCE(cached_tokens, cached_ttft_est_tokens)"
    # Efficiency inputs come from rows where BOTH sides are known (a pre-0046
    # row has C but no R), so served/reusable can never mix rows or exceed 1.
    # Rule 1 (R := C when C > R) applies to measured C only, as in classify():
    # a TTFT estimate on a cold request would otherwise count as fully reused,
    # so an estimated row is capped at its R instead and never raises it.
    both = f"{fact} IS NOT NULL AND {reuse_col} IS NOT NULL"
    est = "cached_source = 'estimated'"
    eff = (
        f"COALESCE(SUM(CASE WHEN {both} THEN CASE WHEN {est} "
        f"THEN MIN({fact}, {reuse_col}) ELSE {fact} END END),0), "
        f"COALESCE(SUM(CASE WHEN {both} THEN CASE WHEN {est} "
        f"THEN {reuse_col} ELSE MAX({reuse_col}, {fact}) END END),0)"
    )
    cur = await db.execute(
        "SELECT model_id, COUNT(*), COALESCE(SUM(prompt_tokens),0), "
        f"COALESCE(SUM({fact}),0), "
        f"COALESCE(SUM(CASE WHEN {fact} IS NOT NULL THEN prompt_tokens END),0), "
        "SUM(CASE WHEN cached_source='engine' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN cached_source='estimated' THEN 1 ELSE 0 END), "
        f"{eff}, "
        # The prefix broke mid-prompt yet enough came before the break to
        # read hit or partial: the rewrite's waste is otherwise invisible.
        f"SUM(CASE WHEN {div_col} IS NOT NULL AND {outcome_col} IN ('hit','partial') "
        "THEN 1 ELSE 0 END), "
        # B4: engine-measured only (no estimate mixed in), and the prompts of
        # the same rows -- the page's one cache figure.
        "COALESCE(SUM(cached_tokens),0), "
        "COALESCE(SUM(CASE WHEN cached_tokens IS NOT NULL THEN prompt_tokens END),0), "
        # Own lens: tokens reused from the key's OWN earlier prompts.
        "COALESCE(SUM(CASE WHEN cached_tokens IS NOT NULL AND reusable_tokens_own IS NOT NULL "
        "THEN MIN(cached_tokens, reusable_tokens_own) END),0), "
        "COALESCE(SUM(CASE WHEN cached_tokens IS NOT NULL AND reusable_tokens_own IS NOT NULL "
        "THEN prompt_tokens END),0), "
        "SUM(CASE WHEN cached_tokens IS NOT NULL AND reusable_tokens_own IS NOT NULL "
        "THEN 1 ELSE 0 END) "
        f"FROM request_history{where} GROUP BY model_id",
        args,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in await cur.fetchall():
        (
            mid,
            n,
            prompt,
            cached,
            known_prompt,
            meas,
            est,
            reused,
            reuse,
            deep,
            m_cached,
            m_prompt,
            own_reused,
            own_prompt,
            own_n,
        ) = row
        out[mid] = {
            "model_id": mid,
            "requests": n,
            "prompt_tokens": prompt,
            #: SUM of every known C (engine or estimate); "from cache" divides
            #: it by known_prompt_tokens, the prompts of those same rows
            "cached_tokens": cached,
            "known_prompt_tokens": known_prompt,
            "measured_requests": meas or 0,
            "estimated_requests": est or 0,
            #: B4: engine-measured cached tokens and the prompts of those rows
            "measured_cached_tokens": m_cached,
            "measured_prompt_tokens": m_prompt,
            #: efficiency = reused_tokens / reusable_tokens (same rows)
            "reused_tokens": reused,
            "reusable_tokens": reuse,
            "outcomes": dict.fromkeys(_OUTCOMES, 0),
            #: hit/partial requests whose prefix still broke deep in the prompt
            "deep_diverged": deep or 0,
            "by_rank": [],
        }
        if token_id:
            #: Own lens only: min(cached, reusable_own) and its population.
            out[mid]["own_reused_tokens"] = own_reused
            out[mid]["own_known_prompt_tokens"] = own_prompt
            out[mid]["own_known_requests"] = own_n or 0
    cur = await db.execute(
        f"SELECT model_id, {outcome_col}, COUNT(*) FROM request_history{where} "
        f"AND {outcome_col} IS NOT NULL GROUP BY model_id, {outcome_col}",
        args,
    )
    for mid, oc, n in await cur.fetchall():
        if mid in out and oc in out[mid]["outcomes"]:
            out[mid]["outcomes"][oc] = n
    # cobs2: why prefixes broke, where lineage explains it. Counted over the
    # broken-prefix population only -- the rows the UI's share is taken of
    # (prefix changed, the relabelled misses, deep-diverged hits/partials). A
    # compacted hit with no divergence keeps its row label but is
    # informational, not a broken prefix, so it is left out here.
    cause_col = "cobs2_divergence_cause_own" if token_id else "cobs2_divergence_cause"
    for mid in out:
        out[mid]["divergence_causes"] = dict.fromkeys(DIVERGENCE_CAUSES, 0)
    cur = await db.execute(
        f"SELECT model_id, {cause_col}, COUNT(*) FROM request_history{where} "
        f"AND {cause_col} IS NOT NULL AND ("
        f"{outcome_col} IN ('diverged','compacted','harness_rewrite') "
        f"OR ({outcome_col} IN ('hit','partial') AND {div_col} IS NOT NULL)"
        f") GROUP BY model_id, {cause_col}",
        args,
    )
    for mid, cause, n in await cur.fetchall():
        if mid in out and cause in out[mid]["divergence_causes"]:
            out[mid]["divergence_causes"][cause] = n
    cur = await db.execute(
        f"SELECT model_id, dp_rank, COUNT(*), COALESCE(SUM({fact}),0), {eff} "
        f"FROM request_history{where} "
        "AND dp_rank IS NOT NULL GROUP BY model_id, dp_rank ORDER BY dp_rank",
        args,
    )
    for mid, rank, n, cached, reused, reuse in await cur.fetchall():
        if mid in out:
            out[mid]["by_rank"].append(
                {
                    "dp_rank": rank,
                    "requests": n,
                    "cached_tokens": cached,
                    "reused_tokens": reused,
                    "reusable_tokens": reuse,
                }
            )
    if not token_id:
        for mid in out:
            out[mid]["top_diverging"] = []
        cur = await db.execute(
            "SELECT model_id, token_id, MAX(token_name), COUNT(*) AS n FROM request_history"
            f"{where} AND (cache_outcome IN ('diverged','harness_rewrite') "
            "OR cobs2_divergence_cause='harness_rewrite') AND token_id IS NOT NULL "
            "GROUP BY model_id, token_id ORDER BY n DESC",
            args,
        )
        for mid, tid, tname, n in await cur.fetchall():
            if mid in out and len(out[mid]["top_diverging"]) < 3:
                out[mid]["top_diverging"].append(
                    {"token_id": tid, "token_name": tname, "requests": n}
                )
    return list(out.values())


def _engine_cache_block(state: Any, model: Any) -> tuple[int | None, bool, int | None]:
    """(B, drops the last unit, match unit) of the engine run serving ``model`` now, the
    values classify() used (app/cache_obs/block_size.py). B is None when
    unknown or a classic small page (the rules ignore it there too), so the
    page only explains a granularity that changes the outcomes."""
    sizes = getattr(state, "engine_block_sizes", None) if state is not None else None
    if sizes is None or model is None:
        return None, False, None
    from app.proxy.routes_dp import engine_epoch

    epoch = engine_epoch(state, model)
    block = sizes.get(model.id, epoch)
    if block is None or block <= BLOCK_AWARE_ABOVE:
        return None, False, None
    unit = sizes.match_unit(model.id, epoch) or block
    return block, sizes.drops_last_block(model.id, epoch), unit


async def decorate_cache_rows(
    db: aiosqlite.Connection,
    rows: list[dict[str, Any]],
    *,
    learned: Any,
    hint: float,
    state: Any = None,
) -> None:
    """Add ``model``, ``backend``, ``prefill_saved_s``, ``rate_source`` and the
    engine's cache granularity to each cache-summary row in place. ``learned``
    is the app's PrefillModelState (or None before it exists); ``hint`` is
    ``settings.prefill_tok_s_hint``; ``state`` is app.state, for the block size
    learned for the engine run serving now. Shared by the operator and the
    per-token routes so they stay identical."""
    repo = ModelRepo(db)
    for s in rows:
        m = await repo.get(s["model_id"])
        s["model"] = m.served_model_name if m else s["model_id"]
        s["backend"] = getattr(m, "backend", None) if m else None
        (
            s["engine_block_tokens"],
            s["engine_drops_last_block"],
            s["engine_match_unit_tokens"],
        ) = _engine_cache_block(state, m)
        rate, src = learned.rate_for(s["model_id"], hint) if learned is not None else (hint, "hint")
        known = s["measured_requests"] + s["estimated_requests"] > 0
        # No row with a known C: prefill saved is unknown, not 0 seconds.
        s["prefill_saved_s"] = s["cached_tokens"] / rate if known and rate > 0 else None
        s["rate_source"] = src


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
    the distribution of the window. Five columns a row kept a full 7d well under
    a second when measured at 200 000 rows; at the 1 000 000-row cap it is
    unmeasured (follow-up).
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


async def session_linkable(db: aiosqlite.Connection, ids: Sequence[str]) -> set[str]:
    """The ids among ``ids`` whose row joined a session (sessions pages): one
    primary-key query per 500 ids."""
    out: set[str] = set()
    ids = list(ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        cur = await db.execute(
            "SELECT id FROM request_history WHERE id IN ("
            + ",".join("?" for _ in chunk)
            + ") AND cobs2_session IS NOT NULL",
            chunk,
        )
        out.update(r[0] for r in await cur.fetchall())
    return out


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
    where = f" WHERE token_id IN ({marks}) AND finished_at >= ? AND finished_at < ?" + LOCAL_ONLY
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
            "  FROM request_history" + where + ") WHERE (rn - 1) % ? = 0",
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


_prune_gen = 0


def prune_generation() -> int:
    """Bumped (``bump_prune_generation``) after a committed prune that may have deleted rows the session forest still
    holds (inside its 7 d window): the forest then rebuilds instead of keeping them (re-review 2, R3)."""
    return _prune_gen


def bump_prune_generation() -> None:
    global _prune_gen
    _prune_gen += 1


async def prune(db: aiosqlite.Connection, *, cutoff: float, max_rows: int) -> dict[str, Any]:
    """Delete rows older than ``cutoff`` and, after that, any beyond ``max_rows``.

    Age first, then count, so the count cap trims the OLDEST survivors. Does
    not commit -- the caller's transaction owns that, as with the other
    pruners. ``newest_deleted`` is the newest ``finished_at`` deleted (None:
    nothing was), so the caller can tell whether the session forest's window
    was touched.
    """
    cur = await db.execute(
        "SELECT MAX(finished_at) FROM request_history WHERE finished_at < ?", (cutoff,)
    )
    (newest,) = await cur.fetchone() or (None,)
    cur = await db.execute("DELETE FROM request_history WHERE finished_at < ?", (cutoff,))
    by_age = cur.rowcount or 0
    by_count = 0
    if max_rows > 0:
        cur = await db.execute(
            "SELECT MAX(finished_at) FROM (SELECT finished_at FROM request_history"
            " ORDER BY finished_at DESC, id DESC LIMIT -1 OFFSET ?)",
            (int(max_rows),),
        )
        (n2,) = await cur.fetchone() or (None,)
        if n2 is not None:
            newest = n2 if newest is None else max(newest, n2)
        cur = await db.execute(
            "DELETE FROM request_history WHERE id IN ("
            "  SELECT id FROM request_history ORDER BY finished_at DESC, id DESC"
            "  LIMIT -1 OFFSET ?"
            ")",
            (int(max_rows),),
        )
        by_count = cur.rowcount or 0
    # Toolsets and tool names: a day of slack covers the name refresh
    # interval, so a tool still in use is never pruned.
    side_cutoff = cutoff - TOOL_NAME_REFRESH_S
    await db.execute("DELETE FROM cobs2_toolsets WHERE last_seen < ?", (side_cutoff,))
    await db.execute("DELETE FROM session_tools WHERE last_seen < ?", (side_cutoff,))
    # Prompt Lens scores go with the requests they were scored from: a session
    # whose last request is older than anything left (age or count cap).
    await db.execute(
        "DELETE FROM lens_findings WHERE EXISTS (SELECT 1 FROM lens_session_scores s"
        " WHERE s.key_id = lens_findings.key_id AND s.sess_ref = lens_findings.sess_ref"
        " AND s.last_at < COALESCE((SELECT MIN(finished_at) FROM request_history), ?))",
        (cutoff,),
    )
    await db.execute(
        "DELETE FROM lens_session_scores WHERE last_at < "
        "COALESCE((SELECT MIN(finished_at) FROM request_history), ?)",
        (cutoff,),
    )
    return {"by_age": by_age, "by_count": by_count, "newest_deleted": newest}


# ---------------------------------------------------------------------------
# Statistics -- pure
# ---------------------------------------------------------------------------

# Bucket edges in seconds. TTFT and ITL reuse vLLM's own histogram boundaries
# so the bars line up with what vLLM's dashboards show for the same model;
# duration reuses its e2e set. The trailing +Inf bucket is encoded as ``None``
# on the wire, exactly as the engine histograms are (JSON has no infinity).
TTFT_EDGES: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
)
ITL_EDGES: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
)
DURATION_EDGES: tuple[float, ...] = (
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
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
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) >= 0]
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
