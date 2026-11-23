"""Prompt Lens "keep captures" (rulings G-R4, G-R12): a key's last finished requests, on disk.

An admin-only per-key switch (``api_tokens.keep_captures``, migration 0061; its
value is the switch's generation, 0 = off). For a key with it on, the section
store (app/proxy/prompt_lens_sections.py) hands each finished request's record to
``KeptCaptures.offer`` at its ``request_end``: the request_start (without its
narrow ``prompt_parts``), kvsig and request_end events as the ADMIN sees them,
and the whole sections (thinking, reply, tool calls, the wide prompt capture)
within the store's own per-section budgets. A background writer puts it in
``prompt_lens_kept``, one column per section, and keeps the newest
``KEPT_PER_KEY`` rows and at most ``KEPT_CHARS_PER_KEY`` chars of that key, so
the capture outlives the ring, the section store's TTL and a restart.

Rules:

* Written only while the key still has the generation the request started
  under: the INSERT checks the key's row in the same statement, so a request
  that finishes after the switch went off -- or off and on again -- or after
  the key was deleted is never written. Turning it off deletes the rows of the
  key's whole rotation lineage (``TokenRepo.update``); deleting a key deletes
  its rows and those of the keys it was rotated from (``TokenRepo.delete``,
  and the FK cascade).
* Stored as the admin sees it; every key-holder read strips it with
  ``prompt_lens_scope.for_key_holder`` (``holder_events``), the same view the key's
  live stream gets.
* A section read reads its own column only, and a text section only from the
  asked offset (``substr``), so a ``?from=`` refetch never parses the row.
* Fail-open and bounded: ``offer`` never blocks or raises; each key has its own
  small queue, so one key's burst never drops another key's record; nothing
  here logs content -- a failure logs its exception type only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import aiosqlite

from app.db.database import open_db
from app.proxy.prompt_lens_scope import admin_view, for_key_holder
from app.proxy.prompt_lens_sections import SectionView, kept_view

logger = logging.getLogger(__name__)

#: Kept captures per key: rows past this are deleted on insert, oldest first.
KEPT_PER_KEY = 50
#: And at most this many chars per key over all its rows (G-R12), oldest first.
KEPT_CHARS_PER_KEY = 32_000_000
#: Records one key may have waiting for the writer; past it that key's record
#: is dropped and counted. Per key, so a burst never drops another key's.
QUEUE_PER_KEY = 8
#: Kept reads (the DB-backed fetches) one key holder may make a minute.
READS_PER_MINUTE = 60
#: JSON at least this long is parsed off the event loop.
_THREAD_CHARS = 256_000

#: Text section kind -> its (text, dropped) columns. A whitelist: the names
#: are put into SQL.
_TEXT_COLS = {"thinking": ("thinking", "thinking_dropped"), "reply": ("reply", "reply_dropped")}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


def section_sizes(record: dict[str, Any]) -> dict[str, Any]:
    """Each section's size, so a reader fetches only the ones it opens."""
    sections = record.get("sections") or {}
    out: dict[str, Any] = {}
    for k in ("thinking", "reply"):
        s = sections.get(k) or {}
        text = s.get("text")
        n = len(text) if isinstance(text, str) else 0
        dropped = _int(s.get("dropped_chars"))
        out[k] = {"chars": dropped + n, "dropped_chars": dropped}
    tc = sections.get("tool_calls") or {}
    calls = tc.get("calls")
    out["tool_calls"] = {"count": len(calls) if isinstance(calls, list) else 0}
    prompt = sections.get("prompt") if isinstance(sections.get("prompt"), dict) else None
    msgs = prompt.get("messages") if prompt is not None else None
    msgs = msgs if isinstance(msgs, list) else []
    out["tool_results"] = {
        "count": sum(
            1 for m in msgs if isinstance(m, dict) and m.get("role") in ("tool", "function")
        )
    }
    out["prompt"] = {"total_chars": prompt.get("total_chars") if prompt is not None else None}
    return out


def summary_of(record: dict[str, Any]) -> dict[str, Any]:
    """What a list of kept captures shows: nothing a key holder may not see."""
    start = record.get("start") or {}
    end = record.get("end") or {}
    return {
        "request_id": record.get("request_id"),
        "model": start.get("served_name") or start.get("model"),
        "stream": start.get("stream"),
        "session_id": start.get("session_id"),
        "turn": start.get("turn"),
        "started_ts": _num(start.get("ts")),
        "ended_ts": _num(end.get("ts")),
        "finish_reason": end.get("finish_reason"),
        "prompt_tokens": end.get("prompt_tokens"),
        "completion_tokens": end.get("completion_tokens"),
        "sections": section_sizes(record),
    }


def holder_events(events: list[Any]) -> list[dict[str, Any]]:
    """Kept events as their key's holder may see them (``for_key_holder``)."""
    return [for_key_holder(e) for e in events if isinstance(e, dict)]


def admin_events(events: list[Any]) -> list[dict[str, Any]]:
    """A kept capture's events as the admin stream sends them (``admin_view``):
    the same bytes for the same event on both admin routes."""
    return [admin_view(e) for e in events if isinstance(e, dict)]


def _row(record: dict[str, Any]) -> tuple[Any, ...]:
    """The record as prompt_lens_kept's columns (in a worker thread: big JSON)."""
    sections = record.get("sections") or {}
    events = [record.get("start"), record.get("kvsig"), record.get("end")]
    events_j = json.dumps([e for e in events if isinstance(e, dict)], default=str)
    texts = {}
    for k in ("thinking", "reply"):
        s = sections.get(k) or {}
        t = s.get("text")
        texts[k] = (t if isinstance(t, str) else "", _int(s.get("dropped_chars")))
    calls_j = json.dumps(sections.get("tool_calls") or {}, default=str)
    prompt_j = json.dumps(sections.get("prompt"), default=str)
    summary_j = json.dumps(summary_of(record), default=str)
    chars = len(events_j) + len(texts["thinking"][0]) + len(texts["reply"][0])
    chars += len(calls_j) + len(prompt_j)
    end = record.get("end") or {}
    return (
        record.get("req_id"),
        record.get("token_id"),
        record.get("request_id"),
        _num(end.get("ts")) or time.time(),
        _int(end.get("seq")),
        chars,
        summary_j,
        events_j,
        texts["thinking"][0],
        texts["thinking"][1],
        texts["reply"][0],
        texts["reply"][1],
        calls_j,
        prompt_j,
    )


async def insert(db: aiosqlite.Connection, record: dict[str, Any], row: tuple[Any, ...]) -> bool:
    """Write one kept capture if its key still has the generation it started
    under, then prune the key to ``KEPT_PER_KEY`` rows and
    ``KEPT_CHARS_PER_KEY`` chars (the newest row always stays). True when
    written. Does not commit."""
    tid = record.get("token_id")
    gen = record.get("keep_gen")
    if not isinstance(gen, int) or isinstance(gen, bool) or gen == 0:
        return False
    cur = await db.execute(
        "INSERT OR REPLACE INTO prompt_lens_kept(req_id, token_id, request_id, ended_ts, end_seq, "
        "chars, summary, events, thinking, thinking_dropped, reply, reply_dropped, tool_calls, "
        "prompt) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? WHERE EXISTS "
        "(SELECT 1 FROM api_tokens WHERE id = ? AND keep_captures = ?)",
        (*row, tid, gen),
    )
    if cur.rowcount <= 0:
        return False
    await db.execute(
        "DELETE FROM prompt_lens_kept WHERE req_id IN ("
        " SELECT req_id FROM ("
        "  SELECT req_id,"
        "   ROW_NUMBER() OVER w AS n,"
        "   SUM(chars) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run"
        "  FROM prompt_lens_kept WHERE token_id = ?"
        "  WINDOW w AS (ORDER BY ended_ts DESC, rowid DESC))"
        " WHERE n > ? OR (n > 1 AND run > ?))",
        (tid, KEPT_PER_KEY, KEPT_CHARS_PER_KEY),
    )
    return True


def _owner(token_id: str | None) -> tuple[str, tuple[Any, ...]]:
    return (" AND token_id = ?", (token_id,)) if token_id is not None else ("", ())


async def _loads(text: str) -> Any:
    try:
        if len(text) >= _THREAD_CHARS:
            return await asyncio.to_thread(json.loads, text)
        return json.loads(text)
    except ValueError:
        return None


async def get_events(
    db: aiosqlite.Connection, req_id: str, *, token_id: str | None
) -> tuple[str, list[Any], dict[str, Any]] | None:
    """(token id, events, summary) of a kept capture -- of key ``token_id``
    only, when given (a key holder's read); any key's for None (the admin's)."""
    where, args = _owner(token_id)
    cur = await db.execute(
        f"SELECT token_id, events, summary FROM prompt_lens_kept WHERE req_id = ?{where}",
        (req_id, *args),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    events, summary = await _loads(row[1]), await _loads(row[2])
    if not isinstance(events, list) or not events:
        return None
    return str(row[0]), events, summary if isinstance(summary, dict) else {}


async def section(
    db: aiosqlite.Connection, req_id: str, kind: str, *, start: int, token_id: str | None
) -> SectionView | None:
    """One section of a kept capture, reading that section's column only (a
    text section only from ``start``). Owner filter as ``get_events``."""
    where, args = _owner(token_id)
    if kind in _TEXT_COLS:
        col, dcol = _TEXT_COLS[kind]
        cur = await db.execute(
            f"SELECT token_id, end_seq, {dcol}, length({col}),"
            f" substr({col}, MAX(?, {dcol}) - {dcol} + 1)"
            f" FROM prompt_lens_kept WHERE req_id = ?{where}",
            (max(0, start), req_id, *args),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        tid, seq, dropped, length, text = row
        offset = min(max(start, dropped), dropped + length)
        return SectionView(
            token_id=str(tid),
            kind=kind,
            seq=int(seq),
            live=False,
            dropped_chars=int(dropped),
            offset=offset,
            text=text or "",
            kept=True,
        )
    json_col = {"tool_calls": "tool_calls", "tool_results": "prompt", "prompt": "prompt"}.get(kind)
    if json_col is None:
        return None
    col = json_col
    cur = await db.execute(
        f"SELECT token_id, end_seq, {col} FROM prompt_lens_kept WHERE req_id = ?{where}",
        (req_id, *args),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    value = await _loads(row[2])
    sections = {"tool_calls": value} if col == "tool_calls" else {"prompt": value}
    return kept_view(
        {"token_id": row[0], "end": {"seq": row[1]}, "sections": sections}, kind, start=start
    )


async def texts_by_request(
    db: aiosqlite.Connection, request_id: str, token_id: str
) -> tuple[list[Any], str, int, str, int] | None:
    """Key ``token_id``'s kept capture of the request ``request_id`` for the
    capture View: (events, reply, reply dropped, thinking, thinking dropped)."""
    cur = await db.execute(
        "SELECT events, reply, reply_dropped, thinking, thinking_dropped FROM prompt_lens_kept "
        "WHERE request_id = ? AND token_id = ? ORDER BY ended_ts DESC LIMIT 1",
        (request_id, token_id),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    events = await _loads(row[0])
    if not isinstance(events, list) or not events:
        return None
    return events, row[1] or "", int(row[2]), row[3] or "", int(row[4])


async def list_for(
    db: aiosqlite.Connection, token_ids: frozenset[str] | None, *, limit: int
) -> list[dict[str, Any]]:
    """Kept captures, newest first: of ``token_ids`` only, or every key's for None."""
    if token_ids is None:
        cur = await db.execute(
            "SELECT req_id, token_id, summary FROM prompt_lens_kept "
            "ORDER BY ended_ts DESC, rowid DESC LIMIT ?",
            (limit,),
        )
    else:
        ids = sorted(token_ids)
        marks = ",".join("?" * len(ids))
        cur = await db.execute(
            f"SELECT req_id, token_id, summary FROM prompt_lens_kept WHERE token_id IN ({marks}) "
            "ORDER BY ended_ts DESC, rowid DESC LIMIT ?",
            (*ids, limit),
        )
    out: list[dict[str, Any]] = []
    for req_id, tid, summary in await cur.fetchall():
        try:
            s = json.loads(summary)
        except ValueError:
            s = {}
        out.append({**(s if isinstance(s, dict) else {}), "req_id": req_id, "token_id": tid})
    return out


class ReadBook:
    """Per-key budget of kept reads (``READS_PER_MINUTE``): the DB-backed
    fetches a key holder makes. Bounded in keys."""

    def __init__(self, per_minute: int = READS_PER_MINUTE) -> None:
        self._per = per_minute
        self._book: dict[str, deque[float]] = {}

    def wait(self, tid: str) -> int:
        """0 when ``tid`` may read now (and counts the read), else seconds to wait."""
        now = time.monotonic()
        if len(self._book) > 1024:
            for k in [k for k, d in self._book.items() if not d or now - d[-1] > 60.0]:
                del self._book[k]
        d = self._book.setdefault(tid, deque(maxlen=self._per))
        if len(d) == self._per and now - d[0] < 60.0:
            return int(60.0 - (now - d[0])) + 1
        d.append(now)
        return 0


class KeptCaptures:
    """The per-key queues and the writer in front of ``prompt_lens_kept``.
    ``offer`` is what the section store calls (sync, never raises);
    ``run_forever`` is the writer the app starts; ``flush`` writes everything
    queued now (tests, shutdown)."""

    def __init__(self, db_path: Path, *, per_key: int = QUEUE_PER_KEY) -> None:
        self._db_path = db_path
        self._per_key = max(1, per_key)
        #: token id -> its waiting records, oldest first; keys in arrival order.
        self._queues: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()
        self._wake = asyncio.Event()
        self._busy = 0
        self._idle = asyncio.Event()
        self._idle.set()
        #: Records refused because their key's queue was full, and rows written.
        self.dropped = 0
        self.written = 0
        self._last_drop_log = float("-inf")

    def offer(self, record: dict[str, Any]) -> bool:
        try:
            tid = str(record.get("token_id"))
            q = self._queues.get(tid)
            if q is None:
                q = self._queues[tid] = deque()
            if len(q) >= self._per_key:
                self.dropped += 1
                now = time.monotonic()
                if now - self._last_drop_log > 60.0:
                    self._last_drop_log = now
                    logger.warning(
                        "prompt_lens: a key's kept captures queue is full, %d dropped so far",
                        self.dropped,
                    )
                return False
            q.append(record)
            self._wake.set()
            return True
        except Exception:  # noqa: BLE001 -- never into the hub
            return False

    def _next(self) -> dict[str, Any] | None:
        """The next record, one key at a time in turn (round robin)."""
        while self._queues:
            tid, q = next(iter(self._queues.items()))
            if not q:
                del self._queues[tid]
                continue
            record = q.popleft()
            self._queues.move_to_end(tid)
            if not q:
                del self._queues[tid]
            return record
        return None

    async def _write(self, record: dict[str, Any]) -> None:
        self._busy += 1
        self._idle.clear()
        try:
            row = await asyncio.to_thread(_row, record)
            async with open_db(self._db_path) as db:
                if await insert(db, record, row):
                    self.written += 1
                await db.commit()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- best effort; the type only, never content
            logger.warning("prompt_lens: kept capture write failed: %s", type(exc).__name__)
        finally:
            self._busy -= 1
            if self._busy == 0:
                self._idle.set()

    async def flush(self) -> None:
        """Write everything queued now, and wait for a write the writer task
        has in hand, so the rows are there when it returns."""
        while (record := self._next()) is not None:
            await self._write(record)
        await self._idle.wait()

    async def run_forever(self) -> None:
        while True:
            record = self._next()
            if record is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._write(record)
