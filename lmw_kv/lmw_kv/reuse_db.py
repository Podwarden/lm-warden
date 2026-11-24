"""``<nvme_dir>/reuse.db``: the reuse counters of NVMe positions (counters design 3.7).

One table ``reuse`` keyed by ``(model, pos)``, beside ``retired.db``. Only the
per-position counters are kept; the tallies since start are not. No isolation property
depends on this file, so every failure is open: the store keeps counting in memory and
reports ``reuse_persist_degraded``.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lmw_kv.reuse import PosCounters, _Pos

_TABLE = """CREATE TABLE IF NOT EXISTS reuse (
    model BLOB NOT NULL, pos BLOB NOT NULL, first_saved REAL NOT NULL, saver BLOB NOT NULL,
    depth INTEGER NOT NULL, loads INTEGER NOT NULL, reloads INTEGER NOT NULL,
    saver_reloads INTEGER NOT NULL, ram_reloads INTEGER NOT NULL,
    nvme_reloads INTEGER NOT NULL, last_reload REAL NOT NULL, d REAL NOT NULL,
    t REAL NOT NULL, tenants_seen BLOB NOT NULL, many INTEGER NOT NULL,
    nvme_written INTEGER NOT NULL, PRIMARY KEY (model, pos))"""
_COLS = (
    "model, pos, first_saved, saver, depth, loads, reloads, saver_reloads, ram_reloads, "
    "nvme_reloads, last_reload, d, t, tenants_seen, many, nvme_written"
)
_UPSERT = f"INSERT OR REPLACE INTO reuse ({_COLS}) VALUES ({', '.join('?' * 16)})"
_ERRORS = (sqlite3.Error, OSError, TypeError, ValueError, OverflowError)


class ReuseDbError(Exception):
    """``reuse.db`` cannot be opened, read or written."""


def _pack(tenants: Sequence[bytes]) -> bytes:
    """Tenants as length-prefixed byte strings."""
    return b"".join(bytes([len(t)]) + t for t in tenants)


def _unpack(blob: bytes) -> tuple[bytes, ...]:
    out: list[bytes] = []
    i = 0
    while i < len(blob):
        n = blob[i]
        out.append(blob[i + 1 : i + 1 + n])
        i += 1 + n
    return tuple(out)


def _row(pos: _Pos, c: PosCounters) -> tuple[Any, ...]:
    return (
        bytes(pos[0]), bytes(pos[1]), float(c.first_saved), bytes(c.saver), int(c.depth),
        int(c.loads), int(c.reloads), int(c.saver_reloads), int(c.ram_reloads),
        int(c.nvme_reloads), float(c.last_reload), float(c.d), float(c.t),
        _pack(c.tenants_seen), int(bool(c.many)), int(c.nvme_written),
    )  # fmt: skip


class ReuseDb:
    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        try:
            db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute(_TABLE)
            except BaseException:
                db.close()
                raise
        except _ERRORS as exc:
            raise ReuseDbError(f"cannot open {path.name}: {exc}") from exc
        self._db = db

    def rows(self) -> list[tuple[_Pos, PosCounters]]:
        with self._lock:
            if self._db is None:
                raise ReuseDbError("closed")
            try:
                got = self._db.execute(f"SELECT {_COLS} FROM reuse").fetchall()
            except _ERRORS as exc:
                raise ReuseDbError(f"cannot read the reuse counters: {exc}") from exc
        out: list[tuple[_Pos, PosCounters]] = []
        for r in got:
            out.append((
                (bytes(r[0]), bytes(r[1])),
                PosCounters(
                    first_saved=r[2], saver=bytes(r[3]), depth=r[4], loads=r[5],
                    reloads=r[6], saver_reloads=r[7], ram_reloads=r[8], nvme_reloads=r[9],
                    last_reload=r[10], d=r[11], t=r[12], tenants_seen=_unpack(bytes(r[13])),
                    many=bool(r[14]), nvme_written=r[15],
                ),
            ))  # fmt: skip
        return out

    def write(self, upserts: Sequence[tuple[_Pos, PosCounters]], deletes: Sequence[_Pos]) -> None:
        """Upsert and delete in one transaction: all of it lands, or none of it."""
        try:
            rows = [_row(p, c) for p, c in upserts]
            dels = [(bytes(m), bytes(p)) for m, p in deletes]
        except _ERRORS as exc:
            raise ReuseDbError(f"unwritable counters: {exc}") from exc
        with self._lock:
            if self._db is None:
                raise ReuseDbError("closed")
            try:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    self._db.executemany("DELETE FROM reuse WHERE model = ? AND pos = ?", dels)
                    self._db.executemany(_UPSERT, rows)
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise
                self._db.execute("COMMIT")
            except _ERRORS as exc:
                raise ReuseDbError(f"cannot write the reuse counters: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            db, self._db = self._db, None
        if db is not None:
            db.close()
