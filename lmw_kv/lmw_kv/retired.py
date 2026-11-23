"""The set of retired isolation scope tags, kept for the whole store.

A block whose scope tag is in this set is never served again. The set always lives in
memory, and memory is the authority: a tag is retired the moment ``add`` sees it. With a
path the set is also written to SQLite so it survives a restart. A RAM-only daemon keeps
it in memory and relies on the api's re-sync after a reconnect.

Failure rules (isolation fails closed):

- A failed write keeps the tag retired in memory, logs at ERROR and sets
  ``persist_degraded``. Every later write, and ``close``, retries the unwritten tags.
- A store file that cannot be opened, read or written at ``attach`` raises
  ``RetiredStoreError`` and sets ``isolation_refused``. The file's set is unknown, so
  the daemon must refuse isolated saves and loads until an ``attach`` succeeds. It never
  treats an unreadable file as an empty set.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)

RETIRED_NAME = "retired.db"
TAG_LEN = 16

_TABLE = "CREATE TABLE IF NOT EXISTS retired (tag BLOB PRIMARY KEY, at REAL NOT NULL)"
_ERRORS = (sqlite3.Error, OSError)


class RetiredStoreError(Exception):
    """The retired-scope file cannot be trusted: refuse isolated saves and loads."""


def _check(tag: bytes) -> bytes:
    if not isinstance(tag, bytes) or len(tag) != TAG_LEN:
        raise ValueError(f"a scope tag is {TAG_LEN} bytes")
    return tag


class RetiredScopes:
    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._tags: set[bytes] = set()
        self._pending: set[bytes] = set()  # retired in memory, not yet known to be on disk
        self._db: sqlite3.Connection | None = None
        self._refused = False
        if path is not None:
            self.attach(path)

    def __repr__(self) -> str:  # never show the tags
        return f"RetiredScopes(<{len(self)} tags>)"

    @property
    def persist_degraded(self) -> bool:
        """A file-backed set holds tags the disk lacks (a write or an ``attach`` failed)."""
        with self._lock:
            return self._refused or bool(self._pending)

    @property
    def isolation_refused(self) -> bool:
        """The last ``attach`` failed: isolated saves and loads must be refused."""
        with self._lock:
            return self._refused

    def attach(self, path: Path) -> None:
        """Persist to ``path``: merge its tags into memory, then write the ones it lacks.

        Raises ``RetiredStoreError`` if the file cannot be opened, read or written;
        memory keeps every tag it had plus any it read.
        """
        db: sqlite3.Connection | None = None
        try:
            if path.exists() and not os.access(path, os.R_OK | os.W_OK):
                raise PermissionError(f"retired scope store is not read-write: {path.name}")
            db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(_TABLE)
            on_disk = {bytes(r[0]) for r in db.execute("SELECT tag FROM retired").fetchall()}
        except _ERRORS as exc:
            if db is not None:
                db.close()
            with self._lock:
                self._refused = True
            log.error("retired scope store unreadable, isolation refused: %s", exc)
            raise RetiredStoreError("retired scope store unreadable") from exc
        with self._lock:
            self._tags |= on_disk  # merge first: nothing the file retired is ever live
            self._pending = self._tags - on_disk
            old, self._db = self._db, db
            ok = self._flush_locked()
            self._refused = not ok
        if old is not None:
            old.close()
        if not ok:
            raise RetiredStoreError("retired scope store not writable")

    def add(self, tags: Iterable[bytes]) -> int:
        """Retire ``tags``; returns how many were not retired before.

        Memory is updated first. A failed write is logged and retried later; it never
        un-retires a tag and never raises.
        """
        checked = [_check(t) for t in tags]
        with self._lock:
            new = [t for t in dict.fromkeys(checked) if t not in self._tags]
            self._tags.update(new)
            if self._db is not None or self._refused:  # file-backed: must reach the disk
                self._pending.update(new)
                self._flush_locked()
            return len(new)

    def _flush_locked(self) -> bool:
        """Write the pending tags; True when none is left. Caller holds the lock."""
        if not self._pending:
            return True
        if self._db is None:
            return False
        try:
            now = time.time()
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.executemany(
                    "INSERT OR IGNORE INTO retired (tag, at) VALUES (?, ?)",
                    [(t, now) for t in self._pending],
                )
        except _ERRORS as exc:
            log.error(
                "retired scope store write failed, %d tag(s) held in memory only: %s",
                len(self._pending),
                exc,
            )
            return False
        self._pending.clear()
        return True

    def __contains__(self, tag: bytes) -> bool:
        with self._lock:
            return tag in self._tags

    def __len__(self) -> int:
        with self._lock:
            return len(self._tags)

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            db, self._db = self._db, None
        if db is not None:
            db.close()
