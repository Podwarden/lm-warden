"""Sign-in sessions (verified finding #2). A session id is minted at login (admin) or key sign-in
(key holder) and carried in every token of that sign-in.

Checks are answered from memory only: the active sessions are loaded at startup and this process
is their only writer, so a check never touches the database (a blocking read on the event loop
would deadlock against aiosqlite's own writer). Writes go to the table off the loop, then to
memory, so an ended session is refused from the very next request."""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
import threading
import time
from pathlib import Path

#: Sessions older than this are forgotten (longer than any refresh token lives).
MAX_AGE_S = 31 * 86400


class AuthSessions:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        #: sid -> (kind, subject, created_at) for every active session.
        self._active: dict[str, tuple[str, str, float]] = {}

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def load(self) -> None:
        """Startup (before serving): the active sessions into memory; old rows dropped."""
        cutoff = time.time() - MAX_AGE_S
        with self._db() as db:
            db.execute("DELETE FROM auth_sessions WHERE created_at < ?", (cutoff,))
            rows = db.execute(
                "SELECT sid, kind, subject, created_at FROM auth_sessions WHERE revoked_at IS NULL"
            ).fetchall()
        with self._lock:
            self._active = {sid: (k, s, c) for sid, k, s, c in rows}

    def is_active(self, sid: str | None, *, kind: str, subject: str) -> bool:
        if not sid:
            return False
        with self._lock:
            hit = self._active.get(sid)
        return hit is not None and hit[0] == kind and hit[1] == subject

    async def create(self, kind: str, subject: str) -> str:
        sid = secrets.token_urlsafe(18)
        now = time.time()

        def write() -> None:
            with self._db() as db:
                db.execute(
                    "INSERT INTO auth_sessions(sid, kind, subject, created_at) VALUES (?, ?, ?, ?)",
                    (sid, kind, subject, now),
                )

        await asyncio.to_thread(write)
        with self._lock:
            self._active[sid] = (kind, subject, now)
        return sid

    async def _end(self, sids: list[str]) -> int:
        if not sids:
            return 0
        with self._lock:
            for s in sids:
                self._active.pop(s, None)
        now = time.time()

        def write() -> None:
            with self._db() as db:
                db.executemany(
                    "UPDATE auth_sessions SET revoked_at = ? WHERE sid = ? AND revoked_at IS NULL",
                    [(now, s) for s in sids],
                )

        await asyncio.to_thread(write)
        return len(sids)

    async def end(self, sid: str | None) -> None:
        if sid:
            await self._end([sid])

    def _of(self, kind: str, subject: str) -> list[str]:
        with self._lock:
            return [s for s, (k, sub, _c) in self._active.items() if k == kind and sub == subject]

    async def end_subject(self, kind: str, subject: str) -> int:
        return await self._end(self._of(kind, subject))

    async def end_subject_except(self, kind: str, subject: str, keep: str | None) -> int:
        return await self._end([s for s in self._of(kind, subject) if s != keep])
