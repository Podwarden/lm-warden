"""The quiet-session scorer (analysis Task 2.3, spec 10).

Every ``lens_scorer_interval_s`` it looks for sessions whose last request is at
least ``lens_quiet_s`` old and whose ``last_at`` moved since the stored row, scores
them on the key's own lens (``service.score_rows``) and writes one
``lens_session_scores`` row and its ``lens_findings`` per session. A session that
resumes and goes quiet again is rewritten in place (Review Focus 5). It reads
stored rows only, so it works with the capture off and after the ring forgets;
a failing session is logged (type only, never content) and skipped.

Window: requests that finished since the last pass's quiet horizon, so each
request is looked at once; a pass that hit ``MAX_PER_RUN`` keeps its horizon
and the rest wait for the next. The first pass reaches back ``CATCHUP_S``.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from contextlib import closing
from typing import Any

from app.lens import service
from app.lens.store import LensStore

logger = logging.getLogger(__name__)

#: How far back the first pass after a start reaches.
CATCHUP_S = 7 * 24 * 3600.0
#: Overlap of consecutive windows (rows are written a moment after they finish).
SLACK_S = 120.0
#: Sessions written per pass at most.
MAX_PER_RUN = 25
#: Candidate (key, session) pairs read per pass at most.
MAX_CANDIDATES = 5_000


def _connect(db_path: Any) -> sqlite3.Connection:
    db = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    db.execute("PRAGMA busy_timeout = 30000")
    db.execute("PRAGMA foreign_keys = ON")
    return db


class ScorerJob:
    def __init__(self, state: Any) -> None:
        self.state = state
        self._horizon: float | None = None

    def _settings(self) -> Any:
        return self.state.settings

    def _candidates(self, db: sqlite3.Connection, lo: float, hi: float) -> list[tuple[str, str]]:
        got = db.execute(
            "SELECT token_id, cobs2_session, MAX(finished_at) AS m FROM request_history"
            " WHERE token_id IS NOT NULL AND cobs2_session IS NOT NULL"
            " AND finished_at > ? AND finished_at <= ?"
            " GROUP BY token_id, cobs2_session ORDER BY m DESC LIMIT ?",
            (lo, hi, MAX_CANDIDATES),
        ).fetchall()
        return [(str(t), str(s)) for t, s, _m in got]

    def run_once_sync(self, now: float) -> int:
        """One pass (a worker thread). Returns the sessions written."""
        cfg = self._settings()
        quiet = float(cfg.lens_quiet_s)
        hi = now - quiet
        lo = (self._horizon if self._horizon is not None else now - CATCHUP_S) - SLACK_S
        written = 0
        complete = True
        with closing(_connect(cfg.db_path)) as db:
            lens = LensStore(db)
            seen: set[tuple[str, str]] = set()
            for token, sess in self._candidates(db, lo, hi):
                for sid, rows in service.session_units(db, cfg.cookie_secret, token, sess):
                    if (token, sid) in seen:
                        continue
                    seen.add((token, sid))
                    last_at = max(r.finished_at for r in rows)
                    if last_at > hi:
                        continue  # still active (a subagent or a later turn)
                    stored = lens.last_at(token, sid)
                    if stored is not None and stored >= last_at:
                        continue
                    if written >= MAX_PER_RUN:
                        complete = False
                        continue
                    try:
                        ss = service.score_rows(db, self.state, rows, sid=sid, strip_admin=False)
                        if ss is not None:
                            lens.upsert_session(ss.to_row(), ss.findings)
                            written += 1
                    except Exception as exc:  # noqa: BLE001 -- one session must not stop the rest
                        logger.warning("lens: scoring a session failed: %s", type(exc).__name__)
        if complete:
            self._horizon = hi
        return written

    async def run_once(self, now: float) -> int:
        return await asyncio.to_thread(self.run_once_sync, now)

    async def run_forever(self) -> None:
        while True:
            try:
                await self.run_once(time.time())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- the loop outlives a bad pass
                logger.warning("lens: scorer pass failed: %s", type(exc).__name__)
            await asyncio.sleep(max(5.0, float(self._settings().lens_scorer_interval_s)))
