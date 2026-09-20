"""admin_audit: one row per request an admin token made (migration 0037).

Written by app/auth/admin_audit.py shortly after each response -- the
middleware enqueues and that module's writer inserts a whole batch per
transaction (``record_many``) -- read one token at a time by GET
/api/admin-tokens/{id}/audit, pruned by app/runtime/stats_pruner.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import aiosqlite

#: Kept 90 days, and at most 200k rows total (spec 2026-09-19, decision 7).
#: PER_TOKEN_MAX_ROWS (issue #257) caps any single token_id's own history
#: first, so one leaked/noisy token can only evict its own rows -- it can no
#: longer flood the shared table and push out every other token's trail. The
#: pruner reads all three at call time, so a test can lower them.
RETENTION_DAYS = 90
MAX_ROWS = 200_000
PER_TOKEN_MAX_ROWS = 20_000

_COLS = "id, ts, token_id, username, method, path, status, duration_ms, client_ip, peer_ip"
#: The write columns (``id`` autoincrements) and the one INSERT both writers
#: use. ``AuditParams`` below is exactly this tuple, in this order.
_WRITE_COLS = "ts, token_id, username, method, path, status, duration_ms, client_ip, peer_ip"
INSERT_SQL = (
    f"INSERT INTO admin_audit({_WRITE_COLS}) VALUES ({', '.join('?' * 9)})"
)

#: One row's INSERT parameters: ``(ts, token_id, username, method, path,
#: status, duration_ms, client_ip, peer_ip)``. The queue in
#: app/auth/admin_audit.py holds these, so nothing richer than a tuple of
#: scalars ever sits in memory waiting for a flush.
AuditParams = tuple[float, str, str, str, str, int, int, str | None, str | None]

#: The per-token cap's DELETE (issue #257): a single covering-index scan of
#: idx_admin_audit_token_ts, windowed per token_id, rather than one query per
#: token_id. Exposed so a test can EXPLAIN QUERY PLAN it directly -- see
#: tests/unit/db/test_admin_audit_repo.py.
PER_TOKEN_CAP_SQL = (
    "DELETE FROM admin_audit WHERE id IN ("
    "  SELECT id FROM ("
    "    SELECT id, ROW_NUMBER() OVER ("
    "      PARTITION BY token_id ORDER BY ts DESC"
    "    ) AS rn FROM admin_audit"
    "  ) WHERE rn > :cap"
    ")"
)


@dataclass(frozen=True)
class AuditRow:
    id: int
    ts: float  # epoch seconds, when the request arrived
    token_id: str
    username: str
    method: str
    path: str  # the route template, e.g. /api/models/{model_id}/load
    status: int
    duration_ms: int
    client_ip: str | None  # proxy-reported (X-Forwarded-For aware), forgeable
    peer_ip: str | None  # the raw socket peer, next to client_ip


def _decode(r: Any) -> AuditRow:
    return AuditRow(
        id=int(r[0]), ts=float(r[1]), token_id=str(r[2]), username=str(r[3]),
        method=str(r[4]), path=str(r[5]), status=int(r[6]), duration_ms=int(r[7]),
        client_ip=None if r[8] is None else str(r[8]),
        peer_ip=None if r[9] is None else str(r[9]),
    )


class AdminAuditRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def record(
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
        peer_ip: str | None = None,
    ) -> None:
        await self.record_many([(
            ts, token_id, username, method, path, status, duration_ms,
            client_ip, peer_ip,
        )])

    async def record_many(self, rows: Sequence[AuditParams]) -> None:
        """One transaction for the whole batch (#258).

        ``executemany`` + one commit: the audit is the only writer on this
        connection and it is opened, used and closed by one ``flush``, so the
        transaction never spans anything but these inserts. Keeping it that
        short is deliberate -- a long write transaction here would contend with
        every other short-lived connection in the process (the WAL
        sync/async lock hazard documented in tests/conftest.py).
        """
        if not rows:
            return
        await self.db.executemany(INSERT_SQL, rows)
        await self.db.commit()

    async def page(
        self, token_id: str, *, limit: int, before: float | None
    ) -> tuple[list[AuditRow], float | None]:
        """One token's rows older than ``before`` (all when None), newest
        first, and the cursor for the next page (None when there is none).

        A page is ``limit`` rows plus every further row sharing the last row's
        ``ts``: the cursor is a timestamp and the next page asks for rows
        strictly older than it, so splitting a timestamp across two pages
        would lose the rest of it.

        ``next_before`` is set ONLY when a further row genuinely exists beyond
        what this page returns. Extending the page to swallow every row tied
        at the boundary timestamp can exhaust the table -- e.g. the last three
        rows all share one ts and the page, after extension, holds all three
        -- in which case a caller must NOT be told there is more to load.
        """
        where = "token_id = ?"
        params: list[object] = [token_id]
        if before is not None:
            where += " AND ts < ?"
            params.append(before)
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM admin_audit WHERE {where} "
            "ORDER BY ts DESC, id DESC LIMIT ?",
            (*params, limit + 1),
        )
        rows = [_decode(r) for r in await cur.fetchall()]
        if len(rows) <= limit:
            return rows, None
        page = rows[:limit]
        boundary = page[-1].ts
        if rows[limit].ts == boundary:
            cur = await self.db.execute(
                f"SELECT {_COLS} FROM admin_audit "
                "WHERE token_id = ? AND ts = ? AND id < ? ORDER BY id DESC",
                (token_id, boundary, page[-1].id),
            )
            page.extend(_decode(r) for r in await cur.fetchall())
        cur = await self.db.execute(
            "SELECT 1 FROM admin_audit WHERE token_id = ? AND ts < ? LIMIT 1",
            (token_id, boundary),
        )
        has_more = await cur.fetchone() is not None
        return page, (boundary if has_more else None)


async def prune(
    db: aiosqlite.Connection,
    *,
    cutoff: float,
    max_rows: int,
    per_token_max_rows: int = PER_TOKEN_MAX_ROWS,
) -> int:
    """Delete rows older than ``cutoff`` (epoch s), then any single token's
    rows beyond its newest ``per_token_max_rows``, then -- as a backstop --
    any row beyond the newest ``max_rows`` overall. Returns the number
    deleted. Does not commit -- the caller's transaction owns that, as with
    request_history.prune.

    The per-token pass runs before the global cap (issue #257): without it, a
    single token flooding the table evicts every other token's history once
    the global cap is hit, well before the 90-day window would. It walks
    idx_admin_audit_token_ts as a single covering-index scan -- one SQL round,
    not a query per token_id -- via a ROW_NUMBER() window partitioned by
    token_id and ordered by ts DESC.
    """
    cur = await db.execute("DELETE FROM admin_audit WHERE ts < ?", (cutoff,))
    removed = cur.rowcount or 0
    if per_token_max_rows > 0:
        cur = await db.execute(PER_TOKEN_CAP_SQL, {"cap": int(per_token_max_rows)})
        removed += cur.rowcount or 0
    if max_rows > 0:
        cur = await db.execute(
            "DELETE FROM admin_audit WHERE id IN ("
            "  SELECT id FROM admin_audit ORDER BY ts DESC, id DESC LIMIT -1 OFFSET ?"
            ")",
            (int(max_rows),),
        )
        removed += cur.rowcount or 0
    return removed
