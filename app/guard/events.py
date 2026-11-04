"""The guard's action log. One row per value acted on: rule, where, length and the placeholder's
keyed hash. Never the value, masked or not."""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any

ACTIONS = ("sanitized", "blocked", "restored", "error")


@dataclass(frozen=True)
class GuardEvent:
    ts: float
    token_id: str | None
    request_id: str | None
    sess_ref: str | None
    action: str
    rule_id: str | None
    role: str | None
    msg_index: int | None
    kind: str | None
    length: int | None
    ph: str | None
    relayed: bool


_COLS = [f.name for f in fields(GuardEvent)]


def record_events(db_path: str | Path, events: list[GuardEvent]) -> None:
    if not events:
        return
    with sqlite3.connect(db_path, timeout=10) as db:
        db.executemany(
            f"INSERT INTO guard_events({', '.join(_COLS)}) VALUES ({', '.join('?' for _ in _COLS)})",
            [astuple(e)[:-1] + (int(e.relayed),) for e in events],
        )


def query_events(
    db: sqlite3.Connection,
    *,
    token_id: str | None,
    since: float,
    action: str | None,
    rule_id: str | None,
    limit: int,
) -> dict[str, Any]:
    where = ["ts >= ?"]
    args: list[Any] = [since]
    if token_id:
        where.append("token_id = ?")
        args.append(token_id)
    if action:
        where.append("action = ?")
        args.append(action)
    if rule_id:
        where.append("rule_id = ?")
        args.append(rule_id)
    w = " AND ".join(where)
    rows = db.execute(
        f"SELECT {', '.join(_COLS)} FROM guard_events WHERE {w} ORDER BY ts DESC, id DESC LIMIT ?",
        (*args, limit),
    ).fetchall()
    tw = " AND ".join(where[: 2 if token_id else 1])
    targs = args[: 2 if token_id else 1]
    counts = dict(
        db.execute(
            f"SELECT action, COUNT(*) FROM guard_events WHERE {tw} GROUP BY action", targs
        ).fetchall()
    )
    distinct = db.execute(
        f"SELECT COUNT(DISTINCT ph) FROM guard_events WHERE {tw} AND ph IS NOT NULL", targs
    ).fetchone()[0]
    return {
        "events": [{**dict(zip(_COLS, r, strict=True)), "relayed": bool(r[-1])} for r in rows],
        "totals": {**{a: int(counts.get(a, 0)) for a in ACTIONS}, "distinct_values": int(distinct)},
    }


def prune_events(db: sqlite3.Connection, cutoff: float) -> int:
    return db.execute("DELETE FROM guard_events WHERE ts < ?", (cutoff,)).rowcount
