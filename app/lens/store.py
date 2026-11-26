"""Prompt Lens score store: the score and findings of a quiet session
(analysis spec 10, 12; migration 0064).

One ``lens_session_scores`` row per ``(key_id, sess_ref)``, rewritten in place
when the session resumes and goes quiet again (never two rows, Review Focus 5).
Everything stored is a number, a rule id, a cause code, a keyed tool hash or a
turn number (A-10). The per-rule ``terms`` let a rule marked "won't fix" (stage
3) be left out on read. Synchronous ``sqlite3``: callers run in a worker thread.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.lens.catalogue import Finding, Saving
from app.lens.scoring import LAYERS, Score

_COLS = (
    "key_id, sess_ref, started, last_at, model, route, harness, version, turns, e, c, r, "
    "overall, ranges, confidence, coverage, terms, rules, score_version"
)


def _loads(v: Any, default: Any) -> Any:
    try:
        d = json.loads(v) if isinstance(v, str) else None
    except ValueError:
        return default
    return d if isinstance(d, type(default)) else default


@dataclass(frozen=True)
class SessionScoreRow:
    """The table row. ``ranges``/``confidence``/``coverage``/``terms`` are the
    JSON the columns hold, decoded; ``rules`` the fired rule ids by layer."""

    key_id: str
    sess_ref: str
    started: float
    last_at: float
    model: str
    route: str
    harness: str
    version: str | None
    turns: int
    e: int | None
    c: int | None
    r: int | None
    overall: int | None
    ranges: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    terms: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    score_version: int = 1

    @staticmethod
    def of(
        *,
        key_id: str,
        sess_ref: str,
        started: float,
        last_at: float,
        model: str,
        route: str,
        harness: str,
        turns: int,
        scores: dict[str, Score],
        terms: Mapping[str, Any],
        coverage: Mapping[str, float],
        fired: Mapping[str, list[str]],
        score_version: int,
        version: str | None = None,
    ) -> SessionScoreRow:
        """Build a row from the scores of ``scoring.score_session``; ``terms``
        is ``{layer: Terms.to_json()}``."""
        per: dict[str, Any] = {}
        for k, s in scores.items():
            t = dict(terms.get(k) or {})
            t.update({"scored": s.scored, "reason": s.reason, "turns": s.turns, "inputs": s.inputs})
            t["excluded"] = list(s.excluded_rules)
            per[k] = t
        return SessionScoreRow(
            key_id=key_id,
            sess_ref=sess_ref,
            started=started,
            last_at=last_at,
            model=model,
            route=route,
            harness=harness,
            version=version,
            turns=turns,
            e=scores["E"].value,
            c=scores["C"].value,
            r=scores["R"].value,
            overall=scores["overall"].value,
            ranges={k: [s.lo, s.hi] for k, s in scores.items() if s.scored},
            confidence={k: s.confidence for k, s in scores.items() if s.scored},
            coverage={k: round(v, 3) for k, v in coverage.items()},
            terms=per,
            rules=dict(fired),
            score_version=score_version,
        )

    def scores(self) -> dict[str, Score]:
        """``Score`` per layer and ``overall`` as stored (value, range, confidence,
        or the reason it was not scored)."""
        out: dict[str, Score] = {}
        for k in (*LAYERS, "overall"):
            t = self.terms.get(k) or {}
            value = {"E": self.e, "C": self.c, "R": self.r, "overall": self.overall}[k]
            rng = self.ranges.get(k) or [None, None]
            if value is None:
                out[k] = Score(
                    k,  # type: ignore[arg-type]
                    False,
                    None,
                    None,
                    None,
                    None,
                    int(t.get("turns") or 0),
                    t.get("reason") or "not scored",
                )
                continue
            out[k] = Score(
                k,  # type: ignore[arg-type]
                True,
                value,
                rng[0],
                rng[1],
                self.confidence.get(k),
                int(t.get("turns") or 0),
                None,
                dict(t.get("inputs") or {}),
                tuple(t.get("excluded") or ()),
            )
        return out

    def to_values(self) -> tuple[Any, ...]:
        dump = json.dumps
        return (
            self.key_id,
            self.sess_ref,
            self.started,
            self.last_at,
            self.model,
            self.route,
            self.harness,
            self.version,
            self.turns,
            self.e,
            self.c,
            self.r,
            self.overall,
            dump(self.ranges),
            dump(self.confidence),
            dump(self.coverage),
            dump(self.terms),
            dump(self.rules),
            self.score_version,
        )


def _row(t: Sequence[Any]) -> SessionScoreRow:
    return SessionScoreRow(
        key_id=t[0],
        sess_ref=t[1],
        started=float(t[2]),
        last_at=float(t[3]),
        model=t[4],
        route=t[5],
        harness=t[6],
        version=t[7],
        turns=int(t[8]),
        e=t[9],
        c=t[10],
        r=t[11],
        overall=t[12],
        ranges=_loads(t[13], {}),
        confidence=_loads(t[14], {}),
        coverage=_loads(t[15], {}),
        terms=_loads(t[16], {}),
        rules=_loads(t[17], {}),
        score_version=int(t[18]),
    )


def finding_of(
    rule: str, turns: str, counts: str, tool: str, cause: Any, saving: str, conf: str
) -> Finding:
    """A stored finding back as a ``Finding``."""
    s = _loads(saving, {})
    return Finding(
        rule=rule,
        turns=tuple(int(n) for n in _loads(turns, [])),
        counts={str(k): int(v) for k, v in _loads(counts, {}).items()},
        tool=tool or None,
        cause=cause,
        saving=Saving(
            int(s.get("tokens") or 0), s.get("prefill_s"), s.get("decode_s"), s.get("money")
        ),
        confidence=conf,  # type: ignore[arg-type]
    )


class LensStore:
    """Reads and writes ``lens_session_scores`` and ``lens_findings`` on a
    caller-owned ``sqlite3`` connection (the caller commits through
    ``upsert_session``)."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def upsert_session(self, row: SessionScoreRow, findings: Sequence[Finding]) -> None:
        """Replace the session's row and findings in one transaction."""
        with self.db:
            self.db.execute(
                f"INSERT OR REPLACE INTO lens_session_scores ({_COLS}) VALUES ({', '.join('?' * 19)})",
                row.to_values(),
            )
            self.db.execute(
                "DELETE FROM lens_findings WHERE key_id = ? AND sess_ref = ?",
                (row.key_id, row.sess_ref),
            )
            for f in findings:
                d = f.to_json()
                self.db.execute(
                    "INSERT OR REPLACE INTO lens_findings"
                    " (key_id, sess_ref, rule, turns, counts, tool, cause, saving, confidence)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row.key_id,
                        row.sess_ref,
                        f.rule,
                        json.dumps(d["turns"]),
                        json.dumps(d["counts"]),
                        f.tool or "",
                        f.cause,
                        json.dumps(d["saving"]),
                        f.confidence,
                    ),
                )

    def session(self, key_id: str, sess_ref: str) -> SessionScoreRow | None:
        t = self.db.execute(
            f"SELECT {_COLS} FROM lens_session_scores WHERE key_id = ? AND sess_ref = ?",
            (key_id, sess_ref),
        ).fetchone()
        return _row(t) if t is not None else None

    def last_at(self, key_id: str, sess_ref: str) -> float | None:
        t = self.db.execute(
            "SELECT last_at FROM lens_session_scores WHERE key_id = ? AND sess_ref = ?",
            (key_id, sess_ref),
        ).fetchone()
        return float(t[0]) if t is not None else None

    def findings(self, key_id: str, sess_ref: str) -> list[Finding]:
        got = self.db.execute(
            "SELECT rule, turns, counts, tool, cause, saving, confidence FROM lens_findings"
            " WHERE key_id = ? AND sess_ref = ? ORDER BY rule, tool",
            (key_id, sess_ref),
        ).fetchall()
        return [finding_of(*t) for t in got]

    def sessions(
        self,
        key_id: str | Sequence[str] | None,
        *,
        since: float,
        model: str | Sequence[str] | None = None,
        harness: Sequence[str] | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> list[SessionScoreRow]:
        """Newest first; ``key_id`` None is every key (admin), a list is those
        keys. ``model`` and ``harness`` are one value or a list (any of them).
        ``cursor`` is the previous page's last ``"<last_at>|<sess_ref>"``
        (opaque to callers)."""
        sql = f"SELECT {_COLS} FROM lens_session_scores WHERE last_at >= ?"
        args: list[Any] = [since]
        for column, want in (("key_id", key_id), ("model", model), ("harness", harness)):
            if want is None or want == "":
                continue
            vals = [want] if isinstance(want, str) else list(want)
            if not vals:
                # An empty list matches nothing, never everything.
                sql += " AND 0"
                continue
            sql += f" AND {column} IN ({','.join('?' * len(vals))})"
            args += vals
        c = decode_cursor(cursor)
        if c is not None:
            sql += " AND (last_at < ? OR (last_at = ? AND sess_ref > ?))"
            args += [c[0], c[0], c[1]]
        sql += " ORDER BY last_at DESC, sess_ref ASC LIMIT ?"
        args.append(int(limit))
        return [_row(t) for t in self.db.execute(sql, args).fetchall()]

    def scores_for(
        self,
        key_ids: Sequence[str] | None,
        *,
        since: float,
        model: str | None = None,
        limit: int = 5_000,
    ) -> list[SessionScoreRow]:
        """The stored rows of the keys (None: every key), newest first, for trends:
        scored rows are the ones with a value; the caller filters. Bounded by ``limit``."""
        sql = f"SELECT {_COLS} FROM lens_session_scores WHERE last_at >= ?"
        args: list[Any] = [since]
        if key_ids is not None:
            sql += f" AND key_id IN ({','.join('?' * len(key_ids))})"
            args += list(key_ids)
        if model:
            sql += " AND model = ?"
            args.append(model)
        sql += " ORDER BY last_at DESC, sess_ref ASC LIMIT ?"
        args.append(int(limit))
        return [_row(t) for t in self.db.execute(sql, args).fetchall()]


def encode_cursor(row: SessionScoreRow) -> str:
    return f"{row.last_at!r}|{row.sess_ref}"


def decode_cursor(cursor: str | None) -> tuple[float, str] | None:
    if not cursor or "|" not in cursor:
        return None
    a, _, b = cursor.partition("|")
    try:
        return float(a), b
    except ValueError:
        return None
