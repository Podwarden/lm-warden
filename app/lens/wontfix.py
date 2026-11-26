"""Won't fix (analysis Task 3.3, spec 11.4, decision D-3).

A key holder, or the admin for any key, marks a rule "won't fix" for a key,
either for every prompt version or for one. A marked rule stops counting in the
scores (its term is left out of E, C or R), stays visible and muted in every
list, never enters the top 3, and the mark is reversible: scores are computed
on read, so an unmark restores them at once. Every mark and unmark writes one
audit row. R-09 (a secret in the prompt) cannot be marked: it is a notice, not
a score term.

Marks follow the key's rotation chain: a mark made on a key counts for its
successors and predecessors (Review Focus 4). The table stores ``''`` for "every
version" (SQLite treats NULLs in a primary key as distinct); the API says null.

Two readers: ``WontfixRepo.active`` (the live scorecard re-scores with the rules
left out) and ``apply_marks`` (a stored session row, for trends: the value is
recomputed from the stored per-rule terms, no rescan). For R, the stored terms
cannot tell a turn bad under two rules from two bad turns, so with a mark on a
rule that fired the figure is a close upper bound on the bad turns (never below
the stored count); every other case is exact.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from fastapi import Request

from app.auth.admin_audit import route_template
from app.lens import scoring
from app.lens.catalogue import RULES
from app.lens.store import SessionScoreRow
from app.lens.versions import lineage_ids
from app.utils.client_ip import client_ip

#: A secret in the prompt is a notice, not a score term: never marked.
UNMARKABLE = frozenset({"R-09"})
By = Literal["key", "admin"]
#: A mark's version, as stored for "every version".
ALL_VERSIONS = ""
#: The audit row's user for a change the key holder made.
KEY_HOLDER = "key-holder"
MARK_ACTION = "lens.wontfix.mark"
UNMARK_ACTION = "lens.wontfix.unmark"


class WontfixError(ValueError):
    """A mark the rules refuse (a route answers 422)."""


@dataclass(frozen=True)
class Mark:
    key_id: str
    rule: str
    #: None: every prompt version.
    version: str | None
    by: str
    at: float


def check_rule(rule: str) -> None:
    if rule not in RULES:
        raise WontfixError("unknown rule")
    if rule in UNMARKABLE:
        raise WontfixError("this rule is a notice and cannot be marked won't fix")


def _stored(version: str | None) -> str:
    return ALL_VERSIONS if version is None else version


class WontfixRepo:
    """``lens_wontfix`` on a caller-owned ``sqlite3`` connection (a worker
    thread's); each write commits."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def mark(self, key_id: str, rule: str, version: str | None, by: str) -> bool:
        """Mark ``rule`` for the key (idempotent: a second mark keeps the first).
        True when a mark was added."""
        check_rule(rule)
        if by not in ("key", "admin"):
            raise WontfixError("unknown actor")
        ids = lineage_ids(self.db, key_id)
        marks = ",".join("?" * len(ids))
        with self.db:
            if self.db.execute(
                f"SELECT 1 FROM lens_wontfix WHERE rule = ? AND version = ? AND key_id IN ({marks})",
                (rule, _stored(version), *ids),
            ).fetchone():
                return False  # already marked on this key or one of its chain
            self.db.execute(
                "INSERT INTO lens_wontfix(key_id, rule, version, by, at) VALUES (?, ?, ?, ?, ?)",
                (key_id, rule, _stored(version), by, time.time()),
            )
        return True

    def unmark(self, key_id: str, rule: str, version: str | None) -> bool:
        """Remove the mark from the key or any key of its chain. True when one existed."""
        ids = lineage_ids(self.db, key_id)
        marks = ",".join("?" * len(ids))
        with self.db:
            cur = self.db.execute(
                f"DELETE FROM lens_wontfix WHERE rule = ? AND version = ? AND key_id IN ({marks})",
                (rule, _stored(version), *ids),
            )
        return cur.rowcount > 0

    def marks(self, key_id: str) -> list[Mark]:
        """Every mark of the key's chain, oldest first."""
        ids = lineage_ids(self.db, key_id)
        q = ",".join("?" * len(ids))
        got = self.db.execute(
            f"SELECT key_id, rule, version, by, at FROM lens_wontfix WHERE key_id IN ({q})"
            " ORDER BY at ASC, rule ASC, version ASC",
            ids,
        ).fetchall()
        return [Mark(k, r, v or None, b, float(a)) for k, r, v, b, a in got]

    def active_marks(self, key_id: str, version: str | None) -> dict[str, str]:
        """Rule -> who marked it (``key`` or ``admin``), for the marks that apply to a
        session of prompt version ``version``: those for every version and those for
        exactly this one."""
        out: dict[str, str] = {}
        for m in self.marks(key_id):
            if m.version is None or m.version == version:
                out.setdefault(m.rule, m.by)
        return out

    def active(self, key_id: str, version: str | None) -> frozenset[str]:
        return frozenset(self.active_marks(key_id, version))


def exclude_for(marks: Sequence[Mark], version: str | None) -> frozenset[str]:
    """The rules ``marks`` leave out of a session of ``version``."""
    return frozenset(m.rule for m in marks if m.version is None or m.version == version)


# -- the audit row ------------------------------------------------------------------------


def audit(
    request: Request,
    *,
    key_id: str,
    username: str,
    action: str,
    rule: str,
    version: str | None,
) -> None:
    """One audit row for a mark or an unmark: who (``username``), when, the key,
    the rule and the version, in ``path`` after the route template (the key and
    the actor sit in ``token_id`` and ``username``). Written for an admin-token
    request too: the middleware's own row for it names only the route."""
    writer = getattr(request.app.state, "admin_audit", None)
    if writer is None:
        return
    ver = "all" if version is None else version
    writer.record(
        ts=time.time(),
        token_id=key_id,
        username=username,
        method=request.method,
        path=f"{route_template(request.scope)} {action}:{rule}@{ver}"[:300],
        status=200,
        duration_ms=0,
        client_ip=client_ip(request),
        peer_ip=request.client.host if request.client else None,
    )


# -- stored rows on read ------------------------------------------------------------------


def _clamp(v: float) -> int:
    return min(100, max(0, round(v)))


def _layer_value(layer: str, row: SessionScoreRow, exclude: frozenset[str]) -> int | None:
    """The layer's value with ``exclude`` left out, from the stored terms. The stored
    value when the layer was not scored or no excluded rule fired in it."""
    stored = {"E": row.e, "C": row.c, "R": row.r}[layer]
    t = row.terms.get(layer) or {}
    by_rule: Mapping[str, Any] = t.get("by_rule") or {}
    if stored is None or not isinstance(by_rule, dict):
        return stored
    fired = {k.split(":", 1)[0] for k in by_rule}
    if not (exclude & fired):
        return stored
    dropped = set(exclude) | set(t.get("excluded") or ())
    total = float(t.get("total") or 0)
    if layer == "E":
        lost = sum(float(v) for k, v in by_rule.items() if k not in dropped)
        return _clamp(100.0 * (1 - lost / total)) if total > 0 else stored
    if layer == "C":
        waste = sum(float(v) for k, v in by_rule.items() if k not in dropped)
        return _clamp(100.0 * max(0.0, 1 - 2 * waste / total)) if total > 0 else stored
    inputs = t.get("inputs") or {}
    turns, comp = int(inputs.get("turns") or 0), float(t.get("base") or 0)
    if turns <= 0:
        return stored
    bad = sum(
        float(v)
        for k, v in by_rule.items()
        if k.endswith(":turns") and k.split(":")[0] not in dropped
    )
    out = sum(
        float(v)
        for k, v in by_rule.items()
        if k.endswith(":out") and k.split(":")[0] not in dropped
    )
    bad = min(bad, float(inputs.get("bad") or 0))
    out = min(out, float(inputs.get("waste_out") or 0))
    token_part = 0.5 * out / comp if comp > 0 else 0.0
    return _clamp(100.0 * (1 - 0.5 * bad / turns - token_part))


def apply_marks(row: SessionScoreRow, exclude: frozenset[str]) -> SessionScoreRow:
    """``row`` with the layer values and the overall recomputed without the rules in
    ``exclude`` (from the stored per-rule terms; ranges are left as stored)."""
    if not exclude:
        return row
    vals: dict[str, int | None] = {k: _layer_value(k, row, exclude) for k in scoring.LAYERS}
    if (vals["E"], vals["C"], vals["R"]) == (row.e, row.c, row.r):
        return row  # no marked rule fired here: the stored scores stand, exactly
    got = [k for k in scoring.LAYERS if vals[k] is not None]
    overall = row.overall
    if overall is not None and len(got) >= scoring.MIN_OVERALL_LAYERS:
        wsum = sum(scoring.WEIGHTS[k] for k in got)
        overall = _clamp(sum(scoring.WEIGHTS[k] * (vals[k] or 0) for k in got) / wsum)
    return replace(row, e=vals["E"], c=vals["C"], r=vals["R"], overall=overall)


def apply_all(rows: Iterable[SessionScoreRow], marks: Sequence[Mark]) -> list[SessionScoreRow]:
    """Every row with the marks that apply to its prompt version left out."""
    return [apply_marks(r, exclude_for(marks, r.version)) for r in rows]
