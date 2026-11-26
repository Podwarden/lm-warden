"""Ranking the fixes of a session (analysis spec 12.2).

Every finding's saving is converted to one unit so they can be ranked: money
per session with a price card, else GPU-seconds (prefill plus decode), else
tokens. The unit is chosen once for the whole list (never money for one fix and
seconds for another). Rank = saving x confidence weight (high 1, medium 0.6, low
0.3); ties by rule order. Muted (won't fix) findings, the notices and the rules
counted in no score are never ranked; R-09 (a secret in the prompt) is pinned
above the top 3 as a notice.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.chat2.ledger import RateCard
from app.lens.catalogue import RULES, Confidence, Finding
from app.lens.response import ADMIN_CAUSES

CONFIDENCE_WEIGHT: dict[Confidence, float] = {"high": 1.0, "medium": 0.6, "low": 0.3}
#: Listed, never ranked: counted in no score (spec 12.2), and server or provider misses (A-11).
NEVER_RANKED = frozenset({"R-08", "R-09", "C-05", "E-08"})
TOP = 3
_ORDER = {rid: i for i, rid in enumerate(RULES)}


def _seconds(f: Finding) -> float | None:
    s = f.saving
    return (
        None
        if s.prefill_s is None and s.decode_s is None
        else (s.prefill_s or 0) + (s.decode_s or 0)
    )


def _value(f: Finding, unit: str) -> float:
    if unit == "money":
        return f.saving.money or 0.0
    if unit == "seconds":
        return _seconds(f) or 0.0
    return float(f.saving.tokens)


def _unit(fs: Sequence[Finding], card: RateCard | None) -> str:
    if card is not None and fs and all(f.saving.money is not None for f in fs if f.saving.tokens):
        return "money"
    if fs and all(_seconds(f) is not None for f in fs if f.saving.tokens):
        return "seconds"
    return "tokens"


def rank_key(f: Finding, card: RateCard | None) -> float:
    """Money when there is a card and the finding is priced, else seconds, else
    tokens; times the confidence weight."""
    unit = (
        "money"
        if card is not None and f.saving.money is not None
        else ("seconds" if _seconds(f) is not None else "tokens")
    )
    return _value(f, unit) * CONFIDENCE_WEIGHT[f.confidence]


def is_fix(f: Finding) -> bool:
    return (
        not f.muted
        and f.rule not in NEVER_RANKED
        and not f.counts.get("info")
        and f.cause not in ADMIN_CAUSES
    )


def top_fixes(findings: Sequence[Finding], card: RateCard | None) -> list[Finding]:
    """At most three fixes, best first."""
    fixes = [f for f in findings if is_fix(f)]
    unit = _unit(fixes, card)
    ranked = sorted(
        fixes,
        key=lambda f: (-_value(f, unit) * CONFIDENCE_WEIGHT[f.confidence], _ORDER[f.rule]),
    )
    return ranked[:TOP]


def pinned_notice(findings: Sequence[Finding]) -> Finding | None:
    """R-09: a secret in the prompt, a notice above the top 3, never ranked."""
    return next((f for f in findings if f.rule == "R-09" and not f.muted), None)
