"""Prompt Lens session analysis: the three objective layers over one session.

``analyse_session`` is pure: the caller (the service, Task 1.8) reads the rows
and everything that is not on a row (toolset entries, the key's recent calls,
the learned prefill rate, the window, the route's tool limit, the guard log's
roles) and passes it in ``AnalyseContext``. The result holds codes, counts and
turn numbers only (A-10).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from app.lens.catalogue import RULES, Finding, Layer
from app.lens.construction import construction_findings, waste_terms
from app.lens.efficiency import CrossSessions, efficiency_findings
from app.lens.response import bad_turns, response_findings
from app.lens.turns import LensTurn, turns_of
from app.sessions.store import Row

Entries = Mapping[str, list[tuple[str, str, int]]]


@dataclass(frozen=True)
class AnalyseContext:
    """Inputs not stored on a row. ``toolsets`` is ``store.toolsets`` (digest
    -> ``(hash8, family, chars)`` in offer order); ``recent_calls`` the key's
    calls per tool hash over its recent sessions (``store.recent_toolset_calls``);
    ``prefill_rate`` tokens/s or None; ``route_tool_limit`` None when the route
    has no modelled limit. ``cross_sessions`` counts the key's own earlier
    sessions with the same E-01 line (per span line: ``service.cross_sessions``); ``guard_where`` is the guard log per
    request id: where code (``response.guard_where_of``) -> values."""

    toolsets: Entries
    recent_calls: Mapping[str, int]
    prefill_rate: float | None
    max_model_len: int | None
    route_tool_limit: int | None
    cross_sessions: CrossSessions = 0
    guard_where: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


@dataclass
class SessionAnalysis:
    """``markers``: turn number -> rule ids of the findings that name it, in
    catalogue order. ``coverage``: per layer, the share of turns carrying every
    field the layer needs. ``bad``: per scored R rule, the bad turns (10.2);
    ``waste``: per C rule, the waste tokens (10.2). S2 scores from these."""

    findings: list[Finding]
    markers: dict[int, list[str]]
    coverage: dict[Layer, float]
    turns: list[LensTurn]
    bad: dict[str, tuple[int, ...]] = field(default_factory=dict)
    waste: dict[str, int] = field(default_factory=dict)


def _has_e(t: LensTurn) -> bool:
    """Own-lens fields: the reuse figure and the layer 1 verdict."""
    return t.reused_own is not None and (t.cause is not None or t.lost_tokens is not None)


def _has_c(t: LensTurn) -> bool:
    """Shape, and the toolset when tools were offered."""
    return t.shape is not None and (t.toolset is not None or not t.tools_in)


def _has_r(t: LensTurn) -> bool:
    """The reply fields of layer 3 (absent on relays and with kvsig off)."""
    return (
        t.reply_chars is not None
        and t.calls_n is not None
        and t.calls_bad is not None
        and t.reasoning_tokens is not None
    )


def _coverage(turns: Sequence[LensTurn]) -> dict[Layer, float]:
    if not turns:
        return {"E": 0.0, "C": 0.0, "R": 0.0}
    return {
        "E": sum(1 for t in turns if _has_e(t)) / len(turns),
        "C": sum(1 for t in turns if _has_c(t)) / len(turns),
        "R": sum(1 for t in turns if _has_r(t)) / len(turns),
    }


def lint_of(turns: Sequence[LensTurn], toolsets: Entries) -> dict[str, int]:
    """Tool hash -> lint flags: each turn's ``tools_lint`` (toolset order)
    against its toolset's entries, OR-ed over the session."""
    out: dict[str, int] = {}
    for t in turns:
        ents = toolsets.get(t.toolset) if t.toolset is not None else None
        if not ents or t.tools_lint is None:
            continue
        for (h, _family, _chars), flags in zip(ents, t.tools_lint, strict=False):
            out[h] = out.get(h, 0) | flags
    return out


def family_sizes(turns: Sequence[LensTurn], toolsets: Entries) -> dict[str, int]:
    """Family -> the most tools of that family any of the session's toolsets offered."""
    out: dict[str, int] = {}
    for ts_id in {t.toolset for t in turns if t.toolset is not None}:
        for fam, n in Counter(f for _h, f, _c in toolsets.get(ts_id) or ()).items():
            out[fam] = max(out.get(fam, 0), n)
    return out


def analyse_session(rows: Sequence[Row], *, own: bool, ctx: AnalyseContext) -> SessionAnalysis:
    """Layers E, C and R over one session's stored rows. ``own=True`` reads
    the key's own lens (a key holder, A-1)."""
    turns = turns_of(rows, own=own)
    if ctx.max_model_len:
        turns = [
            replace(t, max_model_len=ctx.max_model_len) if t.max_model_len is None else t
            for t in turns
        ]
    findings = (
        efficiency_findings(turns, cross_sessions=ctx.cross_sessions, prefill_rate=ctx.prefill_rate)
        + construction_findings(
            turns,
            toolset_entries=ctx.toolsets,
            recent_calls=ctx.recent_calls,
            max_model_len=ctx.max_model_len,
            route_tool_limit=ctx.route_tool_limit,
        )
        + response_findings(
            turns,
            lint=lint_of(turns, ctx.toolsets),
            family_sizes=family_sizes(turns, ctx.toolsets),
            guard_where=ctx.guard_where,
            prefill_rate=ctx.prefill_rate,
        )
    )
    order = {rid: i for i, rid in enumerate(RULES)}
    findings.sort(key=lambda f: order[f.rule])  # stable: keeps each layer's order within a rule
    markers: dict[int, list[str]] = {}
    for f in findings:
        for n in f.turns:
            ids = markers.setdefault(n, [])
            if f.rule not in ids:
                ids.append(f.rule)
    return SessionAnalysis(
        findings=findings,
        markers=markers,
        coverage=_coverage(turns),
        turns=turns,
        bad=bad_turns(turns),
        waste=waste_terms(turns, toolset_entries=ctx.toolsets, recent_calls=ctx.recent_calls),
    )


def findings_for_request(a: SessionAnalysis, req_id: str) -> list[Finding]:
    """The findings whose turns include ``req_id``'s turn; [] when the request
    is not in the session."""
    n = next((t.n for t in a.turns if t.req_id == req_id), None)
    if n is None:
        return []
    return [f for f in a.findings if n in f.turns]
