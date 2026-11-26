"""Prompt Lens construction checks C-01..C-09 (analysis spec 7.2, 11.2, 10.2).

Pure functions over a session's ``LensTurn`` list plus the few inputs that are
not on a row: the toolset entries (``store.toolsets``), the key's recent calls
per tool (``store.recent_toolset_calls``), the model's window and the route's
tool limit. Every token figure goes through the turn's own tokens per char,
never a global constant. A finding holds codes and numbers only (A-10): a tool
is a hash, never a name. ``counts["info"] == 1`` marks an info-severity finding
(a rarely used tool, a head that is reused): it carries no saving and does not
count as a fault.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence

from app.lens.catalogue import RULES, Confidence, Finding, Saving
from app.lens.efficiency import efficiency_findings
from app.lens.turns import LensTurn

Entries = Mapping[str, Sequence[tuple[str, str, int]]]

#: C-01: unused schema tokens per turn, or share of the prompt, and session length.
C01_TOKENS = 1_000
C01_SHARE = 0.05
C01_TURNS = 10
#: C-02: a repeated result under this many tokens is ignored; repeats needed.
C02_TOKENS = 500
C02_REPEATS = 2
#: C-03: head share of the median prompt, head size, reuse share for "info".
C03_SHARE = 0.70
C03_TOKENS = 8_000
C03_REUSED = 0.90
#: C-04: tool results' share of body growth, and the one result's size.
C04_SHARE = 0.70
C04_RESULT = 8_000
#: C-05: window share projected, compaction share, turns needed to project.
C05_FILL = 0.80
C05_COMPACT = 0.50
C05_TURNS = 5
#: C-07: boilerplate share of the prompt; counted at this weight in C (10.2).
C07_SHARE = 0.10
BOIL_WEIGHT = 0.5
#: C-08: outlier factor over the median definition, and its size.
C08_FACTOR = 3
C08_TOKENS = 2_000
#: C-09: tools from which a route's limit is near.
C09_TOOLS = 100


def _finding(
    rule: str,
    turns: Sequence[int],
    counts: dict[str, int],
    *,
    tokens: int = 0,
    tool: str | None = None,
    cause: str | None = None,
    confidence: Confidence = "high",
) -> Finding:
    assert rule in RULES
    return Finding(
        rule=rule,
        turns=tuple(turns),
        counts=counts,
        tool=tool,
        cause=cause,
        saving=Saving(max(int(tokens), 0), None, None, None),
        confidence=confidence,
    )


def _pct(x: float) -> int:
    return int(round(100 * x))


def _called(turns: Sequence[LensTurn]) -> set[str]:
    """Tool hashes the session called: index [1] of ``[family, hash8(, args8)]``."""
    return {e[1] for t in turns for e in t.tools_out if len(e) >= 2}


def _lanes(turns: Sequence[LensTurn]) -> list[list[LensTurn]]:
    by: dict[object, list[LensTurn]] = {}
    for t in turns:
        by.setdefault(t.lane if t.lane is not None else ("solo", t.n), []).append(t)
    return list(by.values())


def _after(turns: Sequence[LensTurn], t: LensTurn) -> int:
    """Turns of ``t``'s lane from ``t`` on: how long a block stays in history."""
    return sum(1 for u in turns if u.n >= t.n and (t.lane is None or u.lane == t.lane))


def _call_hash(turns: Sequence[LensTurn], t: LensTurn) -> str | None:
    """The one tool the turn before ``t`` called, if exactly one."""
    prev = [u for u in turns if u.n == t.n - 1]
    hashes = {e[1] for u in prev for e in u.tools_out if len(e) >= 2}
    return next(iter(hashes)) if len(hashes) == 1 else None


# -- C-01: tools offered, never called -----------------------------------------


def _unused(
    turns: Sequence[LensTurn], entries: Entries, recent: Mapping[str, int]
) -> list[tuple[LensTurn, int, int, set[str], set[str], set[str]]]:
    """Per turn that carried a known toolset: ``(turn, never_tokens,
    rarely_tokens, offered, never, rarely)``. Never: not called in this session
    nor in the key's recent sessions; rarely: called there, not here."""
    called = _called(turns)
    out = []
    for t in turns:
        ents = entries.get(t.toolset) if t.toolset is not None else None
        if not ents:
            continue
        never = rarely = 0
        offered: set[str] = set()
        nev: set[str] = set()
        rar: set[str] = set()
        for h, _family, chars in ents:
            offered.add(h)
            if h in called:
                continue
            tokens = round(max(chars, 0) * t.tokens_per_char)
            if recent.get(h, 0) > 0:
                rarely += tokens
                rar.add(h)
            else:
                never += tokens
                nev.add(h)
        out.append((t, never, rarely, offered, nev, rar))
    return out


def _c01(turns: Sequence[LensTurn], entries: Entries, recent: Mapping[str, int]) -> list[Finding]:
    if len(turns) < C01_TURNS:
        return []
    per = _unused(turns, entries, recent)
    if not per:
        return []
    carrying = [p[0].n for p in per]
    prompt = sum(max(p[0].prompt_tokens, 0) for p in per)
    never = sum(p[1] for p in per)
    rarely = sum(p[2] for p in per)

    def over(tokens: int) -> bool:
        return tokens / len(per) >= C01_TOKENS or (prompt > 0 and tokens / prompt >= C01_SHARE)

    nev_set = set().union(*(p[4] for p in per))
    rar_set = set().union(*(p[5] for p in per))
    counts = {
        "unused": len(nev_set),
        "offered": len(set().union(*(p[3] for p in per))),
        "rarely": len(rar_set),
        "tokens": round(never / len(per)),
        "turns": len(per),
    }
    if over(never):
        return [_finding("C-01", carrying, counts, tokens=never)]
    if rar_set and over(never + rarely):
        # A tool called in a recent session is "rarely used": info, not a fault.
        counts.update(tokens=round((never + rarely) / len(per)), info=1)
        return [_finding("C-01", carrying, counts, confidence="low")]
    return []


# -- C-02: the same output sent again --------------------------------------------


def _events(
    turns: Sequence[LensTurn], value: Callable[[LensTurn], int | None]
) -> list[tuple[LensTurn, int]]:
    """Rises of a within-prompt figure along each lane: one copy entering."""
    out: list[tuple[LensTurn, int]] = []
    for lane in _lanes(turns):
        prev = 0
        for t in lane:
            v = value(t)
            if v is None:
                continue
            if v > prev:
                out.append((t, v - prev))
            prev = v
    return out


def _dup(t: LensTurn) -> int | None:
    return t.dead_weight[0] if t.dead_weight is not None else None


def _c02(turns: Sequence[LensTurn]) -> list[Finding]:
    rep = [(t, s, True) for t, s in _events(turns, lambda t: t.repeat_results) if s >= C02_TOKENS]
    dup = [(t, s, False) for t, s in _events(turns, _dup) if s >= C02_TOKENS]
    ev = rep + dup
    if len(ev) < C02_REPEATS:
        return []
    tools = {_call_hash(turns, t) for t, _s, is_rep in ev if is_rep} if not dup else {None}
    tool = next(iter(tools)) if len(tools) == 1 else None
    saving = sum(s * _after(turns, t) for t, s, _ in ev)
    counts = {
        "repeats": len(ev),
        "tokens": round(statistics.median(s for _, s, _ in ev)),
        "total": sum(s for _, s, _ in ev),
    }
    return [
        _finding(
            "C-02",
            sorted({t.n for t, _, _ in ev}),
            counts,
            tokens=saving,
            tool=tool,
            confidence="high",
        )
    ]


# -- C-03: head share ------------------------------------------------------------


def _int(v: object) -> int | None:
    """``v`` when it is an int (a bool is not), else None."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _head_tokens(t: LensTurn) -> tuple[float, float] | None:
    sh = t.shape
    if not sh:
        return None
    sys_c, tools_c, chars = _int(sh.get("sys")), _int(sh.get("tools_chars")), _int(sh.get("chars"))
    if sys_c is None or tools_c is None or chars is None:
        return None
    if chars <= 0 or sys_c < 0 or tools_c < 0:
        return None
    head = sys_c + tools_c
    return min(head / chars, 1.0), head * t.tokens_per_char


def _c03(turns: Sequence[LensTurn]) -> list[Finding]:
    got = [(t, h) for t in turns if (h := _head_tokens(t)) is not None]
    if not got:
        return []
    share = statistics.median(h[0] for _, h in got)
    tokens = statistics.median(h[1] for _, h in got)
    if share < C03_SHARE or tokens < C03_TOKENS:
        return []
    counts = {"share": _pct(share), "tokens": round(tokens), "turns": len(got)}
    cont = [(t, h) for t, h in got if t.edge != "new" and t.reused_own is not None]
    info = False
    if cont:
        reused = sum(1 for t, h in cont if (t.reused_own or 0) >= h[1]) / len(cont)
        counts["reused"] = _pct(reused)
        info = reused >= C03_REUSED
    if info:
        counts["info"] = 1
    return [_finding("C-03", [t.n for t, _ in got], counts, confidence="low" if info else "high")]


# -- C-04: tool results dominate -------------------------------------------------


def _c04(turns: Sequence[LensTurn]) -> list[Finding]:
    """Per continuing turn the growth of the prompt, less the reply and the new
    user message, is the tool results that came in (shape has no result sizes:
    an estimate, hence medium confidence at most)."""
    growth = 0.0
    results = 0.0
    big: list[tuple[LensTurn, float]] = []
    for lane in _lanes(turns):
        for p, t in zip(lane, lane[1:], strict=False):
            a, b = p.shape, t.shape
            if not a or not b:
                continue
            ca, cb = a.get("chars"), b.get("chars")
            ma, mb = a.get("msgs"), b.get("msgs")
            if not isinstance(ca, int) or not isinstance(cb, int):
                continue
            if not isinstance(ma, dict) or not isinstance(mb, dict) or cb <= ca:
                continue
            g = cb - ca
            growth += g * t.tokens_per_char
            d_tool = (mb.get("tool") or 0) - (ma.get("tool") or 0)
            if d_tool <= 0:
                continue
            new_user = b.get("last_user") if (mb.get("user") or 0) > (ma.get("user") or 0) else 0
            res = max(g - (new_user if isinstance(new_user, int) else 0) - (p.reply_chars or 0), 0)
            tok = res * t.tokens_per_char
            results += tok
            if tok / d_tool >= C04_RESULT:
                big.append((t, tok / d_tool))
    if growth <= 0 or results / growth < C04_SHARE or not big:
        return []
    largest_t, largest = max(big, key=lambda x: x[1])
    saving = sum((r - C04_RESULT) * _after(turns, t) for t, r in big)
    counts = {
        "share": _pct(results / growth),
        "largest": round(largest),
        "results": len(big),
    }
    return [
        _finding(
            "C-04",
            [t.n for t, _ in big],
            counts,
            tokens=round(saving),
            tool=_call_hash(turns, largest_t),
            confidence="medium",
        )
    ]


# -- C-05: context growth --------------------------------------------------------


def _p90(vals: Sequence[int]) -> int:
    s = sorted(vals)
    return s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)]


def _turn_at(n: int, prompt: int, growth: int, target: float) -> int:
    """The turn a prompt of ``prompt`` tokens at turn ``n`` reaches ``target``."""
    return n + max(math.ceil((target - prompt) / growth), 0)


def _c05(turns: Sequence[LensTurn], max_model_len: int | None) -> list[Finding]:
    out: list[Finding] = []
    for lane in _lanes(turns):
        limit = max_model_len or next((t.max_model_len for t in lane if t.max_model_len), None)
        if not limit or limit <= 0:
            continue
        compactions = [
            (p, t)
            for p, t in zip(lane, lane[1:], strict=False)
            if t.edge == "compacted_from" and p.prompt_tokens < C05_COMPACT * limit
        ]
        if compactions:
            at = round(statistics.median(100 * p.prompt_tokens / limit for p, _ in compactions))
            out.append(
                _finding(
                    "C-05",
                    [t.n for _, t in compactions],
                    {"compact_at": int(at), "compactions": len(compactions)},
                    confidence="medium",
                )
            )
        if len(lane) < C05_TURNS:
            continue
        deltas = [b.prompt_tokens - a.prompt_tokens for a, b in zip(lane, lane[1:], strict=False)]
        rises = [d for d in deltas if d > 0]
        if not rises:
            continue
        g = round(statistics.median(rises))
        last = lane[-1]
        room = last.prompt_tokens

        n80 = _turn_at(last.n, room, g, C05_FILL * limit)
        if n80 - last.n <= len(lane):  # reached within as many turns as the session has run
            out.append(
                _finding(
                    "C-05",
                    [last.n],
                    {
                        "growth": g,
                        "p90": _p90(rises),
                        "fill_turn": _turn_at(last.n, room, g, limit),
                        "turns": len(lane),
                    },
                    confidence="medium",
                )
            )
    return out


# -- C-06: volatile before stable ------------------------------------------------


def _c06(turns: Sequence[LensTurn]) -> list[Finding]:
    """The structural fact behind an E-01 finding; its cost is E's (10.2)."""
    out = []
    for f in efficiency_findings(turns, cross_sessions=0):
        if f.rule != "E-01" or "line" not in f.counts:
            continue
        counts = {"line": f.counts["line"], "turns": f.counts["turns"]}
        out.append(_finding("C-06", f.turns, counts, cause=f.cause, confidence=f.confidence))
    return out


# -- C-07: boilerplate -----------------------------------------------------------


def _boil(t: LensTurn) -> int | None:
    return t.dead_weight[1] if t.dead_weight is not None else None


def _c07(turns: Sequence[LensTurn]) -> list[Finding]:
    known = [(t, b) for t in turns if (b := _boil(t)) is not None]
    prompt = sum(max(t.prompt_tokens, 0) for t, _ in known)
    boil = sum(b for _, b in known)
    if not known or prompt <= 0 or boil / prompt < C07_SHARE:
        return []
    counts = {"share": _pct(boil / prompt), "tokens": round(boil / len(known)), "turns": len(known)}
    # Often the harness's reminders, not the app's: medium confidence.
    return [
        _finding("C-07", [t.n for t, b in known if b > 0], counts, tokens=boil, confidence="medium")
    ]


# -- C-08: schema outlier --------------------------------------------------------


def _c08(turns: Sequence[LensTurn], entries: Entries) -> list[Finding]:
    called = [h for t in turns for e in t.tools_out if len(e) >= 2 for h in (e[1],)]
    best: dict[str, Finding] = {}
    for ts_id in {t.toolset for t in turns if t.toolset}:
        ents = entries.get(ts_id)
        if not ents or len(ents) < 2:
            continue
        carrying = [t for t in turns if t.toolset == ts_id]
        med = statistics.median(c for _, _, c in ents)
        tpc = statistics.fmean(t.tokens_per_char for t in carrying)
        for h, _family, chars in ents:
            if med <= 0 or chars < C08_FACTOR * med or chars * tpc < C08_TOKENS:
                continue
            saving = sum(round((chars - med) * t.tokens_per_char) for t in carrying)
            counts = {
                "tokens": round(chars * tpc),
                "x": round(chars / med),
                "used": sum(1 for x in called if x == h),
                "turns": len(carrying),
            }
            f = _finding("C-08", [t.n for t in carrying], counts, tokens=saving, tool=h)
            if h not in best or f.saving.tokens > best[h].saving.tokens:
                best[h] = f
    return list(best.values())


# -- C-09: near a tool limit -----------------------------------------------------


def _c09(turns: Sequence[LensTurn], limit: int | None) -> list[Finding]:
    if not limit:
        return []
    many = [t for t in turns if t.tools_in is not None and t.tools_in >= C09_TOOLS]
    if not many:
        return []
    n = max(t.tools_in or 0 for t in many)
    return [_finding("C-09", [t.n for t in many], {"tools": n, "limit": int(limit)})]


# -- public ----------------------------------------------------------------------


def construction_findings(
    turns: Sequence[LensTurn],
    *,
    toolset_entries: Mapping[str, list[tuple[str, str, int]]],
    recent_calls: Mapping[str, int],
    max_model_len: int | None,
    route_tool_limit: int | None,
) -> list[Finding]:
    """The session's C findings, in catalogue order (largest saving first
    within a rule). ``toolset_entries`` is ``store.toolsets`` (digest ->
    ``(hash8, family, chars)``); ``recent_calls`` the key's calls per tool hash
    over its last sessions with the toolset (``store.recent_toolset_calls``);
    ``route_tool_limit`` the route's modelled tool limit, None if none."""
    found = (
        _c01(turns, toolset_entries, recent_calls)
        + _c02(turns)
        + _c03(turns)
        + _c04(turns)
        + _c05(turns, max_model_len)
        + _c06(turns)
        + _c07(turns)
        + _c08(turns, toolset_entries)
        + _c09(turns, route_tool_limit)
    )
    order = {rid: i for i, rid in enumerate(RULES)}
    return sorted(found, key=lambda f: (order[f.rule], -f.saving.tokens))


def waste_terms(
    turns: Sequence[LensTurn],
    *,
    toolset_entries: Entries | None = None,
    recent_calls: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Per rule id, the waste tokens C counts (10.2): ``C-01`` the unused
    tools' schema (never called here or in the key's recent sessions; zero
    without ``toolset_entries``), ``C-02`` dup plus repeated results, ``C-07``
    boilerplate at half weight. Reasoning counts zero (A-6)."""
    unused = sum(p[1] for p in _unused(turns, toolset_entries or {}, recent_calls or {}))
    c02 = sum(
        (t.dead_weight[0] if t.dead_weight is not None else 0) + max(t.repeat_results or 0, 0)
        for t in turns
    )
    c07 = sum(b for t in turns if (b := _boil(t)) is not None)
    return {"C-01": unused, "C-02": c02, "C-07": round(BOIL_WEIGHT * c07)}
