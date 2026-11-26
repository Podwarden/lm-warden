"""Prompt Lens response signals R-01..R-10 and their likely causes (analysis
spec 8.1, 8.2, 11.3, 10.2).

Pure functions over a session's ``LensTurn`` list. Each finding's cause is the
first rule of spec 8.2 that matches, else ``none_found`` (the finding says so
instead of guessing). Server and engine causes (``server_load``,
``parser_mismatch_admin``) are shown but never count as bad turns and carry no
saving, so they never rank as a prompt fix (A-11). One bad turn is never a
recommendation (A-12); R-03, which the spec reports "even once", shows a single
occurrence as info. R-02 fires from one cut (spec 11.3: ≥ 1, guard none).
Info findings carry
``counts["info"] == 1``, low confidence and no saving (Task 1.6 convention).
Findings hold codes and numbers only (A-10): a tool is its name hash.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from app.lens.catalogue import RULES, Confidence, Finding, Saving
from app.lens.reply_verdicts import CALL_CODES
from app.lens.schema_lint import (
    LINT_NESTED_NO_REQUIRED,
    LINT_NO_DESC,
    LINT_NO_TYPE,
    LINT_SHORT_DESC,
)
from app.lens.turns import LensTurn

#: Spec 8.2's causes, in the order its table lists them; ``none_found`` last.
RESPONSE_CAUSES = (
    "limit_below_need",
    "thinking_used_budget",
    "window_filled",
    "schema_untyped",
    "schema_undocumented",
    "nested_no_required",
    "short_description",
    "similar_tools",
    "parser_mismatch_admin",
    "result_not_final",
    "harness_no_result",
    "error_too_short",
    "same_args_retry",
    "template_swallowed",
    "no_after_tool_instruction",
    "thinking_for_short_step",
    "secret_in_tool_results",
    "secret_in_system",
    "recompute_dominates",
    "server_load",
    "none_found",
)
#: Causes that are the server's or the engine's, never the prompt's (A-11).
ADMIN_CAUSES = frozenset({"parser_mismatch_admin", "server_load"})

#: A-12: the fewest bad turns that make a recommendation. R-01's "≥ 2 turns or
#: ≥ 10 % of turns" and R-07's "≥ 2 or ≥ 5 %" therefore reduce to this floor.
MIN_BAD = 2
#: R-01: ``max_tokens`` at or under this is a probe or a classifier (ignored).
PROBE_MAX_TOKENS = 16
#: R-01/R-02: a cut is the window's when prompt + completion reaches the window less this.
WINDOW_SLACK = 16
#: R-01: reasoning share of a cut reply above which "thinking used the budget".
R01_REASONING = 0.60
#: R-04: bad calls of a tool, or their share of its calls.
R04_CALLS = 3
R04_SHARE = 0.20
#: R-04: offered tools of one family from which the model "mixes up similar tools".
SIMILAR_TOOLS = 20
#: R-05: identical calls in a row, and the result size under which a repeated
#: result reads as "does not say it is done" (tokens).
R05_CALLS = 3
R05_SHORT_RESULT = 1_024
#: R-06: a run is this many consecutive failed calls; runs or total needed.
R06_RUN = 2
R06_RUNS = 2
R06_TOTAL = 3
#: R-06: an error result under this many characters "does not say what to change".
#: ``tools_in`` stores tokens (chars / 4, rounded up): read back at 4 chars per token.
R06_SHORT_CHARS = 40
RESULT_CHARS_PER_TOKEN = 4
#: R-08: reasoning tokens, answer tokens, turns; the floor the saving keeps.
R08_REASONING = 2_000
R08_ANSWER = 40
R08_TURNS = 3
R08_FLOOR = 512
#: R-10: median TTFT, new tokens per turn, and the share that decides the cause.
R10_TTFT_S = 2.0
R10_NEW_TOKENS = 1_000
R10_SHARE = 0.50

#: Finish reasons of a reply that ended normally (an empty one of these is R-07).
_NORMAL_FINISH = frozenset({"stop", "end_turn", "stop_sequence", "tool_calls", "tool_use"})
#: Layer 1 causes whose lost reuse is not the prompt's (A-11): no R-10 recompute.
_SERVER_LOSS = frozenset({"server_evicted", "misrouted", "provider_below_min"})
#: Guard ``where`` codes (see ``guard_where``) and the cause each names.
GUARD_WHERE = ("tool_result", "system", "user", "assistant", "tools")


def guard_where_of(role: str | None, kind: str | None) -> str | None:
    """A ``guard_events`` row's role and kind as a ``guard_where`` code."""
    if kind == "tool result" or role == "tool":
        return "tool_result"
    if role in ("system", "developer"):
        return "system"
    if role in ("user", "assistant", "tools"):
        return role
    return None


# -- helpers ---------------------------------------------------------------------


def _finding(
    rule: str,
    turns: Iterable[int],
    counts: dict[str, int],
    *,
    cause: str,
    tokens: int = 0,
    tool: str | None = None,
    confidence: Confidence = "high",
    prefill_s: float | None = None,
) -> Finding:
    assert rule in RULES and cause in RESPONSE_CAUSES
    if cause in ADMIN_CAUSES:
        # An engine or server cause is not the prompt's: never ranked as a fix.
        tokens, prefill_s = 0, None
    return Finding(
        rule=rule,
        turns=tuple(sorted(set(turns))),
        counts=counts,
        tool=tool,
        cause=cause,
        saving=Saving(max(int(tokens), 0), prefill_s, None, None),
        confidence=confidence,
    )


def _info(f: Finding) -> Finding:
    """Info severity: shown, never a recommendation, no saving."""
    f.counts["info"] = 1
    f.confidence = "low"
    f.saving = Saving(0, None, None, None)
    return f


def _lanes(turns: Sequence[LensTurn]) -> list[list[LensTurn]]:
    by: dict[object, list[LensTurn]] = {}
    for t in turns:
        by.setdefault(t.lane if t.lane is not None else ("solo", t.n), []).append(t)
    return list(by.values())


def _prev_in_lane(turns: Sequence[LensTurn]) -> dict[int, LensTurn]:
    out: dict[int, LensTurn] = {}
    for lane in _lanes(turns):
        for p, t in zip(lane, lane[1:], strict=False):
            out[t.n] = p
    return out


def _p90(vals: Sequence[int]) -> int:
    s = sorted(vals)
    return s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)] if s else 0


def _most(vals: Iterable[object]) -> object:
    c = Counter(v for v in vals if v is not None)
    return c.most_common(1)[0][0] if c else None


def _comp(ts: Iterable[LensTurn]) -> int:
    return sum(max(t.completion_tokens, 0) for t in ts)


def _msgs(t: LensTurn, role: str) -> int | None:
    m = (t.shape or {}).get("msgs")
    v = m.get(role) if isinstance(m, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _got_results(prev: LensTurn | None, t: LensTurn) -> bool | None:
    """Whether ``t``'s prompt carried tool results that ``prev``'s did not:
    from ``tools_in`` when stored, else from the shape's tool-message count."""
    if t.results is not None:
        return len(t.results) > 0
    if prev is None:
        return None
    a, b = _msgs(prev, "tool"), _msgs(t, "tool")
    if a is None or b is None:
        return None
    return b > a


def _calls(t: LensTurn) -> list[tuple[str, ...]]:
    return [e for e in t.tools_out if len(e) >= 2]


# -- per-turn signals (shared by the findings and ``bad_turns``) ------------------


def _cut(t: LensTurn) -> str | None:
    """``"window"``, ``"client"`` or None (not a length cut, a probe, or unknown cap)."""
    if t.finish_reason != "length":
        return None
    mml = t.max_model_len
    if mml and t.prompt_tokens + t.completion_tokens >= mml - WINDOW_SLACK:
        return "window"
    if t.max_tokens_req is None or t.max_tokens_req <= PROBE_MAX_TOKENS:
        return None
    return "client"


def _runaway(t: LensTurn) -> bool:
    return bool(t.finish_reason and t.finish_reason.startswith("runaway_"))


def _empty(t: LensTurn) -> bool:
    """No text and no tool call, on a reply that ended normally (a reply with
    only tool calls is not empty)."""
    return (
        t.reply_chars == 0
        and t.calls_n == 0
        and not _calls(t)
        and (t.finish_reason or "") in _NORMAL_FINISH
    )


def _bad_verdicts(t: LensTurn) -> list[tuple[str, str]]:
    """``(hash8, code)`` of the turn's non-ok verdicts (the call index aside)."""
    return [(h, c) for _k, h, c in (t.calls_bad or ()) if c != "ok"]


# -- R-01 / R-02 / R-03 ------------------------------------------------------------


def _r01(turns: Sequence[LensTurn]) -> list[Finding]:
    cuts = [t for t in turns if _cut(t) == "client"]
    if len(cuts) < MIN_BAD:
        return []
    m = Counter(t.max_tokens_req or 0 for t in cuts).most_common(1)[0][0]
    # A cut reply needed more than it got: it counts one over its size, so a
    # cap that cuts >= 10 % of replies sits below the session's p90 need.
    cut_ns = {t.n for t in cuts}
    p90 = _p90([t.completion_tokens + (1 if t.n in cut_ns else 0) for t in turns])
    thinking = sum(
        1
        for t in cuts
        if t.completion_tokens > 0
        and (t.reasoning_tokens or 0) > R01_REASONING * t.completion_tokens
    )
    if p90 > m:
        cause = "limit_below_need"
    elif 2 * thinking >= len(cuts):
        cause = "thinking_used_budget"
    else:
        cause = "none_found"
    # The p90 that decided the cause (cut replies at +1), never a bare "p90 = m".
    counts = {"cuts": len(cuts), "m": m, "p90": p90}
    return [_finding("R-01", [t.n for t in cuts], counts, cause=cause, tokens=_comp(cuts))]


def _r02(turns: Sequence[LensTurn]) -> list[Finding]:
    cuts = [t for t in turns if _cut(t) == "window"]
    if not cuts:
        return []
    counts = {"cuts": len(cuts), "turn": cuts[0].n, "prompt": cuts[0].prompt_tokens}
    f = _finding("R-02", [t.n for t in cuts], counts, cause="window_filled", tokens=_comp(cuts))
    return [f]


def _r03(turns: Sequence[LensTurn]) -> list[Finding]:
    hits = [t for t in turns if _runaway(t)]
    if not hits:
        return []
    kinds = Counter((t.finish_reason or "")[len("runaway_") :] for t in hits)
    counts = {
        "turns": len(hits),
        "think": kinds.get("think", 0),
        "repeat": kinds.get("repeat", 0),
        "hard": kinds.get("hard", 0),
    }
    f = _finding("R-03", [t.n for t in hits], counts, cause="none_found", tokens=_comp(hits))
    # Reported even once (rare and costly); a recommendation only from 2.
    return [_info(f) if len(hits) < MIN_BAD else f]


# -- R-04 ---------------------------------------------------------------------------


def _schema_cause(flags: int, family_size: int | None) -> str:
    if flags & LINT_NO_TYPE:
        return "schema_untyped"
    if flags & LINT_NO_DESC:
        return "schema_undocumented"
    if flags & LINT_NESTED_NO_REQUIRED:
        return "nested_no_required"
    if flags & LINT_SHORT_DESC:
        return "short_description"
    if family_size is not None and family_size >= SIMILAR_TOOLS:
        return "similar_tools"
    return "none_found"


def _r04(
    turns: Sequence[LensTurn], lint: Mapping[str, int], family_sizes: Mapping[str, int]
) -> list[Finding]:
    judged = [t for t in turns if t.calls_bad is not None]
    calls: Counter[str] = Counter()
    family: dict[str, str] = {}
    for t in judged:
        for e in _calls(t):
            calls[e[1]] += 1
            family.setdefault(e[1], e[0])
    hits: dict[tuple[str, bool], list[tuple[LensTurn, str]]] = {}
    for t in judged:
        for h, code in _bad_verdicts(t):
            hits.setdefault((h, code == "leaked_markup"), []).append((t, code))
    out: list[Finding] = []
    for (h, admin), hs in hits.items():
        k = len(hs)
        n = max(calls.get(h, 0), k)
        ts = {t.n: t for t, _ in hs}
        if len(ts) < MIN_BAD or (k < R04_CALLS and k < R04_SHARE * n):
            continue
        if admin:
            # The engine's tool parser did not match the model's format: admin.
            cause = "parser_mismatch_admin"
        else:
            fam = family.get(h)
            cause = _schema_cause(lint.get(h, 0), family_sizes.get(fam) if fam else None)
        code = str(_most(c for _, c in hs))
        counts = {
            "bad": k,
            "calls": n,
            "code": CALL_CODES.index(code) if code in CALL_CODES else -1,
        }
        out.append(_finding("R-04", ts, counts, cause=cause, tokens=_comp(ts.values()), tool=h))
    return out


# -- R-05 ---------------------------------------------------------------------------


def _single_call(t: LensTurn) -> tuple[str, str] | None:
    cs = _calls(t)
    if len(cs) != 1 or len(cs[0]) < 3:
        return None
    return cs[0][1], cs[0][2]


def _rise(prev: LensTurn, t: LensTurn) -> int:
    """Repeated-result tokens that entered with ``t``: the result of ``prev``'s
    call duplicated an earlier one (``lens_repeat_results`` rose)."""
    if prev.repeat_results is None or t.repeat_results is None:
        return 0
    return max(t.repeat_results - prev.repeat_results, 0)


def _loops(turns: Sequence[LensTurn]) -> list[tuple[list[LensTurn], list[int], bool]]:
    """``(calls, repeated result sizes, a result was missing)`` per loop: the
    same tool and arguments on consecutive turns of a lane, the results in
    between identical (or none came back), at least ``R05_CALLS`` calls."""
    out = []
    for lane in _lanes(turns):
        i = 0
        while i < len(lane):
            key = _single_call(lane[i])
            j = i
            while key is not None and j + 1 < len(lane) and _single_call(lane[j + 1]) == key:
                j += 1
            seg = lane[i : j + 1]
            i = j + 1
            if key is None or len(seg) < R05_CALLS:
                continue
            members = seg[:2]
            rises: list[int] = []
            missing = _got_results(seg[0], seg[1]) is False
            for prev, cur in zip(seg[1:], seg[2:], strict=False):
                none_back = _got_results(prev, cur) is False
                r = _rise(prev, cur)
                if not (none_back or r > 0):
                    break  # a different result: polling or waiting, not a loop
                members.append(cur)
                missing = missing or none_back
                if r > 0:
                    rises.append(r)
            if len(members) >= R05_CALLS:
                out.append((members, rises, missing))
    return out


def _r05(turns: Sequence[LensTurn]) -> list[Finding]:
    out = []
    for members, rises, missing in _loops(turns):
        size = round(statistics.median(rises)) if rises else 0
        if rises and size <= R05_SHORT_RESULT:
            cause = "result_not_final"
        elif missing:
            cause = "harness_no_result"
        else:
            cause = "none_found"
        counts = {"calls": len(members), "result": size}
        tokens = _comp(members[1:]) + sum(rises)
        tool = members[0].tools_out[0][1]
        out.append(
            _finding("R-05", [t.n for t in members], counts, cause=cause, tokens=tokens, tool=tool)
        )
    return out


# -- R-06 ---------------------------------------------------------------------------


_Call = tuple[LensTurn, bool | None, int, str | None]


def _tool_calls(turns: Sequence[LensTurn]) -> dict[tuple[object, str], list[_Call]]:
    """Per (lane, tool hash), its calls in order with the result's status from
    the next turn of the lane: ``(turn, failed, result tokens, args8)``;
    ``failed`` None when the result cannot be matched to the call."""
    out: dict[tuple[object, str], list[_Call]] = {}
    for li, lane in enumerate(_lanes(turns)):
        for k, t in enumerate(lane):
            cs = _calls(t)
            nxt = lane[k + 1] if k + 1 < len(lane) else None
            res = nxt.results if nxt is not None else None
            for idx, e in enumerate(cs):
                failed: bool | None = None
                tokens = 0
                if res is not None:
                    match = None
                    if len(res) == len(cs) and res[idx][0] == e[0]:
                        match = res[idx]
                    elif len(cs) == 1 and len(res) == 1:
                        match = res[0]
                    if match is not None:
                        failed, tokens = match[2], match[1]
                args = e[2] if len(e) >= 3 else None
                out.setdefault((li, e[1]), []).append((t, failed, tokens, args))
    return out


def _runs(calls: Sequence[_Call]) -> list[list[_Call]]:
    """Runs of ``R06_RUN`` or more consecutive failed calls, each followed by
    another call of the tool."""
    runs: list[list[_Call]] = []
    cur: list[_Call] = []
    for i, c in enumerate(calls):
        if c[1] is True and i + 1 < len(calls):
            cur.append(c)
            continue
        if len(cur) >= R06_RUN:
            runs.append(cur)
        cur = []
    if len(cur) >= R06_RUN:
        runs.append(cur)
    return runs


def _r06(turns: Sequence[LensTurn]) -> list[Finding]:
    out = []
    for (_lane, h), calls in _tool_calls(turns).items():
        runs = _runs(calls)
        total = sum(len(r) for r in runs)
        if not runs or (len(runs) < R06_RUNS and total < R06_TOTAL):
            continue
        failed = [c for r in runs for c in r]
        if len({c[0].n for c in failed}) < MIN_BAD:
            continue  # parallel calls in one turn: still one bad turn (A-12)
        chars = round(statistics.median(c[2] for c in failed)) * RESULT_CHARS_PER_TOKEN
        same_args = all(len({c[3] for c in r}) == 1 and r[0][3] is not None for r in runs)
        if chars <= R06_SHORT_CHARS:
            cause = "error_too_short"
        elif same_args:
            cause = "same_args_retry"
        else:
            cause = "none_found"
        counts = {"failed": total, "runs": len(runs), "len": int(chars)}
        ts = [c[0] for c in failed]
        # The error flag is a heuristic on OpenAI-format results: medium at most.
        out.append(
            _finding(
                "R-06",
                [t.n for t in ts],
                counts,
                cause=cause,
                tokens=_comp(ts),
                tool=h,
                confidence="medium",
            )
        )
    return out


# -- R-07 / R-08 --------------------------------------------------------------------


def _r07(turns: Sequence[LensTurn]) -> list[Finding]:
    empty = [t for t in turns if _empty(t)]
    if len(empty) < MIN_BAD:
        return []
    prev = _prev_in_lane(turns)
    swallowed = sum(
        1
        for t in empty
        if (t.reasoning_tokens or 0) > 0 and t.finish_reason in ("stop", "end_turn")
    )
    after_tool = sum(1 for t in empty if _got_results(prev.get(t.n), t))
    if 2 * swallowed >= len(empty):
        cause = "template_swallowed"
    elif 2 * after_tool >= len(empty):
        cause = "no_after_tool_instruction"
    else:
        cause = "none_found"
    counts = {"empty": len(empty), "turns": len(turns)}
    return [_finding("R-07", [t.n for t in empty], counts, cause=cause, tokens=_comp(empty))]


def _r08(turns: Sequence[LensTurn]) -> list[Finding]:
    hits = [
        t
        for t in turns
        if (t.reasoning_tokens or 0) >= R08_REASONING
        and t.completion_tokens - (t.reasoning_tokens or 0) <= R08_ANSWER
        and not t.calls_n
        and not _calls(t)
    ]
    if len(hits) < R08_TURNS:
        return []
    counts = {
        "turns": len(hits),
        "reasoning": round(statistics.median(t.reasoning_tokens or 0 for t in hits)),
        # What a lower budget could save (spec 11.3); info, so not the saving.
        "excess": sum(max((t.reasoning_tokens or 0) - R08_FLOOR, 0) for t in hits),
    }
    f = _finding("R-08", [t.n for t in hits], counts, cause="thinking_for_short_step")
    return [_info(f)]


# -- R-09 ---------------------------------------------------------------------------


def _r09(turns: Sequence[LensTurn], where: Mapping[str, Mapping[str, int]]) -> list[Finding]:
    """Guard matches in the prompt. The role comes from the guard log
    (``where``: request id -> where code -> values) when the caller has it;
    without it, a match that arrived on a turn whose only new messages are tool
    results reads as the results' (medium), anything else as not known (low)."""
    prev = _prev_in_lane(turns)
    groups: dict[tuple[str, Confidence], list[tuple[LensTurn, int, set[str]]]] = {}
    for t in turns:
        if not t.guard:
            continue
        p = prev.get(t.n)
        first = p is None or not p.guard
        vals = sum(c if first else n for _r, c, n in t.guard)
        if vals <= 0:
            continue
        rules = {r for r, c, n in t.guard if (c if first else n) > 0}
        w = where.get(t.req_id)
        cause: str
        conf: Confidence
        if w:
            conf = "high"
            if w.get("tool_result", 0) > 0:
                cause = "secret_in_tool_results"
            elif w.get("system", 0) > 0:
                cause = "secret_in_system"
            else:
                cause = "none_found"
        else:
            u_a, u_b = (_msgs(p, "user") if p else None), _msgs(t, "user")
            only_results = (
                p is not None
                and _got_results(p, t) is True
                and u_a is not None
                and u_b is not None
                and u_b <= u_a
            )
            cause, conf = (
                ("secret_in_tool_results", "medium") if only_results else ("none_found", "low")
            )
        groups.setdefault((cause, conf), []).append((t, vals, rules))
    out = []
    for (cause, conf), hs in groups.items():
        counts = {
            "values": sum(v for _, v, _ in hs),
            "rules": len(set().union(*(r for _, _, r in hs))),
        }
        out.append(_finding("R-09", [t.n for t, _, _ in hs], counts, cause=cause, confidence=conf))
    return out


# -- R-10 ---------------------------------------------------------------------------


def _r10(turns: Sequence[LensTurn], rate: float | None) -> list[Finding]:
    rate = rate if rate and rate > 0 else None
    rows = []
    for t in turns:
        if t.ttft_s is None or t.ttft_s <= 0:
            continue
        reused = t.reused_own if t.reused_own is not None else (t.cached_tokens or 0)
        if t.prompt_tokens - max(reused, 0) < R10_NEW_TOKENS:
            continue
        queue = min(max(t.queued_s or 0.0, 0.0), t.ttft_s)
        lost = 0 if t.cause in _SERVER_LOSS else max(t.lost_tokens or 0, 0)
        rec = min(lost / rate, t.ttft_s - queue) if rate else 0.0
        rows.append((t, t.ttft_s, queue, rec, lost))
    if len(rows) < MIN_BAD:
        return []
    med = statistics.median(r[1] for r in rows)
    if med < R10_TTFT_S:
        return []
    total = sum(r[1] for r in rows)
    queue = sum(r[2] for r in rows)
    rec = sum(r[3] for r in rows)
    counts = {
        "turns": len(rows),
        "ttft_ms": round(1000 * med),
        "recompute_ms": round(1000 * statistics.median(r[3] for r in rows)),
        "queue_ms": round(1000 * statistics.median(r[2] for r in rows)),
    }
    ns = [r[0].n for r in rows]
    if queue / total > R10_SHARE:
        # Server load, not the prompt: no prompt advice (A-11).
        return [_finding("R-10", ns, counts, cause="server_load")]
    if rate and rec / total > R10_SHARE:
        tokens = sum(r[4] for r in rows)
        return [
            _finding("R-10", ns, counts, cause="recompute_dominates", tokens=tokens, prefill_s=rec)
        ]
    conf: Confidence = "medium" if rate else "low"
    return [_finding("R-10", ns, counts, cause="none_found", confidence=conf)]


# -- public -------------------------------------------------------------------------


def response_findings(
    turns: Sequence[LensTurn],
    *,
    lint: Mapping[str, int],
    family_sizes: Mapping[str, int] | None = None,
    guard_where: Mapping[str, Mapping[str, int]] | None = None,
    prefill_rate: float | None = None,
) -> list[Finding]:
    """The session's R findings, in catalogue order (largest saving first
    within a rule). ``lint`` is tool hash -> ``schema_lint`` flags;
    ``family_sizes`` the offered tools per family (for "similar tools");
    ``guard_where`` the guard log's request id -> where code -> values
    (``guard_where_of``); ``prefill_rate`` (tokens/s) splits TTFT."""
    found = (
        _r01(turns)
        + _r02(turns)
        + _r03(turns)
        + _r04(turns, lint, family_sizes or {})
        + _r05(turns)
        + _r06(turns)
        + _r07(turns)
        + _r08(turns)
        + _r09(turns, guard_where or {})
        + _r10(turns, prefill_rate)
    )
    order = {rid: i for i, rid in enumerate(RULES)}
    return sorted(found, key=lambda f: (order[f.rule], -f.saving.tokens))


def bad_turns(turns: Sequence[LensTurn]) -> dict[str, tuple[int, ...]]:
    """Per scored R rule, the turn numbers R counts as bad (spec 10.2), per
    turn and regardless of a finding's threshold: a client or window cut (not a
    probe), a runaway cut, a malformed call (not leaked markup, an engine
    cause: A-11), the 2nd+ call of a loop, the 2nd+ consecutive failed call of
    a tool, an empty reply. R-08, R-09 and R-10 are never in R. A turn can be
    bad under two rules; the score counts it once."""
    loops = sorted({t.n for members, _r, _m in _loops(turns) for t in members[1:]})
    retries = sorted(
        {
            c[0].n
            for calls in _tool_calls(turns).values()
            for r in _runs(calls)
            for c in r[1:]
            if c[0].n != r[0][0].n  # a parallel call in the run's first turn is no retry
        }
    )
    return {
        "R-01": tuple(t.n for t in turns if _cut(t) == "client"),
        "R-02": tuple(t.n for t in turns if _cut(t) == "window"),
        "R-03": tuple(t.n for t in turns if _runaway(t)),
        "R-04": tuple(
            t.n
            for t in turns
            if t.calls_bad is not None and any(c != "leaked_markup" for _h, c in _bad_verdicts(t))
        ),
        "R-05": tuple(loops),
        "R-06": tuple(retries),
        "R-07": tuple(t.n for t in turns if _empty(t)),
    }
