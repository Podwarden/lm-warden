"""Prompt Lens scores: E, C, R and the overall (analysis spec 10).

Pure functions over a ``SessionAnalysis``. Every score is a share of something
measured (tokens or turns) and carries its inputs, so it can be recomputed from
what is shown beside it. Below a layer's minimum data there is no number: a
``Score`` with ``scored=False`` and a ``reason`` (A-7). Server causes never
count (A-11); a lineage gap leaves a turn out of the layer's sums and lowers its
coverage and confidence instead of moving the value (Review Focus 1). Nothing
here holds text (A-10): numbers, rule ids and codes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.lens.analyse import SessionAnalysis
from app.lens.catalogue import RULES, Layer
from app.lens.causes import APP_FIXABLE
from app.lens.turns import LensTurn

#: Bump on a change of the formulas or weights; stored with every row, and a
#: change is listed in the changelog (spec 10.2).
SCORE_VERSION = 1
WEIGHTS: Mapping[str, float] = {"E": 0.45, "C": 0.15, "R": 0.40}
LAYERS: tuple[Layer, ...] = ("E", "C", "R")

# Minimum data (spec 10.3).
MIN_E_TURNS = 3
MIN_E_IDEAL = 4_096
MIN_C_REQUESTS = 2
MIN_C_TOOL_TURNS = 5
MIN_R_TURNS = 5
MIN_R_CALL_TURNS = 3
MIN_OVERALL_LAYERS = 2

# Confidence (spec 10.4).
HIGH_TURNS, HIGH_COVERAGE = 20, 0.90
MEDIUM_TURNS, MEDIUM_COVERAGE = 8, 0.70
BOOTSTRAP_N = 200

#: The R rules that read tool calls (spec 10.3: "tool-call terms need 3 turns
#: with tool calls").
CALL_RULES = frozenset({"R-04", "R-05", "R-06"})
#: Every R rule whose bad turns R counts (R-08, R-09, R-10 never do).
R_RULES = ("R-01", "R-02", "R-03", "R-04", "R-05", "R-06", "R-07")
#: Boilerplate counts half (spec 10.2); mirrors ``construction.BOIL_WEIGHT``.
BOIL_WEIGHT = 0.5

#: The E rule a fixable cause belongs to (spec 11.1), so a rule can be excluded.
E_RULE_OF_CAUSE: Mapping[str, str] = {
    "sys_noise": "E-01",
    "sys_changed": "E-01",
    "tools_reordered": "E-02",
    "tools_key_order": "E-02",
    "tools_added_end": "E-03",
    "tools_changed": "E-03",
    "history_rewritten": "E-04",
    "history_noise": "E-04",
    "reasoning_dropped": "E-05",
    "reasoning_kept": "E-05",
    "provider_no_marks": "E-06",
    "compacted": "E-07",
    "media_changed": "E-09",
}
assert set(E_RULE_OF_CAUSE) == APP_FIXABLE

Confidence = Literal["high", "medium", "low"]
_CONF_ORDER: tuple[Confidence, ...] = ("low", "medium", "high")

#: One bootstrap observation: two numbers per turn (see each layer).
PerTurn = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class Terms:
    """What a layer's score is a share of. ``total`` and ``base`` per layer:
    E: Σ ideal (counterfactual reuse) tokens, ``by_rule`` the fixable lost tokens
    per E rule. C: Σ prompt tokens, ``by_rule`` the waste tokens per C rule.
    R: ``total`` the assistant turns, ``base`` Σ completion tokens, ``by_rule``
    ``"R-01:turns"`` (bad turns) and ``"R-01:out"`` (their completion tokens)
    per R rule (a turn bad under two rules counts once in the score, not here)."""

    layer: Layer
    total: float
    base: float
    by_rule: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "total": self.total,
            "base": self.base,
            "by_rule": dict(self.by_rule),
        }


@dataclass(frozen=True)
class Score:
    """One score, or the reason there is none (A-7). ``inputs`` are the numbers
    the value is computed from; ``excluded_rules`` the rules left out of it
    (won't fix, or a term below its minimum)."""

    layer: Layer | Literal["overall"]
    scored: bool
    value: int | None
    lo: int | None
    hi: int | None
    confidence: Confidence | None
    turns: int
    reason: str | None
    inputs: dict[str, Any] = field(default_factory=dict)
    excluded_rules: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """No ``value`` key at all when not scored (A-7)."""
        if not self.scored:
            return {
                "layer": self.layer,
                "scored": False,
                "reason": self.reason,
                "turns": self.turns,
            }
        return {
            "layer": self.layer,
            "scored": True,
            "value": self.value,
            "lo": self.lo,
            "hi": self.hi,
            "confidence": self.confidence,
            "turns": self.turns,
            "inputs": dict(self.inputs),
            "excluded_rules": list(self.excluded_rules),
        }


def _not_scored(layer: Layer | Literal["overall"], reason: str, turns: int) -> Score:
    return Score(layer, False, None, None, None, None, turns, reason)


# -- the turns each layer reads ---------------------------------------------------


def e_turns(a: SessionAnalysis) -> list[LensTurn]:
    """Continuing turns whose counterfactual and loss are known. A fork is the
    user's own branch; a turn with no parent in memory (a lineage gap) or no
    counterfactual is left out, which lowers coverage, not the value."""
    return [
        t
        for t in a.turns
        if t.edge != "fork" and t.cf_tokens is not None and t.lost_tokens is not None
    ]


def c_turns(a: SessionAnalysis) -> list[LensTurn]:
    return [t for t in a.turns if t.shape is not None]


def r_turns(a: SessionAnalysis) -> list[LensTurn]:
    """Assistant turns: requests whose reply fields were stored."""
    return [
        t
        for t in a.turns
        if t.reply_chars is not None
        and t.calls_n is not None
        and t.calls_bad is not None
        and t.reasoning_tokens is not None
    ]


def _fixable(t: LensTurn) -> str | None:
    """The E rule ``t``'s lost tokens count under, or None (not fixable)."""
    if t.cause not in APP_FIXABLE or (t.lost_tokens or 0) <= 0:
        return None
    if t.cause == "compacted" and t.brk not in ("sys", "tools"):
        return None
    return E_RULE_OF_CAUSE[t.cause]


def _predicted(turns: Sequence[LensTurn]) -> bool:
    """A cache figure that is a prediction: a relayed turn without upstream usage."""
    return any(t.relayed and t.cached_tokens is None for t in turns)


def _c_waste_by_turn(a: SessionAnalysis, turns: Sequence[LensTurn], keep: set[str]) -> list[float]:
    """Waste tokens per C turn, for the range only: dup, repeated results and
    half of boilerplate per turn; the unused-tools term (a session figure)
    spread over the turns that offered tools."""
    carrying = [t for t in turns if (t.tools_in or 0) > 0]
    spread = a.waste.get("C-01", 0) / len(carrying) if carrying and "C-01" in keep else 0.0
    out = []
    for t in turns:
        dw = t.dead_weight
        w = 0.0
        if "C-02" in keep:
            w += (dw[0] if dw is not None else 0) + max(t.repeat_results or 0, 0)
        if "C-07" in keep and dw is not None:
            w += BOIL_WEIGHT * dw[1]
        if "C-01" in keep and (t.tools_in or 0) > 0:
            w += spread
        out.append(w)
    return out


def terms_of(a: SessionAnalysis) -> dict[Layer, Terms]:
    """The three layers' terms, before any minimum or exclusion."""
    et = e_turns(a)
    e_by: dict[str, float] = {}
    for t in et:
        rule = _fixable(t)
        if rule is not None:
            e_by[rule] = e_by.get(rule, 0.0) + (t.lost_tokens or 0)
    ct = c_turns(a)
    rt = r_turns(a)
    r_by: dict[str, float] = {}
    scored = {t.n: t for t in rt}
    for rule in R_RULES:
        ns = [n for n in a.bad.get(rule, ()) if n in scored]
        if ns:
            r_by[f"{rule}:turns"] = float(len(ns))
            r_by[f"{rule}:out"] = float(sum(max(scored[n].completion_tokens, 0) for n in ns))
    return {
        "E": Terms("E", float(sum(t.cf_tokens or 0 for t in et)), 0.0, e_by),
        "C": Terms(
            "C",
            float(sum(max(t.prompt_tokens, 0) for t in ct)),
            0.0,
            {k: float(v) for k, v in a.waste.items() if v},
        ),
        "R": Terms("R", float(len(rt)), float(sum(max(t.completion_tokens, 0) for t in rt)), r_by),
    }


# -- confidence and range -----------------------------------------------------------


def _confidence(turns: int, coverage: float, measured: bool) -> Confidence:
    if turns >= HIGH_TURNS and coverage >= HIGH_COVERAGE and measured:
        return "high"
    if turns >= MEDIUM_TURNS and coverage >= MEDIUM_COVERAGE:
        return "medium"
    return "low"


def _lowest(labels: Sequence[Confidence]) -> Confidence:
    return min(labels, key=_CONF_ORDER.index)


def bootstrap_range(
    per_turn: PerTurn,
    score: Callable[[Sequence[tuple[float, float]]], float],
    seed: str,
    n: int = BOOTSTRAP_N,
) -> tuple[int, int]:
    """10th to 90th percentile of ``score`` over ``n`` resamples of the turns
    (with replacement), seeded so a reload shows the same range."""
    if not per_turn:
        return 0, 0
    rng = random.Random(seed)
    m = len(per_turn)
    vals = sorted(
        min(100.0, max(0.0, score([per_turn[rng.randrange(m)] for _ in range(m)])))
        for _ in range(n)
    )
    lo = vals[round(0.10 * (len(vals) - 1))]
    hi = vals[round(0.90 * (len(vals) - 1))]
    return math.floor(lo), math.ceil(hi)


def _share_score(sample: Sequence[tuple[float, float]]) -> float:
    """E and C: (numerator, denominator) per turn."""
    den = sum(d for _n, d in sample)
    return 100.0 if den <= 0 else 100.0 * (1 - sum(x for x, _d in sample) / den)


def _c_score(sample: Sequence[tuple[float, float]]) -> float:
    den = sum(d for _n, d in sample)
    return 100.0 if den <= 0 else 100.0 * max(0.0, 1 - 2 * sum(x for x, _d in sample) / den)


def _r_score(sample: Sequence[tuple[float, float]]) -> float:
    """R: (bad flag, completion tokens) per turn."""
    if not sample:
        return 100.0
    comp = sum(c for _b, c in sample)
    out = sum(c for b, c in sample if b)
    token_part = 0.5 * out / comp if comp > 0 else 0.0
    return 100.0 * (1 - 0.5 * sum(b for b, _c in sample) / len(sample) - token_part)


def _finish(
    layer: Layer,
    value: float,
    per_turn: PerTurn,
    fn: Callable[[Sequence[tuple[float, float]]], float],
    *,
    turns: int,
    coverage: float,
    measured: bool,
    seed: str,
    inputs: dict[str, Any],
    excluded: set[str],
) -> Score:
    v = min(100, max(0, round(value)))
    lo, hi = bootstrap_range(per_turn, fn, f"{seed}:{layer}")
    return Score(
        layer,
        True,
        v,
        min(lo, v),
        max(hi, v),
        _confidence(turns, coverage, measured),
        turns,
        None,
        {**inputs, "coverage": round(coverage, 3)},
        tuple(sorted(excluded)),
    )


def _score_e(terms: Terms, a: SessionAnalysis, seed: str, exclude: frozenset[str]) -> Score:
    et = e_turns(a)
    ideal = terms.total
    if len(et) < MIN_E_TURNS:
        return _not_scored("E", f"{len(et)} of {MIN_E_TURNS} continuing turns", len(et))
    if ideal < MIN_E_IDEAL:
        return _not_scored("E", f"{round(ideal):,} of {MIN_E_IDEAL:,} reusable tokens", len(et))
    lost = sum(v for rule, v in terms.by_rule.items() if rule not in exclude)
    pairs = [
        (
            float(t.lost_tokens or 0)
            if (r := _fixable(t)) is not None and r not in exclude
            else 0.0,
            float(t.cf_tokens or 0),
        )
        for t in et
    ]
    return _finish(
        "E",
        100.0 * (1 - lost / ideal),
        pairs,
        _share_score,
        turns=len(et),
        coverage=a.coverage["E"],
        measured=not _predicted(et),  # a predicted cache figure caps E at medium
        seed=seed,
        inputs={"ideal": round(ideal), "lost": round(lost)},
        excluded=set(exclude) & _E_IDS,
    )


_E_IDS = frozenset(r for r in RULES if r.startswith("E-"))


def _score_c(terms: Terms, a: SessionAnalysis, seed: str, exclude: frozenset[str]) -> Score:
    ct = c_turns(a)
    if len(ct) < MIN_C_REQUESTS:
        return _not_scored("C", f"{len(ct)} of {MIN_C_REQUESTS} requests", len(ct))
    prompt = terms.total
    if prompt <= 0:
        return _not_scored("C", "no prompt tokens measured", len(ct))
    dropped = set(exclude)
    carrying = sum(1 for t in ct if (t.tools_in or 0) > 0)
    if carrying < MIN_C_TOOL_TURNS:
        dropped.add("C-01")  # the unused-tools term needs 5 turns that offered tools
    waste = sum(v for rule, v in terms.by_rule.items() if rule not in dropped)
    keep = {"C-01", "C-02", "C-07"} - dropped
    per = _c_waste_by_turn(a, ct, keep)
    return _finish(
        "C",
        100.0 * max(0.0, 1 - 2 * waste / prompt),
        [(w, float(max(t.prompt_tokens, 0))) for w, t in zip(per, ct, strict=True)],
        _c_score,
        turns=len(ct),
        coverage=a.coverage["C"],
        measured=True,
        seed=seed,
        inputs={"waste": round(waste), "prompt": round(prompt)},
        excluded={r for r in dropped if r.startswith("C-")},
    )


def _score_r(terms: Terms, a: SessionAnalysis, seed: str, exclude: frozenset[str]) -> Score:
    rt = r_turns(a)
    if len(rt) < MIN_R_TURNS:
        return _not_scored("R", f"{len(rt)} of {MIN_R_TURNS} assistant turns", len(rt))
    dropped = set(exclude)
    if sum(1 for t in rt if (t.calls_n or 0) > 0) < MIN_R_CALL_TURNS:
        dropped |= CALL_RULES  # tool-call terms need 3 turns with tool calls
    in_r = {t.n for t in rt}
    bad: set[int] = set()
    for rule in R_RULES:
        if rule not in dropped:
            bad |= {n for n in a.bad.get(rule, ()) if n in in_r}
    comp = terms.base
    waste_out = sum(max(t.completion_tokens, 0) for t in rt if t.n in bad)
    token_part = 0.5 * waste_out / comp if comp > 0 else 0.0
    value = 100.0 * (1 - 0.5 * len(bad) / len(rt) - token_part)
    return _finish(
        "R",
        value,
        [(1.0 if t.n in bad else 0.0, float(max(t.completion_tokens, 0))) for t in rt],
        _r_score,
        turns=len(rt),
        coverage=a.coverage["R"],
        measured=True,
        seed=seed,
        inputs={
            "turns": len(rt),
            "bad": len(bad),
            "completion": round(comp),
            "waste_out": round(waste_out),
        },
        excluded={r for r in dropped if r.startswith("R-")},
    )


def score_layer(
    layer: Layer,
    terms: Terms,
    a: SessionAnalysis,
    *,
    seed: str,
    exclude: frozenset[str] = frozenset(),
) -> Score:
    """One layer's score from its ``terms`` (``terms_of(a)[layer]``) and the
    analysis (turns, coverage), with the rules in ``exclude`` left out (won't
    fix, spec 13.3). ``seed`` (the session ref) fixes the bootstrap range."""
    if layer == "E":
        return _score_e(terms, a, seed, exclude)
    if layer == "C":
        return _score_c(terms, a, seed, exclude)
    return _score_r(terms, a, seed, exclude)


def score_session(
    a: SessionAnalysis, *, seed: str, exclude: frozenset[str] = frozenset()
) -> dict[str, Score]:
    """E, C, R and the overall of one session."""
    terms = terms_of(a)
    layers: dict[Layer, Score] = {
        k: score_layer(k, terms[k], a, seed=seed, exclude=exclude) for k in LAYERS
    }
    out: dict[str, Score] = {k: v for k, v in layers.items()}
    out["overall"] = overall(layers)
    return out


def overall(scores: Mapping[Layer, Score]) -> Score:
    """The weighted mean over the scored layers, weights shared out in
    proportion; fewer than two scored layers: no overall (spec 10.2)."""
    got = [k for k in LAYERS if k in scores and scores[k].scored]
    if len(got) < MIN_OVERALL_LAYERS:
        return _not_scored("overall", f"{len(got)} of {MIN_OVERALL_LAYERS} layers scored", 0)
    wsum = sum(WEIGHTS[k] for k in got)

    def mix(pick: Callable[[Score], int | None]) -> int:
        return round(sum(WEIGHTS[k] * (pick(scores[k]) or 0) for k in got) / wsum)

    return Score(
        "overall",
        True,
        mix(lambda s: s.value),
        mix(lambda s: s.lo),
        mix(lambda s: s.hi),
        _lowest([c for k in got if (c := scores[k].confidence) is not None]),
        max(scores[k].turns for k in got),
        None,
        {"from": list(got)},
        tuple(sorted({r for k in got for r in scores[k].excluded_rules})),
    )
