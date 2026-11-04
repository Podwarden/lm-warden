"""Dead weight, prompt anatomy and the three time figures of the sessions pages.
Pure: plain numbers in, so store rows of either kind adapt to it. Every figure is
an estimate except the measured cache facts; unknown stays None, never 0."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from typing import Any, Literal

from app.sessions.store import ModelEngine

DW_KINDS = ("unused_tools", "dup", "boil", "reason")


def tokens_per_char(prompt_tokens: int, chars: int | None) -> float:
    if isinstance(chars, int | float) and chars > 0:
        return prompt_tokens / float(chars)
    return 0.25


@dataclass(frozen=True)
class DeadWeight:
    prompt_tokens: int
    unused_tools: float | None
    dup: float | None
    boil: float | None
    reason: float | None
    #: The turn carries a dw measure (dup, boil, reason); a row recorded before
    #: dead weight was tracked has its unused tools only, and is not measured.
    measured: bool = False

    @functools.cached_property
    def total(self) -> float | None:
        known = [v for v in (self.unused_tools, self.dup, self.boil, self.reason) if v is not None]
        if not known:
            return None
        return min(sum(known), float(self.prompt_tokens))


def turn_dead_weight(
    *,
    prompt_tokens: int,
    chars: int | None,
    dw: Mapping[str, int] | tuple[int, int, int] | None,
    unused_tool_chars: int | None,
) -> DeadWeight:
    r = tokens_per_char(prompt_tokens, chars)
    if dw is None:
        dup = boil = reason = None
    else:
        if isinstance(dw, Mapping):
            d, b, re_ = dw.get("dup", 0), dw.get("boil", 0), dw.get("reason", 0)
        else:
            d, b, re_ = dw
        dup, boil, reason = d * r, b * r, re_ * r
    unused = unused_tool_chars * r if unused_tool_chars is not None else None
    return DeadWeight(prompt_tokens, unused, dup, boil, reason, dw is not None)


def dead_weight_turn_json(d: DeadWeight) -> dict[str, Any] | None:
    total = d.total
    if total is None or not d.measured:
        return None

    def rnd(v: float | None) -> int | None:
        return None if v is None else round(v)

    return {
        "unused_tools": rnd(d.unused_tools),
        "dup": rnd(d.dup),
        "boil": rnd(d.boil),
        "reason": rnd(d.reason),
        "total": round(total),
        "share": (total / d.prompt_tokens) if d.prompt_tokens else None,
    }


def dead_weight_json(parts: Sequence[DeadWeight]) -> dict[str, Any]:
    """A session's or conversation's dead weight over the turns where it was
    measured ("measured on N of M turns"); each kind over the turns that know it."""
    measured = [p for p in parts if p.measured and p.total is not None]
    kinds: dict[str, dict[str, float | None]] = {}
    for k in DW_KINDS:
        known = [(getattr(p, k), p.prompt_tokens) for p in parts if getattr(p, k) is not None]
        if not known:
            kinds[k] = {"tokens": None, "share": None, "per_turn": None}
            continue
        tok = sum(v for v, _ in known)
        prompt = sum(pt for _, pt in known)
        kinds[k] = {
            "tokens": round(tok),
            "share": (tok / prompt) if prompt else None,
            "per_turn": tok / len(known),
        }
    total = sum(p.total or 0.0 for p in measured) if measured else None
    prompt = sum(p.prompt_tokens for p in measured)
    return {
        "prompt_tokens": prompt,
        "total_tokens": round(total) if total is not None else None,
        "share": (total / prompt) if total is not None and prompt else None,
        "per_turn": (total / len(measured)) if total is not None else None,
        "turns": len(parts),
        "turns_measured": len(measured),
        "estimated": True,
        "kinds": kinds,
    }


@dataclass(frozen=True)
class Anatomy:
    cached: float
    dead: float
    recomputed: float
    new: float
    unknown: float


def anatomy_of(
    *, prompt: int, cached: int | None, prev_prompt: int | None, dead: float | None
) -> Anatomy:
    """What a prompt was made of: dead weight first, the rest split into from
    cache, recomputed (the previous prompt the cache no longer had) and new."""
    p = float(prompt)
    d = min(float(dead or 0.0), p)
    if cached is None:
        return Anatomy(0.0, d, 0.0, 0.0, p - d)
    c = float(min(cached, prompt))
    n = (p - c) if prev_prompt is None else min(max(0.0, p - prev_prompt), p - c)
    r = p - c - n
    k = (1 - d / p) if p else 0.0
    return Anatomy(c * k, d, r * k, n * k, 0.0)


def add_anatomy(a: Anatomy, b: Anatomy) -> Anatomy:
    # Field by field: astuple deep-copies, and the list scan adds one per row.
    return Anatomy(
        a.cached + b.cached,
        a.dead + b.dead,
        a.recomputed + b.recomputed,
        a.new + b.new,
        a.unknown + b.unknown,
    )


def cache_counts(turns: Iterable[tuple[int, int | None]]) -> tuple[int, int]:
    """(cached, prompt) summed over the turns whose cache is known: the share
    "served from cache", before dead weight is taken out of the anatomy."""
    c = p = 0
    for prompt, cached in turns:
        if cached is not None:
            c += min(cached, prompt)
            p += prompt
    return c, p


def share_of(ck: tuple[int, int]) -> float | None:
    return (ck[0] / ck[1]) if ck[1] else None


def anatomy_json(a: Anatomy) -> dict[str, int]:
    """Rounded fields that still sum to the rounded total."""
    names = [f.name for f in fields(Anatomy)]
    vals = astuple(a)
    out = {n: round(v) for n, v in zip(names, vals, strict=True)}
    diff = round(sum(vals)) - sum(out.values())
    if diff:
        big = max(names, key=lambda n: getattr(a, n))
        out[big] += diff
    return out


def served_s(cached: int | None, rate: float | None) -> float | None:
    if cached is None or not rate:
        return None
    return cached / rate


def recomputed_s(prompt: int, cached: int | None, rate: float | None) -> float | None:
    if cached is None or not rate:
        return None
    return max(0, prompt - cached) / rate


def saved_vs_plain_s(
    cached: int | None, engine: ModelEngine | None, rate: float | None
) -> float | None:
    """Prefill LM Warden's placement spared vs plain vLLM's load-based routing
    (R4): with N replicas, a turn lands on its prefix's replica 1 time in N."""
    if cached is None or engine is None or not rate:
        return None
    if engine.dp_size <= 1 or not engine.dp_affinity:
        return 0.0
    return cached * (engine.dp_size - 1) / engine.dp_size / rate


def where_of(local: int, relayed: int) -> Literal["local", "relayed", "hybrid"]:
    if local and relayed:
        return "hybrid"
    return "relayed" if relayed else "local"
