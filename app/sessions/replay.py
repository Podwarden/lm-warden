"""Replay a session's local turns through each engine's cache rules (R6): this
session alone, no other traffic, the lane's previous prompt taken as fully
reusable. An estimate of prefill recomputed, never a measurement (except LM
Warden's own row); oMLX disk load time is not included."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.sessions.store import ModelEngine

ENGINES = ("lm_warden", "vllm", "llamacpp", "mlx_lm", "omlx")


@dataclass(frozen=True)
class ReplayTurn:
    lane: str | None
    started_at: float
    prompt_tokens: int
    cached_tokens: int | None
    rate: float | None
    engine: ModelEngine | None


def replay(turns: Sequence[ReplayTurn], *, slots: int) -> dict[str, Any] | None:
    if not turns:
        return None
    tok: dict[str, float] = dict.fromkeys(ENGINES, 0.0)
    sec: dict[str, float] = dict.fromkeys(ENGINES, 0.0)
    measured_turns = 0
    skipped = 0
    last_prompt: dict[Any, int] = {}
    lru: OrderedDict[Any, None] = OrderedDict()
    prev_lane: Any = object()
    for i, t in enumerate(sorted(turns, key=lambda x: x.started_at)):
        lane: Any = t.lane if t.lane is not None else ("solo", i)
        p = t.prompt_tokens
        prev = last_prompt.get(lane)
        prefix = min(prev, p) if prev is not None else 0
        costs: dict[str, float | None] = {
            "mlx_lm": float(p - prefix) if prev_lane == lane else float(p),
            "llamacpp": float(p - prefix) if lane in lru else float(p),
            "omlx": float(p - prefix),
            "lm_warden": None,
            "vllm": None,
        }
        if t.cached_tokens is not None:
            c = min(t.cached_tokens, p)
            costs["lm_warden"] = float(p - c)
            eng = t.engine
            if eng is not None and eng.dp_size > 1 and eng.dp_affinity:
                costs["vllm"] = (p - c) + c * (eng.dp_size - 1) / eng.dp_size
            else:
                costs["vllm"] = float(p - c)
            measured_turns += 1
        if not t.rate:
            skipped += 1
        for e, v in costs.items():
            if v is None:
                continue
            tok[e] += v
            if t.rate:
                sec[e] += v / t.rate
        last_prompt[lane] = p
        lru[lane] = None
        lru.move_to_end(lane)
        while len(lru) > max(1, slots):
            lru.popitem(last=False)
        prev_lane = lane
    engines_seen = Counter(t.engine.dp_size for t in turns if t.engine is not None)
    out = []
    for e in ENGINES:
        unknown = e in ("lm_warden", "vllm") and measured_turns == 0
        out.append(
            {
                "engine": e,
                "recompute_tokens": None if unknown else round(tok[e]),
                "recompute_s": None if unknown or skipped == len(turns) else sec[e],
                "measured": e == "lm_warden",
            }
        )
    return {
        "estimated": True,
        "scope": "session",
        "turns": len(turns),
        "skipped_time": skipped,
        "slots": slots,
        "replicas": engines_seen.most_common(1)[0][0] if engines_seen else None,
        "disk_load_excluded": True,
        "engines": out,
    }
