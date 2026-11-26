"""leanprompt's measured score, from a key's stored request rows (spec 14.1).

``leanprompt score = floor(100 x (0.6 x prefill_efficiency x stability + 0.4 x (1 - generation_waste)))``
(leanprompt ``docs/DESIGN.md`` section 8), measured, and only with at least 50 calls:

* ``prefill_efficiency`` = prompt tokens served from cache / reusable prompt tokens, over the
  calls that continue an earlier one and have both figures (the key's own lens);
* ``stability`` = reusable prompt tokens / prompt tokens, over the calls that continue an earlier one;
* ``generation_waste`` = output tokens of calls cut at the token limit or that carried a
  malformed tool call (discarded or retried) / all output tokens.

The inputs are counts and sums of stored numbers; no text. Below 50 calls the answer is
``{"scored": false, ...}`` with no number (A-7).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

MIN_CALLS = 50
SCORE_KIND = "measured"
WEIGHT_PREFILL = 0.6
WEIGHT_GENERATION = 0.4


@dataclass(frozen=True)
class ScoreRow:
    prompt_tokens: int
    cached_tokens: int | None
    completion_tokens: int
    #: The key's own reusable prefix (``reusable_tokens_own``, else ``reusable_tokens``).
    reusable_tokens: int | None
    continues: bool
    finish_reason: str | None
    bad_call: bool


def _ratio(num: int, den: int) -> float | None:
    return None if den <= 0 else min(1.0, max(0.0, num / den))


def leanprompt_score(rows: Sequence[ScoreRow]) -> dict[str, object]:
    calls = len(rows)
    if calls < MIN_CALLS:
        return {
            "scored": False,
            "reason": f"{calls} of {MIN_CALLS} calls needed for a measured score",
            "calls": calls,
        }
    cont = [r for r in rows if r.continues and r.reusable_tokens is not None]
    reusable = sum(min(r.reusable_tokens or 0, r.prompt_tokens) for r in cont)
    prompt = sum(r.prompt_tokens for r in cont)
    served = sum(
        min(r.cached_tokens or 0, min(r.reusable_tokens or 0, r.prompt_tokens))
        for r in cont
        if r.cached_tokens is not None
    )
    served_base = sum(
        min(r.reusable_tokens or 0, r.prompt_tokens) for r in cont if r.cached_tokens is not None
    )
    efficiency = _ratio(served, served_base)
    stability = _ratio(reusable, prompt)
    out_tokens = sum(max(r.completion_tokens, 0) for r in rows)
    wasted = sum(
        max(r.completion_tokens, 0) for r in rows if r.finish_reason == "length" or r.bad_call
    )
    waste = _ratio(wasted, out_tokens) or 0.0
    if efficiency is None or stability is None:
        return {
            "scored": False,
            "reason": "no continuing calls with cache figures in this window",
            "calls": calls,
        }
    value = math.floor(
        100 * (WEIGHT_PREFILL * efficiency * stability + WEIGHT_GENERATION * (1 - waste))
    )
    return {
        "scored": True,
        "kind": SCORE_KIND,
        "leanprompt": max(0, min(100, value)),
        "prefill_efficiency": round(efficiency, 4),
        "stability": round(stability, 4),
        "generation_waste": round(waste, 4),
        "calls": calls,
        "continuing_calls": len(cont),
    }
