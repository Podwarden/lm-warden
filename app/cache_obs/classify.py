# app/cache_obs/classify.py
"""The outcome of one request's cache use. Ordered rules, first match wins
(spec §3.3). Thresholds live here and nowhere else."""

from __future__ import annotations

MIN_REUSABLE = 256
HIT = 0.9
LOST = 0.1
OUTCOMES = ("hit", "partial", "lost", "misrouted", "diverged", "cold")


def bytes_to_tokens(matched: int, total: int, prompt: int) -> int:
    if total <= 0 or matched <= 0 or prompt <= 0:
        return 0
    return round(min(matched, total) / total * prompt)


def classify(
    *,
    cached: int | None,
    reusable: int | None,
    reusable_fleet: int | None,
    diverged_at: int | None,
    allow_misrouted: bool = True,
    cached_is_estimate: bool = False,
) -> str | None:
    if cached is None or reusable is None:
        return None
    r = reusable
    # Rule 1: the engine served more than the index knew of, so R was at least
    # C. Only a MEASURED C proves that; a TTFT estimate on a cold request would
    # otherwise promote itself to a hit.
    if cached > r and not cached_is_estimate:
        r = cached
    if r < MIN_REUSABLE:
        if allow_misrouted and (reusable_fleet or 0) >= MIN_REUSABLE:
            return "misrouted"
        if diverged_at is not None:
            return "diverged"
        return "cold"
    if cached >= HIT * r:
        return "hit"
    if cached < LOST * r:
        return "lost"
    return "partial"
