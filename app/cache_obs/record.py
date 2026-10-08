# app/cache_obs/record.py
"""One finished request's cache fields, in history-column names (spec §3)."""

from __future__ import annotations

from typing import Any

from app.cache_obs.classify import bytes_to_tokens, classify
from app.cache_obs.index import Match

#: an estimate under this share of the prompt is prefill-speed noise, read as 0
EST_NOISE_FLOOR = 0.15


def ttft_estimate(prompt: int, ttft_s: float | None, cold_rate: float | None) -> int | None:
    """Tokens served from cache, inferred from prefill speed. None without a
    LEARNED cold rate: a hint rate would turn a guess into a confident hit.
    Below EST_NOISE_FLOOR of the prompt it is 0: TTFT jitter, not cache."""
    if ttft_s is None or cold_rate is None or cold_rate <= 0 or prompt <= 0:
        return None
    est = int(max(0.0, min(float(prompt), prompt - ttft_s * cold_rate)))
    return 0 if est < EST_NOISE_FLOOR * prompt else est


def _tokens(
    match: Match | None, prompt: int, rank: int | None, rank_known: bool
) -> tuple[int | None, int | None]:
    if match is None:
        return None, None
    best = match.best_bytes
    if rank is not None:
        # A routed request: the None bit holds unrouted requests that could
        # have landed on any rank, so it is no evidence of "another replica".
        best = max((b for r, b in match.by_rank.items() if r is not None), default=0)
    fleet = bytes_to_tokens(best, match.total_bytes, prompt)
    if not rank_known:
        return fleet, fleet
    return bytes_to_tokens(match.by_rank.get(rank, 0), match.total_bytes, prompt), fleet


def observe_fields(
    *,
    prompt: int,
    cached: int | None,
    reporting: str,
    ttft_s: float | None,
    cold_rate: float | None,
    match_all: Match | None,
    match_own: Match | None,
    rank: int | None,
    rank_known: bool,
) -> dict[str, Any]:
    est = None
    if reporting == "none":
        est = ttft_estimate(prompt, ttft_s, cold_rate)
        fact, source = est, ("estimated" if est is not None else None)
    else:
        fact, source = cached, ("engine" if cached is not None else None)
    r, fleet = _tokens(match_all, prompt, rank, rank_known)
    r_own, _ = _tokens(match_own, prompt, rank, rank_known)
    return {
        "reusable_tokens": r,
        "reusable_tokens_fleet": fleet,
        "reusable_tokens_own": r_own,
        "cache_outcome": classify(
            cached=fact,
            reusable=r,
            reusable_fleet=fleet if rank_known else None,
            diverged_at=match_all.diverged_at if match_all else None,
            cached_is_estimate=source == "estimated",
        ),
        "cache_outcome_own": classify(
            cached=fact,
            reusable=r_own,
            reusable_fleet=None,
            diverged_at=match_own.diverged_at if match_own else None,
            allow_misrouted=False,
            cached_is_estimate=source == "estimated",
        ),
        "diverged_at": match_all.diverged_at if match_all else None,
        "diverged_at_own": match_own.diverged_at if match_own else None,
        "cached_source": source,
        "cached_ttft_est_tokens": est,
    }
