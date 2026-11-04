# app/cache_obs/classify.py
"""The outcome of one request's cache use. Ordered rules, first match wins
(spec §3.3). Thresholds live here and nowhere else."""

from __future__ import annotations

MIN_REUSABLE = 256
HIT = 0.9
LOST = 0.1
#: Order is the stats/UI display order; cobs2 miss causes (plan 3 task 6) are
#: appended so nothing that walks the older six moves.
OUTCOMES = (
    "hit",
    "partial",
    "lost",
    "misrouted",
    "diverged",
    "cold",
    "compacted",
    "harness_rewrite",
)
#: The misses a cobs2 cause may relabel: nothing (or a broken prefix) was
#: reusable. Never "lost": "reusable" ends where the rewrite or compaction
#: begins, so a lost request still had a 256+ token head and the engine
#: evicted it; relabelling it would hide real eviction. A hit, partial or
#: misrouted is never relabelled either; a hit or partial's cause goes in
#: divergence_cause, and misrouted gets none.
_COBS2_EXPLAINS = frozenset(("diverged", "cold"))
#: Values of the cobs2_divergence_cause columns (migration 0052).
DIVERGENCE_CAUSES = ("compacted", "harness_rewrite")
#: KV blocks up to this size are the classic 16-token attention pages: their
#: quantisation is noise beside MIN_REUSABLE, and the rules stay as they were.
#: Above it (784 on hybrid Qwen3.5, #301) the engine serves whole blocks only.
BLOCK_AWARE_ABOVE = 16


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
    block_size: int | None = None,
    cobs2_edge: str | None = None,
    loose_match_beyond_exact: bool = False,
) -> str | None:
    out = _classify(
        cached=cached,
        reusable=reusable,
        reusable_fleet=reusable_fleet,
        diverged_at=diverged_at,
        allow_misrouted=allow_misrouted,
        cached_is_estimate=cached_is_estimate,
        block_size=block_size,
    )
    # cobs2 (plan 3 task 6), after every other rule: a miss whose lineage
    # explains it. The conversation was compacted (a new, collapsed body), or
    # the harness rewrote earlier messages (the loose chain matched further
    # than the exact one, e.g. a changed <system-reminder>).
    if out in _COBS2_EXPLAINS:
        cause = _cobs2_cause(cobs2_edge, loose_match_beyond_exact)
        if cause is not None:
            return cause
    return out


def _cobs2_cause(cobs2_edge: str | None, loose_match_beyond_exact: bool) -> str | None:
    if cobs2_edge == "compacted_from":
        return "compacted"
    if loose_match_beyond_exact:
        return "harness_rewrite"
    return None


def divergence_cause(
    outcome: str | None,
    diverged_at: int | None,
    cobs2_edge: str | None,
    loose_match_beyond_exact: bool,
) -> str | None:
    """Why this request's prefix broke, when cobs2 lineage explains it.

    Set on a row ``classify`` relabelled, and on a hit or partial whose
    prefix still broke: a harness rewrite needs the index's deep-divergence
    evidence (``diverged_at``, the deep_diverged condition); a compaction is
    its own evidence, since the collapsed body has no later messages for
    ``diverged_at`` to see. ``lost`` / ``misrouted`` / unknown: None."""
    cause = _cobs2_cause(cobs2_edge, loose_match_beyond_exact)
    if cause is None:
        return None
    if outcome == cause:
        return cause
    if outcome in ("hit", "partial") and (cause == "compacted" or diverged_at is not None):
        return cause
    return None


def _classify(
    *,
    cached: int | None,
    reusable: int | None,
    reusable_fleet: int | None,
    diverged_at: int | None,
    allow_misrouted: bool,
    cached_is_estimate: bool,
    block_size: int | None,
) -> str | None:
    if cached is None or reusable is None:
        return None
    r = reusable
    # Rule 1: the engine served more than the index knew of, so R was at least
    # C. Only a MEASURED C proves that; a TTFT estimate on a cold request would
    # otherwise promote itself to a hit.
    if cached > r and not cached_is_estimate:
        r = cached
    fleet = reusable_fleet or 0
    # Block-aware (#301): the engine caches whole blocks only, so what it
    # could serve is R rounded down to a block; under one block is nothing to
    # reuse at all (cold, never lost). The thresholds then apply unchanged.
    if block_size is not None and block_size > BLOCK_AWARE_ABOVE:
        r -= r % block_size
        fleet -= fleet % block_size
    if r < MIN_REUSABLE:
        if allow_misrouted and fleet >= MIN_REUSABLE:
            return "misrouted"
        if diverged_at is not None:
            return "diverged"
        return "cold"
    if cached >= HIT * r:
        return "hit"
    if cached < LOST * r:
        return "lost"
    return "partial"
