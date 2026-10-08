"""Planning the configuration sweep, and choosing the answer (design §6).

This module answers the question the feature exists for — *what should
``max_model_len`` be for this model on this hardware?* — rather than merely
reporting where things broke. A breaking point is interesting; a setting you
can apply is useful.

Two rules govern everything here.

**Prediction prunes; measurement decides.** ``app/models/fit.py`` can tell us a
candidate is certainly too large for the VRAM present, and that is worth acting
on because every reload costs up to ``load_timeout_s`` = 600 seconds. It cannot
tell us a candidate will work: raising qwen3.8-27b from 65,536 to 131,072 cost
approximately zero additional VRAM (measured on pw-bonus, 2026-09-03), because
sliding-window attention makes long contexts cost sub-linearly. So the estimate
is least reliable exactly where the answer lives. No predicted number is ever
published — prediction only ever removes candidates that cannot possibly work.

**Sweeping is expensive.** Candidates are few and geometric. A dense ladder
would spend hours to distinguish values no operator would set differently.
"""

from __future__ import annotations

from dataclasses import dataclass


def plan_candidates(
    *,
    current: int | None,
    model_ceiling: int | None,
    vram_budget_bytes: int | None,
    kv_bytes_per_token: int | None,
    floor: int = 4096,
) -> list[int]:
    """The ``max_model_len`` values worth actually trying.

    ``model_ceiling`` is the model's own trained context
    (``max_position_embeddings``, or the GGUF's ``n_ctx_train``). Beyond it the
    engine refuses, so a probe there measures the config file rather than the
    hardware.

    A lower value than ``current`` is a legitimate answer: if the running
    context is too ambitious for the card, the correct recommendation is to
    reduce it.
    """
    ceiling = model_ceiling or (current or floor)
    start = current or floor
    candidates: set[int] = set()

    # Climb from the current setting. `current` is the one value with direct
    # evidence behind it -- the model is running at it -- so it is both the
    # baseline and the natural floor. Probing below a value already known to
    # work costs a reload (up to 600s) to confirm what we can already see.
    x = start
    while x < ceiling:
        candidates.add(x)
        x *= 2
    candidates.add(min(start, ceiling))
    candidates.add(ceiling)

    # ...unless there is no headroom at all. When the model is already at its
    # own trained ceiling, "can we go higher" is unaskable and the only open
    # question becomes whether the current setting is too ambitious for this
    # card -- so descend instead.
    if start >= ceiling:
        down = start // 2
        for _ in range(2):
            if down >= floor:
                candidates.add(down)
            down //= 2

    if kv_bytes_per_token and vram_budget_bytes:
        impossible = {
            c for c in candidates if c != current and c * kv_bytes_per_token > vram_budget_bytes
        }
        candidates -= impossible

    return sorted(candidates)


@dataclass(frozen=True)
class CandidateResult:
    """What happened at one ``max_model_len``.

    ``loaded`` False is usually KV-pool OOM, which surfaces at load: both
    engines allocate the pool at startup, so an over-large context fails
    cheaply and cleanly before serving anything, and costs no crash budget.
    """

    max_model_len: int
    loaded: bool
    quality_ok: bool | None
    crashed: bool = False
    gate_tripped: str | None = None


@dataclass(frozen=True)
class Recommendation:
    max_model_len: int
    limited_by: str  # quality | load | crash | model_ceiling
    next_failed_at: int | None
    gate_tripped: str | None = None


def choose_recommendation(results: list[CandidateResult]) -> Recommendation | None:
    """The largest context that loaded, held quality, and did not crash.

    Returns ``None`` when nothing qualified. Silence is a usable answer here;
    a fabricated setting is not, and an operator who applies an invented number
    has been actively misled.

    ``limited_by`` is reported because the remedies differ and collapsing them
    would throw away the actionable half: a memory wall wants more VRAM or a
    smaller quant, a quality wall wants a different model or quantisation, and
    reaching the top of the ladder means nothing failed at all.
    """
    ok = [r for r in results if r.loaded and r.quality_ok is not False and not r.crashed]
    if not ok:
        return None

    best = max(ok, key=lambda r: r.max_model_len)
    failures = sorted(
        (r for r in results if r.max_model_len > best.max_model_len),
        key=lambda r: r.max_model_len,
    )

    if not failures:
        # Every candidate we tried worked, so the only bound established is the
        # model's own trained context. Reporting a quality wall here would
        # invent one that was never hit.
        return Recommendation(
            max_model_len=best.max_model_len,
            limited_by="model_ceiling",
            next_failed_at=None,
        )

    first = failures[0]
    if not first.loaded:
        limited_by = "load"
    elif first.crashed:
        limited_by = "crash"
    else:
        limited_by = "quality"

    return Recommendation(
        max_model_len=best.max_model_len,
        limited_by=limited_by,
        next_failed_at=first.max_model_len,
        gate_tripped=first.gate_tripped,
    )
