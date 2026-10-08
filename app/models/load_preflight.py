"""Pure KV-budget preflight decision for the load path.

Background (the footgun): a user switched the box to a HuggingFace model whose
``config.json`` declares ``max_position_embeddings: 262144``. The model row had
``max_model_len = NULL``, so ``build_vllm_args`` omitted ``--max-model-len``
entirely, vLLM defaulted to the 262144 model max, the KV cache (~16 GiB)
exceeded the single A4000's budget, and the engine subprocess crashed at
startup. The control plane reported only a generic ``last_error``.

This module is the PURE decision half of the preflight. The route
(``app/models/routes_api.py:load_model``) does all the I/O — reads the local
HF-cache ``config.json``, probes GPU VRAM, sums the local snapshot size — and
hands the already-gathered numbers here. Keeping the decision pure makes the
asymmetric NULL-vs-explicit policy exhaustively unit-testable without a GPU or
an HF cache (see ``tests/unit/models/test_load_preflight.py``).

Policy (locked by the orchestrator):
  - ``row_max_model_len is None`` (no user intent) AND the model-max context
    would NOT fit  → **CAP** to the ``fit.recommend_max_model_len`` value.
  - ``row_max_model_len`` is set (explicit user intent) AND the verdict is
    "red" at that length → **BLOCK** (HTTP 422 upstream). Never silently
    rewrite an explicit choice.
  - Otherwise (fits, or no positive recommendation exists) → **PROCEED**.

Fail-open is mandatory: if the config can't supply the KV inputs, or VRAM is
unknown/zero, return PROCEED. The preflight must NEVER produce a false block —
Feature 2 (``log_diagnostics``) makes any resulting crash actionable.

The fit math is reused verbatim from ``app/models/fit.py`` (thresholds LOCKED
there); this module adds no new numeric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.models.fit import (
    classify_fit,
    dtype_bytes_from_torch_dtype,
    kv_reserve_bytes,
    recommend_max_model_len,
    weights_budget_bytes,
)


@dataclass(frozen=True)
class PreflightDecision:
    """Outcome of the pure preflight.

    ``kind``:
      - ``"proceed"`` — load unchanged (pass overrides=None upstream).
      - ``"cap"`` — auto-cap a NULL max_model_len; ``cap_to`` and ``from_len``
        populate the ``context_capped`` response field.
      - ``"block"`` — refuse an explicit-but-won't-fit choice; ``breakdown``
        and ``recommended_max_model_len`` populate the 422 detail.
    """

    kind: Literal["proceed", "cap", "block"]
    cap_to: int | None = None
    from_len: int | None = None
    recommended_max_model_len: int | None = None
    breakdown: dict[str, Any] | None = None


# Singleton — avoids re-allocating identical proceed decisions.
_PROCEED = PreflightDecision(kind="proceed")


def _config_complete(config: dict[str, Any]) -> tuple[bool, int, int, int, int, int, int]:
    """Extract + validate the KV inputs from a config dict.

    Returns ``(complete, hidden_size, num_layers, num_attn_heads, num_kv_heads,
    max_pos_emb, dtype_bytes)``. ``complete`` is False when any of the four KV
    dimensions is missing/zero, which mirrors ``fit_preview``'s degraded-config
    guard. ``num_key_value_heads`` falls back to ``num_attention_heads`` (GQA
    vs MHA), matching ``fit_preview``.
    """
    hidden_size = int(config.get("hidden_size") or 0)
    num_layers = int(config.get("num_hidden_layers") or 0)
    num_attn_heads = int(config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_key_value_heads") or num_attn_heads or 0)
    max_pos_emb = int(config.get("max_position_embeddings") or 0)
    dtype_bytes = dtype_bytes_from_torch_dtype(config.get("torch_dtype"))
    complete = bool(hidden_size and num_layers and num_attn_heads and num_kv_heads)
    return (
        complete,
        hidden_size,
        num_layers,
        num_attn_heads,
        num_kv_heads,
        max_pos_emb,
        dtype_bytes,
    )


def decide_preflight(
    *,
    config: dict[str, Any],
    total_vram: int,
    weights_size: int,
    row_max_model_len: int | None,
    gpu_memory_utilization: float,
    max_batch_size: int = 1,
) -> PreflightDecision:
    """Decide whether to proceed / cap / block a load, given gathered numbers.

    All arguments are already-resolved local values supplied by the route. This
    function performs no I/O and never raises for degraded inputs — it returns
    PROCEED (fail-open) whenever it cannot make a confident negative call.
    """
    # Fail-open guard rails: anything we can't trust → PROCEED.
    if total_vram <= 0 or weights_size <= 0 or gpu_memory_utilization <= 0:
        return _PROCEED

    (
        complete,
        hidden_size,
        num_layers,
        num_attn_heads,
        num_kv_heads,
        max_pos_emb,
        dtype_bytes,
    ) = _config_complete(config)

    if not complete:
        return _PROCEED

    # Effective length: the explicit row value if set, else the model-max from
    # config. When neither is known (max_pos_emb == 0 and row is None) we have
    # no length to evaluate KV against → PROCEED.
    effective_len = row_max_model_len or max_pos_emb
    if effective_len <= 0:
        return _PROCEED

    kv = kv_reserve_bytes(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attn_heads,
        max_model_len=effective_len,
        dtype_bytes=dtype_bytes,
        max_batch_size=max_batch_size,
    )
    budget = weights_budget_bytes(total_vram, gpu_memory_utilization, kv)
    verdict = classify_fit(weights_size, budget)

    # Recommendation (capped at the model's own max_position_embeddings, like
    # fit_preview) — used for both the CAP target and the BLOCK hint.
    rec = recommend_max_model_len(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attn_heads,
        total_vram=total_vram,
        gpu_memory_utilization=gpu_memory_utilization,
        file_size=weights_size,
        dtype_bytes=dtype_bytes,
        max_batch_size=max_batch_size,
    )
    if rec is not None and max_pos_emb > 0:
        rec = min(rec, max_pos_emb)

    ratio = (weights_size / budget) if budget > 0 else float("inf")
    breakdown = {
        "total_vram": total_vram,
        "weights_budget": budget,
        "kv_reserve": kv,
        "file_size": weights_size,
        "ratio": ratio if ratio != float("inf") else 1e9,
        "dtype_bytes": dtype_bytes,
        "max_model_len_used": effective_len,
        "verdict": verdict,
    }

    if row_max_model_len is None:
        # NULL → no user intent. Auto-cap only when the model-max context is
        # the thing that won't fit AND we have a strictly-smaller recommendation
        # that lands out of the red band. If there's no positive recommendation
        # (model too big at any context) PROCEED — capping to a garbage value
        # would just defer the crash, and Feature 2 will explain it.
        if verdict == "red" and rec is not None and rec < effective_len:
            return PreflightDecision(
                kind="cap",
                cap_to=rec,
                from_len=effective_len,
                breakdown=breakdown,
            )
        return _PROCEED

    # Explicit value → user intent. Block only the hard "red" verdict; a
    # tight-but-loadable orange is the user's call to make.
    if verdict == "red":
        return PreflightDecision(
            kind="block",
            recommended_max_model_len=rec,
            breakdown=breakdown,
        )
    return _PROCEED
