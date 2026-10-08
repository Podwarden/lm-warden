"""Unit tests for the pure load-preflight decision function.

``decide_preflight`` takes already-gathered numbers (config dict, total VRAM,
weights size, the row's ``max_model_len``, gpu_memory_utilization) and returns
a ``PreflightDecision`` of one of three kinds: proceed / cap / block. All I/O
(reading the local HF cache, probing the GPU) is done by the route; this
function is pure so the asymmetry logic is exhaustively testable here without a
GPU or HF cache.

Design (locked by the orchestrator, see issue / MR body):
  - NULL row.max_model_len + model-max context won't fit → CAP to the
    recommendation (auto-fit; NULL means "no user intent").
  - Explicit row.max_model_len + verdict red → BLOCK (don't silently rewrite
    a user's explicit choice).
  - Fits, or no recommendation available → PROCEED.
  - Missing / incomplete config → PROCEED (fail-open; never a false block).
"""

from __future__ import annotations

from app.models.fit import (
    classify_fit,
    kv_reserve_bytes,
    recommend_max_model_len,
    weights_budget_bytes,
)
from app.models.load_preflight import decide_preflight

_GIB = 1024**3

# A small model config that comfortably fits at its model-max context on a
# single 24 GiB card — used for the "proceed" cases.
_SMALL_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
}

# A config whose declared model-max (262144) is wildly larger than any single
# A4000 can serve as KV — mirrors the tencent/Hy-MT2-1.8B footgun.
_HUGE_CTX_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}

_A4000_VRAM = 16 * _GIB  # RTX A4000 ~16 GiB total
_WEIGHTS_2B = 4 * _GIB  # ~4 GiB of weights (1.8B bf16)
_GPU_UTIL = 0.90


def _expected_recommendation(config, vram, weights, util):
    return recommend_max_model_len(
        hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        num_kv_heads=config["num_key_value_heads"],
        num_attention_heads=config["num_attention_heads"],
        total_vram=vram,
        gpu_memory_utilization=util,
        file_size=weights,
        dtype_bytes=2,
    )


def test_null_max_model_len_and_model_max_wont_fit_returns_cap():
    """The footgun fix: NULL max_model_len + a 262144 model-max that blows the
    KV budget must produce a CAP decision at the recommended length."""
    decision = decide_preflight(
        config=_HUGE_CTX_CONFIG,
        total_vram=_A4000_VRAM,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "cap"
    expected = _expected_recommendation(_HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B, _GPU_UTIL)
    assert expected is not None and expected < _HUGE_CTX_CONFIG["max_position_embeddings"]
    assert decision.cap_to == expected
    assert decision.from_len == _HUGE_CTX_CONFIG["max_position_embeddings"]


def test_explicit_max_model_len_red_returns_block():
    """An EXPLICIT (non-null) max_model_len that won't fit must BLOCK — we do
    not silently rewrite the user's choice."""
    explicit = 262144
    decision = decide_preflight(
        config=_HUGE_CTX_CONFIG,
        total_vram=_A4000_VRAM,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=explicit,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "block"
    expected = _expected_recommendation(_HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B, _GPU_UTIL)
    assert decision.recommended_max_model_len == expected
    # Breakdown is surfaced to the operator so they understand the verdict.
    assert decision.breakdown is not None
    assert decision.breakdown["total_vram"] == _A4000_VRAM
    assert decision.breakdown["file_size"] == _WEIGHTS_2B
    assert decision.breakdown["max_model_len_used"] == explicit


def test_fits_returns_proceed_null():
    """A small model at its model-max context fits comfortably → PROCEED."""
    decision = decide_preflight(
        config=_SMALL_CONFIG,
        total_vram=24 * _GIB,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "proceed"


def test_fits_with_explicit_value_returns_proceed():
    """An explicit max_model_len that DOES fit → PROCEED (no block)."""
    decision = decide_preflight(
        config=_SMALL_CONFIG,
        total_vram=24 * _GIB,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=4096,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "proceed"


def test_missing_config_returns_proceed_fail_open():
    """If the config can't supply the KV inputs, fail OPEN → PROCEED. Preflight
    must NEVER produce a false block."""
    incomplete = {"max_position_embeddings": 262144}  # no hidden_size etc.
    decision = decide_preflight(
        config=incomplete,
        total_vram=_A4000_VRAM,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "proceed"


def test_empty_config_returns_proceed_fail_open():
    """Empty config dict → PROCEED."""
    decision = decide_preflight(
        config={},
        total_vram=_A4000_VRAM,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "proceed"


def test_zero_vram_fails_open_not_block():
    """A degraded/zero VRAM probe must NOT manufacture a block — fail open."""
    decision = decide_preflight(
        config=_HUGE_CTX_CONFIG,
        total_vram=0,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "proceed"


def test_null_no_recommendation_proceeds():
    """NULL max_model_len but the model is so big NO positive context fits —
    recommend_max_model_len returns None, so we PROCEED (let Feature 2 make any
    resulting crash actionable) rather than cap to a garbage value."""
    # Weights alone nearly exhaust the budget at any context.
    decision = decide_preflight(
        config=_HUGE_CTX_CONFIG,
        total_vram=_A4000_VRAM,
        weights_size=15 * _GIB,  # weights > 0.9*16GiB budget already
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "proceed"


def test_cap_decision_actually_fits_after_capping():
    """Sanity: the capped length must classify out of red against the same
    fit math the route trusts."""
    decision = decide_preflight(
        config=_HUGE_CTX_CONFIG,
        total_vram=_A4000_VRAM,
        weights_size=_WEIGHTS_2B,
        row_max_model_len=None,
        gpu_memory_utilization=_GPU_UTIL,
    )
    assert decision.kind == "cap"
    kv = kv_reserve_bytes(
        hidden_size=_HUGE_CTX_CONFIG["hidden_size"],
        num_layers=_HUGE_CTX_CONFIG["num_hidden_layers"],
        num_kv_heads=_HUGE_CTX_CONFIG["num_key_value_heads"],
        num_attention_heads=_HUGE_CTX_CONFIG["num_attention_heads"],
        max_model_len=decision.cap_to,
        dtype_bytes=2,
    )
    budget = weights_budget_bytes(_A4000_VRAM, _GPU_UTIL, kv)
    assert classify_fit(_WEIGHTS_2B, budget) != "red"
