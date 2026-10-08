"""Unit tests for ``app/models/gpu_capability.py`` (#176).

Pure rule-matrix tests — no I/O, no GPU. Each rule + the None case +
the boundary compute-capability values (7.5 / 8.0 / 8.9 / 10.0) is
locked here so the matrix can't silently drift.
"""

from __future__ import annotations

from app.models.gpu_capability import capability_warnings


def _has(warnings: list[str], *needles: str) -> bool:
    """True if some single warning contains all the lowercase needles."""
    return any(all(n.lower() in w.lower() for n in needles) for w in warnings)


# ---- None compute_cap: never classify -----------------------------------


def test_none_compute_cap_returns_empty():
    assert capability_warnings(None, torch_dtype="float8_e4m3fn", quant_method="fp8") == []
    assert capability_warnings(None, torch_dtype="bfloat16", quant_method=None) == []


# ---- fp8 family on pre-Ada (cc < 8.9) -----------------------------------


def test_fp8_dtype_on_sm86_warns_emulated():
    w = capability_warnings(8.6, torch_dtype="float8_e4m3fn", quant_method=None)
    assert _has(w, "fp8", "emulated")
    assert _has(w, "awq") or _has(w, "bf16")


def test_fp8_quant_method_on_sm86_warns():
    w = capability_warnings(8.6, torch_dtype=None, quant_method="fp8")
    assert _has(w, "fp8", "emulated")


def test_compressed_tensors_fp8_suffix_quant_method_on_sm86_warns():
    # "compressed-tensors-fp8" is unambiguous: warn even with no dtype signal.
    w = capability_warnings(8.6, torch_dtype=None, quant_method="compressed-tensors-fp8")
    assert _has(w, "fp8", "emulated")


def test_compressed_tensors_with_fp8_dtype_on_sm86_warns():
    # Bare "compressed-tensors" counts as fp8 only when the dtype says so.
    w = capability_warnings(8.6, torch_dtype="float8_e4m3fn", quant_method="compressed-tensors")
    assert _has(w, "fp8", "emulated")


def test_compressed_tensors_int8_dtype_on_sm86_no_fp8_warning():
    # compressed-tensors also covers W8A8-INT8 / W4A16 — NOT fp8. A non-fp8
    # dtype (float16) must NOT trigger a spurious "fp8 emulated" warning on a
    # pre-Ada card.
    w = capability_warnings(8.6, torch_dtype="float16", quant_method="compressed-tensors")
    assert w == []


def test_e5m2_dtype_on_sm86_warns():
    w = capability_warnings(8.6, torch_dtype="e5m2", quant_method=None)
    assert _has(w, "fp8", "emulated")


def test_fp8_boundary_cc_8_9_no_warning():
    # Ada (sm_89) has native FP8 — exactly at the boundary, no warning.
    assert capability_warnings(8.9, torch_dtype="float8_e4m3fn", quant_method="fp8") == []


def test_fp8_on_hopper_no_warning():
    assert capability_warnings(9.0, torch_dtype="float8_e4m3fn", quant_method="fp8") == []


# ---- nvfp4 / fp4 needs Blackwell (cc < 10.0) ----------------------------


def test_nvfp4_on_sm89_warns_blackwell():
    w = capability_warnings(8.9, torch_dtype=None, quant_method="nvfp4")
    assert _has(w, "blackwell")


def test_fp4_dtype_on_hopper_warns():
    w = capability_warnings(9.0, torch_dtype="fp4", quant_method=None)
    assert _has(w, "blackwell")


def test_nvfp4_boundary_cc_10_0_no_warning():
    assert capability_warnings(10.0, torch_dtype=None, quant_method="nvfp4") == []


# ---- bf16 needs Ampere (cc < 8.0) ---------------------------------------


def test_bf16_on_turing_warns_ampere():
    w = capability_warnings(7.5, torch_dtype="bfloat16", quant_method=None)
    assert _has(w, "bf16", "ampere")


def test_bf16_boundary_cc_8_0_no_warning():
    assert capability_warnings(8.0, torch_dtype="bfloat16", quant_method=None) == []


def test_bf16_alias_warns_on_old_card():
    w = capability_warnings(7.0, torch_dtype="bf16", quant_method=None)
    assert _has(w, "bf16", "ampere")


# ---- AWQ / GPTQ / INT4 needs Marlin sm_75+ (cc < 7.5) -------------------


def test_awq_on_pre_turing_warns_marlin():
    w = capability_warnings(7.0, torch_dtype=None, quant_method="awq")
    assert _has(w, "marlin")


def test_gptq_on_pre_turing_warns_marlin():
    w = capability_warnings(7.0, torch_dtype=None, quant_method="gptq")
    assert _has(w, "marlin")


def test_awq_boundary_cc_7_5_no_warning():
    assert capability_warnings(7.5, torch_dtype=None, quant_method="awq") == []


def test_awq_on_modern_card_no_warning():
    assert capability_warnings(8.6, torch_dtype=None, quant_method="awq") == []


# ---- Healthy combos: no warnings ----------------------------------------


def test_bf16_on_modern_card_no_warning():
    assert capability_warnings(8.6, torch_dtype="bfloat16", quant_method=None) == []


def test_fp16_anywhere_no_warning():
    assert capability_warnings(7.0, torch_dtype="float16", quant_method=None) == []
