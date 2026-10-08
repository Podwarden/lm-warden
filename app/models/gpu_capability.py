"""GPU compute-capability × quant/dtype capability rules (#176).

Pure functions, no I/O. Given the *minimum* CUDA compute capability across
the selected GPUs (e.g. ``8.6`` for an RTX A4000, sm_86) and a candidate's
quant/dtype signals from ``config.json``, return human-readable warnings
about combinations the hardware can't run *well*.

This is a **warn, never block** layer: the wizard still lets the operator
download anything, but surfaces "this will be emulated / unsupported" so a
sm_86 A4000 user doesn't pick an FP8 build expecting native FP8 throughput
(the 2026-05-26 emulated-fp8 finding) or a GGUF/AWQ dead-end.

Adding a new architecture = one line in the matrix below. The thresholds
are CUDA compute-capability floors per feature:

* FP8 (E4M3/E5M2) native tensor cores → Ada sm_89 (cc 8.9). Pre-Ada runs
  it emulated (slow, often inaccurate).
* NVFP4 / FP4 → Blackwell sm_100 (cc 10.0).
* bfloat16 → Ampere sm_80 (cc 8.0). Turing and older lack native bf16.
* Marlin INT4 (AWQ/GPTQ) → Turing sm_75 (cc 7.5).
"""

from __future__ import annotations

# Compute-capability floors (inclusive) for each feature. A GPU at or above
# the floor runs the feature natively; below it the build is emulated or
# unsupported. New arches only ever raise/lower these numbers or add a row.
FP8_NATIVE_CC = 8.9  # Ada sm_89
NVFP4_NATIVE_CC = 10.0  # Blackwell sm_100
BF16_NATIVE_CC = 8.0  # Ampere sm_80
MARLIN_INT4_CC = 7.5  # Turing sm_75


def capability_warnings(
    compute_cap: float | None,
    *,
    torch_dtype: str | None,
    quant_method: str | None,
) -> list[str]:
    """Cross ``compute_cap`` with the candidate's quant/dtype and return
    warning strings for combinations the detected GPU can't run well.

    ``compute_cap`` is the **minimum** compute capability across the selected
    GPUs (the weakest card gates what the whole tensor-parallel group can do).
    When it's ``None`` (driver reported ``[Not Supported]`` / no GPUs probed)
    we can't classify, so we return ``[]`` rather than fabricate a verdict.

    ``torch_dtype`` is ``config.json``'s ``torch_dtype``; ``quant_method`` is
    ``quantization_config.quant_method`` (awq / gptq / fp8 / compressed-tensors
    / nvfp4 …) lower-cased by the caller-or-here. GGUF candidates have no such
    config — the caller passes ``torch_dtype="bfloat16"`` for them because
    GGUF is dequantized to fp16/bf16 at load (so capability-wise it behaves
    like a bf16 build, not like its on-disk INT quant).
    """
    if compute_cap is None:
        return []

    dt = (torch_dtype or "").lower()
    qm = (quant_method or "").lower()
    warnings: list[str] = []

    # --- FP8 (E4M3/E5M2) ---------------------------------------------------
    # Detected via dtype OR quant_method. NOTE: bare "compressed-tensors" is
    # an umbrella scheme that also covers W8A8-INT8 and W4A16 — those are NOT
    # fp8 — so it only counts as fp8 when the dtype actually says fp8/float8.
    # "compressed-tensors-fp8" is unambiguous and counts unconditionally.
    dt_is_fp8 = any(tok in dt for tok in ("fp8", "float8", "e4m3", "e5m2"))
    is_fp8 = (
        dt_is_fp8
        or qm == "fp8"
        or qm == "compressed-tensors-fp8"
        or (qm == "compressed-tensors" and dt_is_fp8)
    )
    if is_fp8 and compute_cap < FP8_NATIVE_CC:
        warnings.append(
            f"fp8_emulated: native FP8 tensor cores need Ada (sm_89 / cc {FP8_NATIVE_CC})+; "
            f"this GPU is cc {compute_cap:g}, so FP8 runs emulated — slow and often "
            "inaccurate. Prefer an AWQ/GPTQ INT4 build or a bf16/fp16 build instead."
        )

    # --- NVFP4 / FP4 -------------------------------------------------------
    is_fp4 = "fp4" in dt or qm in ("nvfp4", "fp4")  # "fp4" in dt also matches "nvfp4"
    if is_fp4 and compute_cap < NVFP4_NATIVE_CC:
        warnings.append(
            f"fp4_unsupported: NVFP4/FP4 requires Blackwell (sm_100 / cc {NVFP4_NATIVE_CC})+; "
            f"this GPU is cc {compute_cap:g}, which can't run it. Find an HF-format "
            "AWQ/GPTQ INT4 or FP8 (on Ada+) quant instead."
        )

    # --- bfloat16 ----------------------------------------------------------
    is_bf16 = "bf16" in dt or "bfloat16" in dt
    if is_bf16 and compute_cap < BF16_NATIVE_CC:
        warnings.append(
            f"bf16_unsupported: bfloat16 needs Ampere (sm_80 / cc {BF16_NATIVE_CC})+; "
            f"this GPU is cc {compute_cap:g}. Use an fp16 (float16) build instead."
        )

    # --- Marlin INT4 (AWQ / GPTQ) -----------------------------------------
    is_int4 = qm in ("awq", "gptq") or "int4" in dt or "int4" in qm
    if is_int4 and compute_cap < MARLIN_INT4_CC:
        warnings.append(
            f"int4_unsupported: Marlin INT4 (AWQ/GPTQ) kernels need Turing "
            f"(sm_75 / cc {MARLIN_INT4_CC})+; this GPU is cc {compute_cap:g}. "
            "Use an fp16 build instead."
        )

    return warnings
