"""VRAM-fit math for the Add Model wizard (#85, parent #82).

Pure functions, no I/O. Given a model's ``config.json`` shape and a candidate
weights file size + selected GPU VRAM, predict whether the load will fit and
classify into a 4-way verdict. Used by ``POST /api/models/fit-preview`` and
mirrored client-side by dev-2 in #86 so the modal can colour-code rows live
on every checkbox tick.

The math derivation lives on issue #82; the thresholds (green<0.55,
yellow<0.80, orange<1.0, red>=1.0) and the KV-reserve formula are locked
there. This module is the single source of truth — drift here breaks the
contract test in ``tests/unit/models/test_fit.py``.
"""

from __future__ import annotations

from typing import Literal

# Thresholds locked by #82. Anything < 0.55 is "comfortably fits" (green),
# < 0.80 is "probably fits with some KV headroom" (yellow), < 1.0 is
# "tight — only short context will fit" (orange), >= 1.0 won't load (red).
GREEN_RATIO = 0.55
YELLOW_RATIO = 0.80
ORANGE_RATIO = 1.0

# Target ratio for the recommend_max_model_len() solver: aim for "yellow"
# (well-inside-budget) rather than the orange boundary so a small misestimate
# of the weight footprint doesn't immediately tip the user back into "tight".
RECOMMENDATION_TARGET_RATIO = 0.70

Verdict = Literal["green", "yellow", "orange", "red"]


def dtype_bytes_from_torch_dtype(torch_dtype: str | None) -> int:
    """Map ``torch_dtype`` from ``config.json`` to bytes-per-element.

    Recognised: bf16/bfloat16, fp16/float16/half -> 2; fp8/float8/e4m3/e5m2
    -> 1; fp32/float32/float -> 4; int8 -> 1. Unknown / None falls back to 2
    (bf16) since that's the modern transformers default and the wizard's
    worst-case-acceptable assumption — undercounting dtype_bytes would
    silently turn a "red" row green, which is the dangerous direction.
    """
    if torch_dtype is None:
        return 2
    s = torch_dtype.strip().lower()
    if s in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if s in {"fp8", "float8", "float8_e4m3fn", "float8_e5m2", "e4m3", "e5m2", "int8"}:
        return 1
    if s in {"fp32", "float32", "float"}:
        return 4
    return 2


def kv_reserve_bytes(
    *,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    num_attention_heads: int,
    max_model_len: int,
    dtype_bytes: int,
    max_batch_size: int = 1,
    head_dim: int | None = None,
    sliding_layers: int = 0,
    sliding_window: int | None = None,
) -> int:
    """KV-cache reservation. Extends the formula locked on #82.

    ``per_layer_token = 2 * num_kv_heads * head_dim * dtype_bytes``
    ``kv_reserve      = per_layer_token * max_batch_size *
                        (full_layers * max_model_len + sliding_layers * window)``

    The factor of 2 accounts for both K and V tensors. With the defaults
    (``head_dim=None``, ``sliding_layers=0``) this is byte-identical to the
    original ``bytes_per_token * max_model_len * max_batch_size``.

    ``head_dim`` is READ FROM THE CONFIG when the model declares one, and only
    derived as ``hidden_size // num_attention_heads`` when it does not. The two
    disagree in practice: gpt-oss-20b declares ``head_dim: 64`` while the
    derivation gives ``2880 // 64 = 45``, a 30% undercount of the whole KV
    term. Undercounting turns a red row green, which
    ``dtype_bytes_from_torch_dtype`` already names as the dangerous direction.

    ``sliding_layers`` / ``sliding_window`` describe interleaved sliding-window
    attention, where only some layers hold the full context and the rest hold a
    fixed window. Charging every layer the full context roughly doubles the
    estimate for gpt-oss-20b (12 of its 24 layers slide, with a 128-token
    window). The window is capped by ``max_model_len``: a window wider than the
    context reserves the context.

    Callers pass ``sliding_layers`` only from an EXPLICIT per-layer declaration
    (``config.layer_types``). Model families that imply sliding attention some
    other way keep today's all-full-attention estimate, which over-counts —
    the safe direction — rather than inviting a per-family heuristic that would
    silently under-count whenever it guessed wrong.
    """
    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if sliding_layers < 0 or sliding_layers > num_layers:
        raise ValueError("sliding_layers must be between 0 and num_layers")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    per_layer_token = 2 * num_kv_heads * head_dim * dtype_bytes
    full_layers = num_layers - sliding_layers
    window = max_model_len
    if sliding_layers and sliding_window:
        window = min(sliding_window, max_model_len)
    tokens = full_layers * max_model_len + sliding_layers * window
    return per_layer_token * tokens * max_batch_size


def weights_budget_bytes(
    total_vram: int,
    gpu_memory_utilization: float,
    kv_reserve: int,
) -> int:
    """Bytes available for weights after vLLM's gpu_memory_utilization cap and
    KV-cache reservation.

    May return a negative integer when the KV reserve alone exceeds the
    available cap — callers (``classify_fit``, ``recommend_max_model_len``)
    handle that as the "doesn't fit at any size" signal.
    """
    return int(total_vram * gpu_memory_utilization) - kv_reserve


def classify_fit(file_size: int, budget: int) -> Verdict:
    """Verdict for a single candidate weights file.

    A non-positive budget means KV alone overflows the allowed VRAM — the
    weights can't possibly fit even at zero size, so we return "red". For a
    positive budget we compare ``file_size / budget`` against the locked
    thresholds.
    """
    if budget <= 0:
        return "red"
    ratio = file_size / budget
    if ratio < GREEN_RATIO:
        return "green"
    if ratio < YELLOW_RATIO:
        return "yellow"
    if ratio < ORANGE_RATIO:
        return "orange"
    return "red"


def recommend_max_model_len(
    *,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    num_attention_heads: int,
    total_vram: int,
    gpu_memory_utilization: float,
    file_size: int,
    dtype_bytes: int,
    max_batch_size: int = 1,
    target_ratio: float = RECOMMENDATION_TARGET_RATIO,
    head_dim: int | None = None,
    sliding_layers: int = 0,
    sliding_window: int | None = None,
) -> int | None:
    """For tight/red rows, suggest a smaller ``max_model_len`` that lands in
    the yellow band.

    The fit ratio at a candidate length L is::

        ratio(L) = file_size / (total_vram * gpu_util - kv_per_token * L * batch)

    Solving ``ratio(L) = target_ratio`` for L::

        L = (total_vram * gpu_util - file_size / target_ratio)
          / (kv_per_token * batch)

    where ``kv_per_token = 2 * full_layers * num_kv_heads * head_dim * dtype_bytes``.

    ``gpu_memory_utilization`` is the EFFECTIVE VRAM cap fraction, which for a
    backend declaring its own ``vram_cap_fraction`` is that value rather than
    any operator flag -- llama.cpp has no such flag to set.

    Returns ``None`` when there's no positive L that achieves the target —
    that's the "won't run at any context" signal the FE should surface as
    "no recommendation: model too big for selected GPUs".
    """
    if num_attention_heads <= 0:
        return None
    if target_ratio <= 0:
        return None
    if sliding_layers < 0 or sliding_layers > num_layers:
        return None
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    per_layer_token = 2 * num_kv_heads * head_dim * dtype_bytes * max_batch_size
    # Only full-attention layers grow with L. Sliding layers contribute a
    # CONSTANT once L exceeds their window, so they move to the numerator as
    # fixed overhead rather than inflating the per-token slope -- which is
    # what makes this the exact inverse of kv_reserve_bytes.
    full_layers = num_layers - sliding_layers
    kv_per_token = per_layer_token * full_layers
    if kv_per_token <= 0:
        return None
    cap = int(total_vram * gpu_memory_utilization)
    needed_weights = file_size / target_ratio
    sliding_fixed = 0
    if sliding_layers and sliding_window:
        sliding_fixed = per_layer_token * sliding_layers * sliding_window
    numerator = cap - needed_weights - sliding_fixed
    if numerator <= 0:
        return None
    L = int(numerator // kv_per_token)
    # The sliding term was treated as constant, which holds only for
    # L >= window. Below that the true reserve is smaller, so the answer is
    # conservative rather than wrong -- it under-promises context.

    # Floor at 1 to avoid returning 0 (vLLM rejects that); None when the
    # math gives a non-positive recommendation so the FE doesn't paste
    # garbage into the override field.
    return L if L >= 1 else None
