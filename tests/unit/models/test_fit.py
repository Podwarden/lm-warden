"""Pure-function tests for ``app/models/fit.py`` (#85, parent #82).

Hand-computed expected values for realistic configs (gpt-oss-20b on 1x and 2x
A4000 16 GB cards, plus a multi-part safetensors aggregation case) and
boundary tests for the 4-way colour thresholds. If any of these numbers
drift, the wire contract in ``POST /api/models/fit-preview`` has drifted —
update the test intentionally, then update dev-2's #86 mock fixture to match.
"""

from __future__ import annotations

import pytest

from app.models.fit import (
    classify_fit,
    dtype_bytes_from_torch_dtype,
    kv_reserve_bytes,
    recommend_max_model_len,
    weights_budget_bytes,
)

# ---- Constants used by hand-computed cases -------------------------------

GIB = 1024**3
A4000_VRAM_BYTES = 16 * GIB  # nvidia-smi reports 16376 MiB; round to 16 GiB
# for hand math — the route uses MiB*MiB precisely.


# ---- dtype_bytes mapping -------------------------------------------------


@pytest.mark.parametrize(
    "torch_dtype,expected",
    [
        ("bf16", 2),
        ("bfloat16", 2),
        ("fp16", 2),
        ("float16", 2),
        ("half", 2),
        ("fp8", 1),
        ("float8_e4m3fn", 1),
        ("e5m2", 1),
        ("int8", 1),
        ("fp32", 4),
        ("float32", 4),
        ("float", 4),
        # Unknowns fall back to 2 (bf16) so we don't undercount — silently
        # returning 1 here would optimistically green-light fp32 models.
        ("totally-bogus", 2),
        (None, 2),
        ("  BFloat16  ", 2),  # whitespace + case-insensitive
    ],
)
def test_dtype_bytes_from_torch_dtype(torch_dtype, expected):
    assert dtype_bytes_from_torch_dtype(torch_dtype) == expected


# ---- kv_reserve_bytes hand-computation -----------------------------------


def test_kv_reserve_qwen_7b_at_4k_context():
    """Qwen2.5-7B (GQA, 28 attention heads, 4 KV heads) at 4 K context, bf16.

    bytes_per_token = 2 * 28 * 4 * (3584/28) * 2 = 57344
    kv_reserve      = 57344 * 4096 * 1 = 234881024 bytes (~224 MiB)
    """
    got = kv_reserve_bytes(
        hidden_size=3584,
        num_layers=28,
        num_kv_heads=4,
        num_attention_heads=28,
        max_model_len=4096,
        dtype_bytes=2,
    )
    assert got == 234881024  # 224 MiB exactly


def test_kv_reserve_scales_linearly_with_batch():
    base = kv_reserve_bytes(
        hidden_size=4096,
        num_layers=32,
        num_kv_heads=8,
        num_attention_heads=32,
        max_model_len=2048,
        dtype_bytes=2,
    )
    quad = kv_reserve_bytes(
        hidden_size=4096,
        num_layers=32,
        num_kv_heads=8,
        num_attention_heads=32,
        max_model_len=2048,
        dtype_bytes=2,
        max_batch_size=4,
    )
    assert quad == 4 * base


def test_kv_reserve_rejects_zero_num_attention_heads():
    with pytest.raises(ValueError):
        kv_reserve_bytes(
            hidden_size=4096,
            num_layers=32,
            num_kv_heads=8,
            num_attention_heads=0,
            max_model_len=2048,
            dtype_bytes=2,
        )


# ---- weights_budget_bytes ------------------------------------------------


def test_weights_budget_single_a4000_at_90_percent():
    """Single A4000 16 GiB, gpu_util=0.9, 224 MiB KV reserve.

    cap     = 16 GiB * 0.9 = 14.4 GiB
    weights = 14.4 GiB - 224 MiB ~= 14_237_949_133 bytes
    """
    cap = int(16 * GIB * 0.9)
    kv = 234881024
    got = weights_budget_bytes(16 * GIB, 0.9, kv)
    assert got == cap - kv


def test_weights_budget_can_go_negative_when_kv_overflows():
    """KV reserve larger than cap returns a negative budget — classify_fit
    treats anything <= 0 as red."""
    got = weights_budget_bytes(1 * GIB, 0.9, 10 * GIB)
    assert got < 0


# ---- classify_fit boundaries ---------------------------------------------


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0.0, "green"),
        (0.549, "green"),
        # 0.55 boundary itself is yellow (strict < on green).
        (0.55, "yellow"),
        (0.551, "yellow"),
        (0.799, "yellow"),
        (0.80, "orange"),
        (0.801, "orange"),
        (0.999, "orange"),
        (1.0, "red"),
        (1.001, "red"),
        (2.5, "red"),
    ],
)
def test_classify_fit_thresholds(ratio, expected):
    # Pick budget=1_000_000 so file_size = ratio * 1_000_000 cleanly.
    budget = 1_000_000
    file_size = int(ratio * budget)
    # Re-derive ratio from int(file_size)/budget — for 0.549 that's
    # 549000/1000000 = 0.549, exact under IEEE 754 fwiw.
    assert classify_fit(file_size, budget) == expected


def test_classify_fit_red_when_budget_non_positive():
    assert classify_fit(1, 0) == "red"
    assert classify_fit(1, -100) == "red"
    assert classify_fit(0, 0) == "red"


# ---- Realistic case: gpt-oss-20b on 1x and 2x A4000 -----------------------

# Config numbers approximated from a 20 B dense transformer (matches the
# class of gpt-oss-20b for fit-math purposes — exact param-by-param shape
# isn't shipped publicly but the order of magnitude is what counts here).
GPT_OSS_20B_CONFIG = dict(
    hidden_size=6144,
    num_layers=44,
    num_attention_heads=48,
    num_kv_heads=8,  # GQA
    max_model_len=4096,
    dtype_bytes=2,  # bf16
)
GPT_OSS_20B_FILE_SIZE = int(19.8 * GIB)  # ~19.8 GiB single-file weight


def test_gpt_oss_20b_red_on_single_a4000():
    """20 B bf16 model doesn't fit on one 16 GiB card — must be red."""
    kv = kv_reserve_bytes(**GPT_OSS_20B_CONFIG)
    budget = weights_budget_bytes(A4000_VRAM_BYTES, 0.9, kv)
    assert classify_fit(GPT_OSS_20B_FILE_SIZE, budget) == "red"


def test_gpt_oss_20b_fits_on_2x_a4000_with_headroom():
    """Same model across 2x A4000 (32 GiB total) at 4 K context."""
    kv = kv_reserve_bytes(**GPT_OSS_20B_CONFIG)
    budget = weights_budget_bytes(2 * A4000_VRAM_BYTES, 0.9, kv)
    verdict = classify_fit(GPT_OSS_20B_FILE_SIZE, budget)
    # 19.8 / (32*0.9 - kv) ~= 19.8 / 28.59 ~= 0.69 — yellow.
    assert verdict == "yellow"


# ---- Multi-part safetensors aggregation ----------------------------------


def test_classify_fit_aggregates_sharded_safetensors():
    """Operator picks shard 1-of-4 of a 30 GB model; we classify against the
    aggregate 30 GB, not the 7.5 GB slice (the fit-preview route does the
    aggregation; here we assert ``classify_fit`` produces the right verdict
    when fed the summed size).
    """
    aggregate = 30 * GIB
    kv = kv_reserve_bytes(
        hidden_size=8192,
        num_layers=80,
        num_kv_heads=8,
        num_attention_heads=64,
        max_model_len=8192,
        dtype_bytes=2,
    )
    budget_2x = weights_budget_bytes(2 * A4000_VRAM_BYTES, 0.9, kv)
    # 30/(32*0.9 - kv) overflows -> red.
    assert classify_fit(aggregate, budget_2x) == "red"

    budget_4x = weights_budget_bytes(4 * A4000_VRAM_BYTES, 0.9, kv)
    # ~30/57.6 ~= 0.52 -> green.
    assert classify_fit(aggregate, budget_4x) == "green"


# ---- recommend_max_model_len ---------------------------------------------


def test_recommend_max_model_len_brings_red_to_yellow():
    """A red verdict at max context should yield a smaller context whose
    re-classified ratio is at or below the yellow boundary (0.80).
    """
    # 2x A4000, gpt-oss-20b at 64 K context — kv reserve eats most of budget,
    # pushing the verdict into red. (At 16 K context this combo is still
    # yellow; we need a larger context for the recommendation path to fire.)
    cfg = dict(GPT_OSS_20B_CONFIG)
    cfg["max_model_len"] = 65536
    kv = kv_reserve_bytes(**cfg)
    budget = weights_budget_bytes(2 * A4000_VRAM_BYTES, 0.9, kv)
    assert classify_fit(GPT_OSS_20B_FILE_SIZE, budget) in ("orange", "red")

    rec = recommend_max_model_len(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        num_attention_heads=cfg["num_attention_heads"],
        total_vram=2 * A4000_VRAM_BYTES,
        gpu_memory_utilization=0.9,
        file_size=GPT_OSS_20B_FILE_SIZE,
        dtype_bytes=cfg["dtype_bytes"],
    )
    assert rec is not None
    assert rec > 0

    # Re-classify at the recommendation; should be at most yellow.
    new_kv = kv_reserve_bytes(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        num_attention_heads=cfg["num_attention_heads"],
        max_model_len=rec,
        dtype_bytes=cfg["dtype_bytes"],
    )
    new_budget = weights_budget_bytes(2 * A4000_VRAM_BYTES, 0.9, new_kv)
    assert classify_fit(GPT_OSS_20B_FILE_SIZE, new_budget) in ("green", "yellow")


def test_recommend_max_model_len_none_when_weights_alone_overflow():
    """A 30 GB weights file on a single 16 GiB card has no positive
    max_model_len that helps — recommend None rather than 1."""
    rec = recommend_max_model_len(
        hidden_size=4096,
        num_layers=32,
        num_kv_heads=8,
        num_attention_heads=32,
        total_vram=A4000_VRAM_BYTES,
        gpu_memory_utilization=0.9,
        file_size=30 * GIB,
        dtype_bytes=2,
    )
    assert rec is None


def test_recommend_max_model_len_handles_zero_attention_heads():
    rec = recommend_max_model_len(
        hidden_size=4096,
        num_layers=32,
        num_kv_heads=8,
        num_attention_heads=0,
        total_vram=A4000_VRAM_BYTES,
        gpu_memory_utilization=0.9,
        file_size=2 * GIB,
        dtype_bytes=2,
    )
    assert rec is None


# ---- head_dim and sliding-window attention (fit-preview backend-awareness) --
#
# Both are properties of the MODEL, not of the engine, so they belong here
# rather than in the backend-aware budget math. They are in this change
# because without them the backend fix does not actually move the verdict:
# for gpt-oss-20b the two errors are worth ~2x on the KV term between them.

GPT_OSS_20B_REAL = dict(
    hidden_size=2880,
    num_layers=24,
    num_kv_heads=8,
    num_attention_heads=64,
    max_model_len=131072,
    dtype_bytes=2,
)


def test_head_dim_is_used_when_the_config_declares_one():
    """gpt-oss-20b declares head_dim 64; hidden_size // heads gives 45.

    2880 // 64 = 45, but the config says 64. Deriving it undercounts the KV
    cache by 30% -- the direction dtype_bytes_from_torch_dtype's docstring
    calls "the dangerous direction", because it turns a red row green.
    """
    derived = kv_reserve_bytes(**GPT_OSS_20B_REAL)
    declared = kv_reserve_bytes(**GPT_OSS_20B_REAL, head_dim=64)
    assert derived < declared
    # 2 * 24 * 8 * 64 * 2 = 49152 bytes/token
    assert declared == 49152 * 131072


def test_head_dim_defaults_to_the_derived_value():
    """Absent head_dim, behaviour is byte-identical to before this change."""
    assert kv_reserve_bytes(**GPT_OSS_20B_REAL) == kv_reserve_bytes(
        **GPT_OSS_20B_REAL, head_dim=None
    )
    assert kv_reserve_bytes(**GPT_OSS_20B_REAL) == 34560 * 131072


def test_sliding_layers_only_hold_their_window():
    """gpt-oss-20b: 12 full_attention + 12 sliding_attention, window 128.

    Charging all 24 layers the full 131072-token context doubles the KV
    estimate. Per layer per token = 2 * 8 * 64 * 2 = 2048 bytes.
    """
    got = kv_reserve_bytes(**GPT_OSS_20B_REAL, head_dim=64, sliding_layers=12, sliding_window=128)
    expected = 2048 * (12 * 131072 + 12 * 128)
    assert got == expected
    # Just over half the all-full-attention figure, not the whole thing.
    assert got < kv_reserve_bytes(**GPT_OSS_20B_REAL, head_dim=64)


def test_sliding_window_is_capped_by_the_context():
    """A window wider than the context reserves the context, not the window."""
    got = kv_reserve_bytes(
        hidden_size=2880,
        num_layers=2,
        num_kv_heads=8,
        num_attention_heads=64,
        max_model_len=64,
        dtype_bytes=2,
        head_dim=64,
        sliding_layers=1,
        sliding_window=4096,
    )
    # Both layers reserve 64 tokens: the full one by definition, the sliding
    # one because its window exceeds what exists.
    assert got == 2048 * (1 * 64 + 1 * 64)


def test_no_sliding_layers_is_todays_behaviour():
    assert kv_reserve_bytes(
        **GPT_OSS_20B_REAL, head_dim=64, sliding_layers=0, sliding_window=128
    ) == kv_reserve_bytes(**GPT_OSS_20B_REAL, head_dim=64)


def test_recommendation_uses_the_same_kv_shape_as_the_reserve():
    """The two must agree, or the suggested context does not land in yellow.

    `recommend_max_model_len` solves the inverse of `kv_reserve_bytes`. If one
    knows about head_dim and interleaved sliding layers and the other does
    not, the number pasted into the override field produces a different
    verdict than the one that produced the recommendation.
    """
    GIB = 1024**3
    common = dict(
        hidden_size=2880,
        num_layers=24,
        num_kv_heads=8,
        num_attention_heads=64,
        dtype_bytes=2,
        head_dim=64,
        sliding_layers=12,
        sliding_window=128,
    )
    total_vram = 16 * GIB
    # Deliberately NOT the 11.28 GiB gpt-oss Q8_0 case: at the 0.70 yellow
    # target that file needs 16.11 GiB of budget for weights alone on a 16 GiB
    # card, so `recommend_max_model_len` correctly returns None -- there is no
    # context at which it is comfortable, only contexts at which it is tight.
    file_size = int(9 * GIB)

    rec = recommend_max_model_len(
        **common,
        total_vram=total_vram,
        gpu_memory_utilization=1.0,
        file_size=file_size,
    )
    assert rec is not None and rec > 0

    # Feed the recommendation back through the reserve and confirm the
    # verdict it produces is actually the yellow band it aimed for.
    kv = kv_reserve_bytes(**common, max_model_len=rec)
    budget = weights_budget_bytes(total_vram, 1.0, kv)
    assert classify_fit(file_size, budget) in ("green", "yellow")
