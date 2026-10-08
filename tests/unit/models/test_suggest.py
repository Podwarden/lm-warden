"""Unit tests for ``app/models/suggest.py`` — the pure heuristic that
backs ``GET /api/models/{id}/suggest-config``.

Per the S3 plan (#113), the heuristic returns STARTING POINTS, never
auto-applied. The endpoint payload MUST carry a ``disclaimer`` field
explicitly saying so. Tests below cover four representative archetypes
the operator is likely to point at:

* dense 7B (single GPU, no quant marker, modest context)
* MoE 35B (multi-GPU, large config)
* AWQ-INT4 27B (#113 KV-cache fp8 recommendation path)
* GGUF tiny (no config.json — degraded path)

Tests assert structure (keys + disclaimer text) AND values (the
``gpu_memory_utilization = total_vram * 0.92`` and ``kv_cache_dtype=fp8``
branches operators will actually rely on).
"""

from app.models.suggest import (
    DISCLAIMER_TEXT,
    SuggestedConfig,
    suggest_config,
)


def _gpu_total_bytes_for(total_mib: int) -> int:
    """Helper — express VRAM the way callers will (from nvidia-smi MiB)."""
    return total_mib * 1024 * 1024


def test_suggest_dense_7b_single_gpu_24gib():
    """Dense 7B on a single 24 GiB card. Should produce gmu=0.92,
    max_model_len from max_position_embeddings, no kv_cache_dtype
    recommendation (no AWQ marker)."""
    out = suggest_config(
        hf_repo="meta-llama/Llama-3.1-7B-Instruct",
        config={
            "max_position_embeddings": 131072,
            "torch_dtype": "bfloat16",
        },
        total_vram_bytes=_gpu_total_bytes_for(24 * 1024),  # one 24 GiB card
        filenames=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "config.json",
        ],
    )
    assert isinstance(out, SuggestedConfig)
    assert out.gpu_memory_utilization == 0.92
    assert out.max_model_len == 131072
    assert out.kv_cache_dtype is None  # no AWQ → no fp8 recommendation
    assert out.disclaimer == DISCLAIMER_TEXT


def test_suggest_moe_35b_multi_gpu():
    """MoE 35B on 2x 48 GiB cards (96 GiB total). Numbers should still be
    a single gmu=0.92 (per-process util, not aggregate) and max_model_len
    from config."""
    out = suggest_config(
        hf_repo="mistralai/Mixtral-8x7B-v0.1",
        config={
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
            "model_type": "mixtral",
        },
        total_vram_bytes=_gpu_total_bytes_for(2 * 48 * 1024),
        filenames=["model-00001-of-00019.safetensors", "config.json"],
    )
    assert out.gpu_memory_utilization == 0.92
    assert out.max_model_len == 32768
    assert out.kv_cache_dtype is None
    assert out.disclaimer == DISCLAIMER_TEXT


def test_suggest_awq_int4_27b_recommends_fp8_kv_cache():
    """#113 — AWQ-quantized models should get ``kv_cache_dtype='fp8'``.
    The classic 27B-class model that motivated the issue is Gemma-2 27B
    AWQ. The recommendation comes from inspecting EITHER the hf_repo
    name OR file markers; both paths are tested."""
    out = suggest_config(
        hf_repo="hugging-quants/gemma-2-27b-it-AWQ-INT4",
        config={
            "max_position_embeddings": 8192,
            "torch_dtype": "float16",
            "quantization_config": {"quant_method": "awq", "bits": 4},
        },
        total_vram_bytes=_gpu_total_bytes_for(2 * 24 * 1024),
        filenames=["model.safetensors", "config.json"],
    )
    assert out.gpu_memory_utilization == 0.92
    assert out.max_model_len == 8192
    assert out.kv_cache_dtype == "fp8"
    assert out.disclaimer == DISCLAIMER_TEXT


def test_suggest_awq_detected_from_filename_when_repo_name_is_clean():
    """Some operators rename their repo without the AWQ tag but the
    weights file still carries it. We must still recommend fp8 KV cache
    in that case (#113's exact failure mode in the live archive)."""
    out = suggest_config(
        hf_repo="user/private-mirror-of-some-model",
        config={"max_position_embeddings": 4096, "torch_dtype": "float16"},
        total_vram_bytes=_gpu_total_bytes_for(24 * 1024),
        filenames=["model-AWQ.safetensors", "config.json"],
    )
    assert out.kv_cache_dtype == "fp8"


def test_suggest_gguf_tiny_no_config():
    """GGUF model with no transformers config.json. We can't suggest
    ``max_model_len`` (config is None or empty); the field must come
    back as ``None`` rather than guessed or zero so the FE can fall back
    to the row default. gmu still recommended."""
    out = suggest_config(
        hf_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        config=None,
        total_vram_bytes=_gpu_total_bytes_for(8 * 1024),
        filenames=["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"],
    )
    assert out.gpu_memory_utilization == 0.92
    assert out.max_model_len is None
    assert out.kv_cache_dtype is None
    assert out.disclaimer == DISCLAIMER_TEXT


def test_suggest_handles_missing_max_position_embeddings():
    """Some configs ship without ``max_position_embeddings`` even though
    a config.json exists. The heuristic must return ``None`` for
    ``max_model_len`` rather than crashing or substituting a guess."""
    out = suggest_config(
        hf_repo="example/odd-model",
        config={"torch_dtype": "bfloat16"},  # no max_position_embeddings
        total_vram_bytes=_gpu_total_bytes_for(24 * 1024),
        filenames=["model.safetensors"],
    )
    assert out.max_model_len is None
    assert out.gpu_memory_utilization == 0.92


def test_suggest_zero_vram_returns_gmu_unchanged():
    """``total_vram_bytes=0`` is a degraded probe (nvidia-smi failure).
    We still return ``gmu=0.92`` — it's the per-process utilization
    fraction, independent of how much VRAM the host actually has. The
    UI is expected to surface a separate warning about the GPU probe."""
    out = suggest_config(
        hf_repo="some/model",
        config={"max_position_embeddings": 4096},
        total_vram_bytes=0,
        filenames=[],
    )
    assert out.gpu_memory_utilization == 0.92


def test_disclaimer_text_warns_against_auto_apply():
    """The disclaimer string MUST signal 'starting points, never
    auto-applied' so a downstream consumer that auto-applies it ignores
    a contract red flag. Phrasing locked by the S3 plan."""
    assert "starting points" in DISCLAIMER_TEXT.lower()
    assert "never auto-applied" in DISCLAIMER_TEXT.lower()


def test_suggested_config_serializes_to_dict_with_disclaimer():
    """``SuggestedConfig.to_dict`` is what the route returns; the FE
    contract is that the dict always contains the ``disclaimer`` key
    so the wizard can render it next to the suggested values."""
    out = suggest_config(
        hf_repo="meta-llama/Llama-3.1-7B-Instruct",
        config={"max_position_embeddings": 131072},
        total_vram_bytes=_gpu_total_bytes_for(24 * 1024),
        filenames=["model.safetensors"],
    )
    d = out.to_dict()
    assert d["disclaimer"] == DISCLAIMER_TEXT
    assert d["gpu_memory_utilization"] == 0.92
    assert d["max_model_len"] == 131072
    assert "kv_cache_dtype" in d  # present (may be None) so FE can read uniformly
