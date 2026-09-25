"""Built-in model presets — battle-tested configs users can pick from a dropdown.

Each preset has all model-tuning knobs (dtype, max_model_len, tp size, gpu_mem_util,
trust_remote_code, extra_env). The user still chooses gpu_indices and served_model_name
at create time — the preset prefills everything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineSpec:
    channel: str
    vllm_version: str
    image: str | None = None


@dataclass(frozen=True)
class ModelTemplate:
    id: str
    label: str
    hf_repo: str
    hf_revision: str
    dtype: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    trust_remote_code: bool
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    engine: EngineSpec | None = None
    source: str = "builtin"


_GPT_OSS_20B = ModelTemplate(
    id="gpt-oss-20b",
    label="GPT-OSS 20B (battle-tested, TP=2, bfloat16)",
    hf_repo="openai/gpt-oss-20b",
    hf_revision="main",
    dtype="bfloat16",
    max_model_len=32000,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
    extra_args=[],
    engine=EngineSpec(channel="cuda-stable", vllm_version="0.20.0"),
    extra_env={
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
        "VLLM_USE_V2_BLOCK_MANAGER": "1",
        "VLLM_ENABLE_CHUNKED_PREFILL": "1",
        "VLLM_CHUNKED_PREFILL_ENABLED": "1",
        # #91 workaround: disable async output processor on gpt_oss family.
        # The async path concurrently re-enters `vllm/entrypoints/openai/
        # harmony_utils.py::get_encoding()`, which returns a single
        # module-level `_harmony_encoding` reference; the underlying
        # `openai_harmony` Rust extension wraps a `RefCell` and the second
        # in-flight chat-completions call surfaces
        # `RuntimeError: Already borrowed` (HTTP 500). Setting this to "0"
        # forces the synchronous output path, which serializes harmony
        # access per-engine. Upstream still ships the singleton in v0.21.0
        # — TODO: remove once vLLM lands per-request isolation in
        # harmony_utils.py.
        "VLLM_USE_ASYNC_OUTPUT_PROC": "0",
        "VLLM_ASYNC_OUTPUT_PROC_NUM_WORKERS": "2",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
        "VLLM_MAX_NUM_SEQS": "64",
        "VLLM_SWAP_SPACE": "0",
        "VLLM_ENABLE_PREFIX_CACHING": "0",
        "VLLM_SCHEDULER_DELAY_FACTOR": "0.0",
        "VLLM_DISABLE_CUSTOM_ALL_REDUCE": "0",
        "VLLM_ENFORCE_EAGER": "0",
        "VLLM_ENABLE_LORA": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "NCCL_DEBUG": "WARN",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_IB_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
    },
)


# Verified on a production deployment (4x A4000-class TP=4, vLLM 0.26.0 in-container driver,
# 2026-09-23): full 256k context, FP8 KV cache, prefix caching, reasoning and
# XML tool calls. No engine pin — the in-container driver refuses pins, and
# the config has no known version sensitivity beyond needing a vLLM new
# enough to ship the qwen3_xml tool parser.
_QWEN38_27B_FP8 = ModelTemplate(
    id="qwen3.8-27b-fp8",
    label="Qwen3.8 27B FP8 (battle-tested, TP=4, 256k ctx, tools + reasoning)",
    hf_repo="Qwen/Qwen3.8-27B-FP8",
    hf_revision="main",
    dtype="auto",
    max_model_len=262144,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,
    trust_remote_code=False,
    extra_args=[
        "--enable-prefix-caching",
        "--kv-cache-dtype", "fp8",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
    ],
)


_TEMPLATES: dict[str, ModelTemplate] = {
    _GPT_OSS_20B.id: _GPT_OSS_20B,
    _QWEN38_27B_FP8.id: _QWEN38_27B_FP8,
}


def list_templates() -> list[ModelTemplate]:
    return list(_TEMPLATES.values())


def get_template(template_id: str) -> ModelTemplate | None:
    return _TEMPLATES.get(template_id)


def list_builtin_templates() -> list[ModelTemplate]:
    return list(_TEMPLATES.values())


def get_builtin_template(template_id: str) -> ModelTemplate | None:
    return _TEMPLATES.get(template_id)


def template_to_dict(t: ModelTemplate) -> dict:
    return {
        "id": t.id,
        "label": t.label,
        "hf_repo": t.hf_repo,
        "hf_revision": t.hf_revision,
        "dtype": t.dtype,
        "max_model_len": t.max_model_len,
        "tensor_parallel_size": t.tensor_parallel_size,
        "gpu_memory_utilization": t.gpu_memory_utilization,
        "trust_remote_code": t.trust_remote_code,
        "extra_args": list(t.extra_args),
        "extra_env": dict(t.extra_env),
        "engine": (
            None if t.engine is None
            else {"channel": t.engine.channel,
                  "vllm_version": t.engine.vllm_version,
                  "image": t.engine.image}
        ),
        "source": t.source,
    }


def template_from_dict(d: dict) -> ModelTemplate:
    eng = d.get("engine")
    return ModelTemplate(
        id=d["id"], label=d["label"], hf_repo=d["hf_repo"],
        hf_revision=d.get("hf_revision", "main"), dtype=d["dtype"],
        max_model_len=d["max_model_len"],
        tensor_parallel_size=d["tensor_parallel_size"],
        gpu_memory_utilization=d["gpu_memory_utilization"],
        trust_remote_code=d["trust_remote_code"],
        extra_args=list(d.get("extra_args", [])),
        extra_env=dict(d.get("extra_env", {})),
        engine=None if eng is None else EngineSpec(
            channel=eng["channel"], vllm_version=eng["vllm_version"],
            image=eng.get("image"),
        ),
        source=d.get("source", "user"),
    )
