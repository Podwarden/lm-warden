"""Map an engine boot/runtime failure to a category + next-combo hint (#162).

Pure string heuristics over the engine's stderr tail. Deliberately small and
order-sensitive: the first matching rule wins, most-specific first. The
suggestion is human-facing guidance for the trial-and-error loop, not an
executable directive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierResult:
    category: str
    suggestion: str


_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "cuda_arch_unsupported",
        re.compile(r"no kernel image is available|sm_\d+|unsupported gpu architecture", re.I),
        "GPU arch unsupported by this build — try a cuda-legacy channel or an "
        "older vLLM version compiled for this compute capability.",
    ),
    (
        "oom",
        re.compile(r"out of memory|CUDA out of memory|OutOfMemoryError", re.I),
        "Out of VRAM — lower gpu_memory_utilization or max_model_len, raise "
        "tensor_parallel_size, or pick a smaller/quantized checkpoint.",
    ),
    (
        "quant_unsupported",
        re.compile(
            r"quantization.*not supported|unsupported quant|awq.*not supported|gptq.*not supported",
            re.I,
        ),
        "Quantization not supported on this engine/GPU — switch quant scheme "
        "(awq<->gptq), use an fp16 checkpoint, or a newer vLLM version.",
    ),
    (
        "version_mismatch",
        re.compile(
            r"requires torch|version mismatch|incompatible.*version|ImportError.*vllm", re.I
        ),
        "Library/version mismatch — pin a vLLM version whose torch/cuda matches "
        "the engine image (try the adjacent cuda channel).",
    ),
]


def classify(error_text: str) -> ClassifierResult:
    text = error_text or ""
    for category, pattern, suggestion in _RULES:
        if pattern.search(text):
            return ClassifierResult(category=category, suggestion=suggestion)
    return ClassifierResult(
        category="unknown",
        suggestion="No known signature matched — inspect the engine log tail "
        "and try the next-lower vLLM version on the same channel.",
    )
