"""Pure heuristic that backs ``GET /api/models/{id}/suggest-config``.

Per the S3 plan (#113), this module returns STARTING POINTS for the load
configuration — it is NEVER auto-applied. The FE wizard renders the
suggestions alongside the row's current values; the operator decides
whether to copy them across.

Heuristics today (deliberately conservative; expand in S4+ as we learn):

* ``gpu_memory_utilization`` — fixed at ``0.92`` of available VRAM. The
  remainder leaves room for the activation tensors + the KV cache vLLM
  allocates dynamically. Same default used by the upstream vLLM examples.
* ``max_model_len`` — pulled from the model's ``config.max_position_embeddings``
  when present. Returns ``None`` (not a guess) when the config is
  missing — the row default applies and the wizard surfaces a warning.
* ``kv_cache_dtype`` — recommended as ``'fp8'`` ONLY for AWQ-quantized
  models. AWQ weights are int4 in storage but dequantized to bf16 at
  matmul time, so the KV cache dominates VRAM at long context; cutting
  KV cache to fp8 buys ~2× headroom with negligible quality loss in
  practice. Other quant families (GPTQ, GGUF, plain fp16) keep the
  default bf16 KV cache. AWQ detection is "name marker in hf_repo OR
  filename" plus a ``quantization_config.quant_method == 'awq'`` check.

The endpoint payload ALWAYS includes the ``disclaimer`` field so a
downstream consumer that auto-applies the values ignores a contract
red flag.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# Phrasing locked by the S3 plan: "starting points, never auto-applied".
DISCLAIMER_TEXT: str = (
    "These suggestions are starting points only and are never auto-applied. "
    "Review against your workload, edit as needed, and Save to persist."
)

# Default per-process VRAM utilization fraction. Independent of total
# VRAM — vLLM uses it to size the KV cache pool inside its own process,
# not as a host-wide budget.
_DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92

# Re-used from ``app/models/discovery.py`` (#82). Detects ``AWQ`` /
# ``GPTQ`` / ``FP16`` / ``INT8`` / ``INT4`` markers surrounded by ``.``,
# ``-`` or ``_``. We only flag AWQ for fp8-KV-cache (#113); the other
# tags are exposed via discovery but don't change KV cache today.
_AWQ_MARKER_RE = re.compile(
    r"[.\-_]AWQ(?=\.|$|[._-])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SuggestedConfig:
    """Suggested-config blob returned by ``GET /api/models/{id}/suggest-config``.

    Fields mirror the LOADABLE attributes on ``ModelRow`` (subset of what
    ``_PATCHABLE_MODEL_FIELDS`` allows). Any field whose suggestion would
    be a guess is ``None`` rather than fabricated.
    """

    gpu_memory_utilization: float
    max_model_len: int | None
    kv_cache_dtype: str | None
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        # ``asdict`` would happily serialize ``None`` — that's the
        # contract the FE relies on (key always present).
        return asdict(self)


def _is_awq_model(
    hf_repo: str,
    config: dict[str, Any] | None,
    filenames: list[str],
) -> bool:
    """Return True if the model looks AWQ-quantized.

    Three signals, any one is sufficient:

    * ``quantization_config.quant_method == 'awq'`` in the config.json
      (the authoritative signal when present).
    * ``hf_repo`` contains an ``AWQ`` marker.
    * Any weights filename contains an ``AWQ`` marker.

    Detection is broad on purpose — false positives only cost the user a
    suggested ``kv_cache_dtype=fp8`` they can ignore; false negatives
    leave AWQ models on the default bf16 KV cache which is the #113
    bug we're fixing.
    """
    if config is not None:
        quant = config.get("quantization_config")
        if isinstance(quant, dict):
            method = str(quant.get("quant_method") or "").lower()
            if method == "awq":
                return True
    if _AWQ_MARKER_RE.search(hf_repo or ""):
        return True
    for fn in filenames or ():
        if _AWQ_MARKER_RE.search(fn):
            return True
    return False


def suggest_config(
    *,
    hf_repo: str,
    config: dict[str, Any] | None,
    total_vram_bytes: int,  # noqa: ARG001  — reserved for future heuristics
    filenames: list[str],
) -> SuggestedConfig:
    """Compute a starting-point config blob for the given model.

    Inputs:
    * ``hf_repo``: HF repo id (e.g. ``meta-llama/Llama-3.1-8B``).
    * ``config``: parsed ``config.json`` dict, or ``None`` if absent
      (e.g. GGUF-only repos with no transformers metadata).
    * ``total_vram_bytes``: sum of ``memory_total`` across the model's
      selected GPUs. Currently informational only — preserved in the
      signature so we can grow the heuristic (e.g. tensor-parallel
      sizing) in S4+ without a contract break.
    * ``filenames``: list of filenames present in the repo (for marker
      detection).

    Returns a :class:`SuggestedConfig`. Use :meth:`SuggestedConfig.to_dict`
    when wiring up the route response.
    """
    max_model_len: int | None = None
    if config is not None:
        mpe = config.get("max_position_embeddings")
        # Some configs ship ``max_position_embeddings`` as a string; coerce.
        # Reject zero / negative / non-integer values rather than emit a guess.
        if isinstance(mpe, int) and mpe > 0:
            max_model_len = mpe
        elif isinstance(mpe, str):
            try:
                v = int(mpe)
            except ValueError:
                v = 0
            if v > 0:
                max_model_len = v

    kv_cache_dtype: str | None = None
    if _is_awq_model(hf_repo, config, filenames):
        kv_cache_dtype = "fp8"

    return SuggestedConfig(
        gpu_memory_utilization=_DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        disclaimer=DISCLAIMER_TEXT,
    )
