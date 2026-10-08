"""Turn a llama-server log tail into an actionable, operator-facing error.

The same job app/runtime/backends/vllm/diagnostics.py does for vLLM, and the
same discipline: match on STABLE TOKENS rather than exact phrasing, because the
wording drifts between builds; capture numbers opportunistically; return None on
no match so the caller keeps its generic message.

ORDER IS LOAD-BEARING, and for the same reason it is in the vLLM module.
An out-of-memory log routinely also contains context-sized numbers -- ggml
prints the size of the buffer it could not allocate, and llama_context prints
n_ctx_seq alongside -- so a context rule checked first would classify an OOM as
a context overflow and tell the operator to lower max_model_len. That is a loop
with no exit: the weights, not the context, are what did not fit. OOM is checked
FIRST and its message deliberately never mentions the context.

The three levers that actually differ:

  weights do not fit   -> fewer offloaded layers (n_gpu_layers), a smaller
                          quant, or more VRAM. NOT a smaller context.
  context does not fit -> a smaller ctx-size, i.e. max_model_len. This one DOES
                          respond to the obvious knob.
  wrong file           -> point at shard one, or at a file this build can load.

There is deliberately NO auto-retry here: this path only REPORTS.

WHERE THE PATTERNS COME FROM. Every rule is derived from a real captured log in
tests/fixtures/llamacpp/ except the CUDA spellings inside _OOM_RE, which are
read from the same pinned b10731 tree (ggml/src/ggml-cuda/ggml-cuda.cu) because
the only NVIDIA host in scope was read-only when the fixtures were taken -- see
that directory's README. The CPU and CUDA spellings share the backend-neutral
``alloc_tensor_range`` / ``failed to allocate ... buffer`` anchors, so the rule
is exercised by the fixture either way.

IMPORT DIRECTION, stated so it is a decision and not an accident: EngineDiagnosis
is a shared type that happens to live under ``vllm/`` because sub-project B moved
it there with ``git mv``, and that file's comments are pinned verbatim by
tests/unit/runtime/backends/test_incident_comments.py. Importing it from here is
the wrong direction on paper; moving it is a real risk to nine incident records
for zero functional gain, so C imports and files the lift as an open question.
"""

from __future__ import annotations

import re

from app.runtime.backends.vllm.diagnostics import EngineDiagnosis

# --- 1. Out of memory. Checked FIRST -- see the module docstring. ------------
# The CPU spellings are in tests/fixtures/llamacpp/log_oom.txt:
#   ggml_aligned_malloc: insufficient memory (attempted to allocate 21488.50 MB)
#   ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size ...
#   alloc_tensor_range: failed to allocate CPU buffer of size ...
#   llama_init_from_model: failed to initialize the context: failed to allocate
#     buffer for kv cache
# The CUDA spellings come from ggml/src/ggml-cuda/ggml-cuda.cu at b10731:
#   ... cudaMalloc failed: out of memory
#   alloc_tensor_range: failed to allocate CUDA0 buffer of size ...
#   unable to allocate CUDA0 buffer
_OOM_RE = re.compile(
    r"cudaMalloc\s+failed:\s*out\s+of\s+memory"
    r"|insufficient\s+memory\s*\(attempted\s+to\s+allocate"
    r"|failed\s+to\s+allocate\s+\S+\s+buffer"
    r"|failed\s+to\s+allocate\s+buffer\s+(of\s+size|for\s+)"
    r"|unable\s+to\s+allocate\s+\S+\s+buffer",
    re.IGNORECASE,
)

# --- 2. Context larger than the model was trained for, or than fits. ---------
# The capped form is what server-context.cpp prints and what the captured
# log_ctx_capped.txt contains verbatim:
#   the slot context (8192) exceeds the training context of the model (2048) - capping
# The warn-only form comes from llama-context.cpp and appears alongside it:
#   n_ctx_seq (8192) > n_ctx_train (2048) -- possible training context overflow
_CTX_CAP_RE = re.compile(
    r"slot\s+context\s+\((\d+)\)\s+exceeds\s+the\s+training\s+context\s+of\s+"
    r"the\s+model\s+\((\d+)\)",
    re.IGNORECASE,
)
_CTX_TRAIN_RE = re.compile(r"n_ctx_seq\s+\((\d+)\)\s*>\s*n_ctx_train\s+\((\d+)\)", re.IGNORECASE)

# --- 3. Architecture this build cannot load. --------------------------------
# From log_unknown_arch.txt:
#   error loading model: unknown model architecture: 'notarealarch'
_ARCH_RE = re.compile(r"unknown\s+model\s+architecture:\s*'([^']*)'", re.IGNORECASE)

# --- 4. Pointed at the wrong member of a shard set. -------------------------
# From log_wrong_shard.txt:
#   illegal split file idx: 1 (file: ...), model must be loaded with the first split
_SHARD_RE = re.compile(
    r"illegal\s+split\s+file\s+idx:\s*(\d+).*?must\s+be\s+loaded\s+with\s+" r"the\s+first\s+split",
    re.IGNORECASE | re.DOTALL,
)

# --- 5. Generic load failure, the catch-all. --------------------------------
_LOAD_FAIL_RE = re.compile(r"failed\s+to\s+load\s+model|error\s+loading\s+model", re.IGNORECASE)


def diagnose_llamacpp_log(text: str) -> EngineDiagnosis | None:
    """Best-effort diagnosis of a llama-server failure. None when unrecognised."""
    if not text or not text.strip():
        return None

    if _OOM_RE.search(text):
        return EngineDiagnosis(
            message=(
                "llama.cpp ran out of memory placing the model. The weights and "
                "KV cache themselves did not fit, so reduce n_gpu_layers to keep "
                "some layers on the CPU (slower, but it will run), pick a smaller "
                "quantisation, or give the model more or larger GPUs."
            ),
            recommended_max_model_len=None,
        )

    m = _SHARD_RE.search(text)
    if m:
        return EngineDiagnosis(
            message=(
                "This GGUF is one part of a multi-part set and llama.cpp was "
                f"pointed at part {m.group(1)}. Set the model's filename to the "
                "FIRST shard (the one ending -00001-of-000NN.gguf); llama.cpp "
                "finds the rest itself, provided every shard is in the same "
                "directory under its original name."
            ),
            recommended_max_model_len=None,
        )

    m = _ARCH_RE.search(text)
    if m:
        return EngineDiagnosis(
            message=(
                f"This build of llama.cpp does not know the model architecture "
                f"'{m.group(1)}'. The file itself is readable, so this is a "
                "version gap rather than a corrupt download: the architecture "
                "needs a llama.cpp new enough to have it, and llama.cpp's "
                "version is fixed by the warden image."
            ),
            recommended_max_model_len=None,
        )

    m = _CTX_CAP_RE.search(text) or _CTX_TRAIN_RE.search(text)
    if m:
        trained = int(m.group(2))
        return EngineDiagnosis(
            message=(
                f"The requested context is larger than this model was trained "
                f"for ({trained} tokens). llama.cpp caps it, and output quality "
                "past the training context is unreliable. Set max_model_len to "
                f"{trained} or below."
            ),
            recommended_max_model_len=trained,
        )

    if _LOAD_FAIL_RE.search(text):
        return EngineDiagnosis(
            message=(
                "llama.cpp could not load the model file. Check that the pulled "
                ".gguf is complete (re-pull if a download was interrupted) and "
                "that the row's filename points at a weights file rather than a "
                "projector or an imatrix file."
            ),
            recommended_max_model_len=None,
        )

    return None
