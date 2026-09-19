"""Build the `vllm serve` argv from a ModelRow.

The `overrides` parameter lets callers apply a load-config (quantization,
tensor-parallel size, gpu_memory_utilization, max_model_len, max_num_seqs)
in memory without mutating the models row — useful for ad-hoc reload
flows. When `overrides=None`, behaviour matches the row defaults.
"""
from __future__ import annotations

import os
import re

# vLLM 0.20.0 requires GGUF models to be addressed as ``repo_id:quant_type``
# (e.g. ``unsloth/Qwen3.6-27B-GGUF:Q5_K_M``). #85 (v17.13) started shipping
# per-file GGUF downloads via the ``filename`` column, but the launcher kept
# emitting ``--model <hf_repo>`` only — every GGUF deployment died at vllm
# subprocess startup with rc=1. We recover the quant tag from the filename
# (``...-Q5_K_M.gguf``). The pattern allows extended variants like
# ``UD-Q4_K_XL`` by anchoring on ``-Q`` + a digit + arbitrary
# letters/digits/underscores up to ``.gguf``. When the filename is missing or
# the regex doesn't match we fall back to the bare hf_repo — same behaviour as
# pre-#85, so safetensors / non-quantized rows are unaffected. See issue #100.
_GGUF_QUANT_RE = re.compile(r"-(Q\d[_A-Za-z0-9]*)\.gguf$", re.IGNORECASE)


def engine_bind_host(driver: str) -> str:
    """Interface vLLM binds its HTTP server to, chosen by engine driver.

    The engine's own OpenAI server is COMPLETELY UNAUTHENTICATED: whatever
    can open a TCP connection to it gets ``/v1/chat/completions`` directly,
    behind the back of ``require_bearer`` (app/proxy/auth.py), the priority
    scheduler and every byte of token accounting.
    So the bind host is a security boundary, not a connectivity knob, and it
    has to be as narrow as the driver allows.

    ``local`` — the in-container subprocess driver, and the default engine
    driver (app/config.py) i.e. what production actually runs (a Kubernetes
    pod). The engine is a child process sharing the warden's own network
    namespace, so the control plane reaches it over loopback
    (``LocalSubprocessDriver.engine_host`` returns ``127.0.0.1``) and
    ``0.0.0.0`` buys nothing except exposure: on the pod IP, ports
    10000-10999 become a free, unmetered LLM API for every other pod in the
    cluster. Loopback it is. This is issue #211; the pre-#211 default was
    ``0.0.0.0`` for every driver, and the docstring here used to call the
    local driver "legacy" and frame loopback as an opt-in, which is exactly
    backwards and is what made the hole look deliberate.

    ``docker`` — the docker-socket driver runs each engine as a SEPARATE
    container with its own netns, so loopback inside that container is not
    addressable from anywhere else. It genuinely needs ``0.0.0.0`` on the
    engine's eth0: neither the api container (which dials the engine by its
    docker DNS name) nor docker-proxy's published port can reach an
    unbound-on-eth0 engine — docker-proxy forwards to the container's eth0,
    gets RST, the supervisor health probe times out and the row flips to
    ``failed`` despite ``Application startup complete.`` in the engine log
    (that was #180). Containment for this driver belongs at the docker
    network / firewall layer, not here.

    Any other driver name resolves to loopback, same as ``local``. That is
    deliberate: a future driver that needs a wider bind fails LOUDLY (health
    probe times out at first load) instead of silently republishing the
    unauthenticated API, so the failure mode of forgetting to update this
    function is an outage, never a breach.

    ``VW_ENGINE_BIND_HOST`` still wins when set, so an operator keeps the
    escape hatch (e.g. binding one specific pod IP) — but it is no longer
    what decides the safe default. An empty value counts as unset: an
    exported-but-blank env var would otherwise produce ``--host ''`` and a
    dead engine. Read per-call so the env is honoured without re-importing
    the module."""
    explicit = os.environ.get("VW_ENGINE_BIND_HOST", "").strip()
    if explicit:
        return explicit
    return "0.0.0.0" if driver == "docker" else "127.0.0.1"


# Override keys recognised by build_vllm_args. Any other key is a bug
# upstream (the supervisor builds the dict) and is ignored silently to
# stay forward-compatible with future load-config dimensions.
_OVERRIDE_KEYS = frozenset({
    "quantization",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
})


def build_vllm_args(
    model,
    *,
    port: int,
    overrides: dict | None = None,
    driver: str = "local",
    bind_host: str | None = None,
) -> list[str]:
    """Construct the argv vector for `vllm serve <args>`.

    `driver` is the active engine driver name (``settings.engine_driver``)
    and only affects ``--host`` — see :func:`engine_bind_host`. It defaults
    to ``"local"``, i.e. the LOOPBACK bind, on purpose: a caller that forgets
    to pass it gets an engine that is merely unreachable from off-host, not
    one that silently republishes the unauthenticated vLLM API on every
    interface (#211). Fail towards the narrow bind, always.

    `overrides` may contain any subset of:
      - quantization           → emit --quantization <value>
      - tensor_parallel_size   → replace --tensor-parallel-size
      - gpu_memory_utilization → replace --gpu-memory-utilization
      - max_model_len          → replace --max-model-len (omit if None)
      - max_num_seqs           → emit --max-num-seqs <value>

    Overrides are read-only against the model row; the caller decides what
    to pass and is responsible for never persisting this to the DB.
    """
    ov = overrides or {}

    tp = ov.get("tensor_parallel_size", model.tensor_parallel_size)
    gpu_mem = ov.get("gpu_memory_utilization", model.gpu_memory_utilization)
    # `max_model_len` may be explicitly None either on the row or as an
    # override — treat both as "let vLLM pick the default" by omitting the flag.
    max_len = ov["max_model_len"] if "max_model_len" in ov else model.max_model_len
    # `quantization` lives only as an override + DB column; fall back to the
    # row's `quantization` attr if present (added by migration 0011) so the
    # legacy code path stays None-clean when the column hasn't been set.
    quantization = ov.get("quantization", getattr(model, "quantization", None))
    # `max_num_seqs` likewise — DB column from 0011, override slot in v2.
    max_num_seqs = ov.get("max_num_seqs", getattr(model, "max_num_seqs", None))

    # `parallelism_strategy` (#88) — wizard's tp/pp/auto choice from migration
    # 0014. ``auto`` and ``tp`` both emit ``--tensor-parallel-size`` (legacy
    # behaviour); ``pp`` swaps to ``--pipeline-parallel-size``. N is identical
    # in either case because ``ModelCreate._tp_consistent`` validates
    # ``tensor_parallel_size == len(gpu_indices)`` — there is one parallelism
    # dimension wide, only the flag name differs. Single-host PP is fine on
    # vLLM (CTO-decided in #82 plan); we do NOT block at builder level.
    # Legacy ``ModelRow`` instances (no column from migration 0014) decode with
    # ``parallelism_strategy='auto'`` default so this branch is safe pre-#85.
    strategy = getattr(model, "parallelism_strategy", "auto")
    parallelism_flag = (
        "--pipeline-parallel-size" if strategy == "pp" else "--tensor-parallel-size"
    )

    # GGUF: append ``:quant_type`` extracted from ``filename`` (#100). The
    # ``.lower().endswith(".gguf")`` guard skips safetensors rows even if some
    # future caller populates ``filename`` on them (e.g. partial downloads).
    model_arg = model.hf_repo
    filename = getattr(model, "filename", None)
    if filename and filename.lower().endswith(".gguf"):
        m = _GGUF_QUANT_RE.search(filename)
        if m:
            model_arg = f"{model.hf_repo}:{m.group(1)}"

    args: list[str] = [
        "--model", model_arg,
        "--host", bind_host if bind_host is not None else engine_bind_host(driver),
        "--port", str(port),
        "--served-model-name", model.served_model_name,
        parallelism_flag, str(tp),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    # dtype / max_model_len are optional on the model row — vLLM picks safe
    # defaults (auto / model-config max) when omitted. Emitting "--dtype None"
    # passes None into asyncio.create_subprocess_exec, which raises
    # "expected str, bytes or os.PathLike object, not NoneType" — the subprocess
    # never actually starts and the failure surfaces as last_error on the model
    # row rather than as a vLLM log line. Only forward these flags when set.
    if model.dtype:
        args += ["--dtype", model.dtype]
    if max_len is not None:
        args += ["--max-model-len", str(max_len)]
    if quantization:
        args += ["--quantization", str(quantization)]
    if max_num_seqs is not None:
        args += ["--max-num-seqs", str(max_num_seqs)]
    if model.hf_revision:
        args += ["--revision", model.hf_revision]
    # #106: GGUF repos that omit ``config.json`` (common for unsloth republishes)
    # require ``--hf-config-path <original_repo>`` so vLLM can find a config to
    # load. ``--tokenizer`` covers the same upstream-vs-quant split for the
    # tokenizer. Both columns are NULL on legacy / non-GGUF rows; the
    # ``getattr`` keeps stand-in test rows (no migration 0015 column) working.
    hf_config_repo = getattr(model, "hf_config_repo", None)
    if hf_config_repo:
        args += ["--hf-config-path", hf_config_repo]
    tokenizer_repo = getattr(model, "tokenizer_repo", None)
    if tokenizer_repo:
        args += ["--tokenizer", tokenizer_repo]
    # #173 part B — run vLLM's V1 scheduler in priority mode so the engine's
    # own waiting queue is ordered by the per-request ``priority`` field the
    # proxy injects (see app/proxy/routes.py). With all-equal priorities this
    # policy is identical to FCFS, so it's a safe unconditional default; a user
    # who wants plain FCFS can override by putting ``--scheduling-policy fcfs``
    # in ``extra_args`` (appended last → wins in argparse).
    args += ["--scheduling-policy", "priority"]
    extra_args = list(getattr(model, "extra_args", []) or [])
    args += extra_args
    return args
