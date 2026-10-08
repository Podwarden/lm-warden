"""The subprocess env for llama-server, and this backend's extra_env allowlist.

Deliberately much smaller than the vLLM backend's. llama-server is a C++ binary
that reads its model from a path we resolved: it needs no Python runtime
settings, no HuggingFace cache root, and -- see below -- no HuggingFace token.

WHY ``LLAMA_`` IS NOT AN ALLOWED PREFIX
---------------------------------------
Every llama.cpp CLI flag has an ``LLAMA_ARG_*`` environment equivalent, wired by
``set_env(...)`` on each option in common/arg.cpp. That includes LLAMA_ARG_HOST,
LLAMA_ARG_MODEL, LLAMA_ARG_PORT, LLAMA_ARG_ALIAS, LLAMA_ARG_ENDPOINT_METRICS and
LLAMA_API_KEY. An allowlisted ``LLAMA_`` prefix would therefore let a model row's
extra_env re-bind the unauthenticated OpenAI API to 0.0.0.0 (issue #211, from
the other direction), point --model at a different file, move the port out from
under the supervisor's health probe, or silently switch metrics off.

So the prefix list is ``GGML_`` only. GGML_* are backend-tuning knobs
(GGML_CUDA_ENABLE_UNIFIED_MEMORY, GGML_CUDA_P2P) with no equivalent power, and
they are the escape hatch an operator genuinely wants. This is the "narrowing,
not widening" property design spec §6.6 asks of a per-backend allowlist.

The five most dangerous LLAMA_* names are ALSO added to the global
HARD_LOCKED_ENV_KEYS so they are refused loudly at write time rather than
dropped silently -- the same treatment, and the same reasoning, the vLLM backend
gives HF_HUB_CACHE.
"""

from __future__ import annotations

from app.runtime.backends.vllm.env import filter_extra_env

# GGML_ only -- see the module docstring. Adding LLAMA_ here is a security
# regression, not a convenience.
ALLOWED_ENV_PREFIXES: tuple[str, ...] = ("GGML_",)
ALLOWED_ENV_EXACT: frozenset[str] = frozenset({"CUDA_MODULE_LOADING"})


def build_llamacpp_env(
    model,
    *,
    allowed_prefixes: tuple[str, ...] = ALLOWED_ENV_PREFIXES,
    allowed_exact: frozenset[str] = ALLOWED_ENV_EXACT,
) -> dict[str, str]:
    """Construct the env dict for a llama-server subprocess.

    Returns a closed dict; the caller passes it as the env= kwarg to
    asyncio.create_subprocess_exec, which uses ONLY this env (no inheritance).

    No HF_HUB_CACHE, no HUGGING_FACE_HUB_TOKEN, no HF_TOKEN. llama-server only
    touches HuggingFace on the ``-hf`` download path, which this backend never
    emits -- app/runtime/backends/paths.py resolved the file before we got here.
    Handing a credential to a process with no use for it widens the blast radius
    of a compromise for nothing.
    """
    if not model.gpu_indices:
        raise ValueError("gpu_indices must be non-empty")

    extra_env = getattr(model, "extra_env", {}) or {}
    filtered = filter_extra_env(extra_env, prefixes=allowed_prefixes, exact=allowed_exact)

    env = {
        # 2026-05-08, restated for a second engine. ggml enumerates CUDA devices
        # in CUDA's order; without PCI_BUS_ID, NVML may reorder by SM count and
        # gpu_indices stops meaning the cards the operator picked. bonus mixes an
        # RTX A4000 (sm_86) with a Quadro RTX 5000 (sm_75), so this is live.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    # Allowed extra_env first so the hard-set keys below always win.
    env.update(filtered)

    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in model.gpu_indices)
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    return env
