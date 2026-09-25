"""Build subprocess env for vLLM. THIS FIXES THE 2026-05-08 BUG.

CUDA_VISIBLE_DEVICES is derived from model.gpu_indices (a DB column populated by the
user's wizard/CRUD selection). It is NEVER inherited from the parent process. The
parent vllm-warden container is launched with all GPUs visible (the launcher gives
the container `--gpus all` so the supervisor can dispatch any GPU); each per-model
subprocess MUST have CUDA_VISIBLE_DEVICES restricted to exactly that model's
gpu_indices, in the order specified, so vLLM's logical device 0 == gpu_indices[0].

extra_env keys are filtered through an allowlist and cannot override CUDA_VISIBLE_DEVICES,
HF_HUB_CACHE, the HF token, or PATH.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine shared memory (#210)
# ---------------------------------------------------------------------------
# vLLM's tensor-parallel workers talk to each other over POSIX shared memory:
# the shm_broadcast MessageQueue ring plus the custom-all-reduce CUDA-IPC
# handles. The ring is sized VLLM_MQ_MAX_CHUNK_BYTES_MB * n_chunks, and tmpfs
# backs its pages lazily -- ftruncate() and mmap() both SUCCEED even when the
# ring cannot possibly fit in /dev/shm, and the process only finds out when a
# write touches a page tmpfs then refuses to allocate. That fault arrives as
# SIGBUS, which no Python handler can turn into an exception; the engine just
# dies. That is the #210 crash: TP=4 engines taking an uncatchable SIGBUS
# inside shm_broadcast.enqueue 78s-3min into serving.
#
# The docker driver sidesteps this by giving the engine container its own
# large /dev/shm (docker_socket.ENGINE_SHM_SIZE, default 16g, plus
# ipc_mode=host). The local-subprocess driver has no equivalent lever: the
# engine is a child of the warden process and inherits the warden POD's
# /dev/shm, which on Kubernetes is the 64 MiB default unless the pod spec
# asks for an emptyDir with `medium: Memory`.
ENGINE_SHM_PATH = "/dev/shm"

# Threshold for the startup warning below. 2 GiB is the floor at which a TP>1
# engine is safe here, not a comfortable target: a TP=4 engine has been
# observed holding ~1.1 GB of ShmRingBuffer segments, on top of one
# shm_broadcast ring per message queue (160 MiB each at vLLM's stock 16 MB
# chunk size). 2 GiB clears that with headroom while sitting far enough above
# the 64 MiB container/pod default that the warning can only mean "nobody
# configured shared memory for this deployment".
ENGINE_SHM_MIN_BYTES = 2 * 1024**3

# There is deliberately NO VLLM_MQ_MAX_CHUNK_BYTES_MB default here. Capping
# the chunk size at 4 MB (a 40 MiB ring that fits inside a 64 MiB pod) was
# the stop-gap that held production up between the first SIGBUS and the real
# fix, but it costs throughput -- payloads above the chunk limit fall back to
# vLLM's slower zmq overflow copy path, which multimodal requests hit
# routinely -- and it never reaches the ShmRingBuffer segments at all, so it
# narrowed the failure window rather than closing it. The fix is a real
# /dev/shm on the pod (podwarden-core #2417: compose `shm_size:` on the api
# service renders a `medium: Memory` emptyDir), and the warning below is what
# tells an operator when that is missing. A deployment that cannot get real
# shared memory can still set the cap per model via extra_env; it is not a
# hard-locked key.

# Keys that extra_env is never permitted to override — security/incident lockdown.
#
# PYTHONUNBUFFERED is locked even though no ALLOWED_ENV_PREFIXES currently
# accepts the bare "PYTHON" prefix (so today an operator-supplied value would
# be silently dropped by the allowlist). The lock is defence-in-depth against
# a future hand adding a "PYTHON_" prefix for legitimate escape hatches like
# PYTHONFAULTHANDLER / PYTHONHASHSEED — that change would otherwise let
# PYTHONUNBUFFERED=0 flow through and silently undo the v2026.05.15.5 fix
# (block-buffered subprocess stdout swallowing fast-crash tracebacks).
HARD_LOCKED_ENV_KEYS: frozenset[str] = frozenset({
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    # HF_HUB_CACHE is the model-cache target the engine actually reads/writes;
    # it is set below from settings.hf_cache_dir. HF_HOME is NO LONGER set by
    # this builder (see the 2026-06-15 ENOSPC fix) but stays locked so extra_env
    # can't reintroduce it and shadow HF_HUB_CACHE with a different layout/root.
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
    "PATH",
    "PYTHONUNBUFFERED",
    # Spec §2, the mainline-only invariant. Neither key is on any allowlist
    # today, so locking them changes nothing an operator can currently do --
    # which is exactly why it is cheap. They are the two vectors by which
    # extra_env could reintroduce runtime patching through the back door
    # (PYTHONPATH shadowing a module, LD_PRELOAD interposing a symbol), and
    # the invariant says a launch is argv + env and nothing else. The lock
    # documents WHY they can never be added to ALLOWED_ENV_PREFIXES.
    "PYTHONPATH",
    "LD_PRELOAD",
    # Sub-project C. Every llama.cpp CLI flag has an LLAMA_ARG_* env twin, so
    # these five would each breach a boundary if extra_env could set them:
    # HOST re-binds the unauthenticated OpenAI API (#211 from the env side),
    # MODEL serves different weights under the operator's chosen name, PORT
    # moves the engine out from under the supervisor's health probe,
    # ENDPOINT_METRICS blanks the live-stats panel, and LLAMA_CACHE redirects
    # the model cache (the 2026-06-15 ENOSPC class of failure). None is on any
    # allowlist -- llamacpp's prefixes are ("GGML_",) -- so locking them changes
    # nothing an operator can do today; it converts a silent drop into a loud
    # refusal at write time and records WHY they can never be allowlisted.
    "LLAMA_ARG_HOST",
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_PORT",
    "LLAMA_ARG_ENDPOINT_METRICS",
    "LLAMA_CACHE",
})

# extra_env keys are accepted iff they start with one of these prefixes OR are
# exactly CUDA_MODULE_LOADING.  Everything else is silently dropped.
ALLOWED_ENV_PREFIXES: tuple[str, ...] = (
    "VLLM_",
    "TRITON_",
    "NCCL_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
)
ALLOWED_ENV_EXACT: frozenset[str] = frozenset({
    "CUDA_MODULE_LOADING",
    # The escape hatch the HARD_LOCKED comment above anticipated, added by
    # name rather than by a "PYTHON" prefix so PYTHONUNBUFFERED stays
    # unreachable. Defaulted to "1" below; an operator can set "0" to opt out.
    "PYTHONFAULTHANDLER",
    # How long EngineCore waits for a TP worker to answer execute_model /
    # sample_tokens before declaring the engine dead. vLLM's default is 300s.
    #
    # That default is tuned for "a slow batch", not for "a worker is wedged".
    # On 2026-08-18 a TP worker stopped answering mid-request and the engine
    # sat there for the full five minutes -- serving nothing, returning
    # nothing -- before finally erroring out. With automatic restart in place
    # (watchdog.restart_crashed_models) a SHORTER timeout is strictly better:
    # it converts a five-minute silent hang into a fast detect-and-restart.
    #
    # Deliberately NOT defaulted here. Lowering it globally would abort
    # legitimately long batches on a slow GPU, so it stays an operator dial
    # per deployment.
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
})


def filter_extra_env(
    extra_env: dict[str, str],
    *,
    prefixes: tuple[str, ...] = ALLOWED_ENV_PREFIXES,
    exact: frozenset[str] = ALLOWED_ENV_EXACT,
) -> dict[str, str]:
    """Validate and filter extra_env before merging into the subprocess env.

    Raises ValueError if any hard-locked key is present in extra_env.
    Silently drops keys that don't match the allowlist.

    ``prefixes`` / ``exact`` come from the backend's BackendCapabilities. They
    default to vLLM's for callers that predate the backend axis. Making them
    per-backend NARROWS what is accepted -- a llama.cpp row today would accept
    inert VLLM_* keys -- and the hard-locked set below is checked FIRST and is
    global, so no per-backend allowlist can unlock one.
    """
    for key in extra_env:
        if key in HARD_LOCKED_ENV_KEYS:
            raise ValueError(
                f"extra_env key '{key}' is hard-locked and cannot be overridden"
            )

    filtered = {
        key: value
        for key, value in extra_env.items()
        if any(key.startswith(prefix) for prefix in prefixes) or key in exact
    }
    dropped = sorted(set(extra_env) - set(filtered))
    if dropped:
        log.info(
            "env_builder: dropped %d extra_env key(s) not on allowlist: %s",
            len(dropped),
            dropped,
        )
    return filtered


def dev_shm_bytes(path: str = ENGINE_SHM_PATH) -> int | None:
    """Total size of the filesystem mounted at ``path``, in bytes.

    Returns ``None`` when the size cannot be read at all — the path does not
    exist (a slim image, a non-Linux dev box), or ``statvfs`` is unavailable.
    Callers treat ``None`` as "unknown", never as "too small": a missing
    /dev/shm is not evidence of an undersized one.
    """
    try:
        st = os.statvfs(path)
    except (OSError, ValueError, AttributeError):
        return None
    # f_frsize is the fundamental block size; f_blocks counts those blocks.
    return int(st.f_blocks) * int(st.f_frsize)


def warn_if_shm_undersized(
    engine_driver: str,
    *,
    path: str = ENGINE_SHM_PATH,
    minimum: int = ENGINE_SHM_MIN_BYTES,
) -> None:
    """Log a startup warning when the local driver has too little /dev/shm (#210).

    Only the local-subprocess driver is checked: it hands the engine whatever
    /dev/shm the warden process itself has. The docker driver gives each engine
    container its own (``docker_socket.ENGINE_SHM_SIZE`` / ``ipc_mode=host``),
    so the warden's own figure says nothing about it.

    This exists so an operator reads about the problem in the startup log
    instead of discovering it as an uncatchable SIGBUS minutes into serving.
    """
    if engine_driver != "local":
        return
    total = dev_shm_bytes(path)
    if total is None:
        log.warning(
            "engine shm: could not read the size of %s; vLLM tensor-parallel "
            "workers need at least %.0f MiB of shared memory (#210)",
            path,
            minimum / 1024**2,
        )
        return
    if total >= minimum:
        return
    log.warning(
        "engine shm: %s is only %.0f MiB, below the %.0f MiB minimum for "
        "tensor-parallel engines (#210). vLLM's shm_broadcast ring and "
        "custom-all-reduce segments are mmap'd lazily, so an undersized "
        "/dev/shm does not fail at startup — it kills the engine with an "
        "uncatchable SIGBUS in shm_broadcast.enqueue minutes into serving. "
        "The minimum comes from a TP=4 engine's observed ~1.1 GB of "
        "ShmRingBuffer segments plus one 160 MiB message-queue ring. The fix "
        "is a real /dev/shm on the warden pod: an emptyDir with medium: Memory "
        "and an explicit sizeLimit (compose `shm_size:` on the api service, "
        "podwarden-core #2417). As a stop-gap only, "
        "VLLM_MQ_MAX_CHUNK_BYTES_MB=4 in a model's extra_env shrinks the ring "
        "to 40 MiB at a throughput cost on large/multimodal payloads.",
        path,
        total / 1024**2,
        minimum / 1024**2,
    )


def build_subprocess_env(
    model,
    *,
    hf_token: str,
    hf_cache_dir: str,
    engine_driver: str = "local",
    allowed_prefixes: tuple[str, ...] = ALLOWED_ENV_PREFIXES,
    allowed_exact: frozenset[str] = ALLOWED_ENV_EXACT,
) -> dict[str, str]:
    """Construct the env dict for a vLLM subprocess.

    Returns a closed dict. Caller passes this dict as the env= kwarg to
    asyncio.create_subprocess_exec, which uses ONLY this env (no inheritance).

    ``hf_cache_dir`` is ``settings.hf_cache_dir`` — the SAME directory the pull
    task hands to ``snapshot_download(cache_dir=...)``. The engine reads its
    model cache from there via ``HF_HUB_CACHE`` (whose on-disk layout,
    ``<root>/models--org--name/``, matches ``cache_dir``). Pointing the engine
    anywhere else makes it re-download the model — the 2026-06-15 ENOSPC crash,
    where the engine wrote to the tiny ``/data`` PVC instead of the model-cache
    volume and filled it to 0 bytes, taking SQLite down with it.

    ``engine_driver`` is ``settings.engine_driver``. The env is the same under
    both drivers today; the parameter is kept so a driver-specific default can
    be added without touching every caller (the #210 chunk-size cap lived here
    briefly and was removed once the pod got a real /dev/shm).

    extra_env from the model row is merged in after validation: allowed keys
    override defaults (e.g. VLLM_LOGGING_LEVEL=DEBUG overrides INFO), but
    hard-locked keys (CUDA_VISIBLE_DEVICES, HF_HUB_CACHE, HUGGING_FACE_HUB_TOKEN,
    PATH) cannot be overridden.
    """
    if not model.gpu_indices:
        raise ValueError("gpu_indices must be non-empty")
    if model.tensor_parallel_size != len(model.gpu_indices):
        raise ValueError(
            f"tensor_parallel_size ({model.tensor_parallel_size}) must equal "
            f"len(gpu_indices) ({len(model.gpu_indices)})"
        )

    extra_env = getattr(model, "extra_env", {}) or {}
    filtered = filter_extra_env(
        extra_env, prefixes=allowed_prefixes, exact=allowed_exact
    )

    env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        # Pin device ordering to PCI bus so vLLM's logical 0..N matches
        # gpu_indices on heterogeneous-GPU hosts (e.g. a host that mixes a
        # Quadro RTX 4000 + A4000). Without this, NVML may reorder by SM count
        # and break gpu_indices semantics.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # v2026.05.15.5 — Python defaults to block-buffered stdout when
        # stdout is a file fd (the supervisor's behaviour — see
        # supervisor.py opens the log with O_WRONLY|O_CREAT|O_APPEND
        # and passes it as stdout=). On a fast crash (rc=1 within <1s)
        # the subprocess can exit before the buffered output is
        # flushed, so the operator sees nothing in the log even though
        # Python printed the traceback. PYTHONUNBUFFERED=1 forces
        # line-buffered stdio and guarantees flush on exit — defence in
        # depth alongside the routes_logs open-or-create fix.
        "PYTHONUNBUFFERED": "1",
        # 2026-08-18 — EngineCore died 20 times in one night leaving NOTHING:
        # no traceback, no exit code, no core file (the host's core_pattern
        # pipes to apport, which does not exist in this image, so every core
        # is discarded). The only trace was the TP workers noticing "Parent
        # process exited". faulthandler turns a fatal signal (SIGSEGV, SIGBUS,
        # SIGILL, SIGABRT, SIGFPE) into a printed C-level Python stack on
        # stderr, which the supervisor already captures into the engine log.
        # Cost is one-time signal-handler installation; there is no reason a
        # production engine should ever die silently.
        "PYTHONFAULTHANDLER": "1",
    }
    # Apply allowed extra_env first so hard-locked keys below always win.
    env.update(filtered)

    # Hard-locked keys are set last and cannot be overridden.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in model.gpu_indices)
    # Point the engine's HF model cache at settings.hf_cache_dir — the same
    # root the pull task downloads into. HF_HUB_CACHE (not HF_HOME) so the
    # layout matches snapshot_download(cache_dir=...): <root>/models--org--name/.
    env["HF_HUB_CACHE"] = hf_cache_dir
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["HF_TOKEN"] = hf_token
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

    return env
