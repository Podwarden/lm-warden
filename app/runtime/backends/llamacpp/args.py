"""Build the ``llama-server`` argv from a ModelRow.

Flag spellings were checked against llama.cpp b10731's common/arg.cpp and are
re-checked on every test run against tests/fixtures/llamacpp/help.txt, which is
a real --help capture. If the two ever disagree, the capture is right.

Four flags are emitted unconditionally and each one is load-bearing:

``--port``   upstream prints "NOTICE: server default port will be changed to
             :9931 in a future release" when it listens on 8080. An implicit
             port is a future outage that would look like a health-probe bug.
``--alias``  without it the model id llama-server reports is the raw -m PATH.
             Exactly ONE alias is passed: the flag takes a comma-separated SET
             and reports the LEXICOGRAPHICALLY first, not the first typed.
``--metrics``without it GET /metrics is a 501 and every llama.cpp model shows a
             blank live-stats panel.
``--no-webui`` the engine's own port is unauthenticated -- /health is exempt
             even from --api-key -- so it must not additionally serve a web UI.

Four flags are deliberately NOT emitted. Briefly: ``--jinja`` because it is
already the server default and the meaningful flag is now --no-jinja;
``--tensor-split`` because llama.cpp's memory-proportional default is the better
answer on a heterogeneous box; ``--flash-attn`` and the KV cache-type flags
because they are real dials with no column, and inventing a policy for a
mainline default is what spec §2 forbids. All four remain reachable through
extra_args, which is appended last and therefore wins.

A fifth is not emitted either: no colour flag. Task 2's capture confirmed
``--log-colors`` defaults to ``auto`` and correctly detects that the
supervisor's log file descriptor is not a terminal, so the log arrives clean.
"""

from __future__ import annotations

import os


def engine_bind_host(driver_name: str) -> str:
    """Interface llama-server binds its HTTP server to, chosen by engine driver.

    ``driver_name`` is ``settings.engine_driver`` -- a NAME, matching B's
    disambiguated `Backend.bind_host(driver_name)` (`dcdd57e`). The member that
    takes a driver OBJECT is `version_pin_available`; they are deliberately
    spelled differently now, and the backend raises TypeError on the wrong kind.

    Identical policy to the vLLM backend's, and for an identical reason -- this
    is issue #211 restated for a second unauthenticated engine. llama-server's
    OpenAI API has no auth at all unless --api-key is passed, and its /health and
    /v1/health are registered as PUBLIC routes that skip even that check. So the
    bind host is a security boundary, not a connectivity knob.

    ``local`` -- the in-container subprocess driver, i.e. what production runs.
    The engine is a child process sharing the warden's own network namespace, so
    the control plane reaches it over loopback and 0.0.0.0 buys nothing except
    exposure: on the pod IP, ports 10000-10999 become a free, unmetered LLM API
    for every other pod in the cluster.

    ``docker`` -- each engine is a separate container with its own netns, so
    loopback inside it is unreachable from anywhere else; it genuinely needs
    0.0.0.0 on the engine's eth0.

    Any other driver name resolves to loopback. Deliberately: a future driver
    that needs a wider bind fails LOUDLY (the health probe times out at first
    load) instead of silently republishing the unauthenticated API, so the
    failure mode of forgetting to update this function is an outage, never a
    breach.

    ``VW_ENGINE_BIND_HOST`` still wins when set, so the operator keeps the
    escape hatch. An empty value counts as unset -- an exported-but-blank env
    var would otherwise produce ``--host ''`` and a dead engine. Read per call so
    the env is honoured without re-importing the module.
    """
    explicit = os.environ.get("VW_ENGINE_BIND_HOST", "").strip()
    if explicit:
        return explicit
    return "0.0.0.0" if driver_name == "docker" else "127.0.0.1"


def build_llamacpp_args(
    model,
    *,
    port: int,
    bind_host: str,
    resolved,
    overrides: dict | None = None,
) -> list[str]:
    """Construct the argv tail for llama-server (everything after argv[0])."""
    ov = overrides or {}
    if resolved is None or not getattr(resolved, "model_path", None):
        # The resolver deliberately does NOT raise for an unresolvable weights
        # file -- a vLLM GGUF row sets `filename` too and resolves it itself, so
        # refusing there would break rows that load fine today. llama.cpp is the
        # backend that cannot proceed, so llama.cpp is where it becomes an
        # error. Still before any process is spawned, and still inside the
        # supervisor's try/except that releases the GPU claim.
        where = getattr(resolved, "snapshot_dir", None) if resolved else None
        looked = f" Looked in {where}." if where else ""
        raise ValueError(
            "llama.cpp needs a resolved model_path: the row must pin a "
            "`filename` and the file must be in the model cache."
            + looked
            + " Pull the model, or pick a specific .gguf file on the row."
        )

    max_len = ov["max_model_len"] if "max_model_len" in ov else model.max_model_len
    max_num_seqs = ov.get("max_num_seqs", getattr(model, "max_num_seqs", None))
    n_gpu_layers = ov.get("n_gpu_layers", getattr(model, "n_gpu_layers", None))

    args: list[str] = [
        "--model",
        resolved.model_path,
        "--host",
        bind_host,
        "--port",
        str(port),
        "--alias",
        model.served_model_name,
    ]
    mmproj = getattr(resolved, "mmproj_path", None)
    if mmproj:
        args += ["--mmproj", mmproj]
    if max_len is not None:
        args += ["--ctx-size", str(max_len)]
    if max_num_seqs is not None:
        args += ["--parallel", str(max_num_seqs)]
    if n_gpu_layers is not None:
        args += ["--n-gpu-layers", str(n_gpu_layers)]

    # D4: the GPU COUNT is the invariant; which flag it becomes is ours to pick.
    # One GPU -> `none`, and --main-gpu 0 because CUDA_VISIBLE_DEVICES has
    # already renumbered the selected card to index 0. Several -> `layer`, the
    # upstream default: pipeline-parallel layer slices. NOT `row` (deprecated
    # upstream) and NOT `tensor` (experimental, needs flash attention on,
    # forbids KV quantization, disables --fit, and is unimplemented for a long
    # list of architectures). parallelism_strategy is not consulted at all --
    # llama.cpp has no non-experimental tensor-parallel mode, so honouring `tp`
    # would mean presenting it as a peer of vLLM's, which §9.3 forbids. The UI
    # hides that control for this backend rather than asking a question whose
    # answer is discarded.
    if len(model.gpu_indices) == 1:
        args += ["--split-mode", "none", "--main-gpu", "0"]
    else:
        args += ["--split-mode", "layer"]

    args += ["--metrics", "--no-webui"]
    args += list(getattr(model, "extra_args", []) or [])
    return args
