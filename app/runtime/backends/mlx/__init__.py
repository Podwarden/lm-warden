"""The MLX engine backend (Apple silicon, Metal).

Mainline ``mlx_lm.server`` only: no in-house server, no wrapper shim. ``plan()``
is pure and emits the bare console-script name as ``argv[0]`` (the llama.cpp
pattern), resolved at launch through a hard-set ``PATH`` -- so the
effective-argv preview matches what actually runs. The log grammar lives in
``diagnostics`` (p5b); this module is the adapter that presents it as a
``Backend``.

NOT registered in ``app/runtime/backends/registry.py`` yet. Metal design s10
landed the DEFAULT_BACKEND migration first (making the historical default
explicit) and the registry's profile split second; MLX registers only once both
are in, so an ``apple`` profile can default to it without re-pointing rows.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING, Any

from app.models.model_id_flags import model_id_flag_in, model_id_refusal
from app.runtime.backends import BackendCapabilities, LaunchPlan
from app.runtime.backends.vllm.env import filter_extra_env

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.runtime.backends.paths import ResolvedModelPaths
    from app.runtime.backends.vllm.diagnostics import EngineDiagnosis
    from app.runtime.engine import EngineDriver

# argv[0]. The mlx-lm console script; resolved at launch via the hard-set PATH
# below, the same "bare name + PATH" contract llama.cpp makes with
# llama-server. Keeping the name bare is what keeps plan() pure.
BINARY = "mlx_lm.server"


# The hard-set PATH. mlx-lm is a pip install into the warden's OWN venv (Metal
# design s3: python-build-standalone + a relocatable venv inside the signed
# bundle), so its console script sits next to the interpreter running us --
# that directory goes first. It is a constant of this process, read once at
# import, so plan() stays pure. /opt/homebrew/bin and /usr/local/bin are
# deliberately absent: a user-installed mlx_lm.server there would be a
# user-supplied, unsigned engine (s3 forbids those in v1). An empty
# sys.executable (embedded interpreter) contributes nothing, never a leading
# ":" that would put the working directory on PATH.


def engine_path(executable: str) -> str:
    """The hard-set PATH for the engine: the interpreter's own dir, then the fixed fallbacks."""
    return ":".join(
        p for p in ((os.path.dirname(executable) if executable else ""), "/usr/bin", "/bin") if p
    )


_VENV_BIN = os.path.dirname(sys.executable) if sys.executable else ""
ENGINE_PATH = engine_path(sys.executable)


def _model_dir(resolved: ResolvedModelPaths | None) -> str:
    """The directory mlx_lm.server loads: the resolved snapshot directory.

    mlx_lm.server takes a DIRECTORY of MLX weights (config, tokenizer,
    safetensors), never a single file, so ``model_path`` (the file pin a
    llama.cpp row carries) is no substitute. No snapshot means the repo is not
    in the model cache: refuse here, before any process is spawned and inside
    the supervisor's try/except that releases the GPU claim -- the same place
    llama.cpp refuses a missing file. Emitting ``--model None`` instead would
    fail as a TypeError at spawn, or worse, as a Hub download.
    """
    snapshot = getattr(resolved, "snapshot_dir", None) if resolved is not None else None
    if not snapshot:
        raise ValueError(
            "MLX needs a resolved model directory: the repo must be in the "
            "model cache. Pull the model first."
        )
    return str(snapshot)


CAPABILITIES = BackendCapabilities(
    name="mlx",
    display_name="MLX",
    # Apple silicon is one shared unified-memory pool, not N discrete cards.
    # max_gpus is a COUNT of cards the row may touch, and MLX sees exactly one
    # pool whatever the machine has, so 1 is the honest answer -- advertising
    # more would present it as a multi-card engine it is not.
    max_gpus=1,
    # No parallelism modes at all: weights and KV share one pool, so there is
    # nothing to split tensor- or pipeline- or data-ways.
    supports_tensor_parallel=False,
    supports_pipeline_parallel=False,
    supports_data_parallel=False,
    # mlx_lm.server has no priority scheduler; the warden's own admission
    # scheduler still delivers priority lanes, so the UI says so rather than
    # hiding the fact.
    supports_request_priority=False,
    supports_lora=False,
    supports_vision=False,
    # Decisive: without this the engine downloads weights the control plane
    # cannot see (Metal design s5 A hazard). We hand it a resolved local path.
    needs_local_model_path=True,
    # mlx_lm.server serves the standard OpenAI surface; /health is liveness
    # (see health_means_ready below), /metrics is absent (see metrics_path).
    openai_paths=frozenset({"/v1/chat/completions", "/v1/completions", "/v1/models"}),
    health_path="/health",
    # No /metrics endpoint (Metal design s5 C). Telemetry is proxy_observed;
    # nothing to scrape.
    metrics_path=None,
    # The ENGINE fact: mlx-lm has no per-version image to choose, so the
    # version cannot be pinned at all -- and version_pin_available(driver)
    # below is False for the same reason plus the missing resolver.
    supports_version_pin=False,
    env_prefixes=(),
    env_exact=frozenset(),
    # One shared pool the engine will use entirely; there is no
    # gpu_memory_utilization analogue to honour, so 1.0 is the honest cap.
    vram_cap_fraction=1.0,
    # /health is liveness, not readiness (Metal design s5 B): is_healthy()
    # reports the generation thread's liveness, 200 before any weights are
    # resident. Readiness is the warmup probe's job.
    health_means_ready=False,
    # No /metrics to scrape: the control plane builds its EngineReading from
    # what the proxy already observed (Metal design s5 C).
    telemetry="proxy_observed",
    # mlx_lm.server puts no cached_tokens on its responses; the cache fact
    # routes through the TTFT estimate instead, labelled as one.
    cached_tokens_reporting="none",
)


class MlxBackend:
    capabilities = CAPABILITIES

    def version_string(self) -> str | None:
        """The mlx-lm version in THIS install, from its package metadata.

        Reads the installed ``mlx-lm`` dist's version via
        ``importlib.metadata`` -- never ``import mlx``/``mlx_lm``, which would
        drag in the Metal runtime and is far too heavy for a metadata read.
        ``None`` where mlx-lm is not installed (a non-Apple build, the test
        suite). This is what makes MLX report a version on
        /api/system/backends instead of nothing.
        """
        try:
            return _dist_version("mlx-lm")
        except PackageNotFoundError:
            return None

    def bind_host(self, driver_name: str) -> str:
        """Interface this engine should bind, given ``driver_name`` (a NAME).

        The mirror image of version_pin_available below, which takes the driver
        OBJECT -- see the Protocol for why the two differ. mlx_lm.server's
        OpenAI server is unauthenticated, so the same #211 reasoning applies:
        the bind host is a security boundary. The local (subprocess) driver
        shares the warden's network namespace, so loopback; the docker driver
        needs 0.0.0.0 on the engine's own interface.
        """
        if driver_name is not None and not isinstance(driver_name, str):
            raise TypeError(
                f"bind_host() takes the driver NAME (settings.engine_driver), "
                f"not {type(driver_name).__name__}. version_pin_available() is "
                f"the one that takes the object."
            )
        return "0.0.0.0" if driver_name == "docker" else "127.0.0.1"

    def version_pin_available(self, driver: EngineDriver | None) -> bool:
        """Can THIS deployment honour a per-model MLX version pin?

        Takes the driver OBJECT, not its name -- see the Protocol for the
        asymmetry.

        The answer is False for EVERY driver. mlx-lm publishes no per-version
        image to choose (capabilities.supports_version_pin is False), and
        nothing in this build resolves an mlx-lm version to an image. Returning
        getattr(driver, "supports_engine_image", False) would be actively
        harmful: under a docker driver it would light a version selector whose
        selection flows into _resolve_engine_image() and comes back a vLLM
        image. Refusing is a correctness argument, not a cautious one.
        """
        if isinstance(driver, str):
            raise TypeError(
                f"version_pin_available() takes an EngineDriver, not the name "
                f"{driver!r}. bind_host() is the one that takes a name."
            )
        return False

    def plan(
        self,
        model: Any,
        *,
        port: int,
        bind_host: str,
        overrides: dict[str, Any] | None = None,
        resolved: ResolvedModelPaths | None = None,
        hf_token: str = "",
        hf_cache_dir: str = "",
        engine_driver: str = "local",
        image: str | None = None,
        load_max_s: float | None = None,
    ) -> LaunchPlan:
        # mlx_lm.server wants a DIRECTORY of MLX weights, not a file.
        model_dir = _model_dir(resolved)

        # extra_args may not re-point --model: the engine's model id is the
        # LOCAL SNAPSHOT PATH plan() itself passes (Metal design s5 A), and the
        # proxy routes and forwards by engine_model_id -- that same path. A row
        # that overrides --model in extra_args would serve an id the proxy never
        # asks for, so refuse at load time, before any argv is built (OVERSEER
        # model-id decision, planner iter 32; the detector is shared with the
        # other backends and the write-time validator).
        extra_args = list(getattr(model, "extra_args", []) or [])
        flagged = model_id_flag_in("mlx", extra_args)
        if flagged is not None:
            raise ValueError(model_id_refusal("mlx", flagged))

        ov = overrides or {}
        max_num_seqs = ov.get("max_num_seqs", getattr(model, "max_num_seqs", None))

        argv = [
            BINARY,
            "--model",
            model_dir,
            "--host",
            bind_host,
            "--port",
            str(port),
        ]
        # max_num_seqs is the real OOM lever on a 16 GB machine; mlx_lm's flag
        # for it is --decode-concurrency. max_model_len has NO mlx_lm flag and
        # is advisory (it feeds fit math, not argv) -- deliberately emitted as
        # nothing here.
        if max_num_seqs is not None:
            argv += ["--decode-concurrency", str(max_num_seqs)]
        argv += extra_args

        # extra_env goes through the shared filter: a hard-locked key
        # (LD_PRELOAD, DYLD_INSERT_LIBRARIES, PYTHONPATH, PATH, ...) is REFUSED
        # with ValueError, never silently dropped; anything else must match
        # this backend's allowlist, which is empty today, so it is dropped.
        extra_env = getattr(model, "extra_env", {}) or {}
        env = filter_extra_env(
            extra_env,
            prefixes=self.capabilities.env_prefixes,
            exact=self.capabilities.env_exact,
        )
        # Hard-set AFTER the filtered merge so they always win.
        # mlx_lm is a Python process; the supervisor must see its stdout
        # line-by-line, not block-buffered.
        env["PYTHONUNBUFFERED"] = "1"
        # Hard-set PATH so the bare console-script name above resolves at
        # launch -- the contract that keeps plan() pure. See ENGINE_PATH.
        env["PATH"] = ENGINE_PATH

        return LaunchPlan(
            argv=argv,
            env=env,
            image=image,
            port=port,
            gpu_indices=list(model.gpu_indices),
        )

    def engine_model_id(self, model: Any, resolved: ResolvedModelPaths | None = None) -> str:
        """The id this engine serves: what the proxy must put in ``body['model']``.

        mlx_lm.server has no ``--served-model-name`` and registers the model by
        the path it loaded, so the id is the LOCAL MODEL PATH -- exactly the
        string plan() passes to ``--model``. A request body carrying any other
        model id would reach the engine as an unknown name and trigger a Hub
        download (Metal design s5 A). Pure: no I/O, mirroring the other
        backends' method. mlx_lm.server compares the request's id to the
        loaded one as a RAW string, so this must be byte-identical to the
        ``--model`` value: both come from ``_model_dir``. Without a resolved
        snapshot there is no pure answer (the row carries no path), so it
        raises ValueError rather than return an id the engine would treat as a
        model to download.
        """
        return _model_dir(resolved)

    def health_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}{self.capabilities.health_path}"

    def diagnose(self, log_tail: str) -> EngineDiagnosis | None:
        from app.runtime.backends.mlx.diagnostics import diagnose_mlx_log

        return diagnose_mlx_log(log_tail)

    def parse_metrics(self, body: str) -> None:
        # No /metrics endpoint (Metal design s5 C): nothing to parse.
        return None
