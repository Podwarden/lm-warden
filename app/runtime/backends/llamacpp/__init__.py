"""The llama.cpp backend.

Deliberately thin, mirroring VllmBackend: the substance lives in ``args`` (argv
and the #211 bind policy), ``env`` (the subprocess env and its narrow
allowlist), ``diagnostics`` (the log grammar) and ``metrics`` (the ``llamacpp:``
dialect). This module is the adapter that presents them as a Backend.

llama-server is baked into the warden image and launched by the existing
in-container subprocess driver -- decision D2. Its version is therefore pinned by
the warden image, exactly as vLLM's is; bumping it is a warden release, not a
per-model choice, until sub-projects D and F land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.runtime.backends import BackendCapabilities, LaunchPlan
from app.runtime.backends.llamacpp.args import build_llamacpp_args, engine_bind_host
from app.runtime.backends.llamacpp.env import (
    ALLOWED_ENV_EXACT,
    ALLOWED_ENV_PREFIXES,
    build_llamacpp_env,
)

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.runtime.engine import EngineDriver

# argv[0]. Task 13 puts the binary on PATH at /usr/local/bin/llama-server.
BINARY = "llama-server"

CAPABILITIES = BackendCapabilities(
    name="llamacpp",
    display_name="llama.cpp",
    # False, and this is the honest answer rather than the flattering one.
    # llama.cpp's default multi-GPU mode is a LAYER split -- pipeline
    # parallelism. Its `row` mode is deprecated upstream and its `tensor` mode is
    # experimental with hard constraints. Advertising tensor parallelism here
    # would put llama.cpp forward as a drop-in peer of vLLM for "a model too big
    # for one card", which design spec §9.3 explicitly forbids.
    supports_tensor_parallel=False,
    supports_data_parallel=False,
    cached_tokens_reporting="usage",
    supports_pipeline_parallel=True,
    max_gpus=None,
    openai_paths=frozenset({"/v1/chat/completions", "/v1/completions", "/v1/models"}),
    health_path="/health",
    # Only answers once --metrics is on the argv; without it, 501. args.py emits
    # it unconditionally.
    metrics_path="/metrics",
    # llama-server has no priority scheduler. Its request parser ignores unknown
    # fields rather than rejecting them, so the proxy's injected `priority`
    # would be harmlessly discarded -- but advertising True would be a lie, and
    # the proxy uses this flag to stop re-serialising a body for no reason.
    supports_request_priority=False,
    # llama.cpp has LoRA support; this backend does not wire it (no column, no
    # UI), so the honest advertisement is False until it does.
    supports_lora=False,
    # Via --mmproj and the mtmd stack. The projector is a separate GGUF, which
    # is why models.mmproj_filename exists.
    supports_vision=True,
    # llama-server takes ``-m <abs path>``. app/runtime/backends/paths.py is the
    # bridge from a repo id to that path, and this is the flag that asks for it.
    needs_local_model_path=True,
    # The ENGINE fact, driver-invariant: llama.cpp publishes per-build tagged
    # server images upstream (ghcr.io/ggml-org/llama.cpp:server-cuda13-bNNNNN),
    # exactly as vLLM publishes vllm/vllm-openai:vX.Y.Z, so the engine's version
    # CAN be chosen in principle. True, matching VllmBackend.
    #
    # Whether any deployment can HONOUR that is version_pin_available(driver)
    # below, and for llama.cpp the answer is False everywhere today -- for a
    # reason that is neither the engine's nor the driver's. See its docstring.
    supports_version_pin=True,
    env_prefixes=ALLOWED_ENV_PREFIXES,
    env_exact=ALLOWED_ENV_EXACT,
    # llama-server has no gpu_memory_utilization analogue -- verified against
    # `llama-server --help` in the shipped b10731 image, which offers only
    # `-ngl/--n-gpu-layers` (a layer count). Applying vLLM's 0.9 to a llama.cpp
    # row hid 10% of the card from the verdict and reported "won't fit" for
    # models that fit.
    vram_cap_fraction=1.0,
)


class LlamaCppBackend:
    capabilities = CAPABILITIES

    # ------------------------------------------------------------------
    # TWO MEMBERS, TWO KINDS OF DRIVER -- and the base implementation ENFORCES
    # the difference. Mirror it here; do not merely match the signatures.
    #
    #   bind_host(driver_name: str)                      <- the driver NAME
    #   version_pin_available(driver: EngineDriver|None) <- the driver OBJECT
    #
    # version_pin_available is bool(getattr(driver, "supports_engine_image",
    # False)). Hand it a NAME and getattr("docker", ...) returns the default, so
    # a docker deployment quietly answers "you cannot pin a version" and the
    # selector is disabled everywhere. That mistake FAILS CLOSED -- nothing
    # raises, nothing logs -- and it is CAMOUFLAGED, because the subprocess case
    # then returns the right answer for the wrong reason. Every subprocess
    # fixture still passes. Sub-project E shipped exactly this into review.
    #
    # VllmBackend therefore raises TypeError on the wrong KIND in both
    # directions, each message naming the other member. Fail-closed is preserved
    # by kind rather than by value: an unknown driver OBJECT is a legitimate
    # unknown and still answers False; a STRING is a programming error.
    #
    # This backend needs the guards MORE than vLLM does, not less:
    # version_pin_available returns False unconditionally here, so no return
    # value could ever reveal a mistyped argument. Without the raise, a caller
    # passing a name would be wrong and would never find out.
    # ------------------------------------------------------------------

    def bind_host(self, driver_name: str) -> str:
        """Bind interface for ``driver_name`` (``settings.engine_driver``).

        Takes the driver NAME, not the object -- the mirror image of
        version_pin_available below.
        """
        if driver_name is not None and not isinstance(driver_name, str):
            raise TypeError(
                f"bind_host() takes the driver NAME (settings.engine_driver), "
                f"not {type(driver_name).__name__}. version_pin_available() is "
                f"the one that takes the object."
            )
        return engine_bind_host(driver_name)

    def version_pin_available(self, driver: EngineDriver | None) -> bool:
        """Can THIS deployment honour a per-model llama.cpp version pin?

        Takes the driver OBJECT, not its name -- see the block comment above.

        The answer is False for EVERY driver, and not for the reason a reader
        expects. It is not that the subprocess driver cannot swap an image
        (true, but only half of it), and not that llama.cpp is unpinnable (it is
        not -- capabilities.supports_version_pin is True, because upstream
        publishes per-build tagged server images). It is that **nothing in this
        build can resolve a llama.cpp version to an image**:
        app/runtime/backends/vllm/images.py maps (channel, version) ->
        vllm/vllm-openai:vX.Y.Z and has no llama.cpp counterpart. C adds none.

        So returning getattr(driver, "supports_engine_image", False) here would
        be actively harmful rather than merely optimistic. Under a docker or k8s
        driver it would enable a version selector whose selection flows into
        _resolve_engine_image() and comes back a **vLLM** image -- launching
        vLLM under the operator's llama.cpp model name. Refusing is a
        CORRECTNESS argument, not a cautious one.

        Sub-project F must add a llama.cpp image resolver before this method may
        read its ``driver`` argument at all; at that point it should read it
        exactly as VllmBackend's does.
        """
        if isinstance(driver, str):
            raise TypeError(
                f"version_pin_available() takes an EngineDriver, not the name "
                f"{driver!r}. bind_host() is the one that takes a name."
            )
        return False

    def plan(
        self,
        model,
        *,
        port: int,
        bind_host: str,
        overrides: dict | None = None,
        resolved=None,
        hf_token: str = "",
        hf_cache_dir: str = "",
        engine_driver: str = "local",
        image: str | None = None,
    ) -> LaunchPlan:
        argv = [BINARY] + build_llamacpp_args(
            model,
            port=port,
            bind_host=bind_host,
            resolved=resolved,
            overrides=overrides,
        )
        env = build_llamacpp_env(
            model,
            allowed_prefixes=self.capabilities.env_prefixes,
            allowed_exact=self.capabilities.env_exact,
        )
        return LaunchPlan(
            argv=argv,
            env=env,
            image=image,
            port=port,
            gpu_indices=list(model.gpu_indices),
        )

    def health_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}{self.capabilities.health_path}"

    def diagnose(self, log_tail: str):
        from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

        return diagnose_llamacpp_log(log_tail)

    def parse_metrics(self, body: str):
        from app.runtime.backends.llamacpp.metrics import read

        return read(body)
