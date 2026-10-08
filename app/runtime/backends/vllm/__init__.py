"""The vLLM backend.

Deliberately thin. Everything substantial lives in the three modules beside it
-- ``args`` (argv construction and the #211 bind-host policy), ``env`` (the
subprocess env and its allowlist), ``diagnostics`` (the log grammar) -- which
arrived here by ``git mv`` from ``app/runtime/`` and carry the comments
recording nine production incidents. This module is the adapter that presents
them as a ``Backend``; it holds no logic of its own, so reading a vLLM
behaviour question still lands you in the file that explains it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.runtime.backends import BackendCapabilities, LaunchPlan
from app.runtime.backends.vllm.args import build_vllm_args, engine_bind_host
from app.runtime.backends.vllm.diagnostics import diagnose_engine_log
from app.runtime.backends.vllm.env import (
    ALLOWED_ENV_EXACT,
    ALLOWED_ENV_PREFIXES,
    build_subprocess_env,
)
from app.runtime.backends.vllm.metrics import read as read_vllm_metrics

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.runtime.engine import EngineDriver

# argv[0] and its subcommand. This pair used to be hard-coded inside
# LocalSubprocessDriver.spawn, which is what made the driver vLLM-specific.
LAUNCH_HEAD: tuple[str, ...] = ("vllm", "serve")

CAPABILITIES = BackendCapabilities(
    name="vllm",
    display_name="vLLM",
    supports_tensor_parallel=True,
    supports_data_parallel=True,
    cached_tokens_reporting="usage",
    supports_pipeline_parallel=True,
    max_gpus=None,
    openai_paths=frozenset({"/v1/chat/completions", "/v1/completions", "/v1/models"}),
    health_path="/health",
    metrics_path="/metrics",
    # vLLM's V1 scheduler runs in priority mode (#173 part B); the proxy maps
    # vllm_priority = -warden_priority. A backend without a priority scheduler
    # advertises False and the proxy stops injecting the field.
    supports_request_priority=True,
    supports_lora=True,
    supports_vision=True,
    # vLLM takes ``--model <repo-id>`` and resolves the files itself, including
    # a GGUF row's ``repo:QUANT`` form. It never wants a path from us, so the
    # supervisor must not do the cache scan on its behalf.
    needs_local_model_path=False,
    # The ENGINE fact: vLLM publishes many versioned upstream images, so its
    # version CAN be chosen -- independent of how we launch it. Whether a given
    # deployment can honour that is VllmBackend.version_pin_available(driver).
    supports_version_pin=True,
    env_prefixes=ALLOWED_ENV_PREFIXES,
    env_exact=ALLOWED_ENV_EXACT,
    # None = the operator's --gpu-memory-utilization is the cap, because for
    # vLLM it genuinely is one: the engine reserves that fraction up front and
    # refuses to start if the weights plus its KV pool do not fit inside it.
    vram_cap_fraction=None,
)


class VllmBackend:
    capabilities = CAPABILITIES

    def version_pin_available(self, driver: EngineDriver | None) -> bool:
        """Can ``driver`` actually honour a per-model vLLM version pin?

        Takes the driver OBJECT, not its name -- see "Two members, two kinds of
        driver" in app/runtime/backends/__init__.py.

        True exactly when the driver can swap the engine container image. The
        in-container subprocess driver cannot -- its vLLM version is whatever
        the warden image baked in -- which is why `bonus` correctly reports
        False today via /api/system/engine, and must keep doing so.

        ``getattr(driver, "supports_engine_image", False)`` -- note the
        ``False``. ADVERTISING fails closed: an unknown driver, or ``None``,
        must not be told a pin is available, because a pin that is accepted and
        then silently discarded is the #177 bug. ENFORCEMENT in
        ``Supervisor.load`` keeps the opposite default (``True``) so a test
        stand-in driver is never wrongly blocked. The asymmetry is deliberate.
        """
        if isinstance(driver, str):
            # A driver NAME reaching here is the bug E shipped: getattr on a
            # str returns the default, so "docker" would quietly answer False
            # and disable the version selector on every docker deployment.
            # A wrong ANSWER is worse than a crash, because the subprocess case
            # returns the right value for the wrong reason and hides it. An
            # unknown driver OBJECT is a legitimate unknown and still answers
            # False; a string is a programming error and says so.
            raise TypeError(
                f"version_pin_available() takes an EngineDriver, not the name "
                f"{driver!r}. bind_host() is the one that takes a name."
            )
        return bool(getattr(driver, "supports_engine_image", False))

    def bind_host(self, driver_name: str) -> str:
        """Bind interface for ``driver_name`` (``settings.engine_driver``).

        Takes the driver NAME, not the object -- the mirror image of
        version_pin_available above, for the reasons in
        app/runtime/backends/__init__.py.
        """
        if driver_name is not None and not isinstance(driver_name, str):
            # Symmetric guard. A driver object here would compare unequal to
            # "docker" and silently return loopback -- safe under #211's
            # fail-narrow rule, but wrong: a docker engine would be unreachable
            # and the failure would surface 40 seconds later as a health
            # timeout instead of here.
            raise TypeError(
                f"bind_host() takes the driver NAME (settings.engine_driver), "
                f"not {type(driver_name).__name__}. version_pin_available() is "
                f"the one that takes the object."
            )
        return engine_bind_host(driver_name)

    def plan(
        self,
        model,
        *,
        port: int,
        bind_host: str,
        overrides: dict | None = None,
        # Accepted and IGNORED. vLLM takes a repo id and does its own cache
        # lookup, so it needs no resolved path -- but taking the argument is
        # what lets Supervisor.load have one call site rather than a branch on
        # the backend's name. See the Protocol's plan() docstring.
        resolved=None,
        hf_token: str = "",
        hf_cache_dir: str = "",
        engine_driver: str = "local",
        image: str | None = None,
    ) -> LaunchPlan:
        argv = list(LAUNCH_HEAD) + build_vllm_args(
            model, port=port, overrides=overrides, bind_host=bind_host
        )
        env = build_subprocess_env(
            model,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            engine_driver=engine_driver,
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
        return diagnose_engine_log(log_tail)

    def parse_metrics(self, body: str):
        """vLLM exposition text -> a neutral EngineReading, or None.

        The 31 ``vllm:`` names live in ``.metrics``; nothing under
        ``app/stats/`` knows any of them any more.
        """
        try:
            return read_vllm_metrics(body)
        except Exception:  # noqa: BLE001 - a failed parse is "no metrics"
            return None
