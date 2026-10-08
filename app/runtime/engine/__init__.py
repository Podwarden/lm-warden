"""Engine driver contract: how the vLLM engine is spawned/observed/killed.

Supervisor owns lifecycle state, locking, GPU ownership and exit-watchers.
A driver owns only the three engine primitives below. This lets the engine
run as an in-container subprocess (LocalSubprocessDriver, default) or as a
sibling container via the host Docker socket (DockerSocketDriver) without
the Supervisor caring which. Implements #160.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EngineSpec:
    """Everything a driver needs to start one engine. Built by Supervisor
    from a model row + the backend's LaunchPlan; drivers stay model-agnostic
    AND backend-agnostic."""

    model_id: str
    model_arg: str  # the --model value (repo or repo:quant)
    argv: list[str]  # FULL argv, including argv[0]. Before sub-project
    # B this was `args` -- the post-binary tail -- and
    # each driver knew the binary. Now the backend
    # owns it (app/runtime/backends/).
    env: dict[str, str]  # process/container env
    port: int  # host port the engine listens on
    image: str | None = None  # engine container image (docker driver only)
    gpu_indices: list[int] = field(default_factory=list)
    # Which backend produced `argv`. Drivers may use it for naming and labels.
    # Sub-project B does NOT change any existing name that could use it: the
    # container name prefix and the docker label are identity, and changing
    # identity in a byte-identical refactor is out of scope (see the rename,
    # sub-project A). Sub-project C makes the container name backend-aware so a
    # vLLM and a llama.cpp engine for one model id cannot collide.
    backend: str = "vllm"


@runtime_checkable
class EngineHandle(Protocol):
    """Live reference to one running engine."""

    @property
    def pid(self) -> int | None:
        """Host PID of the engine's main process for GPU attribution.
        None if not yet known / not applicable."""

    @property
    def returncode(self) -> int | None:
        """Exit code once exited, else None."""

    async def wait(self) -> int:
        """Block until the engine exits, return its exit code."""


@runtime_checkable
class EngineDriver(Protocol):
    # True if the driver honors ``EngineSpec.image`` (i.e. it can swap the
    # engine container image to a pinned vLLM version); False if the engine
    # version is fixed by the warden image itself (the in-container
    # subprocess driver). Read defensively elsewhere via
    # ``getattr(driver, "supports_engine_image", True)`` so unknown/test
    # stand-in drivers default to capable and are never wrongly blocked.
    supports_engine_image: bool

    async def spawn(self, spec: EngineSpec) -> EngineHandle:
        """Start the engine described by spec and return its handle."""

    async def terminate(self, handle: EngineHandle, *, grace_s: float) -> None:
        """Graceful stop (SIGTERM/`docker stop`); SIGKILL after grace_s."""

    def engine_host(self, model_id: str) -> str:
        """Host the control-plane uses to reach this engine's HTTP API
        (health probe, warmup, client proxy). For an in-container subprocess
        this is loopback; for a sibling container it is the engine's
        on-network DNS name. Synchronous — callers read it off the hot path."""
