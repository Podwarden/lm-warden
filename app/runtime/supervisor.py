import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path

import httpx

from app.runtime.backends import registry
from app.runtime.backends.paths import resolve_model_paths
from app.runtime.backends.vllm.images import resolve_image
from app.runtime.engine import EngineSpec
from app.runtime.engine.local_subprocess import LocalSubprocessDriver
from app.runtime.gpu_ownership import GpuOwnership
from app.runtime.variants import Variant, runtime_facts, variant_of

UNLOAD_GRACE_SECONDS = 30.0


def _driver_engine_host(driver, model_id: str) -> str:
    """Host the control-plane reaches a driver's engine on. Drivers that
    predate the ``engine_host`` protocol method (or third-party stand-ins
    in tests) fall back to loopback — the historical in-container default."""
    fn = getattr(driver, "engine_host", None)
    if fn is None:
        return "127.0.0.1"
    return fn(model_id)


def _resolve_engine_image(model) -> str | None:
    """Engine image for a model's optional engine axis (#161-min).

    Returns None for legacy models with no engine axis so the driver uses
    its own default (the in-container path is unaffected). An explicit
    ``engine_image`` pin wins; otherwise ``(engine_channel,
    engine_vllm_version)`` resolves to an upstream tag. Read defensively
    via getattr because the DB columns land in P3/#162."""
    engine_image = getattr(model, "engine_image", None)
    if engine_image:
        return engine_image
    channel = getattr(model, "engine_channel", None)
    vllm_version = getattr(model, "engine_vllm_version", None)
    if channel and vllm_version:
        return resolve_image(channel, vllm_version)
    return None

# #99 — Default ceiling on how long ``wait_for_health`` waits for the
# vLLM ``/health`` endpoint to return 200 after a load. Surfaced as a
# module-level constant so callers (settings layer, tests) can reference
# the single source of truth instead of parroting the magic number. The
# production load runner overrides this via ``settings.load_timeout_s``;
# this default applies when callers (e.g. ad-hoc scripts, the
# integration suite) do not pass ``timeout_s`` explicitly.
DEFAULT_HEALTH_TIMEOUT_S: float = 600.0


class ModelState(str, Enum):
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    UNLOADING = "unloading"


class EnginePinUnsupported(Exception):
    """Raised by load() when a model carries an engine image/version pin
    that the active driver cannot honor (e.g. the in-container subprocess
    driver, whose vLLM version is fixed by the warden image). Surfaced to
    the operator instead of silently launching the wrong engine version."""


class UnloadRefused(Exception):
    """Raised when unload() is called on a model whose supervisor state
    is not READY and the caller did not pass ``force=True``.

    The exception message names the current state so the HTTP layer can
    translate to a 409 with a useful body.
    """

    def __init__(self, model_id: str, state: ModelState) -> None:
        self.model_id = model_id
        self.state = state
        super().__init__(
            f"refusing to unload model {model_id!r}: state is {state.name}, "
            f"not READY (pass force=True to override)"
        )


class Supervisor:
    def __init__(self, settings, *, driver=None) -> None:
        self.settings = settings
        self.gpus = GpuOwnership()
        self._driver = driver or LocalSubprocessDriver(
            log_dir=str(Path(settings.data_dir) / "logs")
        )
        # model_id -> EngineHandle (driver-owned live engine reference).
        self._handles: dict[str, object] = {}
        self._ports: dict[str, int] = {}
        # model_id -> host the control-plane reaches the engine on. Driver-
        # owned (loopback for the subprocess driver, the engine container's
        # DNS name for the docker driver). Populated at spawn alongside the
        # port; consumed by the health probe, warmup and the client proxy.
        self._hosts: dict[str, str] = {}
        # Per-model overrides dict (the kwarg passed to ``load``). Tracks
        # the LIVE configuration of each running model so callers can
        # snapshot it before reloading and restore it later. ``None`` means
        # "row defaults" (load was called with overrides=None); a dict means
        # those keys override row defaults.
        # Populated by ``load()``; cleared by ``unload()`` / ``_watch_exit``.
        self._overrides: dict[str, dict | None] = {}
        # model_id -> the VARIANT the running engine was launched as
        # (app/runtime/variants.py). Fixed at spawn, so a later edit of the
        # models row does not re-attribute the engine's traffic until it is
        # restarted. Cleared with the other per-engine state.
        self._variants: dict[str, Variant] = {}
        self._watchers: dict[str, asyncio.Task] = {}
        self._state: dict[str, ModelState] = {}
        self._lock = asyncio.Lock()

    async def _watch_exit(
        self,
        model_id: str,
        on_exit: Callable[[int], Awaitable[None]] | None,
    ) -> None:
        handle = self._handles.get(model_id)
        if handle is None:
            return
        try:
            rc = await handle.wait()
        except asyncio.CancelledError:
            return  # unload() cancelled us; it'll clean up
        async with self._lock:
            if model_id not in self._handles:
                return  # unload() got there first
            self._handles.pop(model_id, None)
            self._ports.pop(model_id, None)
            self._hosts.pop(model_id, None)
            self._overrides.pop(model_id, None)
            self._variants.pop(model_id, None)
            self._watchers.pop(model_id, None)
            self._state.pop(model_id, None)
            self.gpus.release(model_id)
        if on_exit is not None:
            await on_exit(rc)

    async def load(
        self,
        model,
        *,
        port: int,
        on_exit: Callable[[int], Awaitable[None]] | None = None,
        overrides: dict | None = None,
    ) -> None:
        async with self._lock:
            if model.id in self._handles:
                raise RuntimeError(f"model {model.id} already running")
            # Resolve the engine-image pin once (reused for EngineSpec below)
            # and refuse it up front — BEFORE claiming any GPU — when the
            # active driver cannot swap the engine image. Otherwise the pin is
            # silently discarded and the warden-baked vLLM launches instead
            # (the #177 bug). Unknown/test stand-in drivers default to capable.
            engine_image = _resolve_engine_image(model)
            if engine_image and not getattr(
                self._driver, "supports_engine_image", True
            ):
                raise EnginePinUnsupported(
                    f"engine version pin ({engine_image}) cannot be honored: "
                    "this deployment runs the in-container subprocess engine, "
                    "whose vLLM version is fixed by the warden image. Clear the "
                    "engine pin on this model, or run LM Warden with "
                    "VW_ENGINE_DRIVER=docker to select engine versions."
                )
            # Pass the served name so a refusal can say which model is in the
            # way; the id alone is meaningless to whoever reads last_error.
            self.gpus.claim(
                model.id,
                model.gpu_indices,
                label=getattr(model, "served_model_name", None),
            )
            try:
                hf_token_path = Path(self.settings.hf_token_path)

                def _read_hf_token() -> str:
                    return hf_token_path.read_text().strip() if hf_token_path.exists() else ""

                hf_token = await asyncio.to_thread(_read_hf_token)
                # D6 — models.backend is nullable and NULL decodes to "vllm";
                # registry.get owns that default. An unknown name raises
                # UnknownBackendError here, INSIDE the try that releases the
                # GPU claim below, so a row asking for a backend this build
                # does not have fails before anything is spawned rather than
                # silently launching vLLM under the operator's chosen name.
                backend_name = getattr(model, "backend", None)
                backend = registry.get(backend_name)
                # #211 / #210 — the DRIVER decides how wide the engine binds
                # its (unauthenticated) OpenAI server and whether the shm
                # mitigation applies; the BACKEND decides what the flag and the
                # env are called. getattr so stand-in settings in tests still
                # resolve to the safe local/loopback default.
                driver_name = getattr(self.settings, "engine_driver", "local")
                # Resolve the row's pinned files ONCE, here, so Backend.plan()
                # stays pure and so a missing file is reported as itself rather
                # than as a subprocess that exits rc=1 forty seconds later. This
                # runs inside the try/except that releases the GPU claim, so a
                # mistyped filename cannot leave a card reserved.
                #
                # ONLY for a backend that actually needs a path. vLLM takes a
                # repo id and does its own lookup, so resolving on its behalf is
                # a cache scan of a directory it never opens -- and Path.is_dir()
                # propagates EACCES, so an unreadable hf_cache_dir would fail a
                # vLLM load that has no business reading that directory at all.
                # Gated on the CAPABILITY, never on the backend's name: a name
                # check here is per-backend branching in the control plane, which
                # is the O(n*m) multiplication decision D1 exists to remove.
                resolved = None
                if backend.capabilities.needs_local_model_path:
                    resolved = await asyncio.to_thread(
                        resolve_model_paths,
                        model,
                        hf_cache_dir=self.settings.hf_cache_dir,
                    )
                plan = backend.plan(
                    model,
                    port=port,
                    bind_host=backend.bind_host(driver_name),
                    overrides=overrides,
                    resolved=resolved,
                    hf_token=hf_token,
                    hf_cache_dir=str(self.settings.hf_cache_dir),
                    engine_driver=driver_name,
                    image=engine_image,
                )
                spec = EngineSpec(
                    model_id=model.id,
                    model_arg=model.hf_repo,
                    argv=plan.argv,
                    env=plan.env,
                    port=plan.port,
                    image=plan.image,
                    gpu_indices=plan.gpu_indices,
                    backend=backend.capabilities.name,
                )
                # The variant this launch IS (app/runtime/variants.py): the row
                # and overrides plus what the plan resolved -- the image the
                # driver will actually run (the pin, else the driver's default)
                # or the in-container engine's baked version, and the commit
                # the revision points at. Before the spawn, so nothing after
                # it can fail.
                default_image = getattr(self._driver, "default_image", None)
                image_used = plan.image or (
                    default_image if isinstance(default_image, str) else None
                )
                facts = await asyncio.to_thread(
                    runtime_facts,
                    backend=backend.capabilities.name,
                    image=image_used,
                    hf_cache_dir=getattr(self.settings, "hf_cache_dir", None),
                    hf_repo=getattr(model, "hf_repo", None),
                    revision=getattr(model, "hf_revision", None),
                )
                variant = variant_of(model, overrides, facts)
                handle = await self._driver.spawn(spec)

                self._handles[model.id] = handle
                self._ports[model.id] = port
                self._hosts[model.id] = _driver_engine_host(self._driver, model.id)
                self._overrides[model.id] = overrides
                self._variants[model.id] = variant
                self._state[model.id] = ModelState.LOADING
                self._watchers[model.id] = asyncio.create_task(
                    self._watch_exit(model.id, on_exit)
                )
            except Exception:
                self.gpus.release(model.id)
                raise

    def get_port(self, model_id: str) -> int | None:
        return self._ports.get(model_id)

    def get_host(self, model_id: str) -> str | None:
        """Host the control-plane reaches this model's engine on, or
        ``None`` if no engine is registered. Loopback for the in-container
        subprocess driver; the engine container's DNS name for the docker
        driver. Consumed by the health probe, warmup and the client proxy."""
        return self._hosts.get(model_id)

    def get_pid(self, model_id: str) -> int | None:
        """Live PID of the running vLLM subprocess, or ``None`` if no
        process is registered (e.g. unload happened between caller's
        health-ok check and this read).

        Callers should treat ``None`` as "process crashed mid-operation"
        and classify accordingly. Public replacement for the previous
        ``sup._processes[model_id].pid`` private-state read.
        """
        h = self._handles.get(model_id)
        return h.pid if h is not None else None

    def get_state(self, model_id: str) -> ModelState | None:
        """Current supervisor lifecycle state for ``model_id``.

        ``None`` if no process is registered. Public read for callers
        that need to display lifecycle (e.g. the UI status badge).
        """
        return self._state.get(model_id)

    async def mark_warming(self, model_id: str) -> None:
        """Transition ``model_id`` from LOADING to WARMING.

        Called by the load runner after ``wait_for_health`` succeeds and
        before the warmup verification probe runs.
        """
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not ModelState.LOADING:
                raise RuntimeError(
                    f"cannot mark warming from state {cur}: expected LOADING"
                )
            self._state[model_id] = ModelState.WARMING

    async def mark_ready(self, model_id: str) -> None:
        """Transition ``model_id`` from WARMING to READY.

        Called by the load runner after the warmup probe succeeds.
        After this transition, ``unload()`` is permitted without force.
        """
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not ModelState.WARMING:
                raise RuntimeError(
                    f"cannot mark ready from state {cur}: expected WARMING"
                )
            self._state[model_id] = ModelState.READY

    def get_overrides(self, model_id: str) -> dict | None:
        """Snapshot of the overrides dict in effect for ``model_id``.

        Returns the same shape the caller passed to :meth:`load`:
        ``None`` if the model was loaded with row defaults, a dict
        otherwise. Returns ``None`` if no model is registered — callers
        in that state should treat the absence as "nothing to restore".
        """
        return self._overrides.get(model_id)

    def get_variant(self, model_id: str) -> Variant | None:
        """The variant ``model_id``'s engine was launched as, or None when it
        was not launched by this supervisor (or is not running)."""
        return self._variants.get(model_id)

    def is_running(self, model_id: str) -> bool:
        h = self._handles.get(model_id)
        return h is not None and h.returncode is None

    def parent_pid_to_model(self) -> dict[int, str]:
        """Snapshot of {parent_pid: model_id} for live PID→model attribution.

        Only includes engines still running (``returncode is None``). Used by
        the live GPU probe to label nvidia-smi compute holders.
        """
        out: dict[int, str] = {}
        for model_id, h in self._handles.items():
            if h.returncode is None and h.pid is not None:
                out[h.pid] = model_id
        return out

    async def ensure_unloadable(self, model_id: str, *, force: bool = False) -> None:
        """Fast pre-flight check: raise :class:`UnloadRefused` if the model is
        in a transient state (LOADING/WARMING/…) and ``force`` is not set.

        #166 — split out from :meth:`unload` so the route can surface the
        refusal **synchronously** (HTTP 409) while running the slow engine
        teardown in a background task. Does no teardown and holds the lock only
        briefly. :meth:`unload` re-checks under the same lock, so this is an
        advisory pre-flight, not a substitute for the in-``unload`` guard.
        """
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not None and cur is not ModelState.READY and not force:
                raise UnloadRefused(model_id, cur)

    async def unload(self, model_id: str, *, force: bool = False) -> None:
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not None and cur is not ModelState.READY and not force:
                raise UnloadRefused(model_id, cur)
            # Past the refusal gate the model is being torn down for good. The
            # GPU ownership + lifecycle bookkeeping MUST be released even if
            # ``terminate()`` raises or the await is cancelled (client/proxy
            # disconnect): the old code released GPUs only as its trailing
            # statement, so a teardown exception stranded the claim and left
            # the GPUs permanently "already claimed" by a model that is gone —
            # unrecoverable without a control-plane restart (#166-adjacent
            # leak, observed in production). Release in ``finally`` to close that gap.
            try:
                watcher = self._watchers.pop(model_id, None)
                if watcher is not None and not watcher.done():
                    watcher.cancel()
                handle = self._handles.get(model_id)
                if handle is None:
                    return
                self._state[model_id] = ModelState.UNLOADING
                if handle.returncode is None:
                    await self._driver.terminate(handle, grace_s=UNLOAD_GRACE_SECONDS)
            finally:
                self._handles.pop(model_id, None)
                self._ports.pop(model_id, None)
                self._hosts.pop(model_id, None)
                self._overrides.pop(model_id, None)
                self._variants.pop(model_id, None)
                self._state.pop(model_id, None)
                self.gpus.release(model_id)


async def _http_get(url: str, timeout: float):  # noqa: ASYNC109
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await c.get(url)


async def wait_for_health(
    *,
    port: int,
    host: str = "127.0.0.1",
    timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    interval_s: float = 2.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        try:
            r = await _http_get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(interval_s)
    return False
