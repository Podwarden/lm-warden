"""Sibling-container engine driver (Docker-out-of-Docker). Asks the HOST
docker daemon (via mounted /var/run/docker.sock) to run the engine image
as a sibling container. GPUs are mapped by the host NVIDIA Container
Toolkit; volumes MUST be named (bind paths resolve on the host). The
handle's pid is the engine container's HOST pid (docker inspect .State.Pid)
which is exactly what nvidia-smi attribution keys on. Implements #160.

GPU mapping note: the Python ``docker`` SDK has NO ``gpus=`` kwarg on
``containers.run`` — that string is a docker *CLI* flag only. The SDK maps
GPUs through ``device_requests=[DeviceRequest(...)]``. We import the SDK
lazily so the default (local subprocess) path never requires it installed.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

from app.runtime.engine.run_marker import format_run_sentinel

HFCACHE_VOLUME = "vllm-warden-hfcache"
# Where the engine container sees the model cache. This is the path every
# shipped deployment shape gives the WARDEN too (docker-compose.yml and the
# PodWarden Hub catalogue template's compose both mount the cache volume
# here, and settings.hf_cache_dir defaults to it), which is what makes
# HF_HUB_CACHE — set from settings.hf_cache_dir by the backend's env builder —
# resolve to the same bytes on both sides. app/main.py says so out loud at
# startup if a deployment has moved VW_HF_CACHE_DIR away from it.
HFCACHE_MOUNT = "/root/.cache/huggingface"

# THE WARDEN'S DATA VOLUME IS DELIBERATELY NOT MOUNTED HERE (#265).
#
# Until this change the driver mounted ``vllm-warden-data`` at ``/data`` with
# mode "rw" in every engine container. That volume holds ``vllm-warden.db`` —
# api tokens, the admin-token audit trail, models, sessions — and the
# auto-minted ``jwt_secret`` (app/config.py). The engine container runs a
# model's code, under an image an operator or an admin token names
# (``engine_image``), with argv an admin token can extend (``extra_args``), so
# that mount handed every engine read AND write access to the control plane's
# own state: read the database, mint a session, edit the audit trail. It also
# made the docker driver no more isolated from warden state than the
# local-subprocess driver, which shares the warden's filesystem by nature.
#
# The engine needs nothing there. Its env carries no path under /data
# (``HF_HUB_CACHE`` points at the cache volume above; see
# app/runtime/backends/*/env.py), its argv names files only inside the model
# cache (app/runtime/backends/paths.py), and the per-model log file under
# ``<data_dir>/logs`` is written by the WARDEN — ``_start_log_pump`` below
# streams ``container.logs()`` from outside and writes it, exactly as the
# watchdog assembles crash bundles from outside. The container writes to its
# own stdout/stderr and nothing else.
#
# The one path that ever did write there was the model cache, before the
# 2026-06-15 ENOSPC incident pointed it at its own volume (the comment in
# app/runtime/backends/vllm/env.py records it). Re-adding this mount would
# re-open both.

# vLLM tensor-parallel workers talk over POSIX shared memory (the
# shm_broadcast message queue + custom-all-reduce CUDA-IPC handles). A
# sibling engine container's default 64MB /dev/shm is far too small for
# TP>1, and the workers hang at the distributed-init barrier with no crash
# and no error — the model just sits in "loading" forever. The
# control-plane compose override already sets ipc:host for the same reason;
# mirror it on the engine siblings. Both knobs are env-overridable for
# hosts that need an isolated IPC namespace (set VLLM_ENGINE_IPC_MODE="" to
# fall back to a private but enlarged /dev/shm).
ENGINE_SHM_SIZE = os.environ.get("VLLM_ENGINE_SHM_SIZE", "16g")
ENGINE_IPC_MODE = os.environ.get("VLLM_ENGINE_IPC_MODE", "host")

# A sibling engine container lands on the default bridge by default — a
# DIFFERENT network from the control-plane's compose network, so the API
# container cannot reach it at all (no shared subnet, no shared DNS) and
# every health probe/proxy hangs while the model sits in "loading" forever.
# Attach the engine to the control-plane's network instead; docker's
# embedded DNS then resolves the engine by its container name
# (``vllm-warden-engine-<model_id>``), which is exactly what engine_host()
# hands the supervisor. Empty => leave the engine on the default bridge
# (only correct when the control-plane itself runs on the host network).
ENGINE_NETWORK = os.environ.get("VW_ENGINE_NETWORK", "vllm-warden_default")
ENGINE_NAME_PREFIX = "vllm-warden-engine-"

# Backends whose engine container keeps the historical, un-suffixed name.
# ``None`` is decision D6's NULL-means-vLLM; "vllm" is the same thing spelled
# out. Both must map to the old name or every running container is orphaned.
_UNSUFFIXED_BACKENDS = (None, "", "vllm")


def _engine_name(model_id: str, backend: str | None = None) -> str:
    """Container name for one model's engine.

    Sub-project B deliberately deferred this to C (its self-review reads
    "DockerSocketDriver derives the name from ``backend`` -- deliberately
    omitted -- identity change, sub-project C"), because ``backend`` is a
    per-model column an operator can switch. Switching it leaves the previous
    engine's container behind under a name the new engine would claim, and the
    two are different programs.

    The vLLM name is BYTE-IDENTICAL to what shipped, so no running container is
    orphaned; only a non-default backend adds a segment. The
    ``vllm-warden-engine-`` prefix itself does not change -- that identity
    belongs to the rename (§8.3, sub-project A).
    """
    suffix = "" if backend in _UNSUFFIXED_BACKENDS else f"-{backend}"
    return f"{ENGINE_NAME_PREFIX}{model_id}{suffix}"


def _gpu_device_requests(gpu_indices: list[int]):
    """Build the docker SDK device_requests for the pinned GPUs.

    Empty ``gpu_indices`` => request all GPUs (count=-1)."""
    from docker.types import DeviceRequest

    if gpu_indices:
        # The SDK wants ``device_ids`` as a list of individual id strings
        # (``["0", "1"]``). The comma-joined ``"0,1"`` form is the docker
        # *CLI* `--gpus device=0,1` syntax and is silently mis-mapped by the
        # SDK — it would request a single device literally named "0,1"
        # instead of two GPUs.
        return [
            DeviceRequest(
                device_ids=[str(i) for i in gpu_indices],
                capabilities=[["gpu"]],
            )
        ]
    return [DeviceRequest(count=-1, capabilities=[["gpu"]])]


def _relative_cuda_visible_devices(env: dict[str, str], gpu_indices: list[int]) -> dict[str, str]:
    """Remap ``CUDA_VISIBLE_DEVICES`` to container-relative indices.

    ``env_builder`` sets ``CUDA_VISIBLE_DEVICES`` to the model's *host* GPU
    indices (e.g. ``"1"`` or ``"2,3"``). That is correct for the in-process
    ``LocalSubprocessDriver``, whose parent container is launched ``--gpus all``
    and therefore sees every GPU at its host index.

    The docker driver is different: ``device_requests`` pins exactly the
    physical GPUs in ``gpu_indices``, and the NVIDIA container runtime renumbers
    those passed-through devices to ``0..N-1`` INSIDE the sibling engine
    container. The host indices no longer exist there, so a host-index
    ``CUDA_VISIBLE_DEVICES`` makes vLLM call ``nvmlDeviceGetHandleByIndex(host_idx)``
    against an NVML view that only has indices ``0..N-1`` — raising
    ``NVMLError_InvalidArgument`` and crashing the engine (rc=1) at startup. The
    GPU-0 model happens to survive only because host index 0 == relative 0.

    The device set is already isolated by ``device_requests``; here we simply
    select all of it in container-relative order.
    """
    if not gpu_indices:
        return env
    env = dict(env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(gpu_indices)))
    return env


class DockerHandle:
    """Mirrors ``LocalHandle`` semantics so the Supervisor's synchronous
    reads (``get_pid`` / ``is_running`` / ``parent_pid_to_model``) never
    block the event loop on a docker-daemon round-trip.

    The container's HOST pid is immutable for the container's lifetime, so
    it is captured ONCE off-thread at spawn. ``returncode`` stays ``None``
    until ``wait()`` reaps the container and caches the exit code — exactly
    like ``asyncio.subprocess.Process.returncode``.
    """

    def __init__(self, container, *, pid: int | None) -> None:
        self._c = container
        self._pid = pid
        self._returncode: int | None = None
        self._wait_task: asyncio.Task | None = None

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self) -> int:
        # Memoize ONE in-flight reap. The exit-watcher and a concurrent
        # terminate() both reap the same container; without sharing, each
        # spawns its own ``to_thread(self._c.wait)`` — and ``to_thread`` is
        # not cancellable, so cancelling the watcher (unload does this) does
        # NOT stop its thread. Two live ``container.wait()`` calls on one
        # container is wasteful and fragile; one shared task avoids it.
        # Cancelling a coroutine that ``await``s this task does not cancel
        # the task itself, so terminate() can still await the same reap.
        #
        # #171: but ``Supervisor.unload()`` cancels the exit-watcher, which
        # cancels the watcher's *task* (not just the awaiting coroutine) —
        # poisoning this memoized ``_wait_task``. A later terminate() that
        # reused that cancelled task would re-raise ``asyncio.CancelledError``
        # (a BaseException, so it escapes terminate()'s ``except Exception``
        # guard) and collapse unload() to HTTP 500. Never reuse a poisoned
        # future: if the prior task is done-with-cancellation, recreate it so
        # the reap runs against the live container instead.
        if self._wait_task is None or (
            self._wait_task.done() and self._wait_task.cancelled()
        ):
            self._wait_task = asyncio.create_task(asyncio.to_thread(self._c.wait))
        res = await self._wait_task
        self._returncode = int(res.get("StatusCode", 0))
        return self._returncode


class DockerSocketDriver:
    # The engine runs as a sibling container whose image we choose at spawn
    # time, so ``EngineSpec.image`` (an engine-version pin) is honored:
    # ``image = spec.image or self._image``. Surfaced so Supervisor.load knows
    # an engine-version pin is safe under this driver.
    supports_engine_image = True

    def __init__(self, *, client, image: str, log_dir: str | None = None) -> None:
        self._client = client
        self._image = image
        # When set, spawn() mirrors the engine container's combined
        # stdout+stderr into ``<log_dir>/<model_id>.log`` so the SSE log
        # endpoint (app/models/routes_logs.py), which reads ONLY from that
        # per-model file, streams live container output. Without this the
        # docker-driver UI Live-logs panel shows stale/empty content (the
        # subprocess driver writes the file; the docker driver did not). When
        # None, behaviour is unchanged (no file written) so existing tests and
        # log_dir-less callers keep working. (#177 follow-up)
        self._log_dir = Path(log_dir) if log_dir is not None else None
        # model_id -> the container name spawn() actually used. Needed because
        # the name now depends on the row's backend and ``engine_host`` is
        # handed only a model_id: without this record a llama.cpp engine would
        # be spawned as ...-<id>-llamacpp and then reached at ...-<id>, and the
        # failure would surface as a health-probe timeout with no clue why.
        self._names: dict[str, str] = {}

    @property
    def default_image(self) -> str:
        """The image an engine runs when its spec pins none. Read by the
        supervisor to record which engine a model variant actually ran
        (app/runtime/variants.py): a warden upgrade that moves this default
        is a new variant even for a model that pins nothing."""
        return self._image

    def engine_host(self, model_id: str) -> str:
        # The sibling engine is reachable from the control-plane container by
        # its docker DNS name once both share ENGINE_NETWORK. The published
        # host port is for operator debugging only — the warden talks to the
        # engine container-to-container, never via 127.0.0.1 (that is the
        # control-plane's OWN loopback, where no engine listens).
        #
        # Falls back to the un-suffixed name for a model this driver instance
        # never spawned (a warden restart with a container still up), which is
        # the pre-C behaviour and correct for every vLLM row.
        return self._names.get(model_id) or _engine_name(model_id)

    async def spawn(self, spec) -> DockerHandle:
        name = _engine_name(spec.model_id, getattr(spec, "backend", None))
        self._names[spec.model_id] = name
        # A per-model engine image (resolved from the template's engine axis
        # by app.runtime.backends.vllm.images) arrives on spec.image and wins; the
        # driver-level image is the fallback default.
        image = spec.image or self._image
        # argv[0] is set as the container ENTRYPOINT rather than being folded
        # into `command`. Before sub-project B the driver passed the
        # post-binary tail as `command` and relied on the engine image's baked
        # ENTRYPOINT to be `vllm serve`. Now that the backend owns argv[0],
        # passing the whole vector as `command` would double-invoke it. Setting
        # the entrypoint explicitly also removes the dependency on what a
        # pinned image happened to bake in.
        kwargs = dict(
            entrypoint=spec.argv[0],
            command=list(spec.argv[1:]),
            detach=True,
            environment=_relative_cuda_visible_devices(
                dict(spec.env), list(spec.gpu_indices)
            ),
            ports={f"{spec.port}/tcp": spec.port},
            # The model cache and NOTHING ELSE — see HFCACHE_MOUNT above for
            # why the warden's data volume is not here (#265).
            volumes={
                HFCACHE_VOLUME: {"bind": HFCACHE_MOUNT, "mode": "rw"},
            },
            device_requests=_gpu_device_requests(list(spec.gpu_indices)),
            shm_size=ENGINE_SHM_SIZE,
            name=name,
            labels={"vllm-warden.model_id": spec.model_id},
        )
        # ipc_mode="host" makes docker ignore shm_size and use the host's
        # (large) /dev/shm — the reliable fix for TP>1. Only pass it when
        # set so an empty override leaves the enlarged private shm in place.
        if ENGINE_IPC_MODE:
            kwargs["ipc_mode"] = ENGINE_IPC_MODE
        # Join the control-plane's network so the API container can resolve
        # and reach the engine by container name. Skipped only when unset.
        if ENGINE_NETWORK:
            kwargs["network"] = ENGINE_NETWORK

        def _run():
            # Reload of the same model reuses the deterministic name; a stale
            # container (left behind by a swallowed teardown failure) would
            # otherwise make ``run`` fail with a name-conflict APIError. Remove
            # any leftover before claiming the name.
            try:
                self._client.containers.get(name).remove(force=True)
            except Exception:
                pass
            container = self._client.containers.run(image, **kwargs)
            # The HOST pid is captured ONCE (the handle's pid is immutable
            # for the container's lifetime). A just-started container can
            # briefly report Pid 0 before the runtime publishes it; poll
            # briefly so docker-mode GPU attribution (parent_pid_to_model)
            # doesn't permanently lose a slow-starting model.
            pid = None
            for _ in range(20):
                container.reload()
                pid = container.attrs.get("State", {}).get("Pid") or None
                if pid:
                    break
                time.sleep(0.1)
            return container, pid

        container, pid = await asyncio.to_thread(_run)
        self._start_log_pump(container, spec.model_id)
        return DockerHandle(container, pid=pid)

    def _start_log_pump(self, container, model_id: str) -> None:
        """Mirror the engine container's combined stdout+stderr into
        ``<log_dir>/<model_id>.log`` so routes_logs.py can stream it.

        ``container.logs(stream=True, follow=True, ...)`` returns a BLOCKING
        generator (it sits on the docker daemon's chunked HTTP body until the
        next byte arrives), so it CANNOT be driven from the asyncio event loop
        or an ``asyncio.to_thread`` slot (which it would hold for the engine's
        entire lifetime, starving the bounded default thread pool). Run it on
        a dedicated daemon thread instead. The file is truncated on each spawn
        ("wb") so a fresh run never shows the previous engine's log.
        """
        if self._log_dir is None:
            return
        log_dir = self._log_dir
        log_path = log_dir / f"{model_id}.log"

        def _pump():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                # Truncate: a fresh engine run must clear stale content (the
                # the observed false "still v0.20.0" reading came from a stale file).
                with open(log_path, "wb") as f:
                    # Run boundary (#234). This driver truncates, so the
                    # sentinel is redundant for scoping here -- write it anyway
                    # so the reader's contract is "logs are delimited", not
                    # "logs are delimited unless the docker driver wrote them",
                    # and so an operator greps one marker on both drivers.
                    f.write(f"{format_run_sentinel()}\n".encode())
                    f.flush()
                    for chunk in container.logs(
                        stream=True, follow=True, stdout=True, stderr=True
                    ):
                        f.write(chunk)
                        f.flush()
            except Exception:
                # The stream ends (or raises) when the container is removed /
                # the daemon connection drops — that's the normal terminal
                # state, not an error worth surfacing.
                pass

        threading.Thread(
            target=_pump, name=f"engine-log-pump-{model_id}", daemon=True
        ).start()

    async def terminate(self, handle: DockerHandle, *, grace_s: float) -> None:
        def _stop():
            try:
                handle._c.stop(timeout=int(grace_s))
            except Exception:
                pass

        await asyncio.to_thread(_stop)
        # Reap the exit code before removal so ``handle.returncode`` becomes
        # non-None — mirroring LocalSubprocessDriver.terminate(), which awaits
        # ``wait()``. Go through the handle's memoized ``wait()`` so we share
        # the single reap with the exit-watcher instead of opening a second
        # concurrent ``container.wait()`` on the same container.
        if handle._returncode is None:
            try:
                await handle.wait()
            except asyncio.CancelledError:
                # #171: a poisoned (already-cancelled) memoized wait task can
                # still re-raise CancelledError here. CancelledError is a
                # BaseException, so the ``except Exception`` below would NOT
                # absorb it and it would escape terminate() -> unload() ->
                # middleware as HTTP 500. The container is being stopped and
                # removed regardless; treat it as exited so reap stays clean.
                handle._returncode = 0
            except Exception:
                handle._returncode = 0

        def _remove():
            try:
                handle._c.remove(force=True)
            except Exception:
                pass

        await asyncio.to_thread(_remove)
