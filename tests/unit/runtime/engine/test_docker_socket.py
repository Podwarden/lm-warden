import asyncio

import pytest

from app.runtime.engine import EngineSpec
from app.runtime.engine.docker_socket import (
    HFCACHE_MOUNT,
    HFCACHE_VOLUME,
    DockerSocketDriver,
)

# The warden's data volume, spelled out here rather than imported: the driver no
# longer names it at all (#265), and a test that imported the constant would
# start passing again the moment somebody re-added it.
DATA_VOLUME_NAME = "vllm-warden-data"


class _FakeContainers:
    def __init__(self, log_chunks=None):
        self.run_kwargs = None
        self.removed = []
        self._log_chunks = log_chunks

    def run(self, image, **kwargs):
        self.run_kwargs = {"image": image, **kwargs}
        return _FakeContainer("cid-1", log_chunks=self._log_chunks)

    def get(self, name):
        # No stale container by default; spawn's pre-clean swallows this.
        raise KeyError(name)


class _FakeContainer:
    def __init__(self, cid, log_chunks=None):
        self.id = cid
        self._stopped = False
        self.wait_calls = 0
        self.attrs = {"State": {"Pid": 9999, "Running": True, "ExitCode": 0}}
        # Byte chunks the fake docker log stream will yield. None => no logs
        # call recorded (lets the existing no-log_dir tests stay untouched).
        self._log_chunks = log_chunks
        self.logs_kwargs = None

    def reload(self):
        if self._stopped:
            self.attrs["State"] = {"Pid": 0, "Running": False, "ExitCode": 0}

    def logs(self, **kwargs):
        # Mirror docker SDK's blocking follow generator: yield the queued
        # byte chunks then terminate (as if the container were removed).
        self.logs_kwargs = kwargs
        yield from (self._log_chunks or [])

    def stop(self, timeout=None):
        self._stopped = True

    def remove(self, force=False):
        pass

    def wait(self):
        self.wait_calls += 1
        return {"StatusCode": 0}


class _FakeClient:
    def __init__(self, log_chunks=None):
        self.containers = _FakeContainers(log_chunks=log_chunks)


@pytest.mark.asyncio
async def test_spawn_runs_engine_image_with_gpus():
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="vllm/vllm-openai:v0.20.0")
    spec = EngineSpec(model_id="m1", model_arg="openai/gpt-oss-20b",
                      argv=["vllm", "serve", "--model", "openai/gpt-oss-20b", "--port", "8001"],
                      env={"VLLM_USE_V1": "1"}, port=8001, gpu_indices=[0, 1])
    handle = await drv.spawn(spec)
    kw = client.containers.run_kwargs
    assert kw["image"] == "vllm/vllm-openai:v0.20.0"
    assert kw["detach"] is True
    # The Python docker SDK has no `gpus=` kwarg (that's a CLI-only flag);
    # GPUs are mapped via device_requests. device_ids MUST be a list of
    # individual id strings (["0", "1"]) — the comma-joined CLI form
    # ("0,1") is silently mis-mapped by the SDK.
    assert "gpus" not in kw
    dreqs = kw["device_requests"]
    assert len(dreqs) == 1
    assert dreqs[0].device_ids == ["0", "1"]
    assert dreqs[0].capabilities == [["gpu"]]
    assert kw["ports"] == {"8001/tcp": 8001}
    assert handle.pid == 9999


@pytest.mark.asyncio
async def test_spawn_remaps_cuda_visible_devices_to_relative_indices():
    # #172: device_requests pins the physical GPUs and the NVIDIA runtime
    # renumbers them to 0..N-1 INSIDE the sibling engine container. env_builder
    # set CUDA_VISIBLE_DEVICES to the HOST indices (correct only for the
    # in-process LocalSubprocessDriver). Leaving the host index here makes vLLM
    # call nvmlDeviceGetHandleByIndex(host_idx) against an NVML view that only
    # has 0..N-1 -> NVMLError_InvalidArgument -> engine crashes rc=1. The driver
    # MUST remap to the container-relative range.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    # Single non-zero GPU: the exact case that crashed (host index 2, but the
    # container only has relative index 0).
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={
        "CUDA_VISIBLE_DEVICES": "2", "VLLM_LOGGING_LEVEL": "INFO"},
        port=8001, gpu_indices=[2])
    await drv.spawn(spec)
    kw = client.containers.run_kwargs
    assert kw["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    # device_requests still pins the PHYSICAL GPU (host index 2).
    assert kw["device_requests"][0].device_ids == ["2"]
    # Other env keys are untouched.
    assert kw["environment"]["VLLM_LOGGING_LEVEL"] == "INFO"


@pytest.mark.asyncio
async def test_spawn_remaps_multi_gpu_cuda_visible_devices():
    # Multi-GPU (TP): two host GPUs [1,3] become relative "0,1" inside the
    # container, while device_requests still pins the physical 1 and 3.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"],
                      env={"CUDA_VISIBLE_DEVICES": "1,3"}, port=8001,
                      gpu_indices=[1, 3])
    await drv.spawn(spec)
    kw = client.containers.run_kwargs
    assert kw["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert kw["device_requests"][0].device_ids == ["1", "3"]


@pytest.mark.asyncio
async def test_spawn_gives_engine_adequate_shared_memory():
    # TP>1 workers hang at the distributed-init barrier on the default 64MB
    # /dev/shm. The driver MUST enlarge it: ipc_mode=host by default (docker
    # then uses the host's large /dev/shm) plus an explicit shm_size fallback.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001,
                      gpu_indices=[0, 1])
    await drv.spawn(spec)
    kw = client.containers.run_kwargs
    assert kw["ipc_mode"] == "host"
    assert kw["shm_size"] == "16g"


@pytest.mark.asyncio
async def test_spawn_ipc_mode_overridable_to_private_shm(monkeypatch):
    # An operator can opt out of the shared host IPC namespace; an empty
    # VLLM_ENGINE_IPC_MODE leaves the enlarged PRIVATE /dev/shm in place
    # (no ipc_mode kwarg passed at all).
    import importlib

    monkeypatch.setenv("VLLM_ENGINE_IPC_MODE", "")
    monkeypatch.setenv("VLLM_ENGINE_SHM_SIZE", "8g")
    from app.runtime.engine import docker_socket as ds

    importlib.reload(ds)
    try:
        client = _FakeClient()
        drv = ds.DockerSocketDriver(client=client, image="img:tag")
        spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={},
                          port=8001, gpu_indices=[0, 1])
        await drv.spawn(spec)
        kw = client.containers.run_kwargs
        assert "ipc_mode" not in kw
        assert kw["shm_size"] == "8g"
    finally:
        # Undo the env BEFORE the restoring reload: monkeypatch only reverts
        # at teardown, so reloading here with the override still in place
        # baked the override into the module for every later test (an
        # ordering bug surfaced by random/xdist ordering).
        monkeypatch.undo()
        importlib.reload(ds)


@pytest.mark.asyncio
async def test_engine_host_is_container_dns_name():
    # A sibling engine container is NOT on the control-plane's loopback —
    # the warden reaches it by its docker DNS name once both share the
    # compose network. engine_host MUST return that name so the health
    # probe / warmup / proxy target the right host.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    assert drv.engine_host("f1b1fdb532f8bfa6") == "vllm-warden-engine-f1b1fdb532f8bfa6"


@pytest.mark.asyncio
async def test_spawn_attaches_engine_to_control_plane_network():
    # Default-bridge isolation: an engine left on the default bridge shares
    # no subnet/DNS with the API container, so every probe hangs. The driver
    # MUST attach the engine to the control-plane's compose network, and the
    # container name MUST match engine_host() so DNS resolves.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001,
                      gpu_indices=[0, 1])
    await drv.spawn(spec)
    kw = client.containers.run_kwargs
    assert kw["network"] == "vllm-warden_default"
    assert kw["name"] == "vllm-warden-engine-m1"
    assert kw["name"] == drv.engine_host("m1")


@pytest.mark.asyncio
async def test_spawn_network_overridable(monkeypatch):
    # An operator whose control-plane runs on the host network sets an empty
    # VW_ENGINE_NETWORK so the engine stays on the default bridge (no network
    # kwarg passed). engine_host still returns the container name.
    import importlib

    monkeypatch.setenv("VW_ENGINE_NETWORK", "")
    from app.runtime.engine import docker_socket as ds

    importlib.reload(ds)
    try:
        client = _FakeClient()
        drv = ds.DockerSocketDriver(client=client, image="img:tag")
        spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={},
                          port=8001, gpu_indices=[0, 1])
        await drv.spawn(spec)
        assert "network" not in client.containers.run_kwargs
    finally:
        # See test_spawn_ipc_mode_overridable_to_private_shm: revert the env
        # first, then reload, or the empty network sticks for later tests.
        monkeypatch.undo()
        importlib.reload(ds)


@pytest.mark.asyncio
async def test_spawn_requests_all_gpus_when_unpinned():
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    await drv.spawn(spec)
    dreqs = client.containers.run_kwargs["device_requests"]
    assert len(dreqs) == 1
    assert dreqs[0].count == -1
    assert dreqs[0].capabilities == [["gpu"]]


@pytest.mark.asyncio
async def test_spawn_prefers_spec_image_over_driver_default():
    # The driver-level image is the fallback default; a per-model engine
    # axis (resolved by app.runtime.backends.vllm.images) arrives on spec.image and
    # MUST win so different models can run on different engine images.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="default:fallback")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001,
                      image="vllm/vllm-openai:v0.21.0")
    await drv.spawn(spec)
    assert client.containers.run_kwargs["image"] == "vllm/vllm-openai:v0.21.0"


@pytest.mark.asyncio
async def test_spawn_falls_back_to_driver_image_when_spec_image_none():
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="default:fallback")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    await drv.spawn(spec)
    assert client.containers.run_kwargs["image"] == "default:fallback"


@pytest.mark.asyncio
async def test_terminate_stops_and_removes():
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)
    await drv.terminate(handle, grace_s=1.0)
    assert handle.returncode == 0


@pytest.mark.asyncio
async def test_handle_wait_is_memoized_single_reap():
    # The exit-watcher and terminate() both reap the same container. They
    # MUST share ONE underlying container.wait() — a second concurrent wait
    # is wasteful and (since to_thread is not cancellable) leaks a blocked
    # thread on unload. Awaiting wait() more than once triggers exactly one
    # container.wait().
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)
    rc1 = await handle.wait()
    rc2 = await handle.wait()
    assert rc1 == rc2 == 0
    assert handle._c.wait_calls == 1


@pytest.mark.asyncio
async def test_terminate_reuses_watcher_reap():
    # When the exit-watcher already has a wait() in flight, terminate must
    # reuse it rather than open a second container.wait().
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)
    watcher = asyncio.ensure_future(handle.wait())  # simulate exit-watcher
    await drv.terminate(handle, grace_s=1.0)
    await watcher
    assert handle.returncode == 0
    assert handle._c.wait_calls == 1


@pytest.mark.asyncio
async def test_terminate_reaps_poisoned_cancelled_wait_task():
    # #171 root cause behind #166: Supervisor.unload() cancels the
    # exit-watcher, which cancels the handle's MEMOIZED ``_wait_task``. A
    # later terminate() that awaits that same cancelled task would re-raise
    # ``asyncio.CancelledError`` (a BaseException, NOT caught by
    # ``except Exception``), escape terminate() -> unload() -> middleware,
    # and collapse to HTTP 500 ("No response returned.").
    #
    # terminate() MUST instead reap cleanly: either the reap guard absorbs
    # the CancelledError, or wait() declines to reuse a poisoned future.
    # Either way terminate() returns without raising and sets a returncode.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)

    # Simulate the exit-watcher's wait() that unload() cancelled: a memoized
    # _wait_task that has been cancelled and finished cancellation.
    async def _never():
        await asyncio.Event().wait()

    poisoned = asyncio.ensure_future(_never())
    poisoned.cancel()
    try:
        await poisoned
    except asyncio.CancelledError:
        pass
    assert poisoned.cancelled()
    handle._wait_task = poisoned

    # Must NOT raise CancelledError out of terminate().
    await drv.terminate(handle, grace_s=1.0)
    assert handle.returncode is not None


@pytest.mark.asyncio
async def test_wait_does_not_reuse_cancelled_task():
    # Defense-in-depth half of #171: a memoized _wait_task that has been
    # cancelled is "poisoned" — awaiting it re-raises CancelledError. wait()
    # MUST recreate it against the live container instead of inheriting the
    # poisoned future, so the normal reap path still yields the exit code.
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)

    async def _never():
        await asyncio.Event().wait()

    poisoned = asyncio.ensure_future(_never())
    poisoned.cancel()
    try:
        await poisoned
    except asyncio.CancelledError:
        pass
    handle._wait_task = poisoned

    rc = await handle.wait()
    assert rc == 0
    assert handle._c.wait_calls == 1


def _read_log_with_retry(log_path, expected: bytes, *, tries=50, delay=0.05):
    # The log pump runs on a daemon thread; poll briefly for it to flush.
    # Matched as a SUFFIX, not equality: since #234 the pump stamps a run
    # sentinel line ahead of the container's own output.
    import time

    for _ in range(tries):
        if log_path.exists() and log_path.read_bytes().endswith(expected):
            return log_path.read_bytes()
        time.sleep(delay)
    return log_path.read_bytes() if log_path.exists() else None


def _split_sentinel(content: bytes) -> bytes:
    """Strip the leading run-boundary sentinel, asserting it was there."""
    from app.runtime.engine.run_marker import is_run_sentinel

    head, _, rest = content.partition(b"\n")
    assert is_run_sentinel(head.decode()), f"no run sentinel at head of {content!r}"
    return rest


@pytest.mark.asyncio
async def test_spawn_mirrors_container_logs_to_per_model_file(tmp_path):
    # Bug: under the docker driver the engine's vLLM output only goes to
    # `docker logs <container>`; the SSE log endpoint reads ONLY from
    # <data_dir>/logs/<model_id>.log, so the UI Live-logs panel shows stale
    # or empty content. The driver MUST mirror the container's combined
    # stdout+stderr into that per-model file.
    chunks = [b"vLLM starting up v0.21.0\n", b"INFO loading weights\n"]
    client = _FakeClient(log_chunks=chunks)
    drv = DockerSocketDriver(
        client=client, image="img:tag", log_dir=str(tmp_path)
    )
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)

    log_path = tmp_path / "m1.log"
    content = _read_log_with_retry(log_path, b"".join(chunks))
    assert _split_sentinel(content) == b"".join(chunks)
    # The driver must request the blocking follow stream of BOTH std streams.
    lk = handle._c.logs_kwargs
    assert lk == {
        "stream": True,
        "follow": True,
        "stdout": True,
        "stderr": True,
    }


@pytest.mark.asyncio
async def test_spawn_truncates_stale_log_file(tmp_path):
    # A fresh run must clear stale content — otherwise an operator sees the
    # previous engine's log (this is the observed "still v0.20.0" false reading).
    log_path = tmp_path / "m1.log"
    log_path.write_bytes(b"STALE old-engine v0.20.0 output\n")

    chunks = [b"fresh v0.21.0 line\n"]
    client = _FakeClient(log_chunks=chunks)
    drv = DockerSocketDriver(
        client=client, image="img:tag", log_dir=str(tmp_path)
    )
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    await drv.spawn(spec)

    content = _read_log_with_retry(log_path, b"".join(chunks))
    assert b"STALE" not in content
    assert _split_sentinel(content) == b"".join(chunks)


@pytest.mark.asyncio
async def test_spawn_without_log_dir_does_not_pump_logs(tmp_path):
    # Backwards compat: when no log_dir is configured the driver must keep
    # today's no-file behaviour (the existing tests construct it this way)
    # and must NOT call container.logs() at all.
    client = _FakeClient(log_chunks=[b"should not be read\n"])
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", argv=["vllm", "serve"], env={}, port=8001)
    handle = await drv.spawn(spec)
    assert handle._c.logs_kwargs is None
    assert not (tmp_path / "m1.log").exists()


@pytest.mark.asyncio
async def test_spawn_splits_argv_into_entrypoint_and_command():
    """argv[0] must reach docker as the ENTRYPOINT, not as the first arg.

    The driver used to pass command=spec.args and rely on the engine image's
    baked ENTRYPOINT to supply 'vllm serve'. Now that argv carries argv[0],
    passing the whole vector as `command` would run 'vllm serve vllm serve
    --model ...'. Setting entrypoint explicitly is also strictly more correct:
    it no longer matters what the chosen image baked in.
    """
    client = _FakeClient()
    driver = DockerSocketDriver(client=client, image="vllm/vllm-openai:v0.26.0")
    spec = EngineSpec(model_id="m1", model_arg="org/m",
                      argv=["vllm", "serve", "--model", "org/m"],
                      env={}, port=10000, gpu_indices=[0])
    await driver.spawn(spec)
    kwargs = client.containers.run_kwargs
    assert kwargs["entrypoint"] == "vllm"
    assert kwargs["command"] == ["serve", "--model", "org/m"]


@pytest.mark.asyncio
async def test_spawn_does_not_mount_the_wardens_data_volume():
    """#265 — the engine container must not be able to reach the warden's
    database.

    The driver used to mount ``vllm-warden-data`` at ``/data`` mode "rw" in
    every engine container. That volume holds ``vllm-warden.db`` (api tokens,
    the admin-token audit trail, models, sessions) and the auto-minted
    ``jwt_secret``, so an engine — a model's code, under an image an operator or
    an admin token names — could read the database and mint a session, and the
    docker driver was no more isolated from warden state than the
    local-subprocess driver it is supposed to improve on.

    Asserted as the WHOLE volumes dict, not as "/data is absent": the point is
    that the model cache is the only thing the engine is handed, so a future
    mount has to be argued for here rather than slipped in.
    """
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="org/m", argv=["vllm", "serve"],
                      env={}, port=8001, gpu_indices=[0])
    await drv.spawn(spec)
    volumes = client.containers.run_kwargs["volumes"]
    assert volumes == {HFCACHE_VOLUME: {"bind": HFCACHE_MOUNT, "mode": "rw"}}
    assert DATA_VOLUME_NAME not in volumes
    binds = {v["bind"] for v in volumes.values()}
    assert not any(b == "/data" or b.startswith("/data/") for b in binds)


@pytest.mark.asyncio
async def test_engine_logs_are_written_by_the_warden_not_through_a_mount(tmp_path):
    """#265 — dropping the ``/data`` mount must not cost the Live-logs panel.

    The per-model log file lives under ``<data_dir>/logs``, i.e. on the volume
    the engine no longer sees. It never needed to: the WARDEN writes it, by
    streaming ``container.logs()`` from outside (``_start_log_pump``). So the
    log arrangement after this change is "the file is written, and the container
    spec carries no mount for it" — both halves asserted together, because
    either one alone would pass on a broken driver.
    """
    chunks = [b"INFO engine up\n"]
    client = _FakeClient(log_chunks=chunks)
    drv = DockerSocketDriver(client=client, image="img:tag", log_dir=str(tmp_path))
    spec = EngineSpec(model_id="m1", model_arg="org/m", argv=["vllm", "serve"],
                      env={}, port=8001, gpu_indices=[0])
    await drv.spawn(spec)

    content = _read_log_with_retry(tmp_path / "m1.log", b"".join(chunks))
    assert _split_sentinel(content) == b"".join(chunks)
    assert client.containers.run_kwargs["volumes"] == {
        HFCACHE_VOLUME: {"bind": HFCACHE_MOUNT, "mode": "rw"}
    }


@pytest.mark.asyncio
async def test_spawn_gives_the_engine_no_env_pointing_at_the_data_volume():
    """The other half of the #265 evidence, pinned.

    The engine can only need ``/data`` if something tells it to go there. The
    backends' env builders point ``HF_HUB_CACHE`` at the model cache and name no
    other directory (the 2026-06-15 ENOSPC fix), and the driver passes that env
    through unchanged apart from ``CUDA_VISIBLE_DEVICES``. If a future env key
    reintroduces a ``/data`` path, this fails next to the mount test rather than
    at 3am on a docker-driver install.
    """
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(
        model_id="m1", model_arg="org/m", argv=["vllm", "serve"],
        env={"HF_HUB_CACHE": HFCACHE_MOUNT, "VLLM_LOGGING_LEVEL": "INFO"},
        port=8001, gpu_indices=[0],
    )
    await drv.spawn(spec)
    env = client.containers.run_kwargs["environment"]
    offenders = {k: v for k, v in env.items()
                 if v == "/data" or v.startswith("/data/")}
    assert offenders == {}
