"""GPU telemetry loss is reported, not rendered as an idle card (#255).

When a systemd reload revokes a container's GPU grant (#254), nvidia-smi in
the container fails with "Failed to initialize NVML: Unknown Error" -- printed
on STDOUT, exit 255 -- while the engine that already had the GPUs open keeps
serving. These tests run the real subprocess path against a fake nvidia-smi
on PATH, so what is pinned is what a production probe does.
"""

import asyncio
import contextlib
import logging
import os
import signal
import stat
import time
from unittest.mock import AsyncMock

import pytest

from app.system import gpu as mod
from app.system.gpu import (
    NVIDIA_SMI_CMD,
    NVIDIA_SMI_LIVE_CMD,
    NVIDIA_SMI_NVLINK_CMD,
    gpu_probe_health,
    query_gpu_snapshot,
    query_gpus,
    reset_gpu_probe_health,
    reset_static_facts_cache,
)

NVML_BROKEN = """#!/bin/sh
echo "Failed to initialize NVML: Unknown Error"
exit 255
"""

# One card, answering the per-card query (NVIDIA_SMI_CMD: index, name,
# memory.total, memory.used, utilization.gpu, power.draw).
HEALTHY = """#!/bin/sh
echo "0, NVIDIA RTX A4000, 16376, 1200, 10, 30.0"
exit 0
"""

# exec makes the fake ONE process like a real nvidia-smi; without it
# `sleep` is a grandchild that keeps the stdout/stderr pipes open after
# the kill, so the transport cannot close and the test itself leaks.
HANGS = """#!/bin/sh
exec sleep 30
"""


@pytest.fixture
def _hermetic_nvidia_smi():
    """Opt out of the conftest autouse fixture: this file drives the REAL
    ``_run_nvidia_smi`` through the fake_smi PATH fake (its own monkeypatched
    module state, no nvidia-smi interception)."""
    yield


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    """Returns install(script): puts an executable nvidia-smi first on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()

    def install(script: str) -> None:
        f = bindir / "nvidia-smi"
        f.write_text(script)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    reset_gpu_probe_health()
    reset_static_facts_cache()
    yield install
    reset_gpu_probe_health()
    reset_static_facts_cache()


@pytest.fixture(autouse=True)
def _no_real_gpu_probe_in_lifespan(monkeypatch):
    # The lifespan's stats sampler would spawn real nvidia-smi / ioreg
    # probes; cancelled mid-spawn at shutdown, their subprocess transports
    # outlive the TestClient loop and print "Event loop is closed" at GC.
    monkeypatch.setattr("app.runtime.stats_sampler.query_gpus", AsyncMock(return_value=[]))


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == mod.__name__
    ]


async def test_nvml_failure_is_recorded_with_the_message_nvidia_smi_printed(fake_smi):
    fake_smi(NVML_BROKEN)
    assert await query_gpus() == []
    h = gpu_probe_health()
    assert h.state == "failing"
    # The NVML line comes from STDOUT; the old log read only stderr and
    # printed "nvidia-smi exit 255: " with nothing after it.
    assert h.error == "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error"
    assert h.since is not None and h.checked_at is not None


async def test_the_snapshot_names_the_real_reason(fake_smi):
    fake_smi(NVML_BROKEN)
    snap = await query_gpu_snapshot()
    assert snap.gpus == []
    assert snap.probe_error == "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error"


async def test_a_failure_is_logged_once_per_state_change_not_per_probe(fake_smi, caplog):
    fake_smi(NVML_BROKEN)
    caplog.set_level(logging.DEBUG, logger=mod.__name__)
    for _ in range(5):
        await query_gpus()
    warnings = _warnings(caplog)
    assert warnings == [
        "GPU telemetry unavailable: nvidia-smi exit 255: Failed to initialize NVML: Unknown Error"
    ]


async def test_recovery_is_logged_and_clears_the_error(fake_smi, caplog):
    fake_smi(NVML_BROKEN)
    await query_gpus()
    since_failing = gpu_probe_health().since
    fake_smi(HEALTHY)
    caplog.set_level(logging.DEBUG, logger=mod.__name__)
    caplog.clear()
    await asyncio.sleep(0.01)
    gpus = await query_gpus()
    assert [g.index for g in gpus] == [0]
    h = gpu_probe_health()
    assert h.state == "ok" and h.error is None
    assert h.since is not None and since_failing is not None and h.since > since_failing
    assert _warnings(caplog) == ["GPU telemetry restored: nvidia-smi is answering again"]


async def test_steady_success_does_not_log(fake_smi, caplog):
    fake_smi(HEALTHY)
    caplog.set_level(logging.DEBUG, logger=mod.__name__)
    for _ in range(3):
        await query_gpus()
    assert gpu_probe_health().state == "ok"
    assert _warnings(caplog) == []


async def test_no_nvidia_smi_is_absent_not_failing(tmp_path, monkeypatch, caplog):
    # A CPU-only install: nothing to warn about.
    monkeypatch.setenv("PATH", str(tmp_path))
    reset_gpu_probe_health()
    caplog.set_level(logging.DEBUG, logger=mod.__name__)
    try:
        assert await query_gpus() == []
        h = gpu_probe_health()
        assert h.state == "absent"
        assert h.error == "nvidia-smi not found"
        assert _warnings(caplog) == []
        snap = await query_gpu_snapshot()
        assert snap.probe_error == "nvidia-smi not found"
    finally:
        reset_gpu_probe_health()


async def test_a_hung_nvidia_smi_is_killed_and_reported(fake_smi):
    fake_smi(HANGS)
    started = time.monotonic()
    out = await mod._run_nvidia_smi(NVIDIA_SMI_CMD, timeout=0.3)
    assert out is None
    assert time.monotonic() - started < 5
    h = gpu_probe_health()
    assert h.state == "failing"
    assert h.error == "nvidia-smi did not answer within 0.3 s"


def _spy_subprocesses(monkeypatch, procs: list) -> None:
    real = asyncio.create_subprocess_exec

    async def spy(*a, **kw):
        p = await real(*a, **kw)
        procs.append(p)
        return p

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)


async def test_a_timed_out_nvidia_smi_child_is_reaped(fake_smi, monkeypatch):
    fake_smi(HANGS)
    procs: list = []
    _spy_subprocesses(monkeypatch, procs)
    assert await mod._run_nvidia_smi(NVIDIA_SMI_CMD, timeout=0.3) is None
    assert len(procs) == 1
    assert procs[0].returncode == -signal.SIGKILL


async def test_a_timed_out_ioreg_child_is_reaped(tmp_path, monkeypatch):
    hang = tmp_path / "hang"
    hang.write_text(HANGS)
    hang.chmod(hang.stat().st_mode | stat.S_IEXEC)
    procs: list = []
    _spy_subprocesses(monkeypatch, procs)
    assert await mod._run_ioreg([str(hang)], timeout=0.3) is None
    assert procs[0].returncode == -signal.SIGKILL


async def test_the_reap_wait_is_bounded(fake_smi, monkeypatch):
    fake_smi(HANGS)
    procs: list = []

    async def never():
        await asyncio.Event().wait()

    def wrapped(real):
        async def spy(*a, **kw):
            p = await real(*a, **kw)
            p.wait = never
            procs.append(p)
            return p

        return spy

    monkeypatch.setattr(asyncio, "create_subprocess_exec", wrapped(asyncio.create_subprocess_exec))
    try:
        started = time.monotonic()
        out = await mod._run_nvidia_smi(NVIDIA_SMI_CMD, timeout=0.3)
        elapsed = time.monotonic() - started
        assert out is None
        # 0.3 s probe timeout + the 1.0 s reap bound: the lower bound proves
        # the wait ran, the upper bound that it is bounded.
        assert 1.2 <= elapsed < 3.0
        assert gpu_probe_health().state == "failing"
    finally:
        with contextlib.suppress(ProcessLookupError):
            asyncio.subprocess.Process.kill(procs[0])
        await asyncio.subprocess.Process.wait(procs[0])


async def test_a_child_already_gone_does_not_raise(fake_smi, monkeypatch):
    fake_smi(HANGS)
    procs: list = []

    def gone():
        raise ProcessLookupError

    async def gone_wait():
        raise ProcessLookupError

    def wrapped(real):
        async def spy(*a, **kw):
            p = await real(*a, **kw)
            p.kill = gone
            p.wait = gone_wait
            procs.append(p)
            return p

        return spy

    monkeypatch.setattr(asyncio, "create_subprocess_exec", wrapped(asyncio.create_subprocess_exec))
    try:
        assert await mod._run_nvidia_smi(NVIDIA_SMI_CMD, timeout=0.3) is None
        assert gpu_probe_health().state == "failing"
    finally:
        with contextlib.suppress(ProcessLookupError):
            asyncio.subprocess.Process.kill(procs[0])
        await asyncio.subprocess.Process.wait(procs[0])


async def test_an_optional_probe_failure_does_not_touch_the_state(fake_smi, caplog):
    # NVLink / thermal / topology queries fail on some drivers by design.
    fake_smi(HEALTHY)
    await query_gpus()
    fake_smi(NVML_BROKEN)
    caplog.set_level(logging.DEBUG, logger=mod.__name__)
    caplog.clear()
    assert await mod._run_nvidia_smi(NVIDIA_SMI_NVLINK_CMD) is None
    assert gpu_probe_health().state == "ok"
    assert _warnings(caplog) == []


async def test_both_primary_probes_feed_the_state(fake_smi):
    fake_smi(NVML_BROKEN)
    assert await mod._run_nvidia_smi(NVIDIA_SMI_LIVE_CMD) is None
    assert gpu_probe_health().state == "failing"


def test_healthz_stays_200_and_carries_the_gpu_state(client):
    reset_gpu_probe_health()
    mod._record_primary_probe(
        "failing", "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error"
    )
    try:
        r = client.get("/healthz")
        # Liveness must not fail over telemetry: a restart would kill a
        # still-serving engine.
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["gpu"]["state"] == "failing"
        assert (
            body["gpu"]["error"] == "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error"
        )
    finally:
        reset_gpu_probe_health()
