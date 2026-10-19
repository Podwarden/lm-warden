"""Tests for ``sample_vitals`` and ``LogFollower``-adjacent vitals — Task 3.

The vitals sampler walks the engine's process group (pgrp == engine pid, since
the supervisor spawns every engine with ``start_new_session=True``) and sums
CPU time (``utime+stime`` ÷ ``SC_CLK_TCK``) and disk read bytes (``rchar``)
from ``/proc/<pid>/io``. GPU memory is the sum of ``GpuComputeApp.memory_mib``
for compute apps whose pid is in the engine's pgrp set. Every probe is wrapped:
exception or > 3 s → that field is ``None``; the sampler itself never raises
and never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from app.runtime.load_progress.vitals import default_compute_apps, sample_vitals
from app.system.gpu import GpuComputeApp

# ---------------------------------------------------------------------------
# Fake /proc helpers
# ---------------------------------------------------------------------------

_CLK_TCK = 100  # the usual Linux value; we hard-code it in the fake stat


def _write_stat(
    proc_root: Path,
    pid: int,
    *,
    comm: str = "vllm",
    ppid: int = 1,
    pgrp: int | None = None,
    utime: int = 0,
    stime: int = 0,
) -> None:
    """Write a fake /proc/<pid>/stat.

    /proc/<pid>/stat layout: ``pid (comm) state ppid pgrp session tty_nr tpgid
    flags ... utime stime cutime cstime ...``. Fields after the closing ``)`` are
    1-indexed: state=1, ppid=2, pgrp=3, session=4, ..., utime=14, stime=15.
    """
    if pgrp is None:
        pgrp = pid
    d = proc_root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    # After ')' the fields (0-based) are:
    #   0 state  1 ppid  2 pgrp  3 session  4 tty_nr  5 tpgid  6 flags
    #   7 minflt  8 cminflt  9 majflt  10 cmajflt  11 utime  12 stime
    # We pad indices 4-10 (7 zeros) then put utime, stime at 11 and 12.
    fields = ["S", str(ppid), str(pgrp), "1", "0", "0", "0", "0", "0", "0", "0"]
    fields += [str(utime), str(stime)]
    rest = " ".join(fields)
    (d / "stat").write_text(f"{pid} ({comm}) {rest}\n")


def _write_io(proc_root: Path, pid: int, *, rchar: int = 0) -> None:
    """Write a fake /proc/<pid>/io. We only consume ``rchar``."""
    d = proc_root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "io").write_text(
        f"rchar: {rchar}\nwchar: 0\nsyscr: 0\nsyscw: 0\n"
        f"read_bytes: 0\nwrite_bytes: 0\ncancelled_write_bytes: 0\n"
    )


def _make_engine_tree(
    proc_root: Path,
    engine_pid: int,
    *,
    worker_pids: list[int],
    utime: int = 0,
    stime: int = 0,
    rchar: int = 0,
    foreign_pids: list[tuple[int, int]] | None = None,
) -> None:
    """Write a fake process tree.

    ``engine_pid`` is the pgrp leader; ``worker_pids`` share that pgrp.
    ``foreign_pids`` is a list of ``(pid, pgrp)`` tuples for unrelated processes
    that must NOT be counted (different pgrp).
    """
    _write_stat(proc_root, engine_pid, pgrp=engine_pid, utime=utime, stime=stime)
    _write_io(proc_root, engine_pid, rchar=rchar)
    for w in worker_pids:
        _write_stat(proc_root, w, pgrp=engine_pid, utime=utime, stime=stime)
        _write_io(proc_root, w, rchar=rchar)
    for pid, pgrp in foreign_pids or []:
        _write_stat(proc_root, pid, pgrp=pgrp, utime=9999, stime=9999)
        _write_io(proc_root, pid, rchar=999_999_999)


# ---------------------------------------------------------------------------
# Local driver (subprocess) vitals
# ---------------------------------------------------------------------------


async def _no_apps() -> list[GpuComputeApp]:
    return []


async def test_sums_cpu_and_rchar_over_tree(tmp_path: Path) -> None:
    """CPU seconds = sum over pgrp members of (utime+stime) / SC_CLK_TCK.
    read_bytes = sum of /proc/<pid>/io rchar over pgrp members."""
    engine = 1000
    workers = [1001, 1002, 1003]
    _make_engine_tree(tmp_path, engine, worker_pids=workers, utime=200, stime=100, rchar=1_000_000)
    # 4 processes × (200 utime + 100 stime) / 100 Hz = 4 × 3.0 = 12.0 s
    # 4 processes × 1_000_000 rchar = 4_000_000 bytes

    sample = await sample_vitals(
        pid=engine,
        handle=None,
        driver=_LocalDriverStub(),
        proc_root=tmp_path,
        compute_apps=_no_apps,
    )
    assert sample.cpu_seconds == pytest.approx(12.0)
    assert sample.read_bytes == 4_000_000
    assert sample.gpu_mem_mib is None  # no compute apps reported
    assert sample.at > 0


async def test_pgrp_members_only(tmp_path: Path) -> None:
    """A process in a different pgrp must NOT be summed even if it is visible
    under the same /proc root."""
    engine = 2000
    workers = [2001]
    # An unrelated process with a different pgrp and very large CPU/IO.
    foreign = [(9999, 8888)]
    _make_engine_tree(
        tmp_path, engine, worker_pids=workers, utime=100, stime=0, rchar=500, foreign_pids=foreign
    )

    sample = await sample_vitals(
        pid=engine,
        handle=None,
        driver=_LocalDriverStub(),
        proc_root=tmp_path,
        compute_apps=_no_apps,
    )
    # Only 2 processes (engine + 1 worker) × (100 utime + 0 stime) / 100 = 2.0 s
    assert sample.cpu_seconds == pytest.approx(2.0)
    assert sample.read_bytes == 1_000  # 2 × 500
    # The foreign process's huge values are excluded.


async def test_gpu_mem_sums_tree_pids_only(tmp_path: Path) -> None:
    """GPU memory = sum of GpuComputeApp.memory_mib for compute apps whose pid
    is in the engine's pgrp set."""
    engine = 3000
    workers = [3001, 3002]
    _make_engine_tree(tmp_path, engine, worker_pids=workers, utime=0, stime=0, rchar=0)

    apps = [
        GpuComputeApp(pid=engine, gpu_uuid="GPU-0", process_name="vllm", memory_mib=1024),
        GpuComputeApp(pid=3001, gpu_uuid="GPU-0", process_name="worker", memory_mib=2048),
        # A different engine's process — different pgrp, must be excluded.
        GpuComputeApp(pid=5555, gpu_uuid="GPU-0", process_name="vllm", memory_mib=8192),
    ]

    async def _apps() -> list[GpuComputeApp]:
        return apps

    sample = await sample_vitals(
        pid=engine,
        handle=None,
        driver=_LocalDriverStub(),
        proc_root=tmp_path,
        compute_apps=_apps,
    )
    assert sample.gpu_mem_mib == 1024 + 2048


async def test_unattributable_gpu_pids_give_none(tmp_path: Path) -> None:
    """If nvidia-smi reports a pid that is not visible in this /proc at all
    (different pid namespace), the GPU field is ``None`` — never count
    unattributable memory."""
    engine = 4000
    _make_engine_tree(tmp_path, engine, worker_pids=[], utime=0, stime=0, rchar=0)

    # All reported pids are invisible in our /proc.
    apps = [
        GpuComputeApp(pid=99999, gpu_uuid="GPU-0", process_name="vllm", memory_mib=4096),
        GpuComputeApp(pid=99998, gpu_uuid="GPU-0", process_name="worker", memory_mib=2048),
    ]

    async def _apps() -> list[GpuComputeApp]:
        return apps

    sample = await sample_vitals(
        pid=engine,
        handle=None,
        driver=_LocalDriverStub(),
        proc_root=tmp_path,
        compute_apps=_apps,
    )
    # No compute-app pid was attributable to our pgrp → GPU mem None.
    assert sample.gpu_mem_mib is None


async def test_nvidia_smi_failure_gives_none_gpu(tmp_path: Path) -> None:
    """If the compute_apps provider returns nothing (nvidia-smi failed / absent),
    GPU memory is None — not 0, not an error."""

    async def _empty_apps() -> list[GpuComputeApp]:
        return []

    engine = 5000
    _make_engine_tree(tmp_path, engine, worker_pids=[5001], utime=10, stime=0, rchar=42)

    sample = await sample_vitals(
        pid=5000,
        handle=None,
        driver=_LocalDriverStub(),
        proc_root=tmp_path,
        compute_apps=_empty_apps,
    )
    assert sample.gpu_mem_mib is None


async def test_hanging_compute_apps_times_out_to_none(tmp_path: Path) -> None:
    """A compute_apps coroutine that sleeps 10 s must time out at 3 s —
    gpu_mem_mib becomes None, the call returns in well under 4 s, and the
    other fields are still populated."""
    engine = 6000
    _make_engine_tree(tmp_path, engine, worker_pids=[6001], utime=10, stime=5, rchar=100)

    async def _slow_apps() -> list[GpuComputeApp]:
        await asyncio.sleep(10)
        return []

    t0 = time.monotonic()
    sample = await sample_vitals(
        pid=engine,
        handle=None,
        driver=_LocalDriverStub(),
        proc_root=tmp_path,
        compute_apps=_slow_apps,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 4.0, f"sample_vitals took {elapsed:.1f}s, should be < 4s"
    assert sample.gpu_mem_mib is None
    # CPU and read_bytes still populated — they don't depend on nvidia-smi.
    assert sample.cpu_seconds is not None
    assert sample.read_bytes is not None


def test_vanished_pid_gives_none_not_raise(tmp_path: Path) -> None:
    """If the engine pid has already exited (no /proc entry), every field
    that depends on the process tree is None and sample_vitals does NOT raise."""
    # No /proc entries written at all.
    sample = asyncio.run(
        sample_vitals(
            pid=7777,
            handle=None,
            driver=_LocalDriverStub(),
            proc_root=tmp_path,
            compute_apps=_no_apps,
        )
    )
    assert sample.cpu_seconds is None
    assert sample.read_bytes is None
    assert sample.gpu_mem_mib is None
    assert sample.at > 0


# ---------------------------------------------------------------------------
# Docker driver stats
# ---------------------------------------------------------------------------


class _LocalDriverStub:
    """A minimal stand-in for LocalSubprocessDriver — has no ``stats`` method,
    so sample_vitals uses the /proc path."""

    supports_engine_image = False


class _DockerDriverStub:
    """A stand-in for DockerSocketDriver with a ``stats`` method."""

    supports_engine_image = True

    def __init__(self, stats: dict | None) -> None:
        self._stats = stats

    async def stats(self, handle: object | None) -> dict | None:
        return self._stats


def _docker_stats(*, cpu_ns: int, read_bytes: int) -> dict[str, Any]:
    """A minimal docker container.stats(stream=False) dict with only the
    fields sample_vitals reads."""
    return {
        "cpu_stats": {
            "cpu_usage": {"total_usage": cpu_ns},
            "system_cpu_usage": 0,
            "online_cpus": 1,
        },
        "blkio_stats": {
            "io_service_bytes_recursive": [
                {"op": "Read", "value": read_bytes},
                {"op": "Write", "value": 0},
                {"op": "Total", "value": read_bytes},
            ],
        },
    }


def test_docker_uses_cpu_and_blkio_from_stats(tmp_path: Path) -> None:
    """Docker driver: cpu_stats.cpu_usage.total_usage (ns → s) and
    blkio_stats Read bytes. GPU memory None unless the container's pids are
    visible in /proc."""
    stats = _docker_stats(cpu_ns=2_500_000_000, read_bytes=8_000_000)
    driver = _DockerDriverStub(stats=stats)
    handle = object()

    sample = asyncio.run(
        sample_vitals(
            pid=None,
            handle=handle,
            driver=driver,
            proc_root=tmp_path,
            compute_apps=_no_apps,
        )
    )
    assert sample.cpu_seconds == pytest.approx(2.5)
    assert sample.read_bytes == 8_000_000
    assert sample.gpu_mem_mib is None


def test_docker_driver_gpu_mem_is_none_by_design(tmp_path: Path) -> None:
    """Docker-driver engines run in sibling containers whose pids are not in
    the warden's /proc namespace, so GPU memory is None by design — even when
    compute_apps reports pids that would be attributable if this were the
    local driver. The docker driver must NOT attempt GPU attribution."""
    stats = _docker_stats(cpu_ns=1_000_000_000, read_bytes=4_000_000)
    driver = _DockerDriverStub(stats=stats)

    # Simulate nvidia-smi reporting GPU holders. Under the local driver these
    # pids would be checked against the pgrp set; under the docker driver they
    # are in a different pid namespace and must never be counted.
    apps = [
        GpuComputeApp(pid=1234, gpu_uuid="GPU-0", process_name="vllm", memory_mib=4096),
        GpuComputeApp(pid=5678, gpu_uuid="GPU-0", process_name="worker", memory_mib=2048),
    ]

    async def _apps() -> list[GpuComputeApp]:
        return apps

    sample = asyncio.run(
        sample_vitals(
            pid=None,
            handle=object(),
            driver=driver,
            proc_root=tmp_path,
            compute_apps=_apps,
        )
    )
    assert sample.gpu_mem_mib is None, (
        "docker driver does not attribute GPU memory (sibling-container pids "
        "are in a different pid namespace)"
    )
    # CPU and read_bytes still come from docker stats.
    assert sample.cpu_seconds == pytest.approx(1.0)
    assert sample.read_bytes == 4_000_000


def test_docker_stats_none_gives_none_fields(tmp_path: Path) -> None:
    """If driver.stats() returns None (container gone / daemon error), cpu
    and read_bytes are None."""
    driver = _DockerDriverStub(stats=None)
    sample = asyncio.run(
        sample_vitals(
            pid=None,
            handle=object(),
            driver=driver,
            proc_root=tmp_path,
            compute_apps=_no_apps,
        )
    )
    assert sample.cpu_seconds is None
    assert sample.read_bytes is None
    assert sample.gpu_mem_mib is None


def test_docker_stats_exception_gives_none(tmp_path: Path) -> None:
    """If driver.stats() raises, cpu and read_bytes are None (not propagated)."""

    class _BoomDriver:
        supports_engine_image = True

        async def stats(self, handle: object | None) -> dict | None:
            raise RuntimeError("docker daemon exploded")

    sample = asyncio.run(
        sample_vitals(
            pid=None,
            handle=object(),
            driver=_BoomDriver(),
            proc_root=tmp_path,
            compute_apps=_no_apps,
        )
    )
    assert sample.cpu_seconds is None
    assert sample.read_bytes is None


# ---------------------------------------------------------------------------
# default_compute_apps
# ---------------------------------------------------------------------------


def test_default_compute_apps_absent_nvidia_smi_returns_empty(monkeypatch) -> None:
    """If nvidia-smi is not on PATH, default_compute_apps returns [] (not raise)."""

    async def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    # create_subprocess_exec is what default_compute_apps uses.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
    result = asyncio.run(default_compute_apps())
    assert result == []


def test_default_compute_apps_parses_csv(monkeypatch) -> None:
    """default_compute_apps pipes the nvidia-smi stdout through
    parse_compute_apps_csv."""
    csv = "1234, GPU-0, python, 1024\n5678, GPU-1, python, 2048\n"

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (csv.encode(), b"")

    async def _fake_exec(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    result = asyncio.run(default_compute_apps())
    assert len(result) == 2
    assert result[0].pid == 1234 and result[0].memory_mib == 1024
    assert result[1].pid == 5678 and result[1].memory_mib == 2048
