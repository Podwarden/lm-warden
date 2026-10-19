"""Sample how much CPU, GPU memory and disk reading the engine's processes
are doing — the "vital signs" that let the load-progress tracker tell a
slow-but-busy load from a stuck one.

Process-tree model (ruling in the plan):
  The supervisor spawns every engine with ``start_new_session=True``, so the
  engine's parent pid IS its process-group leader and every vLLM/llama.cpp
  worker shares that pgrp (see ``app/system/pid_attribution.py``). "The engine's
  process tree" is therefore every ``/proc/<n>`` whose pgrp == the engine pid
  — this also catches workers that got re-parented to init when their parent
  died. We do NOT walk ppid, because a re-parented worker's ppid is 1, not the
  engine pid.

GPU memory pid namespace:
  The warden and the engine run in the same container in the default (local
  subprocess) driver, and the existing GPU probe already maps
  ``nvidia-smi --query-compute-apps`` pids to models via pgrp
  (``app/system/pid_attribution.py``, ``app/system/gpu.py``). We do the same
  here: sum ``memory_mib`` of compute apps whose pid is in the engine's pgrp
  set. If nvidia-smi reports pids that are not visible in this /proc at all
  (different pid namespace), the GPU field is ``None`` — never count
  unattributable memory.

Docker driver:
  ``DockerSocketDriver.stats(handle)`` returns the container's stats dict. We
  read ``cpu_stats.cpu_usage.total_usage`` (ns → s) and
  ``blkio_stats.io_service_bytes_recursive`` Read ops. GPU memory is None
  unless the container's pids are visible in this /proc (same rule as local).

Every probe is wrapped: exception or > 3 s → that field ``None``.
``sample_vitals`` itself never raises and never blocks the event loop — /proc
scans and nvidia-smi run via ``asyncio.to_thread`` with ``asyncio.wait_for(..., 3)``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.runtime.load_progress.tracker import VitalsSample
from app.system.gpu import (
    NVIDIA_SMI_APPS_CMD,
    GpuComputeApp,
    parse_compute_apps_csv,
)
from app.system.pid_attribution import read_pgid

logger = logging.getLogger(__name__)

__all__ = ["sample_vitals", "default_compute_apps"]

# Every probe is individually bounded at 3 s. Anything slower is treated as a
# failure for that field (→ None) rather than a stall of the load-progress loop.
_PROBE_TIMEOUT = 3.0

# /proc/<pid>/stat clock-tick divisor. ``os.sysconf`` may not be available in
# every test environment; we fall back to 100 (the Linux default on essentially
# every distribution). Callers that construct a fake /proc tree for tests use
# this same divisor.
try:
    _CLK_TCK = int(os.sysconf("SC_CLK_TCK"))
except (ValueError, OSError, AttributeError):
    _CLK_TCK = 100


async def sample_vitals(
    *,
    pid: int | None,
    handle: object | None,
    driver: object,
    proc_root: Path = Path("/proc"),
    compute_apps: Callable[[], Awaitable[list[GpuComputeApp]]],
) -> VitalsSample:
    """Collect one vital-sign sample for an engine load.

    Parameters
    ----------
    pid
        The engine's host pid. **Local driver:** pass the engine pid — the
        pgrp walk and GPU attribution both key on it. **Docker driver:** pass
        ``None`` — the engine runs in a sibling container whose pids are not in
        the warden's /proc namespace, so there is no pgrp to walk and
        ``gpu_mem_mib`` is ``None`` by design (see ``DockerSocketDriver``).
    handle
        The engine handle (``LocalHandle`` / ``DockerHandle``). Used by the
        docker driver's ``stats`` call.
    driver
        The engine driver. If it exposes ``async def stats(handle) -> dict |
        None``, the docker path is used for CPU and read_bytes; otherwise the
        /proc walk is used (local-subprocess driver).
    proc_root
        Root of the /proc filesystem to scan. Tests pass a fake tree.
    compute_apps
        A callable returning the current ``nvidia-smi --query-compute-apps``
        result (already parsed). In production this is
        :func:`default_compute_apps`; tests inject a stub.

    Returns
    -------
    VitalsSample
        ``at`` is a monotonic timestamp. Any field may be ``None`` when the
        host could not measure it. ``None`` vitals never count as progress.

    GPU memory by driver
    --------------------
    **Local driver:** GPU memory is the sum of ``GpuComputeApp.memory_mib`` for
    compute apps whose pid is in the engine's pgrp set. ``None`` if no app is
    attributable (never counts unattributable memory).
    **Docker driver:** GPU memory is always ``None`` — the engine's sibling
    container runs in a different pid namespace, so nvidia-smi's pids are not
    visible in the warden's /proc and cannot be attributed. This is by design,
    not a fallback.
    """
    at = time.monotonic()

    cpu_seconds: float | None = None
    read_bytes: int | None = None
    gpu_mem_mib: int | None = None
    pgrp_pids: set[int] | None = None

    # ----- pick the CPU / read_bytes source ---------------------------------
    # The docker driver exposes ``async def stats(handle)``; the local
    # subprocess driver does not. Its presence is what selects the docker path.
    stats_fn = getattr(driver, "stats", None)
    use_docker = callable(stats_fn)

    if use_docker:
        cpu_seconds, read_bytes = await _safe_docker_cpu_read(driver, handle)
    else:
        # Local-subprocess driver: walk the pgrp.
        pgrp_pids = await _safe_pgrp_set(pid, proc_root)
        if pgrp_pids:
            cpu_seconds, read_bytes = await _safe_proc_cpu_read(pgrp_pids, proc_root)

    # ----- GPU memory (local driver only) -----------------------------------
    # GPU memory is attributed by pgrp membership, which requires the engine's
    # pid to be visible in this /proc. The docker driver's engine runs in a
    # sibling container (different pid namespace), so pid is None and
    # gpu_mem_mib stays None BY DESIGN — never attempt attribution for the
    # docker driver.
    if not use_docker and pgrp_pids is None and pid is not None:
        pgrp_pids = await _safe_pgrp_set(pid, proc_root)
    if pgrp_pids:
        gpu_mem_mib = await _safe_gpu_mem(compute_apps, pgrp_pids, proc_root)

    return VitalsSample(
        at=at,
        cpu_seconds=cpu_seconds,
        gpu_mem_mib=gpu_mem_mib,
        read_bytes=read_bytes,
    )


async def _safe_docker_cpu_read(
    driver: object,
    handle: object | None,
) -> tuple[float | None, int | None]:
    """Call ``driver.stats(handle)`` and extract cpu_seconds and read_bytes.

    Returns (None, None) on any exception or timeout.
    """
    stats_fn = getattr(driver, "stats")  # noqa: B009  — driver is typed object; attribute access would fail mypy
    try:
        stats: object = await asyncio.wait_for(stats_fn(handle), timeout=_PROBE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — every failure is a None field.
        logger.debug("docker stats() failed: %s", exc)
        return (None, None)
    if not isinstance(stats, dict):
        return (None, None)

    cpu_seconds: float | None = None
    read_bytes: int | None = None
    try:
        cpu_total_ns = stats["cpu_stats"]["cpu_usage"]["total_usage"]
        cpu_seconds = float(cpu_total_ns) / 1_000_000_000.0
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("docker cpu_stats parse failed: %s", exc)
    try:
        ops = stats["blkio_stats"]["io_service_bytes_recursive"]
        read_bytes = 0
        for op in ops or []:
            if isinstance(op, dict) and op.get("op") == "Read":
                read_bytes += int(op.get("value", 0))
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("docker blkio_stats parse failed: %s", exc)
        read_bytes = None
    return (cpu_seconds, read_bytes)


async def _safe_pgrp_set(
    pid: int | None,
    proc_root: Path,
) -> set[int] | None:
    """Return the set of pids whose pgrp == ``pid``, plus ``pid`` itself.

    Returns ``None`` if ``pid`` is None or the pgrp walk failed. Returns the
    empty set if ``pid`` has already exited (no /proc entry) — callers treat
    that as "no processes to sample" → None fields.
    """
    if pid is None:
        return None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_pgrp_members, pid, proc_root),
            timeout=_PROBE_TIMEOUT,
        )
    except TimeoutError:
        logger.debug("pgrp walk timed out for pid %s", pid)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("pgrp walk failed for pid %s: %s", pid, exc)
        return None


def _pgrp_members(pid: int, proc_root: Path) -> set[int]:
    """Synchronous /proc scan: every pid whose pgrp == ``pid``.

    Includes ``pid`` itself (the supervisor spawns the engine with
    ``start_new_session=True``, so pgrp == pid for the leader too). Workers
    re-parented to init still carry the same pgrp, so this catches them.
    """
    members: set[int] = {pid}
    # The leader's own pgrp should equal pid, but verify defensively — if the
    # leader has exited and its /proc entry is gone, we still want any workers
    # that share its old pgrp.
    try:
        entries = list(proc_root.iterdir())
    except (FileNotFoundError, PermissionError, OSError):
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        other = int(entry.name)
        if other == pid:
            continue
        pgid = read_pgid(other, proc_root=proc_root)
        if pgid == pid:
            members.add(other)
    return members


async def _safe_proc_cpu_read(
    pids: set[int],
    proc_root: Path,
) -> tuple[float | None, int | None]:
    """Sum utime+stime (÷ CLK_TCK) and rchar over the pgrp's pids.

    Returns (None, None) if the scan failed or timed out. A pid that has
    vanished mid-scan contributes 0 — the rest of the tree still sums.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_proc_cpu_read, pids, proc_root),
            timeout=_PROBE_TIMEOUT,
        )
    except TimeoutError:
        logger.debug("proc cpu read timed out for pids %s", pids)
        return (None, None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("proc cpu read failed for pids %s: %s", pids, exc)
        return (None, None)


def _proc_cpu_read(pids: set[int], proc_root: Path) -> tuple[float | None, int | None]:
    """Synchronous sum of utime+stime and rchar over ``pids``.

    Returns ``(None, None)`` only if the entire set is empty; otherwise sums
    what it can. A pid that vanished mid-scan contributes 0.
    """
    if not pids:
        return (None, None)
    total_ticks = 0
    total_rchar = 0
    saw_any = False
    for pid in pids:
        ticks = _read_stat_ticks(pid, proc_root)
        if ticks is not None:
            total_ticks += ticks
            saw_any = True
        rchar = _read_io_rchar(pid, proc_root)
        if rchar is not None:
            total_rchar += rchar
            saw_any = True
    if not saw_any:
        return (None, None)
    cpu_seconds = total_ticks / _CLK_TCK if _CLK_TCK > 0 else None
    return (cpu_seconds, total_rchar)


def _read_stat_ticks(pid: int, proc_root: Path) -> int | None:
    """Return utime + stime (clock ticks) for ``pid``, or None if unreadable.

    /proc/<pid>/stat layout after the closing ``)``: state ppid pgrp session
    tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime ...
    utime is field 14, stime is field 15 (1-indexed including the pid and the
    parenthesized comm). After the ``)`` they are at 0-based indices 11 and 12.
    """
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    rparen = raw.rfind(")")
    if rparen == -1:
        return None
    rest = raw[rparen + 1 :].split()
    # After ')': state(0) ppid(1) pgrp(2) session(3) tty_nr(4) tpgid(5)
    # flags(6) minflt(7) cminflt(8) majflt(9) cmajflt(10) utime(11) stime(12)
    if len(rest) < 13:
        return None
    try:
        return int(rest[11]) + int(rest[12])
    except (ValueError, IndexError):
        return None


def _read_io_rchar(pid: int, proc_root: Path) -> int | None:
    """Return ``rchar`` from /proc/<pid>/io, or None if unreadable.

    ``rchar`` is the bytes read through the syscall interface (read, pread,
    readv) — the right counter for "the engine is pulling weights off disk".
    It counts reads from the page cache too, which is exactly what we want:
    a cold load reads from the SSD and a warm one reads from the page cache,
    but both keep the engine busy.
    """
    try:
        raw = (proc_root / str(pid) / "io").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    for line in raw.splitlines():
        if line.startswith("rchar:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


async def _safe_gpu_mem(
    compute_apps: Callable[[], Awaitable[list[GpuComputeApp]]],
    pgrp_pids: set[int],
    proc_root: Path,
) -> int | None:
    """Sum GPU memory for compute apps whose pid is in the engine's pgrp set.

    Returns ``None`` if the compute-apps call failed/timed out, or if NO
    compute-app pid was attributable to the pgrp (different pid namespace).
    Never counts unattributable memory.
    """
    try:
        apps = await asyncio.wait_for(compute_apps(), timeout=_PROBE_TIMEOUT)
    except TimeoutError:
        logger.debug("compute_apps timed out")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("compute_apps failed: %s", exc)
        return None

    if not apps:
        return None

    total = 0
    saw_attributable = False
    for app in apps:
        if app.pid in pgrp_pids:
            total += app.memory_mib
            saw_attributable = True
    if not saw_attributable:
        # No compute-app pid was attributable to our pgrp. This includes the
        # pid-namespace-mismatch case (host pids invisible in our /proc): the
        # GPU field is None, never an unattributed sum.
        return None
    return total


async def default_compute_apps() -> list[GpuComputeApp]:
    """Run ``nvidia-smi --query-compute-apps`` and parse the CSV.

    Reuses the existing command and parser from ``app/system/gpu.py``. Returns
    ``[]`` if nvidia-smi is absent, fails, or times out — callers treat that as
    "no GPU memory attributable" (→ ``None`` gpu_mem_mib).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *NVIDIA_SMI_APPS_CMD,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as exc:
        logger.debug("nvidia-smi not available for compute_apps: %s", exc)
        return []
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_PROBE_TIMEOUT)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-smi compute_apps failed: %s", exc)
        return []
    if proc.returncode != 0:
        return []
    return parse_compute_apps_csv(stdout.decode())
