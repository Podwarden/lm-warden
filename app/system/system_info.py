"""Hardware + OS + Docker inventory probe for ``GET /api/system/info`` (#148).

The stats page needs static system context (CPU model, total RAM, GPU model
+ VRAM + driver/CUDA versions, OS, Docker runtime) to make the live numbers
on the same page interpretable. None of these values change minute-to-minute,
and several of them require shelling out to ``nvidia-smi`` / ``docker info``,
so we cache the assembled response in-process for 60 s.

The probe functions in this module are split out from the route handler so
the unit tests can drive each source in isolation with mocked
``subprocess.run`` / fake file reads.

When a source is unavailable (``nvidia-smi`` not on PATH, ``docker info``
denied because the API container has no docker socket, ``/proc/cpuinfo``
empty in a non-Linux sandbox) we return ``None`` / an empty list rather
than raising, so the endpoint never 500s. The UI renders "unavailable" in
the missing slot instead of failing the whole panel.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 60s cache — issue #148 AC: "Cache the response for 60 seconds (hardware
# does not change minute-to-minute; nvidia-smi shell-outs are non-trivial)."
CACHE_TTL_SECONDS = 60.0

# Wall-clock timeout for each blocking subprocess. ``nvidia-smi`` typically
# returns in <100 ms but a wedged driver can hang a process for minutes;
# fail-fast keeps the API responsive. ``docker info`` also has hard timeouts
# upstream but we add our own for the same reason.
_SUBPROCESS_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# CPU — /proc/cpuinfo
# ---------------------------------------------------------------------------


def _read_proc_cpuinfo() -> str:
    """Return the contents of ``/proc/cpuinfo`` or ``""`` if unavailable.

    Split out so tests can monkeypatch the file source without touching
    the filesystem.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("/proc/cpuinfo unreadable: %s", e)
        return ""


def parse_cpuinfo(text: str) -> dict[str, Any] | None:
    """Parse ``/proc/cpuinfo`` text into ``{model, physical_cores, threads}``.

    - ``model`` — value of the first ``model name`` field (all processors
      report the same string on every system we'll encounter).
    - ``threads`` — number of ``processor`` lines (one per logical CPU).
    - ``physical_cores`` — count of unique ``(physical id, core id)`` pairs,
      so a hyperthreaded box reports the true physical-core count and not
      the thread count. ARM SoCs and some VMs omit ``physical id``/
      ``core id``; in that case we fall back to ``threads`` (best we can do
      without lscpu).

    Returns ``None`` when ``text`` is empty so the caller can mark CPU as
    unavailable rather than emitting bogus zeros.
    """
    if not text.strip():
        return None

    blocks = [b for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return None

    threads = len(blocks)
    model: str | None = None
    physical_pairs: set[tuple[str, str]] = set()

    for block in blocks:
        phys_id: str | None = None
        core_id: str | None = None
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if model is None and key == "model name":
                model = value
            elif key == "physical id":
                phys_id = value
            elif key == "core id":
                core_id = value
        if phys_id is not None and core_id is not None:
            physical_pairs.add((phys_id, core_id))

    physical_cores = len(physical_pairs) if physical_pairs else threads

    return {
        "model": model or "unknown",
        "physical_cores": physical_cores,
        "threads": threads,
    }


# ---------------------------------------------------------------------------
# RAM — /proc/meminfo (preferred) or `free -m` fallback.
# ---------------------------------------------------------------------------


def _read_proc_meminfo() -> str:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("/proc/meminfo unreadable: %s", e)
        return ""


def parse_meminfo(text: str) -> dict[str, Any] | None:
    """Parse ``/proc/meminfo`` for ``MemTotal`` (kB) → ``{total_mb}``.

    ``/proc/meminfo`` reports kB explicitly (it's the only file in /proc
    that uses kB for memory sizes). We round-divide by 1024 so a 64 GiB
    box reports 65536 MB, matching what ``free -m`` would print.

    Returns ``None`` if the file is empty or ``MemTotal`` is missing.
    """
    if not text:
        return None
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    if not m:
        return None
    kb = int(m.group(1))
    return {"total_mb": kb // 1024}


# ---------------------------------------------------------------------------
# GPU — nvidia-smi
# ---------------------------------------------------------------------------


_NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]
_NVIDIA_SMI_VERSION = ["nvidia-smi", "--version"]
_NVIDIA_SMI_BANNER = ["nvidia-smi"]


def parse_nvidia_smi_gpus(stdout: str, cuda_version: str | None) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu`` CSV into the per-GPU payload list.

    ``cuda_version`` is reported globally by ``nvidia-smi --version`` (it's
    the CUDA toolkit the driver was built against, not a per-GPU value).
    We attach it to each GPU object so the frontend doesn't have to model
    "one GPU array + one global CUDA version" as two separate slots.
    """
    out: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            out.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    # nvidia-smi memory.total with --nounits reports MiB.
                    # The issue spec asks for vram_total_mb — keep the field
                    # name as documented even though the unit is technically
                    # MiB; consistent with `mibToGib` formatter on the FE.
                    "vram_total_mb": int(parts[2]),
                    "driver_version": parts[3],
                    "cuda_version": cuda_version,
                }
            )
        except ValueError:
            logger.warning("nvidia-smi /info row skipped: %r", line)
            continue
    return out


# Matches "CUDA Version : 12.4" (R550 and earlier) AND "CUDA UMD version : 13.3"
# (R610+). Case-insensitive because R610 lower-cased "version", and the value
# group demands digits so the R610 line
#     CUDA version : Deprecated, see "CUDA UMD version" instead
# is skipped over rather than captured. That deprecated line is exactly why
# the panel showed "CUDA —" on the 610.57 host: the old pattern wanted
# "CUDA Version" with a capital V followed by a number, and R610 prints
# neither.
_CUDA_VERSION_RE = re.compile(
    r"CUDA(?:\s+UMD)?\s+version\s*[:=]\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def parse_cuda_version(stdout: str) -> str | None:
    """Extract the driver's CUDA version from ``nvidia-smi --version`` (or the
    plain ``nvidia-smi`` banner — same line, same regex).

    The format has shifted between driver releases: ``CUDA Version : 12.4``
    through R550, ``CUDA Version: 11.8`` (no space) on older builds, and
    from R610 ``CUDA UMD version : 13.3`` with the old key deprecated to a
    text sentence. Some virtualised drivers omit the line entirely. We
    match liberally and return ``None`` if nothing numeric is found —
    caller surfaces "not reported" rather than crashing.
    """
    m = _CUDA_VERSION_RE.search(stdout)
    return m.group(1) if m else None


def _run_subprocess(args: list[str]) -> str | None:
    """Run ``args`` and return stdout, or ``None`` if it failed.

    Synchronous: ``GET /api/system/info`` runs once per minute (with the
    60s cache) so the 3 shell-outs per cache-miss don't merit the
    complexity of asyncio.create_subprocess_exec. FastAPI runs sync
    handlers on a threadpool which is fine for this cadence.
    """
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as e:
        logger.info("%s not on PATH: %s", args[0], e)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out after %ss", args[0], _SUBPROCESS_TIMEOUT_S)
        return None
    except OSError as e:
        logger.warning("%s spawn failed: %s", args[0], e)
        return None
    if proc.returncode != 0:
        logger.info("%s exit %d: %s", args[0], proc.returncode, (proc.stderr or "").strip())
        return None
    return proc.stdout


def collect_gpus() -> list[dict[str, Any]]:
    """Run nvidia-smi twice (queries + version) and assemble the GPU list.

    Returns ``[]`` when ``nvidia-smi`` is missing or every call failed —
    the endpoint still returns 200 with ``gpus: []`` so the UI renders
    an empty state.
    """
    query_out = _run_subprocess(_NVIDIA_SMI_QUERY)
    if query_out is None:
        return []
    version_out = _run_subprocess(_NVIDIA_SMI_VERSION) or ""
    cuda_version = parse_cuda_version(version_out)
    if cuda_version is None:
        # Drivers that predate ``--version`` exit non-zero on it; the plain
        # banner has carried "CUDA Version: X.Y" for a decade, so read that
        # before giving up.
        cuda_version = parse_cuda_version(_run_subprocess(_NVIDIA_SMI_BANNER) or "")
    return parse_nvidia_smi_gpus(query_out, cuda_version)


# ---------------------------------------------------------------------------
# OS — /etc/os-release + uname
# ---------------------------------------------------------------------------


def _read_os_release() -> str:
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("/etc/os-release unreadable: %s", e)
        return ""


_OS_KV_RE = re.compile(r'^([A-Z_]+)=("(.*)"|(.*))$', re.MULTILINE)


def parse_os_release(text: str) -> dict[str, str]:
    """Parse ``/etc/os-release`` k=v lines into a dict (values dequoted).

    Standard freedesktop.org format: ``KEY=VALUE`` or ``KEY="VALUE"``. We
    don't try to handle shell-escapes inside the quotes — none of the
    distros we'll encounter use them in the keys we care about
    (``NAME``, ``VERSION``).
    """
    out: dict[str, str] = {}
    for m in _OS_KV_RE.finditer(text):
        key = m.group(1)
        # Group 3 is the unquoted contents when quoted; group 4 is the
        # bare value when not quoted.
        value = m.group(3) if m.group(3) is not None else m.group(4)
        out[key] = value or ""
    return out


def collect_os() -> dict[str, Any]:
    """Assemble the ``{name, version, kernel}`` OS slot.

    Falls back to ``platform.system()`` + ``"unknown"`` for distros
    without ``/etc/os-release`` (macOS, BSDs, stripped containers).
    Kernel is always available via ``platform.release()`` / ``uname -r``.
    """
    raw = parse_os_release(_read_os_release())
    name = raw.get("NAME") or platform.system() or "unknown"
    # Prefer the human-friendly ``VERSION`` ("22.04.4 LTS") over
    # ``VERSION_ID`` ("22.04") so operators see what they'd see in
    # ``lsb_release``. Fall back to ``VERSION_ID`` then "unknown".
    version = raw.get("VERSION") or raw.get("VERSION_ID") or "unknown"
    try:
        kernel = platform.release() or os.uname().release
    except (OSError, AttributeError):
        kernel = "unknown"
    return {"name": name, "version": version, "kernel": kernel}


# ---------------------------------------------------------------------------
# Docker — `docker info --format json`
# ---------------------------------------------------------------------------


_DOCKER_INFO = ["docker", "info", "--format", "{{json .}}"]


def parse_docker_info(stdout: str) -> dict[str, Any] | None:
    """Pick ``{version, runtime}`` out of ``docker info --format json``.

    ``ServerVersion`` is the Docker Engine version (matches ``docker
    --version`` minus the build hash). ``DefaultRuntime`` is ``"runc"`` on
    a stock host and ``"nvidia"`` on a GPU-enabled host — the field the
    operator actually cares about when interpreting GPU telemetry.

    Returns ``None`` on malformed JSON / missing fields rather than
    raising — the docker slot is optional from the API's point of view.
    """
    try:
        info = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(info, dict):
        return None
    version = info.get("ServerVersion")
    runtime = info.get("DefaultRuntime")
    if not version and not runtime:
        return None
    return {
        "version": version or "unknown",
        "runtime": runtime or "unknown",
    }


def collect_docker() -> dict[str, Any] | None:
    """Shell out to ``docker info`` and return parsed payload or ``None``.

    The API container typically does NOT have ``/var/run/docker.sock``
    mounted (we don't manage docker from inside the warden), so this
    returns ``None`` most of the time. The endpoint surfaces that as
    ``{"version": null, "runtime": null, "available": false}`` — the
    frontend renders an "unavailable" placeholder.
    """
    out = _run_subprocess(_DOCKER_INFO)
    if out is None:
        return None
    return parse_docker_info(out)


# ---------------------------------------------------------------------------
# Aggregator + cache
# ---------------------------------------------------------------------------


def collect_system_info() -> dict[str, Any]:
    """Build the full response payload for ``GET /api/system/info``.

    Each sub-collector handles its own failure modes; the aggregator never
    raises. Missing pieces are filled with stable placeholders so the
    frontend can render without conditionals on every field.
    """
    cpu = parse_cpuinfo(_read_proc_cpuinfo()) or {
        "model": "unknown",
        "physical_cores": 0,
        "threads": 0,
    }
    ram = parse_meminfo(_read_proc_meminfo()) or {"total_mb": 0}
    gpus = collect_gpus()
    os_payload = collect_os()
    docker = collect_docker()
    docker_payload = (
        {"version": docker["version"], "runtime": docker["runtime"], "available": True}
        if docker is not None
        else {"version": None, "runtime": None, "available": False}
    )

    return {
        "cpu": cpu,
        "ram": ram,
        "gpus": gpus,
        "os": os_payload,
        "docker": docker_payload,
    }


@dataclass
class SystemInfoCache:
    """In-process 60s cache for ``collect_system_info``.

    Lives on ``app.state.system_info_cache`` so tests can inject their own
    instance (e.g. with ``ttl=0`` to disable caching, or a frozen clock).
    Concurrency: the underlying probe is idempotent and cheap on a
    cache-hit, so we don't bother with a lock — the worst case is two
    parallel cache-misses both running collect_system_info and racing to
    write _cached. The result is identical either way.
    """

    ttl: float = CACHE_TTL_SECONDS
    clock: Any = field(default=time.monotonic)
    collector: Any = field(default=collect_system_info)
    _cached: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _cached_at: float = field(default=-1e9, init=False, repr=False)
    invocations: int = field(default=0, init=False)  # exposed for tests

    def get(self) -> dict[str, Any]:
        now = self.clock()
        if self._cached is not None and (now - self._cached_at) < self.ttl:
            return self._cached
        self.invocations += 1
        payload = self.collector()
        self._cached = payload
        self._cached_at = now
        return payload
