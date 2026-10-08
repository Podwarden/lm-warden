"""Unit tests for the /api/system/info data pipeline (#148).

Two layers:
  1. Pure-parser tests for each source (cpuinfo, meminfo, nvidia-smi,
     os-release, docker info) — no subprocess, no filesystem.
  2. Route-level tests against the real FastAPI app with the collector
     shimmed via ``SystemInfoCache``'s injectable ``collector`` argument
     so we don't shell out from inside the test runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.system.system_info import (
    SystemInfoCache,
    collect_system_info,
    parse_cpuinfo,
    parse_cuda_version,
    parse_docker_info,
    parse_meminfo,
    parse_nvidia_smi_gpus,
    parse_os_release,
)
from tests.conftest import jwt_login, seed_admin_user

# ---------------------------------------------------------------------------
# Fixtures — representative source outputs
# ---------------------------------------------------------------------------

CPUINFO_HT_DUAL_SOCKET = """\
processor\t: 0
model name\t: Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz
physical id\t: 0
core id\t\t: 0

processor\t: 1
model name\t: Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz
physical id\t: 0
core id\t\t: 0

processor\t: 2
model name\t: Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz
physical id\t: 0
core id\t\t: 1

processor\t: 3
model name\t: Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz
physical id\t: 0
core id\t\t: 1

processor\t: 4
model name\t: Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz
physical id\t: 1
core id\t\t: 0

processor\t: 5
model name\t: Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz
physical id\t: 1
core id\t\t: 0
"""

CPUINFO_NO_PHYSICAL_ID = """\
processor\t: 0
model name\t: AMD EPYC 7402P 24-Core Processor

processor\t: 1
model name\t: AMD EPYC 7402P 24-Core Processor
"""

MEMINFO_64GIB = """\
MemTotal:       65820492 kB
MemFree:        62000000 kB
SwapTotal:      0 kB
"""

NVIDIA_SMI_GPU_CSV = (
    "0, NVIDIA RTX A4000, 16376, 550.54.15\n" "1, NVIDIA RTX A4000, 16376, 550.54.15\n"
)

NVIDIA_SMI_VERSION_OUT = """\
NVIDIA-SMI version  : 550.54.15
NVML version        : 12.550
DRIVER version      : 550.54.15
CUDA Version        : 12.4
"""

NVIDIA_SMI_VERSION_OUT_R610 = """\
NVIDIA-SMI version  : 610.57.04
NVML version        : 610.57
DRIVER version      : Deprecated, see "KMD version" instead
CUDA version        : Deprecated, see "CUDA UMD version" instead
KMD version         : 610.57.04
CUDA UMD version    : 13.3
"""

OS_RELEASE_UBUNTU = """\
PRETTY_NAME="Ubuntu 22.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.4 LTS (Jammy Jellyfish)"
ID=ubuntu
"""

DOCKER_INFO_JSON = '{"ServerVersion": "27.3.1", "DefaultRuntime": "nvidia",' ' "Containers": 12}'


# ---------------------------------------------------------------------------
# Pure parser tests
# ---------------------------------------------------------------------------


class TestParseCpuinfo:
    def test_dual_socket_hyperthreaded(self):
        """3 unique (phys, core) pairs across 6 threads → 3 physical, 6 threads."""
        result = parse_cpuinfo(CPUINFO_HT_DUAL_SOCKET)
        assert result == {
            "model": "Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz",
            "physical_cores": 3,
            "threads": 6,
        }

    def test_missing_physical_id_falls_back_to_threads(self):
        """Without physical_id / core_id (ARM SoCs, some VMs) physical_cores = threads."""
        result = parse_cpuinfo(CPUINFO_NO_PHYSICAL_ID)
        assert result == {
            "model": "AMD EPYC 7402P 24-Core Processor",
            "physical_cores": 2,
            "threads": 2,
        }

    def test_empty_returns_none(self):
        assert parse_cpuinfo("") is None
        assert parse_cpuinfo("   \n  ") is None

    def test_no_model_name_falls_back_to_unknown(self):
        result = parse_cpuinfo("processor\t: 0\n")
        assert result is not None
        assert result["model"] == "unknown"
        assert result["threads"] == 1


class TestParseMeminfo:
    def test_meminfo_kb_to_mb(self):
        """65820492 kB → 65820492 // 1024 = 64277 MB (integer truncation).

        We intentionally floor-divide rather than round; the slight
        under-report (a fraction of a MB) is invisible in the UI and
        matches what ``free -m`` does internally.
        """
        result = parse_meminfo(MEMINFO_64GIB)
        assert result == {"total_mb": 64277}

    def test_missing_memtotal_returns_none(self):
        assert parse_meminfo("MemFree: 1234 kB\n") is None

    def test_empty_returns_none(self):
        assert parse_meminfo("") is None


class TestParseNvidiaSmi:
    def test_two_gpus_with_cuda_version(self):
        gpus = parse_nvidia_smi_gpus(NVIDIA_SMI_GPU_CSV, cuda_version="12.4")
        assert gpus == [
            {
                "index": 0,
                "name": "NVIDIA RTX A4000",
                "vram_total_mb": 16376,
                "driver_version": "550.54.15",
                "cuda_version": "12.4",
            },
            {
                "index": 1,
                "name": "NVIDIA RTX A4000",
                "vram_total_mb": 16376,
                "driver_version": "550.54.15",
                "cuda_version": "12.4",
            },
        ]

    def test_no_cuda_version_propagates_null(self):
        gpus = parse_nvidia_smi_gpus(NVIDIA_SMI_GPU_CSV, cuda_version=None)
        assert all(g["cuda_version"] is None for g in gpus)

    def test_malformed_row_skipped(self):
        """Truncated row gets logged + skipped, valid rows preserved."""
        bad_then_good = "this is garbage\n0, RTX A4000, 16376, 550.54.15\n"
        gpus = parse_nvidia_smi_gpus(bad_then_good, cuda_version="12.4")
        assert len(gpus) == 1
        assert gpus[0]["index"] == 0

    def test_empty_input_returns_empty(self):
        assert parse_nvidia_smi_gpus("", cuda_version="12.4") == []


class TestParseCudaVersion:
    def test_standard_format(self):
        assert parse_cuda_version(NVIDIA_SMI_VERSION_OUT) == "12.4"

    def test_no_space_before_colon(self):
        assert parse_cuda_version("CUDA Version: 11.8") == "11.8"

    def test_missing_returns_none(self):
        assert parse_cuda_version("just some other output") is None

    def test_r610_deprecated_key_reads_cuda_umd_version(self):
        # Recorded from a 610.57.04 host: the old key is now a text sentence
        # and the number moved under "CUDA UMD version". This is what left
        # the panel showing "CUDA —".
        assert parse_cuda_version(NVIDIA_SMI_VERSION_OUT_R610) == "13.3"

    def test_deprecated_sentence_alone_is_none_not_garbage(self):
        out = 'CUDA version        : Deprecated, see "CUDA UMD version" instead\n'
        assert parse_cuda_version(out) is None

    def test_plain_banner_r610(self):
        banner = (
            "+-----------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 610.57.04   KMD Version: 610.57.04   CUDA UMD Version: 13.3 |\n"
        )
        assert parse_cuda_version(banner) == "13.3"

    def test_plain_banner_r550(self):
        banner = "| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4  |"
        assert parse_cuda_version(banner) == "12.4"


class TestCollectGpusCudaFallback:
    def test_falls_back_to_banner_when_version_flag_fails(self, monkeypatch: pytest.MonkeyPatch):
        """Drivers without ``--version`` exit non-zero; the banner still has it."""
        from app.system import system_info as mod

        def fake_run(args: list[str]) -> str | None:
            if "--query-gpu" in " ".join(args):
                return NVIDIA_SMI_GPU_CSV
            if "--version" in args:
                return None
            if args == ["nvidia-smi"]:
                return "| NVIDIA-SMI 470.57.02  Driver Version: 470.57.02  CUDA Version: 11.4 |"
            return None

        monkeypatch.setattr(mod, "_run_subprocess", fake_run)
        gpus = mod.collect_gpus()
        assert [g["cuda_version"] for g in gpus] == ["11.4", "11.4"]

    def test_r610_version_output_end_to_end(self, monkeypatch: pytest.MonkeyPatch):
        from app.system import system_info as mod

        def fake_run(args: list[str]) -> str | None:
            if "--query-gpu" in " ".join(args):
                return NVIDIA_SMI_GPU_CSV
            if "--version" in args:
                return NVIDIA_SMI_VERSION_OUT_R610
            raise AssertionError(f"banner fallback must not run: {args}")

        monkeypatch.setattr(mod, "_run_subprocess", fake_run)
        assert mod.collect_gpus()[0]["cuda_version"] == "13.3"


class TestParseOsRelease:
    def test_ubuntu(self):
        kv = parse_os_release(OS_RELEASE_UBUNTU)
        assert kv["NAME"] == "Ubuntu"
        assert kv["VERSION"] == "22.04.4 LTS (Jammy Jellyfish)"
        assert kv["VERSION_ID"] == "22.04"
        assert kv["ID"] == "ubuntu"

    def test_empty_returns_empty_dict(self):
        assert parse_os_release("") == {}


class TestParseDockerInfo:
    def test_full_payload(self):
        assert parse_docker_info(DOCKER_INFO_JSON) == {
            "version": "27.3.1",
            "runtime": "nvidia",
        }

    def test_malformed_json_returns_none(self):
        assert parse_docker_info("not json") is None

    def test_non_dict_top_level_returns_none(self):
        assert parse_docker_info("[]") is None

    def test_missing_both_fields_returns_none(self):
        assert parse_docker_info('{"Containers": 1}') is None

    def test_partial_fields_default_to_unknown(self):
        """Only ServerVersion present — runtime falls back to "unknown"."""
        assert parse_docker_info('{"ServerVersion": "27.3.1"}') == {
            "version": "27.3.1",
            "runtime": "unknown",
        }


# ---------------------------------------------------------------------------
# SystemInfoCache tests
# ---------------------------------------------------------------------------


class TestSystemInfoCache:
    def test_collapses_within_ttl(self):
        invocations = {"n": 0}

        def collector():
            invocations["n"] += 1
            return {"n": invocations["n"]}

        # Frozen clock — every call sees t=0, so cache should hold forever
        # within the 60s window.
        cache = SystemInfoCache(ttl=60.0, clock=lambda: 0.0, collector=collector)
        for _ in range(5):
            cache.get()
        assert invocations["n"] == 1
        assert cache.invocations == 1

    def test_re_runs_after_ttl(self):
        invocations = {"n": 0}

        def collector():
            invocations["n"] += 1
            return {"n": invocations["n"]}

        t = {"now": 0.0}
        cache = SystemInfoCache(ttl=60.0, clock=lambda: t["now"], collector=collector)
        cache.get()
        t["now"] = 61.0
        cache.get()
        assert invocations["n"] == 2

    def test_returns_payload_dict(self):
        cache = SystemInfoCache(ttl=60.0, clock=lambda: 0.0, collector=lambda: {"hello": "world"})
        assert cache.get() == {"hello": "world"}


# ---------------------------------------------------------------------------
# Aggregator integration — collect_system_info with each source mocked
# ---------------------------------------------------------------------------


class TestCollectSystemInfo:
    def test_with_gpus_and_docker(self, monkeypatch: pytest.MonkeyPatch):
        """Happy path: all sources return data, GPU present, docker available."""
        from app.system import system_info as mod

        monkeypatch.setattr(mod, "_read_proc_cpuinfo", lambda: CPUINFO_HT_DUAL_SOCKET)
        monkeypatch.setattr(mod, "_read_proc_meminfo", lambda: MEMINFO_64GIB)
        monkeypatch.setattr(mod, "_read_os_release", lambda: OS_RELEASE_UBUNTU)

        # _run_subprocess is called with three different argv lists —
        # dispatch on the first element + length so we can differentiate
        # the nvidia-smi query, the version probe, and docker info.
        def fake_run(args: list[str]) -> str | None:
            if args[0] == "nvidia-smi" and "--query-gpu" in " ".join(args):
                return NVIDIA_SMI_GPU_CSV
            if args[0] == "nvidia-smi" and "--version" in args:
                return NVIDIA_SMI_VERSION_OUT
            if args[0] == "docker" and "info" in args:
                return DOCKER_INFO_JSON
            return None

        monkeypatch.setattr(mod, "_run_subprocess", fake_run)
        result = collect_system_info()

        assert result["cpu"]["model"] == "Intel(R) Xeon(R) Gold 6234 CPU @ 3.30GHz"
        assert result["cpu"]["physical_cores"] == 3
        assert result["cpu"]["threads"] == 6
        assert result["ram"] == {"total_mb": 64277}
        assert len(result["gpus"]) == 2
        assert result["gpus"][0]["cuda_version"] == "12.4"
        assert result["os"]["name"] == "Ubuntu"
        assert result["os"]["version"] == "22.04.4 LTS (Jammy Jellyfish)"
        assert result["docker"] == {
            "version": "27.3.1",
            "runtime": "nvidia",
            "available": True,
        }

    def test_no_gpu_no_docker(self, monkeypatch: pytest.MonkeyPatch):
        """nvidia-smi + docker both missing: gpus=[] + docker.available=false.

        Endpoint must NOT 500 in this case — common on dev laptops and
        anywhere the API container isn't running with --gpus / the docker
        socket mounted.
        """
        from app.system import system_info as mod

        monkeypatch.setattr(mod, "_read_proc_cpuinfo", lambda: CPUINFO_NO_PHYSICAL_ID)
        monkeypatch.setattr(mod, "_read_proc_meminfo", lambda: MEMINFO_64GIB)
        monkeypatch.setattr(mod, "_read_os_release", lambda: OS_RELEASE_UBUNTU)
        monkeypatch.setattr(mod, "_run_subprocess", lambda args: None)

        result = collect_system_info()
        assert result["gpus"] == []
        assert result["docker"] == {
            "version": None,
            "runtime": None,
            "available": False,
        }
        # Other slots still populated.
        assert result["cpu"]["threads"] == 2
        assert result["ram"]["total_mb"] == 64277
        assert result["os"]["name"] == "Ubuntu"

    def test_all_sources_missing_returns_placeholders(self, monkeypatch: pytest.MonkeyPatch):
        """Every source fails (sandbox / non-Linux): endpoint still returns 200-shaped dict."""
        from app.system import system_info as mod

        monkeypatch.setattr(mod, "_read_proc_cpuinfo", lambda: "")
        monkeypatch.setattr(mod, "_read_proc_meminfo", lambda: "")
        monkeypatch.setattr(mod, "_read_os_release", lambda: "")
        monkeypatch.setattr(mod, "_run_subprocess", lambda args: None)

        result = collect_system_info()
        assert result["cpu"]["model"] == "unknown"
        assert result["ram"] == {"total_mb": 0}
        assert result["gpus"] == []
        assert result["docker"]["available"] is False
        # OS falls back to platform.system() — at least "name" is always populated.
        assert isinstance(result["os"]["name"], str) and result["os"]["name"]


# ---------------------------------------------------------------------------
# Route-level tests — JWT gate + cache wiring + JSON contract.
# ---------------------------------------------------------------------------


def _seed_done(db_path: Path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0])


def _jwt_auth(client) -> dict[str, str]:
    return jwt_login(client)


def test_system_info_requires_jwt(tmp_data_dir: Path, client) -> None:
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")

    r = client.get("/api/system/info")
    assert r.status_code == 401


def test_system_info_returns_full_payload(tmp_data_dir: Path, client) -> None:
    """Inject a stub collector via the cache → assert the full contract."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")

    stub_payload: dict[str, Any] = {
        "cpu": {
            "model": "AMD EPYC 7402P 24-Core Processor",
            "physical_cores": 24,
            "threads": 48,
        },
        "ram": {"total_mb": 257812},
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX A4000",
                "vram_total_mb": 16376,
                "driver_version": "550.54.15",
                "cuda_version": "12.4",
            },
        ],
        "os": {"name": "Ubuntu", "version": "22.04.4 LTS", "kernel": "6.8.0-49-generic"},
        "docker": {"version": "27.3.1", "runtime": "nvidia", "available": True},
    }
    client.app.state.system_info_cache = SystemInfoCache(
        ttl=60.0, clock=lambda: 0.0, collector=lambda: stub_payload
    )

    auth = _jwt_auth(client)
    r = client.get("/api/system/info", headers=auth)
    assert r.status_code == 200
    assert r.json() == stub_payload


def test_system_info_collapses_via_cache(tmp_data_dir: Path, client) -> None:
    """Two requests within the TTL hit the collector exactly once."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")

    invocations = {"n": 0}

    def collector():
        invocations["n"] += 1
        return {
            "cpu": {"model": "x", "physical_cores": 1, "threads": 1},
            "ram": {"total_mb": 0},
            "gpus": [],
            "os": {"name": "x", "version": "x", "kernel": "x"},
            "docker": {"version": None, "runtime": None, "available": False},
        }

    client.app.state.system_info_cache = SystemInfoCache(
        ttl=60.0, clock=lambda: 0.0, collector=collector
    )
    auth = _jwt_auth(client)
    for _ in range(3):
        r = client.get("/api/system/info", headers=auth)
        assert r.status_code == 200
    assert invocations["n"] == 1


def test_system_info_no_gpu_returns_empty_list(tmp_data_dir: Path, client) -> None:
    """Dev box without NVIDIA: gpus=[] and docker.available=false; HTTP 200."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")

    client.app.state.system_info_cache = SystemInfoCache(
        ttl=60.0,
        clock=lambda: 0.0,
        collector=lambda: {
            "cpu": {"model": "Apple M3", "physical_cores": 8, "threads": 8},
            "ram": {"total_mb": 24576},
            "gpus": [],
            "os": {"name": "Darwin", "version": "unknown", "kernel": "23.4.0"},
            "docker": {"version": None, "runtime": None, "available": False},
        },
    )
    auth = _jwt_auth(client)
    r = client.get("/api/system/info", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["gpus"] == []
    assert body["docker"]["available"] is False
