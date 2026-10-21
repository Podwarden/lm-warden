"""P6 (Metal design s7) — the sysctl inputs of the unified-memory capacity
formula: ``hw.memsize_usable`` and ``iogpu.wired_limit_mb``.

Same contract as ``app/system/apple/memory.py``: never raise; a field the
probe could not read is None, never 0 — except ``wired_limit_mb`` where 0
means "the OS default" and capacity.py relies on it."""

import platform
import sys

import pytest


def test_parse_sysctl_uint_values():
    from app.system.apple.capacity_sysctl import parse_sysctl_uint

    assert parse_sysctl_uint("16554000384\n") == 16554000384
    assert parse_sysctl_uint("0") == 0
    assert parse_sysctl_uint("  100  ") == 100


def test_parse_sysctl_uint_invalid():
    from app.system.apple.capacity_sysctl import parse_sysctl_uint

    assert parse_sysctl_uint("") is None
    assert parse_sysctl_uint("   ") is None
    assert parse_sysctl_uint("-1") is None
    assert parse_sysctl_uint("1.5") is None
    assert parse_sysctl_uint("abc") is None
    assert parse_sysctl_uint("١٢") is None
    assert parse_sysctl_uint("1\n2") is None


def test_parse_sysctl_uint_bound():
    from app.system.apple.capacity_sysctl import parse_sysctl_uint

    assert parse_sysctl_uint(str(2**63 - 1)) == 2**63 - 1
    assert parse_sysctl_uint(str(2**63)) is None


def test_read_capacity_sysctls_both_set():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    calls = []

    def run(argv):
        calls.append(list(argv))
        if argv == ["sysctl", "-n", "hw.memsize_usable"]:
            return "16554000384\n"
        if argv == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            return "4096\n"
        raise AssertionError(f"unexpected argv: {argv}")

    result = read_capacity_sysctls(run=run)
    assert result.memsize_usable == 16554000384
    assert result.wired_limit_mb == 4096
    assert calls == [
        ["sysctl", "-n", "hw.memsize_usable"],
        ["sysctl", "-n", "iogpu.wired_limit_mb"],
    ]


def test_read_capacity_sysctls_zero_semantics():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    def run(argv):
        if argv == ["sysctl", "-n", "hw.memsize_usable"]:
            return "0\n"
        if argv == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            return "0\n"
        raise AssertionError(f"unexpected argv: {argv}")

    result = read_capacity_sysctls(run=run)
    # A machine with zero usable memory is an unreadable probe; the wired
    # limit of 0 is a real value meaning "the OS default".
    assert result.memsize_usable is None
    assert result.wired_limit_mb == 0


def test_read_capacity_sysctls_runner_none():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    result = read_capacity_sysctls(run=lambda argv: None)
    assert result.memsize_usable is None
    assert result.wired_limit_mb is None


def test_read_capacity_sysctls_runner_raises():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    def boom(argv):
        raise OSError("sysctl not found")

    result = read_capacity_sysctls(run=boom)
    assert result.memsize_usable is None
    assert result.wired_limit_mb is None


def test_read_capacity_sysctls_runner_raises_non_oserror():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    def boom(argv):
        raise RuntimeError("timed out")

    result = read_capacity_sysctls(run=boom)
    assert result.memsize_usable is None
    assert result.wired_limit_mb is None


def test_read_capacity_sysctls_partial_failure_memsize_fails():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    def run(argv):
        if argv == ["sysctl", "-n", "hw.memsize_usable"]:
            raise OSError("unavailable")
        if argv == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            return "4096\n"
        raise AssertionError(f"unexpected argv: {argv}")

    result = read_capacity_sysctls(run=run)
    assert result.memsize_usable is None
    assert result.wired_limit_mb == 4096


def test_read_capacity_sysctls_partial_failure_wired_fails():
    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    def run(argv):
        if argv == ["sysctl", "-n", "hw.memsize_usable"]:
            return "16554000384\n"
        if argv == ["sysctl", "-n", "iogpu.wired_limit_mb"]:
            raise OSError("unavailable")
        raise AssertionError(f"unexpected argv: {argv}")

    result = read_capacity_sysctls(run=run)
    assert result.memsize_usable == 16554000384
    assert result.wired_limit_mb is None


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="needs a live Apple Silicon host",
)
@pytest.mark.macos
def test_darwin_live_capacity_sysctls():
    import subprocess

    from app.system.apple.capacity_sysctl import read_capacity_sysctls

    def run(argv):
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return proc.stdout if proc.returncode == 0 else None

    result = read_capacity_sysctls(run=run)
    assert isinstance(result.memsize_usable, int), "hw.memsize_usable must be readable"
    memsize = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10
        ).stdout
    )
    assert 0 < result.memsize_usable <= memsize
    assert isinstance(
        result.wired_limit_mb, int
    ), f"iogpu.wired_limit_mb must be readable (macOS {platform.mac_ver()[0]})"
    assert result.wired_limit_mb >= 0
