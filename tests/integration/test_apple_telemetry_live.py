"""Live Apple telemetry on a real Apple Silicon host (Metal design s8).

The unit tests feed the parsers fixture text; these run the *same* parsers
against real host output so the M1 actually gets proven, not just the format.
No model is loaded — no live-model lock is needed. Skips (not fails) off
darwin/arm64, where libproc and ``ioreg -c AGXAccelerator`` do not exist.

All imports sit inside the test functions: a module-level import of a name
that predates p6b-p6d would ImportError the whole file on such a base and
collect nothing, red for the wrong reason.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time

import pytest

pytestmark = [
    pytest.mark.macos,
    pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="live Apple telemetry needs a darwin/arm64 host (vm_stat page-size "
        "math, ioreg AGXAccelerator and libproc proc_pid_rusage)",
    ),
]

_BUDGET_MIB = 1024


def _run(argv: list[str]) -> str | None:
    """argv -> stdout, or None when the probe fails. Mirrors the ``run``
    contract ``read_memory_pressure`` takes: a live probe that cannot run
    degrades to None fields, never raises out of the test."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def test_read_memory_pressure_real_runner():
    from app.system.apple.memory import read_memory_pressure

    mp = read_memory_pressure(run=_run)

    # Page-size-derived fields: every key vm_stat reports on a live host must
    # be non-None bytes, and each must be a whole positive multiple of the
    # page size the header declared (16384 on Apple Silicon) — the exact
    # multiple pins the header-driven math, not an assumed 4096.
    header = _run(["vm_stat"])
    assert header is not None, "vm_stat produced no output on a darwin/arm64 host"
    assert "page size of" in header
    page_size = int(header.split("page size of", 1)[1].split(" bytes", 1)[0])
    assert page_size > 0
    for name in ("free", "active", "inactive", "wired", "compressed", "speculative"):
        value = getattr(mp, name)
        assert value is not None, f"vm_stat field {name!r} was not reported"
        assert isinstance(value, int)
        assert value % page_size == 0, f"{name}={value} is not a multiple of {page_size}"
    # A live kernel always has free, active, inactive and wired pages. The
    # compressor and the speculative pool can legitimately sit at 0 (a freshly
    # booted big-RAM host, a host under pressure), so those two are pinned as
    # reported, not as positive -- the test must not depend on the machine.
    for name in ("free", "active", "inactive", "wired"):
        assert getattr(mp, name) > 0, f"vm_stat field {name!r} read 0 on a live host"

    # Swap: all three fields must be readable on a live host. free is None
    # only in the odd case where the swap file does not exist; then total
    # and used are the honest pair, both 0.
    assert mp.swap_total_bytes is not None
    assert mp.swap_used_bytes is not None
    assert mp.swap_free_bytes is not None
    # sysctl prints each amount rounded to 0.01M and the parse truncates to a
    # whole byte, so free and total - used agree to within two roundings
    # (0.02 MiB), not to the byte.
    drift = abs(mp.swap_free_bytes - (mp.swap_total_bytes - mp.swap_used_bytes))
    assert drift <= 2**20 // 50, f"swap free/total/used disagree by {drift} bytes"
    if mp.swap_total_bytes == 0:
        assert mp.swap_used_bytes == 0


def test_parse_ioreg_agx_real_output():
    ioreg = shutil.which("ioreg")
    assert ioreg is not None, "ioreg not found on a darwin/arm64 host"
    out = subprocess.run(
        [ioreg, "-r", "-c", "AGXAccelerator", "-d", "2"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"ioreg exited {out.returncode}: {out.stderr}"

    from app.system.gpu_apple import parse_ioreg_agx

    gpus = parse_ioreg_agx(out.stdout, budget_mib=_BUDGET_MIB)
    assert len(gpus) >= 1, "an M1 reports no AGXAccelerator objects"
    # Independent read of the same stdout: the parser must carry the host's
    # own numbers through, not a floor/default (0 is the parser's "not
    # reported", so a range check alone could never go red).
    raw_util = [int(v) for v in re.findall(r'"Device Utilization %"=(\d+)', out.stdout)]
    raw_used = [int(v) for v in re.findall(r'"In use system memory"=(\d+)', out.stdout)]
    assert len(raw_util) == len(gpus), "every AGXAccelerator reports a utilization"
    assert len(raw_used) == len(gpus), "every AGXAccelerator reports in-use memory"
    for i, gpu in enumerate(gpus):
        assert gpu.index == i
        assert 0 <= gpu.utilization_pct <= 100
        assert gpu.utilization_pct == raw_util[i]
        assert gpu.memory_used_mib == raw_used[i] // 2**20
        assert gpu.compute_cap is None
        # The parsed budget is the input budget, verbatim — not hw.memsize.
        assert gpu.memory_total_mib == _BUDGET_MIB
        # free is clamped at 0: a busy GPU can hold more than the budget.
        assert gpu.memory_free_mib == max(0, _BUDGET_MIB - gpu.memory_used_mib)


def test_phys_footprint_self_and_dead_child():
    from app.system.apple import proc

    me = proc.phys_footprint(os.getpid())
    assert (
        isinstance(me, int) and me > 0
    ), f"own footprint should be a positive byte count on darwin, got {me!r}"

    # Cross-check against the OS's own `footprint` tool on an idle child (a
    # stable footprint, unlike this test process): the same counter, so the
    # two agree closely; a wrong struct field or offset would not.
    footprint_tool = shutil.which("footprint")
    assert footprint_tool is not None, "footprint not found on a darwin/arm64 host"
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        time.sleep(0.3)
        ours = proc.phys_footprint(sleeper.pid)
        report = subprocess.run(
            [footprint_tool, "-p", str(sleeper.pid)], capture_output=True, text=True
        ).stdout
    finally:
        sleeper.kill()
        sleeper.wait()
    m = re.search(r"Footprint: ([\d.]+) (B|KB|MB|GB)\b", report)
    assert m is not None, f"unparseable footprint output: {report[:200]!r}"
    theirs = float(m.group(1)) * {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[m.group(2)]
    assert isinstance(ours, int) and ours > 0
    assert 0.75 * theirs <= ours <= 1.25 * theirs, f"phys_footprint {ours} vs footprint {theirs}"

    # A process that exited: the pid is valid but proc_pid_rusage must fail,
    # so the contract's None (never a stale number, never a raise) is what a
    # just-exited child returns. (subprocess.Popen, not run(): we need the
    # child's pid, and CompletedProcess does not keep it.)
    child = subprocess.Popen(["true"], stdout=subprocess.DEVNULL)
    assert child.wait() == 0
    dead = proc.phys_footprint(child.pid)
    assert dead is None, f"exited pid {child.pid} should read None, got {dead!r}"

    # The boundary just below the dead-pid arm: a non-positive pid is refused
    # before any libproc call, and so is pid 0.
    assert proc.phys_footprint(0) is None
    assert proc.phys_footprint(-1) is None
