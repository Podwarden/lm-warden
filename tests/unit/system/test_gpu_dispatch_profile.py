"""p6f: gpu.query_gpus / query_gpu_snapshot dispatch to gpu_apple on the apple
platform profile.

The dispatch keys on ``app.platform_profile.PROFILE`` -- NEVER ``sys.platform``
(Metal design s8 / OVERSEER planner iter 22): resolving by the running OS would
make the whole unit suite see a different backend on a developer's Mac and in
CI. On ``apple`` the live probe runs ``ioreg -r -c AGXAccelerator -d 2`` (async
subprocess, timeout, never raises) and returns gpu_apple's parse; on ``linux``
the nvidia-smi path is byte-identical to before.

All names introduced by this task are imported INSIDE the test functions: a
module-level import of a not-yet-existing name would ImportError the whole file
on the base tree, which runs nothing and is red for the wrong reason.
"""

import pytest

# One AGXAccelerator object, then its dictionary (same shape as
# test_gpu_apple._entry). util/used come from PerformanceStatistics.
_IOREG_ONE = (
    "+-o AGXAcceleratorG13X  <class AGXAccelerator, id 0x100000000, retain 3>\n"
    "| {\n"
    '|   "model" = "Apple M1"\n'
    '|   "gpu-core-count" = 8\n'
    '|   "IOSourceVersion" = "341.5.2"\n'
    '|   "PerformanceStatistics" = {"Device Utilization %"=37,'
    '"In use system memory"=1395589120}\n'
    "| }\n"
)
# 1395589120 // 2**20 = 1330 (floor).
_BUDGET = 11264
# 16 GiB in bytes -> 16384 MiB.
_MEMSIZE_BYTES = "17179869184"


def _fake_ioreg_factory():
    """A _run_ioreg stand-in that records every call and answers the ioreg
    AGX command with one object and the sysctl hw.memsize command with 16 GiB."""
    calls = []

    async def fake(argv):
        calls.append(tuple(argv))
        if argv == ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "2"]:
            return _IOREG_ONE
        if argv == ["sysctl", "-n", "hw.memsize"]:
            return _MEMSIZE_BYTES
        return ""

    return fake, calls


@pytest.fixture(autouse=True)
def _no_live_metal(monkeypatch):
    """Pin the device-ceiling probe to "no Metal" so this file reads the same
    on a Mac as on Linux; tests that want a ceiling set one explicitly."""
    from app.system.apple import metal

    monkeypatch.setattr(metal, "device_ceiling", lambda: None)


def _set_profile(monkeypatch, profile: str) -> None:
    import app.platform_profile as pp

    monkeypatch.setattr(pp, "PROFILE", profile)


# ---------------------------------------------------------------------------
# (a) apple + ioreg ok
# ---------------------------------------------------------------------------


async def test_apple_query_gpus_runs_ioreg_and_parses(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "apple")
    fake, calls = _fake_ioreg_factory()
    monkeypatch.setattr(gpu, "_run_ioreg", fake)

    out = await gpu.query_gpus()
    # the ioreg AGX command ran, and so did the budget probe (sysctl
    # hw.memsize) feeding the default budget provider.
    assert ("ioreg", "-r", "-c", "AGXAccelerator", "-d", "2") in calls
    assert ("sysctl", "-n", "hw.memsize") in calls
    assert len(out) == 1
    g = out[0]
    assert g.index == 0
    assert g.name == "Apple M1"
    assert g.memory_total_mib == 16384
    assert g.memory_used_mib == 1330
    assert g.utilization_pct == 37


async def test_apple_query_gpus_injected_budget_provider(monkeypatch):
    import app.system.gpu as gpu

    # The budget_mib comes from an injectable provider (gpu._apple_budget_mib);
    # overriding it pins that the dispatch threads the budget through to the
    # parse (gpu_apple's memory_total_mib) rather than hard-coding a number.
    _set_profile(monkeypatch, "apple")
    fake, _calls = _fake_ioreg_factory()

    async def fake_budget():
        return _BUDGET

    monkeypatch.setattr(gpu, "_run_ioreg", fake)
    monkeypatch.setattr(gpu, "_apple_budget_mib", fake_budget)

    out = await gpu.query_gpus()
    # GpuInfo has no memory_free_mib; the budget shows up as the total.
    assert out[0].memory_total_mib == _BUDGET


async def test_apple_query_gpu_snapshot_runs_ioreg_and_parses(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "apple")
    fake, _calls = _fake_ioreg_factory()
    monkeypatch.setattr(gpu, "_run_ioreg", fake)

    snap = await gpu.query_gpu_snapshot()
    assert len(snap.gpus) == 1
    g = snap.gpus[0]
    assert g.name == "Apple M1"
    assert g.memory_total_mib == 16384
    assert g.memory_used_mib == 1330
    assert g.utilization_pct == 37
    # (d) no per-process GPU attribution on Apple (s8): apps is always [].
    assert snap.apps == []
    assert snap.probe_error is None
    assert snap.topology is None


# ---------------------------------------------------------------------------
# (b) apple + ioreg missing -> [] and snapshot probe_error set
# ---------------------------------------------------------------------------


async def test_apple_ioreg_missing_query_gpus_returns_empty(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "apple")

    async def fake(argv):
        return None

    monkeypatch.setattr(gpu, "_run_ioreg", fake)

    assert await gpu.query_gpus() == []


async def test_apple_ioreg_missing_snapshot_sets_probe_error(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "apple")

    async def fake(argv):
        return None

    monkeypatch.setattr(gpu, "_run_ioreg", fake)

    snap = await gpu.query_gpu_snapshot()
    assert snap.gpus == []
    assert snap.apps == []
    assert snap.probe_error == "ioreg AGXAccelerator unavailable"


async def test_apple_ioreg_garbage_parses_empty(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "apple")

    async def fake(argv):
        if argv == ["sysctl", "-n", "hw.memsize"]:
            return _MEMSIZE_BYTES
        return "no AGXAccelerator object here, just words and { braces }"

    monkeypatch.setattr(gpu, "_run_ioreg", fake)

    # A successful ioreg that contains no AGX object degrades to the "no
    # Apple GPU found" parse result (gpu_apple returns []), not a probe_error:
    # the probe ran, it just saw nothing.
    assert await gpu.query_gpus() == []
    snap = await gpu.query_gpu_snapshot()
    assert snap.gpus == []
    assert snap.probe_error is None


# ---------------------------------------------------------------------------
# (c) linux path unchanged: nvidia-smi still called, ioreg never
# ---------------------------------------------------------------------------


async def test_linux_query_gpus_calls_nvidia_smi_not_ioreg(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "linux")
    nvidia_calls = []
    ioreg_calls = []

    async def fake_nvidia(args):
        nvidia_calls.append(tuple(args))
        return "0, NVIDIA A100, 8192, 100, 50, 70.0\n"

    async def fake_ioreg(argv):
        ioreg_calls.append(tuple(argv))
        return _IOREG_ONE

    monkeypatch.setattr(gpu, "_run_nvidia_smi", fake_nvidia)
    monkeypatch.setattr(gpu, "_run_ioreg", fake_ioreg)

    out = await gpu.query_gpus()
    assert nvidia_calls == [tuple(gpu.NVIDIA_SMI_CMD)]
    assert ioreg_calls == []
    assert len(out) == 1
    assert out[0].name == "NVIDIA A100"


async def test_linux_query_gpu_snapshot_calls_nvidia_smi_not_ioreg(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "linux")
    nvidia_calls = []
    ioreg_calls = []

    async def fake_nvidia(args):
        nvidia_calls.append(tuple(args))
        if args is gpu.NVIDIA_SMI_LIVE_CMD:
            # 8-col live row (index, uuid, name, mem.total, mem.used, mem.free,
            # util, compute_cap) -- the oldest accepted shape.
            return "0, gpu-uuid-0, NVIDIA A100, 8192, 100, 8000, 50, 8.0\n"
        if args is gpu.NVIDIA_SMI_APPS_CMD:
            return ""
        return ""

    async def fake_ioreg(argv):
        ioreg_calls.append(tuple(argv))
        return _IOREG_ONE

    monkeypatch.setattr(gpu, "_run_nvidia_smi", fake_nvidia)
    monkeypatch.setattr(gpu, "_run_ioreg", fake_ioreg)

    snap = await gpu.query_gpu_snapshot()
    # the LIVE cmd ran (plus the apps cmd in the gather); ioreg never did
    assert tuple(gpu.NVIDIA_SMI_LIVE_CMD) in nvidia_calls
    assert tuple(gpu.NVIDIA_SMI_APPS_CMD) in nvidia_calls
    assert ioreg_calls == []
    assert snap.probe_error is None
    assert len(snap.gpus) == 1
    assert snap.gpus[0].name == "NVIDIA A100"


async def test_apple_never_calls_nvidia_smi(monkeypatch):
    import app.system.gpu as gpu

    _set_profile(monkeypatch, "apple")
    nvidia_calls = []
    ioreg_calls = []

    async def fake_nvidia(args):
        nvidia_calls.append(tuple(args))
        return "0, NVIDIA A100, 8192, 100, 50, 70.0\n"

    async def fake_ioreg(argv):
        ioreg_calls.append(tuple(argv))
        if argv == ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "2"]:
            return _IOREG_ONE
        if argv == ["sysctl", "-n", "hw.memsize"]:
            return _MEMSIZE_BYTES
        return ""

    monkeypatch.setattr(gpu, "_run_nvidia_smi", fake_nvidia)
    monkeypatch.setattr(gpu, "_run_ioreg", fake_ioreg)

    await gpu.query_gpus()
    await gpu.query_gpu_snapshot()
    # apple dispatch must never touch the nvidia-smi path
    assert nvidia_calls == []
    # the AGX probe ran once per call (query_gpus + snapshot); the budget
    # probe is also ioreg-plumbed, so it is excluded here by exact match.
    agx = [c for c in ioreg_calls if c == ("ioreg", "-r", "-c", "AGXAccelerator", "-d", "2")]
    assert len(agx) == 2


# ---------------------------------------------------------------------------
# the injectable budget provider: default derives from hw.memsize (bytes)
# ---------------------------------------------------------------------------


async def test_apple_budget_mib_from_memsize_bytes(monkeypatch):
    import app.system.gpu as gpu

    # default provider: reads hw.memsize (bytes) via the ioreg-plumbed runner,
    # 16 GiB -> 16384 MiB. With no device ceiling (no Metal) and no capacity
    # sysctls the budget is the pre-P6 hw.memsize stand-in, unchanged, and it
    # is NOT floored by the 15% OS reserve (that is admission's business).
    async def fake(argv):
        assert argv[:2] == ["sysctl", "-n"]
        return _MEMSIZE_BYTES if argv == ["sysctl", "-n", "hw.memsize"] else None

    monkeypatch.setattr(gpu, "_run_ioreg", fake)
    assert await gpu._apple_budget_mib() == 16384


def _sysctls(**values):
    async def fake(argv):
        return values.get(argv[-1])

    return fake


async def test_apple_budget_mib_is_the_device_ceiling(monkeypatch):
    import app.system.gpu as gpu
    from app.system.apple import metal

    monkeypatch.setattr(metal, "device_ceiling", lambda: 12713115648)
    monkeypatch.setattr(gpu, "_run_ioreg", _sysctls(**{"hw.memsize": _MEMSIZE_BYTES}))
    assert await gpu._apple_budget_mib() == 12713115648 // 2**20


async def test_apple_capacity_reads_every_probe(monkeypatch):
    import app.system.gpu as gpu
    from app.models.capacity import UnifiedCapacity
    from app.system.apple import metal

    monkeypatch.setattr(metal, "device_ceiling", lambda: 12713115648)
    monkeypatch.setattr(
        gpu,
        "_run_ioreg",
        _sysctls(
            **{
                "hw.memsize": _MEMSIZE_BYTES,
                "hw.memsize_usable": "16554000384\n",
                "iogpu.wired_limit_mb": "0\n",
            }
        ),
    )
    assert await gpu._apple_capacity() == UnifiedCapacity(
        memsize=17179869184,
        memsize_usable=16554000384,
        wired_limit_mb=0,
        device_ceiling=12713115648,
    )


async def test_apple_budget_mib_wired_limit_lowers_it(monkeypatch):
    import app.system.gpu as gpu
    from app.system.apple import metal

    monkeypatch.setattr(metal, "device_ceiling", lambda: 12713115648)
    monkeypatch.setattr(
        gpu,
        "_run_ioreg",
        _sysctls(**{"hw.memsize": _MEMSIZE_BYTES, "iogpu.wired_limit_mb": "8192"}),
    )
    assert await gpu._apple_budget_mib() == 8192


async def test_apple_budget_mib_ceiling_but_no_memsize_fails_closed(monkeypatch):
    import app.system.gpu as gpu
    from app.system.apple import metal

    monkeypatch.setattr(metal, "device_ceiling", lambda: 12713115648)
    monkeypatch.setattr(gpu, "_run_ioreg", _sysctls())
    assert await gpu._apple_budget_mib() == 0


async def test_apple_budget_mib_raising_ceiling_probe_degrades(monkeypatch):
    import app.system.gpu as gpu
    from app.system.apple import metal

    def boom():
        raise RuntimeError("metal broke")

    monkeypatch.setattr(metal, "device_ceiling", boom)
    monkeypatch.setattr(gpu, "_run_ioreg", _sysctls(**{"hw.memsize": _MEMSIZE_BYTES}))
    assert await gpu._apple_budget_mib() == 16384


async def test_apple_device_ceiling_runs_off_the_event_loop(monkeypatch):
    """The pyobjc probe is synchronous (its first call imports Metal and
    creates the device): it must run in a worker thread, never on the loop."""
    import threading

    import app.system.gpu as gpu
    from app.system.apple import metal

    loop_thread = threading.get_ident()
    seen = []

    def probe():
        seen.append(threading.get_ident())
        return 12713115648

    monkeypatch.setattr(metal, "device_ceiling", probe)
    assert await gpu._apple_device_ceiling() == 12713115648
    assert seen and seen[0] != loop_thread


async def test_apple_budget_mib_degrades_when_memsize_unreadable(monkeypatch):
    import app.system.gpu as gpu

    async def fake(argv):
        return None

    monkeypatch.setattr(gpu, "_run_ioreg", fake)
    # memsize unreadable -> 0 budget (degrade, never raise); the GPU row still
    # appears with a zero total.
    assert await gpu._apple_budget_mib() == 0


async def test_apple_budget_mib_non_numeric_degrades(monkeypatch):
    import app.system.gpu as gpu

    async def fake(argv):
        return "not-a-number"

    monkeypatch.setattr(gpu, "_run_ioreg", fake)
    assert await gpu._apple_budget_mib() == 0


# ---------------------------------------------------------------------------
# the runner itself (review fix): real subprocesses, every failure -> None
# ---------------------------------------------------------------------------


async def test_run_ioreg_returns_stdout_on_success():
    import sys

    import app.system.gpu as gpu

    assert await gpu._run_ioreg([sys.executable, "-c", "print('agx')"]) == "agx\n"


async def test_run_ioreg_nonzero_exit_is_none():
    import sys

    import app.system.gpu as gpu

    argv = [sys.executable, "-c", "import sys; print('partial'); sys.exit(3)"]
    assert await gpu._run_ioreg(argv) is None


async def test_run_ioreg_missing_binary_is_none(tmp_path):
    import app.system.gpu as gpu

    assert await gpu._run_ioreg([str(tmp_path / "no-such-ioreg")]) is None


async def test_run_ioreg_non_executable_is_none(tmp_path):
    import app.system.gpu as gpu

    # PermissionError, not FileNotFoundError: the spawn failure still
    # degrades to None instead of raising into the stats loop.
    script = tmp_path / "ioreg"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    assert await gpu._run_ioreg([str(script)]) is None


async def test_run_ioreg_undecodable_output_does_not_raise():
    import sys

    import app.system.gpu as gpu

    argv = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff ok')"]
    out = await gpu._run_ioreg(argv)
    assert out is not None
    assert out.endswith(" ok")


async def test_run_ioreg_timeout_is_none_and_kills_the_child(monkeypatch):
    import asyncio
    import sys

    import app.system.gpu as gpu

    killed = []
    real_exec = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        proc = await real_exec(*args, **kwargs)
        real_kill = proc.kill

        def spy_kill():
            killed.append(proc.pid)
            real_kill()

        proc.kill = spy_kill
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    assert await gpu._run_ioreg(argv, timeout=0.3) is None
    # the wedged probe is killed on the timeout, so repeated probes cannot
    # accumulate children
    assert len(killed) == 1
