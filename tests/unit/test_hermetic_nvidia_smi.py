"""The hermetic nvidia-smi autouse fixture (tests/conftest.py) makes the GPU
probe look ABSENT in every unit test: a host with /usr/bin/nvidia-smi but no
working driver (ollama.h: it exits 9, "Failed to initialize NVML") made the
probe report 'failing' and nulled vram/util/power in the stats tests.

These tests pin the seam without trusting the autouse fixture itself: a
PATH-installed nvidia-smi that fails must NOT change what a test observes --
the primary probe comes back None with the state 'absent' (the
FileNotFoundError record), never 'failing'.
"""

import asyncio
import os
import stat


def test_a_failing_nvidia_smi_on_path_does_not_reach_a_test(tmp_path, monkeypatch):
    # A fake nvidia-smi that answers like ollama.h's: NVML error on stdout,
    # exit 9. Whatever PATH is, the probe must not run it: the fixture's
    # _absent stand-in is what a test's probe hits.
    import app.system.gpu as gpu

    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "nvidia-smi"
    fake.write_text('#!/bin/sh\necho "Failed to initialize NVML: Unknown Error"\nexit 9\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    assert asyncio.run(gpu._run_nvidia_smi(gpu.NVIDIA_SMI_CMD)) is None
    h = gpu.gpu_probe_health()
    assert h.state == "absent"
    assert h.error == "nvidia-smi not found"


def test_the_probe_state_starts_unknown_in_every_test():
    # The fixture resets the tracker before and after each test, so no test
    # inherits another test's probe result.
    import app.system.gpu as gpu

    h = gpu.gpu_probe_health()
    assert h.state == "unknown"
    assert h.checked_at is None


def test_an_optional_probe_leaves_the_state_alone():
    # A non-primary probe's failure (or absence) only logs debug; the
    # tracker's state is for primary probes.
    import app.system.gpu as gpu

    assert asyncio.run(gpu._run_nvidia_smi(gpu.NVIDIA_SMI_NVLINK_CMD)) is None
    h = gpu.gpu_probe_health()
    assert h.state == "unknown"
    assert h.checked_at is None
