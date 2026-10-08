import os

import pytest

from app.runtime.engine import EngineSpec
from app.runtime.engine.local_subprocess import LocalSubprocessDriver
from app.runtime.engine.run_marker import is_run_sentinel

# 64 KiB cap, 16-byte lines, so the copy boundary always lands on a line
# edge and the "rotated file holds the most recent content" assertions are
# exact.
_CAP = 64 * 1024
_LINE = 16  # len(f"line-{i:010d}\n")


def _line(i: int) -> bytes:
    return f"line-{i:010d}\n".encode()


@pytest.mark.asyncio
async def test_spawn_runs_command_and_handle_waits(tmp_path):
    # Use /bin/sh as a stand-in "engine" that exits 0 immediately.
    spec = EngineSpec(
        model_id="m1", model_arg="x", argv=["/bin/sh", "-c", "exit 0"], env={}, port=8001
    )
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    assert handle.pid is not None
    rc = await handle.wait()
    assert rc == 0
    assert handle.returncode == 0


@pytest.mark.asyncio
async def test_terminate_kills_long_running(tmp_path):
    spec = EngineSpec(
        model_id="m2", model_arg="x", argv=["/bin/sh", "-c", "sleep 60"], env={}, port=8002
    )
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    await driver.terminate(handle, grace_s=0.5)
    assert handle.returncode is not None  # exited after term/kill


@pytest.mark.asyncio
async def test_spawn_execs_argv_zero_verbatim(tmp_path):
    """The driver no longer knows what program it is starting.

    Before sub-project B, spawn() prepended ['vllm', 'serve'] to spec.args --
    the single line that made a PLACEMENT concern (where does a process run)
    also decide WHICH program runs. argv[0] now comes from the backend's
    LaunchPlan, so the driver is backend-agnostic.
    """
    spec = EngineSpec(
        model_id="m3", model_arg="x", argv=["/bin/sh", "-c", "exit 7"], env={}, port=8003
    )
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    assert await handle.wait() == 7


@pytest.mark.asyncio
async def test_spawn_delimits_each_run_and_keeps_history(tmp_path):
    """#234: the log is append-only, so each spawn must stamp a run boundary
    that the diagnosis reader can start from. History stays -- the operator
    comparing attempt N-1 with attempt N is why we delimit instead of truncate.
    """
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    for i, marker in enumerate(("run-one-output", "run-two-output")):
        spec = EngineSpec(
            model_id="m4",
            model_arg="x",
            argv=["/bin/sh", "-c", f"echo {marker}"],
            env={},
            port=8004 + i,
        )
        await (await driver.spawn(spec)).wait()

    lines = (tmp_path / "m4.log").read_text().splitlines()
    sentinels = [i for i, ln in enumerate(lines) if is_run_sentinel(ln)]
    assert len(sentinels) == 2
    # Previous run preserved for the operator...
    assert "run-one-output" in lines[sentinels[0] : sentinels[1]]
    # ...but strictly before the boundary the reader starts from.
    assert "run-two-output" in lines[sentinels[1] :]
    assert "run-one-output" not in lines[sentinels[1] :]


@pytest.mark.asyncio
async def test_first_load_rotation_is_silent(tmp_path, caplog):
    """A model's FIRST load has no log file yet -- rotating must say nothing.

    ``_rotate`` stat()s the log before every spawn. On a first load that
    raised FileNotFoundError, which the blanket ``except OSError`` logged as
    ``could not rotate ...`` with a full traceback. Nothing was lost (spawn
    creates the file immediately afterwards and the engine log was captured
    correctly) but it is the very first thing a new operator sees in the logs
    and it reads like a fault.
    """
    caplog.set_level("WARNING")
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=1024)
    spec = EngineSpec(
        model_id="fresh", model_arg="x", argv=["/bin/sh", "-c", "exit 0"], env={}, port=8010
    )
    await (await driver.spawn(spec)).wait()

    assert (tmp_path / "fresh.log").exists()
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_rotation_still_happens_once_the_log_is_big(tmp_path):
    """The silence above must not have cost us the rotation itself.

    The log shares /data with the SQLite DB; filling that volume takes the
    database down with it, which is the whole reason rotation exists.
    """
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=64)
    log_path = tmp_path / "big.log"
    log_path.write_text("x" * 200)

    spec = EngineSpec(
        model_id="big", model_arg="x", argv=["/bin/sh", "-c", "exit 0"], env={}, port=8011
    )
    await (await driver.spawn(spec)).wait()

    assert (tmp_path / "big.log.1").read_text() == "x" * 200
    assert "x" * 200 not in log_path.read_text()


def test_driver_has_no_binary_kwarg():
    """The 'binary' escape hatch existed only so tests could inject /bin/sh
    past the hard-coded head. With argv[0] on the spec it is dead weight, and
    leaving it would be a second, competing way to decide argv[0]."""
    import inspect

    sig = inspect.signature(LocalSubprocessDriver.__init__)
    assert "binary" not in sig.parameters


async def test_in_run_sweep_copy_truncates_live_log(tmp_path):
    """#216 (slice b): a long-lived engine holds its O_APPEND log fd for its
    whole life, so the spawn-time rename reclaims nothing mid-run — the
    engine keeps appending to the file for weeks. The periodic sweep must
    copy-truncate instead: the live file is bounded by the cap, the rotated
    file keeps the newest content, and a write through the SAME fd afterwards
    lands at the fresh start of the live file (no reopen, no sparse hole).
    """
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=_CAP)
    log_path = tmp_path / "m1.log"
    # Hold the fd open the way spawn() does: O_WRONLY|O_CREAT|O_APPEND,
    # handed to the child. Here the test process plays the engine.
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        # 6000 lines x 16 bytes = 96 KiB, past the 64 KiB cap.
        for start in range(0, 6000, 500):
            os.write(fd, b"".join(_line(i) for i in range(start, start + 500)))
        assert os.path.getsize(log_path) == 6000 * _LINE

        assert driver.sweep_engine_logs({"m1"}) == 1

        # The live file is back at/under the cap ...
        assert os.path.getsize(log_path) <= _CAP
        # ... and the rotated file holds the most recent content: exactly the
        # last CAP bytes (4096 whole lines), the head loss marked the same
        # way the watchdog's crash-log tail marks it.
        rotated = (tmp_path / "m1.log.1").read_bytes()
        assert rotated.endswith(b"line-0000005999\n")
        assert b"line-0000000000\n" not in rotated
        assert rotated.splitlines()[0].startswith(b"[truncated:")

        # A write through the SAME fd lands at the start of the live file:
        # O_APPEND on a truncated file re-appends at the new end (offset 0),
        # so the size equals exactly what was written — a sparse hole would
        # make it far bigger.
        os.write(fd, b"post-rotation\n")
        assert os.path.getsize(log_path) == len(b"post-rotation\n")
        assert log_path.read_bytes() == b"post-rotation\n"
    finally:
        os.close(fd)


def test_in_run_sweep_leaves_small_logs_alone(tmp_path):
    """Control: a log under the cap is untouched — no rotation, no
    .log.1, no truncate."""
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=_CAP)
    log_path = tmp_path / "small.log"
    log_path.write_text("a few lines\n")

    assert driver.sweep_engine_logs({"small"}) == 0
    assert log_path.read_bytes() == b"a few lines\n"
    assert not (tmp_path / "small.log.1").exists()


def test_in_run_sweep_disabled_without_a_cap(tmp_path):
    """Control: log_max_bytes <= 0 means rotation is off, at spawn AND in-run."""
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=0)
    (tmp_path / "m.log").write_text("x" * 100)

    assert driver.sweep_engine_logs({"m"}) == 0
    assert (tmp_path / "m.log").read_text() == "x" * 100
    assert not (tmp_path / "m.log.1").exists()


def test_in_run_sweep_only_touches_live_models(tmp_path):
    """Control: only logs with a live writer are rotated. A stopped model's
    log is static, so it cannot grow — and rotating it would move the tail
    the Live Logs backlog reads out of the live file, which the next
    spawn-time rename would then clobber. Leave it alone."""
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=16)
    (tmp_path / "dead.log").write_text("x" * 100)

    assert driver.sweep_engine_logs(set()) == 0
    assert (tmp_path / "dead.log").read_text() == "x" * 100
    assert not (tmp_path / "dead.log.1").exists()
