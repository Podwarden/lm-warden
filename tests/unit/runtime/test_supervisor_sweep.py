"""#216 (slice b) wiring: the watchdog's periodic sweep reaches the driver
only through the supervisor, which decides which engines are live and which
drivers participate at all."""

import asyncio
import contextlib
import threading

import pytest

from app.runtime import watchdog
from app.runtime.engine.local_subprocess import LocalSubprocessDriver
from app.runtime.supervisor import Supervisor
from tests.fakes.fake_engine import FakeDriver, FakeHandle


class _Settings:
    def __init__(self, tmp_path):
        self.data_dir = str(tmp_path)


def test_sweep_rotates_only_logs_of_live_engines(tmp_path):
    """A model whose handle has exited is not live: its log must not be
    rotated, only the live engine's."""
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=16)
    (tmp_path / "live.log").write_text("x" * 100)
    (tmp_path / "dead.log").write_text("x" * 100)

    sup = Supervisor(_Settings(tmp_path), driver=driver)
    sup._handles["live"] = FakeHandle()  # returncode None -> running
    dead = FakeHandle()
    dead._finish(0)
    sup._handles["dead"] = dead

    assert sup.sweep_engine_logs(sup.live_engine_ids()) == 1
    assert (tmp_path / "live.log").read_bytes() == b""
    assert (tmp_path / "dead.log").read_text() == "x" * 100


def test_sweep_is_noop_for_drivers_without_the_capability(tmp_path):
    """The docker driver mirrors the log itself and is out of scope: a
    driver that does not implement the sweep must be left alone, not
    crashed on."""
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver())
    sup._handles["m"] = FakeHandle()

    assert sup.sweep_engine_logs(sup.live_engine_ids()) == 0


def test_sweep_is_noop_with_no_engines(tmp_path):
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=16)
    (tmp_path / "stale.log").write_text("x" * 100)

    sup = Supervisor(_Settings(tmp_path), driver=driver)
    assert sup.sweep_engine_logs(sup.live_engine_ids()) == 0
    assert (tmp_path / "stale.log").read_text() == "x" * 100


# --- the handle table is loop-owned (R9) ------------------------------------


class _ThreadRecordingHandles(dict):
    """Records which thread touches the handle table, so a sweep that reads
    it off the event loop cannot slip through unnoticed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reading_threads: list[threading.Thread] = []

    def items(self):
        self.reading_threads.append(threading.current_thread())
        return super().items()

    def __iter__(self):
        self.reading_threads.append(threading.current_thread())
        return super().__iter__()


class _RecordingSweepDriver:
    def __init__(self):
        self.swept: list[set[str]] = []

    def sweep_engine_logs(self, live_model_ids):
        self.swept.append(set(live_model_ids))
        return len(self.swept[-1])


@pytest.mark.asyncio
async def test_handles_table_is_read_on_the_event_loop_thread(tmp_path):
    """The watchdog sends the sweep to a worker thread, but the handle table
    belongs to the event loop: a load or unload racing a worker-thread read
    would raise 'dictionary changed size during iteration'. Only the file
    work may run off the loop."""
    driver = _RecordingSweepDriver()
    sup = Supervisor(_Settings(tmp_path), driver=driver)
    handles = _ThreadRecordingHandles({"m1": FakeHandle()})
    sup._handles = handles
    app_state = type("S", (), {"supervisor": sup})()
    loop_thread = threading.current_thread()

    await watchdog._sweep_engine_logs(app_state)

    assert handles.reading_threads, "the handle table was never read"
    assert all(
        t is loop_thread for t in handles.reading_threads
    ), f"the handle table was read off the event loop: {handles.reading_threads}"
    assert driver.swept == [{"m1"}], "the live set built on the loop must reach the driver's sweep"


@pytest.mark.asyncio
async def test_watchdog_tick_sweeps_even_when_probing_disabled(monkeypatch):
    """A disabled watchdog stops probing, not housekeeping: one tick with
    enabled=False must still run the log sweep, and nothing else."""
    calls = {"sweep": 0, "check": 0}
    swept = asyncio.Event()

    async def fake_live_config(settings):
        return {
            "enabled": False,
            "restore_on_boot": False,
            "interval_s": 3600.0,
            "failure_threshold": 3,
            "max_restarts": 3,
        }

    async def fake_sweep(app_state):
        calls["sweep"] += 1
        swept.set()

    async def fake_check(settings, app_state, state, budget, cfg):
        calls["check"] += 1

    monkeypatch.setattr(watchdog, "live_config", fake_live_config)
    monkeypatch.setattr(watchdog, "_sweep_engine_logs", fake_sweep)
    monkeypatch.setattr(watchdog, "check_once", fake_check)

    settings = type("S", (), {"watchdog_restart_window_s": 3600.0})()
    app_state = type("S", (), {"supervisor": object()})()

    task = asyncio.create_task(watchdog.run_watchdog_forever(settings, app_state))
    try:
        await asyncio.wait_for(swept.wait(), timeout=5)
    except TimeoutError:
        pass  # the assertions below then fail with the observed call counts
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert calls["sweep"] == 1, "a disabled watchdog must still sweep engine logs"
    assert calls["check"] == 0, "a disabled watchdog must not probe"
