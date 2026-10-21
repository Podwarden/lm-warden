"""Lifespan wires the VW_PARENT_PID belt (Metal design s2).

``app/main.py``'s lifespan must, BEFORE it starts any background task, call
``parse_parent_pid(os.environ)``; a ``ValueError`` propagates so boot fails
loud. When the var is set it starts ONE ``asyncio`` task running
``watch_parent(pid)`` alongside the other loops and cancels + gathers it in
the ``finally`` with them. When it is unset no such task is created (Linux
byte-identical). The watcher's default ``on_death`` (SIGTERM to self) is the
production path; these tests inject a patched ``on_death`` so they never
signal the test process.
"""

import asyncio
import os
import shutil
from unittest.mock import AsyncMock

import pytest


def _watcher_tasks() -> list[asyncio.Task]:
    """The tasks on the current loop whose coroutine is named ``watch_parent``
    (the watcher the lifespan starts), excluding the currently-running test
    task. Must be called from inside a running loop."""
    current = asyncio.current_task()
    return [
        t
        for t in asyncio.all_tasks()
        if t is not current and t.get_coro().__name__ == "watch_parent"
    ]


def _patch_watch_parent(
    monkeypatch, deaths, sleeps, calls=None, cancel_delay=0.0, **watch_overrides
):
    """Patch ``parent_watch.watch_parent`` with a wrapper that injects a fast
    clock, a liveness probe, and an ``on_death`` that records instead of
    signalling. Returns the (async) wrapper; the delegate to the real
    watcher is captured up-front so it never recurses."""
    from app.runtime import parent_watch

    async def fake_sleep(s):
        sleeps.append(s)
        await asyncio.sleep(0)  # a real await point so cancellation can land

    def fake_on_death():
        deaths.append(1)

    def fake_getppid():
        return os.getppid()

    def fake_kill(pid, sig):
        # default: the parent is already gone
        raise ProcessLookupError()

    overrides = {
        "interval": 0.001,
        "sleep": fake_sleep,
        "getppid": fake_getppid,
        "kill": fake_kill,
        "on_death": fake_on_death,
    }
    overrides.update(watch_overrides)
    real_watch_parent = parent_watch.watch_parent

    async def watch_parent(pid, **kwargs):
        # Named watch_parent so the task the lifespan spawns is identifiable.
        if calls is not None:
            calls.append({"pid": pid, "kwargs": dict(kwargs), "finished": False})
        merged = {**overrides, **kwargs}
        try:
            await real_watch_parent(pid, **merged)
        except asyncio.CancelledError:
            # A slow unwind: only an awaited (gathered) task is guaranteed to
            # have finished it before the lifespan returns.
            if cancel_delay:
                await asyncio.sleep(cancel_delay)
            if calls is not None:
                calls[-1]["finished"] = True
            raise

    monkeypatch.setattr(parent_watch, "watch_parent", watch_parent)
    return watch_parent


async def _drive_lifespan(
    tmp_data_dir, migrated_db_template, monkeypatch, cancel_delay=0.0, **watch_overrides
):
    """Drive the real lifespan on the test's own loop (like
    test_shutdown_background_tasks). Yields inside the lifespan so the
    just-created watcher task runs its first liveness check before the loop is
    snapshotted; without that it is only scheduled and the finally cancels it
    before it ever runs. Returns the watcher task seen mid-lifespan, plus the
    death/sleep call logs."""
    from app.main import build_app

    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    app = build_app()

    deaths: list[int] = []
    sleeps: list[float] = []
    calls: list[dict] = []
    _patch_watch_parent(
        monkeypatch, deaths, sleeps, calls=calls, cancel_delay=cancel_delay, **watch_overrides
    )
    # The lifespan's stats sampler would spawn real nvidia-smi / ioreg
    # probes; cancelled mid-spawn at shutdown, their subprocess transports
    # outlive the TestClient loop and print "Event loop is closed" at GC.
    monkeypatch.setattr("app.runtime.stats_sampler.query_gpus", AsyncMock(return_value=[]))

    seen_mid: list[asyncio.Task] = []
    async with app.router.lifespan_context(app):
        for _ in range(50):
            await asyncio.sleep(0)
        seen_mid.extend(_watcher_tasks())

    return {"mid": seen_mid, "deaths": deaths, "sleeps": sleeps, "calls": calls}


# ------------------------------------------------------------- unset arm ----


async def test_unset_creates_no_watcher(tmp_data_dir, migrated_db_template, monkeypatch):
    monkeypatch.delenv("VW_PARENT_PID", raising=False)
    result = await _drive_lifespan(tmp_data_dir, migrated_db_template, monkeypatch)
    assert result["mid"] == [], (
        "VW_PARENT_PID unset: no watch_parent task may exist during the "
        f"lifespan, found {result['mid']!r} (Linux must stay byte-identical)"
    )
    assert result["deaths"] == [], "no on_death may fire without a watcher"


# ------------------------------------------------------------- dead parent ---


async def test_set_dead_parent_triggers_on_death(tmp_data_dir, migrated_db_template, monkeypatch):
    """Red today: nothing reads VW_PARENT_PID, so the lifespan never starts a
    watcher and the (patched) on_death is never called."""
    monkeypatch.setenv("VW_PARENT_PID", "99999")
    result = await _drive_lifespan(tmp_data_dir, migrated_db_template, monkeypatch)
    assert result["deaths"] == [1], (
        "a set VW_PARENT_PID whose kill(pid,0) raises ProcessLookupError must "
        f"trigger the (patched) on_death exactly once; got {result['deaths']!r}"
    )


# ------------------------------------------------------------- alive parent ---


async def test_set_alive_parent_keeps_task_running_then_cancels(
    tmp_data_dir, migrated_db_template, monkeypatch
):
    def alive_kill(pid, sig):
        return None  # the parent is still here

    monkeypatch.setenv("VW_PARENT_PID", "4242")
    result = await _drive_lifespan(
        tmp_data_dir, migrated_db_template, monkeypatch, cancel_delay=0.3, kill=alive_kill
    )
    # Production calls watch_parent(pid) with the parsed pid and the module's
    # defaults (2 s poll, SIGTERM-to-self on_death): no overriding kwargs.
    assert [(c["pid"], c["kwargs"]) for c in result["calls"]] == [(4242, {})], result["calls"]
    # Gathered, not only cancelled: its (slow) unwind finished before the
    # lifespan returned.
    assert result["calls"][0]["finished"], "the watcher must be awaited in the finally"
    assert len(result["mid"]) == 1, (
        "a set, live VW_PARENT_PID must have exactly one watch_parent task "
        f"running during the lifespan, found {result['mid']!r}"
    )
    # the finally has run: the watcher was cancelled and gathered with the
    # other loops, so it must not outlive the lifespan.
    assert all(t.cancelled() for t in result["mid"]), (
        "the watcher must be cancelled (and gathered) by the lifespan's "
        f"finally, found {result['mid']!r}"
    )
    assert result["deaths"] == [], "an alive parent must not fire on_death"


# ------------------------------------------------------------- bad value ----


async def test_bad_value_fails_boot_naming_var(tmp_data_dir, migrated_db_template, monkeypatch):
    """1 is just past the >= 2 bound (2 is accepted): boot must fail loud."""
    from app.main import build_app

    shutil.copyfile(migrated_db_template, tmp_data_dir / "vllm-warden.db")
    monkeypatch.setenv("VW_PARENT_PID", "1")
    app = build_app()
    before = set(asyncio.all_tasks())
    with pytest.raises(ValueError, match="VW_PARENT_PID"):
        async with app.router.lifespan_context(app):
            pass
    # Parsed BEFORE any background task starts: a failed boot leaves none
    # running (they would outlive the app, since the finally never runs).
    leaked = set(asyncio.all_tasks()) - before
    for t in leaked:
        t.cancel()
    assert leaked == set(), f"a failed boot must start no background task, leaked {leaked!r}"
