import asyncio
import logging
import os
import signal

import pytest


def test_parse_returns_pid(monkeypatch):
    from app.runtime.parent_watch import parse_parent_pid

    assert parse_parent_pid({"VW_PARENT_PID": "4242"}) == 4242


# ---------------------------------------------------------------- parse ----


@pytest.mark.parametrize(
    "env",
    [
        {},  # unset
        {"VW_PARENT_PID": ""},  # empty
    ],
)
def test_parse_unset_or_empty_is_none(env):
    from app.runtime.parent_watch import parse_parent_pid

    assert parse_parent_pid(env) is None


def test_parse_exact_bound_is_ok():
    from app.runtime.parent_watch import parse_parent_pid

    # 2 is the smallest accepted pid; 1 is just past the bound and is refused.
    assert parse_parent_pid({"VW_PARENT_PID": "2"}) == 2
    with pytest.raises(ValueError, match="VW_PARENT_PID"):
        parse_parent_pid({"VW_PARENT_PID": "1"})


@pytest.mark.parametrize(
    "raw",
    [
        "0",  # not a real pid
        "-5",  # sign
        "+5",  # sign
        " 5",  # leading whitespace
        "5 ",  # trailing whitespace
        "5\n",  # control character
        "abc",  # non-digits
        "0x10",  # hex is not plain decimal
    ],
)
def test_parse_refused_raises_naming_env(raw):
    from app.runtime.parent_watch import parse_parent_pid

    with pytest.raises(ValueError, match="VW_PARENT_PID"):
        parse_parent_pid({"VW_PARENT_PID": raw})


def test_parse_leading_zero_is_accepted():
    from app.runtime.parent_watch import parse_parent_pid

    assert parse_parent_pid({"VW_PARENT_PID": "042"}) == 42


# ---------------------------------------------------------------- alive ----


def _raise(exc):
    def fake_kill(pid, sig):
        raise exc

    return fake_kill


def test_alive_true_when_kill_ok_and_ppid_unchanged():
    from app.runtime.parent_watch import parent_alive

    def ok_kill(pid, sig):
        return None

    assert parent_alive(4242, start_ppid=1000, getppid=lambda: 1000, kill=ok_kill) is True


def test_alive_false_on_process_lookup_error():
    from app.runtime.parent_watch import parent_alive

    assert (
        parent_alive(4242, start_ppid=1000, getppid=lambda: 1000, kill=_raise(ProcessLookupError()))
        is False
    )


def test_alive_false_when_reparented():
    from app.runtime.parent_watch import parent_alive

    def ok_kill(pid, sig):
        return None

    # kill succeeds but the ppid changed: reparented, so dead.
    assert parent_alive(4242, start_ppid=1000, getppid=lambda: 1, kill=ok_kill) is False


def test_alive_true_on_permission_error():
    from app.runtime.parent_watch import parent_alive

    assert (
        parent_alive(4242, start_ppid=1000, getppid=lambda: 1000, kill=_raise(PermissionError()))
        is True
    )


def test_alive_true_on_other_oserror():
    from app.runtime.parent_watch import parent_alive

    assert (
        parent_alive(
            4242,
            start_ppid=1000,
            getppid=lambda: 1000,
            kill=_raise(OSError(22, "invalid argument")),
        )
        is True
    )


# ---------------------------------------------------------------- watch ----


def test_watch_rejects_nonpositive_interval():
    from app.runtime.parent_watch import watch_parent

    for bad in (0, -1):
        with pytest.raises(ValueError):
            asyncio.run(watch_parent(4242, interval=bad))


def test_watch_dead_before_first_sleep():
    from app.runtime.parent_watch import watch_parent

    sleeps = []

    async def never_sleep(s):
        sleeps.append(s)
        # Fail fast instead of parking: a watcher that sleeps before its first
        # check (or keeps looping after on_death) goes red here, not a hang.
        raise AssertionError("slept before acting on an already-dead parent")

    deaths = []

    def on_death():
        deaths.append(1)

    # Parent already gone: fires on the very first check, before any sleep.
    asyncio.run(
        watch_parent(
            4242,
            getppid=lambda: 1,
            kill=_raise(ProcessLookupError()),
            sleep=never_sleep,
            on_death=on_death,
        )
    )
    assert deaths == [1]
    assert sleeps == []


@pytest.mark.asyncio
async def test_watch_fires_once_after_two_sleeps(caplog):
    from app.runtime.parent_watch import watch_parent

    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) > 2:
            raise AssertionError("watcher kept polling after on_death")
        await asyncio.sleep(0)  # yield, but no real waiting

    check = {"n": 0}

    def fake_kill(pid, sig):
        check["n"] += 1
        if check["n"] < 3:
            return None  # first two checks: alive
        raise ProcessLookupError()  # third check: dead

    deaths = []

    def on_death():
        deaths.append(1)

    caplog.set_level(logging.DEBUG, logger="app.runtime.parent_watch")
    await watch_parent(
        4242,
        interval=2.0,
        getppid=lambda: 1000,
        kill=fake_kill,
        sleep=fake_sleep,
        on_death=on_death,
    )

    assert deaths == [1], "on_death must fire exactly once"
    assert sleeps == [2.0, 2.0], "exactly two sleeps of the interval before death"
    warning_recs = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "app.runtime.parent_watch"
    ]
    assert len(warning_recs) == 1, "exactly one WARNING"
    assert "4242" in warning_recs[0].getMessage(), "the WARNING must name the pid"


def test_watch_detects_reparent_against_the_ppid_recorded_at_start():
    from app.runtime.parent_watch import watch_parent

    # start_ppid is read ONCE: the launcher (1000) is our parent at start and
    # for the first check; after one sleep we have been reparented (to 1) while
    # kill(pid, 0) still succeeds (pid reuse). A watcher that re-read getppid()
    # as its baseline on every pass would never notice.
    ppids = iter([1000, 1000])

    def getppid():
        return next(ppids, 1)

    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) > 3:
            raise AssertionError("watcher never noticed the reparent")

    deaths = []
    asyncio.run(
        watch_parent(
            4242,
            interval=2.0,
            getppid=getppid,
            kill=lambda p, s: None,
            sleep=fake_sleep,
            on_death=lambda: deaths.append(1),
        )
    )
    assert deaths == [1]
    assert sleeps == [2.0]


def test_watch_default_on_death_sends_sigterm(monkeypatch):
    from app.runtime.parent_watch import watch_parent

    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))

    async def no_sleep(s):
        raise AssertionError("watcher kept polling after on_death")

    asyncio.run(
        watch_parent(4242, getppid=lambda: 1, kill=_raise(ProcessLookupError()), sleep=no_sleep)
    )
    assert sent == [(os.getpid(), signal.SIGTERM)]


def test_watch_cancel_propagates():
    from app.runtime.parent_watch import watch_parent

    sleeps = []

    async def blocking_sleep(s):
        sleeps.append(s)
        await asyncio.Event().wait()  # parks until the task is cancelled

    deaths = []

    def on_death():
        deaths.append(1)

    async def drive():
        task = asyncio.create_task(
            watch_parent(
                4242,
                interval=2.0,
                getppid=lambda: 1000,
                kill=lambda p, s: None,
                sleep=blocking_sleep,
                on_death=on_death,
            )
        )
        # let the task run until it parks in blocking_sleep
        for _ in range(100):
            await asyncio.sleep(0)
            if sleeps:
                break
        assert sleeps == [2.0], "entered the first sleep before being cancelled"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert deaths == [], "cancellation must not trigger on_death"
