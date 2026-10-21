"""Tests for the load-progress watcher loop (app/runtime/load_progress/watcher.py).

The watcher runs alongside the engine health wait: it polls the engine log
and vital signs, feeds a LoadTracker, evaluates a StallPolicy, and fires an
``on_give_up`` callback exactly once when the policy says to give up.  The
loop is total: any exception inside one tick is logged and the loop
continues — it must never fail a load by raising.

These tests use fake clocks, fake followers, and fake vitals samplers so the
loop is deterministic and fast.  The clock is a list of values popped per
``clock()`` call so the tests control the time the watcher sees.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.runtime.load_progress.follow import LogFollower
from app.runtime.load_progress.parse import Step
from app.runtime.load_progress.policy import StallPolicy, Verdict
from app.runtime.load_progress.registry import LoadProgressRegistry
from app.runtime.load_progress.tracker import LoadTracker, VitalsSample
from app.runtime.load_progress.watcher import watch_load

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(started_at: float = 0.0) -> LoadTracker:
    return LoadTracker(
        model_id="m",
        generation=1,
        backend="vllm",
        started_at=started_at,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _policy(**kw) -> StallPolicy:
    defaults = dict(warn_s=300.0, give_up_s=1200.0, max_s=7200.0)
    defaults.update(kw)
    return StallPolicy(**defaults)


class _FakeFollower:
    """A fake LogFollower that returns pre-programmed chunks, one per call."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self._chunks = list(chunks or [])
        self._idx = 0

    def read_new(self) -> str:
        if self._idx >= len(self._chunks):
            return ""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class _Clock:
    """A fake clock that returns ``now`` and can be advanced."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _flat_vitals(at: float) -> VitalsSample:
    """Vitals that never count as progress (all fields flat/None)."""
    return VitalsSample(at=at, cpu_seconds=0.0, gpu_mem_mib=0, read_bytes=0)


def _rising_gpu_vitals(at: float, mem: int) -> VitalsSample:
    """Vitals with GPU memory changing — counts as progress."""
    return VitalsSample(at=at, cpu_seconds=0.0, gpu_mem_mib=mem, read_bytes=0)


# ---------------------------------------------------------------------------
# Total loop: never raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_never_raises_on_sample_exception():
    """A ``sample()`` that raises must not kill the loop — the tick is
    logged and the loop continues."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    call_count = {"n": 0}

    async def sample():
        call_count["n"] += 1
        clock.advance(1.0)
        raise RuntimeError("vitals probe exploded")

    on_give_up = AsyncMock()

    async def watch():
        await watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=100.0, max_s=10000.0),
            still_current=lambda: True,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )

    # Run for a short time, then cancel — the loop must not have raised.
    task = asyncio.create_task(watch())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert call_count["n"] >= 2, "the loop should have survived the exception and continued"
    assert on_give_up.await_count == 0


@pytest.mark.asyncio
async def test_watcher_never_raises_on_follower_exception():
    """A ``follower.read_new()`` that raises must not kill the loop."""

    follower_calls = {"n": 0}

    class _BoomFollower:
        def read_new(self) -> str:
            follower_calls["n"] += 1
            raise OSError("log file vanished")

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    async def sample():
        clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_BoomFollower(),
            sample=sample,
            policy=_policy(give_up_s=100.0, max_s=10000.0),
            still_current=lambda: True,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # The follower raised on every tick but the loop kept going.
    assert follower_calls["n"] >= 2
    assert on_give_up.await_count == 0


# ---------------------------------------------------------------------------
# Returns when still_current() is false
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_returns_when_superseded():
    """When ``still_current()`` returns False the watcher returns without
    firing ``on_give_up`` — a superseded load must not write anything."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    current = {"v": True}

    async def sample():
        clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=100.0, max_s=10000.0),
            still_current=lambda: current["v"],
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    await asyncio.sleep(0.05)
    current["v"] = False
    await asyncio.wait_for(task, timeout=2.0)
    assert on_give_up.await_count == 0


# ---------------------------------------------------------------------------
# Returns when the tracker is done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_returns_when_tracker_done():
    """When the tracker is ``done`` (``mark_done`` was called) the watcher
    returns without firing ``on_give_up``."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    async def sample():
        clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=100.0, max_s=10000.0),
            still_current=lambda: True,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    await asyncio.sleep(0.03)
    tracker.mark_done(clock.now)
    await asyncio.wait_for(task, timeout=2.0)
    assert on_give_up.await_count == 0


# ---------------------------------------------------------------------------
# Give-up: fires on_give_up once and returns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_fires_on_give_up_once_and_returns():
    """A flat load (no log, no vitals progress, no CPU) hits the give_up
    threshold.  ``on_give_up`` fires exactly once and the loop returns."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    ticks = {"n": 0}

    async def sample():
        ticks["n"] += 1
        # Advance the clock past give_up_s on the second tick.
        if ticks["n"] >= 2:
            clock.advance(20.0)  # total > give_up_s (10.0)
        else:
            clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()
    given_verdicts: list[Verdict] = []

    async def _on_give_up(v: Verdict) -> None:
        given_verdicts.append(v)
        await on_give_up(v)

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=10.0, max_s=100.0, warn_s=5.0),
            still_current=lambda: True,
            on_give_up=_on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    await asyncio.wait_for(task, timeout=5.0)
    assert on_give_up.await_count == 1
    assert len(given_verdicts) == 1
    assert given_verdicts[0].state == "give_up"


@pytest.mark.asyncio
async def test_watcher_does_not_fire_on_give_up_when_superseded():
    """If ``still_current()`` is False on the give-up tick, ``on_give_up``
    is NOT called — the load was superseded and the verdict is stale."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    ticks = {"n": 0}
    current = {"v": True}

    async def sample():
        ticks["n"] += 1
        if ticks["n"] >= 2:
            clock.advance(100.0)  # past give_up_s
            current["v"] = False  # superseded on the same tick
        else:
            clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=10.0, max_s=200.0, warn_s=5.0),
            still_current=lambda: current["v"],
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    await asyncio.wait_for(task, timeout=5.0)
    assert on_give_up.await_count == 0


# ---------------------------------------------------------------------------
# Progressing: GPU memory rising → not killed before cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progressing_slow_load_is_not_killed_before_cap():
    """A load with GPU memory rising every tick is making progress.  The
    watcher must NOT fire ``on_give_up`` even after 3× the give_up_s
    threshold — progress resets the clock."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    ticks = {"n": 0}

    async def sample():
        ticks["n"] += 1
        # Each tick: advance the clock by give_up_s/2 (well under give_up_s)
        # and bump GPU memory by 128 MiB (above the 64 MiB threshold).
        clock.advance(5.0)
        return VitalsSample(
            at=clock.now,
            cpu_seconds=0.0,
            gpu_mem_mib=ticks["n"] * 128,  # rising 128 MiB each tick
            read_bytes=0,
        )

    on_give_up = AsyncMock()

    give_up_s = 10.0
    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=give_up_s, max_s=10000.0, warn_s=5.0),
            still_current=lambda: True,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    # Let it run for several ticks — well past 3× give_up_s of wall-clock,
    # but each tick has GPU-mem progress so the policy never gives up.
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # The clock advanced past 3 × give_up_s (30 s) but the load was making
    # progress every tick, so the watcher must not have given up.
    assert clock.now >= 3 * give_up_s, f"expected to run past 3×give_up_s, got {clock.now}"
    assert on_give_up.await_count == 0, "a progressing load must not be given up"


# ---------------------------------------------------------------------------
# Registry: verdict stored every tick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_stores_verdict_in_registry():
    """Every tick the latest verdict is stored in the registry so the UI
    can read live progress."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    async def sample():
        clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower([]),
            sample=sample,
            policy=_policy(give_up_s=100.0, max_s=10000.0, warn_s=5.0),
            still_current=lambda: True,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    v = registry.verdict("m")
    assert v is not None
    assert v.state in ("progressing", "stalled")


# ---------------------------------------------------------------------------
# Registry: finish only removes matching generation
# ---------------------------------------------------------------------------


def test_registry_finish_removes_only_matching_generation():
    """``finish`` must not evict a newer load's tracker."""
    reg = LoadProgressRegistry()
    t1 = _make_tracker()
    t1.generation = 1
    reg.start(t1)
    # A newer load (generation 2) replaces the tracker.
    t2 = _make_tracker()
    t2.generation = 2
    reg.start(t2)
    # The old generation finishing must NOT evict t2.
    reg.finish("m", generation=1)
    assert reg.get("m") is t2
    # The current generation finishing DOES evict.
    reg.finish("m", generation=2)
    assert reg.get("m") is None


def test_registry_start_replaces_prior_tracker():
    reg = LoadProgressRegistry()
    t1 = _make_tracker()
    reg.start(t1)
    assert reg.get("m") is t1
    t2 = _make_tracker()
    reg.start(t2)
    assert reg.get("m") is t2
    assert reg.active() == ["m"]


def test_registry_set_verdict_drops_after_finish():
    """A verdict arriving after ``finish`` is dropped — the load is over."""
    reg = LoadProgressRegistry()
    t = _make_tracker()
    reg.start(t)
    reg.finish("m", generation=1)
    v = Verdict(state="stalled", stalled_for_s=10.0, give_up_at_s=100.0, reason="x", hint=None)
    reg.set_verdict("m", v)
    assert reg.verdict("m") is None


# ---------------------------------------------------------------------------
# Real LogFollower: log lines feed the tracker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_feeds_log_lines_into_tracker(tmp_path: Path):
    """A real LogFollower reading from a file: vLLM progress lines update
    the tracker's WEIGHTS counter, which counts as progress."""

    clock = _Clock(start=0.0)
    tracker = _make_tracker(started_at=0.0)
    registry = LoadProgressRegistry()
    registry.start(tracker)

    log_file = tmp_path / "engine.log"
    log_file.write_text("")  # start empty

    follower = LogFollower(log_file)

    async def sample():
        clock.advance(1.0)
        return _flat_vitals(clock.now)

    on_give_up = AsyncMock()

    task = asyncio.create_task(
        watch_load(
            tracker=tracker,
            follower=follower,
            sample=sample,
            policy=_policy(give_up_s=100.0, max_s=10000.0, warn_s=5.0),
            still_current=lambda: True,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.01,
            clock=clock,
        )
    )
    # Append a vLLM weights-counter line — the tracker should see it.
    log_file.write_text(
        "(EngineCore_DP0 pid=123) Loading safetensors checkpoint shards:  "
        "45% Completed | 30/66 [00:01<00:02]\n"
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # The tracker saw the counter event.
    progress = tracker.counter_progress(Step.WEIGHTS)
    assert progress is not None
    assert progress == (30, 66)
    assert on_give_up.await_count == 0


# ---------------------------------------------------------------------------
# MLX fail-fast (BACKLOG-macos P5): the engine's own fatal-load line
# ---------------------------------------------------------------------------


def _tracker_for(backend: str, started_at: float = 0.0) -> LoadTracker:
    return LoadTracker(
        model_id="m",
        generation=1,
        backend=backend,
        started_at=started_at,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def _run_fatal_case(backend: str, chunks: list[str], *, ticks: int = 4, current=True):
    """Drive the watcher over ``chunks`` (one per tick) with flat vitals and a
    policy whose give-up is hours away; return (verdicts given up with,
    registry). Stops after ``ticks`` ticks if nothing gave up."""
    clock = _Clock(start=100.0)
    tracker = _tracker_for(backend, started_at=97.5)
    registry = LoadProgressRegistry()
    registry.start(tracker)
    given: list[Verdict] = []
    n = {"ticks": 0}

    async def sample():
        n["ticks"] += 1
        clock.advance(1.0)
        return _flat_vitals(clock.now)

    def still_current() -> bool:
        if n["ticks"] >= ticks:
            return False  # end the run: superseded
        return current

    async def on_give_up(v: Verdict) -> None:
        given.append(v)

    await asyncio.wait_for(
        watch_load(
            tracker=tracker,
            follower=_FakeFollower(chunks),
            sample=sample,
            policy=_policy(warn_s=3000.0, give_up_s=12000.0, max_s=72000.0),
            still_current=still_current,
            on_give_up=on_give_up,
            registry=registry,
            interval_s=0.0,
            clock=clock,
        ),
        timeout=5.0,
    )
    return given, registry


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["empty_dir", "corrupt_safetensors", "unsupported_arch"])
async def test_mlx_fatal_load_line_gives_up_on_that_tick(case):
    """A captured mlx_lm.server load failure gives the load up on the tick the
    fatal line arrives -- not after load_stall_timeout_s -- exactly once, with a
    ``fatal`` verdict whose reason quotes the engine's own line."""
    from tests.unit.runtime.backends.mlx_load_failure_logs import CAPTURED

    log, fatal = CAPTURED[case]
    engine_words = fatal[fatal.index("mlx_lm.server generation thread died:") :]
    given, registry = await _run_fatal_case("mlx", ["Fetching 9 files: 100%\n", log, log], ticks=10)
    assert len(given) == 1
    v = given[0]
    assert v.state == "give_up"
    assert v.fatal is True
    assert engine_words in (v.reason or "")
    # Seconds since load start at the tick it fired (clock 100, started 97.5):
    # the first tick read only the "Fetching" chunk, the second read the log.
    assert v.give_up_at_s == pytest.approx(3.5)
    assert registry.verdict("m") == v


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
async def test_fatal_line_is_ignored_for_other_backends(backend):
    """Scoped to MLX: the same text in a vLLM or llama.cpp row's log does not
    give the load up (their engines exit on a failed load; the stall policy and
    the alive check own them, unchanged)."""
    from tests.unit.runtime.backends.mlx_load_failure_logs import CORRUPT_SAFETENSORS_LOG

    given, _ = await _run_fatal_case(backend, [CORRUPT_SAFETENSORS_LOG], ticks=3)
    assert given == []


@pytest.mark.asyncio
async def test_mlx_without_a_fatal_line_is_left_to_the_policy():
    """An MLX log of a slow-but-healthy load (503 chatter, warnings, the bare
    traceback/exception lines without the server's fatal anchor) is NOT given
    up early."""
    chatter = (
        "UserWarning: mlx_lm.server is not recommended for production\n"
        "2026-10-21 01:18:41,122 - INFO - Starting httpd at 127.0.0.1 on port 18951...\n"
        '127.0.0.1 - - [21/Oct/2026 01:18:41] "GET /health HTTP/1.1" 503 -\n'
        "RuntimeError: generation thread died\n"
    )
    given, _ = await _run_fatal_case("mlx", [chatter, chatter], ticks=3)
    assert given == []


@pytest.mark.asyncio
async def test_mlx_fatal_line_does_not_fire_when_superseded():
    """The generation re-check runs before on_give_up for the fatal path too."""
    from tests.unit.runtime.backends.mlx_load_failure_logs import EMPTY_DIR_LOG

    given, _ = await _run_fatal_case("mlx", [EMPTY_DIR_LOG], ticks=10, current=False)
    assert given == []


def test_policy_evaluate_never_sets_fatal():
    from app.runtime.load_progress.policy import evaluate

    t = _tracker_for("mlx")
    v = evaluate(t, 99999.0, _policy(give_up_s=1.0, max_s=2.0, warn_s=0.5))
    assert v.state == "give_up"
    assert v.fatal is False
