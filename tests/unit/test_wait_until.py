"""The shared bounded-polling helpers (tests/conftest.py, R20).

A test that observes background work must wait for the CONDITION with an
explicit budget, not sleep a fixed number of ticks: these pin the helper
contract -- the failure names what it waited for and how long, a passing
predicate returns its value early, an async predicate works in the async
twin, and a child that dies fails the wait at once with its return code.
"""

import asyncio
import subprocess
import sys
import time

import pytest

from tests.conftest import wait_until, wait_until_async


def test_sync_times_out_naming_what_and_the_budget():
    with pytest.raises(AssertionError) as ei:
        wait_until(lambda: False, timeout_s=0.3, interval=0.05, what="the never-true condition")
    msg = str(ei.value)
    assert "the never-true condition" in msg
    assert "0.3" in msg


def test_sync_returns_early_with_the_predicate_value():
    calls = []

    def pred():
        calls.append(1)
        return f"up after {len(calls)}" if len(calls) >= 3 else None

    start = time.monotonic()
    assert wait_until(pred, timeout_s=5.0, interval=0.01, what="three polls") == "up after 3"
    assert len(calls) == 3
    # Early exit: a helper that slept the full budget would take ~5 s.
    assert time.monotonic() - start < 1.0


def test_sync_dead_child_fails_at_once_with_its_return_code():
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    child.wait()
    start = time.monotonic()
    with pytest.raises(AssertionError) as ei:
        wait_until(lambda: False, timeout_s=5.0, what="the child to stay alive", child=child)
    assert "7" in str(ei.value)
    assert "the child to stay alive" in str(ei.value)
    # At once, not after the 5 s budget.
    assert time.monotonic() - start < 1.0


def test_sync_refuses_an_async_predicate():
    async def pred():
        return True

    with pytest.raises(TypeError, match="wait_until_async"):
        wait_until(pred, timeout_s=0.1, what="an async predicate")


async def test_async_times_out_naming_what_and_the_budget():
    with pytest.raises(AssertionError) as ei:
        await wait_until_async(lambda: False, timeout_s=0.3, interval=0.05, what="async never-true")
    msg = str(ei.value)
    assert "async never-true" in msg
    assert "0.3" in msg


async def test_async_returns_early_with_value_and_supports_async_predicates():
    calls = []

    async def pred():
        calls.append(1)
        if len(calls) < 2:
            await asyncio.sleep(0.01)
            return None
        return "second"

    start = time.monotonic()
    assert await wait_until_async(pred, timeout_s=5.0, interval=0.01, what="two polls") == "second"
    assert len(calls) == 2
    assert time.monotonic() - start < 1.0


async def test_async_dead_child_fails_at_once_with_its_return_code():
    child = await asyncio.create_subprocess_exec(sys.executable, "-c", "import sys; sys.exit(9)")
    await child.wait()
    start = time.monotonic()
    with pytest.raises(AssertionError) as ei:
        await wait_until_async(
            lambda: False, timeout_s=5.0, what="the child to stay alive", child=child
        )
    assert "9" in str(ei.value)
    assert time.monotonic() - start < 1.0
