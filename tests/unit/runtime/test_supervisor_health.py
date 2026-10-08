import inspect
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.runtime.supervisor import DEFAULT_HEALTH_TIMEOUT_S, wait_for_health


@pytest.mark.asyncio
async def test_wait_for_health_returns_when_endpoint_200():
    responses = [Exception("conn refused"), Exception("conn refused"), 200]

    async def fake_get(url, timeout):  # noqa: ASYNC109
        v = responses.pop(0)
        if isinstance(v, Exception):
            raise v

        class R:
            status_code = v

        return R()

    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=fake_get)):
        ok = await wait_for_health(port=10001, timeout_s=5, interval_s=0.05)
    assert ok is True


@pytest.mark.asyncio
async def test_wait_for_health_times_out():
    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=Exception("never"))):
        ok = await wait_for_health(port=10001, timeout_s=0.2, interval_s=0.05)
    assert ok is False


def test_wait_for_health_default_timeout_matches_module_constant():
    """#99 — the default value of ``timeout_s`` must be sourced from the
    module-level ``DEFAULT_HEALTH_TIMEOUT_S`` constant, not a hard-coded
    literal at the function signature. The constant is the single source
    of truth so callers can reference it (e.g. settings layer) without
    parroting the magic number — and so tests can assert the contract
    without coupling to ``600.0``.
    """
    sig = inspect.signature(wait_for_health)
    assert sig.parameters["timeout_s"].default == DEFAULT_HEALTH_TIMEOUT_S


@pytest.mark.asyncio
async def test_wait_for_health_respects_injected_timeout_s():
    """#99 — passing a tiny timeout_s MUST cause the loop to give up
    quickly even though the module default is 600s. Regression guard
    against a future refactor that would hard-code 600.0 inside the
    loop and ignore the parameter.
    """
    call_count = {"n": 0}

    async def slow_fake(url, timeout):  # noqa: ASYNC109
        call_count["n"] += 1
        raise Exception("conn refused")

    start = time.monotonic()
    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=slow_fake)):
        ok = await wait_for_health(port=10001, timeout_s=0.1, interval_s=0.02)
    elapsed = time.monotonic() - start
    assert ok is False
    # Should have given up well before the module default would have us
    # waiting; a generous upper bound to avoid CI flakes is 1 s.
    assert elapsed < 1.0, f"expected fast failure with timeout_s=0.1, took {elapsed:.3f}s"
    assert call_count["n"] >= 1


@pytest.mark.asyncio
async def test_wait_for_health_without_alive_or_hard_timeout_is_unchanged():
    """#282 — neither new kwarg passed: behaviour must be byte-for-byte the
    old fixed-deadline wait, so every existing caller (and test) that never
    passes ``alive``/``hard_timeout_s`` keeps its original semantics."""
    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=Exception("never"))):
        ok = await wait_for_health(port=10001, timeout_s=0.1, interval_s=0.02)
    assert ok is False


@pytest.mark.asyncio
async def test_wait_for_health_keeps_polling_past_soft_deadline_while_alive():
    """#282 — a live engine that answers /health only after the soft
    ``timeout_s`` has elapsed must still succeed, as long as ``hard_timeout_s``
    has not elapsed and ``alive()`` keeps returning True. This is the actual
    incident: a slow-but-live NVFP4/Blackwell load must not be judged by the
    soft deadline alone."""
    calls = {"n": 0}

    async def fake_get(url, timeout):  # noqa: ASYNC109
        calls["n"] += 1

        class R:
            status_code = 200 if calls["n"] >= 4 else 599

        return R()

    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=fake_get)):
        ok = await wait_for_health(
            port=10001,
            timeout_s=0.05,
            interval_s=0.02,
            alive=lambda: True,
            hard_timeout_s=1.0,
        )
    assert ok is True
    assert calls["n"] >= 4


@pytest.mark.asyncio
async def test_wait_for_health_gives_up_at_hard_deadline_if_never_healthy():
    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=Exception("never"))):
        start = time.monotonic()
        ok = await wait_for_health(
            port=10001,
            timeout_s=0.05,
            interval_s=0.02,
            alive=lambda: True,
            hard_timeout_s=0.2,
        )
        elapsed = time.monotonic() - start
    assert ok is False
    assert elapsed >= 0.2
    assert elapsed < 1.0, f"expected to give up near the hard deadline, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_wait_for_health_fails_fast_when_engine_dies():
    """A dead process will never answer /health -- there is no reason to
    keep polling it out to the (let alone the extended hard) deadline."""
    calls = {"n": 0}

    async def fake_get(url, timeout):  # noqa: ASYNC109
        calls["n"] += 1
        raise Exception("conn refused")

    is_alive = {"v": True}

    def alive():
        # Dies after the first couple of polls.
        if calls["n"] >= 2:
            is_alive["v"] = False
        return is_alive["v"]

    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=fake_get)):
        start = time.monotonic()
        ok = await wait_for_health(
            port=10001,
            timeout_s=10.0,
            interval_s=0.02,
            alive=alive,
            hard_timeout_s=10.0,
        )
        elapsed = time.monotonic() - start
    assert ok is False
    # Must give up almost immediately once alive() flips False, nowhere near
    # either the soft or hard deadline.
    assert elapsed < 1.0, f"expected fast failure on process death, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_wait_for_health_fires_on_soft_timeout_once():
    fired = {"n": 0}

    async def on_soft_timeout():
        fired["n"] += 1

    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=Exception("never"))):
        ok = await wait_for_health(
            port=10001,
            timeout_s=0.05,
            interval_s=0.02,
            alive=lambda: True,
            hard_timeout_s=0.15,
            on_soft_timeout=on_soft_timeout,
        )
    assert ok is False
    assert fired["n"] == 1
