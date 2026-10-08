"""
Regression tests for supervisor crash detection via _watch_exit watcher task.
Covers: unexpected subprocess exit triggers on_exit callback, cleanup is correct,
unload() cancels the watcher without firing on_exit, and no_on_exit does not break.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.supervisor import Supervisor
from tests.conftest import wait_until_async


class _Settings:
    pass


class _M:
    id = "qwen"
    hf_repo = "Qwen/Qwen3.5-9B"
    hf_revision = "main"
    served_model_name = "qwen3.5-9b"
    gpu_indices = [1, 2]
    tensor_parallel_size = 2
    max_model_len = 8192
    dtype = "auto"
    gpu_memory_utilization = 0.90
    extra_env = {}


def _make_settings(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "hf-token")
    s.hf_cache_dir = str(tmp_path / "hf-cache")
    (tmp_path / "hf-token").write_text("hf_xxx")
    return s


def _make_proc(*, returncode_after_wait=None, sleep_s=None):
    """Return a MagicMock subprocess whose .wait() either resolves quickly or sleeps."""
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = None

    if sleep_s is not None:

        async def _hang():
            await asyncio.sleep(sleep_s)
            return 0

        proc.wait = AsyncMock(side_effect=_hang)
    else:
        rc = returncode_after_wait if returncode_after_wait is not None else 0

        async def _exit():
            proc.returncode = rc
            return rc

        proc.wait = AsyncMock(side_effect=_exit)

    return proc


@pytest.mark.asyncio
async def test_crash_calls_on_exit(tmp_path):
    """Subprocess exits rc=139 (segfault) → on_exit fired, state cleaned up."""
    settings = _make_settings(tmp_path)
    sup = Supervisor(settings)
    model = _M()
    rc_received = []

    async def spy(rc: int) -> None:
        rc_received.append(rc)

    proc = _make_proc(returncode_after_wait=139)

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        await sup.load(model, port=10000, on_exit=spy)

    # Wait for the watcher task to fire
    await wait_until_async(lambda: rc_received, what="the watcher to fire on_exit with the rc")

    assert rc_received == [139], f"Expected [139], got {rc_received}"
    assert model.id not in sup._handles
    assert model.id not in sup._ports
    assert model.id not in sup._watchers
    assert sup.gpus.owner_of(1) is None
    assert sup.gpus.owner_of(2) is None


@pytest.mark.asyncio
async def test_unload_cancels_watcher_no_on_exit(tmp_path):
    """unload() cancels the watcher task; on_exit spy must NOT be called."""
    settings = _make_settings(tmp_path)
    sup = Supervisor(settings)
    model = _M()
    spy_called = []

    async def spy(rc: int) -> None:
        spy_called.append(rc)

    proc = _make_proc(sleep_s=5.0)

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        await sup.load(model, port=10000, on_exit=spy)

    # Transition through state machine to READY so unload() doesn't refuse
    await sup.mark_warming(model.id)
    await sup.mark_ready(model.id)

    # Immediately unload — watcher should be cancelled
    # Patch os.killpg so we don't fail trying to signal a fake pid
    async def _immediate_wait():
        proc.returncode = -15
        return -15

    proc.wait = AsyncMock(side_effect=_immediate_wait)

    with patch("os.killpg"):
        await sup.unload(model.id)

    # Give the event loop a tick to settle any residual tasks
    await asyncio.sleep(0.05)

    assert spy_called == [], f"Expected spy not called, got {spy_called}"
    assert model.id not in sup._watchers


@pytest.mark.asyncio
async def test_no_on_exit_does_not_break(tmp_path):
    """on_exit=None: subprocess crash still cleans up state without exception."""
    settings = _make_settings(tmp_path)
    sup = Supervisor(settings)
    model = _M()

    proc = _make_proc(returncode_after_wait=139)

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        await sup.load(model, port=10000, on_exit=None)

    # Wait for watcher to finish
    await wait_until_async(
        lambda: model.id not in sup._handles,
        what=f"the watcher to clean up state for '{model.id}'",
    )
    assert model.id not in sup._ports
    assert model.id not in sup._watchers
    assert sup.gpus.owner_of(1) is None
    assert sup.gpus.owner_of(2) is None


@pytest.mark.asyncio
async def test_on_exit_only_fires_once_on_concurrent_unload(tmp_path):
    """Race between proc exit and unload() — on_exit fires at most once."""
    settings = _make_settings(tmp_path)
    sup = Supervisor(settings)
    model = _M()
    call_count = []

    async def spy(rc: int) -> None:
        call_count.append(rc)

    proc = _make_proc(returncode_after_wait=1)

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        await sup.load(model, port=10000, on_exit=spy)

    # Patch unload's proc.wait to complete immediately after SIGTERM
    async def _sigterm_wait():
        proc.returncode = -15
        return -15

    proc.wait = AsyncMock(side_effect=_sigterm_wait)

    # Race: schedule unload concurrently with the watcher already running
    with patch("os.killpg"):
        await asyncio.gather(
            sup.unload(model.id),
            asyncio.sleep(0),  # yield to let watcher potentially fire
        )

    # Allow any residual tasks to settle
    await asyncio.sleep(0.1)

    assert len(call_count) <= 1, f"on_exit fired {len(call_count)} times; expected at most 1"


@pytest.mark.asyncio
async def test_watcher_task_cleared_on_normal_unload(tmp_path):
    """After sup.unload() returns, the watcher task is no longer in _watchers."""
    settings = _make_settings(tmp_path)
    sup = Supervisor(settings)
    model = _M()

    proc = _make_proc(sleep_s=5.0)

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        await sup.load(model, port=10000)

    assert model.id in sup._watchers

    # Transition through state machine to READY so unload() doesn't refuse
    await sup.mark_warming(model.id)
    await sup.mark_ready(model.id)

    async def _immediate_wait():
        proc.returncode = -15
        return -15

    proc.wait = AsyncMock(side_effect=_immediate_wait)

    with patch("os.killpg"):
        await sup.unload(model.id)

    assert model.id not in sup._watchers
    assert model.id not in sup._handles
