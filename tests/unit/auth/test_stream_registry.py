import asyncio

import pytest

from app.auth.stream_registry import StreamRegistry


@pytest.mark.asyncio
async def test_register_and_cancel():
    reg = StreamRegistry()

    cancelled = asyncio.Event()

    async def fake_stream():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(fake_stream())
    reg.register("admin", task)
    assert reg.count("admin") == 1

    # Yield control so the task starts and enters its try block before we cancel.
    await asyncio.sleep(0)

    reg.cancel_user("admin")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert reg.count("admin") == 0


@pytest.mark.asyncio
async def test_unregister_on_completion():
    reg = StreamRegistry()
    task = asyncio.create_task(asyncio.sleep(0))
    reg.register("admin", task)
    await asyncio.sleep(0.01)
    reg.unregister("admin", task)
    assert reg.count("admin") == 0
