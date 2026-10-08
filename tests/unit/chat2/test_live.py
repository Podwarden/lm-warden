"""Unit tests for the detached-turn registry (app/chat2/live.py)."""

import asyncio

from app.chat2.live import LiveTurns


async def _collect(agen) -> list[bytes]:
    return [f async for f in agen]


async def test_subscribe_replays_then_follows_and_is_byte_identical() -> None:
    reg = LiveTurns(done_ttl_s=60.0)
    live = reg.start("c1", request_id="r1", message_id="m1")

    # first subscriber attaches before any frame exists
    first = asyncio.ensure_future(_collect(reg.subscribe(live)))
    await asyncio.sleep(0)
    await reg.append(live, b"data: one\n\n")
    await reg.append(live, b"data: two\n\n")
    await asyncio.sleep(0)

    # second subscriber joins MID-stream: replay of frames[0:] first
    second = asyncio.ensure_future(_collect(reg.subscribe(live)))
    await asyncio.sleep(0)
    await reg.append(live, b"data: three\n\n")
    await reg.finish(live)

    assert await first == [b"data: one\n\n", b"data: two\n\n", b"data: three\n\n"]
    assert await second == await first  # byte-identical, replay included
    await reg.shutdown()


async def test_subscriber_cancellation_leaves_other_subscribers_alone() -> None:
    reg = LiveTurns(done_ttl_s=60.0)
    live = reg.start("c1", request_id="r1", message_id="m1")
    survivor = asyncio.ensure_future(_collect(reg.subscribe(live)))
    doomed = asyncio.ensure_future(_collect(reg.subscribe(live)))
    await asyncio.sleep(0)
    await reg.append(live, b"data: one\n\n")
    await asyncio.sleep(0)
    doomed.cancel()  # the client disconnect shape
    await asyncio.gather(doomed, return_exceptions=True)
    await reg.append(live, b"data: two\n\n")
    await reg.finish(live)
    assert await survivor == [b"data: one\n\n", b"data: two\n\n"]
    await reg.shutdown()


async def test_finished_entry_is_reaped_after_the_ttl() -> None:
    reg = LiveTurns(done_ttl_s=0.01)
    live = reg.start("c1", request_id="r1", message_id="m1")
    await reg.append(live, b"data: one\n\n")
    await reg.finish(live)
    # inside the grace window a late subscriber still gets the full replay
    assert reg.get("c1") is live
    assert await _collect(reg.subscribe(live)) == [b"data: one\n\n"]
    await asyncio.sleep(0.05)
    assert reg.get("c1") is None


async def test_a_newer_turn_replaces_the_old_entry_immediately() -> None:
    reg = LiveTurns(done_ttl_s=0.0)
    old = reg.start("c1", request_id="r1", message_id="m1")
    await reg.finish(old)
    new = reg.start("c1", request_id="r2", message_id="m2")
    assert reg.get("c1") is new
    # the displaced entry's reaper must not remove the NEW entry
    await reg._reap(old)
    assert reg.get("c1") is new
    await reg.shutdown()
