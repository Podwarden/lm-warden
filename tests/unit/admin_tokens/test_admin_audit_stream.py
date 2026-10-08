"""The audit middleware does not break a stream, and records it once the
client goes away -- driven over raw ASGI so a real SSE stream opens."""

import asyncio
import dataclasses
import sqlite3

from app.proxy.routes_godmode import CONNECTED_FRAME, STREAM_PATH
from tests.asgi_stream import first_chunk
from tests.conftest import seed_admin_token


async def test_a_stream_is_audited_after_it_closes(live_app, tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    live_app.state.settings = dataclasses.replace(live_app.state.settings, godmode_enabled=True)
    # Off the loop (M7): live_app's lifespan just started several background
    # writers (pruner, sampler, ...) that hold aiosqlite transactions across
    # an ``await``; a sync BEGIN IMMEDIATE on the loop thread would block the
    # loop those writers need to reach their own commit.
    tid, secret = await asyncio.to_thread(seed_admin_token, db_path)
    status, body = await first_chunk(live_app, STREAM_PATH, {"Authorization": f"Bearer {secret}"})
    assert (status, body.startswith(CONNECTED_FRAME.encode())) == (200, True)
    # The row is enqueued when the stream closes and batched by the background
    # flusher (#258); drain it explicitly rather than sleeping past the
    # interval. On the app's own loop, so it is the app's own writer.
    assert await live_app.state.admin_audit.flush() == 1
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT token_id, method, path, status FROM admin_audit").fetchall()
    assert rows == [(tid, "GET", STREAM_PATH, 200)]
