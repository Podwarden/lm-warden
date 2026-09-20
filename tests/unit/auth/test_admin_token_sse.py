"""An admin bearer header opens the SSE streams without a ticket (spec
2026-09-19, decision 9), and a token-authenticated stream does not outlive
its token (app/auth/stream_guard.py): revoking the token -- or refreshing it
with no grace -- cancels its streams through the stream registry, and an open
stream re-checks its token and ends once it has expired, been revoked or run
out of rotation grace. That holds for every stream an admin token can open:
the four ticket streams, model pull progress and the chat2 turn stream here,
and the chat playground's completion proxy in
tests/unit/chat/test_playground_stream_guard.py (it needs a POST body and a
stubbed upstream, so it is driven separately). Driven over raw ASGI so a real
stream opens."""

import asyncio
import dataclasses
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request

from app.auth import stream_guard
from app.auth.deps import admin_principal, session_stream_key
from app.auth.stream_guard import guard_stream
from app.chat2.repo import Chat2Repo
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.tokens import sqlite_utc_in, sqlite_utc_now
from app.db.repos.users import UserRepo
from app.proxy.routes_godmode import CONNECTED_FRAME, STREAM_PATH
from app.system.gpu import GpuSnapshot
from app.system.routes_gpus import _ProbeCache
from tests.asgi_stream import first_chunk, open_stream
from tests.conftest import bearer, seed_admin_token

#: The four ticket streams (require_sse_ticket), by name.
TICKET_STREAMS = {
    "godmode": STREAM_PATH,
    "model-logs": "/api/models/demo/logs/stream",
    "stats-live": "/api/stats/live",
    "header-metrics": "/api/header/metrics/stream",
}
#: Every stream an admin token can open. The chat2 path is filled in by the
#: fixture (it needs a chat id); the POST turn stream wraps the same
#: subscriber in the same guard as GET turn/live.
STREAM_NAMES = [*TICKET_STREAMS, "pull-progress", "chat2-turn-live"]


@pytest.fixture
async def streams(live_app, tmp_data_dir: Path) -> dict[str, str]:
    """Every stream able to send its first chunk at once: path by name."""
    live_app.state.settings = dataclasses.replace(live_app.state.settings, godmode_enabled=True)
    # The log stream replays the file's tail before it follows it.
    (tmp_data_dir / "logs").mkdir(exist_ok=True)
    (tmp_data_dir / "logs" / "demo.log").write_text("seed-line\n")

    async def no_gpus() -> GpuSnapshot:
        return GpuSnapshot(gpus=[], apps=[], probe_error=None)

    live_app.state.gpu_probe_cache = _ProbeCache(probe=no_gpus)
    async with open_db(tmp_data_dir / "vllm-warden.db") as db:
        # Pull progress follows a model while it is 'pulling'.
        await ModelRepo(db).insert(ModelRow(
            id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
            gpu_indices=[0], tensor_parallel_size=1, dtype=None,
            max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=[], extra_env={}, status="pulling", pulled_bytes=0,
            pulled_total=None, last_error=None,
        ))
        # The chat2 turn stream follows a live turn of the owner's chat.
        user = await UserRepo(db).get_by_username("admin")
        assert user is not None
        chat = await Chat2Repo(db).create_chat(user.id, model=None, settings={})
    turns = live_app.state.chat2_live_turns
    live = turns.start(chat.id, request_id="req-1", message_id="msg-1")
    await turns.append(live, b"data: {}\n\n")
    return {
        **TICKET_STREAMS,
        "pull-progress": "/api/models/m1/pull/progress",
        "chat2-turn-live": f"/api/chat2/chats/{chat.id}/turn/live",
    }


@pytest.fixture
def streams_app(live_app, streams):
    return live_app


def _set_token(db_path: Path, token_id: str, **cols: str) -> None:
    with sqlite3.connect(db_path) as db:
        sets = ", ".join(f"{k} = ?" for k in cols)
        db.execute(f"UPDATE api_tokens SET {sets} WHERE id = ?", (*cols.values(), token_id))


def _delete_token(db_path: Path, token_id: str) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))


# Every sync sqlite3 seed/mutate below runs through asyncio.to_thread (M7):
# these tests share the event loop with live_app's background writers
# (pruner, sampler, watchdog, history, gc -- main.py's lifespan), each of
# which holds an aiosqlite write transaction across an ``await``. A sync
# call made directly on the loop thread blocks that loop, so a background
# writer whose next step needs the loop can never reach its own commit and
# release the lock -- and a sync retry does not help, since retrying still
# blocks the loop. Running the sync call in a worker thread instead lets the
# loop keep scheduling the writer while sqlite's busy handler waits it out.


async def test_an_admin_bearer_opens_the_godmode_stream(streams_app, tmp_data_dir):
    _, secret = await asyncio.to_thread(seed_admin_token, tmp_data_dir / "vllm-warden.db")
    status, body = await first_chunk(streams_app, STREAM_PATH, bearer(secret))
    assert status == 200
    assert body.startswith(CONNECTED_FRAME.encode())


async def test_an_admin_bearer_opens_the_live_stats_stream(streams_app, tmp_data_dir):
    _, secret = await asyncio.to_thread(seed_admin_token, tmp_data_dir / "vllm-warden.db")
    status, body = await first_chunk(streams_app, "/api/stats/live", bearer(secret))
    assert status == 200
    assert body.startswith(b"data: ")


async def test_a_stream_without_any_credential_is_refused(live_app):
    status, _ = await first_chunk(live_app, "/api/stats/live", {})
    assert status == 401


@pytest.mark.parametrize("name", STREAM_NAMES)
async def test_revoking_the_token_ends_its_stream(streams_app, streams, tmp_data_dir, name):
    """What DELETE /api/admin-tokens/{id} and a zero-grace refresh do:
    cancel_user(admin_token:<id>). A logout -- cancel_user(session:<owner>)
    -- does not touch a script's stream."""
    tid, secret = await asyncio.to_thread(
        seed_admin_token, tmp_data_dir / "vllm-warden.db", created_by="admin"
    )
    registry = streams_app.state.stream_registry
    async with open_stream(streams_app, streams[name], bearer(secret)) as stream:
        assert stream.status == 200, stream.body
        assert registry.count(admin_principal(tid)) == 1
        # The owner's browser logs out: a script's stream stays open.
        assert registry.cancel_user(session_stream_key("admin")) == 0
        assert not await stream.ended(within_s=0.2)
        assert registry.cancel_user(admin_principal(tid)) == 1
        assert await stream.ended(within_s=5.0)
    assert registry.count(admin_principal(tid)) == 0


@pytest.mark.parametrize("name", STREAM_NAMES)
async def test_a_stream_ends_when_its_token_expires(
    streams_app, streams, tmp_data_dir, name, monkeypatch
):
    monkeypatch.setattr(stream_guard, "TOKEN_RECHECK_INTERVAL_S", 0.05)
    db_path = tmp_data_dir / "vllm-warden.db"
    tid, secret = await asyncio.to_thread(
        seed_admin_token, db_path, expires_at=sqlite_utc_in(timedelta(hours=1))
    )
    async with open_stream(streams_app, streams[name], bearer(secret)) as stream:
        assert stream.status == 200, stream.body
        await asyncio.to_thread(_set_token, db_path, tid, expires_at=sqlite_utc_in(timedelta(minutes=-1)))
        assert await stream.ended(within_s=5.0)
    assert streams_app.state.stream_registry.count(admin_principal(tid)) == 0


@pytest.mark.parametrize(
    "dead",
    [
        # A refreshed predecessor whose grace window has run out.
        {"rotated_at": "now", "revoked_at": "past"},
        # Revoked without the registry hearing of it (another process, SQL).
        {"revoked_at": "now"},
        # Gone altogether.
        None,
    ],
    ids=["grace-ended", "revoked", "deleted"],
)
async def test_a_stream_ends_when_its_token_stops_working(
    streams_app, tmp_data_dir, dead, monkeypatch
):
    monkeypatch.setattr(stream_guard, "TOKEN_RECHECK_INTERVAL_S", 0.05)
    db_path = tmp_data_dir / "vllm-warden.db"
    when = {"now": sqlite_utc_now(), "past": sqlite_utc_in(timedelta(minutes=-1))}
    tid, secret = await asyncio.to_thread(
        seed_admin_token,
        db_path, rotated_at=sqlite_utc_now(), revoked_at=sqlite_utc_in(timedelta(hours=1)),
    )  # inside a rotation's grace window: still works
    async with open_stream(streams_app, STREAM_PATH, bearer(secret)) as stream:
        assert stream.status == 200, stream.body
        assert not await stream.ended(within_s=0.3)  # several re-checks, all fine
        if dead is None:
            await asyncio.to_thread(_delete_token, db_path, tid)
        else:
            await asyncio.to_thread(_set_token, db_path, tid, **{k: when[v] for k, v in dead.items()})
        assert await stream.ended(within_s=5.0)


async def test_a_ticket_stream_is_not_rechecked_and_logout_ends_it(
    streams_app, tmp_data_dir, monkeypatch
):
    """A browser's ticket stream registers under session:<username> (logout
    ends it) and never starts a re-check; an admin-token stream does."""
    rechecked: list[str] = []
    real_recheck = stream_guard._recheck

    async def spy(app: Any, token_id: str, stop: Any) -> None:
        rechecked.append(token_id)
        await real_recheck(app, token_id, stop)

    monkeypatch.setattr(stream_guard, "_recheck", spy)
    ticket = streams_app.state.sse_tickets.mint("admin", "/api/stats/live")
    registry = streams_app.state.stream_registry
    async with open_stream(streams_app, f"/api/stats/live?ticket={ticket}", {}) as stream:
        assert stream.status == 200, stream.body
        await asyncio.sleep(0.05)  # a watchdog, were there one, has started
        assert rechecked == []
        assert registry.cancel_user(session_stream_key("admin")) == 1
        assert await stream.ended(within_s=5.0)
    # The spy does see an admin-token stream's re-check.
    tid, secret = await asyncio.to_thread(seed_admin_token, tmp_data_dir / "vllm-warden.db")
    async with open_stream(streams_app, "/api/stats/live", bearer(secret)) as stream:
        assert stream.status == 200, stream.body
        await asyncio.sleep(0.05)
        assert rechecked == [tid]


async def test_the_recheck_ends_the_generator_normally(live_app, tmp_data_dir, monkeypatch):
    """"Cleanly": the stream's generator returns -- its task is not left
    cancelled -- so the response is completed rather than cut off. (Through
    the full app a BaseHTTPMiddleware layer completes the response either
    way, so this is checked on the guard itself.)"""
    monkeypatch.setattr(stream_guard, "TOKEN_RECHECK_INTERVAL_S", 0.05)
    db_path = tmp_data_dir / "vllm-warden.db"
    tid, _ = await asyncio.to_thread(
        seed_admin_token, db_path, expires_at=sqlite_utc_in(timedelta(hours=1))
    )
    request = Request({"type": "http", "app": live_app, "headers": [], "state": {}})
    chunks: list[str] = []

    async def gen():  # the shape of every guarded stream: an idle wait inside
        async with guard_stream(request, admin_principal(tid)):
            yield "first"
            await asyncio.Event().wait()

    async def consume() -> None:
        async for chunk in gen():
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    assert not task.done()
    await asyncio.to_thread(_set_token, db_path, tid, expires_at=sqlite_utc_in(timedelta(minutes=-1)))
    await asyncio.wait_for(task, timeout=5.0)
    assert (task.cancelled(), task.cancelling(), chunks) == (False, 0, ["first"])
    assert live_app.state.stream_registry.count(admin_principal(tid)) == 0
