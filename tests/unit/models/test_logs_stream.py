import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiofiles
import pytest
from fastapi.responses import StreamingResponse

from tests.conftest import jwt_login, seed_admin_user


# #55 fix — route through the shared barrier+retry helpers.
def _seed_done(db_path):
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


async def test_logs_stream_creates_missing_log_file(tmp_data_dir):
    """Regression for v2026.05.15.5 — used to raise HTTPException(404)
    when the log file didn't exist yet. That 404 surfaced through
    useEventSource's ticket-mint preflight as a terminal-error and the
    UI stuck on "stream unavailable" even though the subprocess would
    create the file moments later. The route now matches the
    supervisor's O_CREAT | O_APPEND behaviour — the file is touched
    and the stream opens normally.

    Verified at the handler level (rather than via TestClient) because
    the StreamingResponse is endless from the HTTP transport's view —
    httpx wouldn't see the disconnect until the response is consumed,
    so a TestClient `.get` would block. The file-creation side-effect
    happens synchronously inside the handler before the
    StreamingResponse is returned, so we can assert on it without
    draining the body.
    """
    from app.auth.stream_registry import StreamRegistry
    from app.models.routes_logs import stream_logs

    (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = tmp_data_dir / "logs" / "missing.log"
    assert not log_path.exists(), "precondition: file must not exist yet"

    settings = MagicMock()
    settings.data_dir = tmp_data_dir

    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(side_effect=[True])

    resp = await stream_logs("missing", request, user="admin")
    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    # Assert BEFORE aclose() — the file-creation side-effect must happen
    # synchronously in the handler prologue, NOT lazily inside the generator.
    # If this assertion ever needs to move after aclose(), the open-or-create
    # logic has regressed into lazy creation.
    assert log_path.exists(), "log file was not created by the route handler"
    # Close the iterator so the generator's finally runs.
    await resp.body_iterator.aclose()


async def test_logs_stream_500_when_logs_dir_missing(tmp_data_dir):
    """Defensive: a missing logs/ parent directory is a deployment bug,
    not a "subprocess hasn't started" race. The route must NOT silently
    paper over it with O_CREAT — that would create the file outside the
    expected directory structure on systems where the parent path is a
    typo or symlink target.
    """
    from fastapi import HTTPException

    from app.auth.stream_registry import StreamRegistry
    from app.models.routes_logs import stream_logs

    # Deliberately do NOT create the logs/ subdirectory.
    assert not (tmp_data_dir / "logs").exists()

    settings = MagicMock()
    settings.data_dir = tmp_data_dir

    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(side_effect=[True])

    with pytest.raises(HTTPException) as exc:
        await stream_logs("orphan", request, user="admin")
    assert exc.value.status_code == 500
    assert "log directory does not exist" in exc.value.detail


async def test_logs_stream_observes_post_create_append(tmp_data_dir):
    """Open-or-create round-trip: route creates the file, an external
    writer appends a line, the stream surfaces it. This is the
    end-to-end contract the UI relies on for the "subprocess starts
    after stream opens" race window.
    """
    from app.auth.stream_registry import StreamRegistry
    from app.models import routes_logs

    # Speed up the tail poll cadence so the writer-then-read round-trip
    # finishes within a couple of cycles instead of the default 0.5s.
    # We're testing the open-or-create + tail contract, not the poll
    # cadence itself.
    saved_poll = routes_logs.TAIL_POLL_S
    routes_logs.TAIL_POLL_S = 0.05
    try:
        (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
        log_path = tmp_data_dir / "logs" / "post-append.log"
        assert not log_path.exists()

        settings = MagicMock()
        settings.data_dir = tmp_data_dir

        # Stay connected indefinitely (we exit via aclose, not via
        # disconnect). The Mock side_effect returns False on every
        # call rather than running out and raising StopIteration.
        request = MagicMock()
        request.app.state.settings = settings
        request.app.state.stream_registry = StreamRegistry()
        request.is_disconnected = AsyncMock(return_value=False)

        resp = await routes_logs.stream_logs("post-append", request, user="admin")
        # File must be created by the time the handler returns.
        assert log_path.exists(), "open-or-create did not run"

        # Append the line BEFORE we start draining, so we don't race
        # the tail loop's seek-to-end. aiofiles.open keeps the test
        # async-clean (ASYNC230) and mirrors the route's own I/O path.
        async with aiofiles.open(log_path, "a") as f:
            await f.write("hello from writer\n")

        seen: list[str] = []
        try:

            async def drain():
                async for chunk in resp.body_iterator:
                    seen.append(chunk)
                    if any('"line": "hello from writer"' in c for c in seen):
                        return

            await asyncio.wait_for(drain(), timeout=5.0)
        finally:
            await resp.body_iterator.aclose()
        assert any('"line": "hello from writer"' in c for c in seen), seen
    finally:
        routes_logs.TAIL_POLL_S = saved_poll


async def test_logs_stream_emits_keepalive_when_idle(tmp_data_dir):
    """v2026.05.15.5: idle SSE streams must emit `: keepalive\\n\\n`
    so intermediate proxies (and EventSource's own heuristics) don't
    tear the connection down during long pre-first-line silences.

    We override KEEPALIVE_INTERVAL_S and TAIL_POLL_S to sub-second
    values rather than waiting 15s of real time per test. The contract
    under test is "a keepalive comment line appears on the wire when
    no log lines arrive within the interval".
    """
    from app.auth.stream_registry import StreamRegistry
    from app.models import routes_logs

    saved_ka = routes_logs.KEEPALIVE_INTERVAL_S
    saved_poll = routes_logs.TAIL_POLL_S
    routes_logs.KEEPALIVE_INTERVAL_S = 0.1
    routes_logs.TAIL_POLL_S = 0.05
    try:
        (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
        log_path = tmp_data_dir / "logs" / "idle.log"
        # Pre-create empty so the seeded-history section emits nothing
        # and the generator drops straight into the idle _tail loop.
        log_path.touch()

        settings = MagicMock()
        settings.data_dir = tmp_data_dir

        request = MagicMock()
        request.app.state.settings = settings
        request.app.state.stream_registry = StreamRegistry()
        # Stay connected indefinitely — we exit via aclose() below.
        request.is_disconnected = AsyncMock(return_value=False)

        resp = await routes_logs.stream_logs("idle", request, user="admin")

        seen: list[str] = []
        try:

            async def drain():
                async for chunk in resp.body_iterator:
                    seen.append(chunk)
                    if any(c.startswith(": keepalive") for c in seen):
                        return

            await asyncio.wait_for(drain(), timeout=3.0)
        finally:
            await resp.body_iterator.aclose()
        assert any(c.startswith(": keepalive") for c in seen), seen
        # SSE comment lines are terminated by \n\n — pin the exact
        # framing so a future "let's simplify" pass can't accidentally
        # turn this into a data: event or skip the blank-line terminator.
        keepalives = [c for c in seen if c.startswith(": keepalive")]
        assert all(c == ": keepalive\n\n" for c in keepalives), keepalives
    finally:
        routes_logs.KEEPALIVE_INTERVAL_S = saved_ka
        routes_logs.TAIL_POLL_S = saved_poll


async def test_tail_survives_in_run_truncation(tmp_path):
    """#216 (slice b): the in-run sweep copy-truncates the log while a
    follower has it open. The follower's read position then sits past the
    end of the (much shorter) file; without rewinding it would return EOF
    forever and the Live Logs panel would go silent for the rest of the run.
    The follower must notice the size drop below its read position, seek to
    the start, and keep following — which is exactly where the writer's
    O_APPEND fd appends next.
    """
    from app.models import routes_logs

    saved_poll = routes_logs.TAIL_POLL_S
    routes_logs.TAIL_POLL_S = 0.02
    try:
        log_path = tmp_path / "m.log"
        # The pre-rotation body. The post-rotation rewrite below is SHORTER,
        # so the follower's old offset ends up past the new end of file.
        log_path.write_text("".join(f"old {i}\n" for i in range(200)))

        request = MagicMock()
        request.is_disconnected = AsyncMock(return_value=False)
        stream = routes_logs._tail(log_path, request)
        try:

            async def take(n: int) -> list:
                got: list = []

                async def collect():
                    async for item in stream:
                        got.append(item)
                        if len(got) >= n:
                            return

                await asyncio.wait_for(collect(), timeout=5.0)
                return got

            # Two idle ticks: the follower is parked at the end of the
            # pre-rotation body.
            await take(2)
            # The rotation, as the sweep performs it: the live file is zeroed
            # in place and the writer appends again.
            log_path.write_text("new 0\nnew 1\n")
            # A tick may come first (the rewind happens in the idle branch);
            # the new lines must follow within a few polls.
            got = await take(4)
        finally:
            await stream.aclose()

        lines = [g for g in got if g is not routes_logs._TAIL_TICK]
        assert "new 0" in lines, got
    finally:
        routes_logs.TAIL_POLL_S = saved_poll


async def test_logs_stream_returns_sse_content_type(tmp_data_dir):
    """Verify stream_logs returns StreamingResponse with text/event-stream.

    Calls the route handler directly (bypassing ASGI/HTTP) to avoid the
    httpx-transport limitation where receive() blocks on response_complete
    before delivering http.disconnect — which hangs any HTTP-level test of
    an infinite SSE generator.
    """
    from app.auth.stream_registry import StreamRegistry
    from app.models.routes_logs import stream_logs

    log_path = tmp_data_dir / "logs" / "qwen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello world\n")

    # Minimal settings stub.
    settings = MagicMock()
    settings.data_dir = tmp_data_dir

    # Request stub: always disconnected after first check so _tail exits.
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.stream_registry = StreamRegistry()
    # First call returns False (let the buffered lines flow), subsequent True.
    request.is_disconnected = AsyncMock(side_effect=[False, True, True])

    resp = await stream_logs("qwen", request, user="admin")

    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    # Anti-buffering headers MUST be present on every SSE response
    # (#50). nginx/Caddy buffer text/event-stream by default; without
    # X-Accel-Buffering: no the live-logs panel would see chunks arrive
    # in proxy-buffer-sized bursts rather than as they're emitted.
    # Cache-Control: no-cache is defensive against CDNs that don't
    # recognise the SSE media type.
    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.headers["cache-control"] == "no-cache"

    # Registration happens inside the generator on first iteration, so the
    # registry is still empty until we pull at least one chunk.
    assert request.app.state.stream_registry.count("admin") == 0

    # Consume the generator to ensure it yields SSE data and terminates.
    lines = []
    async for chunk in resp.body_iterator:
        lines.append(chunk)
        if not lines:
            continue
        if len(lines) == 1:
            # While the generator is actively streaming, the user MUST be
            # registered. This catches a regression where the register/
            # unregister pair drifts out of the generator body.
            assert request.app.state.stream_registry.count("admin") >= 1
        if len(lines) >= 5:
            # Stop pulling but also close the iterator so the generator's
            # `finally` runs and unregisters before we assert below.
            await resp.body_iterator.aclose()
            break

    assert any(chunk.startswith("data:") for chunk in lines)
    # After the generator has been closed, the registry must be empty.
    # Guards the success-path register/unregister contract.
    assert request.app.state.stream_registry.count("admin") == 0


async def test_logs_stream_emits_json_per_line(tmp_data_dir):
    """Pin the SSE payload shape: `data: {"line": "<raw>"}\\n\\n`.

    The frontend useEventSource hook JSON.parses every event payload (see
    frontend/src/lib/sse.ts) and silently drops parse failures. Emitting raw
    text would compile and ship with no obvious symptom — the FE would just
    render nothing forever. This test pins the contract so a future "let's
    simplify" pass can't quietly break the live-log UI.
    """
    from app.auth.stream_registry import StreamRegistry
    from app.models.routes_logs import stream_logs

    log_path = tmp_data_dir / "logs" / "qwen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("first line\nsecond line\n")

    settings = MagicMock()
    settings.data_dir = tmp_data_dir

    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(side_effect=[False, True, True])

    resp = await stream_logs("qwen", request, user="admin")

    # Drain just the seeded-history chunks (two lines from the file) — the
    # _tail loop will then exit on the True from is_disconnected.
    seen: list[str] = []
    async for chunk in resp.body_iterator:
        seen.append(chunk)
        if len(seen) >= 2:
            await resp.body_iterator.aclose()
            break

    assert len(seen) >= 2
    # Each chunk is exactly `data: <json>\n\n` and the JSON has shape
    # {"line": "<text>"}.
    for chunk, expected in zip(seen, ["first line", "second line"], strict=False):
        assert chunk.startswith("data: ") and chunk.endswith("\n\n"), chunk
        payload = json.loads(chunk[len("data: ") : -len("\n\n")])
        assert payload == {"line": expected}
