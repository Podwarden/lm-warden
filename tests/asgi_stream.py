"""Open streaming requests straight through the ASGI app (tests only).

TestClient and httpx's ASGITransport both buffer a response until the app
returns, so neither can open an endless SSE stream. ``open_stream`` sends a
GET -- or, with ``method`` and ``body``, a POST that carries one -- waits for
the first non-empty body chunk (or a non-200 start), and hands the test an
``OpenStream`` while the response is still running; leaving the block answers
the app's next ``receive()`` with ``http.disconnect`` -- a client closing the
tab -- and waits for the app to wind down. ``first_chunk`` is that round trip
in one call.

The scope omits ``asgi.spec_version``, so Starlette 0.41's StreamingResponse
takes its listen-for-disconnect path -- the one production takes too (uvicorn
reports spec_version 2.3).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class OpenStream:
    """What the app has sent so far on one streaming request."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.task: asyncio.Task[None] | None = None

    @property
    def status(self) -> int | None:
        start = next((m for m in self.messages if m["type"] == "http.response.start"), None)
        return None if start is None else int(start["status"])

    @property
    def body(self) -> bytes:
        return b"".join(
            m.get("body", b"") for m in self.messages if m["type"] == "http.response.body"
        )

    async def ended(self, within_s: float) -> bool:
        """Whether the app returns by itself within ``within_s`` seconds, with
        the client still connected."""
        assert self.task is not None
        done, _ = await asyncio.wait({self.task}, timeout=within_s)
        return bool(done)


def _scope(
    path: str, headers: dict[str, str], *, method: str = "GET", body: bytes = b""
) -> dict[str, Any]:
    route, _, query = path.partition("?")
    if body:
        # A JSON body, since every streaming POST in the app takes one. The
        # length has to be on the scope: FastAPI reads the body through
        # ``receive`` and Starlette's own middleware reads Content-Length.
        headers = {
            "content-type": "application/json",
            "content-length": str(len(body)),
            **headers,
        }
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": route,
        "raw_path": route.encode(),
        "root_path": "",
        "query_string": query.encode(),
        "headers": [(b"host", b"testserver")]
        + [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "state": {},
    }


@asynccontextmanager
async def open_stream(
    app: Any,
    path: str,
    headers: dict[str, str],
    *,
    method: str = "GET",
    body: bytes = b"",
    deadline_s: float = 10.0,
) -> AsyncIterator[OpenStream]:
    stream = OpenStream()
    ready = asyncio.Event()
    disconnect = asyncio.Event()
    # A request body is delivered once, on the first receive(); every receive
    # after it is the disconnect wait, exactly as for a GET.
    to_send = [{"type": "http.request", "body": body, "more_body": False}] if body else []

    async def receive() -> dict[str, Any]:
        if to_send:
            return to_send.pop(0)
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        stream.messages.append(message)
        if message["type"] == "http.response.start" and message["status"] != 200:
            ready.set()
        if message["type"] == "http.response.body" and (
            message.get("body") or not message.get("more_body")
        ):
            ready.set()

    task = asyncio.create_task(
        app(_scope(path, headers, method=method, body=body), receive, send)
    )
    task.add_done_callback(lambda _: ready.set())
    stream.task = task
    try:
        await asyncio.wait_for(ready.wait(), deadline_s)
        yield stream
    finally:
        disconnect.set()
        await asyncio.wait_for(task, deadline_s)


async def first_chunk(
    app: Any,
    path: str,
    headers: dict[str, str],
    *,
    method: str = "GET",
    body: bytes = b"",
    deadline_s: float = 10.0,
) -> tuple[int, bytes]:
    """Open ``path``, take the first chunk, disconnect: ``(status, body)``."""
    async with open_stream(
        app, path, headers, method=method, body=body, deadline_s=deadline_s
    ) as stream:
        pass
    assert stream.status is not None
    return stream.status, stream.body
