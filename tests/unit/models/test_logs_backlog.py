"""#216 (slice a): the Live Logs initial backlog must not read the whole log.

The backlog is only the last 200 lines, but it used to be produced by
``content = await f.read()`` over the entire engine log. A long-lived engine's
log can be gigabytes; opening Live Logs then OOMed the api process (which also
supervises every model). These tests pin the bounded trailing-slice read: the
backlog is exactly the last 200 lines, and the read never pulls more than the
cap regardless of total log size.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import aiofiles

# The bound the backlog read must respect, independent of the implementation
# constant so the test keeps failing if the cap is raised above 1 MiB.
CAP_BYTES = 1024 * 1024  # 1 MiB


class _CountingFileProxy:
    """Async context manager proxying an aiofiles file; records how many bytes
    each ``read()`` actually returns. Drives the bounded-read assertion."""

    def __init__(self, cm, counter):
        self._cm = cm
        self._counter = counter
        self._f = None

    async def __aenter__(self):
        self._f = await self._cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return await self._cm.__aexit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._f, name)

    async def read(self, size=-1):
        data = await self._f.read(size)
        self._counter.append(len(data))
        return data


def _write_numbered_log(log_path, n_lines, chunk=200_000):
    """Write ``n_lines`` lines of ``line <i>\\n`` (i from 0), ~13 bytes each."""
    with open(log_path, "wb") as f:
        for start in range(0, n_lines, chunk):
            end = min(start + chunk, n_lines)
            f.write("".join(f"line {i}\n" for i in range(start, end)).encode("ascii"))


def _stream_request(tmp_data_dir):
    from app.auth.stream_registry import StreamRegistry

    settings = MagicMock()
    settings.data_dir = tmp_data_dir
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.stream_registry = StreamRegistry()
    return request


async def _drain_backlog(resp, want):
    """Collect the first ``want`` data lines from the stream, then close it."""
    seen = []
    try:
        async for chunk in resp.body_iterator:
            if chunk.startswith("data: "):
                seen.append(json.loads(chunk[len("data: ") : -len("\n\n")])["line"])
                if len(seen) >= want:
                    break
    finally:
        await resp.body_iterator.aclose()
    return seen


async def test_backlog_is_last_200_and_read_is_bounded(tmp_data_dir, monkeypatch):
    """#216 slice a — red today: the whole ~50 MB log is read (> 1 MiB).

    A long-lived engine's log is gigabytes; the old ``f.read()`` pulled it all
    into memory and OOMed the api process (which also supervises every model)
    the moment the Live Logs panel opened. The backlog is only the last 200
    lines, so the read must be a bounded trailing slice.
    """
    from app.models.routes_logs import stream_logs

    n_lines = 3_900_000  # ~50 MB of "line <i>\n"
    (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = tmp_data_dir / "logs" / "big.log"
    _write_numbered_log(log_path, n_lines)
    expected = [f"line {i}" for i in range(n_lines - 200, n_lines)]

    read_bytes: list[int] = []
    real_open = aiofiles.open

    def spy_open(*a, **k):
        return _CountingFileProxy(real_open(*a, **k), read_bytes)

    monkeypatch.setattr(aiofiles, "open", spy_open)

    request = _stream_request(tmp_data_dir)
    request.is_disconnected = AsyncMock(return_value=False)

    resp = await stream_logs("big", request, user="admin")
    seen = await _drain_backlog(resp, 200)
    assert seen == expected, "initial backlog is not exactly the last 200 lines"

    total = sum(read_bytes)
    assert total <= CAP_BYTES, f"backlog read {total} bytes, more than the {CAP_BYTES} cap"


async def test_backlog_small_log_matches_whole_read(tmp_data_dir):
    """Control: a log smaller than the cap is read in full, so the backlog is
    the whole log — byte-for-byte the old whole-read behaviour."""
    from app.models.routes_logs import stream_logs

    (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = tmp_data_dir / "logs" / "small.log"
    lines = [f"row {i}" for i in range(10)]
    log_path.write_text("".join(line + "\n" for line in lines))

    request = _stream_request(tmp_data_dir)
    request.is_disconnected = AsyncMock(return_value=True)

    resp = await stream_logs("small", request, user="admin")
    seen = await _drain_backlog(resp, len(lines))
    assert seen == lines


async def test_backlog_last_line_without_trailing_newline(tmp_data_dir):
    """A final line with no trailing newline is still part of the backlog."""
    from app.models.routes_logs import stream_logs

    (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = tmp_data_dir / "logs" / "nonl.log"
    log_path.write_text("one\ntwo\nthree")  # no trailing newline

    request = _stream_request(tmp_data_dir)
    request.is_disconnected = AsyncMock(return_value=True)

    resp = await stream_logs("nonl", request, user="admin")
    seen = await _drain_backlog(resp, 3)
    assert seen == ["one", "two", "three"]


async def test_backlog_empty_log_emits_nothing(tmp_data_dir):
    """An empty log yields no backlog lines; the connected-but-empty
    placeholder in the UI carries the UX."""
    from app.models.routes_logs import stream_logs

    (tmp_data_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = tmp_data_dir / "logs" / "empty.log"
    log_path.touch()

    request = _stream_request(tmp_data_dir)
    request.is_disconnected = AsyncMock(return_value=True)

    resp = await stream_logs("empty", request, user="admin")
    seen = await _drain_backlog(resp, 1)
    assert seen == []
