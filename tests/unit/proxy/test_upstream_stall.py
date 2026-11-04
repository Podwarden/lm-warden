"""A stalled engine never pins a slot forever (verified finding #4)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from tests.unit.proxy.test_messages import AUTH, _ready

CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}


def _settings(client, **kw):
    client.app.state.settings = dataclasses.replace(client.app.state.settings, **kw)


def _stalling_stream():
    async def aiter():
        yield b'data: {"choices":[{"index":0,"delta":{"content":"he"}}]}\n\n'
        await asyncio.sleep(3600)
        yield b"never"

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _inflight(client) -> int:
    sched = client.app.state.scheduler
    return sum(sched._inflight.values())


def test_a_stream_that_goes_silent_ends_and_frees_its_slot(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _settings(client, upstream_idle_s=0.5)

    async def send(request, *, stream=False, **kw):
        return _stalling_stream()

    t0 = time.monotonic()
    with patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=send)):
        r = client.post("/v1/chat/completions", headers=AUTH, json={**CHAT, "stream": True})
    assert r.status_code == 200 and '"he"' in r.text
    assert time.monotonic() - t0 < 5
    assert _inflight(client) == 0


def test_an_engine_that_never_answers_a_non_stream_request_is_504(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _settings(client, nonstream_max_s=0.5, request_max_wall_s=0.0)

    async def send(request, *, stream=False, **kw):
        await asyncio.sleep(3600)

    t0 = time.monotonic()
    with patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=send)):
        r = client.post("/v1/chat/completions", headers=AUTH, json=CHAT)
    assert r.status_code == 504, r.text
    assert time.monotonic() - t0 < 5
    assert _inflight(client) == 0
    assert "stalled" in json.loads(r.text)["detail"]
