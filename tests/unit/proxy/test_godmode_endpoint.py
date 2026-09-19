"""God-mode SSE endpoint — GET /api/admin/godmode/stream.

  * Unauthenticated (no/bad SSE ticket) → 401.
  * Disabled → clear 409 disabled response (not a dead stream).
  * Enabled → replay the ring snapshot then stream live events; unsubscribe on
    close.
"""

import asyncio
import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.auth.stream_registry import StreamRegistry
from app.proxy.godmode import GodModeHub
from app.proxy.routes_godmode import STREAM_PATH
from tests.conftest import jwt_login, seed_admin_user


def test_stream_rejects_unauthenticated(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    # A bogus ticket hits require_sse_ticket's consume path → 401.
    r = client.get("/api/admin/godmode/stream?ticket=bogus.sig")
    assert r.status_code == 401


# --------------------------------------------------------------------------
# Ticket-mint gate — the browser's first request. EventSource cannot read the
# stream endpoint's 409 (onerror has no HTTP status), so the shared SSE-ticket
# mint MUST itself refuse a god-mode ticket while the feature is off. That
# 4xx is what the frontend classifies as the "disabled" placeholder.
# --------------------------------------------------------------------------
def test_mint_godmode_ticket_disabled_returns_409(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    # godmode_enabled is false by default.
    r = client.post("/api/auth/sse-ticket", json={"path": STREAM_PATH}, headers=auth)
    assert r.status_code == 409, r.text
    assert "disabled" in r.text.lower()


def test_mint_godmode_ticket_enabled_returns_ticket(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, godmode_enabled=True
    )
    r = client.post("/api/auth/sse-ticket", json={"path": STREAM_PATH}, headers=auth)
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["ticket"], str) and r.json()["ticket"]


def test_mint_nongodmode_ticket_unaffected_when_godmode_disabled(tmp_data_dir, client):
    # Regression: the gate must be scoped to the god-mode path only. The
    # model-log ticket flow keeps working while god mode is off.
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/models/abc/logs/stream"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["ticket"], str) and r.json()["ticket"]


async def test_stream_disabled_returns_409(tmp_data_dir):
    from app.proxy.routes_godmode import godmode_stream

    settings = MagicMock()
    settings.godmode_enabled = False
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.godmode_hub = GodModeHub()

    with pytest.raises(HTTPException) as exc:
        await godmode_stream(request, user="admin")
    assert exc.value.status_code == 409
    assert "disabled" in exc.value.detail.lower()


async def test_stream_replays_ring_then_streams_live_then_unsubscribes(tmp_data_dir):
    from app.proxy import routes_godmode

    hub = GodModeHub()
    hub.publish({"type": "request_start", "req_id": "r1", "prompt": "history"})

    settings = MagicMock()
    settings.godmode_enabled = True
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.godmode_hub = hub
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(return_value=False)

    resp = await routes_godmode.godmode_stream(request, user="admin")
    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    # Anti-buffering headers on every SSE response (#50).
    assert resp.headers["x-accel-buffering"] == "no"

    seen: list[str] = []
    try:
        async def drain():
            async for chunk in resp.body_iterator:
                seen.append(chunk)
                # After the replay frame flushes, inject a live event.
                if len(seen) == 1:
                    hub.publish({"type": "delta", "req_id": "r1",
                                 "channel": "content", "text": "livetoken"})
                if any("livetoken" in c for c in seen):
                    return

        await asyncio.wait_for(drain(), timeout=5.0)
    finally:
        await resp.body_iterator.aclose()

    # The connected comment (#251), then the ring replay, then the live event.
    assert seen[0] == ": connected\n\n"
    assert any('"history"' in c for c in seen)
    assert any("livetoken" in c for c in seen)
    # Every data frame is valid SSE JSON.
    for chunk in seen:
        if chunk.startswith("data: "):
            json.loads(chunk[len("data: "):-len("\n\n")])
    # The finally must have unsubscribed.
    assert hub.subscriber_count() == 0


async def test_stream_emits_keepalive_when_idle(tmp_data_dir):
    from app.proxy import routes_godmode

    hub = GodModeHub()
    settings = MagicMock()
    settings.godmode_enabled = True
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.godmode_hub = hub
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(return_value=False)

    saved = routes_godmode.KEEPALIVE_INTERVAL_S
    routes_godmode.KEEPALIVE_INTERVAL_S = 0.05
    try:
        resp = await routes_godmode.godmode_stream(request, user="admin")
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
        assert any(c == ": keepalive\n\n" for c in seen), seen
    finally:
        routes_godmode.KEEPALIVE_INTERVAL_S = saved
    assert hub.subscriber_count() == 0
