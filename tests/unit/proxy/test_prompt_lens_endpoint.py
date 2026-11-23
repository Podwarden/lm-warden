"""Prompt Lens SSE endpoint — GET /api/admin/prompt-lens/stream.

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
from app.proxy.prompt_lens import PromptLensHub
from app.proxy.routes_prompt_lens import STREAM_PATH
from tests.conftest import jwt_login, seed_admin_user


def test_stream_rejects_unauthenticated(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    # A bogus ticket hits require_sse_ticket's consume path → 401.
    r = client.get("/api/admin/prompt-lens/stream?ticket=bogus.sig")
    assert r.status_code == 401


# --------------------------------------------------------------------------
# Ticket-mint gate — the browser's first request. EventSource cannot read the
# stream endpoint's 409 (onerror has no HTTP status), so the shared SSE-ticket
# mint MUST itself refuse a Prompt Lens ticket while the feature is off. That
# 4xx is what the frontend classifies as the "disabled" placeholder.
# --------------------------------------------------------------------------
def test_mint_prompt_lens_ticket_disabled_returns_409(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    # prompt_lens_enabled is false by default.
    r = client.post("/api/auth/sse-ticket", json={"path": STREAM_PATH}, headers=auth)
    assert r.status_code == 409, r.text
    assert "disabled" in r.text.lower()


def test_mint_prompt_lens_ticket_enabled_returns_ticket(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )
    r = client.post("/api/auth/sse-ticket", json={"path": STREAM_PATH}, headers=auth)
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["ticket"], str) and r.json()["ticket"]


def test_mint_nonprompt_lens_ticket_unaffected_when_prompt_lens_disabled(tmp_data_dir, client):
    # Regression: the gate must be scoped to the Prompt Lens path only. The
    # model-log ticket flow keeps working while Prompt Lens is off.
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
    from app.proxy.routes_prompt_lens import prompt_lens_stream

    settings = MagicMock()
    settings.prompt_lens_enabled = False
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.prompt_lens_hub = PromptLensHub()

    with pytest.raises(HTTPException) as exc:
        await prompt_lens_stream(request, user="admin")
    assert exc.value.status_code == 409
    assert "disabled" in exc.value.detail.lower()


async def test_stream_replays_ring_then_streams_live_then_unsubscribes(tmp_data_dir):
    from app.proxy import routes_prompt_lens

    hub = PromptLensHub()
    hub.publish({"type": "request_start", "req_id": "r1", "prompt": "history"})

    settings = MagicMock()
    settings.prompt_lens_enabled = True
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.prompt_lens_hub = hub
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(return_value=False)

    resp = await routes_prompt_lens.prompt_lens_stream(request, user="admin")
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
                    hub.publish(
                        {"type": "delta", "req_id": "r1", "channel": "content", "text": "livetoken"}
                    )
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
            json.loads(chunk[len("data: ") : -len("\n\n")])
    # The finally must have unsubscribed.
    assert hub.subscriber_count() == 0


async def test_stream_emits_keepalive_when_idle(tmp_data_dir):
    from app.proxy import routes_prompt_lens

    hub = PromptLensHub()
    settings = MagicMock()
    settings.prompt_lens_enabled = True
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.prompt_lens_hub = hub
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(return_value=False)

    saved = routes_prompt_lens.KEEPALIVE_INTERVAL_S
    routes_prompt_lens.KEEPALIVE_INTERVAL_S = 0.05
    try:
        resp = await routes_prompt_lens.prompt_lens_stream(request, user="admin")
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
        routes_prompt_lens.KEEPALIVE_INTERVAL_S = saved
    assert hub.subscriber_count() == 0


# --------------------------------------------------------------------------
# G9: GET /api/admin/prompt-lens/requests/{req_id}/section/{kind}
# --------------------------------------------------------------------------
def _section_traffic(hub):
    for tid, text in (("tok1", "alpha reply"), ("tok2", "bravo reply")):
        rid = f"r-{tid}"
        hub.publish({"type": "request_start", "req_id": rid, "token_id": tid, "request_id": rid})
        hub.publish(
            {"type": "delta", "req_id": rid, "token_id": tid, "channel": "content", "text": text}
        )


def test_admin_section_any_key(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, prompt_lens_enabled=True
    )
    _section_traffic(client.app.state.prompt_lens_hub)
    for tid, text in (("tok1", "alpha reply"), ("tok2", "bravo reply")):
        r = client.get(f"/api/admin/prompt-lens/requests/r-{tid}/section/reply", headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"] == text and body["live"] is True and body["offset"] == 0
        assert isinstance(body["seq"], int) and body["seq"] > 0
        assert "token_id" not in body and body["kept"] is False
    r = client.get("/api/admin/prompt-lens/requests/r-tok1/section/reply?from=6", headers=auth)
    assert (r.json()["offset"], r.json()["text"]) == (6, "reply")
    r = client.get("/api/admin/prompt-lens/requests/nope/section/reply", headers=auth)
    assert r.status_code == 404
    r = client.get("/api/admin/prompt-lens/requests/r-tok1/section/bogus", headers=auth)
    assert r.status_code == 422
    r = client.get("/api/admin/prompt-lens/requests/r-tok1/section/reply?from=-1", headers=auth)
    assert r.status_code == 422


def test_admin_section_off_serves_no_memory(tmp_data_dir, client):
    # Off: only kept captures (on disk) are served; what memory holds is not.
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    _section_traffic(client.app.state.prompt_lens_hub)
    r = client.get("/api/admin/prompt-lens/requests/r-tok1/section/reply", headers=auth)
    assert r.status_code == 404


def test_admin_section_needs_a_session(tmp_data_dir, client):
    client.get("/healthz")
    r = client.get("/api/admin/prompt-lens/requests/r-tok1/section/reply")
    assert r.status_code == 401
