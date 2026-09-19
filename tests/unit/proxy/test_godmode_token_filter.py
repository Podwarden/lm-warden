"""God mode per token (spec 2026-09-18 §3.5).

  * parse_token_ids / event_matches -- the filter's rules
  * GET /api/admin/godmode/stream?token_ids= -- filters the replay AND the live
    events, never splits a request from its deltas, keeps a busy-but-filtered
    stream alive, and without the parameter is byte-for-byte today's stream
    after one leading ": connected" comment (#251)
  * a ticket minted for the BARE path is accepted with a query string
  * GET /api/admin/godmode/status
"""

import asyncio
import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.auth.stream_registry import StreamRegistry
from app.proxy import routes_godmode
from app.proxy.godmode import MAX_TOKEN_IDS, GodModeHub, event_matches, parse_token_ids
from app.proxy.routes_godmode import STREAM_PATH
from tests.conftest import jwt_login, seed_admin_user

# ---- pure --------------------------------------------------------------------


def test_parse_token_ids():
    assert parse_token_ids(None) is None
    assert parse_token_ids("a,b") == {"a", "b"}
    assert parse_token_ids(" a , ,b ") == {"a", "b"}
    assert parse_token_ids(",".join(f"t{i}" for i in range(MAX_TOKEN_IDS))) is not None


@pytest.mark.parametrize(
    "raw", ["", ",", " , ", ",".join(f"t{i}" for i in range(MAX_TOKEN_IDS + 1))]
)
def test_parse_token_ids_refuses(raw):
    with pytest.raises(ValueError):
        parse_token_ids(raw)


def test_event_matches():
    assert event_matches({"token_id": "a"}, None) is True
    assert event_matches({}, None) is True
    assert event_matches({"token_id": "a"}, frozenset({"a"})) is True
    assert event_matches({"token_id": "b"}, frozenset({"a"})) is False
    assert event_matches({}, frozenset({"a"})) is False


# ---- the stream, called directly (as test_godmode_endpoint.py does) ----------


def _request(hub):
    settings = MagicMock()
    settings.godmode_enabled = True
    request = MagicMock()
    request.app.state.settings = settings
    request.app.state.godmode_hub = hub
    request.app.state.stream_registry = StreamRegistry()
    request.is_disconnected = AsyncMock(return_value=False)
    return request


async def _drain(resp, *, until, on_first=None, deadline_s=5.0) -> list[str]:
    seen: list[str] = []

    async def run():
        async for chunk in resp.body_iterator:
            seen.append(chunk)
            if len(seen) == 1 and on_first is not None:
                on_first()
            if until(seen):
                return

    try:
        await asyncio.wait_for(run(), timeout=deadline_s)
    finally:
        await resp.body_iterator.aclose()
    return seen


def _events(chunks: list[str]) -> list[dict]:
    return [json.loads(c[len("data: "):-2]) for c in chunks if c.startswith("data: ")]


def _ev(kind, req, tok, text=None):
    e = {"type": kind, "req_id": req, "token_id": tok}
    if text is not None:
        e["text"] = text
    return e


async def test_filter_keeps_only_the_named_keys_in_replay_and_live():
    hub = GodModeHub()
    for e in (
        _ev("request_start", "r1", "tok-a"),
        _ev("request_start", "r2", "tok-b"),
        _ev("delta", "r1", "tok-a", "a-old"),
        _ev("delta", "r2", "tok-b", "b-old"),
        _ev("request_end", "r1", "tok-a"),
    ):
        hub.publish(e)

    def live():
        # tok-b's live event is published FIRST: if the filter leaked, it
        # would arrive before the tok-a sentinel that ends the drain.
        hub.publish(_ev("delta", "r2", "tok-b", "b-live"))
        hub.publish(_ev("delta", "r3", "tok-a", "a-live"))

    resp = await routes_godmode.godmode_stream(_request(hub), user="admin", token_ids="tok-a")
    seen = await _drain(resp, on_first=live, until=lambda s: any("a-live" in c for c in s))
    events = _events(seen)
    assert {e["token_id"] for e in events} == {"tok-a"}
    # The whole of r1 -- start, delta, end -- survived: nothing was split.
    assert [e["type"] for e in events if e["req_id"] == "r1"] == [
        "request_start", "delta", "request_end",
    ]
    assert events[-1]["text"] == "a-live"
    assert hub.subscriber_count() == 0


async def test_without_token_ids_the_stream_is_unchanged():
    hub = GodModeHub()
    hub.publish(_ev("request_start", "r1", "tok-a"))
    hub.publish(_ev("delta", "r2", "tok-b", "b-old"))
    hub.publish({"type": "request_start", "req_id": "r0"})  # no token_id at all
    probe, ring = hub.subscribe()
    hub.unsubscribe(probe)
    live_event = _ev("delta", "r1", "tok-a", "live")

    resp = await routes_godmode.godmode_stream(_request(hub), user="admin")
    seen = await _drain(
        resp,
        on_first=lambda: hub.publish(live_event),
        until=lambda s: any('"live"' in c for c in s),
    )
    # Byte for byte: every ring event, in order, then the live one -- after
    # the one ": connected" comment every stream now opens with (#251). An SSE
    # comment line is dropped by every parser, so a script reading `data:`
    # frames sees exactly what it saw before.
    assert seen == [
        routes_godmode.CONNECTED_FRAME,
        *(f"data: {json.dumps(e)}\n\n" for e in [*ring, live_event]),
    ]


@pytest.mark.parametrize("token_ids", [None, "tok-a"])
async def test_the_first_bytes_are_a_connected_comment(token_ids):
    # #251: EventSource fires `open` (the "Live" badge) on the first bytes.
    # With an empty ring and a quiet hub those used to be a keepalive up to
    # KEEPALIVE_INTERVAL_S (15 s, left at its default here) later.
    hub = GodModeHub()
    resp = await routes_godmode.godmode_stream(_request(hub), user="admin", token_ids=token_ids)
    seen = await _drain(resp, until=lambda s: len(s) >= 1, deadline_s=1.0)
    assert seen == [": connected\n\n"]
    assert hub.subscriber_count() == 0


async def test_a_filtered_stream_keeps_alive_while_other_keys_are_busy(monkeypatch):
    monkeypatch.setattr(routes_godmode, "KEEPALIVE_INTERVAL_S", 0.1)
    hub = GodModeHub()
    stop = asyncio.Event()

    async def other_key_traffic():
        # Faster than the keepalive interval, so the queue is never idle long
        # enough for the wait_for timeout to fire.
        while not stop.is_set():
            hub.publish(_ev("delta", "rb", "tok-b", "x"))
            await asyncio.sleep(0.01)

    resp = await routes_godmode.godmode_stream(_request(hub), user="admin", token_ids="tok-a")
    pump = asyncio.create_task(other_key_traffic())
    try:
        seen = await _drain(resp, until=lambda s: ": keepalive\n\n" in s, deadline_s=3.0)
    finally:
        stop.set()
        await pump
    assert ": keepalive\n\n" in seen
    assert _events(seen) == []


async def test_a_filtered_stream_keeps_alive_on_time_when_other_keys_are_sparse(monkeypatch):
    # #251: one other-key event just before the keepalive is due used to
    # restart a FULL-interval wait, so the gap reached ~2x the interval. The
    # wait is now the time remaining, so the keepalive still lands on time.
    interval = 0.5
    monkeypatch.setattr(routes_godmode, "KEEPALIVE_INTERVAL_S", interval)
    hub = GodModeHub()
    loop = asyncio.get_running_loop()
    resp = await routes_godmode.godmode_stream(_request(hub), user="admin", token_ids="tok-a")
    started = loop.time()
    loop.call_later(0.8 * interval, hub.publish, _ev("delta", "rb", "tok-b", "x"))
    seen = await _drain(resp, until=lambda s: ": keepalive\n\n" in s, deadline_s=5.0)
    elapsed = loop.time() - started
    assert _events(seen) == []
    # On time, not 0.8 + 1.0 intervals. The 1.5x bound leaves room for a
    # loaded test worker while still failing the old behaviour (~1.8x).
    assert elapsed < 1.5 * interval, elapsed


async def test_too_many_token_ids_is_422_and_subscribes_nothing():
    hub = GodModeHub()
    ids = ",".join(f"t{i}" for i in range(MAX_TOKEN_IDS + 1))
    with pytest.raises(HTTPException) as exc:
        await routes_godmode.godmode_stream(_request(hub), user="admin", token_ids=ids)
    assert exc.value.status_code == 422
    assert hub.subscriber_count() == 0


# ---- over HTTP -----------------------------------------------------------------


def _enable_godmode(client):
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, godmode_enabled=True
    )


def _mint(client, auth, path):
    r = client.post("/api/auth/sse-ticket", json={"path": path}, headers=auth)
    assert r.status_code == 200, r.text
    return r.json()["ticket"]


def test_a_ticket_for_the_bare_path_works_with_a_query_string(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    _enable_godmode(client)
    too_many = ",".join(f"t{i}" for i in range(MAX_TOKEN_IDS + 1))

    # The ticket check compares request.url.path only, so the query string
    # does not matter: getting as far as the handler's own 422 proves the
    # ticket was accepted (a refused ticket is a 401 from the dependency).
    ticket = _mint(client, auth, STREAM_PATH)
    r = client.get(STREAM_PATH, params={"token_ids": too_many, "ticket": ticket})
    assert r.status_code == 422, r.text

    # Minted for the path WITH the query: a different path, so refused.
    ticket = _mint(client, auth, f"{STREAM_PATH}?token_ids=tok-a")
    r = client.get(STREAM_PATH, params={"token_ids": "tok-a", "ticket": ticket})
    assert r.status_code == 401


@pytest.mark.parametrize("raw", ["", ","])
def test_an_empty_token_ids_is_422_over_http(tmp_data_dir, client, raw):
    # #251: `token_ids=` is present but names no key. It must be refused, not
    # read as "absent" -- that would stream EVERY key's traffic to a dock that
    # asked for none.
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    _enable_godmode(client)
    hub = client.app.state.godmode_hub
    before = hub.subscriber_count()
    ticket = _mint(client, auth, STREAM_PATH)
    r = client.get(f"{STREAM_PATH}?token_ids={raw}&ticket={ticket}")
    assert r.status_code == 422, r.text
    assert "token_ids" in r.json()["detail"]
    assert hub.subscriber_count() == before


def test_status_requires_a_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/admin/godmode/status").status_code == 401


def test_status_reports_disabled_then_enabled(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/admin/godmode/status", headers=auth)
    assert r.status_code == 200
    assert r.json() == {"enabled": False}  # off by default
    _enable_godmode(client)
    assert client.get("/api/admin/godmode/status", headers=auth).json() == {"enabled": True}
