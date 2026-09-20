"""The playground completion stream ends with its credential (#258).

``POST /api/chat/completions`` is the seventh token-authenticated stream in the
app -- the chat playground's own generation, proxied through the warden to
``/v1/chat/completions``. It was the only one not wrapped in
``app/auth/stream_guard.py``, so revoking the admin token that opened it left
the generation running to completion. It is now registered under the request's
stream-registry key and re-checked like the other six.

Driven over raw ASGI (tests/asgi_stream.py) so a real stream opens, with the
upstream stubbed to send one frame and then hang -- the shape of a long
generation. The six GET streams are covered together in
tests/unit/auth/test_admin_token_sse.py; this one needs a POST body and a
stubbed upstream, so it lives here.
"""

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.auth import stream_guard
from app.auth.deps import admin_principal, session_stream_key
from app.db.repos.tokens import sqlite_utc_in
from tests.asgi_stream import open_stream
from tests.conftest import bearer, seed_admin_token

PATH = "/api/chat/completions"
BODY = json.dumps(
    {"model": "m1", "messages": [{"role": "user", "content": "hello"}]}
).encode()
FIRST_FRAME = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'


class _HangingUpstream:
    """What ``httpx.AsyncClient.send(stream=True)`` returns here: a
    /v1/chat/completions response that sends one SSE frame and then keeps the
    connection open, like a model still generating."""

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self) -> None:
        self.closed = False

    async def aiter_raw(self):
        yield FIRST_FRAME
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
async def upstreams(live_app, monkeypatch) -> list[_HangingUpstream]:
    """The playground wired to a hanging upstream; the responses it handed out."""
    opened: list[_HangingUpstream] = []

    async def fake_send(self: httpx.AsyncClient, _request: Any, **_kw: Any) -> Any:
        resp = _HangingUpstream()
        opened.append(resp)
        return resp

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    # The handler 409s without a cached plaintext (POST /playground/ensure
    # normally puts it there).
    await live_app.state.playground_store.set("pg1", "vw_playground_plaintext")
    return opened


async def test_revoking_the_token_ends_an_in_flight_playground_stream(
    live_app, tmp_data_dir: Path, upstreams
) -> None:
    """What DELETE /api/admin-tokens/{id} and a zero-grace refresh do:
    cancel_user(admin_token:<id>). A logout does not touch a script's stream."""
    tid, secret = await asyncio.to_thread(
        seed_admin_token, tmp_data_dir / "vllm-warden.db", created_by="admin"
    )
    registry = live_app.state.stream_registry
    async with open_stream(
        live_app, PATH, bearer(secret), method="POST", body=BODY
    ) as stream:
        assert stream.status == 200, stream.body
        assert stream.body.startswith(FIRST_FRAME), stream.body
        # Registered under the token, not its owner: a browser logout must not
        # cut a script's generation.
        assert registry.count(admin_principal(tid)) == 1
        assert registry.cancel_user(session_stream_key("admin")) == 0
        assert not await stream.ended(within_s=0.2)
        assert registry.cancel_user(admin_principal(tid)) == 1
        assert await stream.ended(within_s=5.0)
    assert registry.count(admin_principal(tid)) == 0
    # The upstream and the in-flight counter are still released.
    assert upstreams[0].closed
    assert live_app.state.chat_active_requests.count() == 0


async def test_an_expiring_token_ends_an_in_flight_playground_stream(
    live_app, tmp_data_dir: Path, upstreams, monkeypatch
) -> None:
    """The 60 s re-check, for a revoke this process never heard about."""
    monkeypatch.setattr(stream_guard, "TOKEN_RECHECK_INTERVAL_S", 0.05)
    db_path = tmp_data_dir / "vllm-warden.db"
    tid, secret = await asyncio.to_thread(
        seed_admin_token, db_path, expires_at=sqlite_utc_in(timedelta(hours=1))
    )

    def expire() -> None:
        import sqlite3

        with sqlite3.connect(db_path) as db:
            db.execute(
                "UPDATE api_tokens SET expires_at = ? WHERE id = ?",
                (sqlite_utc_in(timedelta(minutes=-1)), tid),
            )

    async with open_stream(
        live_app, PATH, bearer(secret), method="POST", body=BODY
    ) as stream:
        assert stream.status == 200, stream.body
        assert not await stream.ended(within_s=0.2)  # several re-checks, all fine
        # Off the loop (M7): live_app's background writers hold aiosqlite
        # transactions across an await, and a sync sqlite3 call on the loop
        # thread would block them from reaching their commit.
        await asyncio.to_thread(expire)
        assert await stream.ended(within_s=5.0)
    assert live_app.state.stream_registry.count(admin_principal(tid)) == 0
    assert live_app.state.chat_active_requests.count() == 0


async def test_a_session_playground_stream_registers_under_the_session(
    live_app, tmp_data_dir: Path, upstreams
) -> None:
    """The browser's own generation: keyed ``session:<username>``, so
    POST /api/auth/logout ends it (the other streams' contract)."""
    from app.auth.csrf import generate_csrf_token
    from app.auth.jwt import mint_access

    jwt = mint_access("admin", live_app.state.jwt_secret, ttl_minutes=5)
    # A session POST keeps the CSRF check an admin bearer skips: bind a known
    # csrf id through the cookie and send the token derived from it.
    csrf_id = "playground-guard-test"
    session = {
        **bearer(jwt),
        "cookie": f"vw_csrf_id={csrf_id}",
        "x-csrf-token": generate_csrf_token(
            csrf_id, secret=live_app.state.settings.cookie_secret
        ),
    }
    registry = live_app.state.stream_registry
    async with open_stream(live_app, PATH, session, method="POST", body=BODY) as stream:
        assert stream.status == 200, stream.body
        assert registry.count(session_stream_key("admin")) == 1
        assert registry.cancel_user(session_stream_key("admin")) == 1
        assert await stream.ended(within_s=5.0)
    assert registry.count(session_stream_key("admin")) == 0
