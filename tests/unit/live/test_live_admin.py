"""Unit tests for tests/live/_admin.py's list_tokens() pagination (I-b) --
no real network, httpx.MockTransport stands in for the server.
"""

from __future__ import annotations

import httpx
import pytest

from tests.live._admin import AdminSession


def _login_and_csrf_routes(request: httpx.Request) -> httpx.Response | None:
    if request.url.path == "/api/auth/login" and request.method == "POST":
        return httpx.Response(200, json={"access_token": "tok", "expires_in": 900})
    if request.url.path == "/api/csrf" and request.method == "GET":
        return httpx.Response(200, json={"csrf": "csrftoken"})
    return None


def _token(i: int) -> dict:
    return {"id": f"t{i}", "name": f"token-{i}", "is_expired": False, "is_revoked": False, "is_paused": False}


@pytest.mark.asyncio
async def test_list_tokens_current_unpaged_shape_is_one_call():
    """No `total` field (today's production shape) -- treated as already
    complete, exactly one GET."""
    calls = {"tokens": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        common = _login_and_csrf_routes(request)
        if common is not None:
            return common
        assert request.url.path == "/api/tokens"
        calls["tokens"] += 1
        return httpx.Response(200, json={"items": [_token(1), _token(2)]})

    admin = AdminSession(
        "https://example.invalid", "admin", "pw", transport=httpx.MockTransport(handler)
    )
    items = await admin.list_tokens()
    await admin.aclose()

    assert [i["id"] for i in items] == ["t1", "t2"]
    assert calls["tokens"] == 1


@pytest.mark.asyncio
async def test_list_tokens_paged_shape_follows_every_page():
    """`total` present -- keep requesting until offset + len(items) >= total."""
    all_tokens = [_token(i) for i in range(7)]
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        common = _login_and_csrf_routes(request)
        if common is not None:
            return common
        assert request.url.path == "/api/tokens"
        offset = int(request.url.params.get("offset", "0"))
        limit = int(request.url.params.get("limit", "50"))
        calls.append(offset)
        page = all_tokens[offset : offset + limit]
        return httpx.Response(
            200,
            json={"items": page, "total": len(all_tokens), "limit": limit, "offset": offset},
        )

    admin = AdminSession(
        "https://example.invalid", "admin", "pw", transport=httpx.MockTransport(handler)
    )
    items = await admin.list_tokens()
    await admin.aclose()

    assert [i["id"] for i in items] == [f"t{i}" for i in range(7)]


@pytest.mark.asyncio
async def test_list_tokens_paged_shape_with_small_pages_still_gets_everything():
    """A page smaller than our requested limit=500 (server enforces its own
    cap) still must be followed until total is reached."""
    all_tokens = [_token(i) for i in range(5)]
    PAGE = 2

    def handler(request: httpx.Request) -> httpx.Response:
        common = _login_and_csrf_routes(request)
        if common is not None:
            return common
        offset = int(request.url.params.get("offset", "0"))
        page = all_tokens[offset : offset + PAGE]
        return httpx.Response(
            200,
            json={"items": page, "total": len(all_tokens), "limit": PAGE, "offset": offset},
        )

    admin = AdminSession(
        "https://example.invalid", "admin", "pw", transport=httpx.MockTransport(handler)
    )
    items = await admin.list_tokens()
    await admin.aclose()

    assert [i["id"] for i in items] == [f"t{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_list_tokens_stops_on_empty_page_even_if_total_not_reached():
    """Defensive: a server bug returning an empty page before offset
    reaches total must not spin forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        common = _login_and_csrf_routes(request)
        if common is not None:
            return common
        return httpx.Response(200, json={"items": [], "total": 100, "limit": 500, "offset": 0})

    admin = AdminSession(
        "https://example.invalid", "admin", "pw", transport=httpx.MockTransport(handler)
    )
    items = await admin.list_tokens()
    await admin.aclose()

    assert items == []
