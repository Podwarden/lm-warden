import asyncio

import pytest

from app.auth.sse_tickets import TicketStore


@pytest.mark.asyncio
async def test_mint_and_consume_once():
    store = TicketStore(secret="s", ttl_seconds=60)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    user = store.consume(t, "/api/models/abc/logs/stream")
    assert user == "admin"
    with pytest.raises(ValueError):
        store.consume(t, "/api/models/abc/logs/stream")


@pytest.mark.asyncio
async def test_path_binding_enforced():
    store = TicketStore(secret="s", ttl_seconds=60)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    with pytest.raises(ValueError):
        store.consume(t, "/api/models/xyz/logs/stream")


@pytest.mark.asyncio
async def test_expired_ticket_rejected():
    store = TicketStore(secret="s", ttl_seconds=0)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    await asyncio.sleep(0.01)
    with pytest.raises(ValueError):
        store.consume(t, "/api/models/abc/logs/stream")


@pytest.mark.asyncio
async def test_tampered_signature_rejected():
    store = TicketStore(secret="s", ttl_seconds=60)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    bad = t[:-1] + ("A" if t[-1] != "A" else "B")
    with pytest.raises(ValueError):
        store.consume(bad, "/api/models/abc/logs/stream")


@pytest.mark.asyncio
async def test_non_dict_payload_rejected():
    # A correctly-signed but non-dict payload must raise ValueError, not AttributeError.
    import base64
    import hashlib
    import hmac
    import json

    payload = base64.urlsafe_b64encode(json.dumps(42).encode()).decode().rstrip("=")
    sig = (
        base64.urlsafe_b64encode(hmac.new(b"s", payload.encode(), hashlib.sha256).digest())
        .decode()
        .rstrip("=")
    )
    ticket = f"{payload}.{sig}"

    store = TicketStore(secret="s", ttl_seconds=60)
    with pytest.raises(ValueError):
        store.consume(ticket, "/x")


@pytest.mark.asyncio
async def test_missing_iat_rejected():
    # Validly-signed payload without an iat field must raise ValueError, not KeyError.
    import base64
    import hashlib
    import hmac
    import json

    body = {"sub": "admin", "path": "/x"}  # no iat
    payload = (
        base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    sig = (
        base64.urlsafe_b64encode(hmac.new(b"s", payload.encode(), hashlib.sha256).digest())
        .decode()
        .rstrip("=")
    )
    ticket = f"{payload}.{sig}"

    store = TicketStore(secret="s", ttl_seconds=60)
    with pytest.raises(ValueError):
        store.consume(ticket, "/x")


@pytest.mark.asyncio
async def test_same_second_mints_are_distinct_and_independently_consumable():
    # Two back-to-back mints for the same (sub, path) must produce DIFFERENT
    # ticket strings, even when they land in the same whole-second iat. Without
    # the per-mint nonce the payload is byte-identical and HMAC signing is
    # deterministic, so the second mint collides with the first and the
    # single-use deny-list rejects the second consume as "already consumed".
    # No time.sleep() between mints — the test must demonstrate same-second
    # resilience, not paper over it.
    store = TicketStore(secret="s", ttl_seconds=60)
    path = "/api/models/abc/logs/stream"
    t1 = store.mint("admin", path)
    t2 = store.mint("admin", path)
    assert t1 != t2, "same-second re-mint produced byte-identical tickets"
    assert store.consume(t1, path) == "admin"
    # Consuming the first ticket must NOT poison the second.
    assert store.consume(t2, path) == "admin"


@pytest.mark.asyncio
async def test_future_iat_rejected():
    # A signed payload with iat well in the future must be rejected.
    import base64
    import hashlib
    import hmac
    import json
    import time as _t

    body = {"sub": "admin", "path": "/x", "iat": int(_t.time()) + 3600}
    payload = (
        base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    sig = (
        base64.urlsafe_b64encode(hmac.new(b"s", payload.encode(), hashlib.sha256).digest())
        .decode()
        .rstrip("=")
    )
    ticket = f"{payload}.{sig}"

    store = TicketStore(secret="s", ttl_seconds=60)
    with pytest.raises(ValueError):
        store.consume(ticket, "/x")
