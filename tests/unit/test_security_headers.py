"""Security headers on every api response (app/utils/security_headers.py),
HSTS only over HTTPS -- derived like the session cookie's Secure flag."""

from __future__ import annotations

import dataclasses

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.utils.security_headers import (
    API_CSP,
    HSTS,
    PERMISSIONS_POLICY,
    SecurityHeadersMiddleware,
)
from app.utils.sse import sse_headers

BASE = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    "x-frame-options": "DENY",
    "permissions-policy": PERMISSIONS_POLICY,
    "content-security-policy": API_CSP,
}


def _plain_http(client: TestClient) -> None:
    """Pin the install to plain HTTP (an http origin allowlist, no trusted
    proxy), so the only thing under test is how the request arrived."""
    s = client.app.state.settings
    client.app.state.settings = dataclasses.replace(
        s, allowed_origins=("http://localhost:8080",), trust_proxy_origin=False
    )


def test_every_response_carries_the_base_headers(client):
    for path in ("/healthz", "/api/does-not-exist", "/api/tokens"):
        r = client.get(path)
        for name, value in BASE.items():
            assert r.headers.get(name) == value, (path, name)


def test_no_hsts_over_plain_http(client):
    _plain_http(client)
    assert "strict-transport-security" not in client.get("/healthz").headers


def test_hsts_over_https(client):
    _plain_http(client)
    https = TestClient(client.app, base_url="https://testserver")
    assert https.get("/healthz").headers["strict-transport-security"] == HSTS


def test_forwarded_proto_counts_only_from_a_trusted_front_door(client):
    _plain_http(client)
    spoof = {"X-Forwarded-Proto": "https"}
    assert "strict-transport-security" not in client.get("/healthz", headers=spoof).headers
    s = client.app.state.settings
    client.app.state.settings = dataclasses.replace(s, trust_proxy_origin=True)
    assert client.get("/healthz", headers=spoof).headers["strict-transport-security"] == HSTS


def test_an_all_https_origin_allowlist_means_https(client):
    s = client.app.state.settings
    client.app.state.settings = dataclasses.replace(
        s, allowed_origins=("https://lmwarden.com",), trust_proxy_origin=False
    )
    assert client.get("/healthz").headers["strict-transport-security"] == HSTS


def test_a_routes_own_header_wins(client):
    # The landing page ships its own CSP; the api default must not replace it.
    r = client.get("/_landing")
    assert r.status_code == 200
    assert r.headers["content-security-policy"] != API_CSP
    assert r.headers["x-frame-options"] == "DENY"


def test_sse_hints_survive():
    app = FastAPI()

    @app.get("/s")
    async def s():
        async def gen():
            yield b"data: 1\n\n"
            yield b"data: 2\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers())

    app.add_middleware(SecurityHeadersMiddleware)
    with TestClient(app) as c, c.stream("GET", "/s") as r:
        assert r.headers["x-accel-buffering"] == "no"
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert b"".join(r.iter_bytes()) == b"data: 1\n\ndata: 2\n\n"
