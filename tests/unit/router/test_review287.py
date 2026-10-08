"""#287 review fixes: each test here was red before its fix."""

import asyncio
import json
import sqlite3
import time

import httpx
import pytest

from tests.unit.router.harness import (
    CLAUDE_CODE,
    PLAINTEXT,
    decisions,
    fake_engine,
    haiku_body,
    sse_upstream,
)
from tests.unit.router.test_data_plane import (
    HAIKU_RULE,
    _count_body,
    _engine_502,
    _engine_returning,
    _half_open,
    _post,
    _setup,
)


def raw_asgi(client, method, raw_path: bytes, headers: dict[str, str], body: bytes = b""):
    """One request with an exact raw_path (TestClient normalises paths)."""
    app = client.app
    hdrs = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    hdrs.append((b"content-length", str(len(body)).encode()))
    path, _, query = raw_path.partition(b"?")

    async def go():
        sent: list[dict] = []
        msgs = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            if msgs:
                return msgs.pop(0)
            await asyncio.sleep(0.05)
            return {"type": "http.disconnect"}

        async def send(m):
            sent.append(m)

        from urllib.parse import unquote

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": unquote(path.decode("latin-1")),
            "raw_path": path,
            "query_string": query,
            "root_path": "",
            "headers": hdrs,
            "client": ("1.2.3.4", 1),
            "server": ("testserver", 80),
            "state": {},
        }
        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return status, data

    return client.portal.call(go)


# review #2 ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_path",
    [
        b"/v1/%2E%2e/admin",
        b"/v1/.%2e/x",
        b"/v1/%2e./x",
        b"/v1/..%2fx",
        b"/v1/../admin",
    ],
)
def test_dot_segments_are_refused_400_with_their_own_reason(tmp_data_dir, client, raw_path):
    _, up = _setup(client, tmp_data_dir)
    status, data = raw_asgi(client, "GET", raw_path, CLAUDE_CODE)
    assert status == 400, data
    assert "bad_path" in data.decode()
    assert not up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"], d["status"]) == ("refused", "bad_path", 400)
    snap = client.app.state.router.snapshot(None)
    assert snap["passthrough"]["requests"] == 0 and snap["passthrough"]["errors"] == 0


# review #3 ------------------------------------------------------------------

import base64  # noqa: E402


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": f"vw_{'a' * 30}"},
        {"Authorization": f"Basic vw_{'a' * 30}"},
        {"Authorization": f"Bearer VW_{'a' * 30}"},
        {"Authorization": "Bearer vwa_adminsecret1234567890abcdef"},
        {"Authorization": "Basic " + base64.b64encode(f"vw_{'a' * 30}:x".encode()).decode()},
        {"X-API-Key": f"vw_{'a' * 30}"},
        {"X-API-Key": f"Bearer vwa_{'a' * 30}"},
        {"Authorization": "Bearer sk-ant-ok", "X-API-Key": f"vw_{'a' * 30}"},
    ],
)
def test_a_warden_or_admin_key_in_any_upstream_credential_is_never_relayed(
    tmp_data_dir, client, headers
):
    _, up = _setup(client, tmp_data_dir)
    hdrs = {"X-LMWarden-Key": PLAINTEXT, "anthropic-version": "2023-06-01", **headers}
    r = _post(client, haiku_body(model="claude-sonnet-4-5"), headers=hdrs)
    assert r.status_code == 403, r.text
    r2 = client.get("/v1/models/claude-x", headers=hdrs)
    assert r2.status_code == 403
    assert not up.called


def test_an_anthropic_key_that_merely_contains_vw_inside_a_token_is_relayed(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    hdrs = {
        "X-LMWarden-Key": PLAINTEXT,
        "Authorization": "Bearer sk-ant-api03-abcdVw_efghVWA_ijkl",
        "anthropic-version": "2023-06-01",
    }
    r = _post(client, haiku_body(model="claude-sonnet-4-5"), headers=hdrs)
    assert r.status_code == 200 and up.called


# review #5 ------------------------------------------------------------------
# The catch-all must not change what an UNAUTHENTICATED caller sees on paths the
# warden owns or does not have: ownership is decided before authentication.

from tests.unit.router.harness import enable_router, ready  # noqa: E402


@pytest.mark.parametrize("router_on", [False, True])
@pytest.mark.parametrize(
    "method,path,status",
    [
        ("DELETE", "/v1/chat/completions", 405),
        ("GET", "/v1/messages", 405),
        ("GET", "/v1/messages/count_tokens", 405),
        ("OPTIONS", "/v1/chat/completions", 405),
        ("POST", "/v1/chat/completions/", 307),
        ("POST", "/v1/messages/", 307),
        ("OPTIONS", "/v1/foo", 404),
        ("PUT", "/v1/models", 405),
    ],
)
def test_owned_paths_answer_before_authentication(
    tmp_data_dir, client, router_on, method, path, status
):
    ready(client, tmp_data_dir)
    if router_on:
        enable_router(client, tmp_data_dir, [HAIKU_RULE])
    r = client.request(method, path, follow_redirects=False)
    assert r.status_code == status, r.text


def test_unknown_path_without_credential_is_401_router_off_or_on(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    assert client.get("/v1/nothing").status_code == 401
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    assert client.get("/v1/nothing").status_code == 401


def test_options_to_an_unknown_path_is_never_relayed(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    assert client.options("/v1/nothing", headers=CLAUDE_CODE).status_code == 404
    assert not up.called


# review #6 ------------------------------------------------------------------


def test_fallback_never_relays_a_model_outside_the_keys_allow_list(tmp_data_dir, client):
    ready(client, tmp_data_dir, allowed_models="qwen")
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    from tests.unit.router.harness import fake_anthropic

    up = fake_anthropic(client)
    with fake_engine(client, _engine_502):
        r = _post(client, haiku_body())  # served by qwen, engine down -> would fall back
    assert r.status_code == 403 and "token_not_allowed" in r.json()["error"]["message"]
    assert not up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"]) == ("refused", "token_not_allowed")


def test_count_tokens_fallback_respects_the_allow_list_too(tmp_data_dir, client):
    ready(client, tmp_data_dir, allowed_models="qwen")
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    from tests.unit.router.harness import fake_anthropic

    up = fake_anthropic(client)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='pulled' WHERE id='qwen'")
        db.commit()
    r = _post(client, _count_body(), path="/v1/messages/count_tokens")
    assert r.status_code == 529 and not up.called  # the local error, not a relay refusal


def test_catch_all_is_closed_when_passthrough_unmatched_is_off(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, passthrough_unmatched=False)
    r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, json=haiku_body())
    assert r.status_code == 404 and not up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"], d["status"]) == ("refused", "passthrough_disabled", 404)


def test_catch_all_is_refused_for_keys_with_a_model_allow_list(tmp_data_dir, client):
    ready(client, tmp_data_dir, allowed_models="qwen")
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    from tests.unit.router.harness import fake_anthropic

    up = fake_anthropic(client)
    r = client.post(
        "/v1/messages/batches", headers=CLAUDE_CODE, json={"requests": [haiku_body(model="x")]}
    )
    assert r.status_code == 403 and "token_not_allowed" in r.json()["error"]["message"]
    assert not up.called


# review #8 ------------------------------------------------------------------


def _asgi_stream(client, headers: dict[str, str], chunks: list[bytes]):
    """POST /v1/messages with an exact sequence of body messages and no
    content-length unless given; returns (status, body, messages_consumed)."""
    app = client.app
    hdrs = [(k.lower().encode(), v.encode()) for k, v in headers.items()]

    async def go():
        sent: list[dict] = []
        pending = [
            {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
            for i, c in enumerate(chunks)
        ]
        consumed = 0

        async def receive():
            nonlocal consumed
            if pending:
                consumed += 1
                return pending.pop(0)
            await asyncio.sleep(0.05)
            return {"type": "http.disconnect"}

        async def send(m):
            sent.append(m)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"",
            "root_path": "",
            "headers": hdrs,
            "client": ("1.2.3.4", 1),
            "server": ("testserver", 80),
            "state": {},
        }
        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return status, data, consumed

    return client.portal.call(go)


def test_chunked_body_over_the_cap_is_413_and_reading_stops(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, max_body_mb=1)
    chunk = b"x" * (256 * 1024)
    hdrs = {**CLAUDE_CODE, "content-type": "application/json", "transfer-encoding": "chunked"}
    with fake_engine(client) as calls:
        status, data, consumed = _asgi_stream(client, hdrs, [chunk] * 40)  # 10 MiB offered
    assert status == 413 and b"request_too_large" in data
    assert consumed <= 6  # cap is 4 chunks; never the rest
    assert not up.called and not calls
    assert decisions(client)[0]["reason"] == "request_too_large"


def test_chunked_body_under_the_cap_works(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, max_body_mb=1)
    raw = json.dumps(haiku_body(model="claude-sonnet-4-5")).encode()
    hdrs = {**CLAUDE_CODE, "content-type": "application/json", "transfer-encoding": "chunked"}
    status, data, _ = _asgi_stream(client, hdrs, [raw[:20], raw[20:]])
    assert status == 200, data
    assert up.requests[0]["body"] == raw


def test_declared_content_length_over_the_cap_is_refused_without_reading(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, max_body_mb=1)
    hdrs = {**CLAUDE_CODE, "content-type": "application/json", "content-length": str(5 << 20)}
    status, data, consumed = _asgi_stream(client, hdrs, [b"{}"])
    assert status == 413 and consumed == 0
    assert not up.called


# review #9 ------------------------------------------------------------------


def test_relay_client_has_explicit_pool_limits_and_a_finite_pool_timeout():
    from app.router.passthrough import make_relay_client

    c = make_relay_client()
    pool = c._transport._pool  # type: ignore[attr-defined]
    assert pool._max_connections == 256 and pool._max_keepalive_connections == 32
    t = c.timeout
    assert t.pool is not None and 0 < t.pool <= 30
    assert t.connect is not None
    assert t.read is None and t.write is None  # long generations stay legitimate
    assert c.follow_redirects is False


def test_app_uses_the_bounded_relay_client(client):
    client.get("/healthz")
    c = client.app.state.router_http
    assert c.timeout.pool is not None
    assert c._transport._pool._max_connections == 256  # type: ignore[attr-defined]


def test_pool_exhaustion_is_a_529_not_a_hang(tmp_data_dir, client):
    def boom(_req):
        raise httpx.PoolTimeout("pool full")

    _, up = _setup(client, tmp_data_dir, anthropic=boom)
    r = _post(client, haiku_body(model="claude-sonnet-4-5"))
    assert r.status_code == 529, r.text
    assert r.json()["error"]["type"] == "overloaded_error"
    d = decisions(client)[0]
    assert (d["route"], d["reason"], d["status"]) == ("error", "relay_pool_exhausted", 529)


# review #10 -----------------------------------------------------------------


def _explode(*_a, **_k):
    raise RuntimeError("db is locked")


def test_unexpected_local_error_is_a_local_failure_that_falls_back(
    tmp_data_dir, client, monkeypatch
):
    import app.proxy.routes_messages as rm

    _, up = _setup(client, tmp_data_dir)
    monkeypatch.setattr(rm, "_serve_local", _explode)
    r = _post(client, haiku_body())
    assert r.status_code == 200 and up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"]) == ("fallback", "local_error")
    assert client.app.state.router._breakers["qwen"].failures == 1  # breaker-counted
    assert "db is locked" not in r.text and "db is locked" not in json.dumps(d)


def test_unexpected_local_error_without_fallback_is_a_529(tmp_data_dir, client, monkeypatch):
    import app.proxy.routes_messages as rm

    _, up = _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "fallback": False}])
    monkeypatch.setattr(rm, "_serve_local", _explode)
    r = _post(client, haiku_body())
    assert r.status_code == 529 and "local_error" in r.json()["error"]["message"]
    assert "db is locked" not in r.text
    assert not up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"], d["status"]) == ("refused", "local_error", 529)


def test_tokenizer_crash_in_count_tokens_falls_back_without_the_breaker(tmp_data_dir, client):
    from unittest.mock import AsyncMock

    _, up = _setup(client, tmp_data_dir)
    client.app.state.tokenizers.count = AsyncMock(side_effect=RuntimeError("tokenizer died"))
    r = _post(client, _count_body(), path="/v1/messages/count_tokens")
    assert r.status_code == 200 and up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"]) == ("fallback", "local_error")
    assert "qwen" not in client.app.state.router._breakers or (
        client.app.state.router._breakers["qwen"].failures == 0
    )


def test_a_real_client_cancel_is_still_client_closed_499(tmp_data_dir, client, monkeypatch):
    import app.proxy.routes_messages as rm

    _, up = _setup(client, tmp_data_dir)

    async def cancelled(*_a, **_k):
        raise asyncio.CancelledError

    monkeypatch.setattr(rm, "_serve_local", cancelled)
    with pytest.raises(BaseException):  # noqa: B017 - the cancel propagates
        _post(client, haiku_body())
    d = decisions(client)[0]
    assert (d["route"], d["reason"], d["status"]) == ("local", "client_closed", 499)
    assert not up.called
    assert "qwen" not in client.app.state.router._breakers


# test gap: _discard must release the admission slot and the DP rank -----------


def _slow_first_chunk():
    from unittest.mock import AsyncMock, MagicMock

    from tests.unit.router.harness import chunk

    async def aiter():
        await asyncio.sleep(3)
        yield b"data: " + json.dumps(chunk({"role": "assistant", "content": ""})).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _discard_case(client, tmp_data_dir, make_engine_response):
    def anthropic(_req):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"event: message_start\n\n"
        )

    _, up = _setup(client, tmp_data_dir, anthropic=anthropic, local_header_timeout_s=1)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET data_parallel_size=2 WHERE id='qwen'")
        db.commit()

    async def h(_req):
        return make_engine_response()

    with fake_engine(client, h):
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
        ) as r:
            b"".join(r.iter_bytes())
    assert up.called
    assert client.app.state.scheduler._inflight_for_test("qwen") == 0
    assert client.app.state.dp_routing.snapshot("qwen", 2)["totals"]["in_flight"] == 0
    return decisions(client)[0]


def test_first_byte_timeout_releases_slot_and_rank(tmp_data_dir, client):
    d = _discard_case(client, tmp_data_dir, _slow_first_chunk)
    assert (d["route"], d["reason"]) == ("fallback", "first_byte_timeout")


def test_first_frame_error_releases_slot_and_rank(tmp_data_dir, client):
    d = _discard_case(client, tmp_data_dir, lambda: sse_upstream([{"error": {"message": "x"}}]))
    assert (d["route"], d["reason"]) == ("fallback", "first_frame_error")


# survivors of the first mutation run ----------------------------------------


def test_upstream_credentials_are_empty_unless_the_request_used_x_lmwarden_key():
    from types import SimpleNamespace

    from app.proxy.auth import upstream_credential_headers

    def req(flag: bool):
        return SimpleNamespace(
            state=SimpleNamespace(lmwarden_key_header=flag),
            headers={"authorization": "Bearer vw_x", "x-api-key": "k"},
        )

    assert upstream_credential_headers(req(False)) == {}
    assert upstream_credential_headers(req(True)) == {
        "authorization": "Bearer vw_x",
        "x-api-key": "k",
    }


def test_target_allow_list_miss_never_falls_back_even_when_the_requested_model_is_allowed(
    tmp_data_dir, client
):
    from tests.unit.router.harness import fake_anthropic

    ready(client, tmp_data_dir, allowed_models="claude-haiku-4-5")  # not 'qwen', the target
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    up = fake_anthropic(client)
    r = _post(client, haiku_body())
    assert r.status_code == 403 and "token_not_allowed" in r.json()["error"]["message"]
    assert not up.called


def test_probe_slot_is_released_at_the_first_frame_not_the_end_of_the_stream(tmp_data_dir, client):
    import threading

    from tests.unit.router.harness import chunk

    _, up = _setup(client, tmp_data_dir, local_header_timeout_s=10)
    st = client.app.state.router
    for _ in range(3):
        st.record_failure("qwen", "status_502", threshold=3, open_s=60)
    st._breakers["qwen"].open_until = time.time() - 1

    from unittest.mock import AsyncMock, MagicMock

    async def aiter():
        yield b"data: " + json.dumps(chunk({"role": "assistant", "content": ""})).encode() + b"\n\n"
        await asyncio.sleep(1.5)
        yield b"data: " + json.dumps(chunk({"content": "hi"})).encode() + b"\n\n"
        yield b"data: " + json.dumps(chunk({}, finish="stop")).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"

    def slow_stream():
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/event-stream"}
        resp.aiter_bytes = aiter
        resp.aclose = AsyncMock()
        return resp

    def go():
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
        ) as r:
            b"".join(r.iter_bytes())

    with fake_engine(client, _engine_returning(slow_stream)):
        t = threading.Thread(target=go)
        t.start()
        time.sleep(0.6)  # first frame delivered, stream still running
        mid = st._breakers["qwen"].probe_claim
        mid_open = st._breakers["qwen"].open_until
        t.join()
    assert mid is None
    assert mid_open is None  # the first frame closes the breaker (N3), not the stream end
    assert st.breaker_allows("qwen") and st._breakers["qwen"].open_until is None


def test_probe_claim_outlives_the_local_timeout_plus_margin():
    from app.router import state as state_mod
    from app.router.state import RouterState

    t = [1000.0]
    st = RouterState(now=lambda: t[0])
    for _ in range(3):
        st.record_failure("m", "x", threshold=3, open_s=60)
    t[0] += 61
    assert st.breaker_admit("m", timeout_s=3600) > 0
    t[0] += 3600  # a hung engine is still within its own timeout
    assert st.breaker_admit("m", timeout_s=3600) is None
    t[0] += state_mod.PROBE_MARGIN_S + 1
    assert st.breaker_admit("m", timeout_s=3600) > 0


from tests.unit.router.harness import chat_completion, json_upstream  # noqa: E402


def test_no_outcome_fallback_releases_the_probe_before_the_relay(tmp_data_dir, client):
    import threading

    async def slow_anth(_r):
        await asyncio.sleep(1.5)
        return httpx.Response(200, json={"id": "m", "type": "message"})

    _, up = _setup(client, tmp_data_dir, anthropic=slow_anth)
    _half_open(client)
    raw = json.dumps(haiku_body()).encode()
    seen = []

    async def eng(_r):
        seen.append(1)
        if len(seen) == 1:
            return json_upstream({"error": {"message": "bad"}}, 400)
        return chat_completion()

    res = []

    def go():
        res.append(_post(client, None, raw=raw).status_code)

    with fake_engine(client, eng):
        t1 = threading.Thread(target=go)
        t1.start()
        time.sleep(0.5)  # probe 1 is waiting on Anthropic's headers
        t2 = threading.Thread(target=go)
        t2.start()
        t1.join()
        t2.join()
    assert len(seen) == 2  # the second request was admitted as the next probe
    assert ("fallback", "breaker_open") not in [
        (d["route"], d["reason"]) for d in decisions(client)
    ]


def test_no_v1_route_registered_after_the_catch_all(client):
    from app.proxy.routes_messages import find_routes_after_catch_all

    assert find_routes_after_catch_all(client.app) == []
