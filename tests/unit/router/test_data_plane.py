"""#287 task 3: the router data plane through the app, fake upstreams on both legs."""

import asyncio
import inspect
import json
import sqlite3
import time
from unittest.mock import patch

import httpx

from tests.unit.router.harness import (
    ANTHROPIC_TOKEN,
    CLAUDE_CODE,
    PLAINTEXT,
    chat_completion,
    chunk,
    decisions,
    enable_router,
    fake_anthropic,
    fake_engine,
    good_stream,
    haiku_body,
    json_upstream,
    ready,
    sse_events,
    sse_upstream,
)

HAIKU_RULE = {"pattern": "claude-haiku*", "target": "qwen"}


def _post(client, body, *, headers=CLAUDE_CODE, path="/v1/messages?beta=true", raw=None):
    if raw is not None:
        return client.post(
            path, headers={**headers, "content-type": "application/json"}, content=raw
        )
    return client.post(path, headers=headers, json=body)


def _setup(client, tmp_data_dir, rules=(HAIKU_RULE,), relay=True, anthropic=None, **settings):
    ready(client, tmp_data_dir, relay=relay)
    ids = enable_router(client, tmp_data_dir, list(rules), **settings)
    up = fake_anthropic(client, anthropic)
    return ids, up


def _err(r, status, kind):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["type"] == "error" and body["error"]["type"] == kind, body
    return body["error"]["message"]


async def _engine_502(_req):
    raise httpx.ConnectError("refused")


def _engine_returning(resp):
    async def h(_req):
        return resp() if inspect.isfunction(resp) else resp

    return h


# 1 -------------------------------------------------------------------------


def test_rule_serves_locally_with_thinking_off_and_requested_name_echoed(tmp_data_dir, client):
    ids, up = _setup(client, tmp_data_dir)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        r = _post(client, haiku_body())
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "claude-haiku-4-5"
    assert not up.called
    (req,) = calls
    sent = json.loads(req.content)
    assert sent["model"] == "qwen"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    hdrs = {k.lower() for k in req.headers}
    assert not hdrs & {"authorization", "x-api-key", "x-lmwarden-key", "cookie"}
    assert ANTHROPIC_TOKEN not in str(req.headers) and PLAINTEXT not in str(req.headers)
    (d,) = decisions(client)
    assert d["route"] == "local" and d["rule_id"] == ids[0] and d["model_out"] == "qwen"
    assert d["model_in"] == "claude-haiku-4-5" and d["token_name"] == "laptop"


def test_strip_thinking_off_keeps_client_thinking(tmp_data_dir, client):
    _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "strip_thinking": False}])
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        assert _post(client, haiku_body()).status_code == 200
    assert "chat_template_kwargs" not in json.loads(calls[0].content)


# 2 -------------------------------------------------------------------------


def test_min_max_tokens_raises_but_never_lowers(tmp_data_dir, client):
    _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "min_max_tokens": 1024}])
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        _post(client, haiku_body(max_tokens=32))
        _post(client, haiku_body(max_tokens=3000))
    assert [json.loads(c.content)["max_tokens"] for c in calls] == [1024, 3000]


def test_min_max_tokens_zero_leaves_it(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        _post(client, haiku_body(max_tokens=32))
    assert json.loads(calls[0].content)["max_tokens"] == 32


# 3 -------------------------------------------------------------------------


def test_unmatched_model_is_passed_through_byte_for_byte(tmp_data_dir, client):
    def handler(_req):
        return httpx.Response(
            529,
            content=b'{"type":"error","error":{"type":"overloaded_error","message":"x"}}',
            headers={"request-id": "req_9", "content-type": "application/json"},
        )

    _, up = _setup(client, tmp_data_dir, anthropic=handler)
    raw = b'{"max_tokens": 5,  "model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}'
    with fake_engine(client) as engine_calls:
        r = _post(
            client,
            None,
            raw=raw,
            headers={**CLAUDE_CODE, "Cookie": "vw_session=zzz", "X-LMWarden-Trace": "t"},
        )
    assert not engine_calls
    assert r.status_code == 529 and r.headers["request-id"] == "req_9"
    assert r.content == b'{"type":"error","error":{"type":"overloaded_error","message":"x"}}'
    (seen,) = up.requests
    assert seen["method"] == "POST" and seen["path"] == "/v1/messages"
    assert seen["query"] == "beta=true"
    assert seen["host"] == "anthropic.example"
    assert seen["body"] == raw
    h = seen["headers"]
    assert h["authorization"] == f"Bearer {ANTHROPIC_TOKEN}"
    assert h["anthropic-version"] == "2023-06-01"
    assert "x-lmwarden-key" not in h and "x-lmwarden-trace" not in h and "cookie" not in h
    assert h["accept-encoding"] == "gzip, deflate"  # the test client's own, forwarded as sent
    assert PLAINTEXT not in json.dumps(h)
    (d,) = decisions(client)
    assert d["route"] == "passthrough" and d["rule_id"] is None and d["status"] == 529


def test_x_api_key_credential_is_forwarded_too(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    headers = {"X-LMWarden-Key": PLAINTEXT, "x-api-key": "sk-ant-api03-FAKE"}
    r = _post(client, haiku_body(model="claude-sonnet-4-5"), headers=headers)
    assert r.status_code == 200
    assert up.requests[0]["headers"]["x-api-key"] == "sk-ant-api03-FAKE"
    assert "authorization" not in up.requests[0]["headers"]


# 4 -------------------------------------------------------------------------


def test_passthrough_stream_is_delivered_frame_by_frame(tmp_data_dir, client):
    holder: dict[str, asyncio.Event] = {}
    frames = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n", b"event: c\ndata: 3\n\n"]

    def handler(_req):
        async def gen():
            yield frames[0]
            await holder["gate"].wait()
            yield frames[1]
            yield frames[2]

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())

    _, up = _setup(client, tmp_data_dir, anthropic=handler)
    body = json.dumps(haiku_body(model="claude-sonnet-4-5", stream=True)).encode()
    app = client.app

    async def run():
        gate = holder["gate"] = asyncio.Event()
        chunks: list[bytes] = []
        first, done = asyncio.Event(), asyncio.Event()
        started: dict = {}
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            await done.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.start":
                started.update(status=msg["status"], headers=dict(msg["headers"]))
            elif msg["type"] == "http.response.body" and msg.get("body"):
                chunks.append(msg["body"])
                first.set()

        headers = [(k.lower().encode(), v.encode()) for k, v in CLAUDE_CODE.items()]
        headers += [(b"content-type", b"application/json"), (b"host", b"testserver")]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"beta=true",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 5000),
            "server": ("testserver", 80),
            "state": {},
        }
        task = asyncio.ensure_future(app(scope, receive, send))
        await asyncio.wait_for(first.wait(), 5)  # frame 1 reaches the client...
        before = list(chunks)
        gate_was_set = gate.is_set()  # ...while the upstream is still blocked
        gate.set()
        await asyncio.wait_for(task, 5)
        done.set()
        return started, before, gate_was_set, chunks

    started, before, gate_was_set, chunks = client.portal.call(run)
    assert gate_was_set is False and before == [frames[0]]
    assert b"".join(chunks) == b"".join(frames)
    assert started["status"] == 200
    assert started["headers"][b"content-type"] == b"text/event-stream"
    (d,) = decisions(client)
    assert d["route"] == "passthrough" and d["stream"] is True and d["ttfb_ms"] is not None
    assert up.called


# 5 -------------------------------------------------------------------------


def test_served_name_with_router_on_is_local_without_rule(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        r = _post(client, haiku_body(model="qwen"))
    assert r.status_code == 200 and r.json()["model"] == "qwen"
    assert not up.called and len(calls) == 1
    (d,) = decisions(client)
    assert d["route"] == "local" and d["rule_id"] is None


# 6 -------------------------------------------------------------------------


def test_fallback_replays_original_to_anthropic_and_breaker_opens(tmp_data_dir, client):
    ids, up = _setup(client, tmp_data_dir)
    clock = {"t": time.time()}
    client.app.state.router._now = lambda: clock["t"]
    raw = json.dumps(haiku_body()).encode()
    with fake_engine(client, _engine_502) as calls:
        for _ in range(3):
            r = _post(client, None, raw=raw)
            assert r.status_code == 200
        assert len(calls) == 3
        # original model, thinking intact, client's own credential
        first = up.requests[0]
        assert first["body"] == raw
        assert first["headers"]["authorization"] == f"Bearer {ANTHROPIC_TOKEN}"
        assert [d["route"] for d in decisions(client)] == ["fallback"] * 3
        assert decisions(client)[0]["reason"] == "status_502"
        snap = client.app.state.router.snapshot(None)
        assert snap["by_reason"]["status_502"] == 3
        # 4th: breaker open -> Anthropic WITHOUT touching the engine
        r = _post(client, None, raw=raw)
        assert r.status_code == 200 and len(calls) == 3
        assert decisions(client)[0]["reason"] == "breaker_open"
        assert decisions(client)[0]["route"] == "fallback"
        assert len(up.requests) == 4
    # after breaker_open_s the next request probes the engine again
    clock["t"] += 61
    with fake_engine(client, _engine_returning(chat_completion)) as calls2:
        r = _post(client, None, raw=raw)
    assert r.status_code == 200 and len(calls2) == 1
    assert decisions(client)[0]["route"] == "local"
    assert client.app.state.router.breaker_allows("qwen")


# 7 -------------------------------------------------------------------------


def test_no_fallback_rule_refuses_with_529_and_never_calls_anthropic(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "fallback": False}])
    with fake_engine(client, _engine_502):
        r = _post(client, haiku_body())
    msg = _err(r, 529, "overloaded_error")
    assert msg.startswith("router: ")
    assert not up.called
    (d,) = decisions(client)
    assert d["route"] == "refused" and d["reason"] == "status_502"


def test_no_fallback_engine_400_is_400_invalid_request(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "fallback": False}])
    bad = json_upstream({"object": "error", "message": "too long", "code": 400}, status=400)
    with fake_engine(client, _engine_returning(bad)):
        r = _post(client, haiku_body())
    _err(r, 400, "invalid_request_error")
    assert not up.called
    assert client.app.state.router.snapshot(None)["targets"] == []


def test_target_not_loaded_refuses_and_leaves_breaker_alone(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "fallback": False}])
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='pulled' WHERE id='qwen'")
        db.commit()
    with fake_engine(client) as calls:
        r = _post(client, haiku_body())
    _err(r, 529, "overloaded_error")
    assert not calls and not up.called
    assert decisions(client)[0]["reason"] == "target_not_loaded"
    assert client.app.state.router.breaker_allows("qwen")
    assert client.app.state.router._breakers == {}


def test_target_row_missing_falls_back_with_reason(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "target": "gone"}])
    r = _post(client, haiku_body())
    assert r.status_code == 200 and up.called
    assert decisions(client)[0]["reason"] == "target_missing"


# 8 -------------------------------------------------------------------------


def test_first_frame_error_falls_back_to_anthropic_stream(tmp_data_dir, client):
    def handler(_req):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"event: message_start\ndata: {}\n\n",
        )

    _, up = _setup(client, tmp_data_dir, anthropic=handler)
    bad = sse_upstream([{"error": {"message": "engine died"}}])
    with fake_engine(client, _engine_returning(bad)):
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
        ) as r:
            raw = b"".join(r.iter_bytes())
    assert r.status_code == 200 and raw.startswith(b"event: message_start")
    assert up.called
    assert decisions(client)[0]["route"] == "fallback"
    assert decisions(client)[0]["reason"] == "first_frame_error"


def test_stream_failure_after_first_frame_injects_error_and_counts(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    usage = {"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9}
    broken = sse_upstream(
        [
            chunk({"role": "assistant", "content": ""}, usage=usage),
            chunk({"content": "Hel"}, usage=usage),
        ],
        then=RuntimeError("boom"),
    )
    with fake_engine(client, _engine_returning(broken)):
        try:
            with client.stream(
                "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
            ) as r:
                raw = b"".join(r.iter_bytes())
        except Exception:  # noqa: BLE001
            raw = b""
    events = sse_events(raw)
    assert events[0][0] == "message_start"
    assert events[-1][0] == "error" and events[-1][1]["error"]["type"] == "api_error"
    assert "stream interrupted" in events[-1][1]["error"]["message"]
    assert not up.called
    st = client.app.state.router
    assert st._breakers["qwen"].failures == 1
    assert st._breakers["qwen"].last_reason == "stream_interrupted"
    assert decisions(client)[0]["route"] == "local"


def test_good_stream_is_local_with_ttfb(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    with fake_engine(client, _engine_returning(good_stream)):
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
        ) as r:
            events = sse_events(b"".join(r.iter_bytes()))
    assert events[0][0] == "message_start" and events[-1][0] == "message_stop"
    assert events[0][1]["message"]["model"] == "claude-haiku-4-5"
    d = decisions(client)[0]
    assert d["route"] == "local" and d["stream"] is True and d["ttfb_ms"] is not None
    assert not up.called


# 9 -------------------------------------------------------------------------


def test_header_timeout_falls_back_and_releases_slot_and_rank(tmp_data_dir, client):
    def handler(_req):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"x")

    _, up = _setup(client, tmp_data_dir, anthropic=handler, local_header_timeout_s=1)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET data_parallel_size=2 WHERE id='qwen'")
        db.commit()

    async def slow(_req):
        await asyncio.sleep(3)
        return good_stream()

    with fake_engine(client, slow):
        t0 = time.monotonic()
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
        ) as r:
            b"".join(r.iter_bytes())
        assert time.monotonic() - t0 < 2.5
    assert r.status_code == 200
    assert decisions(client)[0]["reason"] == "header_timeout"
    assert decisions(client)[0]["route"] == "fallback"
    assert client.app.state.scheduler._inflight_for_test("qwen") == 0
    assert client.app.state.dp_routing.snapshot("qwen", 2)["totals"]["in_flight"] == 0
    assert client.app.state.router._breakers["qwen"].failures == 1
    assert up.called


# 10 ------------------------------------------------------------------------


def test_token_allow_list_miss_is_403_never_fallback(tmp_data_dir, client):
    ready(client, tmp_data_dir, allowed_models="other")
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    up = fake_anthropic(client)
    with fake_engine(client) as calls:
        r = _post(client, haiku_body())
    assert "token_not_allowed" in _err(r, 403, "permission_error")
    assert not up.called and not calls
    d = decisions(client)[0]
    assert d["route"] == "refused" and d["reason"] == "token_not_allowed"


def test_restricted_token_cannot_pass_through_other_models(tmp_data_dir, client):
    ready(client, tmp_data_dir, allowed_models="qwen")
    enable_router(client, tmp_data_dir, [HAIKU_RULE])
    up = fake_anthropic(client)
    r = _post(client, haiku_body(model="claude-opus-4"))
    assert "token_not_allowed" in _err(r, 403, "permission_error")
    assert not up.called


# 11 ------------------------------------------------------------------------


def _count_body(model="claude-haiku-4-5"):
    return {"model": model, "messages": [{"role": "user", "content": "hello world"}]}


def test_count_tokens_matched_rule_is_local(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    r = _post(client, _count_body(), path="/v1/messages/count_tokens")
    assert r.status_code == 200 and r.json()["input_tokens"] > 0
    assert not up.called
    assert client.app.state.tokenizers.count.await_count == 1


def test_count_tokens_unloaded_target_goes_to_anthropic_even_if_no_fallback(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "fallback": False}])
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='pulled' WHERE id='qwen'")
        db.commit()
    r = _post(client, _count_body(), path="/v1/messages/count_tokens")
    assert r.status_code == 200
    assert up.requests[0]["path"] == "/v1/messages/count_tokens"
    assert decisions(client)[0]["route"] == "fallback"


def test_count_tokens_failure_without_relay_returns_the_local_error(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, relay=False)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='pulled' WHERE id='qwen'")
        db.commit()
    r = _post(client, _count_body(), path="/v1/messages/count_tokens")
    # decision 13: the local error stands; the relay refusal is not what the client sees
    assert r.status_code == 529 and "target_not_loaded" in r.json()["error"]["message"]
    assert "relay_not_allowed" not in r.text
    assert not up.called
    d = decisions(client)[0]
    assert (d["route"], d["reason"], d["status"]) == ("refused", "target_not_loaded", 529)


def test_count_tokens_ignores_an_open_breaker(tmp_data_dir, client):
    # It generates nothing, so a breaker tripped by generation failures must not
    # divert it: the local tokenizer still answers.
    _, up = _setup(client, tmp_data_dir)
    st = client.app.state.router
    for _ in range(3):
        st.record_failure("qwen", "status_502", threshold=3, open_s=60)
    assert not st.breaker_allows("qwen")
    r = _post(client, _count_body(), path="/v1/messages/count_tokens")
    assert r.status_code == 200 and r.json()["input_tokens"] > 0
    assert not up.called and decisions(client)[0]["route"] == "local"


# 12 ------------------------------------------------------------------------


def test_body_cap_is_413_and_nothing_is_called(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, max_body_mb=1)
    raw = json.dumps(haiku_body(model="claude-sonnet-4-5", pad="x" * int(1.1 * 1024 * 1024)))
    with fake_engine(client) as calls:
        r = _post(client, None, raw=raw.encode())
    _err(r, 413, "request_too_large")
    assert not up.called and not calls


# 13 ------------------------------------------------------------------------


def test_catch_all_get_and_streamed_post(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    r = client.get("/v1/models/claude-x?limit=2", headers=CLAUDE_CODE)
    assert r.status_code == 200
    assert up.requests[0]["method"] == "GET"
    assert up.requests[0]["path"] == "/v1/models/claude-x" and up.requests[0]["query"] == "limit=2"
    payload = b"batch-bytes" * 1000
    r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, content=payload)
    assert r.status_code == 200
    assert up.requests[1]["body"] == payload and up.requests[1]["path"] == "/v1/messages/batches"
    assert "x-lmwarden-key" not in up.requests[1]["headers"]
    assert up.requests[1]["headers"]["content-length"] == str(len(payload))


def test_catch_all_requires_a_credential(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    assert client.get("/v1/nothing").status_code == 401
    assert not up.called


def test_catch_all_router_disabled_is_plain_404(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r = client.get("/v1/nothing", headers=CLAUDE_CODE)
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}


def test_warden_owned_paths_are_not_relayed(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    # /v1/models is the warden's own listing, not Anthropic's
    r = client.get("/v1/models", headers=CLAUDE_CODE)
    assert r.status_code == 200 and not up.called
    # wrong method on a warden route stays a 405, never a relay
    r = client.post("/v1/models", headers=CLAUDE_CODE)
    assert r.status_code == 405 and r.json() == {"detail": "Method Not Allowed"}
    assert "GET" in r.headers["allow"] and not up.called


# 14 ------------------------------------------------------------------------


def test_passthrough_unmatched_false_keeps_todays_404(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir, passthrough_unmatched=False)
    r = _post(client, haiku_body(model="claude-sonnet-4-5"))
    _err(r, 404, "not_found_error")
    assert not up.called


# 15 ------------------------------------------------------------------------


def test_anthropic_unreachable_is_502_api_error(tmp_data_dir, client):
    def boom(_req):
        raise httpx.ConnectError("no route")

    _, up = _setup(client, tmp_data_dir, anthropic=boom)
    r = _post(client, haiku_body(model="claude-sonnet-4-5"))
    assert _err(r, 502, "api_error").startswith("router: ")
    assert decisions(client)[0]["route"] == "error"


# 16/17 ---------------------------------------------------------------------


def test_stats_and_decisions_after_a_mix(tmp_data_dir, client):
    ids, up = _setup(client, tmp_data_dir)
    with fake_engine(client, _engine_returning(chat_completion)):
        _post(client, haiku_body())
    _post(client, haiku_body(model="claude-sonnet-4-5"))
    snap = client.app.state.router.snapshot(None)
    assert snap["totals"]["local"] == 1 and snap["totals"]["passthrough"] == 1
    ds = decisions(client)
    assert [d["route"] for d in ds] == ["passthrough", "local"]
    assert ds[1]["rule_id"] == ids[0]


def test_passthrough_does_not_touch_godmode_or_content_log(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    with (
        patch("app.proxy.routes._gm_publish") as gm,
        patch("app.proxy.routes.content_log.should_log") as cl,
    ):
        r = _post(client, haiku_body(model="claude-sonnet-4-5"))
        assert r.status_code == 200
        assert gm.call_count == 0 and cl.call_count == 0
        with fake_engine(client, _engine_returning(chat_completion)):
            _post(client, haiku_body())
        assert cl.call_count == 1  # as today for a local request


def test_passthrough_usage_counted_in_router_stats_only(tmp_data_dir, client):
    def handler(_req):
        return httpx.Response(
            200, json={"usage": {"input_tokens": 7, "output_tokens": 11}, "type": "message"}
        )

    _, up = _setup(client, tmp_data_dir, anthropic=handler)
    _post(client, haiku_body(model="claude-sonnet-4-5"))
    pt = client.app.state.router.snapshot(None)["passthrough"]
    assert (pt["input_tokens"], pt["output_tokens"]) == (7, 11)


# resp.background carried (the #286 unstarted-stream guard) -----------------


def test_local_stream_wrapper_carries_background(tmp_data_dir, client):
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask

    from app.proxy import routes_messages as rm

    ready(client, tmp_data_dir)
    marker = BackgroundTask(lambda: None)
    inner = StreamingResponse(good_stream_iter(), background=marker)

    async def fake_forward(*a, **k):
        return inner

    body = rm.MessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 5,
            "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        }
    )

    async def run():
        class _Req:  # only what _serve_local touches before _forward
            _body = b""
            _receive = None

        with patch.object(rm, "_forward", fake_forward):
            return await rm._serve_local(
                _Req(), None, body, {}, (None, "h", 1, None), response_model_name="qwen"
            )

    resp = client.portal.call(run)
    assert resp.background is marker


def good_stream_iter():
    async def gen():
        yield b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'

    return gen()


# review #1: half-open admits exactly one probe -----------------------------------


def _half_open(client) -> None:
    """Breaker for 'qwen' open and its period just over."""
    st = client.app.state.router
    for _ in range(3):
        st.record_failure("qwen", "status_502", threshold=3, open_s=60)
    st._breakers["qwen"].open_until = time.time() - 1


def test_half_open_lets_one_probe_through_concurrently(tmp_data_dir, client):
    import threading

    _, up = _setup(client, tmp_data_dir)
    _half_open(client)
    raw = json.dumps(haiku_body()).encode()

    async def slow(_req):
        await asyncio.sleep(0.6)
        return chat_completion()

    results: list[int] = []

    def go():
        results.append(_post(client, None, raw=raw).status_code)

    with fake_engine(client, slow) as calls:
        threads = [threading.Thread(target=go) for _ in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.1)
        for t in threads:
            t.join()
    assert results == [200, 200, 200]
    assert len(calls) == 1  # one probe; the others never touched the engine
    routes = sorted((d["route"], d["reason"]) for d in decisions(client))
    assert routes == [("fallback", "breaker_open")] * 2 + [("local", None)]
    assert len(up.requests) == 2
    assert client.app.state.router.breaker_allows("qwen")  # probe succeeded: closed


def test_probe_ending_in_a_local_4xx_releases_the_slot(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    _half_open(client)
    raw = json.dumps(haiku_body()).encode()
    with fake_engine(client, _engine_returning(json_upstream({"error": {"message": "no"}}, 404))):
        r = _post(client, None, raw=raw)
    assert r.status_code == 404
    # neither success nor failure: still half-open, but the next request is the probe
    assert client.app.state.router.breaker_allows("qwen")
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        r = _post(client, None, raw=raw)
    assert r.status_code == 200 and len(calls) == 1
    assert decisions(client)[0]["route"] == "local"


# max_tokens clamp to the local model's remaining context --------------------


def _set_ctx(tmp_data_dir, value):
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET max_model_len=? WHERE id='qwen'", (value,))
        db.commit()


def _long_body(chars=3000, **extra):
    body = haiku_body(max_tokens=32000, **extra)
    body["messages"] = [{"role": "user", "content": "x" * chars}]
    return body


def _clamp_margin(prompt_tokens):
    from app.proxy.messages_clamp import clamp_margin

    return clamp_margin(prompt_tokens, estimated=False)


def test_max_tokens_clamped_to_remaining_context(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    _set_ctx(tmp_data_dir, 4096)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        r = _post(client, _long_body())
    assert r.status_code == 200, r.text
    assert not up.called
    sent = json.loads(calls[0].content)["max_tokens"]
    assert 0 < sent <= 4096 - 3000 - _clamp_margin(3000)
    assert sent > 400  # not absurdly conservative
    (d,) = decisions(client)
    assert d["route"] == "local"


def test_max_tokens_clamped_when_streaming(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    _set_ctx(tmp_data_dir, 4096)
    with fake_engine(client, _engine_returning(good_stream)) as calls:
        r = _post(client, _long_body(stream=True))
    assert r.status_code == 200, r.text
    assert not up.called
    assert json.loads(calls[0].content)["max_tokens"] <= 4096 - 3000 - _clamp_margin(3000)


def test_small_request_is_not_clamped(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _set_ctx(tmp_data_dir, 4096)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        _post(client, haiku_body(max_tokens=500))
    assert json.loads(calls[0].content)["max_tokens"] == 500


def test_unknown_max_model_len_leaves_max_tokens(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _set_ctx(tmp_data_dir, None)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        _post(client, _long_body())
    assert json.loads(calls[0].content)["max_tokens"] == 32000


def test_prompt_over_context_is_not_clamped_and_falls_back(tmp_data_dir, client):
    _, up = _setup(client, tmp_data_dir)
    _set_ctx(tmp_data_dir, 4096)

    async def overflow(_req):
        return json_upstream({"error": {"message": "maximum context length"}}, status=400)

    with fake_engine(client, overflow) as calls:
        r = _post(client, _long_body(chars=5000))
    assert json.loads(calls[0].content)["max_tokens"] == 32000
    assert up.called and r.status_code == 200
    (d,) = decisions(client)
    assert d["route"] == "fallback" and d["reason"] == "status_400"


def test_clamp_wins_over_min_max_tokens_floor(tmp_data_dir, client):
    _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "min_max_tokens": 8000}])
    _set_ctx(tmp_data_dir, 4096)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        _post(client, _long_body(chars=3000) | {"max_tokens": 10})
    sent = json.loads(calls[0].content)["max_tokens"]
    assert 10 < sent <= 4096 - 3000 - _clamp_margin(3000)


def test_floor_still_applies_when_it_fits(tmp_data_dir, client):
    _setup(client, tmp_data_dir, rules=[{**HAIKU_RULE, "min_max_tokens": 1024}])
    _set_ctx(tmp_data_dir, 4096)
    with fake_engine(client, _engine_returning(chat_completion)) as calls:
        _post(client, haiku_body(max_tokens=32))
    assert json.loads(calls[0].content)["max_tokens"] == 1024


def test_count_tokens_unaffected_by_clamp(tmp_data_dir, client):
    _setup(client, tmp_data_dir)
    _set_ctx(tmp_data_dir, 4096)
    body = _long_body()
    body.pop("max_tokens")
    with fake_engine(client) as calls:
        r = _post(client, body, path="/v1/messages/count_tokens")
    assert r.status_code == 200 and r.json() == {"input_tokens": 3000}
    assert not calls


def test_cold_tokenizer_load_does_not_burn_the_header_timeout(tmp_data_dir, client):
    """The clamp's token count (a cold tokenizer can take seconds on the first
    request) runs outside the local header-timeout window."""
    _, up = _setup(client, tmp_data_dir, local_header_timeout_s=1)
    _set_ctx(tmp_data_dir, 4096)

    cold = {"first": True}

    async def slow_count(repo, text, *, fallback_repo=None):
        if cold.pop("first", False):  # only the first call loads the tokenizer
            await asyncio.sleep(1.6)
        return len(text)

    client.app.state.tokenizers.count = slow_count
    with fake_engine(client, _engine_returning(good_stream)):
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=_long_body(stream=True)
        ) as r:
            b"".join(r.iter_bytes())
    assert r.status_code == 200
    (d,) = decisions(client)
    assert d["route"] == "local", d
    assert not up.called
