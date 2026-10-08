"""Cache-affine replica routing through the real forward path (#286 task 3)."""

import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from starlette.requests import Request

from app.db.repos.tokens import TokenRow, hash_token
from app.proxy.dp_affinity import RANK_HEADER, affinity_key
from app.proxy.routes import _forward

PLAINTEXT = "vw_validtoken1234567890abcdef12345"
AUTH = {"Authorization": f"Bearer {PLAINTEXT}"}
OK_BODY = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _seed(db_path, *, dp=7, affinity=1, threshold=None, extra_args=("--max-num-seqs", "32")):
    gpus = list(range(dp))
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": gpus}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, data_parallel_size, dp_affinity_enabled, dp_spill_threshold, "
            "dtype, max_model_len, gpu_memory_utilization, trust_remote_code, extra_args, "
            "status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,?,?,?,'auto',4096,0.9,0,?,"
            "'loaded',0,NULL,NULL)",
            (json.dumps(gpus), dp, affinity, threshold, json.dumps(list(extra_args))),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", PLAINTEXT[:8], hash_token(PLAINTEXT), "inference"),
        )
        db.commit()


def _ready(client, tmp_data_dir, **kw):
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db", **kw)
    client.app.state.supervisor._ports["qwen"] = 19099
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text))
    client.app.state.tokenizers = cache


def _update_model(client, tmp_data_dir, **values):
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    async def go():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            await ModelRepo(db).update_fields("qwen", values)

    client.portal.call(go)


def _resp(body=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body or OK_BODY).encode())
    resp.aclose = AsyncMock()
    return resp


def _sse_resp(chunks, then=None):
    async def aiter():
        for c in chunks:
            yield c
        if then is not None:
            raise then

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _sender(resp):
    seen: list[httpx.Request] = []

    async def _send(request, *, stream=False, **kw):
        seen.append(request)
        if isinstance(resp, BaseException):
            raise resp
        return resp

    return AsyncMock(side_effect=_send), seen


def _post(client, body=None, headers=None, resp=None):
    send, seen = _sender(resp or _resp())
    body = body or {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/chat/completions", headers={**AUTH, **(headers or {})}, json=body)
    return r, seen


def _snap(client, dp=7):
    return client.app.state.dp_routing.snapshot("qwen", dp)


def _msgs(user):
    return {"model": "qwen", "user": user, "messages": [{"role": "user", "content": "hi"}]}


def test_same_user_stays_on_the_rank_it_was_placed_on(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _, seen1 = _post(client, _msgs("alice-user-1"))
    _, seen2 = _post(client, _msgs("alice-user-1"))
    want = seen1[0].headers[RANK_HEADER]
    assert seen2[0].headers[RANK_HEADER] == want
    totals = _snap(client)["totals"]
    assert totals["placed"] == 1 and totals["sticky"] == 1


def _prime(client, user="alice-user-1"):
    """First sight of ``user`` (placed on an idle rank); returns that rank."""
    _, seen = _post(client, _msgs(user))
    return int(seen[0].headers[RANK_HEADER])


def test_different_users_land_on_different_ranks(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    users = [f"user-session-{i}" for i in range(20)]
    ranks = set()
    for u in users:
        _, seen = _post(client, _msgs(u))
        ranks.add(seen[0].headers[RANK_HEADER])
    assert len(ranks) > 1


def test_client_supplied_rank_is_forwarded_untouched_any_case(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _, seen = _post(client, _msgs("alice-user-1"), headers={"x-DATA-parallel-RANK": "5"})
    assert seen[0].headers.get_list(RANK_HEADER) == ["5"]
    snap = _snap(client)
    assert snap["totals"]["client_pinned"] == 1
    assert snap["ranks"][5]["client_pinned"] == 1
    assert snap["totals"]["in_flight"] == 0


def test_client_supplied_invalid_rank_is_still_not_overridden(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _, seen = _post(client, _msgs("alice-user-1"), headers={RANK_HEADER: "99"})
    assert seen[0].headers.get_list(RANK_HEADER) == ["99"]
    assert _snap(client)["totals"]["sticky"] == 0


def test_spill_to_least_loaded_rank(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    state = client.app.state.dp_routing
    home = _prime(client)
    for r in range(7):
        for _ in range(8 if r == home else 3):
            state.route("qwen", dp=7, key=None, threshold=99, pinned=r, affinity_enabled=True)
    # threshold = 32 // 4 = 8: home is at it. Least loaded = lowest idx among the 3s.
    want = min(r for r in range(7) if r != home)
    _, seen = _post(client, _msgs("alice-user-1"))
    assert seen[0].headers[RANK_HEADER] == str(want)
    assert _snap(client)["totals"]["spilled"] == 1


def test_explicit_spill_threshold_setting_applies_without_reload(tmp_data_dir, client):
    _ready(client, tmp_data_dir, threshold=2)
    state = client.app.state.dp_routing
    home = _prime(client)
    for _ in range(2):
        state.route("qwen", dp=7, key=None, threshold=99, pinned=home, affinity_enabled=True)
    _, seen = _post(client, _msgs("alice-user-1"))
    assert seen[0].headers[RANK_HEADER] != str(home)
    # flip the setting through ModelRepo: the very next request sees it (the
    # write drops the model cache, #293)
    _update_model(client, tmp_data_dir, dp_spill_threshold=50)
    _, seen = _post(client, _msgs("alice-user-1"))
    assert seen[0].headers[RANK_HEADER] == str(home)


def test_affinity_disabled_no_header_and_flip_without_reload(tmp_data_dir, client):
    _ready(client, tmp_data_dir, affinity=0)
    _, seen = _post(client, _msgs("alice-user-1"))
    assert RANK_HEADER not in seen[0].headers
    assert _snap(client)["totals"]["unrouted"] == 1
    _update_model(client, tmp_data_dir, dp_affinity_enabled=1)
    _, seen = _post(client, _msgs("alice-user-1"))
    assert RANK_HEADER in seen[0].headers


def test_affinity_disabled_still_forwards_client_header(tmp_data_dir, client):
    _ready(client, tmp_data_dir, affinity=0)
    _, seen = _post(client, _msgs("a"), headers={RANK_HEADER: "2"})
    assert seen[0].headers.get_list(RANK_HEADER) == ["2"]


def test_dp1_model_has_no_header_and_never_touches_state(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _, seen = _post(client, _msgs("alice-user-1"))
    assert RANK_HEADER not in seen[0].headers
    snap = _snap(client, dp=1)
    assert snap["since"] is None
    assert snap["totals"]["unrouted"] == 0


def test_inflight_released_after_nonstream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, _ = _post(client, _msgs("alice-user-1"))
    assert r.status_code == 200
    snap = _snap(client)
    assert snap["totals"]["placed"] == 1 and snap["totals"]["in_flight"] == 0


def test_inflight_released_after_stream_drained(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    send, seen = _sender(
        _sse_resp([b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n', b"data: [DONE]\n\n"])
    )
    body = {**_msgs("alice-user-1"), "stream": True}
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream("POST", "/v1/chat/completions", headers=AUTH, json=body) as r:
            b"".join(r.iter_bytes())
    assert RANK_HEADER in seen[0].headers
    snap = _snap(client)
    assert snap["totals"]["placed"] == 1 and snap["totals"]["in_flight"] == 0


def test_inflight_released_after_upstream_connect_error(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, seen = _post(client, _msgs("alice-user-1"), resp=httpx.ConnectError("refused"))
    assert r.status_code == 502
    snap = _snap(client)
    assert snap["totals"]["placed"] == 1 and snap["totals"]["in_flight"] == 0


def test_inflight_released_after_upstream_error_midstream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    send, _ = _sender(
        _sse_resp([b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'], then=RuntimeError("boom"))
    )
    body = {**_msgs("alice-user-1"), "stream": True}
    with patch("httpx.AsyncClient.send", new=send):
        try:
            with client.stream("POST", "/v1/chat/completions", headers=AUTH, json=body) as r:
                b"".join(r.iter_bytes())
        except Exception:  # noqa: BLE001 — the transport surfaces the app error
            pass
    snap = _snap(client)
    assert snap["totals"]["placed"] == 1 and snap["totals"]["in_flight"] == 0


def _direct_forward(client, body, resp):
    """Run ``_forward`` on the app's own loop and hand back the response object."""
    raw = json.dumps(body).encode()
    app = client.app

    async def go():
        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "app": app,
            "query_string": b"",
            "client": ("127.0.0.1", 1),
        }
        request = Request(scope, receive)
        from app.db.database import open_db
        from app.db.repos.models import ModelRepo

        async with open_db(app.state.settings.db_path) as db:
            model = await ModelRepo(db).get_by_served_name("qwen")
        token = TokenRow(
            id="tok1",
            name="test",
            prefix="vw_valid",
            scope="inference",
            allowed_models=None,
            rate_limit_rpm=None,
            rate_limit_tpm=None,
            revoked_at=None,
            last_used_at=None,
            created_at="2026-01-01",
            expires_at=None,
            rotated_at=None,
            rotated_from=None,
        )
        send, _ = _sender(resp)
        with patch("httpx.AsyncClient.send", new=send):
            return await _forward(request, model, "127.0.0.1", 19099, "/v1/chat/completions", token)

    return go


def test_inflight_released_on_client_disconnect_midstream(tmp_data_dir, client):
    """The consumer stops after the first chunk and closes the body iterator
    (what Starlette does on http.disconnect)."""
    _ready(client, tmp_data_dir)
    chunks = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'] * 3
    body = {**_msgs("alice-user-1"), "stream": True}
    go = _direct_forward(client, body, _sse_resp(chunks))

    async def run():
        resp = await go()
        it = resp.body_iterator
        await it.__anext__()
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 1
        await it.aclose()

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0


def test_inflight_released_on_cancellation_midstream(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    gate = asyncio.Event  # noqa: F841 — placeholder for symmetry; upstream just hangs
    started = {}

    async def hang():
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        await asyncio.sleep(3600)

    resp_up = MagicMock()
    resp_up.status_code = 200
    resp_up.headers = {"content-type": "text/event-stream"}
    resp_up.aiter_bytes = hang
    resp_up.aclose = AsyncMock()
    body = {**_msgs("alice-user-1"), "stream": True}
    go = _direct_forward(client, body, resp_up)

    async def run():
        resp = await go()

        async def consume():
            async for _ in resp.body_iterator:
                started["first"] = True

        task = asyncio.create_task(consume())
        for _ in range(200):
            if started.get("first"):
                break
            await asyncio.sleep(0.01)
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0


def test_inflight_released_exactly_once_when_deregister_runs_twice(tmp_data_dir, client):
    """Two requests in flight on one rank: finishing one must leave the other counted."""
    _ready(client, tmp_data_dir)
    body = {**_msgs("alice-user-1"), "stream": True}
    chunks = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n']
    go1 = _direct_forward(client, body, _sse_resp(chunks))
    go2 = _direct_forward(client, body, _sse_resp(chunks))

    async def run():
        r1 = await go1()
        r2 = await go2()
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 2
        async for _ in r1.body_iterator:
            pass
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 1
        async for _ in r2.body_iterator:
            pass

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0


def test_routing_failure_is_fail_open(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    client.app.state.dp_routing = MagicMock()
    client.app.state.dp_routing.route.side_effect = RuntimeError("boom")
    r, seen = _post(client, _msgs("alice-user-1"))
    assert r.status_code == 200
    assert RANK_HEADER not in seen[0].headers
    client.app.state.dp_routing.release.assert_not_called()


def test_missing_dp_state_is_fail_open(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    del client.app.state.dp_routing
    r, seen = _post(client, _msgs("alice-user-1"))
    assert r.status_code == 200 and RANK_HEADER not in seen[0].headers


def test_session_header_used_when_no_user(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    _, seen = _post(client, body, headers={"X-Session-Id": "sess-0001"})
    first = seen[0].headers[RANK_HEADER]
    # same session id, a different prompt: still the same session, same replica
    other = {"model": "qwen", "messages": [{"role": "user", "content": "something else"}]}
    _, seen2 = _post(client, other, headers={"X-Session-Id": "sess-0001"})
    assert seen2[0].headers[RANK_HEADER] == first
    assert _snap(client)["totals"]["placed"] == 1


def test_prompt_hash_shared_for_same_first_user_message(tmp_data_dir, client):
    _ready(client, tmp_data_dir)

    def body(system, tail="x"):
        return {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "same opener"},
                {"role": "assistant", "content": tail},
            ],
        }

    _, a = _post(client, body("S1"))
    _, b = _post(client, body("S2", tail="y"))
    assert a[0].headers[RANK_HEADER] == b[0].headers[RANK_HEADER]
    # the system message alone is not the hash input
    k1 = affinity_key(body("S1"), {}, "tok1")
    k2 = affinity_key(
        {"messages": [{"role": "system", "content": "S1"}, {"role": "user", "content": "other"}]},
        {},
        "tok1",
    )
    assert k1 != k2
    assert k1[0] == affinity_key(body("S2"), {}, "tok1")[0]


def test_admission_cap_multiplies_by_replicas(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    real = client.app.state.scheduler.acquire
    caps = []

    def spy(*a, **kw):
        caps.append(kw.get("cap"))
        return real(*a, **kw)

    client.app.state.scheduler.acquire = spy
    _post(client, _msgs("alice-user-1"))
    assert caps == [32 * 7]


def _engine_inflight(client):
    return client.app.state.scheduler._inflight_for_test("qwen")


def test_unstarted_stream_closed_releases_rank_and_slot(tmp_data_dir, client):
    """Review #4a: the body iterator is closed before the first chunk (client gone
    before the first read). Closing a never-started generator skips its finally,
    so the rank count and the admission slot used to leak."""
    _ready(client, tmp_data_dir)
    chunks = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'] * 2
    go = _direct_forward(client, {**_msgs("alice-user-1"), "stream": True}, _sse_resp(chunks))

    async def run():
        resp = await go()
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 1
        assert _engine_inflight(client) == 1
        await resp.body_iterator.aclose()  # never iterated
        await resp.body_iterator.aclose()  # idempotent: no double release

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0
    assert _engine_inflight(client) == 0


def test_unstarted_stream_background_task_releases(tmp_data_dir, client):
    """Starlette runs ``background`` after the response; if the body was never
    iterated (send of the response start failed), it must still release once."""
    _ready(client, tmp_data_dir)
    go = _direct_forward(
        client, {**_msgs("alice-user-1"), "stream": True}, _sse_resp([b"data: [DONE]\n\n"])
    )

    async def run():
        resp = await go()
        assert resp.background is not None
        await resp.background()
        await resp.background()

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0
    assert _engine_inflight(client) == 0


def test_stream_drained_then_closed_does_not_double_release(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {**_msgs("alice-user-1"), "stream": True}
    chunks = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n']
    go1 = _direct_forward(client, body, _sse_resp(chunks))
    go2 = _direct_forward(client, body, _sse_resp(chunks))

    async def run():
        r1 = await go1()
        r2 = await go2()
        async for _ in r1.body_iterator:
            pass
        await r1.body_iterator.aclose()
        await r1.background()
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 1
        assert _engine_inflight(client) == 1
        await r2.body_iterator.aclose()

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0
    assert _engine_inflight(client) == 0


def test_cancellation_during_registry_register_releases_rank_and_slot(tmp_data_dir, client):
    """Review #4b: ``except Exception`` does not catch CancelledError."""
    _ready(client, tmp_data_dir)
    entered = asyncio.Event()

    calls = []

    async def hang(_req):
        # The first (pre-admission, "queued") registration fails open; the
        # second, after the slot and rank are held, is the one that hangs.
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("registry down")
        entered.set()
        await asyncio.sleep(3600)

    client.app.state.request_registry.register = hang
    go = _direct_forward(client, _msgs("alice-user-1"), _resp())

    async def run():
        task = asyncio.create_task(go())
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    client.portal.call(run)
    assert _snap(client)["totals"]["in_flight"] == 0
    assert _engine_inflight(client) == 0


def test_guard_abandon_is_a_noop_once_the_stream_started():
    """``abandon`` must honour ``_started``: after the first read the generator's
    own ``finally`` owns cleanup, and a second release would double-count."""
    from app.proxy.routes import _UnstartedStreamGuard

    async def run():
        released = []

        async def gen():
            yield b"a"
            yield b"b"

        async def on_abandon():
            released.append(1)

        started = _UnstartedStreamGuard(gen(), on_abandon)
        await started.__anext__()
        await started.abandon()
        assert released == []
        await started.aclose()

        fresh = _UnstartedStreamGuard(gen(), on_abandon)
        await fresh.abandon()
        await fresh.abandon()
        await fresh.aclose()
        assert released == [1]

    asyncio.run(run())


def test_forward_reuses_one_pooled_upstream_client(tmp_data_dir, client):
    """#279 stage 2: two requests to the same engine run share one pooled
    client, held by app.state.upstream_clients, and never close it."""
    _ready(client, tmp_data_dir)
    used: list[httpx.AsyncClient] = []

    async def _send(self, request, *, stream=False, **kw):
        used.append(self)
        return _resp()

    with patch("httpx.AsyncClient.send", new=_send):
        for _ in range(2):
            r = client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
    pool = client.app.state.upstream_clients
    assert len(pool._clients) == 1
    assert len(used) == 2 and used[0] is used[1]
    assert used[0] is next(iter(pool._clients.values()))
    assert not used[0].is_closed
