import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
from fastapi.responses import StreamingResponse

from app.chat2 import routes_turn
from tests.conftest import csrf_header, jwt_login, seed_admin_user

# The LLM title is fire-and-forget on its own task. Every test that patches
# `httpx.AsyncClient.send` stubs it out too: otherwise the background call also
# hits the mock, so `send.call_args` stops being the turn's request and a
# `purpose='title'` ledger row can race into the assertions.
_NO_TITLE = ("app.chat2.routes_turn._llm_title", AsyncMock(return_value=None))


def _auth(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return {**jwt_login(client), **csrf_header(client)}


def _model(db_path, served="qwen", window=8192, tools=1):
    c = sqlite3.connect(db_path)
    c.execute("INSERT INTO models(id, served_model_name, hf_repo, gpu_indices, tensor_parallel_size, "
              "status, max_model_len, supports_tools, created_at, updated_at) "
              "VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
              (served + "-id", served, "org/" + served, "[0]", 1, "loaded", window, tools))
    c.commit()


def _sse(*objs, done=True) -> bytes:
    out = b"".join(f"data: {json.dumps(o)}\n\n".encode() for o in objs)
    return out + (b"data: [DONE]\n\n" if done else b"")


def _fake_stream(body: bytes, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "text/event-stream"}

    async def aiter_raw():
        for i in range(0, len(body), 7):
            # A real upstream suspends between chunks; without a checkpoint
            # here the whole stream would drain in one event-loop step and a
            # mid-stream cancellation could never be delivered.
            await asyncio.sleep(0)
            yield body[i:i + 7]
    resp.aiter_raw = aiter_raw
    resp.aread = AsyncMock(return_value=body)

    # `AsyncMock.await_count` is incremented BEFORE the side effect is awaited,
    # so `assert_awaited()` cannot tell a completed close from one that was
    # cancelled half-way. `closed_cleanly` records only a close that ran to the
    # end -- and the `sleep(0)` is the checkpoint a real socket close has, where
    # a cancelled scope re-delivers CancelledError.
    resp.closed_cleanly = []

    async def _aclose():
        await asyncio.sleep(0)
        resp.closed_cleanly.append(True)

    resp.aclose = AsyncMock(side_effect=_aclose)
    return resp


def _aborting_after(n: int):
    """A `StreamingResponse` stand-in that consumes `n` frames from the
    response's SUBSCRIBER generator and then `aclose()`s it — a client
    disconnect.

    That is the closest thing to a disconnect available under `TestClient`,
    whose transport always runs the app to completion and buffers the whole
    body (`_TestClientTransport.handle_request` only yields `http.disconnect`
    *after* `response_complete`). Since the detached-turns change the body is
    only a subscriber over the live-turn registry: closing it must NOT touch
    the runner, which keeps streaming to completion on its own task. The
    `_BACKGROUND` drain afterwards waits for the runner (and its shielded
    persistence) so the assertions do not race it — in production the runner
    simply outlives the request.
    """

    class _Aborting(StreamingResponse):
        def __init__(self, content, **kwargs):
            async def drive():
                agen = content.__aiter__()
                try:
                    for _ in range(n):
                        yield await agen.__anext__()
                finally:
                    await agen.aclose()
                while routes_turn._BACKGROUND:
                    await asyncio.gather(*list(routes_turn._BACKGROUND), return_exceptions=True)
                await asyncio.sleep(0)  # let the done-callbacks (lock release) run

            super().__init__(drive(), **kwargs)

    return _Aborting


def _cancelling_after(n: int):
    """A `StreamingResponse` stand-in that runs the response's SUBSCRIBER
    generator in a task and cancels it after `n` frames -- the production
    client-disconnect shape.

    The cancel is issued through an `anyio.CancelScope`, which is what Starlette
    actually wraps a response body in and cancels on `http.disconnect`. Since
    the detached-turns change this must only kill the subscriber: the runner
    task keeps streaming to completion and persists the FULL assistant row.

    Afterwards it drains `routes_turn._BACKGROUND` so the runner and its
    shielded persistence finish before TestClient tears down its per-request
    event loop -- in production those tasks simply outlive the request.
    """

    class _Cancelling(StreamingResponse):
        def __init__(self, content, **kwargs):
            async def drive():
                frames: list[bytes] = []
                agen = content.__aiter__()

                async def pump():
                    with anyio.CancelScope() as scope:
                        async for f in agen:
                            frames.append(f)
                            if len(frames) >= n:
                                scope.cancel()

                await asyncio.ensure_future(pump())
                while routes_turn._BACKGROUND:
                    await asyncio.gather(*list(routes_turn._BACKGROUND), return_exceptions=True)
                await asyncio.sleep(0)  # let the done-callbacks (lock release) run
                for f in frames:
                    yield f

            super().__init__(drive(), **kwargs)

    return _Cancelling


def _draining():
    """A `StreamingResponse` stand-in that drains `routes_turn._BACKGROUND`
    once the body is exhausted.

    The counterpart of `_NO_TITLE` for the tests that want the title task to
    actually run: it is fire-and-forget, so without this the assertions race
    the background write (and, under `TestClient`, the task may not be driven
    at all before the next request). Same trick `_cancelling_after` uses -- the
    drain happens on the app's own event loop, so nothing about the turn's
    ordering changes; in production those tasks simply outlive the request.
    """

    class _Draining(StreamingResponse):
        def __init__(self, content, **kwargs):
            async def drive():
                async for f in content:
                    yield f
                while routes_turn._BACKGROUND:
                    await asyncio.gather(*list(routes_turn._BACKGROUND), return_exceptions=True)
                await asyncio.sleep(0)  # let the done-callbacks (lock release) run

            super().__init__(drive(), **kwargs)

    return _Draining


def _events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.split("\n\n") if line.startswith("data: ")]


def test_turn_streams_persists_and_records_ledger(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    # The usage tail arrives as its own chunk with an EMPTY choices list — that
    # is what `stream_options.include_usage` buys us, and the only way a turn
    # gets exact (non-estimated) usage.
    body = _sse(
        {"id": "cmpl-9", "choices": [{"index": 0, "delta": {"reasoning_content": "hm"}, "finish_reason": None}]},
        {"id": "cmpl-9", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"id": "cmpl-9", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "cmpl-9", "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}},
    )
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))) as send, \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-0001", "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    evs = _events(r.text)
    types = [e["type"] for e in evs]
    # spec §3: context state is emitted at the TOP of the turn (the same state
    # the `context_full` 409 is decided from) as well as at the end of it.
    assert types[:3] == ["message-persisted", "message-start", "context"]
    assert types.count("context") == 2
    assert "reasoning-delta" in types and "text-delta" in types and "usage" in types
    assert types[-2] == "context" and types[-1] == "done" and evs[-1]["finishReason"] == "stop"
    assert evs[2] == {"type": "context", "promptTokens": 0, "window": 8192, "full": False}
    usage_ev = next(e for e in evs if e["type"] == "usage")
    assert usage_ev["prompt"] == 11 and usage_ev["completion"] == 2
    assert usage_ev["estimated"] is False
    sent = json.loads(send.call_args.args[0].content)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["tools"][0]["function"]["name"] == "present_options"
    assert sent["messages"][-1] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    c = sqlite3.connect(db_path)
    rows = c.execute("SELECT role, usage_json, finish_reason FROM messages ORDER BY seq").fetchall()
    assert rows[0][0] == "user" and rows[1][0] == "assistant" and rows[1][2] == "stop"
    assert json.loads(rows[1][1])["prompt"] == 11
    assert json.loads(rows[1][1])["estimated"] is False
    # usage_ledger.request_id is globally UNIQUE, so the client-supplied id is
    # stored namespaced by user id.
    (uid,) = c.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
    led = c.execute("SELECT request_id, provider_request_id, outcome, cost_status, purpose, "
                    "estimated FROM usage_ledger WHERE purpose = 'turn'").fetchall()
    assert led == [(f"{uid}:req-0001", "cmpl-9", "ok", "unpriced", "turn", 0)]
    (title,) = c.execute("SELECT title FROM chats").fetchone()
    assert title == "hi"
    assert client.app.state.chat_active_requests.count() == 0


def test_turn_advertises_only_the_chat_s_enabled_tools(tmp_data_dir, client) -> None:
    """The Tools checkbox writes `enabled_tools`; unchecking it must actually
    stop `present_options` reaching the model, even on a tool-capable model."""
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h,
                       json={"model": "qwen", "settings": {"enabled_tools": []}}).json()
    assert chat["settings"]["enabled_tools"] == []
    body = _sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": "stop"}]})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))) as send, \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-notools",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    assert "tools" not in json.loads(send.call_args.args[0].content)


def test_turn_ignores_unknown_enabled_tool_names(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h,
                       json={"model": "qwen",
                             "settings": {"enabled_tools": ["nope", "present_options"]}}).json()
    body = _sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": "stop"}]})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))) as send, \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-unknown",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    tools = json.loads(send.call_args.args[0].content)["tools"]
    assert [t["function"]["name"] for t in tools] == ["present_options"]


def test_turn_idempotent_retry_and_in_flight_409(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "x", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r1 = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                         json={"request_id": "req-dup-01", "user_parts": [{"type": "text", "text": "a"}]})
        r2 = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                         json={"request_id": "req-dup-01", "user_parts": [{"type": "text", "text": "a"}]})
    assert r1.status_code == 200 and r2.status_code == 200
    replay = _events(r2.text)
    assert [e["type"] for e in replay] == ["done"] and replay[0]["finishReason"] == "stop"
    c = sqlite3.connect(db_path)
    assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    client.app.state.chat2_turn_locks.try_acquire(chat["id"])
    r3 = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                     json={"request_id": "req-0002", "user_parts": [{"type": "text", "text": "b"}]})
    assert r3.status_code == 409 and r3.json()["detail"]["code"] == "turn_in_flight"
    client.app.state.chat2_turn_locks.release(chat["id"])


def test_the_idempotent_replay_stream_runs_under_the_stream_guard(
    tmp_data_dir, client
) -> None:
    """M5: the replay of a recorded turn was the one credentialed stream the app
    opened outside ``guard_stream``, so "every stream the app opens under a
    credential is one of them" was false as written.

    It is a single pre-computed frame, so there was nothing to cancel -- but the
    exception is not worth keeping: it costs one line, and the claim is easier
    to trust than to qualify.
    """
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "x", "choices": [{"index": 0, "delta": {"content": "ok"},
                                         "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        first = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                            json={"request_id": "req-replay-guard",
                                  "user_parts": [{"type": "text", "text": "a"}]})
    assert first.status_code == 200, first.text

    registry = client.app.state.stream_registry
    registered: list[str] = []
    real_register = registry.register
    registry.register = lambda key, task: (  # type: ignore[method-assign]
        registered.append(key), real_register(key, task)
    )[1]
    try:
        replay = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                             json={"request_id": "req-replay-guard",
                                   "user_parts": [{"type": "text", "text": "a"}]})
    finally:
        registry.register = real_register  # type: ignore[method-assign]
    assert [e["type"] for e in _events(replay.text)] == ["done"]
    assert registered == ["session:admin"], registered
    # And it let go of the registration when it finished.
    assert registry.count("session:admin") == 0


def test_turn_guard_finish_and_upstream_429_are_visible_errors(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "g", "choices": [{"index": 0, "delta": {"content": "loop"},
                                         "finish_reason": "runaway_think"}]})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-guard-01", "user_parts": [{"type": "text", "text": "x"}]})
    evs = _events(r.text)
    assert evs[-1]["finishReason"] == "guard" and any(e["type"] == "error" and e["code"] == "guard" for e in evs)
    c = sqlite3.connect(db_path)
    assert c.execute("SELECT outcome, estimated FROM usage_ledger").fetchall() == [("guard", 1)]
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(b'{"detail":"rate limit"}', 429))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-429-01", "user_parts": [{"type": "text", "text": "y"}]})
    evs = _events(r.text)
    assert evs[-1]["type"] == "error" and evs[-1]["code"] == "rate_limited"
    assert c.execute("SELECT error_json IS NOT NULL FROM messages WHERE role='assistant' "
                     "ORDER BY seq DESC LIMIT 1").fetchone()[0] == 1
    # Replaying a FAILED turn must be truthful: `done` carries the recorded
    # outcome, not a fabricated `stop`, and points at the failed assistant row.
    replay = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                         json={"request_id": "req-429-01", "user_parts": [{"type": "text", "text": "y"}]})
    evs = _events(replay.text)
    assert [e["type"] for e in evs] == ["done"] and evs[0]["finishReason"] == "error"
    (failed_id,) = c.execute("SELECT id FROM messages WHERE role='assistant' "
                             "ORDER BY seq DESC LIMIT 1").fetchone()
    assert evs[0]["messageId"] == failed_id


def test_turn_context_full_409_and_tools_unsupported_422(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path, served="tiny", window=100, tools=0)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "tiny"}).json()
    c = sqlite3.connect(db_path)
    c.execute("INSERT INTO messages(id,chat_id,seq,role,parts_json,settings_snapshot_json,usage_json,"
              "created_at) VALUES ('a1',?,1,'assistant','[]','{}','{\"prompt\":90,\"completion\":5}',"
              "'2026-01-01T00:00:00.000Z')", (chat["id"],))
    c.commit()
    r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                    json={"request_id": "req-full", "user_parts": [{"type": "text", "text": "z"}]})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "context_full"
    c.execute("DELETE FROM messages")
    # widen the window too: a 100-token window is *always* context-full once the
    # 1024-token reply reserve is added, so tools_unsupported would be unreachable.
    c.execute("UPDATE models SET max_model_len = 8192")
    c.commit()
    r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                    json={"request_id": "req-tools", "tool_results": [{"call_id": "c", "result": {}}]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "tools_unsupported"
    r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                    json={"request_id": "req-images", "user_parts": [{"type": "text", "text": "z"}],
                          "attachment_ids": [f"a{i}" for i in range(9)]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "too_many_images"


def test_turn_rejects_malformed_part_without_leaking_the_counter(tmp_data_dir, client) -> None:
    """A bad `user_parts` entry must 422 at the schema boundary — before the
    upstream socket and the active-request counter are ever taken."""
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                    json={"request_id": "req-bad-01", "user_parts": [{"type": "text", "text": 123}]})
    assert r.status_code == 422, r.text
    assert client.app.state.chat_active_requests.count() == 0
    c = sqlite3.connect(tmp_data_dir / "vllm-warden.db")
    assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    # the lock must be free again for the next turn
    assert client.app.state.chat2_turn_locks.try_acquire(chat["id"]) is True
    client.app.state.chat2_turn_locks.release(chat["id"])


def test_turn_missing_attachment_persists_nothing(tmp_data_dir, client) -> None:
    """Step 4 is atomic: a rejected attachment id must not leave an orphan
    user row committed in history."""
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                    json={"request_id": "req-gone-01", "user_parts": [{"type": "text", "text": "q"}],
                          "attachment_ids": ["does-not-exist"]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "attachment_missing"
    c = sqlite3.connect(tmp_data_dir / "vllm-warden.db")
    assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 0
    assert client.app.state.chat_active_requests.count() == 0


def test_turn_request_id_is_scoped_to_the_user(tmp_data_dir, client) -> None:
    """`request_id` is client-supplied and unique only per user: one user's id
    must never replay (and thereby swallow) another user's turn."""
    db_path = tmp_data_dir / "vllm-warden.db"
    h = _auth(tmp_data_dir, client)
    _model(db_path)
    seed_admin_user(db_path, username="bob")
    hb = {**jwt_login(client, username="bob"), **csrf_header(client)}
    chat_a = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    chat_b = client.post("/api/chat2/chats", headers=hb, json={"model": "qwen"}).json()
    body = _sse({"id": "s", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        ra = client.post(f"/api/chat2/chats/{chat_a['id']}/turns", headers=h,
                         json={"request_id": "shared-id-01",
                               "user_parts": [{"type": "text", "text": "a"}]})
        rb = client.post(f"/api/chat2/chats/{chat_b['id']}/turns", headers=hb,
                         json={"request_id": "shared-id-01",
                               "user_parts": [{"type": "text", "text": "b"}]})
    assert ra.status_code == 200 and rb.status_code == 200
    types_b = [e["type"] for e in _events(rb.text)]
    assert types_b[0] == "message-persisted" and "text-delta" in types_b
    assert types_b[-1] == "done", "B's turn was swallowed as a replay of A's"
    c = sqlite3.connect(db_path)
    rows = c.execute("SELECT user_id, request_id FROM usage_ledger "
                     "WHERE request_id LIKE '%:shared-id-01' ORDER BY user_id").fetchall()
    assert len(rows) == 2 and rows[0][0] != rows[1][0]
    assert rows[0][1] != rows[1][1]
    assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4


def test_turn_client_disconnect_lets_the_runner_finish(tmp_data_dir, client) -> None:
    """Closing the response's subscriber generator mid-stream (a disconnect)
    must NOT abort the turn: the detached runner streams on, persists the FULL
    assistant row (finish_reason from upstream, here `stop`) plus its ledger
    entry, exits the counter and releases the lock."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse(
        {"id": "ab", "choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": None}]},
        {"id": "ab", "choices": [{"index": 0, "delta": {"content": "tial"}, "finish_reason": "stop"}]},
    )
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_aborting_after(4)):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-abort-01",
                              "user_parts": [{"type": "text", "text": "x"}]})
    assert r.status_code == 200, r.text
    types = [e["type"] for e in _events(r.text)]
    # the disconnected subscriber saw only the first four frames...
    assert types == ["message-persisted", "message-start", "context", "text-delta"]
    # ...but the runner finished the turn: the FULL text, upstream's own
    # finish_reason, and an `ok` ledger row — not an aborted tail.
    c = sqlite3.connect(db_path)
    role, parts, finish = c.execute(
        "SELECT role, parts_json, finish_reason FROM messages ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert role == "assistant" and finish == "stop"
    assert json.loads(parts)[0] == {"type": "text", "text": "partial"}
    assert c.execute("SELECT outcome FROM usage_ledger WHERE purpose = 'turn'"
                     ).fetchall() == [("ok",)]
    assert client.app.state.chat_active_requests.count() == 0
    assert client.app.state.chat2_turn_locks.try_acquire(chat["id"]) is True
    client.app.state.chat2_turn_locks.release(chat["id"])
    # the registry still holds the finished turn for its replay grace window,
    # but the detail endpoint already reports no live turn
    detail = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert detail["live_turn"] is None


def test_turn_cancelled_subscriber_closes_upstream_and_persists_full_row(tmp_data_dir, client) -> None:
    """The production disconnect shape: Starlette cancels the response body
    (an anyio scope that re-delivers CancelledError at every checkpoint).
    Only the subscriber dies — the runner still closes the upstream response
    and the loopback client (the proxy's slot is released on that close),
    exits the counter, persists the FULL row + ledger entry, and releases the
    lock."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse(
        {"id": "cx", "choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": None}]},
        {"id": "cx", "choices": [{"index": 0, "delta": {"content": "tial"}, "finish_reason": "stop"}]},
    )
    fake = _fake_stream(body)
    closed: list[httpx.AsyncClient] = []
    real_aclose = httpx.AsyncClient.aclose

    async def _spy_aclose(self):
        closed.append(self)
        await real_aclose(self)

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)), \
            patch("httpx.AsyncClient.aclose", new=_spy_aclose), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_cancelling_after(4)):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-cancel-01",
                              "user_parts": [{"type": "text", "text": "x"}]})
    assert r.status_code == 200, r.text
    types = [e["type"] for e in _events(r.text)]
    # the cancelled subscriber saw only the first four frames
    assert types == ["message-persisted", "message-start", "context", "text-delta"]
    # the two closes the proxy's slot depends on RAN TO COMPLETION on the
    # runner's own task, untouched by the subscriber's cancellation
    fake.aclose.assert_awaited()
    assert fake.closed_cleanly == [True], "upstream response was not closed"
    assert closed and closed[0].is_closed is True, "loopback client was not closed"
    c = sqlite3.connect(db_path)
    role, parts, finish = c.execute(
        "SELECT role, parts_json, finish_reason FROM messages ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert role == "assistant" and finish == "stop"
    assert json.loads(parts)[0] == {"type": "text", "text": "partial"}
    assert c.execute("SELECT outcome FROM usage_ledger WHERE purpose = 'turn'"
                     ).fetchall() == [("ok",)]
    assert client.app.state.chat_active_requests.count() == 0
    assert client.app.state.chat2_turn_locks.try_acquire(chat["id"]) is True
    client.app.state.chat2_turn_locks.release(chat["id"])


def test_auto_title_is_reassessed_every_fourth_user_turn(tmp_data_dir, client) -> None:
    """The auto title is generated on turn 1 and re-checked on turns 4, 8, ...

    A one-shot title is wrong for a chat that wanders: the first exchange names
    the whole thread forever. Re-running the same chat model periodically fixes
    that, and the displaced title is kept in `title_prev` so the rename is
    undoable. `generate_llm_title` is stubbed rather than `_llm_title` itself,
    so the firing rule AND the compare-then-write are both exercised for real.
    """
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "t", "choices": [{"index": 0, "delta": {"content": "ok"},
                                         "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    gen_title = AsyncMock(side_effect=["First topic", "Second topic"])
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch("app.chat2.routes_turn.generate_llm_title", new=gen_title), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_draining()):
        for i in range(4):
            r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                            json={"request_id": f"req-title-{i}",
                                  "user_parts": [{"type": "text", "text": f"m{i}"}]})
            assert r.status_code == 200, r.text
    # turns 1 and 4 only — 2 and 3 must not spend a title call
    assert gen_title.await_count == 2, gen_title.await_args_list
    kwargs = gen_title.await_args.kwargs
    assert kwargs["model"] == "qwen", "the reassessment must use the chat's own model"
    # the last six user/assistant rows — the opening turn has aged out, which
    # is the whole point of reassessing rather than re-titling from turn one
    assert kwargs["context_text"] == (
        "User: m1\nAssistant: ok\nUser: m2\nAssistant: ok\nUser: m3\nAssistant: ok")
    c = sqlite3.connect(db_path)
    assert c.execute("SELECT title, title_prev FROM chats").fetchone() == (
        "Second topic", "First topic")


def test_auto_title_reassessment_skips_an_unchanged_title(tmp_data_dir, client) -> None:
    """An unchanged verdict must not churn `title_prev` into a self-reference.

    The comparison is normalised (case + whitespace), so the model re-emitting
    "first  TOPIC" is recognised as the title already on the chat.
    """
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "t", "choices": [{"index": 0, "delta": {"content": "ok"},
                                         "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    gen_title = AsyncMock(side_effect=["First topic", "first  TOPIC"])
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch("app.chat2.routes_turn.generate_llm_title", new=gen_title), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_draining()):
        for i in range(4):
            client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": f"req-same-{i}",
                              "user_parts": [{"type": "text", "text": f"m{i}"}]})
    c = sqlite3.connect(db_path)
    # nothing was displaced: turn 1 replaced the first-line fallback (which is
    # not an undo target) and turn 4's verdict was the same title again
    assert c.execute("SELECT title, title_prev FROM chats").fetchone() == ("First topic", None)


def test_user_renamed_chats_are_never_reassessed(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    client.patch(f"/api/chat2/chats/{chat['id']}", headers=h, json={"title": "Mine"})
    body = _sse({"id": "t", "choices": [{"index": 0, "delta": {"content": "ok"},
                                         "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    gen_title = AsyncMock(return_value="Something else")
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch("app.chat2.routes_turn.generate_llm_title", new=gen_title), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_draining()):
        for i in range(4):
            client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": f"req-user-{i}",
                              "user_parts": [{"type": "text", "text": f"m{i}"}]})
    assert gen_title.await_count == 0
    c = sqlite3.connect(db_path)
    assert c.execute("SELECT title, title_prev FROM chats").fetchone() == ("Mine", None)


def test_enable_thinking_false_reaches_vllm_as_a_chat_template_kwarg(tmp_data_dir, client) -> None:
    """Turning thinking off is a chat-template decision, not a sampling one.

    vLLM exposes it through `chat_template_kwargs`; there is no OpenAI field for
    it. Only the OFF case is sent -- omitting the kwarg leaves the template's
    own default in place, which is what "on" has to mean for the templates that
    have never heard of the flag.
    """
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h,
                       json={"model": "qwen", "settings": {"enable_thinking": False}}).json()
    assert chat["settings"]["enable_thinking"] is False
    body = _sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": "stop"}]})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))) as send, \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-think-off",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    sent = json.loads(send.call_args.args[0].content)
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_enable_thinking_default_sends_no_chat_template_kwargs(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    assert chat["settings"]["enable_thinking"] is True
    body = _sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": "stop"}]})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))) as send, \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-think-on",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    assert "chat_template_kwargs" not in json.loads(send.call_args.args[0].content)


# The shape of Qwen3.8's chat_template.jinja: resolve, then validate, then raise.
_QWEN38_TEMPLATE = """{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort') }}
{%- endif %}"""


def _write_template(tmp_data_dir, text=_QWEN38_TEMPLATE, repo="org/qwen"):
    snap = tmp_data_dir / "hf-cache" / f"models--{repo.replace('/', '--')}" / "snapshots" / "x"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "chat_template.jinja").write_text(text)


def _turn_payload(tmp_data_dir, client, h, settings):
    chat = client.post("/api/chat2/chats", headers=h,
                       json={"model": "qwen", "settings": settings}).json()
    body = _sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"},
                                         "finish_reason": "stop"}]})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))) as send, \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": f"req-effort-{chat['id'][-6:]}",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    return json.loads(send.call_args.args[0].content)


def test_reasoning_effort_reaches_vllm_only_in_the_models_own_vocabulary(tmp_data_dir, client) -> None:
    """#241: the level rides `chat_template_kwargs` next to `enable_thinking`,
    verbatim -- but only when the loaded model's template is known to accept
    that exact value. A validating template raises on an unknown one, so an
    unlisted value is dropped and the engine default applies."""
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    _write_template(tmp_data_dir)
    sent = _turn_payload(tmp_data_dir, client, h, {"reasoning_effort": "low"})
    assert sent["chat_template_kwargs"] == {"reasoning_effort": "low"}
    # a value this template does not list never leaves the warden
    sent = _turn_payload(tmp_data_dir, client, h, {"reasoning_effort": "ultra"})
    assert "chat_template_kwargs" not in sent
    # "" is the engine default: no key
    sent = _turn_payload(tmp_data_dir, client, h, {"reasoning_effort": ""})
    assert "chat_template_kwargs" not in sent


def test_reasoning_effort_is_inert_while_thinking_is_off(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    _write_template(tmp_data_dir)
    sent = _turn_payload(tmp_data_dir, client, h,
                         {"reasoning_effort": "low", "enable_thinking": False})
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_reasoning_effort_is_dropped_when_the_template_vocabulary_is_unknown(tmp_data_dir, client) -> None:
    """No readable template (a GGUF, a half-fetched repo) means no vocabulary,
    and no vocabulary means nothing is sent -- guessing is what would hand a
    model the value it raises on."""
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    sent = _turn_payload(tmp_data_dir, client, h, {"reasoning_effort": "low"})
    assert "chat_template_kwargs" not in sent


def test_the_first_llm_title_sets_no_undo_target(tmp_data_dir, client) -> None:
    """Turn 1's LLM title replaces the first-line fallback, not a title anyone
    chose. Recording that fallback as `title_prev` would put an "undo to `m0`"
    affordance on every new chat -- noise, and it would be spent before the
    first real reassessment ever arrived."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "t", "choices": [{"index": 0, "delta": {"content": "ok"},
                                         "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch("app.chat2.routes_turn.generate_llm_title",
                  new=AsyncMock(return_value="Mounting a volume")), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_draining()):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-first-title",
                              "user_parts": [{"type": "text", "text": "how do i mount this"}]})
    assert r.status_code == 200, r.text
    c = sqlite3.connect(db_path)
    assert c.execute("SELECT title, title_prev FROM chats").fetchone() == (
        "Mounting a volume", None)


def test_a_user_rename_mid_flight_beats_the_title_task(tmp_data_dir, client) -> None:
    """The title call sits at the lowest proxy priority, so the window between
    "decide to rename" and "write the rename" is seconds wide. A user who
    renames inside it must not have their title stomped."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse({"id": "t", "choices": [{"index": 0, "delta": {"content": "ok"},
                                         "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def rename_then_answer(**kwargs):
        # stands in for the user renaming while the title request is queued
        c = sqlite3.connect(db_path)
        c.execute("UPDATE chats SET title = 'Mine', title_source = 'user' WHERE id = ?",
                  (chat["id"],))
        c.commit()
        return "Generated title"

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch("app.chat2.routes_turn.generate_llm_title",
                  new=AsyncMock(side_effect=rename_then_answer)), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_draining()):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-race-01",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    c = sqlite3.connect(db_path)
    assert c.execute("SELECT title, title_source, title_prev FROM chats").fetchone() == (
        "Mine", "user", None)


# ---------------------------------------------------------------------------
# Detached turns (feat/chat2-detached-turns): abort endpoint + reattach
# ---------------------------------------------------------------------------


def _hanging_stream(first_chunk: bytes):
    """An upstream that yields one chunk and then hangs until cancelled —
    the shape of a long generation the user aborts mid-answer."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}

    async def aiter_raw():
        yield first_chunk
        await asyncio.Event().wait()  # forever, until the runner is cancelled
    resp.aiter_raw = aiter_raw
    resp.aread = AsyncMock(return_value=first_chunk)
    resp.aclose = AsyncMock()
    return resp


def _scope_request(app):
    from starlette.requests import Request as StarletteRequest
    return StarletteRequest({"type": "http", "app": app, "headers": [],
                             "method": "POST", "path": "/", "query_string": b""})


def test_abort_endpoint_persists_partial_tail_as_aborted(tmp_data_dir, client) -> None:
    """POST /chats/{id}/turn/abort cancels the RUNNER: the partial tail is
    persisted as `aborted` (ledger row included), the lock is released before
    the endpoint answers, and subscribers get a terminal aborted `done`."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    uid = client.get("/api/chat2/_whoami", headers=h).json()["user_id"]
    first = b'data: {"id": "ab", "choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": null}]}\n\n'
    app = client.app
    endpoint_result: dict = {}

    class _AbortViaEndpoint(StreamingResponse):
        def __init__(self, content, **kwargs):
            async def drive():
                n = 0
                async for f in content:
                    yield f
                    n += 1
                    if n == 4:  # persisted, start, context, text-delta("par")
                        # the lock is held while the runner lives...
                        assert app.state.chat2_turn_locks.held(chat["id"]) is True
                        res = await routes_turn.turn_abort(chat["id"], _scope_request(app), uid)
                        endpoint_result.update(res)
                        # ...and free the moment the abort endpoint answers
                        assert app.state.chat2_turn_locks.held(chat["id"]) is False
                while routes_turn._BACKGROUND:
                    await asyncio.gather(*list(routes_turn._BACKGROUND), return_exceptions=True)
                await asyncio.sleep(0)

            super().__init__(drive(), **kwargs)

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_hanging_stream(first))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_AbortViaEndpoint):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-live-abort",
                              "user_parts": [{"type": "text", "text": "x"}]})
    assert r.status_code == 200, r.text
    assert endpoint_result.get("aborted") is True
    evs = _events(r.text)
    types = [e["type"] for e in evs]
    # the still-attached subscriber gets the terminal aborted `done`, so a
    # reattached tab commits the partial tail exactly as the aborting client
    assert types == ["message-persisted", "message-start", "context", "text-delta", "done"]
    assert evs[-1]["finishReason"] == "aborted"
    assert evs[-1]["messageId"] == endpoint_result["message_id"]
    c = sqlite3.connect(db_path)
    role, parts, finish = c.execute(
        "SELECT role, parts_json, finish_reason FROM messages ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert role == "assistant" and finish == "aborted"
    assert json.loads(parts)[0] == {"type": "text", "text": "par"}
    assert c.execute("SELECT outcome FROM usage_ledger WHERE purpose = 'turn'"
                     ).fetchall() == [("aborted",)]
    assert client.app.state.chat_active_requests.count() == 0
    assert client.app.state.chat2_turn_locks.try_acquire(chat["id"]) is True
    client.app.state.chat2_turn_locks.release(chat["id"])
    detail = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert detail["live_turn"] is None


def test_abort_endpoint_404s_without_a_live_turn(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    _model(tmp_data_dir / "vllm-warden.db")
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    r = client.post(f"/api/chat2/chats/{chat['id']}/turn/abort", headers=h)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found"
    # an unknown (or foreign) chat is indistinguishable from no live turn
    r = client.post("/api/chat2/chats/nope/turn/abort", headers=h)
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found"


def test_turn_live_replays_byte_identical_and_live_turn_clears(tmp_data_dir, client) -> None:
    """GET /chats/{id}/turn/live must replay the EXACT bytes the original
    subscriber saw (the frontend reducer runs the same path), and the detail's
    `live_turn` is null once the turn is done."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    body = _sse(
        {"id": "r1", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": "stop"}]},
        {"id": "r1", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    )
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-replay-01",
                              "user_parts": [{"type": "text", "text": "hi"}]})
    assert r.status_code == 200, r.text
    # the finished turn stays replayable for the registry's grace window
    replay = client.get(f"/api/chat2/chats/{chat['id']}/turn/live", headers=h)
    assert replay.status_code == 200
    assert replay.content == r.content, "replay is not byte-identical"
    # but the detail already reports no live turn (done=True)
    detail = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert detail["live_turn"] is None
    # a chat with no registered turn at all is a 404 with the shared envelope
    other = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    r404 = client.get(f"/api/chat2/chats/{other['id']}/turn/live", headers=h)
    assert r404.status_code == 404 and r404.json()["detail"]["code"] == "not_found"


def test_reattach_mid_stream_gets_replay_then_live_frames(tmp_data_dir, client) -> None:
    """A second subscriber joining MID-turn replays everything so far and then
    follows the live tail to `done` — and the detail exposes `live_turn` while
    the runner is still streaming."""
    from app.chat2 import routes_chats

    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    uid = client.get("/api/chat2/_whoami", headers=h).json()["user_id"]
    body = _sse(
        {"id": "mt", "choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": None}]},
        {"id": "mt", "choices": [{"index": 0, "delta": {"content": "tial"}, "finish_reason": "stop"}]},
    )
    app = client.app
    captured: dict = {}

    class _MidStreamProbe(StreamingResponse):
        def __init__(self, content, **kwargs):
            async def drive():
                out: list[bytes] = []
                agen = content.__aiter__()
                for _ in range(4):
                    f = await agen.__anext__()
                    out.append(f)
                    yield f
                registry = app.state.chat2_live_turns
                live = registry.get(chat["id"])
                assert live is not None and live.done is False
                detail = await routes_chats.get_chat(chat["id"], _scope_request(app), uid)
                captured["live_turn"] = detail["live_turn"]
                # the second subscriber: full replay, then follows to done
                captured["second"] = [fr async for fr in registry.subscribe(live)]
                async for f in agen:
                    out.append(f)
                    yield f
                captured["first"] = out

            super().__init__(drive(), **kwargs)

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_stream(body))), \
            patch(_NO_TITLE[0], new=_NO_TITLE[1]), \
            patch("app.chat2.routes_turn.StreamingResponse", new=_MidStreamProbe):
        r = client.post(f"/api/chat2/chats/{chat['id']}/turns", headers=h,
                        json={"request_id": "req-mid-reattach",
                              "user_parts": [{"type": "text", "text": "x"}]})
    assert r.status_code == 200, r.text
    evs = _events(r.text)
    assert evs[-1]["type"] == "done" and evs[-1]["finishReason"] == "stop"
    # mid-stream, the detail advertised the live turn with the tail's id
    assert captured["live_turn"] == {"request_id": "req-mid-reattach",
                                     "message_id": evs[-1]["messageId"]}
    # the second subscriber saw the SAME bytes as the first, start to finish
    assert captured["second"] == captured["first"]
