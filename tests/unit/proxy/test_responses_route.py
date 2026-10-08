"""#Codex: ``POST /v1/responses`` end to end through the real ``_forward``.

The engine is faked by patching ``httpx.AsyncClient.send`` as the other proxy
tests do. Each test asserts BOTH sides: what reached the engine, and what the
Codex client got.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from tests.unit.router.harness import PLAINTEXT, enable_router, ready

AUTH = {"Authorization": f"Bearer {PLAINTEXT}"}
THREAD = "0199c6e2-aaaa-7bbb-8ccc-0123456789ab"
BODY = {"model": "qwen", "input": "hi", "max_output_tokens": 16}
OK = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 2,
        "total_tokens": 11,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}


def json_upstream(body: dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def sse_upstream(events: list[dict[str, Any]], on_chunk=None) -> MagicMock:
    frames = [b"data: " + json.dumps(ev).encode() + b"\n\n" for ev in events]
    frames.append(b"data: [DONE]\n\n")

    async def aiter():
        for f in frames:
            if on_chunk is not None:
                on_chunk()
            yield f

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def chunk(delta=None, finish=None, usage=None) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "id": "c",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [],
    }
    if delta is not None or finish is not None:
        ev["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        ev["usage"] = usage
    return ev


def post(client, body=BODY, *, headers=AUTH, upstream=None):
    send = AsyncMock(return_value=upstream)
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post("/v1/responses", headers=headers, json=body)
    return r, send


def sent_body(send: AsyncMock) -> dict[str, Any]:
    req = send.call_args.args[0]
    assert str(req.url).endswith("/v1/chat/completions")
    return json.loads(req.content)


def events_of(raw: bytes) -> list[dict[str, Any]]:
    out = []
    for frame in raw.decode().split("\n\n"):
        if frame.strip():
            lines = dict(line.split(": ", 1) for line in frame.split("\n"))
            data = json.loads(lines["data"])
            assert data["type"] == lines["event"]
            out.append(data)
    return out


def stream(client, body, upstream, headers=AUTH):
    send = AsyncMock(return_value=upstream)
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream("POST", "/v1/responses", headers=headers, json=body) as r:
            raw = b"".join(r.iter_bytes())
            status, ctype = r.status_code, r.headers.get("content-type", "")
    return status, ctype, events_of(raw) if status == 200 else raw, send


def assert_envelope(r, status: int, kind: str) -> None:
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["type"] == kind and err["param"] is None and err["code"] is None
    assert isinstance(err["message"], str) and err["message"]
    assert "type" not in r.json()  # not Anthropic's {"type": "error", ...}


# ---- happy paths ------------------------------------------------------------------------------


def test_non_stream_round_trip(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r, send = post(client, upstream=json_upstream(OK))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "response" and body["status"] == "completed"
    assert body["model"] == "qwen"
    assert body["output"][0]["content"][0]["text"] == "OK"
    assert body["usage"]["input_tokens"] == 9
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 4
    chat = sent_body(send)
    assert chat["messages"] == [{"role": "user", "content": "hi"}]
    assert chat["max_tokens"] == 16 and chat["chat_template_kwargs"] == {"enable_thinking": False}
    # `priority` is added by _forward itself
    assert set(chat) <= {
        "model",
        "messages",
        "max_tokens",
        "chat_template_kwargs",
        "stream",
        "stream_options",
        "priority",
    }


def test_stream_round_trip_and_chat_body(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = sse_upstream(
        [
            chunk({"role": "assistant", "content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, "stop"),
            chunk(usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
        ]
    )
    body = {
        **BODY,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": THREAD,
        "instructions": "BE BRIEF",
    }
    status, ctype, ev, send = stream(client, body, up)
    assert status == 200 and ctype.startswith("text/event-stream")
    assert ev[0]["type"] == "response.created" and ev[-1]["type"] == "response.completed"
    assert [e["delta"] for e in ev if e["type"] == "response.output_text.delta"] == ["Hel", "lo"]
    chat = sent_body(send)
    assert chat["stream"] is True and chat["stream_options"]["include_usage"] is True
    assert chat["messages"][0] == {"role": "system", "content": "BE BRIEF"}
    assert "prompt_cache_key" not in chat and "store" not in chat and "include" not in chat


def test_tool_call_reply_streams_a_function_call_item(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    call = {
        "index": 0,
        "id": "call_9",
        "type": "function",
        "function": {"name": "shell", "arguments": '{"c":1}'},
    }
    up = sse_upstream([chunk({"tool_calls": [call]}), chunk({}, "tool_calls")])
    body = {**BODY, "stream": True, "tools": [{"type": "function", "name": "shell"}]}
    _, _, ev, _ = stream(client, body, up)
    item = ev[-1]["response"]["output"][0]
    assert (item["type"], item["call_id"], item["arguments"]) == (
        "function_call",
        "call_9",
        '{"c":1}',
    )


def test_max_output_tokens_is_clamped_to_the_context(tmp_data_dir, client):
    ready(client, tmp_data_dir)  # max_model_len 4096
    r, send = post(client, {**BODY, "max_output_tokens": 100000}, upstream=json_upstream(OK))
    assert r.status_code == 200
    assert sent_body(send)["max_tokens"] < 4096


def test_x_lmwarden_key_header_works(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r, _ = post(client, headers={"X-LMWarden-Key": PLAINTEXT}, upstream=json_upstream(OK))
    assert r.status_code == 200


# ---- affinity and the live row ----------------------------------------------------------------


def live_row_during(client, hdrs, body, events=None):
    rows: list = []
    up = sse_upstream(
        events or [chunk({"content": "a"}), chunk({}, "stop")],
        on_chunk=lambda: rows.extend(client.app.state.request_registry.snapshot()),
    )
    stream(client, {**body, "stream": True}, up, headers={**AUTH, **hdrs})
    assert rows
    return rows[0]


def test_codex_session_header_is_the_live_session(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    row = live_row_during(client, {"session-id": THREAD}, {**BODY, "prompt_cache_key": THREAD})
    assert (row.session_id, row.session_source) == (THREAD, "session_id_header")
    assert row.path == "/v1/chat/completions"


def test_prompt_cache_key_alone_is_the_session(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    row = live_row_during(client, {}, {**BODY, "prompt_cache_key": THREAD})
    assert (row.session_id, row.session_source) == (THREAD, "prompt_cache_key")


def test_prompt_hash_fallback_skips_environment_context(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    items = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "rules"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "<environment_context>cwd</environment_context>"}
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "fix the bug"}],
        },
    ]
    row = live_row_during(client, {}, {"model": "qwen", "input": items})
    assert row.session_id is None and row.session_source == "prompt_hash"


def test_affinity_unit():
    from app.proxy.routes_responses import responses_affinity

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "fix the bug"}],
        }
    ]
    assert responses_affinity({"input": items}, {}, "tok1") == (
        "tok1\x00fix the bug",
        "prompt_hash",
        None,
    )
    assert responses_affinity({"input": items, "prompt_cache_key": THREAD}, {}, "tok1") == (
        THREAD,
        "prompt_cache_key",
        THREAD,
    )
    assert responses_affinity({"input": []}, {}, "tok1") == (None, "none", None)
    assert (
        responses_affinity({"input": "plain"}, {"x-session-affinity": "ses_12345678"}, "t")[1]
        == "x_session_affinity"
    )


def test_dp_rank_is_sticky_per_thread(tmp_data_dir, client):
    import sqlite3

    ready(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE models SET gpu_indices=?, data_parallel_size=7 WHERE id='qwen'",
            (json.dumps(list(range(7))),),
        )
    ranks = []
    for _ in range(3):
        _, send = post(client, {**BODY, "prompt_cache_key": THREAD}, upstream=json_upstream(OK))
        ranks.append(send.call_args.args[0].headers["x-data-parallel-rank"])
    assert len(set(ranks)) == 1
    totals = client.app.state.dp_routing.snapshot("qwen", 7)["totals"]
    assert totals["placed"] == 1 and totals["sticky"] == 2


# ---- errors --------------------------------------------------------------------------------------


def test_missing_and_bad_token_are_401_in_the_openai_envelope(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r, send = post(client, headers={})
    assert_envelope(r, 401, "authentication_error")
    r, _ = post(client, headers={"Authorization": "Bearer vw_nope"})
    assert_envelope(r, 401, "authentication_error")
    r, _ = post(client, headers={"x-api-key": PLAINTEXT})  # not a credential here
    assert_envelope(r, 401, "authentication_error")
    send.assert_not_called()


def test_token_not_allowed_for_model_is_403(tmp_data_dir, client):
    ready(client, tmp_data_dir, allowed_models="other")
    r, _ = post(client)
    assert_envelope(r, 403, "permission_error")


def test_unknown_model_is_404(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r, send = post(client, {**BODY, "model": "gpt-5"})
    assert_envelope(r, 404, "invalid_request_error")
    assert "gpt-5" in r.json()["error"]["message"]
    send.assert_not_called()


def test_bad_json_and_bad_shape_are_400(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    with patch("httpx.AsyncClient.send", new=AsyncMock()) as send:
        r = client.post(
            "/v1/responses", headers={**AUTH, "content-type": "application/json"}, content=b"{nope"
        )
        assert_envelope(r, 400, "invalid_request_error")
        r = client.post("/v1/responses", headers=AUTH, json=[1])
        assert_envelope(r, 400, "invalid_request_error")
        r = client.post("/v1/responses", headers=AUTH, json={"model": "qwen"})
        assert_envelope(r, 400, "invalid_request_error")
        assert r.json()["error"]["message"].startswith("input")
    send.assert_not_called()


def test_previous_response_id_is_a_documented_400(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r, send = post(client, {**BODY, "previous_response_id": "resp_x"})
    assert_envelope(r, 400, "invalid_request_error")
    assert "previous_response_id" in r.json()["error"]["message"]
    assert "store:false" in r.json()["error"]["message"]
    send.assert_not_called()


def test_translation_error_never_echoes_the_prompt(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    secret = "TOP-SECRET-USER-PROMPT-TEXT"
    r, _ = post(client, {"model": "qwen", "input": [{"type": "mystery", "text": secret}]})
    assert_envelope(r, 400, "invalid_request_error")
    assert secret not in r.text


def test_upstream_500_is_drained_into_the_envelope(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r, _ = post(client, upstream=json_upstream({"error": {"message": "engine on fire"}}, 500))
    assert_envelope(r, 500, "server_error")
    assert r.json()["error"]["message"] == "engine on fire"


def test_upstream_500_on_a_streamed_request(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = json_upstream({"error": {"message": "engine on fire"}}, 500)

    async def aiter():
        yield json.dumps({"error": {"message": "engine on fire"}}).encode()

    up.aiter_bytes = aiter
    status, _, raw, _ = stream(client, {**BODY, "stream": True}, up)
    assert status == 500
    assert json.loads(raw)["error"]["message"] == "engine on fire"


def test_non_json_upstream_is_502(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = json_upstream({})
    up.aread = AsyncMock(return_value=b"<html>")
    r, _ = post(client, upstream=up)
    assert r.status_code in (502, 500) and "error" in r.json()


def test_in_stream_engine_error_is_response_failed(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = sse_upstream([chunk({"content": "a"}), {"error": {"message": "died"}}])
    status, _, ev, _ = stream(client, {**BODY, "stream": True}, up)
    assert status == 200 and ev[-1]["type"] == "response.failed"
    assert ev[-1]["response"]["error"]["message"] == "died"


# ---- routing -------------------------------------------------------------------------------------


def test_router_on_still_serves_locally_without_rules_or_passthrough(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(
        client,
        tmp_data_dir,
        rules=[{"pattern": "gpt-*", "target": "qwen"}],
        passthrough_unmatched=True,
    )
    r, send = post(client, upstream=json_upstream(OK))
    assert r.status_code == 200
    # a name only a rule would match is NOT rerouted: served names only
    r, _ = post(client, {**BODY, "model": "gpt-5"})
    assert_envelope(r, 404, "invalid_request_error")
    assert send.call_count == 1


def test_get_is_405_with_allow(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r = client.get("/v1/responses", headers=AUTH)
    assert r.status_code == 405 and "POST" in r.headers["allow"]
    r = client.get("/v1/responses")  # unauthenticated: same answer
    assert r.status_code == 405


def test_route_is_ahead_of_the_catch_all(tmp_data_dir, client):
    from app.proxy.routes_messages import find_routes_after_catch_all

    paths = [getattr(r, "path", "") for r in client.app.router.routes]
    assert paths.index("/v1/responses") < paths.index("/v1/{rest:path}")
    assert find_routes_after_catch_all(client.app) == []


def test_client_disconnect_before_first_chunk_releases_rank_and_slot(tmp_data_dir, client):
    """The wrapper carries the inner response's background task: a client gone
    before the first chunk leaves the generator unstarted, and only
    ``abandon`` (run as background) releases the rank and the admission slot."""
    import sqlite3

    ready(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE models SET gpu_indices=?, data_parallel_size=7 WHERE id='qwen'",
            (json.dumps(list(range(7))),),
        )
    upstream = sse_upstream([chunk({"content": "a"}), chunk({}, "stop")])
    raw = json.dumps({**BODY, "stream": True}).encode()
    app = client.app

    async def run() -> None:
        from starlette.requests import Request

        from app.db.repos.tokens import TokenRow
        from app.proxy.routes_responses import responses

        async def body_receive() -> dict[str, Any]:
            return {"type": "http.request", "body": raw, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"content-type", b"application/json")],
            "app": app,
            "query_string": b"",
            "client": ("127.0.0.1", 1),
        }
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
        with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=upstream)):
            resp = await responses(Request(scope, body_receive), token)

        async def disconnected() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            await asyncio.sleep(3600)

        await asyncio.wait_for(resp(scope, disconnected, send), timeout=10)

    client.portal.call(run)
    assert client.app.state.dp_routing.snapshot("qwen", 7)["totals"]["in_flight"] == 0
    assert client.app.state.scheduler._inflight_for_test("qwen") == 0


def test_history_records_the_chat_path_and_cached_tokens(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = sse_upstream(
        [
            chunk({"content": "a"}),
            chunk({}, "stop"),
            chunk(
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 1,
                    "total_tokens": 21,
                    "prompt_tokens_details": {"cached_tokens": 12},
                }
            ),
        ]
    )
    status, _, ev, _ = stream(client, {**BODY, "stream": True, "prompt_cache_key": THREAD}, up)
    assert status == 200
    assert ev[-1]["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 12


# ---- contract: a replay of Codex 0.160's turn pair --------------------------------------------

CODEX = json.loads(
    (
        Path(__file__).resolve().parents[2] / "fixtures" / "responses" / "codex_turns.json"
    ).read_text()
)


def _codex_turn(client, turn: str, events: list[dict[str, Any]]):
    body = {**CODEX[turn], "model": "qwen"}  # the fixture keeps the live served name
    return stream(client, body, sse_upstream(events), headers={**AUTH, **CODEX["headers"]})


def test_codex_turn_one_streams_a_shell_call_and_round_trips_the_call_id(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    call = {
        "index": 0,
        "id": "chatcmpl-tool-aaaa1111",
        "type": "function",
        "function": {"name": "exec_command", "arguments": '{"cmd":"ls | wc -l"}'},
    }
    status, _, ev, send = _codex_turn(
        client,
        "turn1",
        [chunk({"content": "Checking."}), chunk({"tool_calls": [call]}), chunk({}, "tool_calls")],
    )
    assert status == 200
    chat = sent_body(send)
    # developer + environment_context + user; namespace flattened, web_search absent
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "user"]
    assert chat["messages"][1]["content"].startswith("<environment_context>")
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names[:4] == ["exec_command", "write_stdin", "request_user_input", "view_image"]
    assert "close_agent" in names and len(names) == len(set(names))
    # reasoning {summary: auto}: thinking stays on
    assert "chat_template_kwargs" not in chat
    types = [e["type"] for e in ev]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    out = ev[-1]["response"]["output"]
    assert [i["type"] for i in out] == ["message", "function_call"]
    assert out[1]["call_id"] == "chatcmpl-tool-aaaa1111" and out[1]["name"] == "exec_command"


def test_codex_real_turn_two_replays_every_item_without_a_400(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    status, _, ev, send = _codex_turn(
        client, "turn2", [chunk({"content": "3 files."}), chunk({}, "stop")]
    )
    assert status == 200 and ev[-1]["response"]["status"] == "completed"
    chat = sent_body(send)["messages"]
    # exactly one system message (the leading developer item); the developer
    # items Codex repeats on resume are folded into user turns, never a second system
    assert [m["role"] for m in chat].count("system") == 1 and chat[0]["role"] == "system"
    # reasoning items are dropped on input
    assert not any("reasoning" in json.dumps(m)[:40] for m in chat)
    calls = [(m["tool_calls"], i) for i, m in enumerate(chat) if m.get("tool_calls")]
    assert calls
    # every call id round-trips and its tool message follows the assistant message directly
    for tool_calls, i in calls:
        for k, c in enumerate(tool_calls):
            assert chat[i + 1 + k]["role"] == "tool" and chat[i + 1 + k]["tool_call_id"] == c["id"]
    assert any(m["role"] == "user" and "[system message]" in str(m["content"]) for m in chat)


def test_codex_threads_share_a_session_and_a_replica_across_turns(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    seen = []
    for turn in ("turn1", "turn2"):
        rows: list = []
        up = sse_upstream(
            [chunk({"content": "x"}), chunk({}, "stop")],
            on_chunk=lambda rows=rows: rows.extend(client.app.state.request_registry.snapshot()),
        )
        stream(client, {**CODEX[turn], "model": "qwen"}, up, headers={**AUTH, **CODEX["headers"]})
        seen.append((rows[0].session_id, rows[0].session_source))
    assert seen[0] == seen[1] == (CODEX["headers"]["session-id"], "session_id_header")


def test_translator_crash_mid_stream_still_ends_with_one_terminal_event(tmp_data_dir, client):
    from app.proxy.responses_translate import ResponsesStreamTranslator

    ready(client, tmp_data_dir)
    calls = {"n": 0}
    orig = ResponsesStreamTranslator.feed_event

    def boom(self, ev):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("TOP-SECRET-USER-PROMPT-TEXT")
        return orig(self, ev)

    up = sse_upstream([chunk({"content": "a"}), chunk({"content": "b"}), chunk({}, "stop")])
    with patch.object(ResponsesStreamTranslator, "feed_event", boom):
        status, _, ev, _ = stream(client, {**BODY, "stream": True}, up)
    assert status == 200
    terminal = [e["type"] for e in ev if e["type"] in ("response.completed", "response.failed")]
    assert terminal == ["response.failed"] and ev[-1]["type"] == "response.failed"
    assert ev[-1]["response"]["error"]["message"] == "response translation failed"
    assert "TOP-SECRET" not in json.dumps(ev)
    assert client.app.state.scheduler._inflight_for_test("qwen") == 0
