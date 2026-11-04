"""Guard phase 2: the router. One scan in messages_entry; both legs (local and Anthropic) only
ever receive the sanitized body; the relayed answer is restored and carries the trailer; a
block is answered in the Anthropic dialect; the catch-all refuses a guarded key."""

import json
import sqlite3

import httpx

from app.guard.placeholders import guard_secret, placeholder_hex, render
from tests.unit.guard.corpus import github_token
from tests.unit.router.harness import (
    CLAUDE_CODE,
    chat_completion,
    enable_router,
    fake_anthropic,
    fake_engine,
    haiku_body,
    ready,
    sse_events,
)

VALUE = github_token()
HAIKU_RULE = {"pattern": "claude-haiku*", "target": "qwen"}
SANITIZE = {"override": {"mode": "sanitize", "packs": ["cloud-api-keys"], "trailer": "always"}}
BLOCK = {"override": {"mode": "block", "packs": ["cloud-api-keys"]}}


def _setup(client, tmp_data_dir, guard, anthropic=None, rules=(HAIKU_RULE,), **settings):
    ready(client, tmp_data_dir, relay=True)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE api_tokens SET guard = ?", (json.dumps(guard),))
        db.commit()
    enable_router(client, tmp_data_dir, list(rules), **settings)
    return fake_anthropic(client, anthropic)


def _ph(client) -> str:
    secret = guard_secret(client.app.state.settings.cookie_secret)
    return render(placeholder_hex(secret, "tok1", VALUE))


def _body(model="claude-sonnet-4-5", **extra):
    return haiku_body(
        **{
            "model": model,
            "messages": [{"role": "user", "content": f"deploy with {VALUE}"}],
            **extra,
        }
    )


def _post(client, body, path="/v1/messages"):
    return client.post(path, headers=CLAUDE_CODE, json=body)


def test_relay_never_sees_the_value_and_the_json_answer_is_restored(tmp_data_dir, client):
    holder = {}

    def handler(req):
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"export T={holder['ph']}"},
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": "bash",
                        "input": {"cmd": f"echo {holder['ph']}"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    up = _setup(client, tmp_data_dir, SANITIZE, anthropic=handler)
    holder["ph"] = _ph(client)
    with fake_engine(client) as engine_calls:
        r = _post(client, _body())
    assert not engine_calls
    assert r.status_code == 200, r.text
    (seen,) = up.requests
    assert VALUE.encode() not in seen["body"] and holder["ph"].encode() in seen["body"]
    assert seen["headers"]["accept-encoding"] == "identity"
    msg = r.json()
    assert msg["content"][0]["text"] == f"export T={VALUE}"
    assert msg["content"][1]["input"] == {"cmd": f"echo {VALUE}"}
    assert msg["content"][-1]["type"] == "text" and "[LM Warden]" in msg["content"][-1]["text"]
    assert VALUE not in msg["content"][-1]["text"]


def test_relay_stream_is_restored_and_the_trailer_comes_before_message_delta(tmp_data_dir, client):
    holder = {}

    def handler(req):
        ph = holder["ph"]
        half = len(ph) // 2
        evs = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "m",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 3, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "use " + ph[:half]},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": ph[half:] + " ok"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "t", "name": "bash", "input": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"cmd": "' + ph[:5]},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ph[5:] + '"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 9},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        raw = b"".join(f"event: {e}\ndata: {json.dumps(d)}\n\n".encode() for e, d in evs)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw)

    up = _setup(client, tmp_data_dir, SANITIZE, anthropic=handler)
    holder["ph"] = _ph(client)
    r = _post(client, _body(stream=True))
    assert r.status_code == 200, r.text
    assert VALUE.encode() not in up.requests[0]["body"]
    events = sse_events(r.content)
    names = [e for e, _ in events]
    text = "".join(
        d["delta"].get("text", "")
        for e, d in events
        if e == "content_block_delta" and d["index"] == 0
    )
    args = "".join(
        d["delta"].get("partial_json", "")
        for e, d in events
        if e == "content_block_delta" and d["index"] == 1
    )
    assert text == f"use {VALUE} ok"
    assert json.loads(args) == {"cmd": VALUE}
    trailer_idx = [
        i
        for i, (e, d) in enumerate(events)
        if e == "content_block_delta" and "[LM Warden]" in d["delta"].get("text", "")
    ]
    assert trailer_idx and trailer_idx[0] < names.index("message_delta")
    assert names[-1] == "message_stop"


def test_block_is_an_anthropic_message_and_nothing_is_relayed(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, BLOCK)
    r = _post(client, _body())
    assert r.status_code == 200 and not up.called
    assert r.headers["x-lmwarden-guard"].startswith("blocked")
    msg = r.json()
    assert msg["type"] == "message" and msg["model"] == "lm-warden-guard"
    assert msg["stop_reason"] == "end_turn" and VALUE not in json.dumps(msg)
    rs = _post(
        client,
        _body(stream=True, messages=[{"role": "user", "content": f"again {github_token()}"}]),
    )
    names = [e for e, _ in sse_events(rs.content)]
    assert names[0] == "message_start" and names[-1] == "message_stop"


def test_fallback_after_a_local_failure_relays_the_sanitized_body(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, SANITIZE, rules=({**HAIKU_RULE, "fallback": True},))

    async def boom(_req):
        raise httpx.ConnectError("refused")

    with fake_engine(client, boom):
        r = _post(client, _body(model="claude-haiku-4-5"))
    assert up.called, r.text
    assert VALUE.encode() not in up.requests[0]["body"]


def test_local_leg_gets_the_sanitized_body_and_the_answer_is_restored(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, SANITIZE)
    ph = _ph(client)

    async def engine(req):
        assert VALUE.encode() not in req.content
        return chat_completion(f"set {ph}")

    with fake_engine(client, engine) as calls:
        r = _post(client, _body(model="claude-haiku-4-5"))
    assert r.status_code == 200, r.text
    assert calls and not up.called
    text = r.json()["content"][0]["text"]
    assert text.startswith(f"set {VALUE}") and "[LM Warden]" in text


def test_count_tokens_is_sanitized_and_never_blocked(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, BLOCK)
    body = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": f"x {VALUE}"}]}
    r = _post(client, body, path="/v1/messages/count_tokens")
    assert r.status_code == 200 and up.called
    assert VALUE.encode() not in up.requests[0]["body"]
    # the count did not use up the block: the real request still blocks once
    r2 = _post(client, _body())
    assert r2.headers.get("x-lmwarden-guard", "").startswith("blocked")


def test_catch_all_refuses_a_guarded_key(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, SANITIZE)
    r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, json={"requests": []})
    assert r.status_code == 403 and "guard_no_passthrough" in r.text and not up.called


def test_a_router_block_names_a_request_id(tmp_data_dir, client):
    _setup(client, tmp_data_dir, BLOCK)
    r = _post(client, _body())
    assert r.headers["x-lmwarden-guard"].startswith("blocked") and r.headers.get("x-lmw-request-id")
