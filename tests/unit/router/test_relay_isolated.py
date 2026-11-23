"""Isolated keys on relayed models (plan 2, task 10, O6, and the O4 router legs).

The provider's prompt cache is shared by every key relaying through the account, so an
isolated key's request leaves for the provider with no ``cache_control`` mark at all,
on every road there (passthrough, fallback, the catch-all). A sharing key's request is
relayed byte for byte. A local leg that refused an isolated key with the O4 403 is the
answer: never relayed.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from typing import Any

import httpx
import pytest

from app.proxy.token_cache import invalidate_all as invalidate_tokens
from tests.unit.router.harness import (
    CLAUDE_CODE,
    chat_completion,
    decisions,
    enable_router,
    fake_anthropic,
    fake_engine,
    ready,
)

HAIKU_RULE = {"pattern": "claude-haiku*", "target": "qwen"}
CC = {"type": "ephemeral"}
NOT_CHECKED = "this key is isolated and the request body cannot be checked"
O4 = "this key is isolated and model qwen cannot isolate its cache"


def _marked(model: str = "claude-sonnet-4-5", *, count: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "cache_control": dict(CC),
        "system": [{"type": "text", "text": "You are terse.", "cache_control": dict(CC)}],
        "tools": [
            {
                "name": "lookup",
                "description": "d",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": dict(CC),
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "cache_control": dict(CC)}],
            }
        ],
    }
    if not count:
        body["max_tokens"] = 16
    return body


def _db(tmp_data_dir):
    return tmp_data_dir / "vllm-warden.db"


def _isolate(tmp_data_dir) -> None:
    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE api_tokens SET kv_isolation = 'isolated' WHERE id = 'tok1'")
        db.commit()
    invalidate_tokens()


def _backend(tmp_data_dir, backend: str) -> None:
    from app.proxy.model_cache import invalidate_all

    with sqlite3.connect(_db(tmp_data_dir)) as db:
        db.execute("UPDATE models SET backend = ? WHERE id = 'qwen'", (backend,))
        db.commit()
    invalidate_all()


def _setup(client, tmp_data_dir, *, isolated: bool, rules=(HAIKU_RULE,), **settings):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, list(rules), **settings)
    if isolated:
        _isolate(tmp_data_dir)
    return fake_anthropic(client)


def _raw(body: dict[str, Any]) -> bytes:
    # Odd spacing on purpose: a re-serialised body would not match these bytes.
    return json.dumps(body, indent=1).encode()


def _post(client, raw: bytes, path: str = "/v1/messages?beta=true"):
    return client.post(
        path, headers={**CLAUDE_CODE, "content-type": "application/json"}, content=raw
    )


def _stripped(body: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(body)
    out.pop("cache_control", None)
    for part in (out["system"], out["tools"], out["messages"][0]["content"]):
        for b in part:
            b.pop("cache_control", None)
    return out


async def _engine_502(_req):
    raise httpx.ConnectError("refused")


# -- passthrough ----------------------------------------------------------------


@pytest.mark.parametrize("count", [False, True])
def test_isolated_relay_leaves_with_no_cache_control(tmp_data_dir, client, count):
    up = _setup(client, tmp_data_dir, isolated=True)
    body = _marked(count=count)
    path = "/v1/messages/count_tokens" if count else "/v1/messages?beta=true"
    with fake_engine(client) as engine:
        r = _post(client, _raw(body), path)
    assert r.status_code == 200, r.text
    assert not engine
    (seen,) = up.requests
    assert b"cache_control" not in seen["body"]
    assert json.loads(seen["body"]) == _stripped(body)
    assert decisions(client)[0]["route"] == "passthrough"


@pytest.mark.parametrize("count", [False, True])
def test_sharing_relay_is_unchanged(tmp_data_dir, client, count):
    up = _setup(client, tmp_data_dir, isolated=False)
    raw = _raw(_marked(count=count))
    path = "/v1/messages/count_tokens" if count else "/v1/messages?beta=true"
    r = _post(client, raw, path)
    assert r.status_code == 200, r.text
    (seen,) = up.requests
    assert seen["body"] == raw


def test_isolated_relay_with_a_lone_surrogate_is_stripped_not_a_500(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    body = _marked()
    body["messages"][0]["content"][0]["text"] = "\ud800"
    r = _post(client, json.dumps(body).encode())
    assert r.status_code == 200, r.text
    (seen,) = up.requests
    assert b"cache_control" not in seen["body"]
    assert json.loads(seen["body"]) == _stripped(body)


def test_isolated_relay_is_always_reserialised(tmp_data_dir, client):
    """The bytes that leave are the object that was checked, marks or not."""
    up = _setup(client, tmp_data_dir, isolated=True)
    raw = _raw(_stripped(_marked()))
    assert _post(client, raw).status_code == 200
    assert up.requests[0]["body"] == json.dumps(json.loads(raw)).encode() != raw


@pytest.mark.parametrize(
    "raw",
    [
        # A first-wins parser would read the mark; Python keeps the last value.
        b'{"model":"claude-sonnet-4-5","max_tokens":8,"cache_control":{"type":"ephemeral"},'
        b'"cache_control":null,"messages":[{"role":"user","content":"hi"}]}',
        b'{"model":"claude-sonnet-4-5","max_tokens":8,"messages":[{"role":"user","content":'
        b'[{"type":"text","text":"a","cache_control":{"type":"ephemeral"},"type":"text"}]}]}',
    ],
)
def test_isolated_relay_with_a_duplicate_key_is_refused(tmp_data_dir, client, raw):
    up = _setup(client, tmp_data_dir, isolated=True)
    r = _post(client, raw)
    assert r.status_code == 400, r.text
    assert NOT_CHECKED in r.text
    assert not up.called


def test_isolated_relay_strips_message_and_document_source_marks(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "cache_control": dict(CC),
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "content",
                            "content": [{"type": "text", "text": "d", "cache_control": dict(CC)}],
                        },
                    }
                ],
            }
        ],
    }
    assert _post(client, _raw(body)).status_code == 200
    assert b"cache_control" not in up.requests[0]["body"]


# -- fallback -------------------------------------------------------------------


def test_isolated_fallback_is_stripped(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    body = _marked("claude-haiku-4-5")
    with fake_engine(client, _engine_502) as engine:
        r = _post(client, _raw(body))
    assert r.status_code == 200, r.text
    assert len(engine) == 1
    (d,) = decisions(client)
    assert d["route"] == "fallback"
    (seen,) = up.requests
    assert b"cache_control" not in seen["body"]
    assert json.loads(seen["body"]) == _stripped(body)


def test_sharing_fallback_is_unchanged(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=False)
    raw = _raw(_marked("claude-haiku-4-5"))
    with fake_engine(client, _engine_502):
        assert _post(client, raw).status_code == 200
    assert up.requests[0]["body"] == raw


@pytest.mark.parametrize("fallback", [True, False])
def test_o4_refusal_on_the_local_leg_is_never_relayed(tmp_data_dir, client, fallback):
    up = _setup(client, tmp_data_dir, isolated=True, rules=[{**HAIKU_RULE, "fallback": fallback}])
    _backend(tmp_data_dir, "llamacpp")
    with fake_engine(client) as engine:
        r = _post(client, _raw(_marked("claude-haiku-4-5")))
    assert r.status_code == 403, r.text
    assert O4 in r.text
    assert not up.called and not engine
    (d,) = decisions(client)
    assert d["route"] == "local" and d["status"] == 403
    snap = client.app.state.router.snapshot(None)
    assert not snap.get("by_reason", {}).get("status_403")
    assert client.app.state.router.breaker_allows("qwen")


@pytest.mark.parametrize("state", ["not_loaded", "breaker_open"])
def test_o4_refusal_comes_before_the_soft_fallbacks(tmp_data_dir, client, state):
    """The same target never answers 403 one moment and relays the next."""
    up = _setup(client, tmp_data_dir, isolated=True)
    _backend(tmp_data_dir, "llamacpp")
    if state == "not_loaded":
        with sqlite3.connect(_db(tmp_data_dir)) as db:
            db.execute("UPDATE models SET status = 'pulled' WHERE id = 'qwen'")
            db.commit()
        from app.proxy.model_cache import invalidate_all

        invalidate_all()
    else:
        client.app.state.router.record_failure("qwen", "status_502", threshold=1, open_s=60)
        assert not client.app.state.router.breaker_allows("qwen")
    with fake_engine(client) as engine:
        r = _post(client, _raw(_marked("claude-haiku-4-5")))
    assert r.status_code == 403, r.text
    assert O4 in r.text
    assert not up.called and not engine


def test_a_deleted_target_still_relays_an_isolated_key_stripped(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True, rules=[{**HAIKU_RULE, "target": "gone"}])
    with fake_engine(client) as engine:
        r = _post(client, _raw(_marked("claude-haiku-4-5")))
    assert r.status_code == 200, r.text
    assert not engine
    assert b"cache_control" not in up.requests[0]["body"]


def test_a_sharing_key_on_llamacpp_is_served_by_the_rule(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=False)
    _backend(tmp_data_dir, "llamacpp")
    with fake_engine(client, _engine_returning_ok()) as engine:
        r = _post(client, _raw(_marked("claude-haiku-4-5")))
    assert r.status_code == 200, r.text
    assert len(engine) == 1 and not up.called


def _engine_returning_ok():
    async def h(_req):
        return chat_completion()

    return h


def test_isolation_unavailable_falls_back_stripped_and_spares_the_breaker(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    client.app.state.cobs2_key = None  # the local leg answers 503 for an isolated key
    body = _marked("claude-haiku-4-5")
    with fake_engine(client) as engine:
        for _ in range(4):  # more than the breaker threshold
            r = _post(client, _raw(body))
            assert r.status_code == 200, r.text
    assert not engine
    assert len(up.requests) == 4
    assert all(b"cache_control" not in s["body"] for s in up.requests)
    assert [d["reason"] for d in decisions(client)] == ["isolation_unavailable"] * 4
    assert client.app.state.router.breaker_allows("qwen")


def test_isolation_unavailable_without_fallback_is_the_503(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True, rules=[{**HAIKU_RULE, "fallback": False}])
    client.app.state.cobs2_key = None
    with fake_engine(client) as engine:
        r = _post(client, _raw(_marked("claude-haiku-4-5")))
    assert r.status_code == 503, r.text
    assert "cache isolation is unavailable right now" in r.text
    assert not up.called and not engine
    assert client.app.state.router.breaker_allows("qwen")


# -- the catch-all --------------------------------------------------------------


def test_isolated_relay_empty_body_passes(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    r = client.get("/v1/models/claude-x?limit=2", headers=CLAUDE_CODE)
    assert r.status_code == 200, r.text
    (seen,) = up.requests
    assert seen["method"] == "GET" and seen["body"] == b""


def test_isolated_relay_non_json_is_refused(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    for payload in (b"batch-bytes" * 100, b"[1, 2]", b"\xff\xfe"):
        r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, content=payload)
        assert r.status_code == 400, r.text
        assert NOT_CHECKED in r.text
    assert not up.called


def test_isolated_catch_all_json_is_stripped(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True)
    batch = {"requests": [{"custom_id": "a", "params": _marked()}]}
    r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, content=_raw(batch))
    assert r.status_code == 200, r.text
    (seen,) = up.requests
    assert b"cache_control" not in seen["body"]
    assert json.loads(seen["body"])["requests"][0]["params"] == _stripped(_marked())


def test_isolated_catch_all_over_the_cap_is_413(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=True, max_body_mb=1)
    big = b'{"x": "' + b"a" * (2 << 20) + b'"}'
    r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, content=big)
    assert r.status_code == 413
    assert not up.called


def test_sharing_catch_all_is_streamed_unchanged(tmp_data_dir, client):
    up = _setup(client, tmp_data_dir, isolated=False)
    raw = _raw({"requests": [{"custom_id": "a", "params": _marked()}]})
    r = client.post("/v1/messages/batches", headers=CLAUDE_CODE, content=raw)
    assert r.status_code == 200
    assert up.requests[0]["body"] == raw


# -- the rules API ----------------------------------------------------------------


def test_rules_api_marks_the_provider_legs(tmp_data_dir, client):
    from tests.conftest import jwt_login, seed_admin_user

    ready(client, tmp_data_dir)
    enable_router(
        client,
        tmp_data_dir,
        [HAIKU_RULE, {"pattern": "claude-sonnet*", "target": "qwen", "fallback": False}],
    )
    seed_admin_user(_db(tmp_data_dir))
    auth = jwt_login(client)
    out = client.get("/api/router/rules", headers=auth).json()
    assert out["passthrough_cache_isolation"] == "not_isolated_at_provider"
    a, b = out["rules"]
    assert a["cache_isolation"] == "isolated"
    assert a["fallback_cache_isolation"] == "not_isolated_at_provider"
    assert b["fallback_cache_isolation"] is None
    _backend(tmp_data_dir, "llamacpp")
    a, _ = client.get("/api/router/rules", headers=auth).json()["rules"]
    assert a["cache_isolation"] == "unsupported"
