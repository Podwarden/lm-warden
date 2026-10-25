"""Thinking policy applied in the forward path (spec 2026-10-25 §17.3)."""

import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.proxy.thinking_policy import apply_thinking_off, effective_policy
from tests.conftest import wait_until
from tests.unit.proxy.test_messages import AUTH, _json_upstream, _ready

OK = {
    "id": "c1",
    "object": "chat.completion",
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
CHAT = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}


# -- pure ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tok", "mod", "want"),
    [
        ("client", "client", "client"),
        ("off", "client", "off"),
        ("client", "off", "off"),
        ("off", "off", "off"),
        (None, None, "client"),
        ("garbage", "client", "client"),
        (None, "off", "off"),
    ],
)
def test_effective_policy_matrix(tok, mod, want):
    assert effective_policy(tok, mod) == want


def test_off_overrides_client_and_keeps_other_kwargs():
    body = {
        "messages": [],
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": True, "other": 1},
    }
    assert apply_thinking_off(body) is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "other": 1}
    assert body["temperature"] == 0.3


def test_off_adds_kwargs_when_absent():
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "x"}]}
    assert apply_thinking_off(body) is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_completions_prompt_body_is_untouched():
    body = {"model": "qwen", "prompt": "hello"}
    assert apply_thinking_off(body) is False
    assert body == {"model": "qwen", "prompt": "hello"}


# -- forward ------------------------------------------------------------------


def _set(db_path, sql: str) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(sql)
        db.commit()


def _set_model_off(client, tmp_data_dir) -> None:
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    async def go():
        async with open_db(str(tmp_data_dir / "vllm-warden.db")) as db:
            await ModelRepo(db).update_fields("qwen", {"thinking_policy": "off"})

    client.portal.call(go)


def _go(client, body=None, *, upstream=None, path="/v1/chat/completions", hook=None):
    resp = upstream or _json_upstream(OK)

    async def _send(request, *, stream=False, **kw):
        if hook is not None:
            hook()
        return resp

    send = AsyncMock(side_effect=_send)
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(path, headers=AUTH, json=body or CHAT)
    return r, send


def _sent(send) -> bytes:
    return send.call_args.args[0].content


def test_client_policy_changes_nothing(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    r, send = _go(client, body)
    assert r.status_code == 200
    # Only the existing priority injection touches the body; the policy adds nothing.
    assert _sent(send) == json.dumps({**body, "priority": -5}).encode()


def test_token_off_forces_enable_thinking_false(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set(tmp_data_dir / "vllm-warden.db", "UPDATE api_tokens SET thinking_policy='off'")
    body = {**CHAT, "chat_template_kwargs": {"enable_thinking": True, "x": 1}}
    r, send = _go(client, body)
    assert r.status_code == 200
    sent = json.loads(_sent(send))
    assert sent["chat_template_kwargs"] == {"enable_thinking": False, "x": 1}
    assert sent["messages"] == body["messages"]


def test_model_off_applies_on_next_request_without_reload(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _, send = _go(client)
    assert "chat_template_kwargs" not in json.loads(_sent(send))
    _set_model_off(client, tmp_data_dir)
    _, send = _go(client)
    assert json.loads(_sent(send))["chat_template_kwargs"] == {"enable_thinking": False}


def test_completions_prompt_forwarded_untouched_under_off(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set(tmp_data_dir / "vllm-warden.db", "UPDATE api_tokens SET thinking_policy='off'")
    body = {"model": "qwen", "prompt": "hello"}
    r, send = _go(client, body, path="/v1/completions")
    assert r.status_code == 200
    assert "chat_template_kwargs" not in json.loads(_sent(send))


def test_messages_thinking_request_is_forced_off(tmp_data_dir, client):
    """The fake engine reasons only when enable_thinking is not false, like a real one."""
    _ready(client, tmp_data_dir)
    _set(tmp_data_dir / "vllm-warden.db", "UPDATE api_tokens SET thinking_policy='off'")
    seen: list[dict[str, Any]] = []

    async def _send(request, *, stream=False, **kw):
        body = json.loads(request.content)
        seen.append(body)
        kw_ = body.get("chat_template_kwargs") or {}
        msg: dict[str, Any] = {"role": "assistant", "content": "answer"}
        if kw_.get("enable_thinking") is not False:
            msg["reasoning_content"] = "pondering"
        out = {**OK, "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}
        return _json_upstream(out)

    req = {
        "model": "qwen",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 2000},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=_send)):
        r = client.post("/v1/messages", headers=AUTH, json=req)
    assert r.status_code == 200
    assert seen[0]["chat_template_kwargs"]["enable_thinking"] is False
    types = [b["type"] for b in r.json()["content"]]
    assert "thinking" not in types and "text" in types


def _history(db_path):
    def _rows():
        with sqlite3.connect(db_path) as db:
            return db.execute("SELECT thinking_forced_off FROM request_history").fetchall() or None

    return wait_until(_rows, what="request_history row")


def test_live_and_history_mark_forced_off(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    live: list[bool] = []

    def _peek():
        live.extend(r.thinking_forced_off for r in client.app.state.request_registry.snapshot())

    _set(db_path, "UPDATE api_tokens SET thinking_policy='off'")
    _go(client, hook=_peek)
    assert live == [True]
    assert _history(db_path) == [(1,)]


def test_history_marks_not_forced_when_client(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    live: list[bool] = []

    def _peek():
        live.extend(r.thinking_forced_off for r in client.app.state.request_registry.snapshot())

    _go(client, hook=_peek)
    assert live == [False]
    assert _history(tmp_data_dir / "vllm-warden.db") == [(0,)]


def test_policy_failure_fails_open(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    _set(tmp_data_dir / "vllm-warden.db", "UPDATE api_tokens SET thinking_policy='off'")

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.proxy.routes.effective_policy", _boom)
    r, send = _go(client)
    assert r.status_code == 200
    assert _sent(send) == json.dumps({**CHAT, "priority": -5}).encode()


def test_messages_without_thinking_on_off_token_is_marked_forced_off(tmp_data_dir, client):
    """The translator already sends enable_thinking=false; the flag still reads 1."""
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _set(db_path, "UPDATE api_tokens SET thinking_policy='off'")
    req = {"model": "qwen", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_json_upstream(OK))):
        r = client.post("/v1/messages", headers=AUTH, json=req)
    assert r.status_code == 200
    assert _history(db_path) == [(1,)]


def test_completions_on_off_token_is_not_marked(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _set(db_path, "UPDATE api_tokens SET thinking_policy='off'")
    _go(client, {"model": "qwen", "prompt": "hello"}, path="/v1/completions")
    assert _history(db_path) == [(0,)]
