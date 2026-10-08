"""Mid-conversation system messages on the router's local leg (same fold as the plain path)."""

import json

from tests.unit.router.harness import (
    CLAUDE_CODE,
    chat_completion,
    enable_router,
    fake_anthropic,
    fake_engine,
    ready,
)

NOTE = "Plan mode is active."


async def _ok(_req):
    return chat_completion()


def _body(model):
    return {
        "model": model,
        "max_tokens": 32,
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "system", "content": [{"type": "text", "text": NOTE}]},
            {"role": "user", "content": "two"},
        ],
    }


def _check(calls):
    (req,) = calls
    sent = json.loads(req.content)["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "user"]
    assert NOTE in sent[2]["content"] and sent[2]["content"].endswith("two")


def test_rule_matched_local_leg_folds_the_system_message(tmp_data_dir, client):
    ready(client, tmp_data_dir, relay=True)
    enable_router(client, tmp_data_dir, [{"pattern": "claude-haiku*", "target": "qwen"}])
    up = fake_anthropic(client)
    with fake_engine(client, _ok) as calls:
        r = client.post(
            "/v1/messages?beta=true", headers=CLAUDE_CODE, json=_body("claude-haiku-4-5")
        )
    assert r.status_code == 200, r.text
    assert not up.called
    _check(calls)


def test_served_name_with_router_on_folds_the_system_message(tmp_data_dir, client):
    ready(client, tmp_data_dir, relay=True)
    enable_router(client, tmp_data_dir, [{"pattern": "claude-haiku*", "target": "qwen"}])
    fake_anthropic(client)
    with fake_engine(client, _ok) as calls:
        r = client.post("/v1/messages?beta=true", headers=CLAUDE_CODE, json=_body("qwen"))
    assert r.status_code == 200, r.text
    _check(calls)


def test_count_tokens_local_leg_accepts_system_role(tmp_data_dir, client):
    ready(client, tmp_data_dir, relay=True)
    enable_router(client, tmp_data_dir, [{"pattern": "claude-haiku*", "target": "qwen"}])
    up = fake_anthropic(client)
    body = _body("claude-haiku-4-5")
    del body["max_tokens"]
    r = client.post("/v1/messages/count_tokens", headers=CLAUDE_CODE, json=body)
    assert r.status_code == 200, r.text
    assert not up.called
    (text,) = [c.args[1] for c in client.app.state.tokenizers.count.await_args_list]
    assert NOTE in text


def test_passthrough_to_anthropic_is_byte_for_byte(tmp_data_dir, client):
    ready(client, tmp_data_dir, relay=True)
    enable_router(client, tmp_data_dir, [{"pattern": "claude-haiku*", "target": "qwen"}])
    up = fake_anthropic(client)
    raw = json.dumps(_body("claude-opus-4")).encode()
    r = client.post(
        "/v1/messages?beta=true",
        headers={**CLAUDE_CODE, "content-type": "application/json"},
        content=raw,
    )
    assert r.status_code == 200, r.text
    assert up.requests[0]["body"] == raw
