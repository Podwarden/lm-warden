"""POST /v1/messages and /v1/messages/count_tokens resolve model aliases the
way the OpenAI routes do: an enabled alias wins, the first loaded target
serves, the alias's settings (params, thinking policy, guard) apply, a
disabled alias is invisible and an alias with no loaded target is a 404.
Covered with the router off (legacy handler) and on (router data plane).
"""

import json
import sqlite3
from unittest.mock import patch

import pytest

from tests.unit.proxy.test_messages import AUTH, _json_upstream, _ready
from tests.unit.router.harness import enable_router, fake_anthropic, fake_engine

_BODY = {"model": "smart", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}


def _completion(model="qwen2"):
    return _json_upstream(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    )


def _engine():
    async def handler(_req):
        return _completion()

    return handler


def _setup(client, tmp_data_dir, *, alias_targets=("qwen2",), enabled=1, loaded2=True, **alias):
    _ready(client, tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen2','qwen2','r','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)",
            (json.dumps([0]), "loaded" if loaded2 else "pulled"),
        )
        c.execute(
            "INSERT INTO model_aliases(id, name, targets, enabled, guard, thinking_policy, params) "
            "VALUES ('a1','smart',?,?,?,?,?)",
            (
                json.dumps(list(alias_targets)),
                enabled,
                alias.get("guard", "inherit"),
                alias.get("thinking_policy", "inherit"),
                json.dumps(alias.get("params", {})),
            ),
        )
        c.commit()
    client.app.state.supervisor._ports["qwen2"] = 19098


def _router_on(client, tmp_data_dir, **kw):
    enable_router(client, tmp_data_dir, [], **kw)
    return fake_anthropic(client)


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
def test_alias_reaches_target_with_rewritten_name(tmp_data_dir, client, on, passthrough):
    _setup(client, tmp_data_dir)
    up = _router_on(client, tmp_data_dir, passthrough_unmatched=passthrough) if on else None
    with fake_engine(client, _engine()) as calls:
        r = client.post("/v1/messages", headers=AUTH, json=_BODY)
    assert r.status_code == 200, r.text
    if up is not None:
        assert not up.called
    (req,) = calls
    assert ":19098" in str(req.url)
    assert json.loads(req.content)["model"] == "qwen2"
    # the OpenAI path hands back the engine's own model name for an alias
    assert r.json()["model"] == "qwen2"


@pytest.mark.parametrize("on", [False, True])
def test_first_loaded_target_serves(tmp_data_dir, client, on):
    _setup(client, tmp_data_dir, alias_targets=("qwen", "qwen2"))
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as c:
        c.execute("UPDATE models SET status='pulled' WHERE id='qwen'")
        c.commit()
    if on:
        _router_on(client, tmp_data_dir)
    with fake_engine(client, _engine()) as calls:
        r = client.post("/v1/messages", headers=AUTH, json=_BODY)
    assert r.status_code == 200, r.text
    assert ":19098" in str(calls[0].url)


@pytest.mark.parametrize("on", [False, True])
def test_alias_params_and_thinking_policy_apply(tmp_data_dir, client, on):
    _setup(
        client,
        tmp_data_dir,
        thinking_policy="off",
        params={"temperature": {"value": 0.25, "mode": "force"}},
    )
    if on:
        _router_on(client, tmp_data_dir)
    with fake_engine(client, _engine()) as calls:
        r = client.post("/v1/messages", headers=AUTH, json={**_BODY, "temperature": 0.9})
    assert r.status_code == 200, r.text
    sent = json.loads(calls[0].content)
    assert sent["temperature"] == 0.25
    assert sent["chat_template_kwargs"]["enable_thinking"] is False


def _openai(client):
    with fake_engine(client, _engine()):
        return client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "smart", "messages": _BODY["messages"]},
        )


@pytest.mark.parametrize("on", [False, True])
def test_disabled_alias_same_error_as_openai(tmp_data_dir, client, on):
    _setup(client, tmp_data_dir, enabled=0)
    oai = _openai(client)
    if on:
        _router_on(client, tmp_data_dir, passthrough_unmatched=False)
    with fake_engine(client, _engine()) as calls:
        r = client.post("/v1/messages", headers=AUTH, json=_BODY)
    assert not calls
    assert r.status_code == oai.status_code == 404
    assert r.json()["error"]["message"] == oai.json()["detail"]


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
def test_alias_without_loaded_target_same_error_as_openai(tmp_data_dir, client, on, passthrough):
    _setup(client, tmp_data_dir, loaded2=False)
    oai = _openai(client)
    up = _router_on(client, tmp_data_dir, passthrough_unmatched=passthrough) if on else None
    with fake_engine(client, _engine()) as calls:
        r = client.post("/v1/messages", headers=AUTH, json=_BODY)
    assert not calls
    if up is not None:
        assert not up.called  # never relayed to Anthropic in the alias's place
    assert r.status_code == oai.status_code == 404
    assert r.json()["error"]["message"] == oai.json()["detail"]
    assert "no model behind alias 'smart' is loaded" in r.json()["error"]["message"]


@pytest.mark.parametrize("on", [False, True])
def test_token_allowed_by_target_served_name(tmp_data_dir, client, on):
    """D4d: a key that lists the serving model, not the alias, may use the alias."""
    _setup(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as c:
        c.execute("UPDATE api_tokens SET allowed_models=?", ("qwen2",))
        c.commit()
    if on:
        _router_on(client, tmp_data_dir)
    with fake_engine(client, _engine()):
        r = client.post("/v1/messages", headers=AUTH, json=_BODY)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("on", [False, True])
def test_count_tokens_resolves_alias(tmp_data_dir, client, on):
    _setup(client, tmp_data_dir)
    up = _router_on(client, tmp_data_dir) if on else None
    body = {"model": "smart", "messages": _BODY["messages"]}
    r = client.post("/v1/messages/count_tokens", headers=AUTH, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["input_tokens"] > 0
    if up is not None:
        assert not up.called
    # counted with the target's tokenizer
    assert client.app.state.tokenizers.count.call_args.args[0] == "r"


def test_router_guard_sees_the_alias(tmp_data_dir, client):
    """The router's own guard scan must apply the alias's guard like _forward does."""
    from app.guard import proxy as guard_proxy

    _setup(client, tmp_data_dir, guard="block")
    _router_on(client, tmp_data_dir)
    seen = []
    real = guard_proxy.guard_for

    def spy(platform, token, alias=None):
        seen.append(getattr(alias, "name", None))
        return real(platform, token, alias)

    with patch("app.router.data_plane.guard_proxy.guard_for", spy):
        with fake_engine(client, _engine()):
            client.post("/v1/messages", headers=AUTH, json=_BODY)
    assert "smart" in seen
