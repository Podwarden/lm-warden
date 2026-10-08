"""#287: with the router disabled (the default) behaviour is today's, byte for byte."""

import json
from unittest.mock import patch

from app.router.state import RouterState
from tests.unit.router.harness import (
    PLAINTEXT,
    chat_completion,
    fake_anthropic,
    fake_engine,
    ready,
)

AUTH = {"Authorization": f"Bearer {PLAINTEXT}"}
BODY = {
    "model": "qwen",
    "max_tokens": 256,
    "temperature": 0.2,
    "stop_sequences": ["###"],
    "messages": [{"role": "user", "content": "hi"}],
}


async def _ok(_req):
    return chat_completion("Hello there")


def test_engine_request_and_client_response_match_today(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    with fake_engine(client, _ok) as calls:
        r = client.post("/v1/messages", headers=AUTH, json=BODY)
    assert r.status_code == 200, r.text
    (req,) = calls
    sent = json.loads(req.content)
    # literal expectation captured from tests/unit/proxy/test_messages.py before task 3
    assert sent == {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "stop": ["###"],
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": -5,
    }
    msg = r.json()
    assert msg["model"] == "qwen" and msg["content"] == [{"type": "text", "text": "Hello there"}]
    assert (msg["usage"]["input_tokens"], msg["usage"]["output_tokens"]) == (12, 3)


def test_ruleset_loaded_once_and_no_router_db_reads_on_the_hot_path(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    loads = []
    real = RouterState._load

    async def spy(self, db_path):
        loads.append(1)
        return await real(self, db_path)

    with patch.object(RouterState, "_load", spy), fake_engine(client, _ok):
        for _ in range(3):
            assert client.post("/v1/messages", headers=AUTH, json=BODY).status_code == 200
        assert (
            client.post(
                "/v1/messages/count_tokens",
                headers=AUTH,
                json={"model": "qwen", "messages": BODY["messages"]},
            ).status_code
            == 200
        )
    assert len(loads) <= 1


def test_router_code_opens_no_db_connection_on_the_second_request(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    with fake_engine(client, _ok):
        client.post("/v1/messages", headers=AUTH, json=BODY)  # boot load happens here
        with patch("app.router.state.open_db", side_effect=AssertionError("router touched DB")):
            assert client.post("/v1/messages", headers=AUTH, json=BODY).status_code == 200


def test_disabled_router_ignores_lookalike_rules_and_headers(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = fake_anthropic(client)
    r = client.post("/v1/messages", headers=AUTH, json={**BODY, "model": "claude-haiku-4-5"})
    assert r.status_code == 404 and "not loaded" in r.json()["error"]["message"]
    assert not up.called


def test_unknown_v1_path_is_the_plain_404(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r = client.get("/v1/anything", headers=AUTH)
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}


def test_wrong_method_on_a_warden_route_is_still_405(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r = client.post("/v1/models", headers=AUTH)
    assert r.status_code == 405 and r.json() == {"detail": "Method Not Allowed"}


def test_trailing_slash_still_redirects(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    r = client.post("/v1/messages/", headers=AUTH, json=BODY, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].endswith("/v1/messages")
