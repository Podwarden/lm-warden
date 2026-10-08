"""#287 secrets + no-open-relay: nothing leaks, nothing relays without every condition."""

import json
import logging

import httpx
import pytest

from tests.conftest import bearer, seed_admin_token
from tests.unit.router.harness import (
    ANTHROPIC_TOKEN,
    CLAUDE_CODE,
    PLAINTEXT,
    USER_TEXT,
    chat_completion,
    decisions,
    enable_router,
    fake_anthropic,
    fake_engine,
    haiku_body,
    ready,
)

RULE = {"pattern": "claude-haiku*", "target": "qwen"}
FORBIDDEN = (ANTHROPIC_TOKEN, PLAINTEXT, "Bearer", USER_TEXT, "sk-ant")


async def _engine_502(_req):
    raise httpx.ConnectError("refused")


async def _engine_ok(_req):
    return chat_completion()


def _post(client, body, headers=CLAUDE_CODE, path="/v1/messages?beta=true"):
    return client.post(path, headers=headers, json=body)


def _err_message(r, status, kind):
    assert r.status_code == status, r.text
    assert r.json()["error"]["type"] == kind
    return r.json()["error"]["message"]


def test_nothing_secret_reaches_logs_stats_decisions_or_registry(tmp_data_dir, client, caplog):
    caplog.set_level(logging.DEBUG)
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [RULE])
    up = fake_anthropic(client)
    admin = "vwa_adminsecret1234567890abcdefabcdef"
    seed_admin_token(tmp_data_dir / "vllm-warden.db", plaintext=admin)
    with fake_engine(client, _engine_ok) as engine_calls:
        _post(client, haiku_body())  # case 1: local
        _post(client, haiku_body(model="claude-sonnet-4-5"))  # case 3: passthrough
    with fake_engine(client, _engine_502):
        _post(client, haiku_body())  # case 6: fallback
    # the engine never saw the Anthropic token; Anthropic never saw the warden key
    for req in engine_calls:
        assert ANTHROPIC_TOKEN not in str(req.headers) + req.content.decode()
        assert PLAINTEXT not in str(req.headers)
    for seen in up.requests:
        assert PLAINTEXT not in json.dumps(seen["headers"])
    dumps = [
        caplog.text,
        json.dumps(client.app.state.router.snapshot(None)),
        json.dumps(decisions(client)),
        client.get("/api/stats/requests", headers=bearer(admin)).text,
        json.dumps(client.app.state.router.decisions(200)),
    ]
    assert len(decisions(client)) == 3
    for dump in dumps:
        for needle in FORBIDDEN:
            assert needle not in dump, needle


def test_decision_ring_has_no_header_or_body_fields(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [RULE])
    fake_anthropic(client)
    _post(client, haiku_body(model="claude-sonnet-4-5"))
    (d,) = decisions(client)
    assert set(d) == {
        "ts", "path", "model_in", "model_out", "rule_id", "route", "reason",
        "status", "latency_ms", "ttfb_ms", "stream", "token_name",
    }  # fmt: skip


def test_vw_key_in_authorization_only_is_never_relayed(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [RULE])
    up = fake_anthropic(client)
    r = _post(
        client,
        haiku_body(model="claude-sonnet-4-5"),
        headers={"Authorization": f"Bearer {PLAINTEXT}"},
    )
    assert "no_upstream_credential" in _err_message(r, 403, "permission_error")
    assert not up.called
    d = decisions(client)[0]
    assert d["route"] == "refused" and d["reason"] == "no_upstream_credential"


def test_vw_key_in_both_places_is_never_relayed(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [RULE])
    up = fake_anthropic(client)
    r = _post(
        client,
        haiku_body(model="claude-sonnet-4-5"),
        headers={"X-LMWarden-Key": PLAINTEXT, "Authorization": f"Bearer {PLAINTEXT}"},
    )
    assert "no_upstream_credential" in _err_message(r, 403, "permission_error")
    r = _post(
        client,
        haiku_body(model="claude-sonnet-4-5"),
        headers={"X-LMWarden-Key": PLAINTEXT, "x-api-key": PLAINTEXT},
    )
    assert r.status_code == 403
    assert not up.called


def test_key_without_relay_flag_is_refused_for_passthrough_and_fallback(tmp_data_dir, client):
    ready(client, tmp_data_dir, relay=False)
    enable_router(client, tmp_data_dir, [RULE])
    up = fake_anthropic(client)
    r = _post(client, haiku_body(model="claude-sonnet-4-5"))
    assert "relay_not_allowed" in _err_message(r, 403, "permission_error")
    with fake_engine(client, _engine_502) as calls:
        r = _post(client, haiku_body())  # fallback rule, engine down
    assert "relay_not_allowed" in _err_message(r, 403, "permission_error")
    assert len(calls) == 1 and not up.called
    assert client.get("/v1/messages/batches", headers=CLAUDE_CODE).status_code == 403
    assert not up.called


def test_relay_flag_with_no_credential_headers_is_refused(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [RULE])
    up = fake_anthropic(client)
    r = _post(client, haiku_body(model="claude-sonnet-4-5"), headers={"X-LMWarden-Key": PLAINTEXT})
    assert "no_upstream_credential" in _err_message(r, 403, "permission_error")
    assert not up.called


def test_garbage_lmwarden_header_wins_over_valid_authorization(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [RULE])
    up = fake_anthropic(client)
    r = _post(
        client,
        haiku_body(),
        headers={"X-LMWarden-Key": "vw_garbage", "Authorization": f"Bearer {PLAINTEXT}"},
    )
    assert r.status_code == 401
    assert not up.called


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_lmwarden_header_authenticates_every_v1_route_and_never_reaches_engine(
    tmp_data_dir, client, path
):
    ready(client, tmp_data_dir)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "prompt": "hi"}
    with fake_engine(client, _engine_ok) as calls:
        r = client.post(path, headers=CLAUDE_CODE, json=body)
    assert r.status_code == 200, r.text
    (req,) = calls
    sent = {k.lower() for k in req.headers}
    assert not sent & {"x-lmwarden-key", "authorization", "x-api-key", "cookie"}


def test_cookie_never_reaches_the_engine_even_with_router_off(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    with fake_engine(client, _engine_ok) as calls:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {PLAINTEXT}", "Cookie": "vw_session=abc"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert "cookie" not in {k.lower() for k in calls[0].headers}


def test_router_off_never_relays(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    up = fake_anthropic(client)
    r = _post(client, haiku_body(model="claude-sonnet-4-5"))
    assert r.status_code == 404
    assert client.get("/v1/messages/batches", headers=CLAUDE_CODE).status_code == 404
    assert not up.called
