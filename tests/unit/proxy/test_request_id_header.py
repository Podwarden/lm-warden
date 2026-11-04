"""Every proxied response carries X-LMW-Request-Id = the history row's id (task 4)."""

import re
from unittest.mock import patch

from app.proxy.request_id import REQUEST_ID_HEADER
from tests.unit.proxy.test_cache_estimate_forward import _live_rows, _spy_registry
from tests.unit.proxy.test_cobs2_forward import BODY
from tests.unit.proxy.test_dp_routing_forward import AUTH, _post, _ready, _resp, _sender, _sse_resp

HEX32 = re.compile(r"[0-9a-f]{32}")


def _check(client, r):
    rid = r.headers[REQUEST_ID_HEADER]
    assert HEX32.fullmatch(rid)
    assert _live_rows(client)[-1].id == rid


def test_chat_nonstream_carries_the_history_id(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    r, _ = _post(client, BODY)
    _check(client, r)


def test_chat_stream_carries_the_history_id(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    frames = [
        b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    send, _ = _sender(_sse_resp(frames))
    with patch("httpx.AsyncClient.send", new=send):
        with client.stream(
            "POST", "/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True}
        ) as r:
            b"".join(r.iter_bytes())
            _check(client, r)


def test_runaway_reaggregated_path_carries_it(tmp_data_dir, client):
    from tests.unit.proxy import test_runaway_forward as rf

    plaintext = rf._prep(client, tmp_data_dir, runaway_mode="log")
    _spy_registry(client)
    chunks = [
        rf._chat_chunk("hel", role="assistant"),
        rf._chat_chunk("lo", finish="stop"),
        b"data: [DONE]\n\n",
    ]
    send, _ = rf._capturing_send(rf._fake_stream_resp(chunks))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": f"Bearer {plaintext}"}, json=BODY
        )
    assert r.status_code == 200
    _check(client, r)


def test_engine_error_status_carries_it_too(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    bad = _resp({"error": {"message": "bad"}})
    bad.status_code = 400
    r, _ = _post(client, BODY, resp=bad)
    assert r.status_code == 400
    _check(client, r)


def _completion():
    return {
        "id": "c",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_messages_translated_path_carries_it(tmp_data_dir, client):
    from tests.unit.proxy import test_messages as tm

    tm._ready(client, tmp_data_dir)
    _spy_registry(client)
    r, _ = tm._post(
        client,
        {"model": "qwen", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        upstream=tm._json_upstream(_completion()),
    )
    assert r.status_code == 200
    _check(client, r)


def test_responses_translated_path_carries_it(tmp_data_dir, client):
    from tests.unit.proxy import test_messages as tm

    tm._ready(client, tmp_data_dir)
    _spy_registry(client)
    r, _ = tm._post(
        client,
        {"model": "qwen", "input": "hi"},
        upstream=tm._json_upstream(_completion()),
        path="/v1/responses",
    )
    assert r.status_code == 200
    _check(client, r)


def test_router_local_leg_stream_carries_it(tmp_data_dir, client):
    from tests.unit.router import test_data_plane as dp
    from tests.unit.router.harness import CLAUDE_CODE, fake_engine, good_stream, haiku_body

    dp._setup(client, tmp_data_dir)
    _spy_registry(client)
    with fake_engine(client, dp._engine_returning(good_stream)):
        with client.stream(
            "POST", "/v1/messages", headers=CLAUDE_CODE, json=haiku_body(stream=True)
        ) as r:
            b"".join(r.iter_bytes())
            _check(client, r)
