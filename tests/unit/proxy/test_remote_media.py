"""#324: remote media is refused by default on every proxy route."""

import pytest

from app.proxy import remote_media
from app.proxy.messages_schemas import MessagesRequest
from app.proxy.messages_translate import TranslationError as MsgTranslationError
from app.proxy.messages_translate import to_openai_request as messages_to_openai
from app.proxy.responses_schemas import ResponsesRequest
from app.proxy.responses_translate import TranslationError as RespTranslationError
from app.proxy.responses_translate import to_openai_request as responses_to_openai
from tests.unit.proxy.test_messages import _json_upstream, _post, _ready

REMOTE = "http://metadata.invalid/latest/meta-data/"
OK_BODY = {"id": "c1", "model": "qwen", "choices": [{"message": {"content": "ok"}}]}


def _chat_body(url, key="image_url"):
    part = {"type": key, key: {"url": url}}
    return {
        "model": "qwen",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, part]}],
    }


def test_anthropic_url_source_refused():
    req = MessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "url", "url": REMOTE}}],
                }
            ],
        }
    )
    with pytest.raises(MsgTranslationError, match="remote image URLs"):
        messages_to_openai(req)


def test_anthropic_data_url_source_accepted():
    url = "data:image/png;base64,AAA"
    req = MessagesRequest.model_validate(
        {
            "model": "qwen",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "url", "url": url}}],
                }
            ],
        }
    )
    out = messages_to_openai(req)
    assert out["messages"][0]["content"][0]["image_url"]["url"] == url


@pytest.mark.parametrize("url", [REMOTE, "file:///etc/passwd"])
def test_responses_remote_image_refused(url):
    req = ResponsesRequest.model_validate(
        {
            "model": "qwen",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": url}],
                }
            ],
        }
    )
    with pytest.raises(RespTranslationError, match="remote image URLs"):
        responses_to_openai(req)


def test_messages_route_answers_400(client, tmp_data_dir):
    _ready(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": REMOTE}}],
            }
        ],
    }
    r, send = _post(client, body, upstream=_json_upstream(OK_BODY))
    assert r.status_code == 400
    send.assert_not_called()


def test_responses_route_answers_400_openai_shape(client, tmp_data_dir):
    _ready(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "input": [{"role": "user", "content": [{"type": "input_image", "image_url": REMOTE}]}],
    }
    r, send = _post(client, body, upstream=_json_upstream(OK_BODY), path="/v1/responses")
    assert r.status_code == 400
    assert "error" in r.json()
    send.assert_not_called()


@pytest.mark.parametrize(
    "url,key",
    [
        (REMOTE, "image_url"),
        ("https://example.com/a.png", "image_url"),
        ("file:///etc/passwd", "image_url"),
        ("http://example.com/v.mp4", "video_url"),
        ("http://example.com/a.wav", "audio_url"),
    ],
)
def test_chat_completions_refuses_remote_media_by_default(
    client, tmp_data_dir, monkeypatch, url, key
):
    monkeypatch.delenv(remote_media.ENV_VAR, raising=False)
    _ready(client, tmp_data_dir)
    r, send = _post(
        client, _chat_body(url, key), upstream=_json_upstream(OK_BODY), path="/v1/chat/completions"
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    send.assert_not_called()  # nothing reached the engine


def test_chat_completions_passes_data_url(client, tmp_data_dir, monkeypatch):
    monkeypatch.delenv(remote_media.ENV_VAR, raising=False)
    _ready(client, tmp_data_dir)
    r, send = _post(
        client,
        _chat_body("data:image/png;base64,AAA"),
        upstream=_json_upstream(OK_BODY),
        path="/v1/chat/completions",
    )
    assert r.status_code == 200
    send.assert_called_once()


def test_chat_completions_allowlist(client, tmp_data_dir, monkeypatch):
    monkeypatch.setenv(remote_media.ENV_VAR, "Images.Example.com")
    _ready(client, tmp_data_dir)
    ok, send = _post(
        client,
        _chat_body("https://images.example.com/a.png"),
        upstream=_json_upstream(OK_BODY),
        path="/v1/chat/completions",
    )
    assert ok.status_code == 200
    send.assert_called_once()
    for bad in (
        "https://example.com/a.png",
        "https://images.example.com.evil.test/a.png",
        "https://images.example.com@evil.test/a.png",
        "file:///etc/passwd",
    ):
        r, send = _post(
            client, _chat_body(bad), upstream=_json_upstream(OK_BODY), path="/v1/chat/completions"
        )
        assert r.status_code == 400, bad
        send.assert_not_called()


def test_allowed_domains_parsing(monkeypatch):
    monkeypatch.setenv(remote_media.ENV_VAR, " A.com, b.com ,,a.com")
    assert remote_media.allowed_domains() == ["a.com", "b.com"]
    monkeypatch.setenv(remote_media.ENV_VAR, "")
    assert remote_media.allowed_domains() == []
