from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.runtime.warmup_probe import warmup_probe


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict):
        self.status_code = status_code
        self._json = json_body
        self.text = str(json_body)

    def json(self):
        return self._json


@pytest.mark.asyncio
async def test_probe_succeeds_on_200_with_choices():
    fake = _FakeResponse(200, {"choices": [{"text": " "}]})
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake)):
        result = await warmup_probe(port=10001, served_model_name="m1", timeout_s=5.0)
    assert result.ok is True
    assert result.detail is None


@pytest.mark.asyncio
async def test_probe_fails_on_5xx():
    fake = _FakeResponse(503, {"error": "engine warming"})
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake)):
        result = await warmup_probe(port=10001, served_model_name="m1", timeout_s=5.0)
    assert result.ok is False
    assert "503" in result.detail


@pytest.mark.asyncio
async def test_probe_fails_on_200_without_choices():
    fake = _FakeResponse(200, {"unexpected": "shape"})
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake)):
        result = await warmup_probe(port=10001, served_model_name="m1", timeout_s=5.0)
    assert result.ok is False
    assert "choices" in result.detail


@pytest.mark.asyncio
async def test_probe_fails_on_timeout():
    with patch(
        "httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.ReadTimeout("timed out")),
    ):
        result = await warmup_probe(port=10001, served_model_name="m1", timeout_s=0.1)
    assert result.ok is False
    assert "timeout" in result.detail.lower()


@pytest.mark.asyncio
async def test_probe_fails_on_connection_error():
    with patch(
        "httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        result = await warmup_probe(port=10001, served_model_name="m1", timeout_s=5.0)
    assert result.ok is False
    assert "connect" in result.detail.lower()


@pytest.mark.asyncio
async def test_probe_sends_max_tokens_1():
    fake = _FakeResponse(200, {"choices": [{"text": " "}]})
    captured = {}

    async def fake_post(self, url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return fake

    with patch("httpx.AsyncClient.post", new=fake_post):
        await warmup_probe(port=10001, served_model_name="m1", timeout_s=5.0)
    assert captured["json"]["max_tokens"] == 1
    assert captured["json"]["model"] == "m1"
    assert captured["json"]["stream"] is False
    assert captured["url"].endswith("/v1/completions")
    assert "127.0.0.1:10001" in captured["url"]
