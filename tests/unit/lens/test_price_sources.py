"""The two public price lists: parsing and the guarded fetch (#347)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.lens import price_sources as src

OPENROUTER = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.5",
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "input_cache_read": "0.0000003",
                "input_cache_write": "0.00000375",
            },
        },
        {"id": "openai/gpt-5:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "x/no-pricing"},
        "junk",
    ]
}
LITELLM = {
    "sample_spec": {"litellm_provider": "x"},
    "claude-sonnet-4-5-20250929": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
        "cache_creation_input_token_cost": 3.75e-06,
    },
    "gemini/gemini-2.5-pro": {
        "litellm_provider": "gemini",
        "mode": "chat",
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 1e-05,
    },
    "bedrock/claude": {"litellm_provider": "bedrock", "input_cost_per_token": 1e-06},
    "dall-e": {"litellm_provider": "openai", "mode": "image_generation", "input_cost_per_token": 1},
}


def test_model_key_folds_dates_and_dots():
    assert src.model_key("claude-sonnet-4-5-20250929") == src.model_key("claude-sonnet-4.5")
    assert src.model_key("Claude-Sonnet-4.5") == "claude-sonnet-4-5"
    assert src.model_key("gpt-4o-2024-08-06") == "gpt-4o"


def test_parse_openrouter_per_token_to_micros_per_million():
    rows = {(r.host, r.model_key): r for r in src.parse_openrouter(OPENROUTER, "T")}
    r = rows[("openrouter.ai", "anthropic/claude-sonnet-4-5")]
    assert (r.input_per_m_micros, r.output_per_m_micros) == (3_000_000, 15_000_000)
    assert (r.cache_read_per_m_micros, r.cache_write_per_m_micros) == (300_000, 3_750_000)
    assert r.source == "openrouter" and r.fetched_at == "T"
    # The vendor's own host gets it too, under the vendor's model name.
    v = rows[("api.anthropic.com", "claude-sonnet-4-5")]
    assert v.model == "claude-sonnet-4.5" and v.input_per_m_micros == 3_000_000
    # A free variant stays under OpenRouter only; a variable price (-1) and no price are skipped.
    assert ("openrouter.ai", "openai/gpt-5:free") in rows
    assert not any(h == "api.openai.com" for h, _ in rows)
    assert ("openrouter.ai", "openrouter/auto") not in rows
    assert len(rows) == 3


def test_parse_openrouter_rejects_a_changed_shape():
    with pytest.raises(ValueError):
        src.parse_openrouter({"models": []})


def test_parse_litellm_keeps_vendor_chat_models_only():
    rows = {(r.host, r.model_key): r for r in src.parse_litellm(LITELLM, "T")}
    r = rows[("api.anthropic.com", "claude-sonnet-4-5")]
    assert r.model == "claude-sonnet-4-5-20250929"
    assert (r.input_per_m_micros, r.output_per_m_micros) == (3_000_000, 15_000_000)
    assert (r.cache_read_per_m_micros, r.cache_write_per_m_micros) == (300_000, 3_750_000)
    g = rows[("generativelanguage.googleapis.com", "gemini-2-5-pro")]
    assert g.cache_read_per_m_micros is None and g.source == "litellm"
    assert len(rows) == 2  # not sample_spec, not bedrock, not an image model


def _transport(handler):
    return httpx.MockTransport(handler)


async def test_fetch_json_refuses_other_hosts_and_plain_http():
    for url in ("https://example.com/x.json", "http://openrouter.ai/api/v1/models"):
        with pytest.raises(ValueError):
            await src.fetch_json(url)


async def test_fetch_json_sends_no_credentials_and_does_not_follow_redirects():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(302, headers={"location": "https://evil.example/x"})

    with pytest.raises(RuntimeError, match="HTTP 302"):
        await src.fetch_json(src.OPENROUTER_URL, transport=_transport(handler))
    h = seen["headers"]
    assert "authorization" not in h and "cookie" not in h


async def test_fetch_json_caps_the_size(monkeypatch):
    monkeypatch.setattr(src, "MAX_BYTES", 100)
    big = httpx.Response(200, content=b"[" + b"1," * 200 + b"1]")
    with pytest.raises(RuntimeError, match="over"):
        await src.fetch_json(src.OPENROUTER_URL, transport=_transport(lambda r: big))


def _many(n):
    return {
        "data": [
            {"id": f"anthropic/m{i}", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
            for i in range(n)
        ]
    }


async def test_fetch_rows_primary_then_fallback():
    def ok(request):
        assert request.url.host == "openrouter.ai"
        return httpx.Response(200, json=_many(30))

    rows, source, note = await src.fetch_rows(transport=_transport(ok))
    assert source == "openrouter" and note is None and len(rows) >= 30

    litellm = {
        f"gpt-{i}": {"litellm_provider": "openai", "input_cost_per_token": 1e-6} for i in range(30)
    }

    def fallback(request):
        if request.url.host == "openrouter.ai":
            return httpx.Response(503)
        assert request.url.host == "raw.githubusercontent.com"
        return httpx.Response(200, content=json.dumps(litellm).encode())

    rows, source, note = await src.fetch_rows(transport=_transport(fallback))
    assert source == "litellm" and "openrouter" in (note or "") and len(rows) == 30

    with pytest.raises(RuntimeError):
        await src.fetch_rows(transport=_transport(lambda r: httpx.Response(500)))
