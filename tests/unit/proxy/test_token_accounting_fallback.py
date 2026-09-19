"""Token accounting must not 500 a request that the engine could have served.

A GGUF-only repo -- the exact shape of the model this whole sub-project exists
to serve -- has no tokenizer.json, so AutoTokenizer.from_pretrained raises. That
call sits on the hot path at app/proxy/routes.py:418 with no try/except, so
today every request to such a model fails before it reaches the engine.

Two layers, tested separately because they fail for different reasons:
tokenizer_repo is the CORRECT answer and gives exact counts; the character
estimate is the SAFETY NET for when even that is unavailable. The estimate is
deliberately visible -- it is logged once per repo and reported through the
cache -- because it feeds the per-token usage figures, and silently inaccurate
billing is worse than loudly approximate billing.
"""

from __future__ import annotations

import logging

import pytest

from app.proxy.tokenizers import TokenizerCache

pytestmark = pytest.mark.asyncio


class _Boom:
    def __init__(self, *a, **k):
        raise OSError("does not appear to have a file named tokenizer.json")


def _break_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "app.proxy.tokenizers.AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(_Boom)}),
    )


async def test_falls_back_to_an_estimate_when_the_tokenizer_will_not_load(monkeypatch):
    _break_tokenizer(monkeypatch)
    cache = TokenizerCache()
    n = await cache.count("org/gguf-only", "hello world", trust_remote_code=False)
    assert n > 0


async def test_the_estimate_is_proportional_to_length(monkeypatch):
    _break_tokenizer(monkeypatch)
    cache = TokenizerCache()
    short = await cache.count("org/gguf-only", "x" * 40, trust_remote_code=False)
    long = await cache.count("org/gguf-only", "x" * 400, trust_remote_code=False)
    assert long > short


async def test_the_fallback_is_logged_once_per_repo(monkeypatch, caplog):
    """Once, not per request: a hot path that logs every call turns a
    degradation into a second incident."""
    _break_tokenizer(monkeypatch)
    cache = TokenizerCache()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            await cache.count("org/gguf-only", "hi", trust_remote_code=False)
    assert sum("estimat" in r.message.lower() for r in caplog.records) == 1


async def test_the_estimating_set_names_the_model_repo_not_the_tried_repo(monkeypatch):
    """An operator reading this is looking for the row to fix, and the row is
    keyed by hf_repo. Reporting the repo we happened to TRY would name a value
    they may not have set."""
    _break_tokenizer(monkeypatch)
    cache = TokenizerCache()
    await cache.count(
        "org/gguf-only", "hi", trust_remote_code=False, fallback_repo="org/also-broken"
    )
    assert cache.estimating() == frozenset({"org/gguf-only"})


async def test_fallback_repo_is_preferred_over_the_gguf_repo(monkeypatch):
    """tokenizer_repo is the CORRECT answer, not a fallback-of-a-fallback: when
    it is set we must never even try the GGUF-only repo, because that attempt is
    a guaranteed miss and, on a cold cache, a network round trip."""
    seen = []

    class _Tok:
        @staticmethod
        def from_pretrained(repo, **k):
            seen.append(repo)
            if repo == "org/gguf-only":
                raise OSError("no tokenizer here")
            return type(
                "t", (), {"encode": staticmethod(lambda s: [0] * len(s.split()))}
            )()

    monkeypatch.setattr("app.proxy.tokenizers.AutoTokenizer", _Tok)
    cache = TokenizerCache()
    n = await cache.count(
        "org/gguf-only", "a b c", trust_remote_code=False, fallback_repo="org/safetensors"
    )
    assert seen == ["org/safetensors"]
    assert n == 3


async def test_empty_text_is_still_zero_without_loading_anything(monkeypatch):
    _break_tokenizer(monkeypatch)
    assert await TokenizerCache().count("org/x", "", trust_remote_code=False) == 0


async def test_evict_clears_the_fallback_marker(monkeypatch):
    """Otherwise a model whose tokenizer_repo the operator has just fixed keeps
    estimating until the process restarts."""
    _break_tokenizer(monkeypatch)
    cache = TokenizerCache()
    await cache.count("org/gguf-only", "hi", trust_remote_code=False)
    await cache.evict("org/gguf-only")
    assert "org/gguf-only" not in cache.estimating()


async def test_a_working_tokenizer_never_enters_the_estimating_set(monkeypatch):
    class _Tok:
        @staticmethod
        def from_pretrained(repo, **k):
            return type(
                "t", (), {"encode": staticmethod(lambda s: [0] * len(s.split()))}
            )()

    monkeypatch.setattr("app.proxy.tokenizers.AutoTokenizer", _Tok)
    cache = TokenizerCache()
    assert await cache.count("org/fine", "a b", trust_remote_code=False) == 2
    assert cache.estimating() == frozenset()
