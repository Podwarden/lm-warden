from unittest.mock import MagicMock, patch


async def test_tokenizer_cache_returns_cached_instance():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok = MagicMock()
    fake_tok.encode = lambda s: list(s.encode())
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ) as load:
        cache = TokenizerCache()
        a = await cache.get("Qwen/Qwen3.5-9B")
        b = await cache.get("Qwen/Qwen3.5-9B")
    assert a is b
    assert load.call_count == 1
    load.assert_called_once_with("Qwen/Qwen3.5-9B", trust_remote_code=False)


async def test_count_tokens_uses_repo_tokenizer():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok = MagicMock()
    fake_tok.encode = lambda s: list(s)
    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=fake_tok):
        cache = TokenizerCache()
        n = await cache.count("Qwen/Qwen3.5-9B", "hello")
    assert n == 5


async def test_count_handles_empty_string():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok = MagicMock()
    fake_tok.encode = lambda s: []
    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=fake_tok):
        cache = TokenizerCache()
        assert await cache.count("Qwen/x", "") == 0


async def test_the_cache_key_is_the_repo_alone():
    """#264 — the key was ``(hf_repo, trust_remote_code)`` while the flag could
    vary. It no longer can (the cache never trusts remote code), so a second
    key component would only ever leave an unreachable slot behind."""
    from app.proxy.tokenizers import TokenizerCache
    fake = MagicMock()
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=fake,
    ) as load:
        cache = TokenizerCache()
        a = await cache.get("some/model")
        b = await cache.get("some/model")

    assert load.call_count == 1
    assert a is b
    assert list(cache._cache) == ["some/model"]
