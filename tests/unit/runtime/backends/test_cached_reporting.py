from app.runtime.backends import registry


def test_shipped_backends_report_cached_tokens_in_usage():
    for name in registry.available():
        assert registry.get(name).capabilities.cached_tokens_reporting == "usage"


def test_only_vllm_honours_cache_salt():
    # #300: vLLM folds cache_salt into its first block hash; llama.cpp b10731
    # never reads the field, so its slots reuse a prefix across salts
    assert registry.get("vllm").capabilities.honours_cache_salt is True
    assert registry.get("llamacpp").capabilities.honours_cache_salt is False
