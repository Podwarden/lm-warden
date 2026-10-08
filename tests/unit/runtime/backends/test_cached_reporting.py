from app.runtime.backends import registry


def test_shipped_backends_report_cached_tokens_in_usage():
    for name in registry.available():
        assert registry.get(name).capabilities.cached_tokens_reporting == "usage"
