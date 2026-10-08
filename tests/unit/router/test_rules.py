"""Pure rule logic: pattern validation, matching, kv parsing (#287)."""

import pytest

from app.router.rules import (
    HEADER_NAME,
    RouterConfig,
    RuleSpec,
    config_from_kv,
    match_rule,
    validate_pattern,
    validate_upstream_url,
)


def _spec(pattern: str, *, id: str = "r", enabled: bool = True, position: int = 0) -> RuleSpec:
    return RuleSpec(
        id=id,
        position=position,
        pattern=pattern,
        target_model_id="m1",
        enabled=enabled,
        fallback=True,
        strip_thinking=True,
        min_max_tokens=0,
    )


def test_header_name():
    assert HEADER_NAME == "X-LMWarden-Key"


@pytest.mark.parametrize("p", ["claude-haiku*", "claude-3-5-haiku-20241022", "claude-?-sonnet*"])
def test_validate_pattern_accepts(p):
    assert validate_pattern(p) == p


@pytest.mark.parametrize("p", ["", "*", "**", "*?", "a[b]", "x" * 129, "a b", "a/b"])
def test_validate_pattern_refuses(p):
    with pytest.raises(ValueError, match="pattern"):
        validate_pattern(p)


def test_validate_pattern_max_length_ok():
    assert validate_pattern("x" * 128)


def test_first_enabled_match_wins_by_order():
    rules = [
        _spec("claude-*-x", id="a"),
        _spec("claude-haiku*", id="b"),
        _spec("claude-h*", id="c"),
    ]
    hit = match_rule("claude-haiku-4", rules)
    assert hit is not None and hit.id == "b"


def test_disabled_rule_skipped():
    rules = [_spec("claude-haiku*", id="a", enabled=False), _spec("claude-*", id="b")]
    hit = match_rule("claude-haiku-4", rules)
    assert hit is not None and hit.id == "b"


def test_match_is_case_sensitive():
    assert match_rule("Claude-Haiku-x", [_spec("claude-haiku*")]) is None


@pytest.mark.parametrize("model", [None, 123, ["claude-haiku"], {"a": 1}, b"claude-haiku-1"])
def test_non_string_model_never_matches(model):
    assert match_rule(model, [_spec("claude-haiku*")]) is None


def test_exact_pattern_matches_only_itself():
    rules = [_spec("claude-haiku-4")]
    assert match_rule("claude-haiku-4", rules) is not None
    assert match_rule("claude-haiku-4-1", rules) is None
    assert match_rule("claude-haiku-", rules) is None


def test_question_mark_is_one_char():
    rules = [_spec("claude-?-sonnet")]
    assert match_rule("claude-3-sonnet", rules) is not None
    assert match_rule("claude-33-sonnet", rules) is None


def test_config_defaults():
    assert config_from_kv({}) == RouterConfig()
    c = RouterConfig()
    assert (c.enabled, c.upstream_url, c.passthrough_unmatched) == (
        False,
        "https://api.anthropic.com",
        True,
    )
    assert (c.local_header_timeout_s, c.local_nonstream_timeout_s) == (60, 120)
    assert (c.breaker_threshold, c.breaker_open_s, c.max_body_mb) == (3, 60, 32)


def test_config_bad_value_falls_back_per_key():
    c = config_from_kv({"router_enabled": "true", "router_breaker_threshold": "x"})
    assert c.enabled is True
    assert c.breaker_threshold == 3


def test_config_parses_values_and_range_checks():
    c = config_from_kv(
        {
            "router_enabled": "true",
            "router_passthrough_unmatched": "false",
            "router_upstream_url": "https://proxy.example/anthropic",
            "router_local_header_timeout_s": "10",
            "router_breaker_open_s": "0",  # out of 1..3600 -> default
            "router_max_body_mb": "999",  # out of 1..256 -> default
        }
    )
    assert c.passthrough_unmatched is False
    assert c.upstream_url == "https://proxy.example/anthropic"
    assert c.local_header_timeout_s == 10
    assert c.breaker_open_s == 60
    assert c.max_body_mb == 32


def test_config_bad_url_falls_back():
    assert config_from_kv({"router_upstream_url": "http://proxy.example"}).upstream_url == (
        "https://api.anthropic.com"
    )


@pytest.mark.parametrize(
    "u", ["https://api.anthropic.com", "https://proxy.example/anthropic", "http://127.0.0.1:9"]
)
def test_upstream_url_accepts(u):
    assert validate_upstream_url(u) == u


@pytest.mark.parametrize(
    "u", ["http://proxy.example", "https://x/?q=1", "https://x/#f", "ftp://x", ""]
)
def test_upstream_url_refuses(u):
    with pytest.raises(ValueError):
        validate_upstream_url(u)
