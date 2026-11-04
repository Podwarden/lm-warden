"""Guard round 2 (second live test, 2026-10-30): the pattern check judged by structure and by a
timed probe, the remaining misses, and the cosmetics."""

from __future__ import annotations

import pytest

from app.guard.config import compile_guard, validate
from app.guard.proxy import mask_for
from app.guard.scanner import scan_text

DEFAULT = compile_guard({"mode": "sanitize"})
PII = compile_guard({"mode": "sanitize", "packs": ["pii-contact"]})


def _rule(p):
    return {"rules": [{"id": "r", "type": "regex", "regex": p}]}


@pytest.mark.parametrize(
    "pattern",
    [
        "\\s*\\s*\\s*\\s*\\s*$",
        "((a+))+$",
        "(a|aa)+$",
        "(a{1,30}){1,30}$",
        "a*a*a*a*b",
        "(\\w+\\s?)*$",
    ],
)
def test_slow_patterns_are_refused(pattern):
    with pytest.raises(ValueError):
        validate(_rule(pattern))


@pytest.mark.parametrize(
    "pattern",
    [
        "(?:[a-z]+\\.)+internal",
        "[a-z]+(?:-[a-z0-9]+)*",
        "corp-[a-z0-9]{8}\\.internal",
        "CORP-[A-Z0-9]{24}",
        "[a-z0-9._%+-]+@corp\\.example\\.com",
    ],
)
def test_useful_patterns_with_a_clear_separator_are_accepted(pattern):
    validate(_rule(pattern))


@pytest.mark.parametrize(
    "text,value",
    [
        ("password: hunterqxzlbv", "hunterqxzlbv"),
        ('"password": "zqhvbnmwpx"', "zqhvbnmwpx"),
        ("db.example.com:5432:app:appuser:Kq7mWp2xLv9z", "Kq7mWp2xLv9z"),
        ("AWS_SECURITY_TOKEN=FQoGZXIvYXdzEJr7k2mWq9xLp4vZ8nR", "FQoGZXIvYXdzEJr7k2mWq9xLp4vZ8nR"),
        ("INTERNAL_TOKEN_PROD=k8Jq2mXw9Lp4vZ7nRt3Y", "k8Jq2mXw9Lp4vZ7nRt3Y"),
    ],
)
def test_remaining_misses_are_caught(text, value):
    assert value in [f.value for f in scan_text(text, DEFAULT.rules)]


@pytest.mark.parametrize(
    "text",
    [
        "const password: string = getPassword();",
        "passwordField: inputRef",
        "WEB_ENDPOINT_TOKEN_URL=https://x.example/t",
    ],
)
def test_still_not_secrets(text):
    assert scan_text(text, DEFAULT.rules) == []


def test_a_phone_written_with_spaces_is_caught():
    assert [f.rule_id for f in scan_text("call +1 415 555 0134", PII.rules)] == ["phone"]


def test_masks_show_a_public_prefix_or_only_the_length():
    assert mask_for("ZGVwbG95OkhqNyNrTDltUTJwVw==", "docker-auth") == "(28 chars)"
    assert mask_for("a" * 64, "hex-private-key") == "(64 chars)"
    assert mask_for("AKIAQWERTYUIOPASDFGH", "aws-access-key").startswith("AKIA…")


def test_a_quadratic_custom_pattern_stays_fast_on_a_megabyte_line():
    import time

    g = compile_guard(
        {"rules": [{"id": "host", "type": "regex", "regex": "(?:[a-z]+\\.)+internal"}]}
    )
    text = "a" * 1_000_000 + " db.corp.internal"
    t = time.perf_counter()
    found = scan_text(text, g.rules)
    assert time.perf_counter() - t < 10
    assert [f.value for f in found][-1:] == ["db.corp.internal"]
