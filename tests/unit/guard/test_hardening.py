"""Guard hardening after the 2026-10-30 live test: custom patterns that could hang the server are
refused, keywords work anywhere in a match, typos in overrides are refused, specific rules beat
generic ones, generic values are masked by length only, and restored placeholders are logged."""

from __future__ import annotations

import pytest

from app.guard.config import compile_guard, validate
from app.guard.proxy import SeenSet, mask_for, run_guard
from app.guard.scanner import scan_text
from tests.unit.guard.corpus import github_token


@pytest.mark.parametrize(
    "pattern",
    ["((a+))+$", "(a|aa)+$", "(?:a|a)*b", "(a{1,30}){1,30}$", "(a+){2,}"],
)
def test_patterns_that_can_take_exponential_time_are_refused(pattern):
    with pytest.raises(ValueError, match="exponential|too long"):
        validate({"rules": [{"id": "r", "type": "regex", "regex": pattern}]})


@pytest.mark.parametrize(
    "pattern",
    ["corp-[a-z0-9]{8}\\.internal", "CORP-[A-Z0-9]{24}", "ab(?:cd|ef)gh", "[a-z]+@corp\\.example"],
)
def test_flat_patterns_are_accepted(pattern):
    validate({"rules": [{"id": "r", "type": "regex", "regex": pattern}]})


def test_a_custom_keyword_can_be_anywhere_in_the_match():
    g = compile_guard(
        {
            "rules": [
                {
                    "id": "host",
                    "type": "regex",
                    "regex": "corp-[a-z0-9]{8}\\.internal",
                    "keywords": ["internal"],
                }
            ]
        }
    )
    assert [f.rule_id for f in scan_text("db at corp-a1b2c3d4.internal:5432", g.rules)] == ["host"]


@pytest.mark.parametrize("key", ["no-such-rule", "Cloud", "cloud-api-key"])
def test_an_override_for_an_unknown_pack_or_rule_is_refused(key):
    with pytest.raises(ValueError, match="unknown"):
        validate({"overrides": {key: "block"}})


def test_overrides_for_packs_builtin_and_custom_rules_are_accepted():
    validate(
        {
            "overrides": {"private-keys": "block", "aws-access-key": "sanitize", "mine": "block"},
            "rules": [{"id": "mine", "type": "fixed", "values": ["halcyon"]}],
        }
    )


def test_a_specific_rule_beats_a_generic_one_on_the_same_text():
    tok = github_token()
    g = compile_guard({"mode": "sanitize"})
    found = scan_text(f"git clone https://x-access-token:{tok}@github.com/acme/app.git", g.rules)
    assert [(f.rule_id, f.value) for f in found] == [("github-token", tok)]


def test_generic_values_are_masked_by_length_only():
    assert mask_for("Zx9qLm2vB7nR4tKw", "password-assignment") == "(16 chars)"
    assert mask_for("AKIAQWERTYUIOPASDFGH", "aws-access-key").startswith("AKIA…")


def test_restored_placeholders_are_logged_without_values():
    from app.guard.placeholders import RestoreMap

    tok = github_token()
    g = compile_guard({"mode": "sanitize", "packs": ["cloud-api-keys"]})
    out = run_guard(
        {"messages": [{"role": "user", "content": f"use {tok}"}]},
        "openai",
        g,
        token_id="t",
        secret=b"k" * 32,
        session="s",
        seen=SeenSet(),
    )
    hx = out.findings[0].hex
    text, n = out.restore.restore(f"run ⟦LMW-SECRET-{hx}⟧ twice ⟦LMW-SECRET-{hx}⟧")
    assert n == 2 and text.count(tok) == 2
    events = out.restored_events(now=1.0)
    assert [(e.action, e.rule_id, e.ph, e.length) for e in events] == [
        ("restored", "github-token", hx, len(tok))
    ]
    assert tok not in repr(events)
    assert isinstance(out.restore, RestoreMap)
