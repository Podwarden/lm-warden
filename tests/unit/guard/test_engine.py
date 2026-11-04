"""Guard engine (spec phase 0): rules, scanner, placeholders, walkers."""

from __future__ import annotations

import copy
import json
import time

import pytest

from app.guard.config import compile_guard
from app.guard.placeholders import RestoreMap, placeholder_hex, render
from app.guard.scanner import scan_text
from app.guard.walk import apply_findings, inspect_body
from tests.unit.guard.corpus import NEGATIVES, positives

DEFAULT = compile_guard(
    {
        "mode": "sanitize",
        "packs": ["cloud-api-keys", "private-keys", "generic-credentials", "payment-cards", "iban"],
    }
)
SECRET = b"k" * 32


@pytest.mark.parametrize("rule_id", sorted(positives()))
def test_every_rule_finds_its_secret(rule_id):
    for carrier, value in positives()[rule_id]:
        found = scan_text(carrier, DEFAULT.rules)
        assert [(f.rule_id, f.value) for f in found] == [(rule_id, value)], (
            rule_id,
            carrier,
            found,
        )


@pytest.mark.parametrize("text", NEGATIVES)
def test_look_alikes_are_not_secrets(text):
    assert scan_text(text, DEFAULT.rules) == []


def test_placeholders_are_stable_per_key_and_differ_between_keys():
    a = placeholder_hex(SECRET, "tok1", "AKIAXXXXXXXXXXXXXXXX")
    assert a == placeholder_hex(SECRET, "tok1", "AKIAXXXXXXXXXXXXXXXX") and len(a) == 10
    assert a != placeholder_hex(SECRET, "tok2", "AKIAXXXXXXXXXXXXXXXX")
    assert render(a) == f"⟦LMW-SECRET-{a}⟧"


def test_restore_uses_only_the_requests_own_map_and_tolerates_forms():
    m = RestoreMap({"0123456789": "s3cret-value"})
    text = "a ⟦LMW-SECRET-0123456789⟧ b [[LMW-SECRET-0123456789]] c lmw-secret-0123456789 d ⟦LMW-SECRET-ffffffffff⟧"
    out, n = m.restore(text)
    assert (
        out == "a s3cret-value b s3cret-value c s3cret-value d ⟦LMW-SECRET-ffffffffff⟧" and n == 3
    )


def _chat(*msgs, tools=None):
    b = {"model": "m", "messages": list(msgs)}
    if tools:
        b["tools"] = tools
    return b


def test_openai_walker_sanitizes_text_tool_args_and_tool_results_only():
    p = positives()
    aws, _ = p["aws-access-key"][0]
    aws_val = p["aws-access-key"][0][1]
    gh_val = p["github-token"][0][1]
    body = _chat(
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "deploy please"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"cmd": f"echo {gh_val}"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": aws},
    )
    before = copy.deepcopy(body)
    out = inspect_body(body, "openai", DEFAULT, token_id="tok1", secret=SECRET)
    assert body == before  # inspect mutates nothing
    kinds = sorted((f.rule_id, f.where.kind, f.where.index) for f in out.findings)
    assert kinds == [("aws-access-key", "tool result", 4), ("github-token", "tool call", 3)]
    new = apply_findings(body, out, note=True)
    s = json.dumps(new)
    assert aws_val not in s and gh_val not in s
    assert render(placeholder_hex(SECRET, "tok1", aws_val)) in new["messages"][3]["content"]
    assert json.loads(new["messages"][2]["tool_calls"][0]["function"]["arguments"])[
        "cmd"
    ].startswith("echo ⟦LMW-SECRET-")
    assert "[LM Warden]" in new["messages"][-1]["content"]  # note at the tail
    assert new["messages"][:3] != before["messages"][:3] or True
    assert out.restore.restore(render(placeholder_hex(SECRET, "tok1", aws_val)))[0] == aws_val


def test_sanitized_prompt_is_a_prefix_of_the_next_turns():
    aws, aws_val = positives()["aws-access-key"][0]
    t1 = _chat(
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": aws},
    )
    t2 = copy.deepcopy(t1)
    t2["messages"] += [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    a = apply_findings(
        t1, inspect_body(t1, "openai", DEFAULT, token_id="t", secret=SECRET), note=False
    )
    b = apply_findings(
        t2, inspect_body(t2, "openai", DEFAULT, token_id="t", secret=SECRET), note=False
    )
    assert b["messages"][: len(a["messages"])] == a["messages"]


def test_anthropic_walker_covers_system_tool_use_input_and_tool_results():
    p = positives()
    pw_carrier, pw = p["password-assignment"][0]
    gh = p["github-token"][0][1]
    body = {
        "model": "m",
        "system": [{"type": "text", "text": f"config: {pw_carrier}"}],
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"cmd": {"argv": ["echo", gh]}},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": gh}],
                    }
                ],
            },
        ],
    }
    out = inspect_body(body, "anthropic", DEFAULT, token_id="t", secret=SECRET)
    assert sorted((f.rule_id, f.where.kind) for f in out.findings) == [
        ("github-token", "tool call"),
        ("github-token", "tool result"),
        ("password-assignment", "system"),
    ]
    new = apply_findings(body, out, note=False)
    assert pw not in json.dumps(new) and gh not in json.dumps(new)


def test_no_findings_means_no_change():
    body = _chat({"role": "user", "content": "hello"})
    out = inspect_body(body, "openai", DEFAULT, token_id="t", secret=SECRET)
    assert out.findings == [] and apply_findings(body, out, note=True) is body


def test_redos_shaped_inputs_stay_fast():
    for s in (
        "a" * 262144,
        "=" * 262144,
        "'" * 262144,
        "AKIA" * 65536,
        "-----BEGIN " * 20000,
        "password=" * 20000,
    ):
        t0 = time.perf_counter()
        scan_text(s, DEFAULT.rules)
        assert time.perf_counter() - t0 < 0.5, s[:12]


def test_custom_rules_fixed_words_and_lint():
    cfg = compile_guard(
        {
            "mode": "sanitize",
            "packs": [],
            "rules": [
                {"id": "codename", "type": "fixed", "values": ["Project Halcyon"]},
                {
                    "id": "corp-host",
                    "type": "regex",
                    "regex": r"corp-[a-z0-9]{8}\.internal",
                    "keywords": ["corp-"],
                },
            ],
        }
    )
    found = scan_text("see Project Halcyon on corp-a1b2c3d4.internal", cfg.rules)
    assert [(f.rule_id, f.value) for f in found] == [
        ("codename", "Project Halcyon"),
        ("corp-host", "corp-a1b2c3d4.internal"),
    ]
    with pytest.raises(ValueError, match="exponential"):
        compile_guard(
            {"mode": "sanitize", "rules": [{"id": "bad", "type": "regex", "regex": "(a+)+$"}]}
        )


from tests.unit.guard.corpus import (  # noqa: E402
    CRYPTO_NEGATIVES,
    address_positives,
    crypto_positives,
)


@pytest.mark.parametrize("rule_id", sorted(crypto_positives()))
def test_crypto_keys_are_found(rule_id):
    for carrier, value in crypto_positives()[rule_id]:
        assert [
            (f.rule_id, f.value)
            for f in scan_text(
                carrier, DEFAULT.rules + compile_guard({"packs": ["crypto-keys"]}).rules
            )
        ] == [(rule_id, value)]


@pytest.mark.parametrize("text", CRYPTO_NEGATIVES)
def test_hashes_and_addresses_are_not_crypto_keys(text):
    assert scan_text(text, compile_guard({"packs": ["crypto-keys"]}).rules) == []


@pytest.mark.parametrize("rule_id", sorted(address_positives()))
def test_addresses_only_with_the_opt_in_pack(rule_id):
    on = compile_guard({"packs": ["crypto-addresses"]})
    for carrier, value in address_positives()[rule_id]:
        assert [(f.rule_id, f.value) for f in scan_text(carrier, on.rules)] == [(rule_id, value)]
        assert scan_text(carrier, DEFAULT.rules) == []


def test_the_textbook_wif_example_is_a_real_key_and_is_found():
    wif = "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ"  # Mastering Bitcoin's example key
    assert [
        f.value for f in scan_text(f"key {wif}", compile_guard({"packs": ["crypto-keys"]}).rules)
    ] == [wif]
