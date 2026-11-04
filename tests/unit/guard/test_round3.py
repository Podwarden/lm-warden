"""Guard round 3 (third live test, 2026-10-30): letters-only password detection must not take code
words, and a false hit must never spread through the session."""

from __future__ import annotations

import json

import pytest

from app.guard.config import compile_guard
from app.guard.proxy import SeenSet, run_guard
from app.guard.scanner import scan_text

G = compile_guard({"mode": "sanitize"})


@pytest.mark.parametrize(
    "text",
    [
        "    password: Optional[str] = None",
        "password: required",
        "password: optional",
        "password: disabled",
        "auth: passwordless",
        "cluster: Kubernetes",
        "password: Kubernetes",
        'confirm_password: "Passwords must match"',
        "db1.internal:5432:replica:streaming:active",
        "localhost:6379:0:cache:primary",
        "10.0.0.5:8080:api:v2:healthy",
        "password: SecretStr",
        "password = models.CharField(max_length=128)",
    ],
)
def test_code_and_config_words_are_not_passwords(text):
    assert scan_text(text, G.rules) == []


@pytest.mark.parametrize(
    "text,value",
    [
        ("password: hunterqxzlbv", "hunterqxzlbv"),
        ('"password": "zqhvbnmwpx"', "zqhvbnmwpx"),
        ("DB_PASSWORD=CorrectHorseBatteryStaple", "CorrectHorseBatteryStaple"),
        ("db.example.com:5432:app:appuser:Kq7mWp2xLv9z", "Kq7mWp2xLv9z"),
        ("SMTP_PASS=Xk9#mP2vL8qR4nT7", "Xk9#mP2vL8qR4nT7"),
    ],
)
def test_real_passwords_are_still_caught(text, value):
    assert value in [f.value for f in scan_text(text, G.rules)]


def test_a_short_letters_only_hit_never_spreads_through_the_session():
    seen = SeenSet()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": 'cfg = {"password": "zqhvbnmwpx"}'}],
    }
    run_guard(body, "openai", G, token_id="t", secret=b"k" * 32, session="s", seen=seen)
    later = {
        "model": "m",
        "messages": [{"role": "user", "content": "the word zqhvbnmwpx in prose"}],
    }
    out = run_guard(later, "openai", G, token_id="t", secret=b"k" * 32, session="s", seen=seen)
    assert out.action == "pass" and "zqhvbnmwpx" in json.dumps(out.body)


def test_try_it_masks_by_prefix_or_length_only():
    from app.guard.proxy import mask_for

    assert mask_for("bWU6c2VjcmV0cGFzc3dvcmQ=", "docker-auth") == "(24 chars)"
