"""Guard recall and precision after the 2026-10-30 live test: encoded secrets, human passwords,
CLI forms, named tokens it missed, and the look-alikes it wrongly caught. All values synthetic."""

from __future__ import annotations

import base64
import json

import pytest

from app.guard.config import compile_guard
from app.guard.scanner import scan_text
from tests.unit.guard.corpus import pem_private

DEFAULT = compile_guard({"mode": "sanitize"})
WITH_PII = compile_guard({"mode": "sanitize", "packs": ["pii-contact"]})


def _found(text: str, guard=DEFAULT) -> list[tuple[str, str]]:
    return [(f.rule_id, f.value) for f in scan_text(text, guard.rules)]


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


# --- encoded ---------------------------------------------------------------------------------


def test_a_kubeconfig_client_key_is_caught_inside_base64():
    blob = _b64(pem_private())
    cfg = f"users:\n- name: admin\n  user:\n    client-key-data: {blob}\n"
    assert _found(cfg) == [("pem-private-key", blob)]


def test_a_base64_blob_holding_an_aws_key_is_caught():
    blob = _b64("aws_access_key_id=AKIAQWERTYUIOPASDFGH\n")
    assert _found(f"payload: {blob}") == [("aws-access-key", blob)]


def test_docker_config_auth_is_caught():
    auth = _b64("deploy:Hj7#kL9mQ2pW")
    cfg = json.dumps({"auths": {"registry.example.com": {"auth": auth}}})
    assert _found(cfg) == [("docker-auth", auth)]


def test_a_pem_key_with_json_escapes_is_caught():
    pem = pem_private().replace("\n", "\\n")
    sa = json.dumps({"type": "service_account", "private_key": "PLACEHOLDER"}).replace(
        "PLACEHOLDER", pem
    )
    assert [r for r, _v in _found(sa)] == ["pem-private-key"]


@pytest.mark.parametrize(
    "text",
    [
        "integrity sha512-" + _b64("x" * 60),  # lockfile hash decodes to text but holds nothing
        "data:image/png;base64," + base64.b64encode(bytes(range(256))).decode(),
        _b64("just a harmless sentence that is long enough to decode"),
    ],
)
def test_base64_without_a_secret_inside_is_left_alone(text):
    assert _found(text) == []


# --- passwords -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("SMTP_PASS=Xk9#mP2vL8qR4nT7", "Xk9#mP2vL8qR4nT7"),
        ("DB_PASSWORD=Summer2024!!", "Summer2024!!"),
        ("DB_PASSWORD=CorrectHorseBatteryStaple", "CorrectHorseBatteryStaple"),
        ("password: 'correct horse battery staple'", "correct horse battery staple"),
        ("REDIS_PASSWORD=hunter2x9", "hunter2x9"),
    ],
)
def test_human_passwords_in_config_are_caught(text, value):
    assert value in [v for _r, v in _found(text)]


@pytest.mark.parametrize(
    "text,value",
    [
        ("redis://default:supersecretpassword@cache.internal:6379/0", "supersecretpassword"),
        ("curl -u admin:Tr0ub4dor3 https://api.example.com", "Tr0ub4dor3"),
        ("mysql -h db -u root -pS3cretPass db", "S3cretPass"),
        ("pg_dump --password=Zq8wPl3mXv db", "Zq8wPl3mXv"),
        ("machine api.example.com login bot password Qp4!mZ9vLx", "Qp4!mZ9vLx"),
    ],
)
def test_passwords_on_command_lines_and_urls_are_caught(text, value):
    assert value in [v for _r, v in _found(text)]


@pytest.mark.parametrize(
    "text",
    [
        "password = user_password_hash",
        "const password = getPasswordFromVault()",
        "password: string",
        "passwordField: PasswordInput",
        "PASSWORD=${DB_PASSWORD}",
        "password = os.environ['DB_PASSWORD']",
        "tokenEndpoint = 'https://login.example.com/oauth2/token'",
        '{"nextToken": "eyJsYXN0S2V5IjoiYWJjMTIzIiwicGFnZSI6Mn0="}',
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "if (password.length < 12) return false;",
    ],
)
def test_code_and_published_examples_are_not_passwords(text):
    assert _found(text) == []


# --- named tokens ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,rule",
    [
        (
            "Authorization: Bearer ya29.a0AfH6SMBx3kLqZ9vR2mT8nWp4QeYc7uIoP1sDfGhJkL",
            "google-oauth-token",
        ),
        ("VAULT_TOKEN=hvs.CAESIJx9kLmQ2pWz8vR4tN7bY3cF6hJ1sD5gK0aE", "vault-token"),
        ("stripe.api_key = 'sk_test_51HxYz9KqL2mN8pR4tV7wB3cF6hJ1sD5g'", "stripe-key"),
    ],
)
def test_named_tokens_it_missed_are_caught(text, rule):
    assert rule in [r for r, _v in _found(text)]


# --- contact data (opt-in) -------------------------------------------------------------------


@pytest.mark.parametrize("text", ["call (415) 555-0134 now", "phone: 415-555-0134"])
def test_us_phone_formats_are_caught(text):
    assert [r for r, _v in _found(text, WITH_PII)] == ["phone"]


def test_build_numbers_are_not_phones():
    assert _found("version +20261029 build", WITH_PII) == []
