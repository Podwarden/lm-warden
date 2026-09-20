
from app.auth.bearer import (
    ADMIN_TOKEN_PREFIX,
    generate_admin_token,
    generate_bearer_token,
    parse_bearer_header,
)


def test_generate_token_format():
    plaintext = generate_bearer_token()
    assert plaintext.startswith("vw_")
    # base32 of 32 bytes = 56 chars, total prefix+body = 59
    assert len(plaintext) == 3 + 56


def test_parse_bearer_header_strips_prefix():
    assert parse_bearer_header("Bearer vw_xxxx") == "vw_xxxx"
    assert parse_bearer_header("bearer vw_xxxx") == "vw_xxxx"


def test_parse_bearer_header_rejects_other_schemes():
    assert parse_bearer_header("Basic abc") is None
    assert parse_bearer_header("vw_xxxx") is None
    assert parse_bearer_header(None) is None
    assert parse_bearer_header("") is None


def test_generate_admin_token_format():
    plaintext = generate_admin_token()
    assert plaintext.startswith(ADMIN_TOKEN_PREFIX) and ADMIN_TOKEN_PREFIX == "vwa_"
    assert len(plaintext) == 4 + 56
    body = plaintext[4:]
    assert body == body.lower() and body.isalnum()
    # /v1's prefix check (`startswith("vw_")`) must refuse an admin secret.
    assert not plaintext.startswith("vw_")
    assert generate_admin_token() != plaintext
