"""Unit tests for the token_allows() helper in app.proxy.auth."""

from app.db.repos.tokens import TokenRow
from app.proxy.auth import token_allows


def _make_token(allowed_models: str | None) -> TokenRow:
    return TokenRow(
        id="tok1",
        name="test",
        prefix="vw_valid",
        scope="inference",
        allowed_models=allowed_models,
        rate_limit_rpm=None,
        rate_limit_tpm=None,
        revoked_at=None,
        last_used_at=None,
        created_at="2025-01-01T00:00:00+00:00",
        expires_at=None,
        rotated_at=None,
        rotated_from=None,
    )


def test_none_allows_any_model():
    token = _make_token(None)
    assert token_allows(token, "qwen3") is True
    assert token_allows(token, "llama3") is True
    assert token_allows(token, "mistral") is True


def test_single_model_allows_exact_match():
    token = _make_token("qwen3")
    assert token_allows(token, "qwen3") is True


def test_single_model_denies_other():
    token = _make_token("qwen3")
    assert token_allows(token, "llama3") is False


def test_csv_handles_whitespace_and_allows_all_listed():
    token = _make_token("qwen3, llama3 ,mistral")
    assert token_allows(token, "qwen3") is True
    assert token_allows(token, "llama3") is True
    assert token_allows(token, "mistral") is True


def test_csv_denies_unlisted():
    token = _make_token("qwen3, llama3 ,mistral")
    assert token_allows(token, "gpt4") is False


def test_empty_string_does_not_match_anything():
    # An empty allowed_models value (after CSV split+strip) should deny all.
    token = _make_token("")
    assert token_allows(token, "") is False
    assert token_allows(token, "qwen3") is False


def test_csv_with_empty_items_does_not_match_empty_string():
    # "qwen3,,llama3" — the middle empty item must not match ""
    token = _make_token("qwen3,,llama3")
    assert token_allows(token, "") is False
    assert token_allows(token, "qwen3") is True
    assert token_allows(token, "llama3") is True
