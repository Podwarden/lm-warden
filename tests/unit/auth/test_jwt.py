# tests/unit/auth/test_jwt.py
from datetime import UTC, datetime, timedelta

import jwt as pyjwt
import pytest

from app.auth.jwt import decode, mint_access, mint_refresh

SECRET = "test-secret-not-for-prod"


def test_access_token_roundtrip():
    tok = mint_access("admin", SECRET, ttl_minutes=15)
    claims = decode(tok, SECRET)
    assert claims["sub"] == "admin"
    assert claims["typ"] == "access"
    assert 0 < claims["exp"] - claims["iat"] <= 15 * 60


def test_refresh_token_roundtrip():
    tok = mint_refresh("admin", SECRET, ttl_days=7)
    claims = decode(tok, SECRET)
    assert claims["sub"] == "admin"
    assert claims["typ"] == "refresh"


def test_rejects_expired():
    now = datetime.now(UTC)
    expired = pyjwt.encode(
        {
            "sub": "admin",
            "typ": "access",
            "iat": int((now - timedelta(hours=2)).timestamp()),
            "exp": int((now - timedelta(hours=1)).timestamp()),
        },
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(pyjwt.ExpiredSignatureError):
        decode(expired, SECRET)


def test_rejects_wrong_secret():
    tok = mint_access("admin", SECRET, ttl_minutes=15)
    with pytest.raises(pyjwt.InvalidSignatureError):
        decode(tok, "different-secret")
