# app/auth/jwt.py
from datetime import UTC, datetime, timedelta

import jwt as pyjwt


def _now() -> datetime:
    return datetime.now(UTC)


def mint_access(sub: str, secret: str, ttl_minutes: int) -> str:
    now = _now()
    return pyjwt.encode(
        {
            "sub": sub,
            "typ": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def mint_refresh(sub: str, secret: str, ttl_days: int) -> str:
    now = _now()
    return pyjwt.encode(
        {
            "sub": sub,
            "typ": "refresh",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=ttl_days)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def mint_forest(token_id: str, secret: str, ttl_minutes: int) -> str:
    """A key holder's session for the forest routes only (typ="forest").

    ``_session_subject`` accepts only typ="access", so every other route
    refuses this token.
    """
    now = _now()
    return pyjwt.encode(
        {
            "sub": token_id,
            "typ": "forest",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def decode(token: str, secret: str) -> dict:
    return pyjwt.decode(token, secret, algorithms=["HS256"])
