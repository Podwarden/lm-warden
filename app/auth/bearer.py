import base64
import secrets

#: Admin tokens call the control API (/api/*); inference tokens call /v1
#: (docs/superpowers/specs/2026-09-19-admin-tokens-design.md, decision 1). The
#: prefix tells a leaked secret's kind at a glance and lets secret scanners key
#: on it. "vwa_" does not start with "vw_", so /v1's prefix check
#: (app/proxy/auth.py) refuses an admin secret before any lookup.
ADMIN_TOKEN_PREFIX = "vwa_"
INFERENCE_TOKEN_PREFIX = "vw_"


def _random_body() -> str:
    """56 lowercase base32 chars (35 random bytes, no padding)."""
    raw = secrets.token_bytes(35)
    return base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def generate_bearer_token() -> str:
    """Returns vw_<56 lowercase base32 chars> (35 random bytes, no padding)."""
    return f"{INFERENCE_TOKEN_PREFIX}{_random_body()}"


def generate_admin_token() -> str:
    """Returns vwa_<56 lowercase base32 chars> (35 random bytes, no padding)."""
    return f"{ADMIN_TOKEN_PREFIX}{_random_body()}"


def parse_bearer_header(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()
