"""Placeholders: ⟦LMW-SECRET-<10 hex>⟧, the hex a keyed HMAC of (key id, value). Stable across
turns and restarts, so the sanitized prompt stays a prefix of the next one; different per key."""

from __future__ import annotations

import hashlib
import hmac
import json
import re

_TOLERANT = re.compile(r"(?i)(?:⟦|\[\[)?LMW-SECRET-([0-9a-f]{10})(?:⟧|\]\])?")
LABEL = b"lmw-guard-v1"


def guard_secret(cookie_secret: str | bytes) -> bytes:
    k = cookie_secret.encode() if isinstance(cookie_secret, str) else cookie_secret
    return hmac.new(k, LABEL, hashlib.sha256).digest()


def placeholder_hex(secret: bytes, token_id: str, value: str) -> str:
    return hmac.new(
        secret, token_id.encode() + b"\x00" + value.encode(), hashlib.sha256
    ).hexdigest()[:10]


def render(hex10: str) -> str:
    return f"⟦LMW-SECRET-{hex10}⟧"


class RestoreMap:
    """Request-local {hex → value}. Restores only placeholders this request produced; any other
    placeholder (guessed, copied, stale) stays as it is."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._m = dict(mapping or {})
        #: Which placeholders were put back (ids only), for the action log.
        self.restored: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self._m)

    def add(self, hex10: str, value: str) -> None:
        self._m[hex10] = value

    def restore(self, text: str, *, json_escape: bool = False) -> tuple[str, int]:
        """``json_escape``: the text is the inside of a JSON string (tool-call arguments as the
        model streams them), so the value goes in escaped."""
        if not self._m or "LMW-SECRET-" not in text.upper():
            return text, 0
        n = 0

        def sub(m: re.Match[str]) -> str:
            nonlocal n
            hx = m.group(1).lower()
            v = self._m.get(hx)
            if v is None:
                return m.group(0)
            n += 1
            self.restored.add(hx)
            return json.dumps(v, ensure_ascii=False)[1:-1] if json_escape else v

        return _TOLERANT.sub(sub, text), n

    def hexes(self) -> list[str]:
        return list(self._m)
