import base64
import hashlib
import hmac
import json
import secrets
import threading
import time


class TicketStore:
    def __init__(self, secret: str, ttl_seconds: int = 60):
        self._secret = secret.encode("utf-8")
        self._ttl = ttl_seconds
        self._deny: dict[str, float] = {}
        self._lock = threading.Lock()

    def _sign(self, payload: bytes) -> str:
        sig = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).decode().rstrip("=")

    def mint(self, user_id: str, path: str) -> str:
        body = {
            "sub": user_id,
            "path": path,
            "iat": int(time.time()),
            "jti": secrets.token_urlsafe(8),
        }
        payload = (
            base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )
        sig = self._sign(payload.encode())
        return f"{payload}.{sig}"

    def consume(self, ticket: str, path: str) -> str:
        try:
            payload, sig = ticket.split(".", 1)
        except ValueError as exc:
            raise ValueError("malformed ticket") from exc
        expected = self._sign(payload.encode())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        try:
            padded = payload + "=" * (-len(payload) % 4)
            body = json.loads(base64.urlsafe_b64decode(padded))
        except Exception as exc:
            raise ValueError("malformed payload") from exc
        if not isinstance(body, dict):
            raise ValueError("malformed payload")
        iat = body.get("iat")
        if not isinstance(iat, int | float) or isinstance(iat, bool):
            raise ValueError("malformed payload")
        if body.get("path") != path:
            raise ValueError("path mismatch")
        now = time.time()
        SKEW_SECONDS = 5
        if iat > now + SKEW_SECONDS:
            raise ValueError("future iat")
        if now - iat > self._ttl:
            raise ValueError("expired")
        with self._lock:
            # Garbage-collect deny entries older than 2*TTL.
            self._deny = {k: v for k, v in self._deny.items() if v > now}
            if ticket in self._deny:
                raise ValueError("already consumed")
            self._deny[ticket] = now + self._ttl + 5
        return body["sub"]
