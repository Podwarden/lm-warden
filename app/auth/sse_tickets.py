import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import threading
import time
from typing import Any

_TICKET_IN_QUERY = re.compile(r"([?&]ticket=)[^&\s\"]*")


def redact_tickets(text: str) -> str:
    """``text`` with every ``ticket=`` query value replaced by ``[redacted]``."""
    return _TICKET_IN_QUERY.sub(r"\1[redacted]", text)


class RedactTicketFilter(logging.Filter):
    """Keeps SSE tickets out of the access log (Task 7 review M3): an
    EventSource can only carry its ticket in the query string, and uvicorn's
    access line logs the whole path. Rewrites the record's arguments; never
    drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and any(isinstance(a, str) and "ticket=" in a for a in args):
            record.args = tuple(redact_tickets(a) if isinstance(a, str) else a for a in args)
        elif isinstance(record.msg, str) and "ticket=" in record.msg:
            record.msg = redact_tickets(record.msg)
        return True


def install_access_log_redaction() -> None:
    """Add ``RedactTicketFilter`` to uvicorn's access logger, once."""
    log = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactTicketFilter) for f in log.filters):
        log.addFilter(RedactTicketFilter())


class TicketStore:
    def __init__(self, secret: str, ttl_seconds: int = 60):
        self._secret = secret.encode("utf-8")
        self._ttl = ttl_seconds
        self._deny: dict[str, float] = {}
        self._lock = threading.Lock()

    def _sign(self, payload: bytes) -> str:
        sig = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).decode().rstrip("=")

    def mint(
        self, user_id: str, path: str, *, sid: str | None = None, exp: int | None = None
    ) -> str:
        """A single-use ticket for ``path``. ``sid`` and ``exp`` (a key holder's
        sign-in and when it expires, app/tokens/routes_me_prompt_lens.py) travel
        signed inside it."""
        body: dict[str, object] = {
            "sub": user_id,
            "path": path,
            "iat": int(time.time()),
            "jti": secrets.token_urlsafe(8),
        }
        if sid is not None:
            body["sid"] = sid
        if exp is not None:
            body["exp"] = exp
        payload = (
            base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )
        sig = self._sign(payload.encode())
        return f"{payload}.{sig}"

    def consume(self, ticket: str, path: str) -> str:
        """Check and spend ``ticket`` for ``path``; its subject."""
        return str(self.consume_claims(ticket, path)["sub"])

    def consume_claims(self, ticket: str, path: str) -> dict[str, Any]:
        """``consume``, returning the whole signed body (``sub``, ``sid``, ...)."""
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
        if not isinstance(body.get("sub"), str):
            raise ValueError("malformed payload")
        return body
