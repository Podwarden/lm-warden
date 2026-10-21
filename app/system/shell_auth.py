"""Guard for the loopback-only shell status feed (GET /api/shell/status).

The macOS shell mints a per-launch random bearer and injects it as
``VW_SHELL_TOKEN``; the backend admits only a loopback, un-forwarded peer that
presents the exact token, compared in constant time. A non-loopback caller
gets a 404 indistinguishable from a disabled endpoint so a LAN caller has no
oracle for whether the feed is enabled. The token value is never logged,
raised into an exception message, echoed in a response body, or set in a
response header.

Same stance as ``app/runtime/parent_watch.py``'s ``VW_PARENT_PID``: a
malformed token fails boot loud rather than silently disabling the belt, but
the offending value is never repeated in the error.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import re
from collections.abc import Mapping

from starlette.exceptions import HTTPException
from starlette.requests import Request

logger = logging.getLogger(__name__)

SHELL_TOKEN_ENV = "VW_SHELL_TOKEN"

# 32–256 chars of [A-Za-z0-9_-]; the fullmatch anchors both ends.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,256}")

# Any of these present (any value, even empty) marks the request as relayed by
# a proxy and disqualifies it — a loopback peer is only the shell when it is
# the direct peer.
_FORWARDED_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")


def parse_shell_token(env: Mapping[str, str]) -> str | None:
    """VW_SHELL_TOKEN -> the bearer, or None when unset/empty.

    A set non-empty value must be 32–256 chars of ``[A-Za-z0-9_-]``; anything
    else raises a ValueError naming the variable and the rule but never the
    offending value (a silently disabled belt is the hazard this exists to
    prevent).
    """
    raw = env.get(SHELL_TOKEN_ENV, "")
    if raw == "":
        return None
    if not _TOKEN_RE.fullmatch(raw):
        raise ValueError(f"{SHELL_TOKEN_ENV} must be 32-256 chars of [A-Za-z0-9_-]")
    return raw


def is_loopback_client(request: Request) -> bool:
    """True only when request.client.host parses as a loopback address.

    Covers 127.0.0.0/8, ::1, and an IPv4-mapped IPv6 address whose
    ipv4_mapped is loopback. Never raises: a missing client, a hostname such
    as ``testclient``/``localhost``, or an unparseable/``0.0.0.0`` value are
    all False.
    """
    client = request.client
    if client is None:
        return False
    host = client.host
    if host is None or host == "":
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        mapped: ipaddress.IPv4Address | None = addr.ipv4_mapped
    else:
        mapped = None
    if mapped is not None:
        return mapped.is_loopback
    return addr.is_loopback


def is_forwarded(request: Request) -> bool:
    """True when any Forwarded / X-Forwarded-For / X-Forwarded-Host / X-Real-IP
    header is present (any value, even empty)."""
    return any(name in request.headers for name in _FORWARDED_HEADERS)


async def require_shell_client(request: Request) -> None:
    """Route dependency: admit only a loopback, un-forwarded peer with the
    exact bearer.

    Refusal order: (1) token unset/empty -> 404; (2) not a loopback client or
    forwarded -> 404 (indistinguishable from a disabled endpoint); (3) the
    Authorization header is not exactly ``<scheme> <token>`` with scheme
    ``Bearer`` (case-insensitive) and one space, or the token mismatches ->
    401 with WWW-Authenticate. A malformed header is refused without
    the constant-time compare. Returns None on success.
    """
    token = getattr(request.app.state, "shell_token", None)
    if token is None or token == "":
        logger.debug("shell status refused: endpoint disabled")
        raise HTTPException(404, "Not Found")

    if not is_loopback_client(request) or is_forwarded(request):
        logger.debug("shell status refused: not a loopback peer or forwarded")
        raise HTTPException(404, "Not Found")

    header = request.headers.get("authorization")
    if header is None:
        scheme, sep, candidate = "", "", ""
    else:
        scheme, sep, candidate = header.partition(" ")
    if not sep or scheme.lower() != "bearer" or candidate == "" or " " in candidate:
        logger.debug("shell status refused: malformed authorization header")
        raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    ok = hmac.compare_digest(
        candidate.encode("utf-8", "surrogateescape"),
        token.encode("utf-8", "surrogateescape"),
    )
    if not ok:
        logger.debug("shell status refused: token mismatch")
        raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
    return None
