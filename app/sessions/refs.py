"""Opaque ids for the sessions pages. A raw harness session key or cobs2 session
never reaches a URL or an API response: URLs carry ``s_<hex>`` (a session, its
lookup ref plus, for a session-less segment, the first request's id prefix) or
``r_<request id>``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.proxy.forest_fields import hkey

SESSION_PREFIX = "s_"
REQUEST_PREFIX = "r_"
SEGMENT_HEX = 12
_SESSION_RE = re.compile(r"s_([0-9a-f]{32})([0-9a-f]{12})?")
_REQUEST_RE = re.compile(r"r_([0-9a-f]{32})")


@dataclass(frozen=True)
class ParsedRef:
    kind: Literal["session", "request"]
    ref: str
    segment: str | None


def sess_ref(secret: str, token_id: str | None, cobs2_session: str | None) -> str | None:
    """The keyed lookup ref of (token, session); None without a session."""
    if not cobs2_session:
        return None
    return hkey(secret, f"{token_id or '-'}|{cobs2_session}", 32)


def session_id(sess_ref: str, segment_first: str | None = None) -> str:
    return SESSION_PREFIX + sess_ref + (segment_first[:SEGMENT_HEX] if segment_first else "")


def request_ref(request_id: str) -> str:
    return REQUEST_PREFIX + request_id


def parse_ref(value: str) -> ParsedRef | None:
    m = _SESSION_RE.fullmatch(value)
    if m is not None:
        return ParsedRef("session", m.group(1), m.group(2))
    m = _REQUEST_RE.fullmatch(value)
    if m is not None:
        return ParsedRef("request", m.group(1), None)
    return None


def is_fallback(cobs2_session: str) -> bool:
    """A cobs2 fallback session (no harness session id): ``f:<digest>``."""
    return cobs2_session.startswith("f:")
