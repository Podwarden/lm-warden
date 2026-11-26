"""Remote media policy (#324).

A media URL in a request is fetched by the ENGINE process, from whatever
network the engine sits on. By default the warden therefore accepts only
``data:`` URLs. An operator can allow named hosts with
``VW_REMOTE_MEDIA_ALLOWED_DOMAINS`` (comma separated, exact host match, the
same rule as vLLM's ``--allowed-media-domains``, which the launcher emits).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

ENV_VAR = "VW_REMOTE_MEDIA_ALLOWED_DOMAINS"
_URL_PART_KEYS = ("image_url", "video_url", "audio_url")


def allowed_domains() -> list[str]:
    """The operator's allowlist: lower-cased, de-duplicated, order kept.
    Empty (the default) means no remote media at all. Read per call so the
    environment is honoured without re-importing."""
    out: list[str] = []
    for raw in os.environ.get(ENV_VAR, "").replace(";", ",").split(","):
        host = raw.strip().lower()
        if host and host not in out:
            out.append(host)
    return out


def check_url(url: str, allowed: list[str]) -> str | None:
    """None when the engine may be handed this URL, else the refusal text."""
    stripped = url.strip()
    if stripped[:5].lower() == "data:":
        return None
    try:
        parts = urlsplit(stripped)
        host = (parts.hostname or "").lower()
    except ValueError:
        return "media URL is not valid"
    if parts.scheme.lower() not in ("http", "https"):
        return "only data: URLs and allow-listed http(s) media URLs are accepted"
    if not allowed:
        return "remote media URLs are not accepted; send the media as a data: URL"
    if host not in allowed:
        return "media host is not in the operator's allowed list"
    return None


def refusal(body_json: Any, allowed: list[str] | None = None) -> str | None:
    """First refusal text for a chat body's media URL parts, or None."""
    if not isinstance(body_json, dict):
        return None
    messages = body_json.get("messages")
    if not isinstance(messages, list):
        return None
    allow = allowed_domains() if allowed is None else allowed
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict):
                continue
            for key in _URL_PART_KEYS:
                if key not in blk:
                    continue
                val = blk[key]
                url = val.get("url") if isinstance(val, dict) else val
                if not isinstance(url, str):
                    return "media URL must be a string"
                why = check_url(url, allow)
                if why:
                    return why
    return None
