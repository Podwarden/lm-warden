"""One extractor for "which conversation is this request part of?".

Used for BOTH the dashboard's session column and the data-parallel router's
affinity key, so what an operator sees and what the router pins can never
disagree. Pure, dict and header reads only, and it never raises into a request.

Signals, first match wins (``SessionInfo.source`` names the one that matched):

  claude_code_header   X-Claude-Code-Session-Id                 (Claude Code)
  anthropic_metadata   session inside metadata.user_id          (Claude Code)
  x_session_id         X-Session-Id          (OpenCode ``ses_...``; Hermes when
                       configured, ``YYYYMMDD_HHMMSS_<hex>`` or ``cron_<job>``;
                       Aider via a static header)
  x_session_affinity   x-session-affinity    (OpenCode duplicate; pi with
                       compat.sendSessionAffinityHeaders)
  session_id_header    ``session_id`` (underscore, pi) and ``session-id``
                       (dash, Codex): distinct header names, both checked
  prompt_cache_key     body prompt_cache_key (Codex, pi on Responses)
  openai_user          id-shaped body ``user``
  client_request_id    x-client-request-id, LAST resort: pi and Codex set it to
                       the session id, other clients use it per request

When none matches, the caller falls back to hashing the conversation's opening
turns (``first_user_text``), which is not a session id and is never shown as one.

Every candidate must look like an OPAQUE id (8-128 chars of ``[A-Za-z0-9_.:-]``,
no spaces, no ``@``); anything else is ignored and the next signal is tried.
A value that is, or embeds, a Claude Code user_id (starts with ``{`` or contains
``_account_``) is reduced to its session part first, so an account id cannot
surface whichever field a client put it in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CLAUDE_SESSION_HEADER = "X-Claude-Code-Session-Id"
SESSION_HEADER = "X-Session-Id"
SESSION_AFFINITY_HEADER = "x-session-affinity"
#: pi sends the underscore spelling, Codex the dash spelling.
SESSION_ID_HEADERS = ("session_id", "session-id")
CLIENT_REQUEST_ID_HEADER = "x-client-request-id"
PARENT_HEADERS = ("x-parent-session-id", "x-codex-parent-thread-id")

PROMPT_HASH_CHARS = 512

_OPAQUE_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_USER_ID_SESSION_RE = re.compile(rf"_session_({_UUID})\b")

#: Leading turns that are the SAME in every chat of a harness, so hashing the
#: first user message would put every chat on one replica. Matched on the
#: normalised (lower-cased, whitespace-collapsed) start of the message; a
#: matching user turn and the assistant reply right after it are skipped.
PREAMBLE_PREFIXES: tuple[str, ...] = (
    # Aider's few-shot "example conversation" that opens every edit-format chat.
    "change the greeting to be more casual",
    # Aider's reset pair after switching repositories ("... Please don't
    # consider the above files ...", answered by a bare "Ok.").
    "i switched to a new code base",
)
#: A first real message that is only context (not the user's question): hash it
#: together with the user message that follows it.
CONTEXT_BLOCK_PREFIXES: tuple[str, ...] = (
    # Aider's repo map, sent as the first user turn once files are indexed.
    "here are summaries of some files",
)
#: /v1/responses ``input`` items that are harness preamble, not the user's turn.
RESPONSES_PREAMBLE_PREFIXES: tuple[str, ...] = (
    # Codex prepends the working directory / shell as a user item.
    "<environment_context>",
)


@dataclass(frozen=True)
class SessionInfo:
    id: str | None
    #: which signal matched (see the module docstring), None when none did
    source: str | None
    #: parent session of a subagent (OpenCode, Codex); display only
    parent: str | None = None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup that works on plain dicts too."""
    want = name.lower()
    for k, v in headers.items():
        if k.lower() == want:
            return v
    return None


def opaque_id(raw: Any) -> str | None:
    """``raw`` if it looks like an opaque id, else None."""
    if not isinstance(raw, str):
        return None
    v = raw.strip()
    return v if _OPAQUE_RE.match(v) else None


def session_from_user_id(user_id: str) -> str | None:
    """Claude Code's ``metadata.user_id`` carries the session id in one of two
    shapes: a JSON string with a ``session_id`` key, or
    ``user_<hex>_account_<uuid>_session_<uuid>``. Only the session part is
    returned -- the rest holds the account id and must never leave here."""
    text = user_id.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except ValueError:
            return None
        return opaque_id(obj.get("session_id")) if isinstance(obj, dict) else None
    m = _USER_ID_SESSION_RE.search(text)
    return m.group(1) if m else None


def clean_candidate(raw: Any) -> str | None:
    """Every candidate goes through here: account-bearing values are reduced to
    their session part, everything else must pass the opaque-id rule."""
    if not isinstance(raw, str):
        return None
    v = raw.strip()
    if v.startswith("{") or "_account_" in v:
        return session_from_user_id(v)
    return opaque_id(v)


def extract_session(
    body_json: Any, headers: Mapping[str, str], explicit_user_id: str | None = None
) -> SessionInfo:
    """The conversation's id and which signal it came from. Fail-open."""
    try:
        return _extract(body_json, headers, explicit_user_id)
    except Exception:  # noqa: BLE001 -- display and routing metadata only
        return SessionInfo(None, None)


def _extract(
    body_json: Any, headers: Mapping[str, str], explicit_user_id: str | None
) -> SessionInfo:
    parent = None
    for name in PARENT_HEADERS:
        parent = clean_candidate(_header(headers, name))
        if parent:
            break

    def hit(source: str, value: str | None) -> SessionInfo | None:
        return SessionInfo(value, source, parent) if value else None

    body = body_json if isinstance(body_json, dict) else {}
    found = (
        hit("claude_code_header", clean_candidate(_header(headers, CLAUDE_SESSION_HEADER)))
        or hit("anthropic_metadata", clean_candidate(explicit_user_id))
        or hit("x_session_id", clean_candidate(_header(headers, SESSION_HEADER)))
        or hit("x_session_affinity", clean_candidate(_header(headers, SESSION_AFFINITY_HEADER)))
    )
    if found:
        return found
    for name in SESSION_ID_HEADERS:
        found = hit("session_id_header", clean_candidate(_header(headers, name)))
        if found:
            return found
    return (
        hit("prompt_cache_key", clean_candidate(body.get("prompt_cache_key")))
        or hit("openai_user", clean_candidate(body.get("user")))
        or hit("client_request_id", clean_candidate(_header(headers, CLIENT_REQUEST_ID_HEADER)))
        or SessionInfo(None, None, parent)
    )


# ---- the fallback: hash the conversation's opening ---------------------------


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return ""


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _starts(text: str, prefixes: tuple[str, ...]) -> bool:
    n = _norm(text[:200])
    return any(n.startswith(p) for p in prefixes)


def responses_first_user_text(input_value: Any) -> str:
    """The first real user turn of a ``/v1/responses`` ``input``: a plain string,
    or the first ``role: user`` item that is not harness preamble (developer and
    system items, ``<environment_context>`` blocks)."""
    if isinstance(input_value, str):
        return input_value
    if not isinstance(input_value, list):
        return ""
    for item in input_value:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        text = text_of(item.get("content"))
        if text and not _starts(text.lstrip(), RESPONSES_PREAMBLE_PREFIXES):
            return text
    return ""


def _chat_opening(msgs: list[Any]) -> str:
    """First user message after any known preamble; if that is only a context
    block (repo map), the next user message is hashed with it."""
    turns = [(m.get("role"), text_of(m.get("content"))) for m in msgs if isinstance(m, dict)]
    i = 0
    while i < len(turns):
        role, text = turns[i]
        if role == "user" and _starts(text, PREAMBLE_PREFIXES):
            i += 1
            if i < len(turns) and turns[i][0] == "assistant":
                i += 1  # the canned reply to the preamble
            continue
        if role == "assistant" and i == 0:
            i += 1
            continue
        if role == "user":
            break
        i += 1  # system and the like
    else:
        # no user turn left: fall back to the first message, as before
        return next((t for _, t in turns if t), "")[:PROMPT_HASH_CHARS]
    first = turns[i][1]
    if _starts(first, CONTEXT_BLOCK_PREFIXES):
        for role, text in turns[i + 1 :]:
            if role == "user" and text:
                return first[:PROMPT_HASH_CHARS] + "\x00" + text[:PROMPT_HASH_CHARS]
    return first[:PROMPT_HASH_CHARS]


def first_user_text(body_json: Any) -> str:
    """Text the prompt-hash key is built from. The system message is
    deliberately not the input: it is identical across sessions of one client
    and would pin a whole token to one replica."""
    if not isinstance(body_json, dict):
        return ""
    msgs = body_json.get("messages")
    if isinstance(msgs, list) and msgs:
        return _chat_opening(msgs)
    if "input" in body_json:
        return responses_first_user_text(body_json.get("input"))[:PROMPT_HASH_CHARS]
    prompt = body_json.get("prompt")
    if isinstance(prompt, str):
        return prompt[:PROMPT_HASH_CHARS]
    if isinstance(prompt, list) and prompt:
        return prompt[0][:PROMPT_HASH_CHARS] if isinstance(prompt[0], str) else ""
    return ""
