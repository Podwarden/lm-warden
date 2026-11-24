"""The guard on the request path (spec phase 1): decide sanitize / block for one request, restore
placeholders in the answer (SSE frames and JSON bodies), the trailer and the block reply in the
OpenAI chat and legacy completions dialects. Translated /v1/messages and /v1/responses consume
these chat frames, so they need nothing of their own.

Nothing here touches the DB or keeps a value past the request: the restore map lives in the
caller's frame, events carry the value's length and keyed id only."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.guard.config import Guard, compile_guard
from app.guard.events import GuardEvent
from app.guard.placeholders import RestoreMap
from app.guard.resolve import compiled_settings, effective
from app.guard.rules import BUILTIN
from app.guard.walk import Hit, Unscannable, apply_findings, inspect_body

MAX_SCAN_BYTES = 8 * 1024 * 1024
MAX_FINDINGS = 2000
HEADER = "X-LMWarden-Guard"
_RULES = {r.id: r for r in BUILTIN}


class GuardFailed(Exception):
    """The guard could not decide (fail closed): the request is answered by LM Warden."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def mask_for(value: str, rule_id: str) -> str:
    """The masked form shown back to the client. A generic rule's value (a password) is shown by
    length only: its first and last characters would go back into the next prompt."""
    rule = _RULES.get(rule_id)
    if rule is None or not rule.public_prefix or not value.startswith(rule.public_prefix):
        return f"({len(value)} chars)"
    return mask(value, rule.public_prefix)


def mask(value: str, public_prefix: str | None = None) -> str:
    """At most a quarter of the characters, first ≤ 4 and last ≤ 2; under 12 characters only the
    length. For the client's own block reply and trailer, and the admin's try-it box."""
    if public_prefix and value.startswith(public_prefix):
        return f"{public_prefix}… ({len(value)} chars)"
    if len(value) < 12:
        return f"({len(value)} chars)"
    head = min(4, len(value) // 4)
    tail = min(2, len(value) // 8)
    return f"{value[:head]}…{value[-tail:] if tail else ''} ({len(value)} chars)"


# --- settings --------------------------------------------------------------------------------


_GUARD_MODE_RANK = {"sanitize": 1, "block": 2}


def _alias_mode(alias: Any) -> str | None:
    """The alias's guard mode when it is a guard (sanitize | block); anything
    else (inherit/absent/garbage) is inert."""
    if alias is None:
        return None
    m = getattr(alias, "guard", None)
    return m if m in _GUARD_MODE_RANK else None


def guard_for(platform: dict[str, Any] | None, token: Any, alias: Any = None) -> Guard | None:
    """The key's compiled guard, or None. With no platform setting and nothing on the key this is
    one attribute read.

    An alias that carries a guard (D4c) can only strengthen: the effective mode
    is the stricter of the key's and the alias's (block > sanitize > none); an
    alias never weakens a key.
    """
    alias_mode = _alias_mode(alias)
    raw = getattr(token, "guard", None)
    if platform is None and not raw and alias_mode is None:
        return None
    try:
        key = json.loads(raw) if raw else None
    except ValueError:
        key = None
    eff = compiled_settings(effective(platform, key))
    if alias_mode is None:
        return compile_guard(eff)
    out = dict(eff) if eff else {}
    cur_rank = _GUARD_MODE_RANK.get(out.get("mode") or "", 0)
    if _GUARD_MODE_RANK[alias_mode] > cur_rank:
        out["mode"] = alias_mode
    return compile_guard(out)


# --- session state ---------------------------------------------------------------------------


class SeenSet:
    """Per (key, session): the values already found, as {placeholder hex: (length, rule id)}.
    The hex is a keyed HMAC of the value, so this holds ids and integers, never a value. It does
    two jobs: block-once (a value already reported sanitizes instead of blocking) and literal
    matching (a value found once is caught again later in the session without the context its
    rule needed). Bounded, in memory; a restart empties it."""

    def __init__(self, sessions: int = 20_000, per_session: int = 4096, ttl_s: float = 86_400.0):
        self._d: OrderedDict[tuple[str, str], tuple[float, dict[str, tuple[Any, ...]]]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._max, self._per, self._ttl = sessions, per_session, ttl_s

    def get(self, key: tuple[str, str]) -> dict[str, tuple[Any, ...]]:
        with self._lock:
            hit = self._d.get(key)
            if hit is None or time.monotonic() - hit[0] > self._ttl:
                return {}
            return dict(hit[1])

    def add(self, key: tuple[str, str], found: dict[str, tuple[Any, ...]]) -> None:
        if not found:
            return
        with self._lock:
            _, cur = self._d.pop(key, (0.0, {}))
            cur = {**cur, **found}
            if len(cur) > self._per:
                cur = dict(list(cur.items())[-self._per :])
            self._d[key] = (time.monotonic(), cur)
            while len(self._d) > self._max:
                self._d.popitem(last=False)


# --- the decision ----------------------------------------------------------------------------


@dataclass
class Outcome:
    action: str  # "pass" | "sanitize" | "block"
    body: dict[str, Any]
    restore: RestoreMap = field(default_factory=RestoreMap)
    trailer: str | None = None
    findings: list[Hit] = field(default_factory=list)
    events: list[GuardEvent] = field(default_factory=list)
    history_json: str = ""
    restore_mode: str = "all"
    new: set[str] = field(default_factory=set)
    token_id: str = ""

    def restored_events(self, *, now: float) -> list[GuardEvent]:
        """One `restored` event per value put back in the answer (ids and lengths only)."""
        done = self.restore.restored
        return [
            dataclasses.replace(
                _event(now, self.token_id, "restored", h),
                role="assistant",
                msg_index=None,
                kind="answer",
            )
            for h in _distinct(self.findings)
            if h.hex in done
        ]


def _where_text(h: Hit) -> str:
    if h.where.index == 0 or h.where.kind == "system":
        return "system prompt" if h.where.kind == "system" else "tool definitions"
    return f"{h.where.kind}, message {h.where.index}"


def _report(hits: list[Hit]) -> list[str]:
    """One line per distinct value: placeholder head, rule, masked value, where."""
    out, done = [], set()
    for h in hits:
        if h.hex in done:
            continue
        done.add(h.hex)
        m = mask_for(h.value, h.rule_id)
        out.append(f"⟦LMW-SECRET-{h.hex[:4]}…⟧ ({h.rule_id}, {m}, {_where_text(h)})")
    return out


def _trailer_text(hits: list[Hit]) -> str:
    lines = _report(hits)
    n = len(lines)
    return (
        f"[LM Warden] {n} value{'s' if n != 1 else ''} {'were' if n != 1 else 'was'} replaced before "
        f"the model saw {'them' if n != 1 else 'it'}: "
        + "; ".join(lines)
        + ". The model answered without them; placeholders in this answer were restored."
    )


def _json_mode(body: dict[str, Any]) -> bool:
    rf = body.get("response_format")
    return isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")


def _history(mode: str, blocked: bool, hits: list[Hit], new: set[str]) -> str:
    per: dict[str, list[int]] = {}
    seen: set[str] = set()
    for h in hits:
        if h.hex in seen:
            continue
        seen.add(h.hex)
        c = per.setdefault(h.rule_id, [0, 0])
        c[0] += 1
        c[1] += 1 if h.hex in new else 0
    f = [[r, c[0], c[1]] for r, c in sorted(per.items())]
    s = json.dumps({"m": mode, "b": int(blocked), "f": f, "r": 0, "u": 0}, separators=(",", ":"))
    while len(s) > 256 and f:
        f.pop()
        s = json.dumps(
            {"m": mode, "b": int(blocked), "f": f, "r": 0, "u": 0}, separators=(",", ":")
        )
    return s


def run_guard(
    body: dict[str, Any],
    dialect: str,
    guard: Guard,
    *,
    token_id: str,
    secret: bytes,
    session: str,
    seen: SeenSet,
    raw_len: int | None = None,
    max_scan_bytes: int = MAX_SCAN_BYTES,
) -> Outcome:
    """Inspect, decide, and (sanitize) build the engine's body. Raises GuardFailed when the guard
    cannot decide and the key fails closed; a key that fails open passes the body through."""
    try:
        size = raw_len if raw_len is not None else len(json.dumps(body))
        if size > max_scan_bytes:
            raise GuardFailed("too_large")
        skey = (token_id, session)
        before = seen.get(skey)
        ins = inspect_body(body, dialect, guard, token_id=token_id, secret=secret, known=before)
        # A value found in this request may also sit elsewhere in it without its context (said in
        # prose before the .env that names it): a second pass with what this request found.
        here = _fingerprints(ins.findings)
        if here.keys() - before.keys():
            ins = inspect_body(
                body, dialect, guard, token_id=token_id, secret=secret, known={**before, **here}
            )
        if len(ins.findings) > MAX_FINDINGS:
            raise GuardFailed("too_many_findings")
    except GuardFailed:
        if guard.fail == "open":
            return Outcome("pass", body)
        raise
    except Unscannable as e:
        if guard.fail == "open":
            return Outcome("pass", body)
        raise GuardFailed("unscannable") from e
    except Exception as e:  # noqa: BLE001 -- a scanner fault must never pass a value through
        if guard.fail == "open":
            return Outcome("pass", body)
        raise GuardFailed("scanner_error") from e
    if not ins.findings:
        return Outcome("pass", body)
    hexes = {h.hex for h in ins.findings}
    new = hexes - before.keys()
    blocking = [
        h
        for h in ins.findings
        if not _head(h)
        and h.hex in new
        and guard.action_for(_RULES.get(h.rule_id) or _rule_stub(h.rule_id)) == "block"
    ]
    # A value inside a signed thinking block cannot be replaced without breaking the signature,
    # so the only way to keep it from the model's provider is to stop the request, in any mode.
    blocking += [h for h in ins.findings if h.where.kind == "thinking" and h not in blocking]
    seen.add(skey, _fingerprints(ins.findings))
    now = time.time()
    if blocking:
        events = [_event(now, token_id, "blocked", h) for h in _distinct(blocking)]
        return Outcome(
            "block",
            body,
            findings=blocking,
            events=events,
            history_json=_history(guard.mode, True, blocking, new),
            new=new,
        )
    suppress = _json_mode(body) or bool(body.get("echo"))
    trailer = None
    if guard.trailer != "off" and not suppress and (guard.trailer == "always" or new):
        trailer = _trailer_text(ins.findings)
    restore = ins.restore if guard.restore != "off" else RestoreMap()
    return Outcome(
        "sanitize",
        apply_findings(body, ins, note=True),
        restore=restore,
        trailer=trailer,
        findings=ins.findings,
        events=[
            _event(now, token_id, "sanitized", h) for h in _distinct(ins.findings) if h.hex in new
        ],
        history_json=_history(guard.mode, False, ins.findings, new),
        restore_mode=guard.restore,
        new=new,
        token_id=token_id,
    )


def _fingerprints(hits: list[Hit]) -> dict[str, tuple[Any, ...]]:
    """{placeholder hex: (length, rule id, word count, in-process hash)}: what literal matching
    needs. Integers and ids only; the in-process hash is Python's per-process keyed hash, a cheap
    prefilter before the HMAC, meaningless outside this process."""
    return {
        h.hex: (len(h.value), h.rule_id, len(h.value.split()), hash(h.value))
        for h in hits
        # A short letters-only value may be a word the rules mistook for a password; remembering it
        # would replace that word everywhere in the session.
        if not (h.value.isalpha() and len(h.value) < 16)
    }


def _head(h: Hit) -> bool:
    """The harness's system prompt and tool definitions: reported, never blocked."""
    return h.where.index == 0 or h.where.role in ("system", "developer")


def _rule_stub(rule_id: str) -> Any:
    class _R:
        id = rule_id
        pack = "custom"

    return _R()


def _distinct(hits: list[Hit]) -> list[Hit]:
    out, done = [], set()
    for h in hits:
        if h.hex not in done:
            done.add(h.hex)
            out.append(h)
    return out


def _event(now: float, token_id: str, action: str, h: Hit) -> GuardEvent:
    return GuardEvent(
        ts=now,
        token_id=token_id,
        request_id=None,
        sess_ref=None,
        action=action,
        rule_id=h.rule_id,
        role=h.where.role,
        msg_index=h.where.index,
        kind=h.where.kind,
        length=len(h.value),
        ph=h.hex,
        relayed=False,
    )


def with_restored(history_json: str, n: int) -> str:
    try:
        d = json.loads(history_json)
    except ValueError:
        return history_json
    d["r"] = n
    return json.dumps(d, separators=(",", ":"))


def log_events(app_state: Any, events: list[GuardEvent], request_id: str | None) -> None:
    """Write the action log off the request path (no value: app/guard/events.py)."""
    if not events:
        return
    import asyncio
    import dataclasses

    from app.guard.events import record_events

    rows = [dataclasses.replace(e, request_id=request_id) for e in events]
    db_path = app_state.settings.db_path

    async def _write() -> None:
        try:
            await asyncio.to_thread(record_events, db_path, rows)
        except Exception:  # noqa: BLE001 -- the log must never fail a request
            logging.getLogger(__name__).warning("guard: could not record events", exc_info=True)

    task = asyncio.get_running_loop().create_task(_write())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


_TASKS: set[Any] = set()


def shared_seen(app_state: Any) -> SeenSet:
    seen = getattr(app_state, "guard_seen", None)
    if seen is None:
        seen = app_state.guard_seen = SeenSet()
    return seen


def shared_secret(app_state: Any) -> bytes:
    secret = getattr(app_state, "guard_secret", None)
    if secret is None:
        from app.guard.placeholders import guard_secret

        secret = app_state.guard_secret = guard_secret(app_state.settings.cookie_secret)
    return secret


# --- restoring -------------------------------------------------------------------------------

_P = "LMW-SECRET-"


def _hold_from(s: str) -> int:
    """Where the suffix of ``s`` that could still become a tolerant placeholder starts (len(s):
    nothing to hold). At most 25 characters are ever held."""
    start = max(0, len(s) - 25)
    for i in range(start, len(s)):
        if _could_be(s[i:]):
            return i
    return len(s)


def _could_be(t: str) -> bool:
    opener = ""
    if t.startswith("⟦"):
        opener, t = "⟦", t[1:]
    elif t.startswith("[["):
        opener, t = "[[", t[2:]
    elif t == "[":
        return True
    up = t.upper()
    if len(up) <= len(_P):
        return _P.startswith(up) and (bool(up) or bool(opener))
    if not up.startswith(_P):
        return False
    rest = t[len(_P) :]
    hexpart = re.match(r"[0-9a-fA-F]*", rest).group(0)  # type: ignore[union-attr]
    if len(hexpart) < len(rest):
        tail = rest[len(hexpart) :]
        return len(hexpart) == 10 and opener == "[[" and tail == "]"
    return len(hexpart) <= 10


class _Channel:
    __slots__ = ("held",)

    def __init__(self) -> None:
        self.held = ""


class ChatStreamRestorer:
    """Restores placeholders in an OpenAI chat (or legacy completions) SSE stream, frame by frame,
    holding back per channel only a suffix that could still become a placeholder. Holds the
    finish chunk to emit the trailer before it. ``restored`` counts replacements."""

    def __init__(self, restore: RestoreMap, *, trailer: str | None, restore_mode: str = "all"):
        self._map = restore
        self._trailer = trailer
        self._mode = restore_mode
        self._buf = b""
        self._ch: dict[tuple[int, str, int], _Channel] = {}
        self._template: dict[str, Any] | None = None
        self._trailer_sent = False
        self.restored = 0

    # -- text --
    def _text(self, key: tuple[int, str, int], s: str, *, final: bool = False) -> str:
        ch = self._ch.setdefault(key, _Channel())
        s = ch.held + s
        cut = len(s) if final else _hold_from(s)
        ready, ch.held = s[:cut], s[cut:]
        if not ready:
            return ""
        out, n = self._map.restore(ready, json_escape=key[1] == "args")
        self.restored += n
        return out

    def _drain(self, choice: dict[str, Any]) -> None:
        """Flush every held channel of this choice into its delta (before a finish)."""
        idx = choice.get("index", 0)
        for (ci, kind, ti), ch in list(self._ch.items()):
            if ci != idx or not ch.held:
                continue
            rest = self._text((ci, kind, ti), "", final=True)
            if kind == "content":
                d = choice.setdefault("delta", {})
                d["content"] = (d.get("content") or "") + rest
            elif kind == "text":
                choice["text"] = (choice.get("text") or "") + rest
            else:
                d = choice.setdefault("delta", {})
                d.setdefault("tool_calls", []).append(
                    {"index": ti, "function": {"arguments": rest}}
                )

    def _event(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        choices = obj.get("choices")
        if not isinstance(choices, list):
            return [obj]
        finishing = []
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            idx = ch.get("index", 0)
            d = ch.get("delta")
            if isinstance(d, dict):
                if self._template is None:
                    self._template = {
                        k: v for k, v in obj.items() if k != "choices" and k != "usage"
                    }
                if isinstance(d.get("content"), str):
                    d["content"] = self._text((idx, "content", 0), d["content"])
                if self._mode == "all":
                    for tc in d.get("tool_calls") or []:
                        fn = tc.get("function") if isinstance(tc, dict) else None
                        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                            fn["arguments"] = self._text(
                                (idx, "args", tc.get("index", 0)), fn["arguments"]
                            )
            elif isinstance(ch.get("text"), str):
                if self._template is None:
                    self._template = {
                        k: v for k, v in obj.items() if k != "choices" and k != "usage"
                    }
                ch["text"] = self._text((idx, "text", 0), ch["text"])
            if ch.get("finish_reason"):
                finishing.append(ch)
        if finishing:
            for ch in finishing:
                self._drain(ch)
            if (
                self._trailer
                and not self._trailer_sent
                and any(c.get("finish_reason") != "error" for c in finishing)
            ):
                self._trailer_sent = True
                legacy = "delta" not in finishing[0]
                # Text the finishing chunk carries is the end of the answer: it goes out before
                # the trailer, or the trailer would split the answer's last token.
                tail = {**(self._template or {}), "choices": []}
                for ch in finishing:
                    if legacy and ch.get("text"):
                        tail["choices"].append(
                            {"index": ch.get("index", 0), "text": ch["text"], "finish_reason": None}
                        )
                        ch["text"] = ""
                    elif not legacy and (ch.get("delta") or {}).get("content"):
                        tail["choices"].append(
                            {
                                "index": ch.get("index", 0),
                                "delta": {"content": ch["delta"]["content"]},
                                "finish_reason": None,
                            }
                        )
                        ch["delta"]["content"] = ""
                if tail["choices"]:
                    out.append(tail)
                pre = {**(self._template or {}), "choices": []}
                for ch in finishing:
                    piece = "\n\n" + self._trailer
                    if legacy:
                        pre["choices"].append(
                            {"index": ch.get("index", 0), "text": piece, "finish_reason": None}
                        )
                    else:
                        pre["choices"].append(
                            {
                                "index": ch.get("index", 0),
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        )
                out.append(pre)
        out.append(obj)
        return out

    def _frame(self, ev: bytes) -> bytes:
        if not ev.startswith(b"data:"):
            return ev + b"\n\n"
        payload = ev[5:].strip()
        if payload == b"[DONE]":
            return self._flush_held() + ev + b"\n\n"
        try:
            obj = json.loads(payload)
        except ValueError:
            return ev + b"\n\n"
        return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in self._event(obj))

    def _flush_held(self) -> bytes:
        held = [(k, c) for k, c in self._ch.items() if c.held]
        if not held:
            return b""
        frames = []
        for (ci, kind, ti), _c in held:
            rest = self._text((ci, kind, ti), "", final=True)
            if kind == "content":
                ch = {"index": ci, "delta": {"content": rest}, "finish_reason": None}
            elif kind == "text":
                ch = {"index": ci, "text": rest, "finish_reason": None}
            else:
                ch = {
                    "index": ci,
                    "delta": {"tool_calls": [{"index": ti, "function": {"arguments": rest}}]},
                    "finish_reason": None,
                }
            frames.append({**(self._template or {}), "choices": [ch]})
        return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in frames)

    def feed(self, chunk: bytes) -> bytes:
        self._buf += chunk
        out = []
        while True:
            i = self._buf.find(b"\n\n")
            if i < 0:
                break
            ev, self._buf = self._buf[:i], self._buf[i + 2 :]
            out.append(self._frame(ev))
        return b"".join(out)

    def flush(self) -> bytes:
        """End of stream (normal, reaped or the client left): whatever is held goes out as is."""
        out = self._flush_held()
        if self._buf:
            out += self._buf
            self._buf = b""
        return out


def restore_json(
    raw: bytes, restore: RestoreMap, *, trailer: str | None, restore_mode: str = "all"
) -> tuple[bytes, int]:
    """Restore a non-streamed chat / legacy completions body and append the trailer."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return raw, 0
    n = 0

    def r(s: str, json_escape: bool = False) -> str:
        nonlocal n
        out, k = restore.restore(s, json_escape=json_escape)
        n += k
        return out

    for ch in obj.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message")
        if isinstance(msg, dict):
            if isinstance(msg.get("content"), str):
                msg["content"] = r(msg["content"])
                if trailer:
                    msg["content"] += "\n\n" + trailer
            if restore_mode == "all":
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        fn["arguments"] = r(fn["arguments"], True)
        elif isinstance(ch.get("text"), str):
            ch["text"] = r(ch["text"])
            if trailer:
                ch["text"] += "\n\n" + trailer
    return json.dumps(obj).encode(), n


# --- LM Warden's own answers -----------------------------------------------------------------


def _block_text(o: Outcome) -> str:
    lines = _report(o.findings)
    n = len(lines)
    return (
        f"[LM Warden] This request was not sent to the model: it contains {n} sensitive "
        f"value{'s' if n != 1 else ''}: " + "; ".join(lines) + ". "
        "Remove the value (or stop reading that file) and send again. If you send the same "
        "conversation again, the value is replaced by a placeholder instead and the model answers. "
        "To always replace instead of stopping, ask the admin to switch this key's sensitive data "
        "guard to sanitize."
    )


def block_reply(o: Outcome, *, stream: bool, path: str) -> Response:
    text = _block_text(o)
    hdr = {HEADER: f"blocked;n={len(_distinct(o.findings))}"}
    cid = "lmw-guard-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    legacy = not path.endswith("/chat/completions")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not stream:
        if legacy:
            body = {
                "id": cid,
                "object": "text_completion",
                "created": created,
                "model": "lm-warden-guard",
                "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
                "usage": usage,
            }
        else:
            body = {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": "lm-warden-guard",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        return JSONResponse(body, headers=hdr)
    base = {"id": cid, "created": created, "model": "lm-warden-guard"}
    if legacy:
        frames = [
            {
                **base,
                "object": "text_completion",
                "choices": [{"index": 0, "text": text, "finish_reason": None}],
            },
            {
                **base,
                "object": "text_completion",
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            },
        ]
    else:
        obj = "chat.completion.chunk"
        frames = [
            {
                **base,
                "object": obj,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **base,
                "object": obj,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            },
            {
                **base,
                "object": obj,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]
    payload = (
        b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames) + b"data: [DONE]\n\n"
    )

    async def it() -> AsyncIterator[bytes]:
        yield payload

    return StreamingResponse(it(), media_type="text/event-stream", headers=hdr)


def failed_reply(reason: str) -> Response:
    return JSONResponse(
        {
            "error": {
                "message": "LM Warden's sensitive data guard could not check this request, so it "
                "was not sent to the model.",
                "type": "guard_failed",
                "code": reason,
            }
        },
        status_code=422,
        headers={HEADER: f"failed;reason={reason}"},
    )
