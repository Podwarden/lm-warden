"""Capture the assistant message LM Warden generated (cobs2 spec section 6.3).

The reply is hashed as a lone message, so its id equals the id of the same
message when a client sends it back in a later request. Bounded and
fail-open: any problem means "no reply message", never an error.
"""

from __future__ import annotations

import logging
from typing import Any

from app.proxy.reaggregate import StreamAggregator, parse_sse_event

logger = logging.getLogger(__name__)

MAX_REPLY_BYTES = 4 << 20
#: Guard against pathological SSE envelopes (the reply bound counts payload only).
MAX_RAW_BYTES = 64 << 20


def _size(msg: dict[str, Any]) -> int:
    n = 0
    for k in ("content", "reasoning_content", "reasoning"):
        v = msg.get(k)
        if isinstance(v, str):
            n += len(v)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(fn, dict):
            n += len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))
    return n


class ReplyCapture:
    def __init__(self) -> None:
        self._agg: StreamAggregator | None = StreamAggregator(is_chat=True)
        self._msg: dict[str, Any] | None = None
        self._bytes = 0
        self._raw = 0
        self._finished = False
        self._over = False

    def feed_sse(self, line: bytes) -> None:
        if self._over or line.lstrip()[:5] != b"data:":
            return
        try:
            self._raw += len(line)
            if self._raw > MAX_RAW_BYTES:
                self._abandon()
                return
            ev = parse_sse_event(line)
            if ev is None:
                if b"[DONE]" in line:
                    self._finished = True
                return
            self._absorb(ev)
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: reply feed failed", exc_info=True)
            self._abandon()

    def feed_event(self, ev: dict[str, Any], raw_len: int = 0) -> None:
        """Feed an already-parsed SSE event (the caller parsed the frame)."""
        if self._over:
            return
        try:
            self._raw += raw_len
            if self._raw > MAX_RAW_BYTES:
                self._abandon()
                return
            self._absorb(ev)
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: reply feed failed", exc_info=True)
            self._abandon()

    def _abandon(self) -> None:
        self._over = True
        self._agg = None

    def _absorb(self, ev: dict[str, Any]) -> None:
        for ch in ev.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            if ch.get("finish_reason") is not None:
                self._finished = True
            d = ch.get("delta")
            if isinstance(d, dict):
                self._bytes += _size(d)
        if self._bytes > MAX_REPLY_BYTES:
            self._abandon()
            return
        if self._agg is not None:
            self._agg.feed_event(ev)

    @property
    def complete(self) -> bool:
        """True once the engine said it finished (a finish_reason or a full JSON body)."""
        return self._finished

    def set_json(self, body: dict[str, Any]) -> None:
        try:
            msg = ((body.get("choices") or [{}])[0] or {}).get("message")
            if isinstance(msg, dict) and _size(msg) <= MAX_REPLY_BYTES:
                self._msg = msg
                self._finished = True
                self._over = False
            else:
                self._over = True
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: reply set_json failed", exc_info=True)
            self._over = True

    def message(self) -> dict[str, Any] | None:
        try:
            if self._over:
                return None
            msg = self._msg
            if msg is None and self._agg is not None:
                built = self._agg.build()
                msg = ((built.get("choices") or [{}])[0] or {}).get("message")
            if not isinstance(msg, dict):
                return None
            out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
            for k in ("reasoning_content", "reasoning"):
                if isinstance(msg.get(k), str):
                    out["reasoning_content"] = msg[k]
                    break
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            return out
        except Exception:  # noqa: BLE001
            logger.debug("cobs2: reply message failed", exc_info=True)
            return None
