"""The guard's Anthropic Messages dialect (spec phase 2): the relayed answer's SSE and JSON with
placeholders restored and the trailer added, LM Warden's block reply and the guard's error.

The stream rewriter does two jobs, one per instance: on the engine side (empty map, trailer
set) it adds the trailer as a text block before ``message_delta``; on the client side (map set,
no trailer) it restores ``text_delta`` and ``input_json_delta`` per content block, holding back
only a suffix that could still become a placeholder."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.guard.placeholders import RestoreMap
from app.guard.proxy import HEADER, Outcome, _block_text, _distinct, _hold_from


def _bytes(c: Any) -> bytes:
    """A body chunk as bytes (Starlette allows str and memoryview too)."""
    if isinstance(c, bytes):
        return c
    if isinstance(c, str):
        return c.encode()
    return bytes(c)


def _frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class AnthropicStreamRewriter:
    def __init__(self, restore: RestoreMap, *, trailer: str | None, restore_mode: str = "all"):
        self._map = restore
        self._trailer = trailer
        self._mode = restore_mode
        self._buf = b""
        self._held: dict[int, str] = {}
        self._kind: dict[int, str] = {}
        self._max_index = -1
        self._trailer_sent = False
        self.restored = 0

    def _text(self, index: int, s: str, *, final: bool = False) -> str:
        s = self._held.pop(index, "") + s
        cut = len(s) if final else _hold_from(s)
        ready, rest = s[:cut], s[cut:]
        if rest:
            self._held[index] = rest
        if not ready:
            return ""
        out, n = self._map.restore(ready, json_escape=self._kind.get(index) == "tool_use")
        self.restored += n
        return out

    def _flush_index(self, index: int) -> bytes:
        if not self._held.get(index):
            return b""
        rest = self._text(index, "", final=True)
        if self._kind.get(index) == "tool_use":
            delta = {"type": "input_json_delta", "partial_json": rest}
        else:
            delta = {"type": "text_delta", "text": rest}
        return _frame(
            "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
        )

    def _trailer_frames(self) -> bytes:
        i = self._max_index + 1
        self._max_index = i
        return (
            _frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            + _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": "\n\n" + (self._trailer or "")},
                },
            )
            + _frame("content_block_stop", {"type": "content_block_stop", "index": i})
        )

    def _event(self, raw: bytes) -> bytes:
        lines = raw.split(b"\n")
        data_line = next((ln for ln in lines if ln.startswith(b"data:")), None)
        if data_line is None:
            return raw + b"\n\n"
        try:
            obj = json.loads(data_line[5:].strip())
        except ValueError:
            return raw + b"\n\n"
        if not isinstance(obj, dict):
            return raw + b"\n\n"
        t = obj.get("type")
        pre = b""
        if t == "content_block_start":
            i = int(obj.get("index", 0))
            self._max_index = max(self._max_index, i)
            cb = obj.get("content_block") or {}
            self._kind[i] = str(cb.get("type"))
        elif t == "content_block_delta" and self._map:
            i = int(obj.get("index", 0))
            d = obj.get("delta") or {}
            if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
                d["text"] = self._text(i, d["text"])
            elif (
                d.get("type") == "input_json_delta"
                and self._mode == "all"
                and isinstance(d.get("partial_json"), str)
            ):
                d["partial_json"] = self._text(i, d["partial_json"])
            else:
                return raw + b"\n\n"
            return _frame("content_block_delta", obj)
        elif t == "content_block_stop":
            pre = self._flush_index(int(obj.get("index", 0)))
        elif t == "message_delta":
            for i in list(self._held):
                pre += self._flush_index(i)
            if self._trailer and not self._trailer_sent:
                self._trailer_sent = True
                pre += self._trailer_frames()
        return pre + raw + b"\n\n"

    def feed(self, chunk: bytes) -> bytes:
        self._buf += chunk
        out = []
        while True:
            i = self._buf.find(b"\n\n")
            if i < 0:
                break
            ev, self._buf = self._buf[:i], self._buf[i + 2 :]
            out.append(self._event(ev))
        return b"".join(out)

    def flush(self) -> bytes:
        out = b"".join(self._flush_index(i) for i in list(self._held))
        if self._buf:
            out += self._buf
            self._buf = b""
        return out


def _restore_values(obj: Any, restore: RestoreMap) -> tuple[Any, int]:
    if isinstance(obj, str):
        return restore.restore(obj)
    n = 0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k], k_n = _restore_values(v, restore)
            n += k_n
        return out, n
    if isinstance(obj, list):
        items = []
        for v in obj:
            r, k_n = _restore_values(v, restore)
            items.append(r)
            n += k_n
        return items, n
    return obj, 0


def restore_message_json(
    raw: bytes, restore: RestoreMap, *, trailer: str | None, restore_mode: str = "all"
) -> tuple[bytes, int]:
    """A non-streamed Messages answer with placeholders restored and the trailer appended."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return raw, 0
    if not isinstance(obj, dict) or not isinstance(obj.get("content"), list):
        return raw, 0
    n = 0
    for block in obj["content"]:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            block["text"], k = restore.restore(block["text"])
            n += k
        elif block.get("type") == "tool_use" and restore_mode == "all":
            block["input"], k = _restore_values(block.get("input"), restore)
            n += k
    if trailer:
        obj["content"].append({"type": "text", "text": "\n\n" + trailer})
    return json.dumps(obj, ensure_ascii=False).encode(), n


def block_reply(o: Outcome, *, stream: bool) -> Response:
    text = _block_text(o)
    hdr = {HEADER: f"blocked;n={len(_distinct(o.findings))}"}
    mid = "msg_lmwguard_" + uuid.uuid4().hex[:16]
    usage = {"input_tokens": 0, "output_tokens": 0}
    if not stream:
        return JSONResponse(
            {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "model": "lm-warden-guard",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": usage,
            },
            headers=hdr,
        )
    payload = (
        _frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": "lm-warden-guard",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            },
        )
        + _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        + _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
        + _frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        + _frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        + _frame("message_stop", {"type": "message_stop"})
    )

    async def it() -> AsyncIterator[bytes]:
        yield payload

    return StreamingResponse(it(), media_type="text/event-stream", headers=hdr)


def error_reply(status: int, kind: str, message: str, header: str) -> Response:
    return JSONResponse(
        {"type": "error", "error": {"type": kind, "message": message}},
        status_code=status,
        headers={HEADER: header},
    )


def failed_reply(reason: str) -> Response:
    return error_reply(
        422,
        "invalid_request_error",
        "LM Warden's sensitive data guard could not check this request, so it was not sent.",
        f"failed;reason={reason}",
    )


def wrap_relay(
    resp: StreamingResponse,
    o: Outcome,
    *,
    on_engine_chunk: Any = None,
) -> StreamingResponse:
    """Rewrite a relayed answer in place: the engine view (trailer added) goes to
    ``on_engine_chunk`` (the recorder), the client gets values restored. Only 2xx SSE and JSON
    answers are touched; anything encoded or an error passes through as it came."""
    raw_headers = dict((k.lower(), v) for k, v in resp.raw_headers)
    ctype = raw_headers.get(b"content-type", b"").decode("latin-1")
    encoded = raw_headers.get(b"content-encoding", b"identity").lower() not in (b"", b"identity")
    ok = 200 <= resp.status_code < 300
    is_sse = ctype.startswith("text/event-stream")
    is_json = ctype.startswith("application/json")
    upstream = resp.body_iterator
    trailer = o.trailer
    restore = o.restore if o.restore else None
    mode = o.restore_mode

    def feed(chunk: bytes) -> None:
        if on_engine_chunk is not None:
            try:
                on_engine_chunk(chunk)
            except Exception:  # noqa: BLE001 - recording never fails a relay
                pass

    if not ok or encoded or not (is_sse or is_json):

        async def passthrough() -> AsyncIterator[bytes]:
            async for c in upstream:
                b = _bytes(c)
                feed(b)
                yield b

        resp.body_iterator = passthrough()
        return resp

    if is_sse:
        engine = AnthropicStreamRewriter(RestoreMap(), trailer=trailer)
        client = (
            AnthropicStreamRewriter(restore, trailer=None, restore_mode=mode) if restore else None
        )

        async def sse() -> AsyncIterator[bytes]:
            async def emit(b: bytes) -> AsyncIterator[bytes]:
                if not b:
                    return
                feed(b)
                out = client.feed(b) if client is not None else b
                if out:
                    yield out

            async for c in upstream:
                async for out in emit(engine.feed(_bytes(c))):
                    yield out
            async for out in emit(engine.flush()):
                yield out
            if client is not None:
                tail = client.flush()
                if tail:
                    yield tail

        resp.body_iterator = sse()
    else:

        async def whole() -> AsyncIterator[bytes]:
            parts = [_bytes(c) async for c in upstream]
            engine_view, _ = restore_message_json(b"".join(parts), RestoreMap(), trailer=trailer)
            feed(engine_view)
            if restore is None:
                yield engine_view
                return
            out, _ = restore_message_json(engine_view, restore, trailer=None, restore_mode=mode)
            yield out

        resp.body_iterator = whole()
    resp.raw_headers = [(k, v) for k, v in resp.raw_headers if k.lower() != b"content-length"]
    resp.raw_headers.append(
        (HEADER.lower().encode(), f"sanitized;n={len(_distinct(o.findings))}".encode())
    )
    return resp


__all__ = [
    "AnthropicStreamRewriter",
    "block_reply",
    "error_reply",
    "failed_reply",
    "restore_message_json",
    "wrap_relay",
]
