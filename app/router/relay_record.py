"""Record relayed (router passthrough) ``/v1/messages`` requests for the sessions
pages: cobs2 ids and lineage from the translated body, tools and the reply id
from the captured response, timings and the upstream's own usage.

Nothing here touches the relayed byte path beyond one ``list.append`` per chunk
(``on_chunk``): translation, hashing and parsing run in tasks the relay never
awaits, in worker threads where they cost CPU. Every entry point is fail-open.
A relayed row is ``relayed=True``: local cache observation, the cache panel,
replica placement and every local stats reader ignore it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from uuid import uuid4

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

    from app.db.repos.tokens import TokenRow

logger = logging.getLogger(__name__)

CAPTURE_MAX_BYTES = 8 << 20
FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}
#: Strong refs to in-flight recording tasks (the loop keeps only weak ones).
_TASKS: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> asyncio.Task[Any]:
    task = asyncio.get_running_loop().create_task(coro)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


@dataclass
class RelayReply:
    message: dict[str, Any] | None
    tool_names: list[str]
    stop_reason: str | None
    usage: dict[str, int] = field(default_factory=dict)
    complete: bool = False


def _decode(raw: bytes, encoding: str) -> bytes | None:
    enc = (encoding or "identity").strip().lower()
    if enc in ("", "identity"):
        return raw
    if enc == "gzip":
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    if enc == "deflate":
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    return None


def _ints(d: Any) -> dict[str, int]:
    if not isinstance(d, dict):
        return {}
    return {k: int(v) for k, v in d.items() if isinstance(v, int) and not isinstance(v, bool)}


def assemble_reply(raw: bytes, *, is_sse: bool, encoding: str) -> RelayReply | None:
    """Fold an Anthropic Messages response (SSE or JSON) into the assistant
    message, tool names, stop reason and usage. None when it cannot be read."""
    try:
        data = _decode(raw, encoding)
    except zlib.error:
        return None
    if data is None:
        return None
    if not is_sse:
        try:
            body = json.loads(data)
        except ValueError:
            return None
        if not isinstance(body, dict) or not isinstance(body.get("content"), list):
            return None
        jblocks = [b for b in body["content"] if isinstance(b, dict)]
        return RelayReply(
            {"role": "assistant", "content": jblocks},
            [str(b.get("name")) for b in jblocks if b.get("type") == "tool_use" and b.get("name")],
            body.get("stop_reason") if isinstance(body.get("stop_reason"), str) else None,
            _ints(body.get("usage")),
            True,
        )
    blocks: dict[int, dict[str, Any]] = {}
    partial: dict[int, list[str]] = {}
    usage: dict[str, int] = {}
    stop: str | None = None
    complete = False
    for frame in data.split(b"\n\n"):
        payload = None
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                payload = line[5:].strip()
        if not payload:
            continue
        try:
            ev = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        typ = ev.get("type")
        if typ == "message_start":
            usage.update(_ints((ev.get("message") or {}).get("usage")))
        elif typ == "content_block_start":
            cb = ev.get("content_block")
            if isinstance(cb, dict) and isinstance(ev.get("index"), int):
                blocks[ev["index"]] = dict(cb)
        elif typ == "content_block_delta":
            i, d = ev.get("index"), ev.get("delta")
            if not isinstance(i, int) or not isinstance(d, dict):
                continue
            b = blocks.get(i)
            if b is None:
                continue
            dt = d.get("type")
            if dt == "text_delta":
                b["text"] = b.get("text", "") + str(d.get("text", ""))
            elif dt == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + str(d.get("thinking", ""))
            elif dt == "signature_delta":
                b["signature"] = b.get("signature", "") + str(d.get("signature", ""))
            elif dt == "input_json_delta":
                partial.setdefault(i, []).append(str(d.get("partial_json", "")))
        elif typ == "message_delta":
            d = ev.get("delta")
            if isinstance(d, dict) and isinstance(d.get("stop_reason"), str):
                stop = d["stop_reason"]
            usage.update(_ints(ev.get("usage")))
        elif typ == "message_stop":
            complete = True
    for i, parts in partial.items():
        try:
            blocks[i]["input"] = json.loads("".join(parts)) if parts else {}
        except ValueError:
            complete = False
    ordered = [blocks[i] for i in sorted(blocks)]
    return RelayReply(
        {"role": "assistant", "content": ordered},
        [str(b.get("name")) for b in ordered if b.get("type") == "tool_use" and b.get("name")],
        stop,
        usage,
        complete,
    )


def _compute(oai: dict[str, Any], key: Any, secret: str) -> Any:
    """The cobs2 pure pass over the translated body (worker thread)."""
    from app.proxy.routes import _cobs2_compute

    return _cobs2_compute(oai, key, secret)


def _translate(body: bytes) -> tuple[Any, dict[str, Any]]:
    from app.proxy.messages_schemas import MessagesRequest
    from app.proxy.messages_translate import to_openai_request

    req = MessagesRequest.model_validate_json(body)
    return req, to_openai_request(req)


def _reply_ids(message: dict[str, Any], model: str, key: Any, salt: bytes) -> tuple[Any, Any]:
    """The reply's ids exactly as the NEXT request will carry it: translated as
    an assistant message by the same Messages→chat translation."""
    from app.cobs2.canonical import canonicalize
    from app.cobs2.ids import ids_for
    from app.proxy.messages_schemas import MessagesRequest
    from app.proxy.messages_translate import to_openai_request

    req = MessagesRequest.model_validate(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "."}, message],
        }
    )
    oai = to_openai_request(req)
    msgs = [m for m in oai.get("messages", []) if isinstance(m, dict)]
    last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    if last is None:
        return None, None
    canon = canonicalize({"messages": [last]})
    if canon is None or not canon.messages:
        return None, None
    return ids_for(canon, key, salt), canon


@dataclass
class _Prep:
    model_in: str | None = None
    session_key: str | None = None
    parent_session_key: str | None = None
    session_id: str | None = None
    session_source: str | None = None
    pre: Any = None
    seen: Any = None


class RelayRecorder:
    """One relayed ``/v1/messages`` request's history row. Never raises."""

    def __init__(self, request: Request, token: TokenRow, body: bytes, *, upstream: str) -> None:
        self.request_id = uuid4().hex
        self._request = request
        self._token = token
        self._body = body
        try:
            self._upstream = urlparse(upstream).hostname
        except Exception:  # noqa: BLE001
            self._upstream = None
        self._t0 = time.monotonic()
        self._started_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._ttfb: float | None = None
        self._chunks: list[bytes] = []
        self._size = 0
        self._over = False
        self._usage_in: int | None = None
        self._usage_out: int | None = None
        self._prep: asyncio.Task[_Prep] | None = None
        self._closed = False

    # -- relay callbacks (hot path: cheap, guarded) -----------------------
    def start(self) -> None:
        try:
            self._prep = _spawn(self._prepare())
        except Exception:  # noqa: BLE001
            logger.debug("relay record: start failed", exc_info=True)

    def on_chunk(self, chunk: bytes) -> None:
        if self._over:
            return
        self._size += len(chunk)
        if self._size > CAPTURE_MAX_BYTES:
            self._over = True
            self._chunks.clear()
            return
        self._chunks.append(chunk)

    def on_ttfb(self) -> None:
        if self._ttfb is None:
            self._ttfb = time.monotonic()

    def note_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self._usage_in, self._usage_out = input_tokens, output_tokens

    def on_close(self, resp: Response) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            status = int(resp.status_code)
            headers = resp.headers
            ctype = headers.get("content-type", "")
            enc = headers.get("content-encoding", "identity")
            # Both clocks now: the row is written later, but it describes this moment.
            _spawn(self._finish(status, ctype, enc, time.monotonic(), time.time()))
        except Exception:  # noqa: BLE001
            logger.debug("relay record: close failed", exc_info=True)

    # -- off the stream path -------------------------------------------------
    async def _prepare(self) -> _Prep:
        from app.proxy import forest_fields
        from app.proxy.routes import _cobs2_link
        from app.proxy.routes_messages import anthropic_affinity_key
        from app.proxy.session_id import extract_session

        prep = _Prep()
        state = self._request.app.state
        secret = state.settings.cookie_secret
        try:
            req, oai = await asyncio.to_thread(_translate, self._body)
        except Exception:  # noqa: BLE001
            logger.debug("relay record: translate failed", exc_info=True)
            return prep
        prep.model_in = req.model
        try:
            sess = extract_session(oai, self._request.headers, anthropic_affinity_key(req))
            prep.session_id = sess.id
            prep.session_source = sess.source
            prep.session_key = forest_fields.hkey(secret, sess.id)
            prep.parent_session_key = forest_fields.hkey(secret, sess.parent)
        except Exception:  # noqa: BLE001
            logger.debug("relay record: session failed", exc_info=True)
        key = getattr(state, "cobs2_key", None)
        try:
            snap = {k: oai[k] for k in ("messages", "tools", "cache_salt") if k in oai}
            prep.pre = await asyncio.to_thread(_compute, snap, key, secret)
        except Exception:  # noqa: BLE001
            logger.debug("relay record: ids failed", exc_info=True)
            prep.pre = None
        if prep.pre is not None:
            try:
                prep.seen = _cobs2_link(
                    prep.pre,
                    lineage=getattr(state, "cobs2_lineage", None),
                    model_id="relay:" + str(req.model),
                    session_key=prep.session_key,
                    parent_session_key=prep.parent_session_key,
                    token_scope=forest_fields.hkey(secret, self._token.id),
                    inflight_heads=[],
                    request_id=self.request_id,
                )
            except Exception:  # noqa: BLE001
                logger.debug("relay record: lineage failed", exc_info=True)
        return prep

    async def _finish(
        self, status: int, ctype: str, enc: str, t_end: float, wall_end: float
    ) -> None:
        try:
            await self._record(status, ctype, enc, t_end, wall_end)
        except Exception:  # noqa: BLE001
            logger.debug("relay record: finish failed", exc_info=True)

    async def _record(
        self, status: int, ctype: str, enc: str, t_end: float, wall_end: float
    ) -> None:
        from app.cobs2.shape import break_index, shape_json
        from app.proxy import forest_fields
        from app.proxy.request_registry import LiveRequest, finished_record
        from app.proxy.routes import _COBS2_REPLY_SKETCH_CHARS, _cobs2_text
        from app.sessions.refs import sess_ref
        from app.utils.client_ip import client_ip

        state = self._request.app.state
        secret = state.settings.cookie_secret
        prep = _Prep()
        if self._prep is not None:
            try:
                prep = await self._prep
            except Exception:  # noqa: BLE001
                prep = _Prep()
        ok = 200 <= status < 300
        reply: RelayReply | None = None
        if ok and not self._over and self._chunks:
            raw = b"".join(self._chunks)
            self._chunks = []
            try:
                reply = await asyncio.to_thread(
                    assemble_reply, raw, is_sse=ctype.startswith("text/event-stream"), encoding=enc
                )
            except Exception:  # noqa: BLE001
                reply = None
        usage = reply.usage if reply is not None else {}
        inp = usage.get("input_tokens", self._usage_in or 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        out = usage.get("output_tokens", self._usage_out or 0)
        model = prep.model_in or "unknown"
        pre, seen = prep.pre, prep.seen
        obs = seen.obs if seen is not None else None
        live = LiveRequest(
            id=self.request_id,
            token_id=self._token.id,
            token_name=self._token.name,
            client_ip=client_ip(self._request),
            model=model,
            model_row_id=model,
            path="/v1/messages",
            prompt_tokens=int(inp) + int(cache_read) + int(cache_create),
            max_model_len=None,
            started_monotonic=self._t0,
            started_iso=self._started_iso,
        )
        live.completion_tokens = int(out)
        live.cached_tokens = int(cache_read)
        live.first_token_monotonic = self._ttfb
        live.finish_reason = (
            "error"
            if not ok
            else (
                FINISH.get(reply.stop_reason, reply.stop_reason)
                if reply and reply.stop_reason
                else None
            )
        )
        live.session_id = prep.session_id
        live.session_source = prep.session_source
        live.session_key = prep.session_key
        live.parent_session_key = prep.parent_session_key
        live.relayed = True
        live.upstream = self._upstream
        if reply is not None and reply.tool_names:
            live.tools_out = [
                [forest_fields.tool_family(n), forest_fields.hkey(secret, n, 8)]
                for n in reply.tool_names
            ]
        if pre is not None:
            ids = pre.ids
            live.cobs2_head = ids.head
            live.cobs2_node = ids.nodes[-1] if ids.nodes else None
            live.cobs2_conv = obs.conv if obs is not None and obs.conv else pre.conv
            live.cobs2_toolset = pre.toolset
            if seen is not None:
                live.cobs2_compaction_request = seen.compaction
            if obs is not None:
                live.cobs2_edge = obs.edge
                live.cobs2_parent_node = obs.parent
                live.cobs2_parent_req = obs.parent_req
                live.cobs2_loose_beyond_exact = obs.loose_match_beyond_exact
                live.cobs2_session = obs.session
            if pre.shape is not None:
                try:
                    live.cobs2_shape = shape_json(pre.shape, break_index(obs))
                except Exception:  # noqa: BLE001
                    pass
        if not live.cobs2_session and prep.session_key:
            live.cobs2_session = prep.session_key
        if live.cobs2_session:
            live.cobs2_sess_ref = sess_ref(secret, self._token.id, live.cobs2_session)
        key = getattr(state, "cobs2_key", None)
        if (
            reply is not None
            and reply.complete
            and reply.message
            and key is not None
            and pre is not None
        ):
            try:
                rids, canon = _reply_ids(reply.message, model, key, pre.salt)
                if rids is not None and rids.msgs:
                    live.cobs2_reply_msg = rids.msgs[0]
                    live.cobs2_reply_loose_msg = rids.loose_msgs[0]
                    lineage = getattr(state, "cobs2_lineage", None)
                    if lineage is not None and obs is not None and pre.ids.nodes:
                        lineage.remember_reply(
                            model_id="relay:" + model,
                            scope=pre.salt,
                            session=obs.session,
                            node=pre.ids.nodes[-1],
                            reply_ids=rids,
                            reply_text_loose=_cobs2_text(
                                canon.messages[0], _COBS2_REPLY_SKETCH_CHARS
                            ),
                        )
            except Exception:  # noqa: BLE001
                logger.debug("relay record: reply ids failed", exc_info=True)
        history = getattr(state, "request_history", None)
        if history is not None:
            history.record(finished_record(live, now=t_end), finished_at=wall_end)
