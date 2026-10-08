"""Anthropic Messages API <-> OpenAI chat completions translation (#281).

``POST /v1/messages`` (routes_messages.py) turns an Anthropic request into the
OpenAI chat body every engine behind the warden already speaks, hands it to
the ordinary ``/v1/chat/completions`` forward, and turns the answer back.
Everything here is pure: no I/O, no app state, so each direction is testable
on dicts alone.

Decisions that are not obvious from either spec:

* **Tool results come first.** An Anthropic user turn may carry
  ``tool_result`` blocks next to text. OpenAI wants each result as its own
  ``role: tool`` message *immediately* after the assistant turn that made the
  calls, so the results are emitted first and the turn's remaining text (and
  images) follow as one user message.
* **Images in tool results move to the user message.** An OpenAI ``tool``
  message carries text only. Claude Code's Read tool answers an image file
  with an image block inside ``tool_result``; dropping it would blind a vision
  model, so it rides in the user message that follows, labelled with the
  ``tool_use_id`` it came from.
* **Streamed tool calls are emitted whole, at the end.** Text and thinking
  stream as they arrive. Tool-call fragments are merged by
  ``StreamAggregator`` (reaggregate.py -- the same per-index merge the
  runaway re-aggregation uses) and each call goes out as one
  ``content_block_start`` / ``input_json_delta`` / ``content_block_stop``
  triple once the engine is done. That lets the arguments be validated as
  JSON before a client's SDK tries to parse them, and copes with engines
  that interleave fragments of parallel calls -- which Anthropic's
  one-block-at-a-time event order cannot express.
* **Reasoning becomes ``thinking`` only when asked for.** The Messages API
  returns thinking blocks only to a request with ``thinking`` enabled, so the
  engine's reasoning channel is dropped otherwise. Thinking blocks a client
  sends back are dropped on the way in: they carry an Anthropic signature
  no local engine can check, and chat templates discard past reasoning.
"""

import json
import logging
from typing import Any
from uuid import uuid4

from app.proxy.messages_schemas import CountTokensRequest, MessagesRequest
from app.proxy.reaggregate import StreamAggregator, parse_sse_event
from app.utils.sse import delta_reasoning

logger = logging.getLogger(__name__)


class TranslationError(ValueError):
    """The request uses something the OpenAI side cannot carry (-> 400)."""


# --------------------------------------------------------------------------
# Request: Anthropic -> OpenAI
# --------------------------------------------------------------------------


def _text_of(blocks: str | list[Any] | None, where: str) -> str:
    """Join the text of a ``system`` value or a tool_result's content."""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    parts = []
    for blk in blocks:
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(str(blk.get("text", "")))
        else:
            kind = blk.get("type") if isinstance(blk, dict) else type(blk).__name__
            raise TranslationError(f"{where}: unsupported block type '{kind}'")
    return "\n\n".join(parts)


def _image_part(blk: dict[str, Any]) -> dict[str, Any]:
    src = blk.get("source")
    if not isinstance(src, dict):
        raise TranslationError("image block without a 'source'")
    if src.get("type") == "base64":
        media = src.get("media_type") or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{src.get('data', '')}"},
        }
    if src.get("type") == "url" and isinstance(src.get("url"), str):
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    raise TranslationError(f"image source type '{src.get('type')}' is not supported")


def _user_turn(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One Anthropic user message -> OpenAI ``tool`` messages, then a user one."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    tool_msgs: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for blk in content:
        kind = blk.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(blk.get("text", ""))})
        elif kind == "image":
            parts.append(_image_part(blk))
        elif kind == "document":
            src = blk.get("source") or {}
            if src.get("type") != "text":
                raise TranslationError(
                    "document blocks are supported only with a plain-text source"
                )
            parts.append({"type": "text", "text": str(src.get("data", ""))})
        elif kind == "tool_result":
            tool_msgs.append(_tool_result(blk, parts))
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            raise TranslationError(f"user content block type '{kind}' is not supported")
    out = list(tool_msgs)
    if parts:
        if all(p["type"] == "text" for p in parts):
            out.append({"role": "user", "content": "\n\n".join(p["text"] for p in parts)})
        else:
            out.append({"role": "user", "content": parts})
    return out


def _tool_result(blk: dict[str, Any], user_parts: list[dict[str, Any]]) -> dict[str, Any]:
    tool_use_id = blk.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        raise TranslationError("tool_result block without a 'tool_use_id'")
    raw = blk.get("content")
    texts: list[str] = []
    if isinstance(raw, str):
        texts.append(raw)
    elif isinstance(raw, list):
        for part in raw:
            ptype = part.get("type") if isinstance(part, dict) else None
            if ptype == "text":
                texts.append(str(part.get("text", "")))
            elif ptype == "image":
                # See the module docstring: a tool message cannot carry it.
                user_parts.append(
                    {"type": "text", "text": f"[image from tool result {tool_use_id}]"}
                )
                user_parts.append(_image_part(part))
                texts.append("[image attached in the next user message]")
            else:
                raise TranslationError(f"tool_result content block type '{ptype}' is not supported")
    text = "\n\n".join(texts)
    if blk.get("is_error"):
        text = f"Error: {text}" if text else "Error"
    return {"role": "tool", "tool_call_id": tool_use_id, "content": text}


def _assistant_turn(content: str | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for blk in content:
        kind = blk.get("type")
        if kind == "text":
            texts.append(str(blk.get("text", "")))
        elif kind == "tool_use":
            calls.append(
                {
                    "id": blk.get("id") or f"toolu_{uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": blk.get("name", ""),
                        "arguments": json.dumps(blk.get("input") or {}),
                    },
                }
            )
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            raise TranslationError(f"assistant content block type '{kind}' is not supported")
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else None}
    if calls:
        msg["tool_calls"] = calls
    elif msg["content"] is None:
        msg["content"] = ""
    return msg


def _tools(req: MessagesRequest | CountTokensRequest) -> list[dict[str, Any]]:
    out = []
    for tool in req.tools or []:
        if tool.input_schema is None:
            # An Anthropic server tool (web search, code execution, ...): the
            # engine cannot run it, and a client like Claude Code only sends
            # one on a dedicated request. Leave it out rather than fail the
            # whole request.
            logger.debug("messages: dropping server tool %r (type %r)", tool.name, tool.type)
            continue
        fn: dict[str, Any] = {"name": tool.name, "parameters": tool.input_schema}
        if tool.description:
            fn["description"] = tool.description
        out.append({"type": "function", "function": fn})
    return out


def system_note(text: str) -> str:
    return f"[system message]\n{text}\n[end system message]"


def prefix_content(content: str | list[dict[str, Any]], note: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return f"{note}\n\n{content}" if content else note
    return [{"type": "text", "text": note}, *content]


def append_note(messages: list[dict[str, Any]], note: str) -> None:
    """Put a trailing system note after everything already translated.

    Joins the last message when it is a plain user turn; otherwise (an
    assistant turn, or ``tool`` messages that must stay adjacent to their
    ``tool_calls``) it becomes a user message of its own.
    """
    last = messages[-1] if messages else None
    if last is not None and last["role"] == "user":
        content = last["content"]
        if isinstance(content, str):
            last["content"] = f"{content}\n\n{note}" if content else note
        else:
            content.append({"type": "text", "text": note})
    else:
        messages.append({"role": "user", "content": note})


def to_openai_request(req: MessagesRequest | CountTokensRequest) -> dict[str, Any]:
    """Build the OpenAI chat-completions body for ``req``.

    ``role: "system"`` entries inside ``messages`` (Claude Code sends them
    mid-conversation) cannot go through as-is: chat templates such as Qwen's
    accept a system message only first. A leading run is merged into the
    top-level system prompt. A later one is folded, delimited, into the NEXT
    user turn -- the first thing the model answers after the note, so its
    position in the conversation is kept -- as a leading text block, which
    after ``tool_result`` blocks lands in the user message that follows the
    ``tool`` messages and so never separates a ``tool_call`` from its result.
    With no user turn after it, the note is appended to the end of the
    conversation (see ``append_note``).
    """
    messages: list[dict[str, Any]] = []
    system_parts = [_text_of(req.system, "system")]
    pending: list[str] = []
    started = False  # a user/assistant message has been seen
    for i, m in enumerate(req.messages):
        if m.role == "system":
            text = _text_of(m.content, f"messages.{i}")
            if not text.strip():
                continue
            if started:
                pending.append(system_note(text))
            else:
                system_parts.append(text)
            continue
        started = True
        if m.role == "user":
            content = m.content
            if pending:
                content = prefix_content(content, "\n\n".join(pending))
                pending = []
            messages.extend(_user_turn(content))
        else:
            messages.append(_assistant_turn(m.content))
    if pending:
        append_note(messages, "\n\n".join(pending))
    system = "\n\n".join(p for p in system_parts if p)
    if system:
        messages.insert(0, {"role": "system", "content": system})

    body: dict[str, Any] = {"model": req.model, "messages": messages}
    tools = _tools(req)
    if tools:
        body["tools"] = tools
        choice = req.tool_choice
        if choice is not None:
            if choice.type == "auto":
                body["tool_choice"] = "auto"
            elif choice.type == "any":
                body["tool_choice"] = "required"
            elif choice.type == "none":
                body["tool_choice"] = "none"
            else:
                if not choice.name:
                    raise TranslationError("tool_choice type 'tool' needs a 'name'")
                body["tool_choice"] = {"type": "function", "function": {"name": choice.name}}
            if choice.disable_parallel_tool_use:
                body["parallel_tool_calls"] = False

    if isinstance(req, MessagesRequest):
        body["max_tokens"] = req.max_tokens
        if req.stop_sequences:
            body["stop"] = req.stop_sequences
        for field in ("temperature", "top_p", "top_k"):
            value = getattr(req, field)
            if value is not None:
                body[field] = value
        if req.stream:
            body["stream"] = True
            # include_usage: the engine's own counts on the last frame, for
            # message_delta and for the warden's accounting. continuous_usage_stats
            # (vLLM): usage on EVERY frame, so message_start can report
            # input_tokens up front -- clients (Claude Code's context meter and
            # auto-compact) read it there. Engines that do not know the flag
            # ignore it.
            body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        # Thinking is a chat-template decision, not a sampling one: vLLM only
        # exposes it through `chat_template_kwargs`, so there is no OpenAI
        # field to set. The Messages API returns thinking blocks only when a
        # request turns them on (see thinking_requested), but without this the
        # engine was never told -- it reasoned on every request and the
        # translator just discarded the blocks, spending time and tokens on
        # output nobody asked for. Only the OFF case is sent, matching
        # chat2/routes_turn.py: a template that has never heard of the flag
        # must keep its own default, which is what "on" has to mean here.
        if not thinking_requested(req):
            body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def count_text(oai: dict[str, Any]) -> str:
    """Everything in a translated body that costs prompt tokens, as text."""
    parts: list[str] = []
    for m in oai.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if p.get("type") == "text")
        for call in m.get("tool_calls") or []:
            parts.append(call["function"]["name"])
            parts.append(call["function"]["arguments"])
    if oai.get("tools"):
        parts.append(json.dumps(oai["tools"]))
    return "\n".join(parts)


def thinking_requested(req: MessagesRequest) -> bool:
    return req.thinking is not None and req.thinking.type != "disabled"


# --------------------------------------------------------------------------
# Response: OpenAI -> Anthropic
# --------------------------------------------------------------------------


def new_message_id() -> str:
    return f"msg_{uuid4().hex[:24]}"


def map_stop_reason(
    finish_reason: Any,
    matched_stop: Any,
    stop_sequences: list[str] | None,
    has_tool_use: bool,
) -> tuple[str, str | None]:
    """OpenAI ``finish_reason`` -> Anthropic ``(stop_reason, stop_sequence)``.

    ``matched_stop`` is vLLM's per-choice ``stop_reason``: the stop string
    that ended the generation, which is exactly Anthropic's ``stop_sequence``.
    A turn that produced tool calls is ``tool_use`` whatever the engine said:
    some engines finish such a turn with ``stop``, and Claude Code acts on
    the stop reason.
    """
    if finish_reason in ("tool_calls", "function_call") or (
        has_tool_use and finish_reason in (None, "stop")
    ):
        return "tool_use", None
    if finish_reason == "length":
        return "max_tokens", None
    if finish_reason == "content_filter":
        return "refusal", None
    if isinstance(finish_reason, str) and finish_reason.startswith("runaway_"):
        # The warden's runaway detector cut the generation (runaway.py): the
        # output is truncated, which is what max_tokens tells a client.
        return "max_tokens", None
    if isinstance(matched_stop, str) and stop_sequences and matched_stop in stop_sequences:
        return "stop_sequence", matched_stop
    return "end_turn", None


def _tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        logger.warning("messages: tool call arguments are not JSON; sending {}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_use_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid4().hex[:24]}",
                "name": fn.get("name") or "",
                "input": _tool_input(fn.get("arguments")),
            }
        )
    return blocks


def anthropic_usage(usage: Any) -> dict[str, int]:
    """OpenAI usage -> Anthropic usage. A prefix-cache hit the engine reports
    (``prompt_tokens_details.cached_tokens``) is Anthropic's cache read, and
    Anthropic's ``input_tokens`` excludes it."""
    usage = usage if isinstance(usage, dict) else {}
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cached = min(cached, prompt)
    return {
        "input_tokens": prompt - cached,
        "output_tokens": usage.get("completion_tokens") or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def to_anthropic_message(
    oai: dict[str, Any],
    *,
    model: str,
    stop_sequences: list[str] | None,
    emit_thinking: bool,
) -> dict[str, Any]:
    """A non-streamed OpenAI chat completion -> an Anthropic ``message``."""
    choices = oai.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    reasoning = delta_reasoning(message)
    if emit_thinking and reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    tools = tool_use_blocks(message)
    content.extend(tools)
    stop_reason, stop_sequence = map_stop_reason(
        choice.get("finish_reason"), choice.get("stop_reason"), stop_sequences, bool(tools)
    )
    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": anthropic_usage(oai.get("usage")),
    }


def sse_frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class AnthropicStreamTranslator:
    """OpenAI ``chat.completion.chunk`` SSE bytes -> Anthropic SSE events.

    Feed upstream bytes with :meth:`feed_bytes`; call :meth:`finish` once the
    upstream ends. Only choice 0 is translated (the Messages API has no
    ``n``). ``done`` turns true after an upstream error frame, which is
    relayed as an Anthropic ``error`` event and ends the stream.
    """

    def __init__(
        self,
        *,
        model: str,
        stop_sequences: list[str] | None,
        emit_thinking: bool,
    ) -> None:
        self._model = model
        self._stop_sequences = stop_sequences
        self._emit_thinking = emit_thinking
        self._agg = StreamAggregator(is_chat=True)
        self._buf = b""
        self._started = False
        self._next_index = 0
        self._open: str | None = None  # "text" | "thinking"
        self._matched_stop: Any = None
        self.done = False

    # -- input ---------------------------------------------------------------

    def feed_bytes(self, chunk: bytes) -> list[bytes]:
        self._buf += chunk.replace(b"\r\n", b"\n")
        out: list[bytes] = []
        while b"\n\n" in self._buf and not self.done:
            frame, self._buf = self._buf.split(b"\n\n", 1)
            out.extend(self._feed_frame(frame))
        return out

    def _feed_frame(self, frame: bytes) -> list[bytes]:
        ev = None
        for line in frame.split(b"\n"):
            ev = parse_sse_event(line)
            if ev is not None:
                break
        return self.feed_event(ev) if ev is not None else []

    def feed_event(self, ev: dict[str, Any]) -> list[bytes]:
        if self.done:
            return []
        # A mid-stream engine error: `{"error": {...}}` (current vLLM,
        # llama.cpp) or the older flat `{"object": "error", "message": ...}`.
        if (ev.get("error") is not None or ev.get("object") == "error") and not ev.get("choices"):
            self.done = True
            return [sse_frame("error", error_body(500, upstream_error_text(ev)))]
        self._agg.feed_event(ev)
        out = self._start(ev.get("usage"))
        for choice in ev.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            if choice.get("stop_reason") is not None:
                self._matched_stop = choice["stop_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            reasoning = delta_reasoning(delta)
            if reasoning and self._emit_thinking:
                out.extend(self._delta("thinking", reasoning))
            if delta.get("content"):
                out.extend(self._delta("text", delta["content"]))
        return out

    # -- output --------------------------------------------------------------

    def _start(self, usage: Any) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        start_usage = anthropic_usage(usage)
        start_usage["output_tokens"] = 0
        message = {
            "id": new_message_id(),
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        }
        return [sse_frame("message_start", {"type": "message_start", "message": message})]

    def _close_open(self) -> list[bytes]:
        if self._open is None:
            return []
        self._open = None
        index = self._next_index
        self._next_index += 1
        return [sse_frame("content_block_stop", {"type": "content_block_stop", "index": index})]

    def _delta(self, kind: str, text: str) -> list[bytes]:
        out: list[bytes] = []
        if self._open != kind:
            out.extend(self._close_open())
            self._open = kind
            block = (
                {"type": "text", "text": ""}
                if kind == "text"
                else {"type": "thinking", "thinking": "", "signature": ""}
            )
            out.append(
                sse_frame(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self._next_index,
                        "content_block": block,
                    },
                )
            )
        delta = (
            {"type": "text_delta", "text": text}
            if kind == "text"
            else {"type": "thinking_delta", "thinking": text}
        )
        out.append(
            sse_frame(
                "content_block_delta",
                {"type": "content_block_delta", "index": self._next_index, "delta": delta},
            )
        )
        return out

    def finish(self) -> list[bytes]:
        """Close the message: pending tool calls, stop reason, final usage."""
        if self.done:
            return []
        out: list[bytes] = []
        if self._buf.strip():
            out.extend(self._feed_frame(self._buf))
            self._buf = b""
            if self.done:
                return out
        out.extend(self._start(None))
        out.extend(self._close_open())
        built = self._agg.build()
        choices = built.get("choices") or [{}]
        choice = choices[0]
        tools = tool_use_blocks(choice.get("message") or {})
        for block in tools:
            index = self._next_index
            self._next_index += 1
            args = block["input"]
            out.append(
                sse_frame(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    },
                )
            )
            out.append(
                sse_frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)},
                    },
                )
            )
            out.append(
                sse_frame("content_block_stop", {"type": "content_block_stop", "index": index})
            )
        stop_reason, stop_sequence = map_stop_reason(
            choice.get("finish_reason"), self._matched_stop, self._stop_sequences, bool(tools)
        )
        self.done = True
        out.append(
            sse_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                    "usage": anthropic_usage(built.get("usage")),
                },
            )
        )
        out.append(sse_frame("message_stop", {"type": "message_stop"}))
        return out


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def error_body(status: int, message: str) -> dict[str, Any]:
    """Anthropic's error envelope for an HTTP status."""
    kind = _ERROR_TYPES.get(status) or ("api_error" if status >= 500 else "invalid_request_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


def upstream_error_text(obj: Any) -> str:
    """The human message inside an engine's (or FastAPI's) error body."""
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err:
            return err
        for key in ("message", "detail"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict | list):
                return json.dumps(value)
    if isinstance(obj, str) and obj:
        return obj
    return "upstream engine error"


def upstream_error_message(body: bytes) -> str:
    try:
        return upstream_error_text(json.loads(body))
    except (TypeError, ValueError):
        text = body.decode("utf-8", "replace").strip()
        return text[:2000] or "upstream engine error"
