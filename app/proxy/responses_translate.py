"""OpenAI Responses API <-> OpenAI chat completions translation.

``POST /v1/responses`` (routes_responses.py) turns a Responses request into the
chat body every engine behind the warden speaks, runs it through the ordinary
``_forward``, and turns the answer back -- the same shape as ``/v1/messages``
(messages_translate.py), and for the same reason: everything the warden adds
(admission, accounting, live phases, prefix-cache estimate, replica affinity,
the runaway detector) lives on the chat path. vLLM's own ``/v1/responses`` is
not used: for non-harmony models its stream emits tool-call XML as text deltas
(vllm-project/vllm#36435) so Codex would never see a function call.
Everything here is pure: no I/O, no app state.

Decisions that are not obvious from the spec:

* **Stateless.** Codex sends ``store: false`` and the whole conversation in
  ``input`` every turn. ``previous_response_id``, ``conversation`` and
  ``background`` are refused (400) rather than silently ignored.
* **Tool calls are emitted whole, at the end** (like the Messages stream):
  ``StreamAggregator`` merges the engine's fragments, so engines that
  interleave parallel calls and invalid-JSON arguments are handled in one place.
* **Reasoning goes on the summary channel.** Codex shows summaries by default
  and ignores raw reasoning text unless configured.
* **A stream always ends with ``response.completed``**, with ``status:
  "incomplete"`` for a truncated turn: Codex treats ``response.incomplete`` as
  a stream error and retries the whole turn. An in-stream engine error is
  ``response.failed``.
* **Invalid-JSON tool arguments are passed through raw** (the Messages
  translator sends ``{}``): the model's text is more useful to the client than
  a silently empty call.
* **No request content is logged.** Errors and log lines name item types and
  field paths only.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.proxy.messages_translate import (
    TranslationError,
    append_note,
    prefix_content,
    system_note,
    upstream_error_text,
)
from app.proxy.reaggregate import StreamAggregator, parse_sse_event
from app.proxy.responses_schemas import ResponsesRequest
from app.utils.sse import delta_reasoning

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}
_CUSTOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}
_CUSTOM_HINT = "Freeform tool: put the entire raw tool input in the string argument `input`."

#: Items that only exist on the Responses side of the wire.
_DROPPED_ITEMS = frozenset({"reasoning", "web_search_call"})


@dataclass
class Translation:
    """A translated request plus what the answer needs to know about it."""

    oai: dict[str, Any]
    #: names of ``custom`` (freeform) tools: their calls come back as
    #: ``custom_tool_call`` items, not ``function_call``
    custom_names: frozenset[str] = frozenset()
    #: tool name -> the ``namespace`` tool it was flattened out of
    namespaces: dict[str, str] = field(default_factory=dict)
    emit_thinking: bool = False
    tool_choice: Any = "auto"
    parallel_tool_calls: bool = True
    #: item types dropped on the way in (debug aid, never content)
    dropped: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Request: Responses -> OpenAI chat
# --------------------------------------------------------------------------


def thinking_requested(req: ResponsesRequest) -> bool:
    """ON iff ``reasoning`` is an object whose effort is not ``none``."""
    return req.reasoning is not None and req.reasoning.get("effort") != "none"


def _text_parts(content: Any, where: str) -> list[dict[str, Any]]:
    """A message's ``content`` as chat parts: text parts and image_url parts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise TranslationError(f"{where}: content must be a string or a list")
    parts: list[dict[str, Any]] = []
    for j, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        here = f"{where}[{j}]"
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "refusal":
            parts.append({"type": "text", "text": str(part.get("refusal", ""))})
        elif kind == "input_image":
            parts.append(_image_part(part, here))
        else:
            raise TranslationError(f"{here}: content type '{kind}' is not supported")
    return parts


def _image_part(part: dict[str, Any], where: str) -> dict[str, Any]:
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        raise TranslationError(
            f"{where}: input_image needs an 'image_url' (file ids are not supported)"
        )
    return {"type": "image_url", "image_url": {"url": url}}


def _collapse(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _plain_text(content: Any, where: str) -> str:
    return "\n\n".join(p["text"] for p in _text_parts(content, where) if p["type"] == "text")


def _tool_output(output: Any, call_id: str, where: str, user_parts: list[dict[str, Any]]) -> str:
    """The text of a ``*_call_output``. Images cannot ride in a ``tool``
    message, so they are queued for the user message that follows it."""
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    if not isinstance(output, list):
        raise TranslationError(f"{where}: output must be a string or a list")
    texts: list[str] = []
    for j, part in enumerate(output):
        parts = _text_parts([part], f"{where}[{j}]")  # raises on unknown types
        for p in parts:
            if p["type"] == "text":
                texts.append(p["text"])
            else:
                user_parts.append({"type": "text", "text": f"[image from tool output {call_id}]"})
                user_parts.append(p)
                texts.append("[image attached in the next user message]")
    return "\n\n".join(texts)


def _arguments(raw: Any) -> str:
    if isinstance(raw, str):
        return raw or "{}"
    if raw is None:
        return "{}"
    return json.dumps(raw)


def _tools(
    req: ResponsesRequest,
) -> tuple[list[dict[str, Any]], frozenset[str], dict[str, str], list[str]]:
    """``(chat tools, custom tool names, name -> namespace, dropped tool types)``.

    A ``namespace`` tool (Codex 0.160 sends ``multi_agent_v1``) is a named group
    of function tools: the chat API has no groups, so its members are
    flattened under their own names and remembered, so the call that comes
    back can carry the ``namespace`` Codex looks the tool up by.
    """
    out: list[dict[str, Any]] = []
    custom: set[str] = set()
    namespaces: dict[str, str] = {}
    seen: set[str] = set()
    dropped: list[str] = []

    def add(tool: Any, where: str, namespace: str | None) -> None:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if isinstance(kind, str) and kind.startswith("web_search"):
            logger.debug("responses: dropping hosted tool type %r", kind)
            dropped.append(kind)
            return
        if kind == "namespace" and namespace is None:
            ns = tool.get("name")
            members = tool.get("tools")
            if not isinstance(ns, str) or not ns or not isinstance(members, list):
                raise TranslationError(f"{where}: namespace tool needs a 'name' and 'tools'")
            for k, member in enumerate(members):
                add(member, f"{where}.tools[{k}]", ns)
            return
        if kind not in ("function", "custom"):
            raise TranslationError(f"{where}: tool type '{kind}' is not supported")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise TranslationError(f"{where}: tool without a 'name'")
        if name in seen:
            if namespace is not None:
                # A namespace member that collides must not kill the whole
                # thread: the first tool of that name wins, this one is skipped.
                logger.debug("responses: skipping namespace member with a duplicate name")
                dropped.append(f"duplicate:{namespace}")
                return
            raise TranslationError(f"{where}: duplicate tool name '{name}'")
        seen.add(name)
        if namespace is not None:
            namespaces[name] = namespace
        fn: dict[str, Any] = {"name": name}
        description = tool.get("description") or ""
        if kind == "function":
            fn["parameters"] = tool.get("parameters") or dict(_EMPTY_SCHEMA)
        else:
            custom.add(name)
            fmt = tool.get("format")
            extra = _CUSTOM_HINT
            if isinstance(fmt, dict) and fmt.get("type") == "grammar":
                extra += f"\nGrammar ({fmt.get('syntax', 'lark')}):\n{fmt.get('definition', '')}"
            description = f"{description}\n\n{extra}" if description else extra
            fn["parameters"] = dict(_CUSTOM_SCHEMA)
        if description:
            fn["description"] = description
        out.append({"type": "function", "function": fn})

    for i, tool in enumerate(req.tools or []):
        add(tool, f"tools[{i}]", None)
    return out, frozenset(custom), namespaces, dropped


def _tool_choice(choice: Any, custom: frozenset[str]) -> Any:
    if isinstance(choice, str):
        if choice in ("auto", "none", "required"):
            return choice
        raise TranslationError(f"tool_choice: '{choice}' is not supported")
    if isinstance(choice, dict):
        kind = choice.get("type")
        name = choice.get("name")
        if kind in ("function", "custom"):
            if not isinstance(name, str) or not name:
                raise TranslationError(f"tool_choice type '{kind}' needs a 'name'")
            if kind == "custom" and name not in custom:
                raise TranslationError(
                    f"tool_choice names custom tool '{name}' that is not in tools"
                )
            return {"type": "function", "function": {"name": name}}
        raise TranslationError(f"tool_choice type '{kind}' is not supported")
    raise TranslationError("tool_choice must be a string or an object")


def _response_format(text: Any) -> dict[str, Any] | None:
    fmt = text.get("format") if isinstance(text, dict) else None
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema: dict[str, Any] = {
            "name": fmt.get("name") or "response",
            "schema": fmt.get("schema") or {},
        }
        if fmt.get("strict") is not None:
            schema["strict"] = fmt["strict"]
        return {"type": "json_schema", "json_schema": schema}
    raise TranslationError(f"text.format type '{kind}' is not supported")


class _Builder:
    """Folds ``input`` items into chat messages (see to_openai_request)."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.system_parts: list[str] = []
        self.pending: list[str] = []
        self.images: list[dict[str, Any]] = []
        self.started = False
        self.call_ids: set[str] = set()
        self.dropped: list[str] = []

    def flush_images(self) -> None:
        if self.images:
            self.messages.append({"role": "user", "content": self.images})
            self.images = []

    def add(self, i: int, item: dict[str, Any]) -> None:
        where = f"input[{i}]"
        kind = item.get("type")
        if kind is None and "role" in item:
            kind = "message"
        if kind in ("function_call_output", "custom_tool_call_output"):
            self.output(item, where)
            return
        self.flush_images()
        if kind == "message":
            self.message(item, where)
        elif kind in ("function_call", "custom_tool_call"):
            self.call(item, where, custom=kind == "custom_tool_call")
        elif kind in _DROPPED_ITEMS:
            logger.debug("responses: dropping %s item", kind)
            self.dropped.append(str(kind))
        elif kind is None:
            raise TranslationError(f"{where}: item without a 'type'")
        else:
            raise TranslationError(
                f"{where}: unsupported item type '{kind}'; this warden cannot translate it, "
                "start a new Codex thread (or resend the conversation without that item)"
            )

    def message(self, item: dict[str, Any], where: str) -> None:
        role = item.get("role")
        content = item.get("content")
        if role in ("system", "developer"):
            text = _plain_text(content, f"{where}.content")
            if not text.strip():
                return
            if self.started:
                self.pending.append(system_note(text))
            else:
                self.system_parts.append(text)
            return
        self.started = True
        if role == "user":
            parts = _text_parts(content, f"{where}.content")
            body = _collapse(parts)
            if self.pending:
                body = prefix_content(body, "\n\n".join(self.pending))
                self.pending = []
            self.messages.append({"role": "user", "content": body})
        elif role == "assistant":
            self.messages.append(
                {"role": "assistant", "content": _plain_text(content, f"{where}.content")}
            )
        else:
            raise TranslationError(f"{where}: role '{role}' is not supported")

    def call(self, item: dict[str, Any], where: str, *, custom: bool) -> None:
        self.started = True
        call_id = item.get("call_id")
        name = item.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise TranslationError(f"{where}: call without a 'call_id'")
        if not isinstance(name, str) or not name:
            raise TranslationError(f"{where}: call without a 'name'")
        if custom:
            arguments = json.dumps({"input": item.get("input", "")})
        else:
            arguments = _arguments(item.get("arguments"))
        entry = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "assistant":
            last.setdefault("tool_calls", []).append(entry)
            if last["content"] == "":
                last["content"] = None
        else:
            self.messages.append({"role": "assistant", "content": None, "tool_calls": [entry]})
        self.call_ids.add(call_id)

    def output(self, item: dict[str, Any], where: str) -> None:
        self.started = True
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise TranslationError(f"{where}: output without a 'call_id'")
        if call_id not in self.call_ids:
            raise TranslationError(f"{where}: output for unknown call_id")
        text = _tool_output(item.get("output"), call_id, f"{where}.output", self.images)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})


def to_openai_request(req: ResponsesRequest) -> Translation:
    """Build the OpenAI chat-completions body for ``req``.

    Only fields the engine understands are written (an allow-list): vLLM logs
    every ignored field, and Codex sends many. ``developer``/``system`` items
    in a leading run join ``instructions`` as the system prompt; later ones are
    folded, delimited, into the next user turn (chat templates such as Qwen's
    accept a system message only first), or appended at the end.
    """
    if req.previous_response_id is not None:
        raise TranslationError(
            "previous_response_id is not supported: the warden stores no responses; "
            "send the full conversation in 'input' with store:false"
        )
    if req.conversation is not None:
        raise TranslationError("conversation is not supported: send the conversation in 'input'")
    if req.background:
        raise TranslationError("background responses are not supported")

    b = _Builder()
    if isinstance(req.input, str):
        b.started = True
        b.messages.append({"role": "user", "content": req.input})
    else:
        for i, item in enumerate(req.input):
            b.add(i, item)
        b.flush_images()
    if b.pending:
        append_note(b.messages, "\n\n".join(b.pending))
    system = "\n\n".join(p for p in [req.instructions or "", *b.system_parts] if p)
    messages = b.messages
    if system:
        messages.insert(0, {"role": "system", "content": system})
    if not any(m["role"] != "system" for m in messages):
        raise TranslationError("input: no conversation turns")

    body: dict[str, Any] = {"model": req.model, "messages": messages}
    tools, custom, namespaces, dropped = _tools(req)
    choice: Any = "auto"
    if tools:
        body["tools"] = tools
        if req.tool_choice is not None:
            choice = _tool_choice(req.tool_choice, custom)
            body["tool_choice"] = choice
        if req.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = req.parallel_tool_calls
    elif req.tool_choice is not None:
        choice = _tool_choice(req.tool_choice, custom)
    if req.max_output_tokens is not None:
        body["max_tokens"] = req.max_output_tokens
    for name in ("temperature", "top_p"):
        value = getattr(req, name)
        if value is not None:
            body[name] = value
    fmt = _response_format(req.text)
    if fmt is not None:
        body["response_format"] = fmt
    if req.stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
    emit = thinking_requested(req)
    # Only the OFF case is sent: a template that has never heard of the flag
    # must keep its own default (same rule as /v1/messages).
    if not emit:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return Translation(
        oai=body,
        custom_names=custom,
        namespaces=namespaces,
        emit_thinking=emit,
        tool_choice=req.tool_choice if req.tool_choice is not None else "auto",
        parallel_tool_calls=True if req.parallel_tool_calls is None else req.parallel_tool_calls,
        dropped=[*b.dropped, *dropped],
    )


# --------------------------------------------------------------------------
# Response: OpenAI chat -> Responses
# --------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:24]}"


def responses_usage(usage: Any) -> dict[str, Any]:
    """Chat usage -> Responses usage. ``input_tokens`` INCLUDES the cached
    prefix (unlike Anthropic's), which is reported in ``cached_tokens``."""
    usage = usage if isinstance(usage, dict) else {}
    prompt = usage.get("prompt_tokens") or 0
    out = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": min(cached, prompt)},
        "output_tokens": out,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": usage.get("total_tokens") or prompt + out,
    }


def finish_status(finish_reason: Any, produced: bool = False) -> tuple[str, dict[str, str] | None]:
    """chat ``finish_reason`` -> ``(status, incomplete_details)``.

    ``produced``: the turn yielded output. A stream that ends with output but
    no finish_reason was cut (the wall-clock reaper), so it is incomplete.
    """
    if finish_reason is None and produced:
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "length" or (
        isinstance(finish_reason, str) and finish_reason.startswith("runaway_")
    ):
        # A runaway cut (runaway.py) is a truncated output, which is what
        # max_output_tokens tells a client.
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None


def reasoning_item(text: str, item_id: str | None = None) -> dict[str, Any]:
    return {
        "id": item_id or new_id("rs"),
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
        "encrypted_content": None,
    }


def message_item(text: str, item_id: str | None = None) -> dict[str, Any]:
    return {
        "id": item_id or new_id("msg"),
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _arguments_text(raw: Any) -> str:
    if isinstance(raw, str):
        if raw:
            try:
                json.loads(raw)
            except ValueError:
                logger.warning("responses: tool call arguments are not JSON; passing them raw")
        return raw or "{}"
    if isinstance(raw, dict):
        return json.dumps(raw)
    return "{}"


def call_item(call: dict[str, Any], tr: Translation) -> dict[str, Any]:
    """One chat ``tool_calls`` entry -> a ``function_call`` / ``custom_tool_call`` item."""
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    call_id = call.get("id") or new_id("call")
    args = _arguments_text(fn.get("arguments"))
    ns = tr.namespaces.get(name)
    if name in tr.custom_names:
        raw: Any = args
        try:
            parsed = json.loads(args)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
            raw = parsed["input"]
        item: dict[str, Any] = {
            "id": new_id("ctc"),
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": name,
            "input": raw,
        }
    else:
        item = {
            "id": new_id("fc"),
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": args,
            "status": "completed",
        }
    if ns is not None:
        item["namespace"] = ns
    return item


def output_items_from_message(message: dict[str, Any], tr: Translation) -> list[dict[str, Any]]:
    """Reasoning, then the message, then the calls (the order the stream uses)."""
    items: list[dict[str, Any]] = []
    reasoning = delta_reasoning(message)
    if tr.emit_thinking and reasoning:
        items.append(reasoning_item(reasoning))
    if message.get("content"):
        items.append(message_item(message["content"]))
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            items.append(call_item(call, tr))
    return items


def response_object(
    *,
    rid: str,
    created_at: int,
    status: str,
    model: str,
    tr: Translation,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    incomplete_details: dict[str, str] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": rid,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage,
        "incomplete_details": incomplete_details,
        "error": error,
        "parallel_tool_calls": tr.parallel_tool_calls,
        "tool_choice": tr.tool_choice,
        "store": False,
        "instructions": None,
        "metadata": {},
    }


def to_response_object(oai: dict[str, Any], *, model: str, tr: Translation) -> dict[str, Any]:
    """A non-streamed chat completion -> a ``response`` object."""
    choices = oai.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    output = output_items_from_message(message, tr)
    status, details = finish_status(choice.get("finish_reason"), bool(output))
    return response_object(
        rid=new_id("resp"),
        created_at=int(time.time()),
        status=status,
        model=model,
        tr=tr,
        output=output,
        usage=responses_usage(oai.get("usage")),
        incomplete_details=details,
    )


def sse_frame(seq: int, event: str, data: dict[str, Any]) -> bytes:
    payload = {"type": event, "sequence_number": seq, **data}
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


class ResponsesStreamTranslator:
    """Chat ``chat.completion.chunk`` SSE bytes -> Responses SSE events.

    Feed upstream bytes with :meth:`feed_bytes`; call :meth:`finish` once the
    upstream ends. Only choice 0 is translated. Text and reasoning stream as
    they arrive; tool calls come out whole after the engine is done (see the
    module docstring). ``done`` turns true after the terminal event:
    ``response.completed`` (also for a truncated turn) or ``response.failed``
    for an engine error frame. ``response.incomplete`` is never sent.
    """

    def __init__(self, *, model: str, tr: Translation) -> None:
        self._model = model
        self._tr = tr
        self._agg = StreamAggregator(is_chat=True)
        self._buf = b""
        self._rid = new_id("resp")
        self._created = int(time.time())
        self._seq = 0
        self._started = False
        self._next_index = 0
        self._outputs: list[dict[str, Any]] = []
        # the open streamed item: {"kind", "id", "index", "text"}
        self._open: dict[str, Any] | None = None
        self.done = False

    # -- input ---------------------------------------------------------------

    def feed_bytes(self, chunk: bytes) -> list[bytes]:
        # normalise after joining: a CRLF may be split across two chunks
        self._buf = (self._buf + chunk).replace(b"\r\n", b"\n")
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
        if (ev.get("error") is not None or ev.get("object") == "error") and not ev.get("choices"):
            return self._fail(upstream_error_text(ev))
        self._agg.feed_event(ev)
        out = self._start()
        for choice in ev.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            reasoning = delta_reasoning(delta)
            if reasoning and self._tr.emit_thinking:
                out.extend(self._delta("reasoning", reasoning))
            if delta.get("content"):
                out.extend(self._delta("message", delta["content"]))
        return out

    # -- output --------------------------------------------------------------

    def _emit(self, event: str, **data: Any) -> bytes:
        frame = sse_frame(self._seq, event, data)
        self._seq += 1
        return frame

    def _snapshot(self, status: str, **kw: Any) -> dict[str, Any]:
        return response_object(
            rid=self._rid,
            created_at=self._created,
            status=status,
            model=self._model,
            tr=self._tr,
            output=kw.pop("output", []),
            **kw,
        )

    def _start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        snap = self._snapshot("in_progress")
        return [
            self._emit("response.created", response=snap),
            self._emit("response.in_progress", response=snap),
        ]

    def _delta(self, kind: str, text: str) -> list[bytes]:
        out: list[bytes] = []
        if self._open is None or self._open["kind"] != kind:
            out.extend(self._close_open())
            out.extend(self._open_item(kind))
        item = self._open
        assert item is not None
        item["text"] += text
        if kind == "reasoning":
            out.append(
                self._emit(
                    "response.reasoning_summary_text.delta",
                    item_id=item["id"],
                    output_index=item["index"],
                    summary_index=0,
                    delta=text,
                )
            )
        else:
            out.append(
                self._emit(
                    "response.output_text.delta",
                    item_id=item["id"],
                    output_index=item["index"],
                    content_index=0,
                    delta=text,
                    logprobs=[],
                )
            )
        return out

    def _open_item(self, kind: str) -> list[bytes]:
        index = self._next_index
        self._next_index += 1
        item_id = new_id("rs" if kind == "reasoning" else "msg")
        self._open = {"kind": kind, "id": item_id, "index": index, "text": ""}
        if kind == "reasoning":
            shell = {"id": item_id, "type": "reasoning", "summary": [], "encrypted_content": None}
            part: dict[str, Any] = {"type": "summary_text", "text": ""}
            return [
                self._emit("response.output_item.added", output_index=index, item=shell),
                self._emit(
                    "response.reasoning_summary_part.added",
                    item_id=item_id,
                    output_index=index,
                    summary_index=0,
                    part=part,
                ),
            ]
        shell = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        part = {"type": "output_text", "text": "", "annotations": []}
        return [
            self._emit("response.output_item.added", output_index=index, item=shell),
            self._emit(
                "response.content_part.added",
                item_id=item_id,
                output_index=index,
                content_index=0,
                part=part,
            ),
        ]

    def _close_open(self) -> list[bytes]:
        item = self._open
        if item is None:
            return []
        self._open = None
        text, item_id, index = item["text"], item["id"], item["index"]
        if item["kind"] == "reasoning":
            done = reasoning_item(text, item_id)
            out = [
                self._emit(
                    "response.reasoning_summary_text.done",
                    item_id=item_id,
                    output_index=index,
                    summary_index=0,
                    text=text,
                ),
                self._emit(
                    "response.reasoning_summary_part.done",
                    item_id=item_id,
                    output_index=index,
                    summary_index=0,
                    part={"type": "summary_text", "text": text},
                ),
            ]
        else:
            done = message_item(text, item_id)
            out = [
                self._emit(
                    "response.output_text.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    text=text,
                    logprobs=[],
                ),
                self._emit(
                    "response.content_part.done",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part=done["content"][0],
                ),
            ]
        out.append(self._emit("response.output_item.done", output_index=index, item=done))
        self._outputs.append(done)
        return out

    def _call(self, call: dict[str, Any]) -> list[bytes]:
        index = self._next_index
        self._next_index += 1
        item = call_item(call, self._tr)
        if item["type"] == "custom_tool_call":
            shell = {**item, "input": ""}
            events = [
                ("response.custom_tool_call_input.delta", {"delta": item["input"]}),
                ("response.custom_tool_call_input.done", {"input": item["input"]}),
            ]
        else:
            shell = {**item, "arguments": "", "status": "in_progress"}
            events = [
                ("response.function_call_arguments.delta", {"delta": item["arguments"]}),
                ("response.function_call_arguments.done", {"arguments": item["arguments"]}),
            ]
        out = [self._emit("response.output_item.added", output_index=index, item=shell)]
        for name, data in events:
            out.append(self._emit(name, item_id=item["id"], output_index=index, **data))
        out.append(self._emit("response.output_item.done", output_index=index, item=item))
        self._outputs.append(item)
        return out

    def _fail(self, message: str) -> list[bytes]:
        out = self._start()
        self.done = True
        error = {"code": "server_error", "message": message}
        out.append(self._emit("response.failed", response=self._snapshot("failed", error=error)))
        return out

    def fail(self, message: str) -> list[bytes]:
        """End the stream with ``response.failed`` (no-op once it has ended)."""
        return [] if self.done else self._fail(message)

    def finish(self) -> list[bytes]:
        """Close the turn: pending calls, then ``response.completed``."""
        if self.done:
            return []
        out: list[bytes] = []
        if self._buf.strip():
            out.extend(self._feed_frame(self._buf))
            self._buf = b""
            if self.done:
                return out
        out.extend(self._start())
        out.extend(self._close_open())
        built = self._agg.build()
        choices = built.get("choices") or [{}]
        choice = choices[0]
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                out.extend(self._call(call))
        status, details = finish_status(choice.get("finish_reason"), bool(self._outputs))
        self.done = True
        out.append(
            self._emit(
                "response.completed",
                response=self._snapshot(
                    status,
                    output=list(self._outputs),
                    usage=responses_usage(built.get("usage")),
                    incomplete_details=details,
                ),
            )
        )
        return out


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "invalid_request_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


def error_body(status: int, message: str) -> dict[str, Any]:
    """OpenAI's error envelope for an HTTP status."""
    kind = _ERROR_TYPES.get(status) or (
        "server_error" if status >= 500 else "invalid_request_error"
    )
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}
