"""Rebuild vLLM's non-stream JSON from a forced-streaming SSE chunk sequence.

The runaway detector needs to watch tokens as they decode, so the proxy forces
``stream=true`` upstream even when the client asked for ``stream=false`` (see
docs/superpowers/specs/2026-07-21-runaway-detector-design.md §4). For those
clients we must hand back exactly what vLLM's *non-stream* endpoint would have
produced — reconstructed from the ``chat.completion.chunk`` / ``text_completion``
deltas plus the ``include_usage`` tail chunk.

Fidelity is the risk. :class:`StreamAggregator` accumulates state per choice
index and emits the canonical non-stream envelope:
``id``/``object``/``created``/``model``/``choices``/``usage``. It is defensive by
construction — an unrecognized chunk shape is skipped, never fatal — because the
spec's hard rule is *never 500 a request that generated fine*.
"""

import json

from app.utils.sse import delta_reasoning, delta_reasoning_field


def parse_sse_event(line: bytes | str) -> dict | None:
    """Parse one SSE ``data:`` frame into its JSON object.

    Returns ``None`` for the ``[DONE]`` sentinel, any non-``data:`` line
    (comments, blanks, event: fields), or a payload that does not parse — the
    caller treats ``None`` as "nothing to aggregate from this line".
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8", "replace")
        except Exception:
            return None
    line = line.lstrip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        ev = json.loads(payload)
    except Exception:
        return None
    return ev if isinstance(ev, dict) else None


class _ChoiceState:
    __slots__ = (
        "index",
        "role",
        "content",
        "reasoning_content",
        "reasoning_field",
        "text",
        "finish_reason",
        "tool_calls",
    )

    def __init__(self, index: int):
        self.index = index
        self.role: str | None = None
        self.content: str | None = None
        self.reasoning_content: str | None = None
        # The field name the engine used for the reasoning channel
        # (`reasoning_content`, or `reasoning` on newer builds — R22): the
        # rebuilt message must carry it under the same name, so the body
        # matches what the engine's own non-stream endpoint would have
        # produced. Set on the first reasoning delta, kept afterwards.
        self.reasoning_field: str | None = None
        self.text: str | None = None
        self.finish_reason = None
        # tool_calls reassembled by their own nested index.
        self.tool_calls: dict[int, dict] = {}


class StreamAggregator:
    """Fold a stream of parsed SSE events into one non-stream response dict."""

    def __init__(self, is_chat: bool):
        self.is_chat = is_chat
        self._choices: dict[int, _ChoiceState] = {}
        self._usage = None
        self._id: str | None = None
        self._model: str | None = None
        self._created: int | None = None
        self._system_fingerprint = None

    def feed_event(self, ev: dict) -> None:
        """Absorb one parsed chunk. Malformed shapes are skipped, never raised."""
        if not isinstance(ev, dict):
            return
        # Top-level envelope fields are constant across chunks; take the first
        # non-null we see for each.
        if self._id is None and isinstance(ev.get("id"), str):
            self._id = ev["id"]
        if self._model is None and isinstance(ev.get("model"), str):
            self._model = ev["model"]
        if self._created is None and isinstance(ev.get("created"), int):
            self._created = ev["created"]
        if self._system_fingerprint is None and ev.get("system_fingerprint") is not None:
            self._system_fingerprint = ev["system_fingerprint"]
        # ``include_usage`` puts usage on a tail chunk (empty choices); earlier
        # chunks carry usage: null. Keep the last non-null one.
        if ev.get("usage") is not None:
            self._usage = ev["usage"]

        choices = ev.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._feed_choice(choice)

    def _feed_choice(self, choice: dict) -> None:
        idx = choice.get("index", 0)
        if not isinstance(idx, int):
            idx = 0
        st = self._choices.get(idx)
        if st is None:
            st = _ChoiceState(idx)
            self._choices[idx] = st

        fr = choice.get("finish_reason")
        if fr is not None:
            st.finish_reason = fr

        if self.is_chat:
            delta = choice.get("delta")
            if isinstance(delta, dict):
                role = delta.get("role")
                if role is not None:
                    st.role = role
                content = delta.get("content")
                if content is not None:
                    st.content = (st.content or "") + content
                reasoning = delta_reasoning(delta)
                if reasoning is not None:
                    if st.reasoning_field is None:
                        st.reasoning_field = delta_reasoning_field(delta)
                    st.reasoning_content = (st.reasoning_content or "") + reasoning
                self._merge_tool_calls(st, delta.get("tool_calls"))
        else:
            text = choice.get("text")
            if text is not None:
                st.text = (st.text or "") + text

    @staticmethod
    def _merge_tool_calls(st: _ChoiceState, tool_calls) -> None:
        if not isinstance(tool_calls, list):
            return
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tidx = tc.get("index", 0)
            if not isinstance(tidx, int):
                tidx = 0
            acc = st.tool_calls.get(tidx)
            if acc is None:
                acc = {"id": None, "type": None, "function": {"name": None, "arguments": ""}}
                st.tool_calls[tidx] = acc
            if tc.get("id") is not None:
                acc["id"] = tc["id"]
            if tc.get("type") is not None:
                acc["type"] = tc["type"]
            fn = tc.get("function")
            if isinstance(fn, dict):
                if fn.get("name") is not None:
                    acc["function"]["name"] = fn["name"]
                args = fn.get("arguments")
                if args is not None:
                    acc["function"]["arguments"] = acc["function"]["arguments"] + args

    def build(self, finish_reason_override: str | None = None) -> dict:
        """Emit the reconstructed non-stream response.

        ``finish_reason_override`` (set on a runaway trip) replaces every
        choice's finish_reason so the client sees the runaway signal on the
        partial content it did receive.
        """
        out_choices = []
        for idx in sorted(self._choices):
            st = self._choices[idx]
            finish = finish_reason_override or st.finish_reason
            if self.is_chat:
                message: dict = {"role": st.role or "assistant"}
                # content is always present on a chat.completion message (null
                # only when tool_calls fully replace it); default to "".
                message["content"] = (
                    st.content if st.content is not None else (None if st.tool_calls else "")
                )
                if st.reasoning_content is not None:
                    # Emit under the name the engine used (first delta wins).
                    message[st.reasoning_field or "reasoning_content"] = st.reasoning_content
                if st.tool_calls:
                    message["tool_calls"] = [st.tool_calls[t] for t in sorted(st.tool_calls)]
                out_choices.append(
                    {
                        "index": idx,
                        "message": message,
                        "logprobs": None,
                        "finish_reason": finish,
                    }
                )
            else:
                out_choices.append(
                    {
                        "index": idx,
                        "text": st.text if st.text is not None else "",
                        "logprobs": None,
                        "finish_reason": finish,
                    }
                )

        out: dict = {
            "id": self._id,
            "object": "chat.completion" if self.is_chat else "text_completion",
            "created": self._created,
            "model": self._model,
            "choices": out_choices,
        }
        if self._usage is not None:
            out["usage"] = self._usage
        if self._system_fingerprint is not None:
            out["system_fingerprint"] = self._system_fingerprint
        return out
