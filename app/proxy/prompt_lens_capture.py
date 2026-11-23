"""Prompt Lens structured prompt capture (G1, spec 2026-11-22 prompt_lens dock).

Display only: everything here reads a chat body and builds a bounded,
windowed copy for the Prompt Lens tap. It NEVER touches the forwarded request,
never logs content, and never raises (any error means no capture).

The head/body split is the canonical form's (``app.cobs2.canonical``): every
``system`` / ``developer`` message, wherever it sits, is the head; the other
dict messages are the body, numbered 0.. in order, so a body message's ``i``
equals its kvsig node index. ``split_head_body`` is the one copy of that
split; later Prompt Lens passes reuse it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

from app.cobs2.canonical import _HEAD_ROLES

logger = logging.getLogger(__name__)

#: Most collapsed (text-less) body messages listed in one capture, as compact
#: parallel arrays (``older``). Older collapsed messages are counted in
#: ``omitted`` instead, so a very long history cannot pin a big start event in
#: the ring or the starts map (about 10 KB of heap at this cap).
COLLAPSED_LIST_MAX = 256
#: Most chars one listed message carries: text, reasoning and tool-call
#: arguments together. Tool calls past it are counted in ``calls_omitted``.
ENTRY_CHARS = 24_000

#: One-char role codes of the collapsed list; any other role is "?".
_ROLE_CODES = {"user": "u", "assistant": "a", "tool": "t", "function": "f"}

_TEXT_TYPES = ("text", "input_text", "output_text")
_IMAGE_TYPES = ("image_url", "input_image", "image")


def _prompt_lens_prompt_window(text: str, max_chars: int, tail_chars: int) -> tuple[str, bool]:
    """Display-only capture window for the Prompt Lens tap.

    NEVER touches the forwarded request — only the tapped copy. If the prompt
    fits within ``max_chars`` it is returned verbatim. Otherwise keep the head
    ``(max_chars - tail_chars)`` chars PLUS the last ``tail_chars`` chars,
    joined by the marker ``\\n\\n…[{n} chars elided]…\\n\\n`` where ``n`` is the
    elided count. This guarantees the newest turn (which lives at the tail)
    survives even under a giant repeated system prompt, instead of the old
    head-slice spending the whole budget on the system prompt.

    Returns ``(windowed_text, elided)`` where ``elided`` is True iff the middle
    was dropped. O(len) — pure slices, no extra copies.
    """
    n = len(text)
    if n <= max_chars:
        return text, False
    # Clamp the tail so a degenerate config (tail >= max) can't make the head
    # length negative; the head is whatever budget remains after the tail.
    tail = max(0, min(tail_chars, max_chars))
    head_len = max_chars - tail
    head = text[:head_len]
    tail_str = text[n - tail :] if tail > 0 else ""
    elided = n - head_len - tail
    return f"{head}\n\n…[{elided} chars elided]…\n\n{tail_str}", True


def _role(m: dict[str, Any]) -> str:
    r = m.get("role")
    return r if isinstance(r, str) else ""


def split_head_body(
    messages: list[Any],
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    """``(head, body)`` of a chat ``messages`` list, each as ``(raw index,
    message)`` pairs. Non-dict entries are skipped (the canonical form skips
    them too), so ``body``'s position is the kvsig node index and the raw index
    points back into the client's list."""
    head: list[tuple[int, dict[str, Any]]] = []
    body: list[tuple[int, dict[str, Any]]] = []
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        (head if _role(m) in _HEAD_ROLES else body).append((idx, m))
    return head, body


def _content(content: Any) -> tuple[str, int]:
    """``(text, images)`` of a chat ``content``: text parts joined by a newline,
    image parts counted (never copied)."""
    if isinstance(content, str):
        return content, 0
    if not isinstance(content, list):
        return "", 0
    texts: list[str] = []
    images = 0
    for p in content:
        if isinstance(p, str):
            texts.append(p)
        elif not isinstance(p, dict):
            continue
        elif p.get("type") in _IMAGE_TYPES:
            images += 1
        elif (p.get("type") in _TEXT_TYPES or "type" not in p) and isinstance(p.get("text"), str):
            texts.append(p["text"])
    return "\n".join(texts), images


def _content_chars(content: Any) -> int:
    """``len(_content(content)[0])`` without building the joined text."""
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    parts = 0
    for p in content:
        if isinstance(p, str):
            n = len(p)
        elif (
            isinstance(p, dict)
            and p.get("type") not in _IMAGE_TYPES
            and (p.get("type") in _TEXT_TYPES or "type" not in p)
            and isinstance(p.get("text"), str)
        ):
            n = len(p["text"])
        else:
            continue
        total += n
        parts += 1
    return total + max(0, parts - 1)


def _dump(v: Any) -> str:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=str)


def _tool_name(t: Any) -> str | None:
    if not isinstance(t, dict):
        return None
    fn = t.get("function")
    if isinstance(fn, dict):
        fname = fn.get("name")
        if isinstance(fname, str):
            return fname
    name = t.get("name")
    return name if isinstance(name, str) else None


def _reasoning(m: dict[str, Any]) -> str:
    """An assistant message's reasoning, as the canonical form takes it
    (``reasoning_content``, else ``reasoning``)."""
    if _role(m) != "assistant":
        return ""
    for key in ("reasoning_content", "reasoning"):
        r = m.get(key)
        if isinstance(r, str) and r:
            return r
    return ""


def _calls(m: dict[str, Any]) -> list[tuple[str | None, str, str]]:
    """``(id, name, raw args)`` of an assistant message's tool calls."""
    calls = m.get("tool_calls")
    if _role(m) != "assistant" or not isinstance(calls, list):
        return []
    out: list[tuple[str | None, str, str]] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function")
        fn = fn if isinstance(fn, dict) else {}
        cid = c.get("id")
        name = fn.get("name")
        args = fn.get("arguments")
        if args is None:
            args = ""
        elif not isinstance(args, str):
            args = _dump(args)
        out.append(
            (cid if isinstance(cid, str) else None, name if isinstance(name, str) else "", args)
        )
    return out


def _msg_size(m: dict[str, Any], calls: list[tuple[str | None, str, str]] | None = None) -> int:
    """A body message's whole size in chars: text, reasoning and tool-call
    arguments (the parts of its kvsig node)."""
    if calls is None:
        calls = _calls(m)
    return _content_chars(m.get("content")) + len(_reasoning(m)) + sum(len(a) for _, _, a in calls)


def _tool_list(body: dict[str, Any]) -> list[Any]:
    raw = body.get("tools")
    return raw if isinstance(raw, list) else []


def _tools_chars(tool_list: list[Any]) -> int:
    return sum(len(_dump(t)) for t in tool_list)


class BodyLayout(NamedTuple):
    """Sizes and roles of a chat body, by kvsig part (``body_layout``)."""

    #: Body node ``i``'s index in the raw ``messages`` list.
    raw_body_index: list[int]
    #: Body node ``i``'s size in chars (``_msg_size``).
    parts_chars: list[int]
    #: Body node ``i``'s role (client text: never echo it unchecked).
    roles: list[str]
    #: The head's size: its texts joined by a blank line, as ``sys`` captures it.
    sys_chars: int
    #: The tool list's size: each tool's compact JSON.
    tools_chars: int
    #: The head messages' raw indices and roles.
    raw_head_index: list[int]
    head_roles: list[str]


def body_layout(body: Any) -> BodyLayout | None:
    """Per-part sizes and roles of a chat body, with ``split_head_body`` (the
    one head/body split) and the sizes ``structured_prompt`` reports; no text
    is copied. None when ``body`` is not a chat body. Raises on a bug (the
    caller fails open)."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None
    head, msgs = split_head_body(body["messages"])
    sys_chars = sum(_content_chars(m.get("content")) for _, m in head)
    if head:
        sys_chars += 2 * (len(head) - 1)
    return BodyLayout(
        raw_body_index=[r for r, _ in msgs],
        parts_chars=[_msg_size(m) for _, m in msgs],
        roles=[_role(m) for _, m in msgs],
        sys_chars=sys_chars,
        tools_chars=_tools_chars(_tool_list(body)),
        raw_head_index=[r for r, _ in head],
        head_roles=[_role(m) for _, m in head],
    )


def structured_prompt(
    body: dict[str, Any],
    *,
    budget_chars: int,
    sys_chars: int = 2000,
    msg_chars: int = 6000,
) -> dict[str, Any] | None:
    """The structured, windowed capture of a chat body (``PromptParts``), or
    None when the body is not a chat body or cannot be read (fail open).

    ``sys`` is the head joined by a blank line, windowed to ``sys_chars``. A
    body message's text, its reasoning (assistant only) and each tool call's
    arguments are windowed to ``msg_chars``; one message carries at most
    ``ENTRY_CHARS`` of them, and the tool calls past that are counted in
    ``calls_omitted``. ``chars`` is the message's whole size: text, reasoning
    and arguments, the parts of its kvsig node.

    Walking newest to oldest, messages are listed in ``messages`` while the
    running total stays within ``budget_chars`` (the newest always is). The
    older ones are collapsed: only role and size, in ``older`` as parallel
    arrays (``roles`` one char per message, ``chars`` ints) starting at body
    index ``older["from"]``, at most ``COLLAPSED_LIST_MAX`` of them; the older
    rest are counted in ``omitted``. ``collapsed`` counts both."""
    try:
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return None
        head, msgs = split_head_body(body["messages"])

        sys_part: dict[str, Any] | None = None
        if head:
            sys_text = "\n\n".join(_content(m.get("content"))[0] for _, m in head)
            win, el = _prompt_lens_prompt_window(sys_text, sys_chars, sys_chars // 4)
            sys_part = {"chars": len(sys_text), "text": win, "elided": el}

        tool_list = _tool_list(body)
        names = [n for n in (_tool_name(t) for t in tool_list) if n is not None]
        tools_chars = _tools_chars(tool_list)
        tools: dict[str, Any] = {"count": len(tool_list), "chars": tools_chars, "names": names}

        # Oldest to newest: sizes, plus the tool-call names a later tool result
        # resolves its call id against (one pass, no text copies).
        call_names: dict[str, str] = {}
        result_names: dict[int, str] = {}
        sizes: list[int] = []
        for n, (_, m) in enumerate(msgs):
            role = _role(m)
            m_calls = _calls(m)
            for cid, name, _args in m_calls:
                if cid is not None:
                    call_names[cid] = name
            size = _msg_size(m, m_calls)
            if role == "tool":
                cid = m.get("tool_call_id")
                if isinstance(cid, str) and cid in call_names:
                    result_names[n] = call_names[cid]
            sizes.append(size)

        tail = msg_chars // 4
        entries: list[dict[str, Any]] = []
        used = 0
        n = len(msgs) - 1
        while n >= 0:
            _, m = msgs[n]
            role = _role(m)
            text, images = _content(m.get("content"))
            win, el = _prompt_lens_prompt_window(text, msg_chars, tail)
            rwin, rel = _prompt_lens_prompt_window(_reasoning(m), msg_chars, tail)
            cost = len(win) + len(rwin)
            listed: list[dict[str, Any]] = []
            all_calls = _calls(m)
            for cid, name, args in all_calls:
                awin, ael = _prompt_lens_prompt_window(args, msg_chars, tail)
                if cost + len(awin) > ENTRY_CHARS:
                    break
                cost += len(awin)
                el = el or ael
                listed.append({"id": cid, "name": name, "args": awin})
            if entries and used + cost > budget_chars:
                break
            used += cost
            e: dict[str, Any] = {
                "i": n,
                "role": role,
                "chars": sizes[n],
                "text": win,
                "elided": el or rel or len(listed) < len(all_calls),
            }
            if rwin:
                e["reasoning"] = rwin
            if all_calls:
                e["tool_calls"] = listed
                if len(listed) < len(all_calls):
                    e["calls_omitted"] = len(all_calls) - len(listed)
            if role == "tool":
                cid = m.get("tool_call_id")
                if isinstance(cid, str):
                    e["tool_call_id"] = cid
                if n in result_names:
                    e["tool_name"] = result_names[n]
            if images:
                e["images"] = images
            entries.append(e)
            n -= 1
        entries.reverse()

        # Body indices 0..n are collapsed; the newest COLLAPSED_LIST_MAX of them
        # are listed compactly, the rest only counted.
        collapsed = n + 1
        first = max(0, collapsed - COLLAPSED_LIST_MAX)
        older = {
            "from": first,
            "roles": "".join(_ROLE_CODES.get(_role(m), "?") for _, m in msgs[first:collapsed]),
            "chars": sizes[first:collapsed],
        }

        return {
            "sys": sys_part,
            "tools": tools,
            "messages": entries,
            "older": older,
            "collapsed": collapsed,
            "omitted": first,
            "total_chars": (sys_part["chars"] if sys_part else 0) + tools_chars + sum(sizes),
        }
    except Exception as exc:  # noqa: BLE001 -- display capture must never fail a request
        # The type only: an exception's text could quote captured content.
        logger.warning("prompt_lens structured prompt failed: %s", type(exc).__name__)
        return None


# --------------------------------------------------------------------------
# G2: tool-call deltas (output side)
# --------------------------------------------------------------------------
class ToolCallDelta(NamedTuple):
    """One tool-call fragment of a reply. ``index`` is the call's slot in the
    reply (parallel calls interleave by it); ``call_id`` and ``name`` are set
    only on the fragment that carries them (the first, for vLLM); ``text`` is
    the arguments fragment, possibly ``""``. (The field ``index`` shadows
    ``tuple.index`` by design: it is the wire name.)"""

    index: int  # type: ignore[assignment]
    call_id: str | None
    name: str | None
    text: str


class GmFrame(NamedTuple):
    """Prompt Lens parse of one SSE frame (choice 0). Any part may be empty."""

    content: str | None
    reasoning: str | None
    finish_reason: str | None
    tool_calls: tuple[ToolCallDelta, ...]


def tool_call_delta(entry: Any, position: int) -> ToolCallDelta | None:
    """A ``tool_calls[i]`` entry (a stream delta or a whole ``message``
    call) as a ``ToolCallDelta``; None for a non-dict entry. ``index`` is the
    entry's own int ``index``, else ``position`` (its place in the list).
    Arguments that are not a string are dumped to compact JSON, as the prompt
    capture does."""
    if not isinstance(entry, dict):
        return None
    index = entry.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        index = position
    cid = entry.get("id")
    fn = entry.get("function")
    fn = fn if isinstance(fn, dict) else {}
    name = fn.get("name")
    args = fn.get("arguments")
    if args is None:
        args = ""
    elif not isinstance(args, str):
        args = _dump(args)
    return ToolCallDelta(
        index,
        cid if isinstance(cid, str) else None,
        name if isinstance(name, str) else None,
        args,
    )


class ToolCallBudget:
    """Per-request bound on the tool-call text Prompt Lens publishes.

    ``chars`` counts every published ``call_id``, ``name`` and argument
    fragment (the request's tool-call section, the same budget as the
    whole-section prompt capture). The capture ends at ONE point and the tail
    after it is dropped and counted (``dropped_chars`` of arguments,
    ``dropped_calls`` for calls whose opening fragment was never published):

    * an argument fragment that crosses the budget is cut to what fits and
      marked ``elided``;
    * a call's opening fragment (``call_id`` or ``name``) is never cut: one
      that does not fit is dropped whole and ends the capture, even with
      budget left, so no later fragment (of that call: a nameless orphan; or
      of any other call) is published;
    * an opener that uses the budget exactly is marked ``elided``, since none
      of its arguments can follow.

    O(1) memory whatever the reply holds.
    """

    __slots__ = ("left", "dropped_chars", "dropped_calls")

    def __init__(self, chars: int) -> None:
        self.left = max(0, chars)
        self.dropped_chars = 0
        self.dropped_calls = 0

    def take(self, d: ToolCallDelta) -> dict[str, Any] | None:
        """The event fields to publish for ``d`` (``index``, ``call_id``?,
        ``name``?, ``text``, ``elided``?), or None when nothing is published."""
        head = len(d.call_id or "") + len(d.name or "")
        opener = d.call_id is not None or d.name is not None
        if not opener and not d.text:
            return None
        if head > self.left or (not opener and self.left == 0):
            self.dropped_chars += len(d.text)
            if opener:
                self.dropped_calls += 1
            self.left = 0  # one clean tail: nothing after this is published
            return None
        self.left -= head
        text = d.text
        out: dict[str, Any] = {"index": d.index}
        if d.call_id is not None:
            out["call_id"] = d.call_id
        if d.name is not None:
            out["name"] = d.name
        if len(text) > self.left or (opener and self.left == 0):
            self.dropped_chars += len(text) - min(len(text), self.left)
            text = text[: self.left]
            out["elided"] = True
        self.left -= len(text)
        out["text"] = text
        return out

    def end_fields(self) -> dict[str, int]:
        """Drop counters for ``request_end``, present only when above 0."""
        out: dict[str, int] = {}
        if self.dropped_chars:
            out["tool_call_chars_dropped"] = self.dropped_chars
        if self.dropped_calls:
            out["tool_calls_dropped"] = self.dropped_calls
        return out
