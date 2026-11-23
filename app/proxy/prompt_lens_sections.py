"""G9: the whole section on expand -- a bounded, in-memory section store.

The Prompt Lens stream carries each request's thinking, reply and tool calls as
deltas, and the ring evicts the oldest of them; the dock keeps a client cap too.
So the dock's pop-up fetches a WHOLE section from here instead: the hub feeds
this store every event it publishes (``PromptLensHub.publish``, kept and live-only
alike), and the store assembles, per request,

* ``thinking`` (the ``reasoning`` channel) and ``reply`` (``content``): text,
  capped per channel at ``max_channel_chars``; the oldest text goes first and
  is counted in ``dropped_chars``, and offsets stay absolute;
* ``tool_calls``: the calls by index (``call_id``, ``name``, ``args``), within
  the same per-channel cap; there the TAIL is dropped and counted, as in the
  publisher's ``ToolCallBudget`` (a streamed call cannot lose its start);
* ``tool_results`` and ``prompt``: from the WIDE structured prompt capture
  (``put_prompt``), whose window is larger than the stream event's.

Bounds (spec 2026-11-22, "Section store budget"): ``max_channel_chars`` per
channel, ``total_chars`` over every stored text (the wide prompt included),
``max_requests`` requests, and ``ttl_s`` after ``request_end``. The newest
request always stays, even alone over the budget.

Isolation (spec section 4, Task 7's model): eviction is charged per key first.
Each key has its own budget (``key_chars``, ``key_requests``), and over it the
key's own oldest finished request goes first (else its oldest), so whether a
key's section is still here depends on its own traffic only. The total bounds
are the memory backstop. Past them, eviction takes, in order: a key over its
own budget (heaviest first), then the fed key's own older requests, then the
heaviest key's oldest request. A key under its own budget is never chosen
while another key is over its budget, so a quiet key keeps its sections.
Residual (accepted, documented): the heaviest key can lose its oldest section
to other keys' combined volume, which tells it only that total traffic was
high, never what or whose.

Never raises into the hub, never logs content, nothing on disk. A key with
``keep_captures`` on (``put_prompt(keep=<its generation>)``) also hands a record of the
finished request to ``on_kept`` (app/proxy/prompt_lens_kept.py persists it).
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from typing import Any, Literal, get_args

SectionKind = Literal["thinking", "reply", "tool_calls", "tool_results", "prompt"]
SECTION_KINDS: tuple[str, ...] = get_args(SectionKind)

#: Delta channel -> its text section.
_TEXT_KINDS = {"reasoning": "thinking", "content": "reply"}
#: Prompt roles the ``tool_results`` section lists.
_RESULT_ROLES = ("tool", "function")


@dataclass(frozen=True)
class SectionView:
    """One section as fetched. ``seq`` is the hub seq of the last event folded
    in (admin only; a key holder never gets the hub's seq). ``offset`` is the
    absolute char offset ``text`` starts at (>= ``dropped_chars``), so
    ``offset + len(text)`` is the section's whole length. ``calls`` (tool
    calls, by index) and ``messages`` (tool results, or the prompt with its
    ``sys`` and ``tools`` entries first) are set for those kinds only.
    ``older`` (prompt only) is the capture's collapsed summary. ``kept`` is
    True when the section comes from a kept (persisted) capture."""

    token_id: str
    kind: str
    seq: int
    live: bool
    dropped_chars: int
    offset: int
    text: str | None = None
    calls: list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]] | None = None
    dropped_calls: int = 0
    older: dict[str, Any] | None = None
    kept: bool = False


#: Small deltas are coalesced into chunks of about this many chars, so a
#: token-per-delta stream does not cost one string object per token (review I2).
_CHUNK = 4096
#: A larger delta is stored in pieces of at most this, so trimming the front of
#: a capped channel never touches more than one piece.
_PIECE = 65536


class _Text:
    """Text, oldest chunks first, capped; drops from the front.

    Deltas are coalesced into ~``_CHUNK``-char chunks (big ones split into
    ``_PIECE`` pieces). Dropping from the front moves ``skip`` into the first
    chunk instead of slicing it, and a read joins without storing the join
    back, so no delta ever copies the whole channel (review M12)."""

    __slots__ = ("chunks", "skip", "size", "dropped")

    def __init__(self) -> None:
        self.chunks: deque[str] = deque()
        self.skip = 0
        self.size = 0
        self.dropped = 0

    def add(self, s: str, cap: int | None = None) -> int:
        """Append ``s``, trim the front to ``cap``; the change in stored chars."""
        if not s:
            return 0
        before = self.size
        last = self.chunks[-1] if self.chunks else None
        if last is not None and len(last) + len(s) <= _CHUNK:
            self.chunks[-1] = last + s
        else:
            for i in range(0, len(s), _PIECE):
                self.chunks.append(s[i : i + _PIECE])
        self.size += len(s)
        if cap is not None:
            while self.size > cap and self.chunks:
                over = self.size - cap
                left = len(self.chunks[0]) - self.skip
                if left <= over:
                    self.chunks.popleft()
                    self.skip = 0
                    self.size -= left
                    self.dropped += left
                else:
                    self.skip += over
                    self.size -= over
                    self.dropped += over
        return self.size - before

    def text(self) -> str:
        if not self.chunks:
            return ""
        if self.skip:
            return self.chunks[0][self.skip :] + "".join(islice(self.chunks, 1, None))
        return "".join(self.chunks)


class _Call:
    __slots__ = ("call_id", "name", "args", "elided")

    def __init__(self) -> None:
        self.call_id: str | None = None
        self.name: str | None = None
        self.args = _Text()
        self.elided = False

    def out(self, index: int) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": index,
            "call_id": self.call_id,
            "name": self.name,
            "args": self.args.text(),
        }
        if self.elided:
            d["elided"] = True
        return d


class _Req:
    __slots__ = (
        "token_id",
        "request_id",
        "seq",
        "live",
        "ended",
        "texts",
        "calls",
        "calls_chars",
        "calls_dropped_chars",
        "calls_dropped",
        "calls_closed",
        "pub_dropped_chars",
        "pub_dropped_calls",
        "prompt",
        "prompt_cost",
        "cost",
        "keep",
        "start",
        "kvsig",
    )

    def __init__(self, token_id: str, request_id: Any, seq: int, start: dict[str, Any]) -> None:
        self.token_id = token_id
        self.request_id = request_id if isinstance(request_id, str) else None
        self.seq = seq
        self.live = True
        self.ended = 0.0
        self.texts: dict[str, _Text] = {}
        self.calls: dict[int, _Call] = {}
        self.calls_chars = 0
        self.calls_dropped_chars = 0
        self.calls_dropped = 0
        self.calls_closed = False
        self.pub_dropped_chars = 0
        self.pub_dropped_calls = 0
        self.prompt: dict[str, Any] | None = None
        self.prompt_cost = 0
        self.cost = 0
        #: The key's keep_captures generation at request start; 0 = not kept.
        self.keep = 0
        #: The request_start and kvsig events, held only for a kept request
        #: (refs to the hub's own dicts, never changed here).
        self.start: dict[str, Any] | None = start
        self.kvsig: dict[str, Any] | None = None


def _chars(node: Any) -> int:
    """Total length of the strings in ``node`` (iterative), as the hub counts."""
    total = 0
    stack: list[Any] = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, str):
            total += len(n)
        elif isinstance(n, dict):
            stack.extend(n.values())
        elif isinstance(n, list | tuple):
            stack.extend(n)
    return total


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


def _clip(text: str, dropped: int, start: int) -> tuple[int, str]:
    """(offset, text from it) for an absolute ``start``."""
    end = dropped + len(text)
    offset = min(max(start, dropped), end)
    return offset, text[offset - dropped :]


def _prompt_messages(
    prompt: dict[str, Any] | None, kind: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    msgs = prompt.get("messages") if isinstance(prompt, dict) else None
    msgs = [m for m in msgs if isinstance(m, dict)] if isinstance(msgs, list) else []
    if kind == "tool_results":
        return [m for m in msgs if m.get("role") in _RESULT_ROLES], None
    head: list[dict[str, Any]] = []
    older: dict[str, Any] | None = None
    if prompt is not None:
        if isinstance(prompt.get("sys"), dict):
            head.append({"role": "sys", **prompt["sys"]})
        if isinstance(prompt.get("tools"), dict):
            head.append({"role": "tools", **prompt["tools"]})
        older = {
            k: prompt.get(k)
            for k in ("older", "collapsed", "omitted", "total_chars")
            if k in prompt
        }
    return head + msgs, older


def _view(
    *,
    token_id: str,
    kind: str,
    seq: int,
    live: bool,
    kept: bool,
    start: int,
    text_of: Callable[[str], tuple[str, int]],
    calls: list[dict[str, Any]],
    calls_dropped_chars: int,
    calls_dropped: int,
    prompt: dict[str, Any] | None,
) -> SectionView | None:
    base: dict[str, Any] = {"token_id": token_id, "kind": kind, "seq": seq, "live": live}
    if kind in ("thinking", "reply"):
        text, dropped = text_of(kind)
        offset, part = _clip(text, dropped, start)
        return SectionView(**base, dropped_chars=dropped, offset=offset, text=part, kept=kept)
    if kind == "tool_calls":
        return SectionView(
            **base,
            dropped_chars=calls_dropped_chars,
            offset=0,
            calls=calls,
            dropped_calls=calls_dropped,
            kept=kept,
        )
    if kind in ("tool_results", "prompt"):
        msgs, older = _prompt_messages(prompt, kind)
        return SectionView(**base, dropped_chars=0, offset=0, messages=msgs, older=older, kept=kept)
    return None


def kept_view(record: dict[str, Any], kind: str, *, start: int = 0) -> SectionView | None:
    """One section of a kept capture (the record ``on_kept`` received, as
    persisted), or None for an unknown kind or an unreadable record."""
    try:
        sections = record.get("sections") or {}
        end = record.get("end") or {}

        def text_of(k: str) -> tuple[str, int]:
            s = sections.get(k) or {}
            t = s.get("text")
            return (t if isinstance(t, str) else ""), _int(s.get("dropped_chars"))

        tc = sections.get("tool_calls") or {}
        calls = tc.get("calls")
        return _view(
            token_id=str(record.get("token_id")),
            kind=kind,
            seq=_int(end.get("seq")),
            live=False,
            kept=True,
            start=max(0, start),
            text_of=text_of,
            calls=[c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else [],
            calls_dropped_chars=_int(tc.get("dropped_chars")),
            calls_dropped=_int(tc.get("dropped_calls")),
            prompt=sections.get("prompt") if isinstance(sections.get("prompt"), dict) else None,
        )
    except Exception:  # noqa: BLE001 -- a bad row reads as "no such capture"
        return None


class PromptLensSections:
    """The bounded section store (module docstring). Instantiated once on
    ``app.state.prompt_lens_sections`` and handed to the hub."""

    def __init__(
        self,
        *,
        total_chars: int = 16_000_000,
        max_channel_chars: int = 2_000_000,
        max_requests: int = 512,
        ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
        key_chars: int | None = None,
        key_requests: int | None = None,
        on_kept: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._total_max = max(0, total_chars)
        self._chan_max = max(0, max_channel_chars)
        self._req_max = max(1, max_requests)
        self._ttl = ttl_s
        self._clock = clock
        #: Per key: a quarter of the total and an eighth of the requests by
        #: default, so four keys at their full budget fit the backstop.
        self._key_chars = self._total_max // 4 if key_chars is None else max(0, key_chars)
        self._key_reqs = max(1, self._req_max // 8 if key_requests is None else key_requests)
        self._on_kept = on_kept
        self._reqs: dict[str, _Req] = {}
        #: token id -> its request ids, oldest first; keys least recently active first.
        self._keys: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()
        self._key_cost: dict[str, int] = {}
        #: (ended at, req id), in end order, for the TTL.
        self._ends: deque[tuple[float, str]] = deque()
        self._total = 0

    # -- feed (sync, never raises: called from PromptLensHub.publish) --

    def feed(self, event: dict[str, Any]) -> None:
        try:
            self._feed(event)
        except Exception:  # noqa: BLE001 -- display only, never into the hub
            pass

    def _feed(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "progress" or not isinstance(kind, str):
            return
        rid = event.get("req_id")
        if not isinstance(rid, str):
            return
        self._expire()
        seq = _int(event.get("seq"))
        if kind == "request_start":
            self._open(rid, event, seq)
            return
        r = self._reqs.get(rid)
        if r is None or event.get("token_id") != r.token_id:
            return
        r.seq = max(r.seq, seq)
        if kind == "delta":
            self._delta(r, event)
        elif kind == "kvsig":
            if r.keep:
                r.kvsig = event
        elif kind == "request_end" and r.live:
            r.live = False
            r.ended = self._clock()
            self._ends.append((r.ended, rid))
            if len(self._ends) > 2 * self._req_max:
                # Entries of evicted requests stay until their TTL: compact (M11).
                self._ends = deque(
                    (t, i)
                    for t, i in self._ends
                    if (q := self._reqs.get(i)) is not None and not q.live and q.ended == t
                )
            r.pub_dropped_chars = _int(event.get("tool_call_chars_dropped"))
            r.pub_dropped_calls = _int(event.get("tool_calls_dropped"))
            if r.keep:
                self._hand_over(rid, r, event)
            r.start = r.kvsig = None
        self._touch(r.token_id)
        self._shrink(rid)

    def _open(self, rid: str, event: dict[str, Any], seq: int) -> None:
        tid = event.get("token_id")
        tid = tid if isinstance(tid, str) else ""
        if rid in self._reqs:
            self._drop(rid)
        self._reqs[rid] = _Req(tid, event.get("request_id"), seq, event)
        self._keys.setdefault(tid, OrderedDict())[rid] = None
        self._key_cost.setdefault(tid, 0)
        self._touch(tid)
        self._shrink(rid)

    def _delta(self, r: _Req, event: dict[str, Any]) -> None:
        channel = event.get("channel")
        text = event.get("text")
        text = text if isinstance(text, str) else ""
        tk = _TEXT_KINDS.get(channel) if isinstance(channel, str) else None
        if tk is not None:
            buf = r.texts.get(tk)
            if buf is None:
                buf = r.texts[tk] = _Text()
            self._charge(r, buf.add(text, self._chan_max))
        elif channel == "tool_call":
            self._tool_call(r, event, text)

    def _tool_call(self, r: _Req, event: dict[str, Any], text: str) -> None:
        index = event.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return
        cid, name = event.get("call_id"), event.get("name")
        cid = cid if isinstance(cid, str) else None
        name = name if isinstance(name, str) else None
        call = r.calls.get(index)
        head = len(cid or "") + len(name or "")
        left = self._chan_max - r.calls_chars
        if r.calls_closed or head > left or (call is None and cid is None and name is None):
            # One clean tail, as ToolCallBudget: nothing after a cut is kept.
            r.calls_dropped_chars += len(text)
            if call is None and (cid is not None or name is not None):
                r.calls_dropped += 1
            r.calls_closed = r.calls_closed or head > left
            return
        if call is None:
            call = r.calls[index] = _Call()
        if cid is not None:
            call.call_id = cid
        if name is not None:
            call.name = name
        left -= head
        if len(text) > left:
            r.calls_dropped_chars += len(text) - left
            text = text[:left]
            call.elided = True
            r.calls_closed = True
        if event.get("elided") is True:
            call.elided = True
        call.args.add(text)
        used = head + len(text)
        r.calls_chars += used
        self._charge(r, used)

    def put_prompt(
        self, req_id: str, token_id: str, parts: dict[str, Any] | None, *, keep: int = 0
    ) -> None:
        """Attach the wide structured prompt capture to ``req_id`` (whose
        request_start was just fed), and the key's keep_captures generation
        (0: not kept; the writer stores the record only while the key still
        has this generation, so a switch turned off and on again in between
        never brings an older capture back).
        Ignored for an unknown request or another key's. Never raises."""
        try:
            r = self._reqs.get(req_id)
            if r is None or r.token_id != token_id:
                return
            r.keep = int(keep) if isinstance(keep, int) else 0
            if not r.keep:
                r.start = None
            if r.prompt is not None:
                self._charge(r, -r.prompt_cost)
            r.prompt = parts if isinstance(parts, dict) else None
            r.prompt_cost = _chars(r.prompt) if r.prompt is not None else 0
            self._charge(r, r.prompt_cost)
            self._shrink(req_id)
        except Exception:  # noqa: BLE001 -- display only
            pass

    def _hand_over(self, rid: str, r: _Req, end: dict[str, Any]) -> None:
        if self._on_kept is None:
            return

        def section(k: str) -> dict[str, Any]:
            buf = r.texts.get(k)
            return {"text": buf.text() if buf else "", "dropped_chars": buf.dropped if buf else 0}

        # The narrow structured prompt is not kept: the wide one is (review I5).
        start = (
            {k: v for k, v in r.start.items() if k != "prompt_parts"}
            if isinstance(r.start, dict)
            else None
        )
        record = {
            "req_id": rid,
            "token_id": r.token_id,
            "request_id": r.request_id,
            "keep_gen": r.keep,
            "start": start,
            "kvsig": r.kvsig,
            "end": end,
            "sections": {
                "thinking": section("thinking"),
                "reply": section("reply"),
                "tool_calls": {
                    "calls": [r.calls[i].out(i) for i in sorted(r.calls)],
                    "dropped_chars": r.calls_dropped_chars + r.pub_dropped_chars,
                    "dropped_calls": r.calls_dropped + r.pub_dropped_calls,
                },
                "prompt": r.prompt,
            },
        }
        try:
            self._on_kept(record)
        except Exception:  # noqa: BLE001 -- persistence is best effort
            pass

    # -- bounds --

    def _charge(self, r: _Req, n: int) -> None:
        r.cost += n
        self._total += n
        self._key_cost[r.token_id] = self._key_cost.get(r.token_id, 0) + n

    def _touch(self, tid: str) -> None:
        if tid in self._keys:
            self._keys.move_to_end(tid)

    def _drop(self, rid: str) -> None:
        r = self._reqs.pop(rid, None)
        if r is None:
            return
        self._total -= r.cost
        ids = self._keys.get(r.token_id)
        if ids is not None:
            ids.pop(rid, None)
            self._key_cost[r.token_id] = self._key_cost.get(r.token_id, 0) - r.cost
            if not ids:
                del self._keys[r.token_id]
                self._key_cost.pop(r.token_id, None)

    def _victim(self, tid: str, keep: str) -> str | None:
        """Key ``tid``'s oldest finished request, else its oldest; never ``keep``."""
        ids = self._keys.get(tid)
        if not ids:
            return None
        oldest = None
        for rid in ids:
            if rid == keep:
                continue
            if oldest is None:
                oldest = rid
            r = self._reqs.get(rid)
            if r is not None and not r.live:
                return rid
        return oldest

    def _shrink(self, fed: str) -> None:
        r = self._reqs.get(fed)
        tid = r.token_id if r is not None else None
        if tid is not None:
            while (
                len(self._keys.get(tid) or ()) > self._key_reqs
                or self._key_cost.get(tid, 0) > self._key_chars
            ):
                v = self._victim(tid, fed)
                if v is None:
                    break
                self._drop(v)
        # The backstop (review C1): never the quiet key. First a key over its
        # own budget (heaviest first), then the fed key's own older requests,
        # then the heaviest key -- by request count when the count binds, by
        # chars when the chars do. So a key under its own budget loses a
        # section to other keys' traffic only when it is itself the heaviest.
        while len(self._reqs) > self._req_max or self._total > self._total_max:
            by_count = len(self._reqs) > self._req_max
            v = self._backstop_victim(fed, by_count=by_count, over_only=True)
            if v is None and tid is not None:
                v = self._victim(tid, fed)
            if v is None:
                v = self._backstop_victim(fed, by_count=by_count, over_only=False)
            if v is None:
                break
            self._drop(v)

    def _over(self, k: str) -> bool:
        return len(self._keys[k]) > self._key_reqs or self._key_cost.get(k, 0) > self._key_chars

    def _backstop_victim(self, fed: str, *, by_count: bool, over_only: bool) -> str | None:
        def weight(k: str) -> int:
            return len(self._keys[k]) if by_count else self._key_cost.get(k, 0)

        keys = [k for k in self._keys if not over_only or self._over(k)]
        for k in sorted(keys, key=weight, reverse=True):
            v = self._victim(k, fed)
            if v is not None:
                return v
        return None

    def _expire(self) -> None:
        now = self._clock()
        while self._ends and now - self._ends[0][0] > self._ttl:
            ended, rid = self._ends.popleft()
            r = self._reqs.get(rid)
            if r is not None and not r.live and r.ended == ended:
                self._drop(rid)

    # -- read --

    def section(self, req_id: str, kind: SectionKind, *, start: int = 0) -> SectionView | None:
        """One section of ``req_id``, text from the absolute offset ``start``;
        None for an unknown, evicted or expired request or an unknown kind."""
        try:
            self._expire()
            r = self._reqs.get(req_id)
            if r is None:
                return None

            def text_of(k: str) -> tuple[str, int]:
                buf = r.texts.get(k)
                return (buf.text(), buf.dropped) if buf is not None else ("", 0)

            return _view(
                token_id=r.token_id,
                kind=kind,
                seq=r.seq,
                live=r.live,
                kept=False,
                start=max(0, start),
                text_of=text_of,
                calls=[r.calls[i].out(i) for i in sorted(r.calls)],
                calls_dropped_chars=r.calls_dropped_chars + r.pub_dropped_chars,
                calls_dropped=r.calls_dropped + r.pub_dropped_calls,
                prompt=r.prompt,
            )
        except Exception:  # noqa: BLE001 -- display only
            return None

    def clear(self) -> None:
        """Forget every section held in memory (Prompt Lens turned off at
        runtime). Kept captures already handed to ``on_kept`` are not touched."""
        self._reqs.clear()
        self._keys.clear()
        self._key_cost.clear()
        self._ends.clear()
        self._total = 0

    def size_chars(self) -> int:
        """Chars held now, over every stored text and the wide prompts."""
        return self._total
