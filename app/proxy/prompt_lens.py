"""Prompt Lens — a process-local, in-memory pub/sub broadcast hub.

The hub taps the proxy hot path (``app/proxy/routes.py::_forward``) and fans
prompt/output events out to the warden's privileged UI stream
(``GET /api/admin/prompt-lens/stream``). It is deliberately tiny and dependency-free:

  * **No DB, no disk.** Everything lives in a bounded in-memory ring; the ring
    dies on pod restart. Prompt/output content is NEVER persisted (see the
    Prompt Lens spec, §2 "History depth").
  * **Never backpressures the proxy.** ``publish`` is *sync* and non-awaiting so
    the tap can call it without ``await`` on the hot path. A slow/full
    subscriber queue drops its OLDEST event rather than blocking ``publish`` —
    Prompt Lens must never break, delay, or backpressure a proxied response.
  * **Heap-bounded.** The ring is capped by BOTH an event count AND a total
    char budget; the oldest events are evicted when either bound is exceeded so
    a burst of huge runaway completions can't blow the heap. The single newest
    event is always kept even if it alone exceeds the char budget (so an
    oversized completion is still visible, not silently swallowed).

The raw bearer token / Authorization header is never placed on an event — the
tap passes only a ``token_label`` (token name, or id prefix) and ``token_id``.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.proxy.prompt_lens_sections import PromptLensSections

# How many events a single subscriber queue holds before drop-oldest kicks in.
# Bounded so one slow SSE client can't pin unbounded memory; the frontend keys
# rows on the monotonic ``seq`` so a gap from a dropped event is recoverable
# (the client can show a "lagged" hint).
DEFAULT_QUEUE_MAXSIZE = 1000

#: Put last on a subscriber's queue to end its stream (app/proxy/prompt_lens_stream.py
#: returns on it): the key holder's re-check, a revoke, and Prompt Lens turned off
#: at runtime (``PromptLensHub.drop_all``).
STOP: Any = object()


def _event_chars(event: dict) -> int:
    """Approximate an event's heap cost as the total length of its string values,
    nested ones included (ruling G-F8).

    The prompt/delta text dominates. Nested text counts too: ``prompt_parts``
    (messages, ``older``), tool-call fields and ``kvsig`` parts would otherwise
    cost 0 and the ring's char cap would understate the heap. Keys and numbers
    are ignored. Used only for ring-eviction accounting, so an approximation is
    fine. Iterative, so a deep event cannot overflow the stack.
    """
    total = 0
    stack: list[Any] = [event]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            total += len(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list | tuple):
            stack.extend(node)
    return total


#: The key holder's own replay window (G7, review I1). Each key's kept events
#: are also held in a window of its own, bounded by these, so what a key holder
#: replays at connect -- and which of its captures are still there -- depends on
#: its own traffic only, never on how busy the other keys are (the shared ring
#: evicts by everyone's traffic, so its depth would count theirs).
KEY_RING_EVENTS = 500
KEY_RING_CHARS = 400_000
#: Starts kept per key by request id (the capture route), beside its window.
#: A start that left the window's events is still charged to its char budget.
KEY_STARTS_MAX = 64
#: Live requests (start seen, no request_end yet) a key's window tracks, so a
#: dock that connects mid-request gets the start back (``restore``). Not
#: charged to the char budget: each entry is a request id plus its kvsig view
#: (64 parts at most); the start itself is the one in ``starts``.
KEY_LIVE_MAX = 32
#: Live requests the hub tracks for the admin's restore (the start and kvsig
#: are the ring's own dicts; entries leave at request_end). Starts held here
#: after the ring evicted them sit outside the ring's char budget: at most
#: LIVE_MAX of them, each bounded by the capture caps.
LIVE_MAX = 512
#: A key's window is dropped this long after its last event (own time only).
KEY_IDLE_S = 1800.0
#: The memory backstop: at most this many key windows. Each holds at most
#: KEY_RING_CHARS of events and kept starts together (plus its newest event,
#: should that one alone be larger), so about 51 M chars in all. Past it the
#: least recently active key's window goes. That is the one place another
#: key's activity shows: only with more than this many keys active within a
#: key's idle time.
KEY_WINDOWS_MAX = 128


class _KeyWindow:
    """One key's own kept events, newest last, bounded (``KEY_RING_*``).

    Holds and is charged with the KEY HOLDER'S VIEW of each event
    (``prompt_lens_scope.for_key_holder``), never the raw event (Task 7 review
    N1): the raw one carries what the key never sees -- kvsig's ``other``
    states and ``parts_key``, ``reusable_any``, request_end's ``cache`` --
    whose size depends on other keys' cache state. Charged with the view,
    what the window keeps and evicts, and the capture route's ``partial``, are
    the same as in a world with no other key.

    ``starts`` maps request id -> (start, cost, orphaned). A start still in
    ``events`` costs nothing more; once evicted from there ("orphaned") its
    cost moves to ``orphan`` and counts against the same char budget (review
    N2), so ``chars + orphan`` stays within it."""

    __slots__ = ("events", "chars", "starts", "orphan", "last", "live")

    def __init__(self) -> None:
        self.events: deque[tuple[dict[str, Any], int]] = deque()
        self.chars = 0
        self.starts: OrderedDict[str, tuple[dict[str, Any], int, bool]] = OrderedDict()
        self.orphan = 0
        self.last = 0.0
        #: req_id -> [request id, kvsig view or None] of each live request.
        self.live: OrderedDict[str, list[Any]] = OrderedDict()

    def _pop_event(self) -> None:
        ev, cost = self.events.popleft()
        self.chars -= cost
        rid = ev.get("request_id") if ev.get("type") == "request_start" else None
        if not isinstance(rid, str):
            return
        ent = self.starts.get(rid)
        if ent is not None and ent[0] is ev and not ent[2]:
            self.starts[rid] = (ev, cost, True)
            self.orphan += cost

    def _drop_start(self, rid: str) -> None:
        _ev, cost, orphaned = self.starts.pop(rid)
        if orphaned:
            self.orphan -= cost

    def _track_live(self, view: dict[str, Any], rid: str | None) -> None:
        kind, req = view.get("type"), view.get("req_id")
        if not isinstance(req, str):
            return
        if kind == "request_start" and rid is not None:
            self.live[req] = [rid, None]
            while len(self.live) > KEY_LIVE_MAX:
                self.live.popitem(last=False)
        elif kind == "kvsig" and req in self.live:
            self.live[req][1] = view
        elif kind == "request_end":
            self.live.pop(req, None)

    def restore(self, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The start (and kvsig) of each live request whose own is not in
        ``snapshot``, marked ``restored``. From this window only: the key's
        own traffic decides what is restored, never another key's."""
        have = {id(e) for e in snapshot}
        out: list[dict[str, Any]] = []
        for rid, kv in self.live.values():
            ent = self.starts.get(rid)
            if ent is None:
                continue  # its start left this window's budget: own traffic did that
            if id(ent[0]) not in have:
                out.append({**ent[0], "restored": True})
            if kv is not None and id(kv) not in have:
                out.append({**kv, "restored": True})
        return out

    def add(self, view: dict[str, Any], rid: str | None, max_events: int, max_chars: int) -> None:
        self._track_live(view, rid)
        cost = _event_chars(view)
        self.events.append((view, cost))
        self.chars += cost
        if rid is not None:
            if rid in self.starts:
                self._drop_start(rid)
            self.starts[rid] = (view, cost, False)
            while len(self.starts) > KEY_STARTS_MAX:
                self._drop_start(next(iter(self.starts)))
        while len(self.events) > max_events:
            self._pop_event()
        # Events first (the newest always stays), then the oldest orphaned starts.
        while len(self.events) > 1 and self.chars + self.orphan > max_chars:
            self._pop_event()
        while self.orphan and self.chars + self.orphan > max_chars:
            oldest = next(r for r, (_e, _c, o) in self.starts.items() if o)
            self._drop_start(oldest)

    def total_chars(self) -> int:
        return self.chars + self.orphan


# Most ids ``GET /api/admin/prompt-lens/stream?token_ids=`` accepts (spec
# 2026-09-18 §3.5). The token page sends one rotation chain -- a handful of
# keys -- so twenty is headroom, and the cap keeps the query string bounded.
MAX_TOKEN_IDS = 20


def parse_token_ids(raw: str | None) -> frozenset[str] | None:
    """``?token_ids=a,b`` -> the ids to keep, or None for "no filter".

    Absent (None) means today's unfiltered stream. Present but naming no id
    (``token_ids=`` or ``token_ids=,``) is refused rather than read as "no
    filter": silently widening an empty selection to every key is exactly
    the substitution a per-token view must never make. Raises ValueError for
    that and for more than MAX_TOKEN_IDS ids; the route answers 422.
    """
    if raw is None:
        return None
    ids = frozenset(part.strip() for part in raw.split(",") if part.strip())
    if not ids:
        raise ValueError("token_ids names no token")
    if len(ids) > MAX_TOKEN_IDS:
        raise ValueError(f"token_ids names more than {MAX_TOKEN_IDS} tokens")
    return ids


def event_matches(event: dict[str, Any], token_ids: frozenset[str] | None) -> bool:
    """Whether a stream filtered to ``token_ids`` carries ``event``.

    Exact because every event of a request -- request_start, each delta and
    request_end -- carries the same ``token_id`` (app/proxy/routes.py), so a
    filter can never keep a request's start and drop its deltas.
    """
    return token_ids is None or event.get("token_id") in token_ids


class PromptLensHub:
    """In-memory pub/sub with a bounded replay ring. Instantiated once on
    ``app.state.prompt_lens_hub`` at startup."""

    def __init__(
        self,
        *,
        ring_events: int = 2000,
        ring_chars: int = 4_000_000,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        key_ring_events: int | None = None,
        key_ring_chars: int | None = None,
        sections: PromptLensSections | None = None,
    ) -> None:
        #: G9: the whole-section store, fed every published event (kept and
        #: live-only) right after its seq is set.
        self._sections = sections
        self._ring_events = ring_events
        self._ring_chars = ring_chars
        # Never larger than the shared ring (a small test hub stays small).
        self._key_events = min(
            ring_events, KEY_RING_EVENTS if key_ring_events is None else key_ring_events
        )
        self._key_chars = min(
            ring_chars, KEY_RING_CHARS if key_ring_chars is None else key_ring_chars
        )
        #: token id -> its own window, least recently active first.
        self._keys: OrderedDict[str, _KeyWindow] = OrderedDict()
        #: req_id -> [request_start, kvsig or None] of each live request.
        self._live: OrderedDict[str, list[Any]] = OrderedDict()
        self._queue_maxsize = queue_maxsize
        # deque of (event, char_cost) so eviction can decrement the running
        # total in O(1) without re-measuring the evicted event.
        self._ring: deque[tuple[dict, int]] = deque()
        self._ring_chars_total = 0
        self._seq = 0
        #: queue -> the token ids it takes (None: every event). A filtered
        #: queue (the key holder's stream, a per-token admin view) is never
        #: offered another key's event, so that traffic can neither wake it
        #: nor push it into "lagged".
        self._subscribers: dict[asyncio.Queue[Any], frozenset[str] | None] = {}
        #: False while Prompt Lens is off at runtime (``drop_all``): publish
        #: then keeps and sends nothing, so a request already running when it
        #: went off captures nothing more.
        self.accepting = True
        #: Bumped by every ``drop_all``. A request binds to it when it starts
        #: (``bound``); its later events are dropped once it has moved on.
        self.epoch = 0

    # -- publish (sync, non-awaiting — safe to call on the proxy hot path) --

    def publish(self, event: dict[str, Any], *, keep: bool = True) -> int:
        """Assign a monotonic ``seq``, append to the ring (evicting as needed),
        then non-blocking-offer to every subscriber. Returns the seq.

        ``keep=False`` is a live-only event (G6 ``progress``): it gets a seq and
        reaches the current subscribers, but is neither appended to the ring nor
        kept by request id, so a reconnect never replays it and it costs the
        ring nothing.

        Never awaits, never blocks, never raises on a full subscriber queue.
        Returns 0 (keeping nothing) while the hub is not ``accepting``.
        """
        if not self.accepting:
            return 0
        self._seq += 1
        event["seq"] = self._seq
        if self._sections is not None:
            # Before the live-only return (Task 6 carry); never into publish.
            try:
                self._sections.feed(event)
            except Exception:  # noqa: BLE001 -- display only
                pass
        if not keep:
            self._fan_out(event)
            return self._seq
        rid = event.get("request_id") if event.get("type") == "request_start" else None
        self._track_live(event)
        cost = _event_chars(event)
        self._ring.append((event, cost))
        self._ring_chars_total += cost
        tid = event.get("token_id")
        if isinstance(tid, str):
            self._keep_for_key(tid, event, rid if isinstance(rid, str) else None)

        # Evict oldest past the event-count bound.
        while len(self._ring) > self._ring_events:
            _old, old_cost = self._ring.popleft()
            self._ring_chars_total -= old_cost
        # Evict oldest past the char budget, but always keep the newest event
        # (a single runaway completion bigger than the whole budget must still
        # be visible rather than evicting itself into the void).
        while len(self._ring) > 1 and self._ring_chars_total > self._ring_chars:
            _old, old_cost = self._ring.popleft()
            self._ring_chars_total -= old_cost

        self._fan_out(event)
        return self._seq

    def _keep_for_key(self, tid: str, event: dict[str, Any], rid: str | None) -> None:
        from app.proxy.prompt_lens_scope import for_key_holder

        now = time.monotonic()
        # Idle windows out first: by their own last event, oldest first.
        while self._keys:
            k, idle = next(iter(self._keys.items()))
            if now - idle.last <= KEY_IDLE_S:
                break
            del self._keys[k]
        w = self._keys.get(tid)
        if w is None:
            w = self._keys[tid] = _KeyWindow()
        else:
            self._keys.move_to_end(tid)
        w.last = now
        w.add(for_key_holder(event), rid, self._key_events, self._key_chars)
        while len(self._keys) > KEY_WINDOWS_MAX:
            self._keys.popitem(last=False)

    def _track_live(self, event: dict[str, Any]) -> None:
        kind, req = event.get("type"), event.get("req_id")
        if not isinstance(req, str):
            return
        if kind == "request_start":
            self._live[req] = [event, None]
            while len(self._live) > LIVE_MAX:
                self._live.popitem(last=False)
        elif kind == "kvsig" and req in self._live:
            self._live[req][1] = event
        elif kind == "request_end":
            self._live.pop(req, None)

    def restore(
        self, snapshot: list[dict[str, Any]], token_ids: frozenset[str] | None = None
    ) -> list[dict[str, Any]]:
        """For the admin's replay: the request_start (and kvsig) of every live
        request whose own already left ``snapshot`` (the ring evicts by size,
        and one token per delta fills it in seconds), oldest first, as copies
        marked ``restored: true``. Filtered by ``token_ids`` as the stream is."""
        have = {id(e) for e in snapshot}
        out: list[dict[str, Any]] = []
        for start, kv in self._live.values():
            if not event_matches(start, token_ids):
                continue
            if id(start) not in have:
                out.append({**start, "restored": True})
            if kv is not None and id(kv) not in have:
                out.append({**kv, "restored": True})
        return out

    def key_restore(self, tid: str, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """``restore`` for key ``tid``'s holder: its own live requests only,
        from its own window, already in the key holder's view."""
        w = self._keys.get(tid)
        return w.restore(snapshot) if w is not None else []

    def key_chars(self, tid: str) -> int:
        """What key ``tid``'s window holds, in chars (events and kept starts)."""
        w = self._keys.get(tid)
        return w.total_chars() if w is not None else 0

    def key_events(self, tid: str) -> list[dict[str, Any]]:
        """Key ``tid``'s own kept events, oldest first (its window, not the
        ring), each as its holder sees it (``for_key_holder``)."""
        w = self._keys.get(tid)
        return [e for e, _c in w.events] if w is not None else []

    def key_start_of(self, tid: str, request_id: str) -> dict[str, Any] | None:
        """Key ``tid``'s request_start of ``request_id``, from its own window."""
        w = self._keys.get(tid)
        ent = w.starts.get(request_id) if w is not None else None
        return ent[0] if ent is not None else None

    def subscribe_key(self, tid: str) -> tuple[asyncio.Queue[Any], list[dict[str, Any]]]:
        """``subscribe`` for one key holder: the live queue takes ``tid``'s
        events only, and the snapshot is the key's own window."""
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._queue_maxsize)
        q._prompt_lens_lagged = False  # type: ignore[attr-defined]
        self._subscribers[q] = frozenset({tid})
        return q, self.key_events(tid)

    def _fan_out(self, event: dict[str, Any]) -> None:
        tid = event.get("token_id")
        for q, wanted in self._subscribers.items():
            if wanted is None or tid in wanted:
                self._offer(q, event)

    @staticmethod
    def _offer(q: asyncio.Queue, event: dict) -> None:
        """Non-blocking put; on a full queue drop the OLDEST item for that
        subscriber and mark it lagged. Never blocks, never raises."""
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
            q._prompt_lens_lagged = True  # type: ignore[attr-defined]

    # -- subscribe / unsubscribe --

    def subscribe(
        self, token_ids: frozenset[str] | None = None
    ) -> tuple[asyncio.Queue[Any], list[dict[str, Any]]]:
        """Register a new subscriber. Returns ``(queue, ring_snapshot)`` where
        the snapshot is a shallow copy of the current ring for replay-on-connect
        (later publishes do not mutate a returned snapshot list).

        ``token_ids`` (``event_matches``' filter) narrows the LIVE queue to those
        keys' events; the snapshot is the whole ring, filtered by the caller."""
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._queue_maxsize)
        q._prompt_lens_lagged = False  # type: ignore[attr-defined]
        self._subscribers[q] = token_ids
        snapshot = [event for event, _cost in self._ring]
        return q, snapshot

    def events(self) -> list[dict[str, Any]]:
        """A shallow copy of the ring, oldest first (no subscription): for tests
        and diagnostics. Key holders read their own window (``key_events``)."""
        return [event for event, _cost in self._ring]

    def ring_span(self) -> dict[str, Any]:
        """What the replay ring holds now: the oldest kept event's ``ts`` (None
        when the ring is empty or that event has none), the event count, and its
        char cost against the budget. Feeds the stream's ``ring`` frame."""
        oldest_ts = None
        if self._ring:
            ts = self._ring[0][0].get("ts")
            if isinstance(ts, int | float) and not isinstance(ts, bool):
                oldest_ts = float(ts)
        return {
            "oldest_ts": oldest_ts,
            "events": len(self._ring),
            "chars": self._ring_chars_total,
            "chars_max": self._ring_chars,
        }

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.pop(q, None)

    def bound(self) -> _BoundHub:
        """This hub as one request sees it: its ``publish`` drops every event
        once Prompt Lens has been turned off after the request started, even
        when it is back on (no orphan rows of a request whose start is gone).
        Everything else reads through to the hub."""
        return _BoundHub(self, self.epoch)

    def drop_all(self) -> None:
        """Prompt Lens turned off at runtime: forget every event held (the
        ring, the key windows, the live requests), stop accepting new ones,
        and end every open stream with ``STOP`` (each queue is emptied first,
        so the mark always fits). ``accepting = True`` turns it back on."""
        self.accepting = False
        self.epoch += 1
        self._ring.clear()
        self._ring_chars_total = 0
        self._keys.clear()
        self._live.clear()
        for q in list(self._subscribers):
            while True:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                q.put_nowait(STOP)
            except asyncio.QueueFull:  # pragma: no cover -- emptied above
                pass

    # -- introspection (tests / diagnostics) --

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @staticmethod
    def is_lagged(q: asyncio.Queue) -> bool:
        return bool(getattr(q, "_prompt_lens_lagged", False))

    @staticmethod
    def take_lagged(q: asyncio.Queue[Any]) -> bool:
        """Whether ``q`` dropped events since the last call; clears the mark."""
        lagged = bool(getattr(q, "_prompt_lens_lagged", False))
        q._prompt_lens_lagged = False  # type: ignore[attr-defined]
        return lagged


class _BoundHub:
    """``PromptLensHub.bound``: a request's view of the hub, fixed to its epoch."""

    def __init__(self, hub: PromptLensHub, epoch: int) -> None:
        self._hub = hub
        self._epoch = epoch

    def publish(self, event: dict[str, Any], *, keep: bool = True) -> int:
        if self._hub.epoch != self._epoch:
            return 0
        return self._hub.publish(event, keep=keep)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hub, name)


# Mime allowlist for stored media: an explicit raster set, not a generic
# image/* pattern. Stored images are rendered via blob: URLs, which inherit
# the WARDEN PAGE'S OWN ORIGIN (unlike a plain <img src>) — so a scripted
# format opened directly (e.g. "open image in new tab") would run same-origin
# against the admin API. image/svg+xml is exactly that: SVG can carry
# <script>, so it is excluded even though it matches image/*. See spec
# 2026-08-03 §5.
_MEDIA_MIME_ALLOW = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/avif",
        "image/bmp",
    }
)


class PromptLensMediaStore:
    """Bounded in-memory blob store for Prompt Lens images (spec 2026-08-03, §3).

    Same contract family as :class:`PromptLensHub`: sync, non-raising,
    heap-bounded, oldest-first eviction, nothing persisted. Values are the
    base64 payload *string* sliced straight out of the data URI — decoding
    happens at serve time in the endpoint, never on the proxy hot path.
    """

    def __init__(
        self,
        *,
        store_chars: int = 64_000_000,
        max_item_chars: int = 14_000_000,
    ) -> None:
        self._store_chars = store_chars
        self._max_item_chars = max_item_chars
        # dict preserves insertion order -> oldest-first eviction via iteration.
        self._items: dict[str, tuple[str, str]] = {}
        self._chars_total = 0

    def put(self, media_id: str, mime: str, b64: str) -> bool:
        """Store one image payload. Returns False (storing nothing) on any
        rejected/garbage input. Never raises — same hot-path contract as
        ``PromptLensHub.publish``."""
        try:
            if not (isinstance(media_id, str) and isinstance(mime, str) and isinstance(b64, str)):
                return False
            if mime not in _MEDIA_MIME_ALLOW:
                return False
            if len(b64) > self._max_item_chars:
                return False
            old = self._items.pop(media_id, None)
            if old is not None:
                self._chars_total -= len(old[1])
            self._items[media_id] = (mime, b64)
            self._chars_total += len(b64)
            # Oldest-first eviction past the total budget; the newest item
            # always survives (mirrors the hub ring's newest-kept rule).
            while len(self._items) > 1 and self._chars_total > self._store_chars:
                oldest_id = next(iter(self._items))
                _mime, old_b64 = self._items.pop(oldest_id)
                self._chars_total -= len(old_b64)
            return True
        except Exception:
            return False

    def get(self, media_id: str) -> tuple[str, str] | None:
        try:
            return self._items.get(media_id)
        except Exception:
            return None

    def size_chars(self) -> int:
        return self._chars_total

    def clear(self) -> None:
        """Forget every stored image (Prompt Lens turned off at runtime)."""
        self._items.clear()
        self._chars_total = 0
