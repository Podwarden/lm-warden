"""Prompt Lens recent heads (lens spec v1 4.6): the cross-session comparison.

Lineage is per session, so a new session's first turn has no parent. For the
head the useful question is "did my previous session of the same app have the
same head?". Per scope (one API key: ``lineage.key_scope``) this keeps, for
each ``(model, loose_sys, tools)``, the last exact ``sys`` id and its line
digests. Ids and keyed digests only, never text. Bounded: ``max_per_key``
entries per scope, ``max_entries`` in all, ``ttl_s`` seconds.

A second index, keyed ``(model, tools)`` only, with the same bounds, serves the
fallback of spec 4.5 (ruling, Task 1.3 fix round 1): a new session in another
directory changes a path line of the system prompt, so its ``loose_sys``
differs and the first index misses. The caller decides whether that entry is
close enough to compare (``app.lens.hot``).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from app.lens.probes import PackedDigests


@dataclass(frozen=True)
class HeadEntry:
    sys_id: str
    digests: PackedDigests
    #: The remembered head's loose sys id (the ``(model, tools)`` index needs it).
    loose_sys: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeadEntry):
            return NotImplemented
        # loose_sys is compared only when both carry it (tests built before it).
        same_loose = not self.loose_sys or not other.loose_sys or self.loose_sys == other.loose_sys
        return self.sys_id == other.sys_id and self.digests == other.digests and same_loose

    def __hash__(self) -> int:
        return hash((self.sys_id, self.digests))


@dataclass(frozen=True)
class _Stamped:
    entry: HeadEntry
    ts: float


_Key = tuple[str, ...]

#: Digests kept per entry: the 1,024 lines and 1,024 chunks after them (16 KB packed).
MAX_DIGESTS = 2_048


class _Index:
    """``max_per_key`` entries per scope and ``max_entries`` in all, oldest first out."""

    def __init__(self, max_per_key: int, ttl_s: float, max_entries: int) -> None:
        self._max = max_per_key
        self._ttl = ttl_s
        self._max_entries = max_entries
        self._scopes: dict[bytes, OrderedDict[_Key, _Stamped]] = {}
        #: Every entry once, oldest first: the global bound.
        self._order: OrderedDict[tuple[bytes, _Key], None] = OrderedDict()

    def _drop(self, scope: bytes, k: _Key) -> None:
        per = self._scopes.get(scope)
        if per is not None:
            per.pop(k, None)
            if not per:
                del self._scopes[scope]
        self._order.pop((scope, k), None)

    def put(self, scope: bytes, k: _Key, entry: HeadEntry, now: float) -> None:
        per = self._scopes.setdefault(scope, OrderedDict())
        per[k] = _Stamped(entry, now)
        per.move_to_end(k)
        self._order[(scope, k)] = None
        self._order.move_to_end((scope, k))
        while len(per) > self._max:
            self._drop(scope, next(iter(per)))
        while len(self._order) > self._max_entries:
            self._drop(*next(iter(self._order)))

    def get(self, scope: bytes, k: _Key, now: float) -> HeadEntry | None:
        per = self._scopes.get(scope)
        hit = per.get(k) if per is not None else None
        if hit is None:
            return None
        if now - hit.ts > self._ttl:
            self._drop(scope, k)
            return None
        return hit.entry

    def __len__(self) -> int:
        return len(self._order)


class RecentHeads:
    """Each index: ``max_per_key`` entries per scope and ``max_entries`` in
    all, each with at most ``MAX_DIGESTS`` packed digests (16 KB): two indexes
    of 4,096 x 16 KB is the worst case (about 130 MB), a few MB in practice.
    An entry is shared by both indexes (one object)."""

    def __init__(
        self, max_per_key: int = 64, ttl_s: float = 21_600, max_entries: int = 4_096
    ) -> None:
        self._main = _Index(max_per_key, ttl_s, max_entries)
        self._by_tools = _Index(max_per_key, ttl_s, max_entries)
        # remember/lookup may run on the event loop and in a worker thread.
        self._lock = threading.Lock()

    def remember(
        self,
        scope: bytes,
        model: str,
        loose_sys: str,
        tools: str,
        sys_id: str,
        digests: Sequence[bytes],
        now: float,
    ) -> None:
        packed = PackedDigests.of(digests)
        if len(packed) > MAX_DIGESTS:
            packed = PackedDigests(packed.raw[: 8 * MAX_DIGESTS])
        entry = HeadEntry(sys_id, packed, loose_sys)
        with self._lock:
            self._main.put(scope, (model, loose_sys, tools), entry, now)
            self._by_tools.put(scope, (model, tools), entry, now)

    def lookup(
        self, scope: bytes, model: str, loose_sys: str, tools: str, now: float
    ) -> HeadEntry | None:
        with self._lock:
            return self._main.get(scope, (model, loose_sys, tools), now)

    def lookup_by_tools(self, scope: bytes, model: str, tools: str, now: float) -> HeadEntry | None:
        """The key's last head with this model and tool list, whatever its
        system prompt (the spec 4.5 fallback)."""
        with self._lock:
            return self._by_tools.get(scope, (model, tools), now)

    def __len__(self) -> int:
        return len(self._main)
