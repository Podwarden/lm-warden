"""0-based turn index per session for the session forest (spec §3).

In memory only. After a restart a session restarts at 0; the forest orders
turns by time, so the index is a hint, never a key.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable


class SessionTurns:
    def __init__(
        self,
        max_entries: int = 10_000,
        ttl_s: float = 86_400.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max(1, int(max_entries))
        self._ttl = float(ttl_s)
        self._clock = clock
        self._d: OrderedDict[str, tuple[int, float]] = OrderedDict()

    def next(self, session_key: str | None) -> int | None:
        if not session_key:
            return None
        now = self._clock()
        prev = self._d.pop(session_key, None)
        n = 0 if prev is None or now - prev[1] > self._ttl else prev[0] + 1
        self._d[session_key] = (n, now)
        while len(self._d) > self._max:
            self._d.popitem(last=False)
        return n
