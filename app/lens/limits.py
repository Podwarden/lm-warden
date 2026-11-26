"""Rate limits of the Prompt Lens API and MCP (analysis Task 4.1).

One token bucket per credential (a key id, or an admin), shared by the REST routes and the MCP
tools, so a client cannot double its budget by using both. Defaults: a key holder 60 calls a
minute and 2 concurrent diffs, an admin 300 a minute (``lens_rate_key_per_min``,
``lens_rate_admin_per_min``). ``check`` returns None when the call may go on, else the seconds
to wait (the ``Retry-After`` of a 429). Bounded in credentials.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

#: Concurrent diffs one key holder may run (each reads two requests and a session).
MAX_DIFFS = 2
_MAX_BUCKETS = 4096


class TokenBucket:
    """``rate_per_min`` tokens a minute, ``burst`` at most; ``take`` spends one."""

    def __init__(
        self, rate_per_min: int, burst: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._rate = max(1, rate_per_min) / 60.0
        self._burst = float(max(1, burst))
        self._clock = clock
        self._tokens = self._burst
        self._at = clock()

    def take(self) -> int | None:
        """None when a token was spent; else the whole seconds until one is there."""
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._at) * self._rate)
        self._at = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return None
        return int((1.0 - self._tokens) / self._rate) + 1

    @property
    def idle(self) -> bool:
        return self._tokens >= self._burst


class LensLimiter:
    def __init__(
        self,
        key_per_min: int = 60,
        admin_per_min: int = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._key = key_per_min
        self._admin = admin_per_min
        self._clock = clock
        self._buckets: dict[tuple[bool, str], TokenBucket] = {}
        self._diffs: dict[str, int] = {}

    def check(self, credential_id: str, *, admin: bool, diff: bool = False) -> int | None:
        """None = allowed (a token spent), else the Retry-After seconds. ``diff`` also needs
        one of the credential's concurrent-diff slots, which ``diff_done`` gives back."""
        if len(self._buckets) > _MAX_BUCKETS:
            for k in [k for k, b in self._buckets.items() if b.idle]:
                del self._buckets[k]
        rate = self._admin if admin else self._key
        bucket = self._buckets.setdefault(
            (admin, credential_id), TokenBucket(rate, rate, self._clock)
        )
        if diff and not admin and self._diffs.get(credential_id, 0) >= MAX_DIFFS:
            return 1
        wait = bucket.take()
        if wait is not None:
            return wait
        if diff and not admin:
            self._diffs[credential_id] = self._diffs.get(credential_id, 0) + 1
        return None

    def diff_done(self, credential_id: str) -> None:
        n = self._diffs.get(credential_id, 0) - 1
        if n > 0:
            self._diffs[credential_id] = n
        else:
            self._diffs.pop(credential_id, None)


def limiter_of(app: Any) -> LensLimiter:
    """The app's limiter, built on first use from the settings."""
    state = app.state
    lim = getattr(state, "lens_limiter", None)
    if lim is None:
        st = getattr(state, "settings", None)
        lim = LensLimiter(
            getattr(st, "lens_rate_key_per_min", 60), getattr(st, "lens_rate_admin_per_min", 300)
        )
        state.lens_limiter = lim
    return lim
