"""Lightweight counter for in-flight chat-playground proxy requests.

The S8 plan flagged this as a MED risk: when a browser tab is closed
mid-stream, the upstream vLLM request must be cancelled so the model's
slot is released for the next caller. ``StreamingResponse`` in FastAPI
propagates the asyncio CancelledError into the generator when the
client disconnects, and we drive ``aclose()`` from the generator's
``finally`` block. To prove that cleanup actually happens (vs. silently
leaking handles), this counter increments at the top of every
playground stream and decrements in the same ``finally`` block — the
Playwright happy-path test polls ``GET /api/admin/active-requests``
before, during, and after the abort to confirm the counter returns to
zero.

The counter is intentionally narrower than a full request registry:

* No per-request metadata. Browser-side abort is the only telemetry
  consumer; richer audit lives in the existing token usage table.
* Uses ``asyncio.Lock`` for the read-modify-write pair so an aborted
  request that lands in the same tick as a fresh start can't see a
  stale snapshot. The lock is held for microseconds — no contention
  concern.
"""

from __future__ import annotations

import asyncio


class ActiveRequestCounter:
    """In-process counter for chat-playground streams.

    Each ``enter()`` returns a token that must be passed to ``exit()``;
    the token is opaque (just an int) but lets future debugging surface
    request IDs without changing the lock surface. ``count()`` is a
    synchronous read of the atomic int — safe to call from any context.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._count = 0
        self._next_id = 0

    async def enter(self) -> int:
        async with self._lock:
            self._count += 1
            self._next_id += 1
            return self._next_id

    async def exit(self, _token: int) -> None:
        async with self._lock:
            # Floor at 0 — a double-decrement would otherwise yield a
            # negative count which is misleading in the diagnostic
            # endpoint. The token is currently unused; keeping it in the
            # signature so we can switch to a dict-based registry later
            # without a wire-format change.
            if self._count > 0:
                self._count -= 1

    def count(self) -> int:
        # Read is racy-but-fine: the snapshot may be one off from the
        # canonical value if a stream is mid-enter/exit, but the
        # diagnostic only needs to converge to 0 after all activity
        # stops — which the asyncio.Lock guarantees.
        return self._count
