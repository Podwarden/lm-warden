"""Bearer token rows cached for a few seconds (issue #279 stage 3).

Every /v1 request used to SELECT its token row. Rows are now cached by secret
hash for TOKEN_CACHE_TTL_S. Any change to api_tokens through TokenRepo calls
invalidate_all(), so revoke / pause / rotate / delete apply to the very next
request in this process; the TTL bounds staleness for anything else (expiry is
evaluated per request from the cached row, so it is never late). Unknown
secrets are NOT cached: the unknown-secret throttle must keep seeing them.

A caller snapshots generation() BEFORE it reads the row and passes it to put():
an entry whose read predates an invalidation is dead on arrival, so a revoke
racing an in-flight lookup always wins. The generation is per process: other
worker processes are bounded by the TTL only.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

TOKEN_CACHE_TTL_S = 5.0
MISS = object()
_generation = 0


def generation() -> int:
    return _generation


def invalidate_all() -> None:
    global _generation
    _generation += 1


class TokenCache:
    def __init__(
        self,
        ttl_s: float = TOKEN_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._d: dict[str, tuple[object, float, int]] = {}

    def get(self, token_hash: str) -> object:
        hit = self._d.get(token_hash)
        if hit is None:
            return MISS
        row, at, gen = hit
        if gen != _generation or self._clock() - at > self._ttl:
            self._d.pop(token_hash, None)
            return MISS
        # A copy, so a caller can never mutate the shared cached row.
        if dataclasses.is_dataclass(row) and not isinstance(row, type):
            return dataclasses.replace(row)
        return row

    def put(self, token_hash: str, row: object, gen: int | None = None) -> None:
        if row is None:
            return
        if len(self._d) >= 10_000:
            self._d.clear()
        self._d[token_hash] = (row, self._clock(), _generation if gen is None else gen)
