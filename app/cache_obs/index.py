# app/cache_obs/index.py
"""Which prefixes were recently processed, on which replica, for whom.

One bounded LRU keyed by (kind, model, epoch, scope, digest). Chunk entries
hold the replica ranks that processed that exact prefix as an int bitmask (bit
0 = None, i.e. single replica / unrouted; bit r+1 = rank r) -- a set per entry
cost ~216 bytes, more than the rest of the entry; message entries just mark a
message as seen. A routed request's fleet best leaves bit 0 out (record.py).
Scope "*" is everyone (operator lens); a token's own scope is a digest of its id (own lens).

``match`` runs BEFORE forward so a request never matches itself; ``observe``
runs after the engine processed the prompt. Two methods are the whole contract,
so the store can later become shared (spec §6).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.cache_obs.canonical import CHUNK_BYTES, Chain, element_of_byte

_ALL = b"*"


@dataclass(frozen=True)
class Match:
    by_rank: dict[int | None, int]
    best_bytes: int
    total_bytes: int
    #: message index where the prefix broke while later messages were seen;
    #: -1 = the tools element; None = no divergence evidence
    diverged_at: int | None


def _bit(rank: int | None) -> int:
    return 1 if rank is None else 1 << (rank + 1)


def _ranks(mask: int) -> list[int | None]:
    out: list[int | None] = []
    b = 0
    while mask:
        if mask & 1:
            out.append(None if b == 0 else b - 1)
        mask >>= 1
        b += 1
    return out


def _scope(token_id: str | None) -> bytes:
    if token_id is None:
        return _ALL
    return hashlib.blake2b(token_id.encode("utf-8", "surrogatepass"), digest_size=16).digest()


class PrefixIndex:
    def __init__(self, max_entries: int = 200_000) -> None:
        self._max = max_entries
        self._d: OrderedDict[tuple[Any, ...], int | None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._d)

    def _put(self, key: tuple[Any, ...], rank: int | None, *, chunk: bool) -> None:
        if chunk:
            self._d[key] = (self._d.get(key) or 0) | _bit(rank)
        else:
            self._d[key] = None
        self._d.move_to_end(key)

    def observe(
        self, model_id: str, epoch: str, rank: int | None, chain: Chain, token_id: str | None
    ) -> None:
        scopes = (_ALL,) if token_id is None else (_ALL, _scope(token_id))
        for sc in scopes:
            for k in chain.chunks:
                self._put(("c", model_id, epoch, sc, k), rank, chunk=True)
            for h in chain.message_hashes:
                self._put(("m", model_id, epoch, sc, h), None, chunk=False)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def match(
        self, model_id: str, epoch: str, chain: Chain, *, token_id: str | None = None
    ) -> Match:
        sc = _scope(token_id)
        lengths: dict[int | None, int] = {}
        active: int | None = None
        n = len(chain.chunks)
        for i, k in enumerate(chain.chunks):
            mask = self._d.get(("c", model_id, epoch, sc, k))
            if not mask:
                break
            if active is None:
                active = mask
            for r in _ranks(active & ~mask):
                lengths[r] = i
            active &= mask
            if not active:
                break
        else:
            i = n
        if active:
            for r in _ranks(active):
                lengths[r] = n if i == n else i
        by_rank = {r: min(c * CHUNK_BYTES, chain.total_bytes) for r, c in lengths.items() if c}
        best = max(by_rank.values(), default=0)
        return Match(
            by_rank, best, chain.total_bytes, self._diverged(model_id, epoch, sc, chain, best)
        )

    def _diverged(
        self, model_id: str, epoch: str, sc: bytes, chain: Chain, best: int
    ) -> int | None:
        if best >= chain.total_bytes or not chain.message_hashes:
            return None
        elem = element_of_byte(chain, best)
        k = elem - chain.msg_offset  # -1 when the break is inside tools
        # A continuation leaves the break-point message intact in the index;
        # a real rewrite changes it. Chunk flooring can land the break inside
        # an earlier, unchanged message.
        if (
            0 <= k < len(chain.message_hashes)
            and ("m", model_id, epoch, sc, chain.message_hashes[k]) in self._d
        ):
            return None
        later = chain.message_hashes[k + 1 :]
        if len(later) < 2:
            return None
        seen = sum(1 for h in later if ("m", model_id, epoch, sc, h) in self._d)
        return k if seen >= 2 and seen * 2 >= len(later) else None
