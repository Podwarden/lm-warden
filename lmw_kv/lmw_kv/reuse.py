"""Reuse counters per position, stage 1 (counters design 3.1 to 3.4, 6.1).

Pure bookkeeping: no I/O, no lock of its own (the store calls every method under its
condition variable). Every hook is O(1) per position (the holder sketch scans at most
``HOLDER_TOP`` entries). The counters are inert (N1): nothing that picks a victim, a
refusal or an admission reads them. Only ``report`` and ``top`` read them, for ``stats``
and ``top_reused``.

- A **position** is ``(model, pos)``; its counters (``PosCounters``) live from the first
  commit of any member (``saved``) until the position leaves both tiers (``left`` with
  ``gone``, or ``forget``).
- **Loads and reloads** are counted once per request tag (the last ``TAGS_MAX`` tags). A
  load without a tag counts toward ``loads`` only, never toward ``reloads``.
- **Two views.** The charged view is the tenant paying for the position now (Plan 2's
  "last user pays") and counts every reload. The saver view is the tenant that saved it
  and counts only its own reloads (``saver_reloads``): "reused" there means
  ``saver_reloads >= 1``, so another key's reload of a sharing key's position never moves
  the saver's figures (no new oracle).
- **Tallies** since the store started: the histogram of reloads when a position is
  evicted from a tier (both views), disk bytes written (to the tenant of the member that
  landed) and read (to the tenant it was read for), disk bytes the saver wrote for
  positions that left NVMe never reused by it, and regret (a save of a position that
  left the store less than ``REGRET_WINDOW_S`` ago). The store-wide totals count every
  byte.
- **Holder top list.** Per tenant, a Space-Saving sketch of ``HOLDER_TOP`` positions by
  that tenant's own reloads.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

HALF_LIFE_S = 21600.0
TAGS_MAX = 10_000
REGRET_RING = 16_384
REGRET_WINDOW_S = 86_400.0
HOLDER_TOP = 64
TENANTS_SEEN_MAX = 4
TOP_MAX = 50
REUSE_PERSIST_S = 30
REUSE_NAME = "reuse.db"

HIST_BUCKETS = ("0", "1", "2-3", "4-7", "8+")
LABEL_PREFIX = b"lmw-reuse-label\x00"

Tier = Literal["ram", "nvme"]
_TIERS: tuple[Tier, ...] = ("ram", "nvme")
_Pos = tuple[bytes, bytes]  # (model, position key)
_Pair = tuple[bytes, bytes]  # (model, tenant)
_LOADED, _RELOADED = 1, 2  # what a tag already counted for a position


@dataclass
class PosCounters:
    first_saved: float
    saver: bytes = field(repr=False)
    depth: int
    loads: int = 0
    reloads: int = 0
    saver_reloads: int = 0
    ram_reloads: int = 0
    nvme_reloads: int = 0
    last_reload: float = 0.0
    d: float = 1.0
    t: float = 0.0
    tenants_seen: tuple[bytes, ...] = field(default=(), repr=False)
    many: bool = False
    nvme_written: int = 0


def decayed(c: PosCounters, now: float, half_life_s: float = HALF_LIFE_S) -> float:
    """The decayed reload count at ``now`` (counters design 3.5)."""
    return c.d * 2 ** (-(now - c.t) / half_life_s)


def label(label_key: bytes, view_tenant: bytes, model: bytes, pos: bytes) -> str:
    """12 hex characters naming a position for one viewer; never the position key."""
    mac = hmac.new(label_key, LABEL_PREFIX + view_tenant + model + pos, hashlib.sha256)
    return mac.digest()[:6].hex()


def bucket(reloads: int) -> str:
    """The histogram bucket of a reload count."""
    if reloads <= 0:
        return "0"
    if reloads == 1:
        return "1"
    if reloads <= 3:
        return "2-3"
    if reloads <= 7:
        return "4-7"
    return "8+"


def _hist() -> dict[str, dict[str, dict[str, int]]]:
    return {t: {b: {"positions": 0, "bytes": 0} for b in HIST_BUCKETS} for t in _TIERS}


def empty_report() -> dict[str, Any]:
    """One ``R`` row with every figure at zero (the shape of every ``reuse`` map)."""
    return {
        "charged": {
            "positions": {"ram": 0, "nvme": 0},
            "positions_reloaded": {"ram": 0, "nvme": 0},
            "reloads": 0,
            "ram_reloads": 0,
            "nvme_reloads": 0,
            "evicted_hist": _hist(),
        },
        "saver": {
            "positions": 0,
            "positions_reloaded": 0,
            "saver_reloads": 0,
            "evicted_hist": _hist(),
            "nvme_bytes_written": 0,
            "nvme_bytes_read": 0,
            "wasted_nvme_bytes": 0,
            "regret": 0,
            "admission_skipped": 0,
            "ghost_admitted": 0,
        },
    }


def _add(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    """Sum ``src`` into ``dst``, nested maps of ints alike."""
    for k, v in src.items():
        if isinstance(v, Mapping):
            _add(dst[k], v)
        else:
            dst[k] += v


@dataclass
class _SaverTally:
    hist: dict[str, dict[str, dict[str, int]]] = field(default_factory=_hist)
    nvme_bytes_written: int = 0
    nvme_bytes_read: int = 0
    wasted_nvme_bytes: int = 0
    regret: int = 0


class ReuseBook:
    """The counters of every stored position, and the tallies since start."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.rows: dict[_Pos, PosCounters] = {}
        self.persist_degraded = False
        # Persistence (``reuse.db``): positions changed or forgotten since the last write.
        # Kept only while a file is attached, so a RAM-only store never accumulates them.
        self._dirty: set[_Pos] = set()
        self._tracking = False
        # tag -> position -> what this tag already counted (_LOADED | _RELOADED)
        self._tags: OrderedDict[str, dict[_Pos, int]] = OrderedDict()
        self._ring: OrderedDict[_Pos, float] = OrderedDict()  # left the store, when
        self._sketch: dict[bytes, dict[_Pos, int]] = {}  # tenant -> position -> own reloads
        self._sketch_of: dict[_Pos, set[bytes]] = {}  # position -> tenants sketching it
        self._charged_hist: dict[_Pair, dict[str, dict[str, dict[str, int]]]] = {}
        self._saver: dict[_Pair, _SaverTally] = {}
        self._written = 0
        self._read = 0

    # ---- hooks (the store's lock held; O(1) each) -------------------------------

    def saved(self, pos: _Pos, saver: bytes, depth: int, now: float) -> None:
        """A member of ``pos`` was committed: the counters start with the first one."""
        c = self.rows.get(pos)
        if c is not None:
            if c.depth < 0 <= depth:
                c.depth = depth
                self._touch(pos)
            return
        self.rows[pos] = PosCounters(first_saved=now, saver=saver, depth=depth, d=1.0, t=now)
        self._touch(pos)
        gone_at = self._ring.pop(pos, None)
        if gone_at is not None and now - gone_at < REGRET_WINDOW_S:
            self._saver_tally((pos[0], saver)).regret += 1

    def loaded(self, pos: _Pos, tag: str) -> None:
        """``begin_load`` returned ``pos``: one load per tag, or per call without one."""
        c = self.rows.get(pos)
        if c is None:
            return
        if not tag or self._mark(tag, pos, _LOADED):
            c.loads += 1
            self._touch(pos)

    def reloaded(self, pos: _Pos, tag: str, tenant: bytes, tier: Tier, now: float) -> None:
        """The adapter used a load of ``pos`` (``end_load`` with ``used``), once per tag."""
        c = self.rows.get(pos)
        if c is None or not tag or not self._mark(tag, pos, _RELOADED):
            return
        c.reloads += 1
        self._touch(pos)
        if tier == "ram":
            c.ram_reloads += 1
        else:
            c.nvme_reloads += 1
        c.d = decayed(c, now, HALF_LIFE_S) + 1.0
        c.t = now
        c.last_reload = now
        if tenant == c.saver:
            c.saver_reloads += 1
        if tenant not in c.tenants_seen:
            if len(c.tenants_seen) < TENANTS_SEEN_MAX:
                c.tenants_seen = (*c.tenants_seen, tenant)
            else:
                c.many = True
        self._sketch_add(tenant, pos)

    def nvme_written(self, pos: _Pos, nbytes: int, tenant: bytes) -> None:
        """Record bytes of a member of ``pos`` landed on NVMe, that member's ``tenant``
        paying for them. The position's own count (the saver's wasted bytes when it
        leaves NVMe) grows only for the saver's members, so another key filling in a
        member never moves the saver's figures."""
        self._written += nbytes
        c = self.rows.get(pos)
        if c is not None:
            if tenant == c.saver:
                c.nvme_written += nbytes
                self._touch(pos)
            self._saver_tally((pos[0], tenant)).nvme_bytes_written += nbytes

    def nvme_read(self, pos: _Pos, nbytes: int, tenant: bytes) -> None:
        """Record bytes of a member of ``pos`` were promoted from NVMe for ``tenant``."""
        self._read += nbytes
        if pos in self.rows:
            self._saver_tally((pos[0], tenant)).nvme_bytes_read += nbytes

    def left(
        self,
        pos: _Pos,
        tier: Tier,
        charged: bytes,
        *,
        evicted: bool,
        gone: bool,
        now: float,
        nbytes: int = 0,
    ) -> None:
        """``pos`` has no member left in ``tier``.

        ``evicted``: by eviction or reclaim (the histograms count it, ``nbytes`` being
        the position's bytes in that tier). ``gone``: it is in neither tier now, so its
        counters go and the regret ring remembers it.
        """
        c = self.rows.get(pos)
        if c is not None:
            if evicted:
                h = self._charged_hist.get((pos[0], charged))
                if h is None:
                    h = self._charged_hist[(pos[0], charged)] = _hist()
                saver_hist = self._saver_tally((pos[0], c.saver)).hist
                for hist, n in ((h, c.reloads), (saver_hist, c.saver_reloads)):
                    cell = hist[tier][bucket(n)]
                    cell["positions"] += 1
                    cell["bytes"] += nbytes
            if tier == "nvme":
                if c.saver_reloads <= 0:
                    self._saver_tally((pos[0], c.saver)).wasted_nvme_bytes += c.nvme_written
                c.nvme_written = 0
                self._touch(pos)
        if gone:
            self._ring[pos] = now
            self._ring.move_to_end(pos)
            while len(self._ring) > REGRET_RING:
                self._ring.popitem(last=False)
            self.forget(pos)

    def forget(self, pos: _Pos) -> None:
        """Drop ``pos``'s counters with no tally (it left without an eviction)."""
        self.rows.pop(pos, None)
        self._touch(pos)
        for tenant in self._sketch_of.pop(pos, ()):
            sk = self._sketch.get(tenant)
            if sk is not None:
                sk.pop(pos, None)
                if not sk:
                    del self._sketch[tenant]

    # ---- persistence (the store's lock held) ---------------------------------------

    def load(self, rows: Iterable[tuple[_Pos, PosCounters]]) -> None:
        """Install rows read from ``reuse.db`` and start tracking changes.

        A position already counted in memory keeps its live counters.
        """
        for pos, c in rows:
            self.rows.setdefault(pos, c)
        self._tracking = True

    def untrack(self) -> None:
        """No file is attached: stop remembering changes."""
        self._tracking = False
        self._dirty.clear()

    def dirty(self) -> tuple[list[tuple[_Pos, PosCounters]], list[_Pos]]:
        """Copies of the changed rows and the forgotten positions; clears the marks.

        A write that fails hands them back through ``redirty``.
        """
        ups: list[tuple[_Pos, PosCounters]] = []
        dels: list[_Pos] = []
        for pos in self._dirty:
            c = self.rows.get(pos)
            if c is None:
                dels.append(pos)
            else:
                ups.append((pos, replace(c)))
        self._dirty.clear()
        return ups, dels

    def redirty(self, upserts: Iterable[tuple[_Pos, PosCounters]], deletes: Iterable[_Pos]) -> None:
        """Mark again what a failed write did not store."""
        if self._tracking:
            self._dirty.update(p for p, _ in upserts)
            self._dirty.update(deletes)

    def seed(self, pos: _Pos, counters: PosCounters) -> None:
        """Start ``pos`` with ``counters`` (a position on NVMe that has no row)."""
        self.rows[pos] = counters
        self._touch(pos)

    def mark(self, pos: _Pos) -> None:
        """Mark ``pos`` changed, so the next write brings the file in line."""
        self._touch(pos)

    def _touch(self, pos: _Pos) -> None:
        if self._tracking:
            self._dirty.add(pos)

    # ---- reports (the store's lock held; O(positions)) --------------------------

    def report(
        self,
        live: Mapping[_Pos, tuple[bool, bool]],
        dead: Callable[[_Pos], bool],
        charged: Callable[[_Pos], bytes],
    ) -> dict[str, Any]:
        """``stats.reuse`` and the ``reuse`` map of each tenant and (model, tenant).

        ``live`` gives (in RAM, on NVMe) of every stored position; a dead one (retired
        scope, unregistered namespace) appears nowhere.
        """
        pairs: dict[_Pair, dict[str, Any]] = {}

        def row(pair: _Pair) -> dict[str, Any]:
            r = pairs.get(pair)
            if r is None:
                r = pairs[pair] = empty_report()
            return r

        for pos, c in self.rows.items():
            where = live.get(pos)
            if where is None or not any(where) or dead(pos):
                continue
            ch = row((pos[0], charged(pos)))["charged"]
            for tier, on in zip(_TIERS, where, strict=True):
                if on:
                    ch["positions"][tier] += 1
                    if c.reloads >= 1:
                        ch["positions_reloaded"][tier] += 1
            ch["reloads"] += c.reloads
            ch["ram_reloads"] += c.ram_reloads
            ch["nvme_reloads"] += c.nvme_reloads
            sv = row((pos[0], c.saver))["saver"]
            sv["positions"] += 1
            if c.saver_reloads >= 1:
                sv["positions_reloaded"] += 1
            sv["saver_reloads"] += c.saver_reloads
        for pair, h in self._charged_hist.items():
            _add(row(pair)["charged"]["evicted_hist"], h)
        for pair, st in self._saver.items():
            sv = row(pair)["saver"]
            _add(sv["evicted_hist"], st.hist)
            sv["nvme_bytes_written"] += st.nvme_bytes_written
            sv["nvme_bytes_read"] += st.nvme_bytes_read
            sv["wasted_nvme_bytes"] += st.wasted_nvme_bytes
            sv["regret"] += st.regret
        by_tenant: dict[str, dict[str, Any]] = {}
        for (_, tenant), r in pairs.items():
            t = by_tenant.get(tenant.hex())
            if t is None:
                t = by_tenant[tenant.hex()] = empty_report()
            _add(t, r)
        return {
            "store": {
                "reuse_persist_degraded": self.persist_degraded,
                "half_life_s": HALF_LIFE_S,
                "order": "lru",
                "nvme_bytes_written": self._written,
                "nvme_bytes_read": self._read,
            },
            "by_tenant": by_tenant,
            "by_model_tenant": pairs,
        }

    def top(
        self,
        model: bytes | None,
        tenant: bytes | None,
        view: Literal["admin", "holder"],
        n: int,
        label_key: bytes,
        live: Mapping[_Pos, tuple[bool, bool]],
        dead: Callable[[_Pos], bool],
        charged: Callable[[_Pos], bytes],
    ) -> list[dict[str, Any]]:
        """The most reloaded stored positions, at most ``n`` rows, never a position key.

        ``holder``: ``tenant``'s own sketch, ranked by its own reloads. ``admin``: every
        position reloaded at least once (of ``tenant``'s charge when given), by reloads.
        """
        now = self._clock()
        cands: Iterable[tuple[_Pos, int]]
        if view == "holder":
            if tenant is None:
                raise ValueError("a holder view needs a tenant")
            cands = list(self._sketch.get(tenant, {}).items())
        elif view == "admin":
            cands = [(pos, c.reloads) for pos, c in self.rows.items() if c.reloads >= 1]
        else:
            raise ValueError(f"unknown view {view!r}")
        ranked: list[tuple[int, float, _Pos, tuple[bool, bool]]] = []
        for pos, count in cands:
            c = self.rows.get(pos)
            where = live.get(pos)
            if c is None or where is None or not any(where):
                continue
            if model is not None and pos[0] != model:
                continue
            if dead(pos):
                continue
            if view == "admin" and tenant is not None and charged(pos) != tenant:
                continue
            ranked.append((count, decayed(c, now, HALF_LIFE_S), pos, where))
        best = heapq.nlargest(n, ranked, key=lambda e: (e[0], e[1]))
        out: list[dict[str, Any]] = []
        for count, dec, pos, where in best:
            c = self.rows[pos]
            if view == "holder":
                assert tenant is not None
                out.append({
                    "label": label(label_key, tenant, pos[0], pos[1]),
                    "model": pos[0].hex(), "depth": c.depth, "reloads": count,
                })  # fmt: skip
                continue
            out.append({
                "label": label(label_key, b"", pos[0], pos[1]),
                "model": pos[0].hex(), "depth": c.depth, "reloads": count,
                "decayed": dec, "last_reload": c.last_reload, "first_saved": c.first_saved,
                "tiers": [t for t, on in zip(_TIERS, where, strict=True) if on],
                "tenants_seen": len(c.tenants_seen), "many": c.many,
                "charged": charged(pos).hex(),
            })  # fmt: skip
        return out

    # ---- internals ----------------------------------------------------------------

    def _mark(self, tag: str, pos: _Pos, bit: int) -> bool:
        """Mark ``bit`` for (``tag``, ``pos``); False when it was already marked."""
        seen = self._tags.get(tag)
        if seen is None:
            seen = self._tags[tag] = {}
            while len(self._tags) > TAGS_MAX:
                self._tags.popitem(last=False)
        else:
            self._tags.move_to_end(tag)
        got = seen.get(pos, 0)
        if got & bit:
            return False
        seen[pos] = got | bit
        return True

    def _sketch_add(self, tenant: bytes, pos: _Pos) -> None:
        """Space-Saving: one more reload of ``pos`` by ``tenant`` (``HOLDER_TOP`` entries)."""
        sk = self._sketch.get(tenant)
        if sk is None:
            sk = self._sketch[tenant] = {}
        if pos in sk:
            sk[pos] += 1
            return
        count = 1
        if len(sk) >= HOLDER_TOP:
            out = min(sk, key=sk.__getitem__)
            count += sk.pop(out)
            owners = self._sketch_of.get(out)
            if owners is not None:
                owners.discard(tenant)
                if not owners:
                    del self._sketch_of[out]
        sk[pos] = count
        self._sketch_of.setdefault(pos, set()).add(tenant)

    def _saver_tally(self, pair: _Pair) -> _SaverTally:
        st = self._saver.get(pair)
        if st is None:
            st = self._saver[pair] = _SaverTally()
        return st
