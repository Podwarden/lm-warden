"""The block store: a RAM tier, an NVMe tier, write-through, promotion, eviction.

Layout and rules:

- **RAM slots.** One slot holds one whole block: every KV cache group's payload,
  packed at offsets inside the slot, each group rounded up to ``mmap.PAGESIZE``,
  so every offset handed to a connector is page-aligned (``cudaHostRegister``).
  The pool file is created and every page committed when the store starts, before
  the daemon answers anyone, so ``register_ns`` never allocates (allocating 16 GiB
  takes seconds, longer than a connector waits). It is sized to the smallest of the
  request, ``/dev/shm`` free space less its reserve, and available memory less a
  safety margin (``ram.grant``). The first namespace to register sets the slot size.
  A new store first unlinks any pool file an earlier store left behind,
  so each store has its own inode and never resizes pages an old engine still maps. A namespace whose blocks need a larger slot re-slabs the pool when
  nothing is pinned, saving or queued (RAM copies are dropped; NVMe copies stay;
  the file keeps its size, the page-rounded budget, so a connector's mapping of it
  stays valid; the tail past the last whole slot is unused),
  otherwise it runs without RAM until it registers again. A smaller namespace
  shares the larger slots.
- **NVMe.** Only for a namespace registered with ``nvme=True`` (the default); a
  namespace registered with ``nvme=False`` never writes to NVMe and never reads from
  it, even when its blocks are already there. ``<nvme_dir>/<ns id hex>/`` holds that namespace's segment files and
  manifest; ``<nvme_dir>/data.key`` is the data key. Only one background thread
  writes segments (``SegmentWriter`` has no lock), from a bounded queue.
- **NVMe cap.** ``nvme_cap_bytes`` caps the bytes of the segment files on disk, not
  the live payload: an evicted block's bytes stay in its segment until the whole
  file is deleted. Over the cap, whole segments are reclaimed, oldest first. Every
  block with a record in the segment leaves the NVMe index, and so does every
  descendant that would be left without a present parent (prefix closure; a RAM
  copy of such a descendant is dropped too). A segment whose blocks are busy in RAM
  (pinned, saving, promoting or queued) is skipped for the next oldest.
- **Presence** is RAM (committed) or NVMe (every group recorded). Presence stays
  prefix-closed: a block whose removal changes presence is removed only as a leaf
  (no child present in either tier). Dropping the RAM copy of a block whose NVMe
  write landed does not change presence, so those copies go in plain LRU order.
  A RAM block is not evictable while it is being saved, promoted, pinned by a
  load, or queued for its NVMe write (October spec 7.7).
- **Loads** return a ``LoadHandle``; ``end_load(handle.id)`` releases exactly the
  pins that load took, never another request's.
- **Failure.** Any NVMe error (ENOSPC, a missing or read-only path, a SQLite
  error) switches the NVMe tier off for the life of the store, with one warning;
  RAM keeps working. Too little shared memory or free memory shrinks the pool or
  means no RAM tier, logged once.
  A damaged NVMe record is a miss and is dropped from the manifest.

- **Owners (I3).** A namespace keeps every owner that registered it, with whether that
  owner wants NVMe. NVMe is on for the namespace only if every owner wants it; the legacy
  owner ``""`` keeps step 1's last-registration-wins for itself. ``release_owner`` drops
  an owner everywhere; a namespace left with none is unregistered (``lookup`` 0, no
  loads, its blocks class dead) until it registers again.
- **Attribution.** Every block carries a tenant (who pays), a scope tag (empty for a
  sharing key), the model of its namespace (fixed at the first registration) and a
  position key (``pos``; its own key when none is given, and for step 1 rows). The
  position index ``(model, pos) -> {(ns, key)}`` covers both tiers. Usage is bytes per
  ``(model, tenant)``: a RAM block counts its pool slot, an NVMe block its record bytes on
  disk (what each tier's cap limits). A load with a tenant takes over the charge of every
  loaded position in both tiers ("last user pays").
- **Retired scopes.** A block whose scope tag is retired is never found, loaded or saved
  again, and is class dead. The set lives in memory and, while NVMe is on, in
  ``<nvme_dir>/retired.db``. If that file cannot be read, every isolated block (non-empty
  scope tag) is refused until it can; sharing blocks are unaffected.
- **Policy** (``set_policy``, ``lmw_kv.fair``). ``lru``, the default, is step 1's victim
  order, exactly (``_evict_one``, ``_reclaim_oldest_segment``); attribution only counts.
  Under ``fair`` a RAM victim is a whole position (every RAM member valid, unpinned, not
  queued, and durable or a leaf; all members go in one step), chosen in this order: class
  dead (retired scope or unregistered namespace), class idle (unused longer than
  ``idle_ram_s``), then the over-share ``(model, tenant)`` pairs (``fair.over_order``),
  each pair's least recently used position, and only when none of those has an evictable
  position, the global LRU. A tenant at its RAM cap evicts its own oldest position before
  any allocation for it, or the allocation fails. The leaf test reads an in-memory child
  count, never SQLite.
- **Fair NVMe** (spec 5.1 erratum: the unit is a whole segment). Under ``fair`` the store
  chooses which segment to reclaim by ``_segment_score``: most dead bytes first (retired
  scope, a namespace with no owner or not registered since start, and the bytes of
  records already dropped), then most idle bytes (unused longer than ``idle_nvme_s``),
  then most over-share bytes (weighted by how far over each pair is), then the oldest.
  The writer's current segments come last. Scores are cached in a heap; each reclaim
  rescores only the segments whose maps changed, a rotating slice and the best few, and
  tries at most ``RECLAIM_TRIES`` plans, so its lock-held cost does not grow with the
  number of segments; the lock is let go between reclaims. A tenant at its
  ``max_nvme_pct`` cap reclaims the segment where its share of the live bytes is
  largest; if even there others hold more live bytes than it does, its write is refused
  (stays RAM only). Dropping a block also drops its position's members in the other
  namespaces from their manifests. Scoring reads per-segment byte maps kept in memory;
  SQLite runs only in ``_plan_reclaim`` (indexed lookups). A namespace not registered
  since the store opened becomes dead after ``UNREGISTERED_GRACE_S``.

One condition variable guards every index mutation; slot copies and segment I/O
run outside it, and so does every whole-table SQLite read (opening a manifest).
"""

from __future__ import annotations

import bisect
import heapq
import itertools
import logging
import mmap
import os
import queue
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lmw_kv import ram
from lmw_kv.fair import DEFAULT_OWNER_WEIGHT, Policy, over_order, parts, shares
from lmw_kv.format import (
    HEADER_LEN,
    SegmentWriter,
    _existing_segments,
    _seg_path,
    load_or_create_data_key,
    read_record,
    record_size,
)
from lmw_kv.keys import Namespace, namespace_id
from lmw_kv.manifest import Entry, Manifest, PosRow
from lmw_kv.ram import RamPool, create_pool_file, discard_stale, fit_memory, fit_shm
from lmw_kv.retired import RETIRED_NAME, RetiredScopes, RetiredStoreError

log = logging.getLogger(__name__)

GIB = 1 << 30
SEGMENT_BYTES = 64 << 20
WRITE_QUEUE_MAX = 1024
HITS_MAX = 10_000
MANIFEST_NAME = "manifest.db"
#: Longest a load waits for another load's promotion of the same block; then a miss.
PROMOTE_WAIT_S = 10.0
#: A namespace on NVMe that has not registered since the store opened is class dead only
#: after this long (models load again after a restart).
UNREGISTERED_GRACE_S = 1800.0
#: Fair NVMe choice, per reclaim: segments rescored from a rotating cursor, the cached
#: best ones rescored exactly, and the most ``_plan_reclaim`` attempts before refusing.
RANK_SLICE = 512
RANK_TOP = 8
RANK_DRAIN = 2048  # stale-heap entries moved to the new heap per reclaim
RECLAIM_TRIES = 8
#: Segments of a capped tenant looked at per reclaim (its oldest first).
CAPPED_SCAN = 1024
#: (model, tenant) pairs whose eviction counts ``stats`` keeps, the most recently counted
#: first: the reply stays bounded for the life of the store (``by_tenant`` keeps all).
PAIR_STATS_MAX = 4096
_NVME_ERRORS = (OSError, sqlite3.Error, RuntimeError)

_SAVING = "saving"
_VALID = "valid"
_PROMOTING = "promoting"

_Ik = tuple[bytes, bytes]  # (namespace id, block key)
_Pos = tuple[bytes, bytes]  # (model, position key)
_Acct = tuple[bytes, bytes, bytes, bytes]  # (namespace id, model, tenant, scope tag)
Tier = Literal["ram", "nvme"]
_TIERS: tuple[Tier, ...] = ("ram", "nvme")
_Pair = tuple[bytes, bytes]  # (model, tenant)
_Victim = list[tuple[_Ik, "_Block"]]  # the RAM members of one evictable position
MODEL_LEN = 16


def _page_round(n: int) -> int:
    return -(-n // mmap.PAGESIZE) * mmap.PAGESIZE


@dataclass
class StoreConfig:
    ram_path: Path
    ram_bytes: int
    nvme_dir: Path | None
    nvme_cap_bytes: int
    min_save_tokens: int = 2048
    #: Wall clock for ages and idle checks (the manifest's ``last_used`` is wall time too).
    clock: Callable[[], float] = time.time
    #: Engine workers whose host memory the pool leaves free (``ram.fit_memory``).
    engine_workers: int = 0


@dataclass
class _Block:
    slot: int
    parent: bytes
    scope: bytes = field(repr=False)
    state: str
    pins: int = 0
    queued: bool = False  # waiting for its NVMe write
    durable: bool = False  # an NVMe copy exists
    model: bytes = b""
    pos: bytes = b""  # the position key, never empty once stored
    tenant: bytes = field(default=b"", repr=False)
    scope_tag: bytes = field(default=b"", repr=False)
    last_used: float = 0.0
    charged: int = 0  # bytes counted in RAM usage (0 until committed)


@dataclass
class _NvRec:
    """One block on NVMe, kept in memory (rebuilt from the manifest at open)."""

    model: bytes
    pos: bytes
    parent: bytes
    tenant: bytes = field(repr=False)
    scope_tag: bytes = field(repr=False)
    nbytes: int  # record bytes on disk, every group
    segments: dict[int, int]  # segment -> this block's record bytes in it
    last_used: float


@dataclass
class _Opened:
    """A manifest read outside the lock, ready to install."""

    ns_id: bytes
    manifest: Manifest
    segs: list[tuple[int, float, int]]  # (segment, mtime, size)
    rows: list[PosRow]


@dataclass
class _ScoreCtx:
    """Shared by every segment's score in one fair NVMe choice."""

    ratios: dict[tuple[bytes, bytes], float]  # over-share pair -> usage / share
    cutoff: float  # last used before this is idle
    dead_ns: dict[bytes, bool]
    dead_tag: dict[bytes, bool] = field(repr=False)


@dataclass
class _TenantStats:
    ram_evictions: int = 0
    nvme_evictions: int = 0
    evicted_by: dict[bytes, int] = field(default_factory=dict)
    ram_tokens: int = 0
    nvme_tokens: int = 0


@dataclass
class LoadHandle:
    """One ``begin_load``: its id for ``end_load`` and the slots it pinned.

    ``nvme_blocks`` counts the returned blocks this call promoted from NVMe.
    """

    id: int
    slots: list[dict[str, int]]
    nvme_blocks: int = 0


@dataclass
class _Ns:
    ns_id: bytes
    groups: dict[str, int]
    offsets: dict[str, int]  # group -> offset inside the slot
    slot_bytes: int
    ram_ok: bool
    nvme: bool = True  # write through to NVMe and load from it
    owners: dict[str, bool] = field(default_factory=dict)  # owner -> wants NVMe
    model: bytes = b""  # fixed at the first registration
    role: str = "attention"
    rank: int = 0
    live: bool = True  # False once its last owner is released


class Store:
    def __init__(self, cfg: StoreConfig) -> None:
        self.cfg = cfg
        self._cond = threading.Condition()
        self._closed = False
        self._ns: dict[bytes, _Ns] = {}
        self._ram: OrderedDict[_Ik, _Block] = OrderedDict()  # least recently used first
        self._children: dict[_Ik, int] = {}  # RAM entries per parent
        self._pool: RamPool | None = None
        self._evictions = 0
        self._nvme_evictions = 0
        self._hits: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._load_ids = itertools.count(1)
        self._loads: dict[int, list[_Block]] = {}  # handle id -> the blocks it pinned
        self._ram_budget = self._fit_budget(cfg)
        self._ram_budget = self._create_pool(cfg, self._ram_budget)
        self._retired = RetiredScopes()  # memory first; attached to a file with NVMe
        self._positions: dict[Tier, dict[_Pos, set[_Ik]]] = {"ram": {}, "nvme": {}}
        self._nv: dict[_Ik, _NvRec] = {}  # every block in an open manifest
        self._seg_bytes: dict[tuple[bytes, int], dict[bytes, int]] = {}  # key -> bytes
        # Per segment, for fair scoring: bytes per (model, tenant, scope tag), and the
        # [oldest, newest] last use of its records (bounds only: never lowered or raised
        # by a drop, so a stale bound only costs an exact scan).
        self._seg_acct: dict[tuple[bytes, int], dict[tuple[bytes, bytes, bytes], int]] = {}
        self._seg_use: dict[tuple[bytes, int], list[float]] = {}
        # Fair NVMe ranking: a heap of (score key, version, segment), stale entries
        # skipped by version; segments to rescore; a ring every segment is in, walked
        # RANK_SLICE at a time so every score is refreshed now and then.
        self._rank_heap: list[tuple[tuple[Any, ...], int, tuple[bytes, int]]] = []
        # A heap too full of stale entries is drained into a new one a slice at a time
        # (popping a heap's last element keeps it a heap), never rebuilt in one go.
        self._rank_old: list[tuple[tuple[Any, ...], int, tuple[bytes, int]]] = []
        self._rank_ver: dict[tuple[bytes, int], int] = {}
        self._rank_vers = itertools.count()
        self._rank_dirty: set[tuple[bytes, int]] = set()
        self._ring: list[tuple[bytes, int]] = []
        self._ring_pos = 0
        self._rank_sig: tuple[Any, ...] = ()
        # Per tenant: its NVMe record bytes (dead ones too) and the segments holding them.
        self._tenant_nv: dict[bytes, int] = {}
        self._tenant_segs: dict[bytes, dict[tuple[bytes, int], int]] = {}
        self._capped_pos: dict[bytes, int] = {}  # where a capped tenant's next scan starts
        self._opened_at = cfg.clock()
        self._grace_done = False
        self._acct: dict[Tier, dict[_Acct, int]] = {"ram": {}, "nvme": {}}
        self._tenants: dict[bytes, _TenantStats] = {}
        # The same eviction counts per (model, tenant): ``by_model[m].tenants[t]``.
        self._pair_stats: OrderedDict[_Pair, _TenantStats] = OrderedDict()
        # The owner that registered each model last: names a model with no owner left.
        self._last_owner: dict[bytes, str] = {}
        self._evict_for: bytes | None = None  # the tenant an eviction makes room for
        self._enable_lock = threading.Lock()
        # Tenant changes of NVMe rows not yet in the manifest, (ns, key) -> tenant. Written
        # in batches outside ``_cond`` (``_persist_tenants``); one writer at a time keeps
        # the latest change last.
        self._tenant_dirty: dict[_Ik, bytes] = {}
        self._persist_lock = threading.Lock()
        self._policy = Policy()
        self._nv_kids: dict[_Ik, int] = {}  # NVMe records per (ns, parent): the leaf test
        # RAM positions for ``fair``. Candidates are positions whose every RAM member is
        # durable or has no RAM child: a superset of the evictable ones that leaves out
        # interior positions of RAM-only chains, so no victim walk scans them. Kept sorted
        # by touch order (least recently used first): one list over all, and one per
        # (model, tenant) a member is charged to.
        self._cand_all: list[tuple[int, _Pos]] = []
        self._cand_pair: dict[_Pair, list[tuple[int, _Pos]]] = {}
        self._cand_seq: dict[_Pos, int] = {}  # the touch number a candidate is listed under
        self._cand_pairs: dict[_Pos, list[_Pair]] = {}  # the pair lists it is in
        self._pos_tenants: dict[_Pos, dict[bytes, int]] = {}  # RAM members per tenant
        self._pos_used: dict[_Pos, float] = {}  # last use, wall clock (class idle)
        self._pos_seq: dict[_Pos, int] = {}  # touch order, to compare across pairs
        self._touches = itertools.count()
        self._saving: dict[bytes, int] = {}  # SAVING blocks per tenant (the RAM cap)

        self._nvme_enabled = False
        self._nvme_off_logged = False
        self._nvme_used = 0  # bytes of segment files on disk (what the cap limits)
        # (ns id, segment) -> [age, bytes on disk]; oldest (smallest age) reclaimed first
        self._segs: dict[tuple[bytes, int], list[float]] = {}
        self._manifests: dict[bytes, Manifest] = {}
        self._data_key = b""
        self._writers: dict[bytes, SegmentWriter] = {}  # writer thread only
        self._cur_seg: dict[bytes, int] = {}  # writer thread only
        self._seg_floor: dict[bytes, int] = {}  # next segment number never below this
        if cfg.nvme_dir is not None and cfg.nvme_cap_bytes > 0:
            try:
                cfg.nvme_dir.mkdir(parents=True, exist_ok=True)
                self._data_key = load_or_create_data_key(cfg.nvme_dir)
                for d in sorted(cfg.nvme_dir.iterdir()):
                    ns_id = _ns_from_dirname(d.name)
                    if ns_id is not None and (d / MANIFEST_NAME).exists():
                        self._open_manifest(ns_id)
                self._nvme_enabled = True
            except _NVME_ERRORS as exc:
                self._nvme_off(exc)
            if self._nvme_enabled:
                self._attach_retired(cfg.nvme_dir)

        self._queue: queue.Queue[_Ik | None] = queue.Queue(WRITE_QUEUE_MAX)
        self._thread = threading.Thread(target=self._writer_loop, name="lmw-kv-writer", daemon=True)
        self._thread.start()

    # ---- public API -------------------------------------------------------------

    def register_namespace(
        self,
        ns: Namespace,
        groups: dict[str, int],
        *,
        nvme: bool = True,
        owner: str = "",
        model: bytes = b"",
        role: str = "attention",
        rank: int = 0,
    ) -> bytes:
        """Register a namespace with its per-block bytes for each group; returns the ns id.

        ``nvme`` False keeps the namespace's blocks in RAM only: no write-through and no
        loads from NVMe. The namespace keeps every ``owner`` that registered it; NVMe is
        on only if every owner wants it (I3). The legacy owner ``""`` registering again
        sets its own flag again (the last one wins). ``model``, ``role`` and ``rank`` are
        fixed by the first registration; an empty ``model`` means the first 16 bytes of
        the namespace id. A later owner with another model joins as an owner only.
        """
        if not groups:
            raise ValueError("a namespace needs at least one group")
        offsets: dict[str, int] = {}
        slot_bytes = 0
        for g, n in groups.items():
            if n <= 0 or len(g.encode("utf-8")) > 255:
                raise ValueError(f"bad group {g!r}: {n} bytes")
            offsets[g] = slot_bytes
            slot_bytes += _page_round(n)
        ns_id = namespace_id(ns)
        self.flush()  # a re-slab needs the write queue drained
        with self._cond:
            known = self._ns.get(ns_id)
            if known is not None:
                if known.groups != dict(groups):
                    raise ValueError("namespace registered again with different group sizes")
                info = known
            else:
                info = _Ns(
                    ns_id, dict(groups), offsets, slot_bytes, ram_ok=False, nvme=nvme,
                    model=model or ns_id[:MODEL_LEN], role=role, rank=rank,
                )  # fmt: skip
                self._ns[ns_id] = info
            info.owners[owner] = nvme
            if owner:
                self._last_owner[info.model] = owner
            info.live = True
            late = self._set_nvme(info)
            if not info.ram_ok:
                info.ram_ok = self._fit_ram(info.slot_bytes)
        if late:
            self._open_late(ns_id)
        return ns_id

    def release_owner(self, owner: str) -> int:
        """Drop ``owner`` from every namespace; returns how many were left with no owner.

        Such a namespace is unregistered until it registers again: ``lookup`` is 0,
        ``begin_load`` is empty, ``begin_save`` refuses, and its blocks are class dead.
        """
        gone = 0
        late: list[bytes] = []
        with self._cond:
            for ns_id, info in self._ns.items():
                if owner not in info.owners:
                    continue
                del info.owners[owner]
                if not info.owners:
                    info.live = False  # its NVMe flag stays: lru treats its blocks as before
                    gone += 1
                elif self._set_nvme(info):
                    late.append(ns_id)
        for ns_id in late:
            self._open_late(ns_id)
        return gone

    def lookup(self, ns_id: bytes, keys: Sequence[bytes]) -> int:
        """Number of leading keys stored in RAM or on NVMe."""
        with self._cond:
            info = self._usable(ns_id)
            if info is None:
                return 0
            n = 0
            for key in keys:
                b = self._ram.get((ns_id, key))
                if b is not None and b.state != _SAVING:
                    if self._hidden(b.scope_tag):
                        break
                    n += 1
                elif b is None and self._nvme_present(info, key):
                    n += 1
                else:
                    break
            return n

    def begin_save(
        self,
        ns_id: bytes,
        key: bytes,
        parent: bytes,
        scope: bytes,
        *,
        tenant: bytes = b"",
        scope_tag: bytes = b"",
        pos: bytes = b"",
    ) -> dict[str, int] | None:
        """Reserve a RAM slot for a new block: group -> byte offset, or None.

        ``tenant`` pays for the block, ``scope_tag`` names its isolation scope (empty
        for a sharing key) and ``pos`` is its position key (empty means ``key``). A save
        under a retired scope, or an isolated save while the retired set is unreadable,
        is refused; so is a save of a key already stored (retired or not).
        """
        with self._cond:
            info = self._usable(ns_id)
            ik = (ns_id, key)
            if info is None or self._hidden(scope_tag):
                return None
            if ik in self._ram or self._nvme_entries(info, key) is not None:
                return None
            pk = (ns_id, parent)
            self._add_child(pk)  # the parent is no longer a leaf while this block exists
            slot = self._alloc(tenant, info.model)
            if slot is None:
                self._drop_child(pk)
                return None
            b = _Block(
                slot, parent, scope, _SAVING, model=info.model, pos=pos or key,
                tenant=tenant, scope_tag=scope_tag, last_used=self.cfg.clock(),
            )  # fmt: skip
            self._ram[ik] = b
            self._pos_add("ram", (b.model, b.pos), ik)
            self._lru_add((b.model, b.pos), tenant, b.last_used)
            self._saving[tenant] = self._saving.get(tenant, 0) + 1
            return self._offsets(info, slot)

    def commit_save(self, ns_id: bytes, key: bytes) -> None:
        """The slot holds the block: make it visible and queue its NVMe write."""
        with self._cond:
            ik = (ns_id, key)
            b = self._ram.get(ik)
            if b is None or b.state != _SAVING:
                return
            self._unsaving(b.tenant)
            b.state = _VALID
            self._ram.move_to_end(ik)
            b.last_used = self.cfg.clock()
            self._lru_touch((b.model, b.pos), b.last_used)
            self._charge_ram(ik, b)
            if not self._nvme_on(ns_id):
                return
            try:
                self._queue.put_nowait(ik)
            except queue.Full:
                log.debug("write-through queue full; block stays in RAM only")
                return
            b.queued = True

    def abort_save(self, ns_id: bytes, key: bytes) -> None:
        """Give back a slot from ``begin_save`` that will never be committed."""
        with self._cond:
            ik = (ns_id, key)
            b = self._ram.get(ik)
            if b is not None and b.state == _SAVING:
                self._remove(ik, b)

    def begin_load(self, ns_id: bytes, keys: Sequence[bytes], *, tenant: bytes = b"") -> LoadHandle:
        """Pin the leading stored keys in RAM (promoting NVMe-only ones first).

        ``slots`` holds group -> offset per pinned key, stopping at the first key
        that cannot be loaded, so it may be shorter than ``keys``. Release exactly
        these pins with ``end_load(handle.id)``. A non-empty ``tenant`` takes over the
        charge of every loaded position, in RAM and on NVMe ("last user pays").
        """
        with self._cond:
            info = self._usable(ns_id)
            handle_id = next(self._load_ids)
        handle = LoadHandle(handle_id, [])
        if info is None:
            return handle
        pinned: list[_Block] = []
        with self._cond:
            self._loads[handle_id] = pinned  # recorded as pins are taken
        try:
            for key in keys:
                # Two arguments when there is no tenant: step 1's call, exactly.
                got = self._load_one(info, key, tenant) if tenant else self._load_one(info, key)
                if got is None:
                    break
                offsets, b, promoted = got
                handle.slots.append(offsets)
                handle.nvme_blocks += promoted
                with self._cond:
                    pinned.append(b)
        except BaseException:
            self.end_load(handle_id)  # the caller never sees this handle: free its pins
            raise
        finally:
            if tenant:
                self._persist_tenants()  # outside the lock, one transaction per manifest
        if not pinned:
            with self._cond:
                self._loads.pop(handle_id, None)
        return handle

    def end_load(self, handle_id: int) -> None:
        """Release one ``begin_load``'s pins; an unknown or ended handle is a no-op."""
        with self._cond:
            pinned = self._loads.pop(handle_id, None)
            if pinned is None:
                log.debug("end_load: unknown or already ended handle %d", handle_id)
                return
            for b in pinned:
                b.pins -= 1

    def record_hits(
        self,
        tag: str,
        ram_tokens: int,
        nvme_tokens: int,
        load_s: float,
        *,
        tenant: bytes = b"",
    ) -> None:
        with self._cond:
            h = self._hits.pop(tag, None) or {"ram_tokens": 0, "nvme_tokens": 0, "load_s": 0.0}
            h["ram_tokens"] += ram_tokens
            h["nvme_tokens"] += nvme_tokens
            h["load_s"] += load_s
            self._hits[tag] = h
            while len(self._hits) > HITS_MAX:
                self._hits.popitem(last=False)
            t = self._tenant(tenant)
            t.ram_tokens += ram_tokens
            t.nvme_tokens += nvme_tokens

    def pop_hits(self, tag: str) -> dict[str, Any] | None:
        with self._cond:
            return self._hits.pop(tag, None)

    def retire_scopes(self, tags: Iterable[bytes]) -> int:
        """Retire isolation scope tags for good; returns how many were new.

        Their blocks are never found, loaded or saved again, and are class dead. The
        set is in memory at once and on disk while NVMe is on (idempotent).
        """
        return self._retired.add(list(tags))  # its own lock; never under ``_cond``

    def usage(self, tier: Tier) -> dict[tuple[bytes, bytes], int]:
        """Bytes per ``(model, tenant)`` in ``tier``, by position; dead blocks excluded."""
        with self._cond:
            return self._usage(tier)

    def set_policy(self, policy: Policy) -> None:
        """Switch the eviction policy; the next victim is chosen by it."""
        with self._cond:
            self._policy = policy
            self._rank_dirty.update(self._segs)  # shares and idle limits may have changed

    def enable_nvme(self, nvme_dir: Path) -> bool:
        """Turn the NVMe tier on after start; True when it is on.

        For a daemon that started RAM-only because another daemon held the directory's
        lock, once it gets the lock. Blocks committed before stay RAM-only. A store whose
        NVMe tier was switched off by an error stays off. The manifests are read outside
        the lock; scopes retired while RAM-only are written to the new retired file.
        """
        with self._enable_lock:
            with self._cond:
                if self._nvme_enabled or self._nvme_off_logged or self.cfg.nvme_cap_bytes <= 0:
                    return self._nvme_enabled
                wanted = [ns_id for ns_id, info in self._ns.items() if self._wants_nvme(info)]
            opened: list[_Opened] = []
            try:
                nvme_dir.mkdir(parents=True, exist_ok=True)
                self.cfg.nvme_dir = nvme_dir  # read only once the tier is on
                data_key = load_or_create_data_key(nvme_dir)
                for d in sorted(nvme_dir.iterdir()):
                    ns_id = _ns_from_dirname(d.name)
                    if ns_id is not None and (d / MANIFEST_NAME).exists():
                        opened.append(self._read_manifest(ns_id))
                have = {o.ns_id for o in opened}
                for ns_id in wanted:
                    if ns_id not in have:
                        opened.append(self._read_manifest(ns_id))
            except _NVME_ERRORS as exc:
                for o in opened:
                    o.manifest.close()
                with self._cond:
                    self._nvme_off(exc)
                return False
            # Before the tier shows as on: the daemon sees "on" as done (its lock retry).
            self._attach_retired(nvme_dir)
            with self._cond:
                self._data_key = data_key
                for o in opened:
                    self._install(o)
                self._nvme_enabled = True
                # Registered meanwhile: NVMe waits for its manifest (never a write without).
                late = [ns_id for ns_id, info in self._ns.items() if self._set_nvme(info)]
            for ns_id in late:
                self._open_late(ns_id)
            return self._nvme_enabled

    def ram_bytes(self) -> int:
        """Size of the RAM file (the budget before the first namespace).

        Lock-free and cheap: ``hello`` answers on the daemon's event loop with it, so the
        api's liveness check sees a live daemon even while every worker thread is busy
        (two plain attribute reads; the pool's size never changes once it exists).
        """
        pool = self._pool
        return pool.size_bytes if pool is not None else self._ram_budget

    def stats(self) -> dict[str, Any]:
        with self._cond:
            pool = self._pool
            ram_only = sum(
                1
                for ik, b in self._ram.items()
                if b.state == _VALID and not (b.durable and self._nvme_on(ik[0]))
            )
            nvme_blocks = len(self._nv) if self._nvme_enabled else 0  # no table scan
            out: dict[str, Any] = {
                "ram_bytes": pool.size_bytes if pool is not None else self._ram_budget,
                "ram_used": (
                    (pool.slots_total - pool.slots_free()) * pool.slot_bytes if pool else 0
                ),
                "nvme_used": self._nvme_used if self._nvme_enabled else 0,
                "nvme_enabled": self._nvme_enabled,
                "blocks": ram_only + nvme_blocks,
                "evictions": self._evictions,
                "nvme_evictions": self._nvme_evictions,
                "policy": self._policy.mode,
                # Wall time the store started: every count below is since then.
                "started_at": self._opened_at,
                "nvme_cap_bytes": self.cfg.nvme_cap_bytes if self._nvme_enabled else 0,
                # The retired set (lock order: _cond, then its own lock, as _hidden).
                "retired_scopes": len(self._retired),
                "isolation_refused": self._retired.isolation_refused,
                "retired_persist_degraded": self._retired.persist_degraded,
            }
            out.update(self._attribution_stats())
            return out

    def flush(self) -> None:
        """Wait until every queued NVMe write has landed (or been dropped)."""
        self._queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._persist_tenants()
        for w in self._writers.values():
            try:
                w.close()
            except OSError as exc:
                log.debug("closing a segment writer failed: %s", exc)
        for m in self._manifests.values():
            m.close()
        self._retired.close()
        with self._cond:
            self._ram.clear()
            self._children.clear()
            self._acct["ram"].clear()
            self._positions["ram"].clear()
            self._lru_clear()
            self._saving.clear()
            if self._pool is not None:
                self._pool.close()
                self._pool = None
            if self._ram_budget > 0:  # the file this store created at start
                self.cfg.ram_path.unlink(missing_ok=True)

    # ---- RAM tier (callers hold the lock unless noted) ---------------------------

    @staticmethod
    def _fit_budget(cfg: StoreConfig) -> int:
        """The pool size; one WARNING when it is smaller than asked (or 0)."""
        wanted = cfg.ram_bytes
        try:
            discard_stale(cfg.ram_path)  # one size, one inode per store (C1)
            shm = fit_shm(cfg.ram_path, wanted)
        except OSError as exc:
            log.warning("no RAM tier: cannot size %s: %s", cfg.ram_path, exc)
            return 0
        mem = fit_memory(wanted, engine_workers=cfg.engine_workers)
        got = ram.grant(wanted, shm, mem)
        if got >= wanted // mmap.PAGESIZE * mmap.PAGESIZE:
            return got
        if mem <= shm:
            limit = "free memory less the safety margin and the engine reserve"
        else:
            limit = f"free space on {cfg.ram_path.parent}"
        if got == 0:
            log.warning(
                "RAM tier off: requested %.2f GiB, only %.2f GiB fits (%s), under the "
                "%.2f GiB minimum",
                wanted / GIB,
                min(shm, mem) / GIB,
                limit,
                ram.MIN_POOL_BYTES / GIB,
            )
        else:
            log.warning(
                "RAM tier shrunk: requested %.2f GiB, granted %.2f GiB (%s)",
                wanted / GIB,
                got / GIB,
                limit,
            )
        return got

    @staticmethod
    def _create_pool(cfg: StoreConfig, size: int) -> int:
        """Create and commit the pool file now; the budget, or 0 when that failed."""
        if size == 0:
            return 0
        t0 = time.monotonic()
        try:
            create_pool_file(cfg.ram_path, size)
        except OSError as exc:
            log.warning("no RAM tier: cannot create %s: %s", cfg.ram_path, exc)
            return 0
        log.info(
            "RAM pool %s: %.2f GiB committed in %.1f s",
            cfg.ram_path,
            size / GIB,
            time.monotonic() - t0,
        )
        return size

    def _fit_ram(self, slot_bytes: int) -> bool:
        if self._ram_budget == 0:
            return False
        if self._pool is not None:
            if slot_bytes <= self._pool.slot_bytes:
                return True
            if any(b.pins or b.queued or b.state != _VALID for b in self._ram.values()):
                log.warning("RAM pool busy: a namespace needing %d-byte slots waits", slot_bytes)
                return False
            self._ram.clear()
            self._children.clear()
            self._acct["ram"].clear()  # every RAM copy went with the old slots
            self._positions["ram"].clear()
            self._lru_clear()
            self._saving.clear()
            self._pool.close()
            self._pool = None
        try:
            self._pool = RamPool(
                self.cfg.ram_path, self._ram_budget, slot_bytes, fixed_size=True, precommitted=True
            )
        except (OSError, ValueError) as exc:
            log.warning("no RAM tier: %s", exc)
            self.cfg.ram_path.unlink(missing_ok=True)  # give its memory back
            self._ram_budget = 0
            for info in self._ns.values():
                info.ram_ok = False
            return False
        for info in self._ns.values():
            info.ram_ok = info.slot_bytes <= slot_bytes
        return True

    def _usable(self, ns_id: bytes) -> _Ns | None:
        info = self._ns.get(ns_id)
        if info is None or not info.live or not info.ram_ok or self._pool is None:
            return None
        return info

    def _offsets(self, info: _Ns, slot: int) -> dict[str, int]:
        assert self._pool is not None
        base = self._pool.offset(slot)
        return {g: base + off for g, off in info.offsets.items()}

    def _add_child(self, pk: _Ik) -> None:
        n = self._children.get(pk, 0) + 1
        self._children[pk] = n
        if n == 1:
            self._cand_refresh_block(pk)  # no longer a leaf

    def _drop_child(self, pk: _Ik) -> None:
        n = self._children.get(pk, 0) - 1
        if n > 0:
            self._children[pk] = n
        else:
            self._children.pop(pk, None)
            self._cand_refresh_block(pk)  # a leaf again

    def _unsaving(self, tenant: bytes) -> None:
        n = self._saving.get(tenant, 0) - 1
        if n > 0:
            self._saving[tenant] = n
        else:
            self._saving.pop(tenant, None)

    def _remove(self, ik: _Ik, b: _Block) -> None:
        if b.state == _SAVING:
            self._unsaving(b.tenant)
        del self._ram[ik]
        assert self._pool is not None
        self._pool.free(b.slot)
        self._drop_child((ik[0], b.parent))
        if b.charged:
            self._charge("ram", (ik[0], b.model, b.tenant, b.scope_tag), -b.charged)
            b.charged = 0
        self._pos_del("ram", (b.model, b.pos), ik)
        self._lru_del((b.model, b.pos), b.tenant)
        if self._evict_for is not None:
            self._count_eviction("ram", b.tenant, b.model)

    def _alloc(self, taker: bytes = b"", model: bytes = b"", *, load: bool = False) -> int | None:
        """A free slot, evicting for it if need be.

        ``taker`` is who needs the room, ``model`` the model of the block it is for and
        ``load`` whether the block is promoted by a load rather than saved (the last two
        are read only under ``fair``).
        """
        assert self._pool is not None
        if self._policy.mode == "fair":
            return self._alloc_fair(taker, model, load)
        slot = self._pool.alloc()
        while slot is None:
            self._evict_for = taker
            try:
                evicted = self._evict_one()
            finally:
                self._evict_for = None
            if not evicted:
                return None
            slot = self._pool.alloc()
        return slot

    def _evict_one(self) -> bool:
        for ik, b in self._ram.items():
            if b.state != _VALID or b.pins or b.queued:
                continue
            if not (b.durable and self._nvme_on(ik[0])) and not self._is_leaf(ik):
                continue
            self._remove(ik, b)
            self._evictions += 1
            return True
        return False

    # ---- fair RAM victims (lock held) ---------------------------------------------

    def _alloc_fair(self, taker: bytes, model: bytes, load: bool = False) -> int | None:
        """``_alloc`` under ``fair``: the taker's cap first, then the victim order.

        A taker over its share in this model, when no over-share pair has an evictable
        position, is refused for a save (R4). A load in that state borrows instead: the
        least recently used position whose every RAM block is also on NVMe, so its
        owner only moves to the slower tier (ruling R14 revised, spec 5.3). The loader
        is charged for the block, so it is the most over and gives the room back first.
        """
        pool = self._pool
        assert pool is not None
        self._evict_for = taker
        try:
            cap = self._ram_cap(taker)
            if cap is not None and (self._saving.get(taker, 0) + 1) * pool.slot_bytes > cap:
                return None  # saves in flight fill the cap: evicting would not make room
            while cap is not None and self._tenant_ram(taker) >= cap:
                own = self._own_victim(taker)
                if own is None:
                    return None  # at its cap with nothing of its own to give back
                self._drop_victim(own)
            slot = pool.alloc()
            while slot is None:
                victim = self._dead_victim() or self._idle_victim()
                if victim is None:
                    victim, order = self._over_victim()
                    if victim is None:
                        if (model, taker) in order:
                            # A saver over its share in this model never takes an
                            # under-share key's block (spec 5.3): refused, a later miss.
                            # A load may demote one that is also on NVMe.
                            if not load:
                                return None
                            victim = self._oldest_durable_victim()
                        else:
                            victim = self._oldest_victim()
                if victim is None:
                    return None
                self._drop_victim(victim)
                slot = pool.alloc()
            return slot
        finally:
            self._evict_for = None

    def _drop_victim(self, victim: _Victim) -> None:
        for ik, b in victim:
            self._remove(ik, b)
            self._evictions += 1

    def _evictable(self, pos: _Pos) -> _Victim | None:
        """The RAM members of ``pos`` if the whole position can go now, else None."""
        members = self._positions["ram"].get(pos)
        if not members:
            return None
        out: _Victim = []
        for ik in sorted(members):
            b = self._ram[ik]
            if b.state != _VALID or b.pins or b.queued:
                return None
            if not (b.durable and self._nvme_on(ik[0])) and not self._leaf_in_memory(ik):
                return None
            out.append((ik, b))
        return out

    def _leaf_in_memory(self, ik: _Ik) -> bool:
        """``_is_leaf`` from the in-memory indexes (no SQLite under the lock)."""
        if self._children.get(ik, 0):
            return False
        return not (self._nvme_on(ik[0]) and self._nv_kids.get(ik, 0))

    def _pos_dead(self, pos: _Pos) -> bool:
        """Class dead: every RAM member of the position is."""
        members = self._positions["ram"].get(pos, set())
        return bool(members) and all(self._dead(ik[0], self._ram[ik].scope_tag) for ik in members)

    def _first_evictable(self, lru: Iterable[_Pos], *, dead: bool = False) -> _Victim | None:
        for pos in lru:
            if dead and not self._pos_dead(pos):
                continue
            got = self._evictable(pos)
            if got is not None:
                return got
        return None

    def _oldest_of(self, pairs: Iterable[_Pair], *, dead: bool = False) -> _Victim | None:
        """The least recently used evictable position over several pairs' lists."""
        best: _Victim | None = None
        best_seq = 0
        for pair in pairs:
            got = self._first_evictable(_listed(self._cand_pair.get(pair, ())), dead=dead)
            if got is None:
                continue
            b = got[0][1]
            seq = self._pos_seq[(b.model, b.pos)]
            if best is None or seq < best_seq:
                best, best_seq = got, seq
        return best

    def _dead_victim(self) -> _Victim | None:
        """Class dead, oldest first; only the pairs holding dead bytes are scanned."""
        pairs = {(m, t) for (ns_id, m, t, tag) in self._acct["ram"] if self._dead(ns_id, tag)}
        return self._oldest_of(sorted(pairs), dead=True) if pairs else None

    def _idle_victim(self) -> _Victim | None:
        """Class idle: unused for longer than ``idle_ram_s``, oldest first."""
        cutoff = self.cfg.clock() - self._policy.idle_ram_s
        for _, pos in self._cand_all:
            if self._pos_used[pos] >= cutoff:
                return None  # touch order: everything after was used later
            got = self._evictable(pos)
            if got is not None:
                return got
        return None

    def _over_victim(self) -> tuple[_Victim | None, list[_Pair]]:
        """The over-share pairs in ``over_order``, each its least recently used position.

        Returns the victim (None when no over pair has one) and the order itself.
        """
        usage = self._usage("ram")
        if not usage:
            return None, []
        part = parts(self._capacity("ram"), self._model_weights())
        share = shares(usage, part, self._weight)
        order = over_order(usage, part, share)
        for pair in order:
            got = self._first_evictable(_listed(self._cand_pair.get(pair, ())))
            if got is not None:
                return got, order
        return None, order

    def _oldest_victim(self) -> _Victim | None:
        """Global LRU over positions, for a taker under its share (or nobody over)."""
        return self._first_evictable(_listed(self._cand_all))

    def _oldest_durable_victim(self) -> _Victim | None:
        """Global LRU over positions whose every RAM member is also on NVMe (a load's
        borrow: the owner keeps every block)."""
        for _, pos in self._cand_all:
            got = self._evictable(pos)
            if got is not None and all(b.durable and self._nvme_on(ik[0]) for ik, b in got):
                return got
        return None

    def _own_victim(self, tenant: bytes) -> _Victim | None:
        return self._oldest_of(sorted(p for p in self._cand_pair if p[1] == tenant))

    def _ram_cap(self, tenant: bytes) -> float | None:
        pct = self._policy.tenant(tenant).max_ram_pct
        if pct is None:
            return None
        return self._capacity("ram") * pct / 100

    def _tenant_ram(self, tenant: bytes) -> int:
        """Live RAM bytes of ``tenant``, its slots still being saved included."""
        pool = self._pool
        saving = self._saving.get(tenant, 0) * (pool.slot_bytes if pool is not None else 0)
        return saving + sum(
            n
            for (ns_id, _, t, tag), n in self._acct["ram"].items()
            if t == tenant and not self._dead(ns_id, tag)
        )

    def _lru_add(self, pos: _Pos, tenant: bytes, used: float) -> None:
        """A RAM member of ``pos``, charged to ``tenant``, joins it; the position is used."""
        ts = self._pos_tenants.setdefault(pos, {})
        ts[tenant] = ts.get(tenant, 0) + 1
        self._lru_touch(pos, used)

    def _lru_touch(self, pos: _Pos, used: float) -> None:
        if pos not in self._pos_tenants:
            return
        self._pos_used[pos] = used
        self._pos_seq[pos] = next(self._touches)
        self._cand_refresh(pos)

    def _lru_del(self, pos: _Pos, tenant: bytes) -> None:
        """A RAM member of ``pos`` charged to ``tenant`` leaves it."""
        ts = self._pos_tenants.get(pos)
        if ts is None or tenant not in ts:
            return
        if ts[tenant] > 1:
            ts[tenant] -= 1
        else:
            del ts[tenant]
        if ts:
            self._cand_refresh(pos)
            return
        del self._pos_tenants[pos]
        self._cand_unlist(pos)
        self._pos_used.pop(pos, None)
        self._pos_seq.pop(pos, None)

    def _lru_clear(self) -> None:
        self._cand_all.clear()
        self._cand_pair.clear()
        self._cand_seq.clear()
        self._cand_pairs.clear()
        self._pos_tenants.clear()
        self._pos_used.clear()
        self._pos_seq.clear()

    def _is_cand(self, pos: _Pos) -> bool:
        """Every RAM member is durable or has no RAM child (cheap; see ``_evictable``)."""
        for ik in self._positions["ram"].get(pos, ()):
            b = self._ram.get(ik)
            if b is None or (not b.durable and self._children.get(ik, 0)):
                return False
        return True

    def _cand_refresh(self, pos: _Pos) -> None:
        """List or unlist ``pos`` as a candidate, under its current touch number."""
        self._cand_unlist(pos)
        ts = self._pos_tenants.get(pos)
        if not ts or not self._is_cand(pos):
            return
        seq = self._pos_seq[pos]
        e = (seq, pos)
        bisect.insort(self._cand_all, e)
        pairs = [(pos[0], t) for t in ts]
        for pair in pairs:
            bisect.insort(self._cand_pair.setdefault(pair, []), e)
        self._cand_seq[pos] = seq
        self._cand_pairs[pos] = pairs

    def _cand_refresh_block(self, ik: _Ik) -> None:
        b = self._ram.get(ik)
        if b is not None:
            self._cand_refresh((b.model, b.pos))

    def _cand_unlist(self, pos: _Pos) -> None:
        seq = self._cand_seq.pop(pos, None)
        if seq is None:
            return
        e = (seq, pos)
        _sorted_del(self._cand_all, e)
        for pair in self._cand_pairs.pop(pos, ()):
            lst = self._cand_pair.get(pair)
            if lst is not None:
                _sorted_del(lst, e)
                if not lst:
                    del self._cand_pair[pair]

    def _is_leaf(self, ik: _Ik) -> bool:
        if self._children.get(ik, 0):
            return False
        m = self._manifests.get(ik[0]) if self._nvme_on(ik[0]) else None
        if m is None:
            return True
        try:
            return not m.has_child(ik[0], ik[1])
        except sqlite3.Error as exc:
            self._nvme_off(exc)
            return True

    def _load_one(
        self, info: _Ns, key: bytes, tenant: bytes = b""
    ) -> tuple[dict[str, int], _Block, bool] | None:
        """Pin one block: (offsets, block, promoted from NVMe by this call), or None.

        A non-empty ``tenant`` takes over the charge of the block's whole position.
        """
        ik = (info.ns_id, key)
        deadline = time.monotonic() + PROMOTE_WAIT_S
        with self._cond:
            while (b := self._ram.get(ik)) is not None and b.state == _PROMOTING:
                left = deadline - time.monotonic()
                if left <= 0:
                    log.warning("a promotion of one block took over %.0f s; a miss", PROMOTE_WAIT_S)
                    return None
                self._cond.wait(left)
            if self._usable(info.ns_id) is None:
                return None
            if b is not None:
                if b.state != _VALID or self._hidden(b.scope_tag):
                    return None
                b.pins += 1
                self._ram.move_to_end(ik)
                b.last_used = self.cfg.clock()
                self._lru_touch((b.model, b.pos), b.last_used)
                self._nv_touch(ik, b.last_used)
                if tenant:
                    self._take_charge(b, tenant)
                return self._offsets(info, b.slot), b, False
            entries = self._nvme_entries(info, key)
            if entries is None or self._hidden(entries[0].scope_tag):
                return None
            e0 = entries[0]
            parent = e0.parent
            self._add_child((info.ns_id, parent))
            slot = self._alloc(tenant, e0.model or info.ns_id[:MODEL_LEN], load=True)
            if slot is None:
                self._drop_child((info.ns_id, parent))
                return None
            scope = bytes.fromhex(e0.scope)
            b = _Block(
                slot, parent, scope, _PROMOTING, pins=1, durable=True,
                model=e0.model or info.ns_id[:MODEL_LEN], pos=e0.pos or key,
                tenant=e0.tenant, scope_tag=e0.scope_tag, last_used=self.cfg.clock(),
            )  # fmt: skip
            self._ram[ik] = b
            self._pos_add("ram", (b.model, b.pos), ik)
            self._lru_add((b.model, b.pos), b.tenant, b.last_used)
            pool = self._pool
        assert pool is not None
        ok: bool | None = None  # None: the promotion raised
        result: tuple[dict[str, int], _Block, bool] | None = None
        try:
            ok = self._promote(pool, slot, info, key, entries)  # outside the lock
        except Exception as exc:
            log.warning("promotion from NVMe failed, a miss: %r", exc)
        finally:
            # Whatever happened, the block leaves PROMOTING and waiters wake up.
            with self._cond:
                try:
                    # A scope retired while the bytes were copied is never served.
                    if ok and not self._hidden(b.scope_tag):
                        b.state = _VALID
                        self._touch_nvme(info.ns_id, key)
                        self._nv_touch(ik, self.cfg.clock())
                        self._charge_ram(ik, b)
                        if tenant:
                            self._take_charge(b, tenant)
                        result = self._offsets(info, slot), b, True
                    elif self._ram.get(ik) is b:
                        self._remove(ik, b)
                        if ok is False:  # a damaged record is a miss, never a load
                            self._drop_nvme(info.ns_id, key)
                finally:
                    self._cond.notify_all()
        return result

    # ---- NVMe tier --------------------------------------------------------------

    def _nvme_on(self, ns_id: bytes) -> bool:
        """Is the NVMe tier on, and on for this namespace?"""
        info = self._ns.get(ns_id)
        return self._nvme_enabled and info is not None and info.nvme

    def _nvme_off(self, exc: BaseException) -> None:
        self._nvme_enabled = False
        if not self._nvme_off_logged:
            self._nvme_off_logged = True
            log.warning("NVMe tier switched off, RAM tier continues: %s", exc)

    def _ns_dir(self, ns_id: bytes) -> Path:
        assert self.cfg.nvme_dir is not None
        return self.cfg.nvme_dir / ns_id.hex()

    def _open_manifest(self, ns_id: bytes) -> None:
        """Open and install a manifest (only where no other thread runs: ``__init__``)."""
        self._install(self._read_manifest(ns_id))

    def _read_manifest(self, ns_id: bytes) -> _Opened:
        """Open a namespace's manifest and read every block and segment (no lock held)."""
        d = self._ns_dir(ns_id)
        d.mkdir(exist_ok=True)
        m = Manifest(d / MANIFEST_NAME)
        try:
            segs = []
            for seg in _existing_segments(d):
                st = _seg_path(d, seg).stat()
                segs.append((seg, st.st_mtime, st.st_size))
            rows = list(m.iter_blocks(ns_id))
        except BaseException:
            m.close()
            raise
        return _Opened(ns_id, m, segs, rows)

    def _install(self, o: _Opened) -> None:
        """Make a read manifest live: segments, the NVMe index and its accounting."""
        ns_id = o.ns_id
        self._manifests[ns_id] = o.manifest
        for seg, mtime, size in o.segs:
            self._segs[(ns_id, seg)] = [mtime, size]
            self._nvme_used += size
            self._rank_new((ns_id, seg))
        for r in o.rows:
            # A step 1 row has no model or position: the namespace's one model, pos = key.
            rec = _NvRec(
                model=r.model or ns_id[:MODEL_LEN], pos=r.pos or r.key, parent=r.parent,
                tenant=r.tenant,
                scope_tag=r.scope_tag, nbytes=sum(n for _, n in r.segments),
                segments=dict(r.segments), last_used=r.last_used,
            )  # fmt: skip
            self._nv_add((ns_id, r.key), rec)

    def _open_late(self, ns_id: bytes) -> None:
        """Open a namespace's manifest after registration, then let NVMe on (no lock held)."""
        try:
            opened: _Opened | None = self._read_manifest(ns_id)
        except _NVME_ERRORS as exc:
            opened = None
            with self._cond:
                self._nvme_off(exc)
        with self._cond:
            if opened is not None:
                if ns_id in self._manifests or not self._nvme_enabled:
                    opened.manifest.close()
                else:
                    self._install(opened)
            info = self._ns.get(ns_id)
            if info is not None:
                info.nvme = self._wants_nvme(info)

    def _wants_nvme(self, info: _Ns) -> bool:
        """I3: every owner wants NVMe (no owner left: the flag stays as it was)."""
        return all(info.owners.values()) if info.owners else info.nvme

    def _set_nvme(self, info: _Ns) -> bool:
        """Set the namespace's NVMe flag; True when its manifest must be opened first.

        NVMe stays off for the namespace until then: a write never runs without its
        manifest.
        """
        want = self._wants_nvme(info)
        late = want and self._nvme_enabled and info.ns_id not in self._manifests
        info.nvme = want and not late
        return late

    def _attach_retired(self, nvme_dir: Path) -> None:
        """Persist the retired set next to the manifests (no lock held).

        An unreadable file leaves the set refusing isolated blocks (``_hidden``).
        """
        try:
            self._retired.attach(nvme_dir / RETIRED_NAME)
        except RetiredStoreError:
            pass  # logged by RetiredScopes; isolation refused until a later attach works

    def _nvme_present(self, info: _Ns, key: bytes) -> bool:
        entries = self._nvme_entries(info, key)
        return entries is not None and not self._hidden(entries[0].scope_tag)

    def _nvme_entries(self, info: _Ns, key: bytes) -> list[Entry] | None:
        """The block's NVMe records if every group is present with the right size."""
        m = self._manifests.get(info.ns_id) if self._nvme_enabled and info.nvme else None
        if m is None:
            return None
        try:
            entries = m.get(info.ns_id, key)
        except sqlite3.Error as exc:
            self._nvme_off(exc)
            return None
        if {e.group: e.nbytes for e in entries} != info.groups:
            return None
        return entries

    def _touch_nvme(self, ns_id: bytes, key: bytes) -> None:
        m = self._manifests.get(ns_id)
        if m is None or not self._nvme_enabled:
            return
        try:
            m.touch(ns_id, key)
        except sqlite3.Error as exc:
            self._nvme_off(exc)

    def _drop_nvme(self, ns_id: bytes, key: bytes) -> list[Entry]:
        m = self._manifests.get(ns_id)
        if m is None:
            return []
        try:
            entries = m.get(ns_id, key)
            m.delete(ns_id, key)
        except sqlite3.Error as exc:
            self._nvme_off(exc)
            return []
        rec = self._nv.get((ns_id, key))
        if rec is not None and self._evict_for is not None:
            self._count_eviction("nvme", rec.tenant, rec.model)
        self._nv_del((ns_id, key))
        # The record bytes stay on disk until their whole segment is deleted.
        b = self._ram.get((ns_id, key))
        if b is not None:
            b.durable = False
            self._cand_refresh((b.model, b.pos))
        return entries

    def _promote(
        self, pool: RamPool, slot: int, info: _Ns, key: bytes, entries: list[Entry]
    ) -> bool:
        d = self._ns_dir(info.ns_id)
        with pool.view(slot) as view:
            for e in entries:
                data = read_record(d, e.loc, self._data_key, key, e.group)
                if data is None or len(data) != info.groups[e.group]:
                    return False
                off = info.offsets[e.group]
                view[off : off + len(data)] = data
        return True

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write_through(item)
            except Exception:
                log.exception("write-through failed")
            finally:
                self._queue.task_done()

    def _write_through(self, ik: _Ik) -> None:
        with self._cond:
            b = self._ram.get(ik)  # a queued block is never evicted: this is the one queued
        try:
            self._write_through_once(ik)
        finally:
            if b is not None:
                with self._cond:
                    b.queued = False  # whatever raised, the block is evictable again

    def _write_through_once(self, ik: _Ik) -> None:
        ns_id, key = ik
        with self._cond:
            b = self._ram.get(ik)
            info = self._ns.get(ns_id)
            if b is None:
                return
            if info is None or not self._nvme_on(ns_id) or self._pool is None:
                b.queued = False
                return
            # Bytes this block adds on disk, plus a header in case a new segment starts.
            need = sum(record_size(g, n) for g, n in info.groups.items()) + HEADER_LEN
            self._evict_for = b.tenant
            try:
                room = self._make_nvme_room(need, b.tenant, ns_id)
            finally:
                self._evict_for = None
            if not room:
                b.queued = False  # stays RAM only, evictable as a leaf
                return
            pool = self._pool
        try:  # a queued block is never evicted, so its slot is stable here
            with pool.view(b.slot) as view:
                payloads = {
                    g: bytes(view[info.offsets[g] : info.offsets[g] + n])
                    for g, n in info.groups.items()
                }
            writer = self._writers.get(ns_id)
            if writer is None:
                writer = SegmentWriter(
                    self._ns_dir(ns_id),
                    ns_id,
                    self._data_key,
                    SEGMENT_BYTES,
                    first_segment=self._seg_floor.get(ns_id, 1),
                )
                self._writers[ns_id] = writer
            locs = {}
            for g, payload in payloads.items():
                locs[g] = writer.append(key, g, payload)
                self._cur_seg[ns_id] = locs[g].segment
        except _NVME_ERRORS as exc:
            with self._cond:
                b.queued = False
                self._nvme_off(exc)
            return
        with self._cond:
            b.queued = False
            for loc in locs.values():  # the records are on disk either way
                self._grow_seg(ns_id, loc.segment, loc.offset + loc.length)
            if not self._nvme_on(ns_id):
                return  # a RAM-only owner registered meanwhile: no row, stays RAM only
            try:
                m = self._manifests[ns_id]
                for g, loc in locs.items():
                    m.put(
                        ns_id, key, b.parent, g, loc, info.groups[g], b.scope.hex(),
                        model=b.model, pos=b.pos, tenant=b.tenant, scope_tag=b.scope_tag,
                    )  # fmt: skip
            except (*_NVME_ERRORS, KeyError) as exc:
                self._nvme_off(exc)
                return
            b.durable = True
            self._cand_refresh((b.model, b.pos))
            segments: dict[int, int] = {}
            for loc in locs.values():
                segments[loc.segment] = segments.get(loc.segment, 0) + loc.length
            rec = _NvRec(
                model=b.model, pos=b.pos, parent=b.parent, tenant=b.tenant,
                scope_tag=b.scope_tag, nbytes=sum(segments.values()), segments=segments, last_used=self.cfg.clock(),
            )  # fmt: skip
            self._nv_add(ik, rec)

    def _grow_seg(self, ns_id: bytes, seg: int, end: int) -> None:
        """A record now ends at ``end`` in ``seg``: its file is that long (lock held)."""
        rec = self._segs.get((ns_id, seg))
        if rec is None:
            rec = self._segs[(ns_id, seg)] = [self.cfg.clock(), 0]
            self._rank_new((ns_id, seg))
        if end > rec[1]:
            self._nvme_used += int(end - rec[1])
            rec[1] = end
            self._rank_dirty.add((ns_id, seg))

    def _make_nvme_room(
        self, need: int, for_tenant: bytes = b"", ns_id: bytes | None = None
    ) -> bool:
        """Reclaim whole segments until ``need`` more bytes fit (writer only).

        ``lru``: oldest first (step 1). ``fair``: ``_reclaim_segment``, which also keeps
        ``for_tenant`` under its ``max_nvme_pct`` cap, and lets the lock go between
        reclaims: then ``ns_id``, the namespace about to be written, must still have NVMe.
        """
        if need > self.cfg.nvme_cap_bytes:
            return False
        if self._policy.mode == "fair":
            return self._make_nvme_room_fair(need, for_tenant, ns_id)
        while self._nvme_used + need > self.cfg.nvme_cap_bytes:
            try:
                if not self._reclaim_oldest_segment():
                    return False
            except _NVME_ERRORS as exc:
                self._nvme_off(exc)
                return False
        return True

    def _make_nvme_room_fair(self, need: int, for_tenant: bytes, ns_id: bytes | None) -> bool:
        """``_make_nvme_room`` under ``fair``. The lock is let go between reclaims, so
        one write's room never holds it for more than one reclaim at a time.

        ``_evict_for`` (who the drops are charged to) is the caller's while a reclaim
        runs and None while the lock is let go: another thread's removals meanwhile are
        never charged to this writer, and its own setting never leaks into a reclaim.
        """
        cap = self._nvme_tenant_cap(for_tenant)
        if cap is not None and need > cap:
            return False
        taker = self._evict_for
        total = self.cfg.nvme_cap_bytes
        first = True
        try:
            while self._nvme_used + need > total or self._nvme_capped(for_tenant, need):
                if not first:
                    self._evict_for = None
                    self._cond.release()
                    os.sched_yield()
                    self._cond.acquire()
                    if not self._nvme_enabled or (ns_id is not None and not self._nvme_on(ns_id)):
                        return False  # NVMe went off, or a RAM-only owner registered
                first = False
                self._evict_for = taker
                try:
                    if not self._reclaim_segment(for_tenant, need):
                        return False
                except _NVME_ERRORS as exc:
                    self._nvme_off(exc)
                    return False
            return True
        finally:
            self._evict_for = taker

    def _reclaim_oldest_segment(self) -> bool:
        """Delete the oldest segment that can go, with its blocks; False when none can."""
        for (ns_id, seg), _ in sorted(self._segs.items(), key=lambda kv: (kv[1][0], kv[0])):
            plan = self._plan_reclaim(ns_id, seg)
            if plan is None:
                continue  # a block in it, or a descendant, is busy in RAM
            self._apply_reclaim(ns_id, seg, plan)
            return True
        return False

    def _apply_reclaim(
        self, ns_id: bytes, seg: int, plan: tuple[set[bytes], list[tuple[_Ik, _Block]]]
    ) -> None:
        """Drop a reclaim plan's blocks and delete the segment (and any it emptied)."""
        nvme_drop, ram_drop = plan
        touched = {seg}
        for key in nvme_drop:
            touched |= {e.loc.segment for e in self._drop_nvme(ns_id, key)}
            self._nvme_evictions += 1
        for ik, b in ram_drop:
            if self._ram.get(ik) is b:
                self._remove(ik, b)
                self._evictions += 1
        self._delete_segment(ns_id, seg)
        m = self._manifests.get(ns_id)
        for other in touched - {seg}:
            if (
                m is not None
                and other != self._cur_seg.get(ns_id)
                and not m.segment_in_use(ns_id, other)
            ):
                self._delete_segment(ns_id, other)

    # ---- fair NVMe segments (writer thread, lock held) ------------------------------

    def _reclaim_segment(self, for_tenant: bytes = b"", need: int = 0) -> bool:
        """Under ``fair``: delete the best-scored segment that can go; False when none can.

        A ``for_tenant`` at its ``max_nvme_pct`` cap (with ``need`` more bytes) may only
        take a segment holding its own records. Every dropped block that is a member of
        a position takes the position's members in the other namespaces with it.
        """
        got = self._choose_segment(for_tenant, need)
        if got is None:
            return False
        ns_id, seg, plan = got
        recs = [self._nv[(ns_id, k)] for k in sorted(plan[0]) if (ns_id, k) in self._nv]
        self._apply_reclaim(ns_id, seg, plan)
        self._drop_position_peers(ns_id, recs)
        return True

    def _choose_segment(
        self, for_tenant: bytes, need: int
    ) -> tuple[bytes, int, tuple[set[bytes], list[tuple[_Ik, _Block]]]] | None:
        """The segment ``_reclaim_segment`` takes, with its plan; None when none can go.

        A capped tenant: ``_choose_capped``. Otherwise the cached ranking: the segments
        whose maps changed, ``RANK_SLICE`` more from the ring and the ``RANK_TOP`` best
        are rescored, then the best that ``_plan_reclaim`` allows wins, the writer's
        current segments last, after at most ``RECLAIM_TRIES`` plans. The cost per call
        is bounded by those constants plus the changes since the last call, not by the
        number of segments.
        """
        if self._nvme_capped(for_tenant, need):
            return self._choose_capped(for_tenant)
        ctx = self._score_ctx()
        self._rank_refresh(ctx)
        current = set(self._cur_seg.items())
        held: list[tuple[tuple[Any, ...], int, tuple[bytes, int]]] = []
        later: list[tuple[bytes, int]] = []
        chosen: tuple[bytes, int, tuple[set[bytes], list[tuple[_Ik, _Block]]]] | None = None
        tries = 0
        while tries < RECLAIM_TRIES:
            e = self._rank_pop()
            if e is None:
                break
            k = e[2]
            held.append(e)
            if k in current:
                later.append(k)
                continue
            tries += 1
            plan = self._plan_reclaim(*k)
            if plan is not None:
                chosen = (k[0], k[1], plan)
                break
        for k in later:
            if chosen is not None or tries >= RECLAIM_TRIES:
                break
            tries += 1
            plan = self._plan_reclaim(*k)
            if plan is not None:
                chosen = (k[0], k[1], plan)
        for e in held:
            if chosen is None or e[2] != (chosen[0], chosen[1]):
                heapq.heappush(self._rank_heap, e)
        return chosen

    def _rank_pop(self) -> tuple[tuple[Any, ...], int, tuple[bytes, int]] | None:
        """The best valid entry of the two heaps, removed; None when both are empty."""
        heap, old = self._rank_heap, self._rank_old
        while heap or old:
            src = heap if not old or (heap and heap[0] < old[0]) else old
            e = heapq.heappop(src)
            if self._rank_ver.get(e[2]) == e[1] and e[2] in self._segs:
                return e
        return None

    def _choose_capped(
        self, tenant: bytes
    ) -> tuple[bytes, int, tuple[set[bytes], list[tuple[_Ik, _Block]]]] | None:
        """A tenant at its NVMe cap reclaims a segment of its own: the one where its share
        of the live bytes is largest (then the oldest). If even there the other tenants'
        live bytes outweigh its own, the write is refused instead (stays RAM only, as at
        the RAM cap): a capped tenant never destroys more of others' data than its own.
        """
        ctx = self._score_ctx(ratios=False)
        cands: list[tuple[float, float, tuple[bytes, int], int, int]] = []
        for k in self._capped_slice(tenant):
            seg = self._segs.get(k)
            if seg is None:
                continue
            own = others = 0
            for (_, t, tag), n in self._seg_acct.get(k, {}).items():
                if self._dead_cached(ctx, k[0], tag):
                    continue
                if t == tenant:
                    own += n
                else:
                    others += n
            if own:
                cands.append((-own / (own + others), seg[0], k, own, others))
        cands.sort()
        for _, _, k, own, others in cands[:RECLAIM_TRIES]:
            if own < others:
                return None
            plan = self._plan_reclaim(*k)
            if plan is not None:
                return k[0], k[1], plan
        return None

    def _capped_slice(self, tenant: bytes) -> list[tuple[bytes, int]]:
        """Up to ``CAPPED_SCAN`` of the tenant's segments; each call moves on to the next
        slice (wrapping), so a good segment past the first slice is reached in turn."""
        segs = self._tenant_segs.get(tenant, {})
        n = len(segs)
        if n <= CAPPED_SCAN:
            self._capped_pos.pop(tenant, None)
            return list(segs)
        start = self._capped_pos.get(tenant, 0) % n
        out = list(itertools.islice(segs, start, start + CAPPED_SCAN))
        if len(out) < CAPPED_SCAN:
            out += itertools.islice(segs, 0, CAPPED_SCAN - len(out))
        self._capped_pos[tenant] = start + CAPPED_SCAN
        return out

    def _rank_new(self, k: tuple[bytes, int]) -> None:
        """A new segment joins the ring (both modes; deleted ones are swept out here)."""
        if len(self._ring) > 2 * len(self._segs) + 1024:
            self._ring = [x for x in self._ring if x in self._segs]
            self._ring_pos = 0
        self._ring.append(k)
        self._rank_dirty.add(k)

    def _rank_push(self, k: tuple[bytes, int], ctx: _ScoreCtx) -> None:
        seg = self._segs.get(k)
        if seg is None:
            return
        dead, idle, over, _ = self._segment_score(k[0], k[1], ctx=ctx)
        ver = next(self._rank_vers)
        self._rank_ver[k] = ver
        heapq.heappush(self._rank_heap, ((-dead, -idle, -over, seg[0], k), ver, k))

    def _rank_refresh(self, ctx: _ScoreCtx) -> None:
        """Rescore what changed, a slice of the ring and the current best (lock held)."""
        # Liveness, retirement or the grace ending change dead bytes everywhere: rare.
        sig = (
            frozenset(n for n, info in self._ns.items() if info.live),
            len(self._retired),
            self._grace_over(),
        )
        if sig != self._rank_sig:
            self._rank_sig = sig
            self._rank_dirty.update(self._segs)
        for k in self._rank_dirty:
            self._rank_push(k, ctx)
        self._rank_dirty.clear()
        ring = self._ring
        for _ in range(min(RANK_SLICE, len(ring))):
            if self._ring_pos >= len(ring):
                self._ring_pos = 0
            k = ring[self._ring_pos]
            self._ring_pos += 1
            self._rank_push(k, ctx)
        if not self._rank_old and len(self._rank_heap) > 2 * len(self._segs) + 1024:
            self._rank_old, self._rank_heap = self._rank_heap, []
        old = self._rank_old
        for _ in range(min(RANK_DRAIN, len(old))):
            e = old.pop()  # the last element: what is left is still a heap
            if self._rank_ver.get(e[2]) == e[1] and e[2] in self._segs:
                heapq.heappush(self._rank_heap, e)
        top: list[tuple[bytes, int]] = []
        while len(top) < RANK_TOP:
            got = self._rank_pop()
            if got is None:
                break
            top.append(got[2])
        for k in top:
            self._rank_push(k, ctx)

    def _score_ctx(self, *, ratios: bool = True) -> _ScoreCtx:
        """What every segment's score shares: computed once per choice."""
        return _ScoreCtx(
            ratios=self._nvme_over_ratios() if ratios else {},
            cutoff=self.cfg.clock() - self._policy.idle_nvme_s,
            dead_ns={},
            dead_tag={},
        )

    def _dead_cached(self, ctx: _ScoreCtx, ns_id: bytes, tag: bytes) -> bool:
        d = ctx.dead_ns.get(ns_id)
        if d is None:
            d = ctx.dead_ns[ns_id] = self._dead(ns_id, b"")
        if d or not tag:
            return d
        t = ctx.dead_tag.get(tag)
        if t is None:
            t = ctx.dead_tag[tag] = tag in self._retired
        return t

    def _segment_score(
        self, ns_id: bytes, seg: int, *, ctx: _ScoreCtx | None = None
    ) -> tuple[int, int, float, float]:
        """``(dead bytes, idle bytes, over-share bytes, -mtime)``, compared largest first.

        Dead: records of retired-scope or unregistered blocks, plus bytes of records
        already dropped (the file is longer than its live records). Idle: live records
        unused for longer than ``idle_nvme_s``. Over-share: live records of the pairs in
        the NVMe ``over_order``, each weighted by ``usage / share``.
        """
        if ctx is None:
            ctx = self._score_ctx()
        k = (ns_id, seg)
        age, size = self._segs.get(k, (0.0, 0.0))
        dead = live = 0
        over = 0.0
        ratios = ctx.ratios
        acct = self._seg_acct.get(k)
        if acct:
            if self._dead_cached(ctx, ns_id, b""):
                live = dead = sum(acct.values())
            else:
                for (model, tenant, tag), n in acct.items():
                    live += n
                    if tag and self._dead_cached(ctx, ns_id, tag):
                        dead += n
                    elif ratios:
                        r = ratios.get((model, tenant))
                        if r is not None:
                            over += n * r
        idle = self._idle_bytes(k, ctx, live - dead)
        dead += max(0, int(size) - HEADER_LEN - live)
        return dead, idle, over, -age

    def _idle_bytes(self, k: tuple[bytes, int], ctx: _ScoreCtx, live: int) -> int:
        """Live bytes of the segment's records last used before the idle cutoff."""
        use = self._seg_use.get(k)
        cutoff = ctx.cutoff
        if use is None or use[0] >= cutoff or live <= 0:
            return 0
        if use[1] < cutoff:
            return live
        n = 0
        for key, nb in self._seg_bytes.get(k, {}).items():
            rec = self._nv.get((k[0], key))
            if (
                rec is not None
                and rec.last_used < cutoff
                and not self._dead_cached(ctx, k[0], rec.scope_tag)
            ):
                n += nb
        return n

    def _nvme_over_ratios(self) -> dict[_Pair, float]:
        """``usage / share`` of every pair in the NVMe ``over_order`` (inf: no share)."""
        usage = self._usage("nvme")
        if not usage:
            return {}
        part = parts(self._capacity("nvme"), self._model_weights())
        share = shares(usage, part, self._weight)
        out: dict[_Pair, float] = {}
        for pair in over_order(usage, part, share):
            q = share.get(pair, 0.0)
            out[pair] = usage[pair] / q if q > 0 else float("inf")
        return out

    def _nvme_tenant_cap(self, tenant: bytes) -> float | None:
        pct = self._policy.tenant(tenant).max_nvme_pct
        if pct is None:
            return None
        return self._capacity("nvme") * pct / 100

    def _nvme_capped(self, tenant: bytes, need: int) -> bool:
        """``tenant`` (fair only) cannot add ``need`` bytes without passing its NVMe cap."""
        cap = self._nvme_tenant_cap(tenant) if self._policy.mode == "fair" else None
        if cap is None or self._tenant_nv.get(tenant, 0) + need <= cap:
            return False  # its bytes, dead ones too, fit: no need to count the live ones
        used = sum(
            n
            for (ns_id, _, t, tag), n in self._acct["nvme"].items()
            if t == tenant and not self._dead(ns_id, tag)
        )
        return used + need > cap

    def _drop_position_peers(self, ns_id: bytes, recs: list[_NvRec]) -> None:
        """Drop from their manifests the members, in other namespaces, of these positions.

        No bytes are freed (their segments stay). A member keeps presence through a valid
        RAM copy; one without is dropped only once it has no child in either tier, so
        presence stays prefix-closed in its namespace (members of a chain go leaf first).
        A member being promoted stays.
        """
        peers: set[_Ik] = set()
        for rec in recs:
            for ik in self._positions["nvme"].get((rec.model, rec.pos), ()):
                if ik[0] != ns_id:
                    peers.add(ik)
        pending = sorted(peers)
        while pending:
            rest: list[_Ik] = []
            for ik in pending:
                if ik not in self._nv:
                    continue
                b = self._ram.get(ik)
                if b is not None and b.state == _PROMOTING:
                    continue
                keeps = b is not None and b.state == _VALID
                if not keeps and (self._nv_kids.get(ik, 0) or self._children.get(ik, 0)):
                    rest.append(ik)
                    continue
                self._drop_nvme(*ik)
                self._nvme_evictions += 1
            if len(rest) == len(pending):
                break  # what is left has a child outside these positions: it stays
            pending = rest

    def _plan_reclaim(
        self, ns_id: bytes, seg: int
    ) -> tuple[set[bytes], list[tuple[_Ik, _Block]]] | None:
        """What deleting ``seg`` removes: (NVMe keys, RAM blocks), or None when busy.

        Every block with a record in the segment loses its NVMe copy. One with a valid
        RAM copy stays present; one without is gone, and so is every descendant (in
        either tier), so no stored block is left without its parent.
        """
        m = self._manifests.get(ns_id)
        keys = m.keys_in_segment(ns_id, seg) if m is not None else []
        nvme_drop: set[bytes] = set(keys)
        ram_drop: list[tuple[_Ik, _Block]] = []
        lost: list[bytes] = []
        for key in keys:
            b = self._ram.get((ns_id, key))
            if b is not None and b.state == _PROMOTING:
                return None  # being read from this very segment
            if b is None or b.state != _VALID:
                lost.append(key)
        if not lost:
            return nvme_drop, ram_drop
        ram_kids: dict[bytes, list[bytes]] = {}
        for (bns, bkey), b in self._ram.items():
            if bns == ns_id:
                ram_kids.setdefault(b.parent, []).append(bkey)
        seen = set(lost)
        while lost:
            parent = lost.pop()
            kids = set(ram_kids.get(parent, []))
            if m is not None:
                kids.update(m.children(ns_id, parent))
            for kid in kids - seen:
                seen.add(kid)
                b = self._ram.get((ns_id, kid))
                if b is not None:
                    if b.state != _VALID or b.pins or b.queued:
                        return None  # a busy descendant cannot be dropped now
                    ram_drop.append(((ns_id, kid), b))
                nvme_drop.add(kid)
                lost.append(kid)
        return nvme_drop, ram_drop

    def _delete_segment(self, ns_id: bytes, seg: int) -> None:
        """Remove one segment file and its bytes from the count (writer thread, lock held)."""
        if self._cur_seg.get(ns_id) == seg:
            writer = self._writers.pop(ns_id, None)
            self._cur_seg.pop(ns_id, None)
            if writer is not None:
                writer.close()  # the next write starts a new segment
        _seg_path(self._ns_dir(ns_id), seg).unlink(missing_ok=True)
        self._seg_floor[ns_id] = max(self._seg_floor.get(ns_id, 1), seg + 1)  # never reused
        rec = self._segs.pop((ns_id, seg), None)
        if rec is not None:
            self._nvme_used -= int(rec[1])
        k = (ns_id, seg)
        self._seg_bytes.pop(k, None)
        for (_, tenant, _), n in list(self._seg_acct.get(k, {}).items()):
            self._tenant_seg_charge(k, tenant, -n)  # normally none left: all were dropped
        self._seg_acct.pop(k, None)
        self._seg_use.pop(k, None)
        self._rank_ver.pop(k, None)
        self._rank_dirty.discard(k)

    # ---- attribution (lock held) ------------------------------------------------

    def _hidden(self, scope_tag: bytes) -> bool:
        """An isolated block that must not be served: its scope is retired, or the
        retired set cannot be trusted (fail closed). Sharing blocks are never hidden."""
        # Lock order: _cond, then RetiredScopes' lock; a slow retired.db write in
        # retire_scopes stalls this check (and so lookups) until it ends. Retires are rare.
        if not scope_tag:
            return False
        return self._retired.isolation_refused or scope_tag in self._retired

    def _dead(self, ns_id: bytes, scope_tag: bytes) -> bool:
        """Class dead: its namespace has no owner (released, or not registered within
        ``UNREGISTERED_GRACE_S`` of this store opening), or its scope is retired."""
        info = self._ns.get(ns_id)
        if info is None:
            if self._grace_over():
                return True
        elif not info.live:
            return True
        return bool(scope_tag) and scope_tag in self._retired

    def _grace_over(self) -> bool:
        if not self._grace_done:
            self._grace_done = self.cfg.clock() - self._opened_at >= UNREGISTERED_GRACE_S
        return self._grace_done

    def _charge(self, tier: Tier, acct: _Acct, n: int) -> None:
        book = self._acct[tier]
        total = book.get(acct, 0) + n
        if total:
            book[acct] = total
        else:
            book.pop(acct, None)

    def _charge_ram(self, ik: _Ik, b: _Block) -> None:
        """Count a block that just became valid in RAM: one pool slot."""
        if b.charged or self._pool is None:
            return
        b.charged = self._pool.slot_bytes
        self._charge("ram", (ik[0], b.model, b.tenant, b.scope_tag), b.charged)

    def _usage(self, tier: Tier) -> dict[_Pair, int]:
        out: dict[_Pair, int] = {}
        for (ns_id, model, tenant, tag), n in self._tier_acct(tier).items():
            if not self._dead(ns_id, tag):
                out[(model, tenant)] = out.get((model, tenant), 0) + n
        return out

    def _capacity(self, tier: Tier) -> int:
        """What the tier's cap limits: the pool's slots, or the NVMe segment-bytes cap."""
        if tier == "ram":
            pool = self._pool
            return pool.slots_total * pool.slot_bytes if pool is not None else 0
        return self.cfg.nvme_cap_bytes if self._nvme_enabled else 0

    def _model_weights(self) -> dict[bytes, int]:
        """Each live model's weight: the largest ``owner_weights`` value of its live owners."""
        owners: dict[bytes, set[str]] = {}
        for info in self._ns.values():
            if info.live:
                owners.setdefault(info.model, set()).update(info.owners)
        weights = self._policy.owner_weights
        out: dict[bytes, int] = {}
        for model, os_ in owners.items():
            got = [weights[o] for o in os_ if o in weights]
            out[model] = max(got) if got else DEFAULT_OWNER_WEIGHT
        return out

    def _weight(self, tenant: bytes) -> int:
        return self._policy.tenant(tenant).weight

    def _tier_acct(self, tier: Tier) -> dict[_Acct, int]:
        if tier == "nvme" and not self._nvme_enabled:
            return {}
        return self._acct[tier]

    def _pos_add(self, tier: Tier, pos: _Pos, ik: _Ik) -> None:
        self._positions[tier].setdefault(pos, set()).add(ik)

    def _pos_del(self, tier: Tier, pos: _Pos, ik: _Ik) -> None:
        members = self._positions[tier].get(pos)
        if members is None:
            return
        members.discard(ik)
        if not members:
            del self._positions[tier][pos]

    def _nv_add(self, ik: _Ik, rec: _NvRec) -> None:
        self._nv_del(ik)  # a rewrite replaces the old rows
        self._nv[ik] = rec
        pk = (ik[0], rec.parent)
        self._nv_kids[pk] = self._nv_kids.get(pk, 0) + 1
        self._pos_add("nvme", (rec.model, rec.pos), ik)
        for seg, n in rec.segments.items():
            k = (ik[0], seg)
            self._seg_bytes.setdefault(k, {})[ik[1]] = n
            self._seg_charge(k, rec, rec.tenant, n)
            use = self._seg_use.get(k)
            if use is None:
                self._seg_use[k] = [rec.last_used, rec.last_used]
            else:
                use[0] = min(use[0], rec.last_used)
                use[1] = max(use[1], rec.last_used)
        self._charge("nvme", (ik[0], rec.model, rec.tenant, rec.scope_tag), rec.nbytes)

    def _nv_del(self, ik: _Ik) -> None:
        self._tenant_dirty.pop(ik, None)  # a stale change never reaches a rewritten row
        rec = self._nv.pop(ik, None)
        if rec is None:
            return
        pk = (ik[0], rec.parent)
        n = self._nv_kids.get(pk, 0) - 1
        if n > 0:
            self._nv_kids[pk] = n
        else:
            self._nv_kids.pop(pk, None)
        self._pos_del("nvme", (rec.model, rec.pos), ik)
        for seg, nb in rec.segments.items():
            k = (ik[0], seg)
            self._seg_charge(k, rec, rec.tenant, -nb)
            keys = self._seg_bytes.get(k)
            if keys is not None:
                keys.pop(ik[1], None)
                if not keys:
                    del self._seg_bytes[k]
                    self._seg_acct.pop(k, None)
                    self._seg_use.pop(k, None)
        self._charge("nvme", (ik[0], rec.model, rec.tenant, rec.scope_tag), -rec.nbytes)

    def _seg_charge(self, k: tuple[bytes, int], rec: _NvRec, tenant: bytes, n: int) -> None:
        book = self._seg_acct.setdefault(k, {})
        a = (rec.model, tenant, rec.scope_tag)
        total = book.get(a, 0) + n
        if total:
            book[a] = total
        else:
            book.pop(a, None)
        self._tenant_seg_charge(k, tenant, n)
        self._rank_dirty.add(k)

    def _tenant_seg_charge(self, k: tuple[bytes, int], tenant: bytes, n: int) -> None:
        t = self._tenant_nv.get(tenant, 0) + n
        if t:
            self._tenant_nv[tenant] = t
        else:
            self._tenant_nv.pop(tenant, None)
        segs = self._tenant_segs.setdefault(tenant, {})
        got = segs.get(k, 0) + n
        if got:
            segs[k] = got
        else:
            segs.pop(k, None)
            if not segs:
                del self._tenant_segs[tenant]
                self._capped_pos.pop(tenant, None)

    def _nv_touch(self, ik: _Ik, now: float) -> None:
        """The block was used (a RAM hit or a promotion): its NVMe record is not idle."""
        rec = self._nv.get(ik)
        if rec is None:
            return
        rec.last_used = now
        for seg in rec.segments:
            use = self._seg_use.get((ik[0], seg))
            if use is not None and now > use[1]:
                use[1] = now
                self._rank_dirty.add((ik[0], seg))

    def _take_charge(self, b: _Block, tenant: bytes) -> None:
        """Last user pays: ``tenant`` now pays for every member of the position of ``b``."""
        pos = (b.model, b.pos)
        members = self._positions["ram"].get(pos, set()) | self._positions["nvme"].get(pos, set())
        for ik in members:
            rb = self._ram.get(ik)
            if rb is not None and rb.tenant != tenant:
                if rb.charged:
                    self._charge("ram", (ik[0], rb.model, rb.tenant, rb.scope_tag), -rb.charged)
                    self._charge("ram", (ik[0], rb.model, tenant, rb.scope_tag), rb.charged)
                used = self._pos_used.get(pos, rb.last_used)
                self._lru_del(pos, rb.tenant)
                if rb.state == _SAVING:
                    self._unsaving(rb.tenant)
                    self._saving[tenant] = self._saving.get(tenant, 0) + 1
                rb.tenant = tenant
                self._lru_add(pos, tenant, used)
            rec = self._nv.get(ik)
            if rec is None or rec.tenant == tenant:
                continue
            self._charge("nvme", (ik[0], rec.model, rec.tenant, rec.scope_tag), -rec.nbytes)
            for seg, nb in rec.segments.items():
                self._seg_charge((ik[0], seg), rec, rec.tenant, -nb)
                self._seg_charge((ik[0], seg), rec, tenant, nb)
            rec.tenant = tenant
            self._charge("nvme", (ik[0], rec.model, rec.tenant, rec.scope_tag), rec.nbytes)
            self._tenant_dirty[ik] = tenant  # the manifest learns it outside the lock

    def _persist_tenants(self) -> None:
        """Write pending tenant changes to the manifests, one transaction each (no lock).

        Only namespaces whose NVMe is on are written; the others keep their changes
        pending until it is on again.
        """
        with self._persist_lock:
            batches: dict[bytes, tuple[Manifest, list[tuple[bytes, bytes]]]] = {}
            with self._cond:
                for ik, tenant in list(self._tenant_dirty.items()):
                    m = self._manifests.get(ik[0])
                    if m is None or not self._nvme_on(ik[0]):
                        continue
                    del self._tenant_dirty[ik]
                    batches.setdefault(ik[0], (m, []))[1].append((ik[1], tenant))
            for ns_id, (m, changes) in batches.items():
                try:
                    m.set_tenants(ns_id, changes)
                except sqlite3.Error as exc:
                    with self._cond:
                        self._nvme_off(exc)
                    return

    def _tenant(self, tenant: bytes) -> _TenantStats:
        t = self._tenants.get(tenant)
        if t is None:
            t = self._tenants[tenant] = _TenantStats()
        return t

    def _count_eviction(self, tier: Tier, victim: bytes, model: bytes) -> None:
        """One block of ``victim`` in ``model`` left ``tier`` to make room for
        ``self._evict_for``. Counted per tenant and per (model, tenant)."""
        taker = self._evict_for if self._evict_for is not None else b""
        pair = self._pair_stats.get((model, victim))
        if pair is None:
            pair = self._pair_stats[(model, victim)] = _TenantStats()
            while len(self._pair_stats) > PAIR_STATS_MAX:
                self._pair_stats.popitem(last=False)
        else:
            self._pair_stats.move_to_end((model, victim))
        for t in (self._tenant(victim), pair):
            if tier == "ram":
                t.ram_evictions += 1
            else:
                t.nvme_evictions += 1
            t.evicted_by[taker] = t.evicted_by.get(taker, 0) + 1

    def _attribution_stats(self) -> dict[str, Any]:
        """``dead_bytes``, ``by_tenant``, ``by_model`` and ``by_namespace`` for ``stats``."""
        dead = {"ram": 0, "nvme": 0}
        by_tenant: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        by_ns: dict[str, dict[str, Any]] = {}

        def tenant_row(tenant: bytes) -> dict[str, Any]:
            row = by_tenant.get(tenant.hex())
            if row is None:
                row = by_tenant[tenant.hex()] = {
                    "ram_bytes": 0, "nvme_bytes": 0, "ram_evictions": 0,
                    "nvme_evictions": 0, "evicted_by": {}, "ram_tokens": 0, "nvme_tokens": 0,
                    "ram_share": 0.0, "nvme_share": 0.0,
                }  # fmt: skip
            return row

        weights = self._model_weights()
        usage = {tier: self._usage(tier) for tier in _TIERS}
        part = {tier: parts(self._capacity(tier), weights) for tier in _TIERS}
        share = {tier: shares(usage[tier], part[tier], self._weight) for tier in _TIERS}

        def model_row(model: bytes) -> dict[str, Any]:
            row = by_model.get(model.hex())
            if row is None:
                row = by_model[model.hex()] = {
                    "owners": set(), "last_owner": self._last_owner.get(model, ""),
                    "ram_bytes": 0, "nvme_bytes": 0,
                    "weight": weights.get(model, DEFAULT_OWNER_WEIGHT),
                    "ram_part": part["ram"].get(model, 0.0),
                    "nvme_part": part["nvme"].get(model, 0.0),
                    "tenants": {},
                }  # fmt: skip
            return row

        for ns_id, info in self._ns.items():
            by_ns[ns_id.hex()] = {
                "model": info.model.hex(), "role": info.role, "rank": info.rank,
                "owners": dict(info.owners), "live": info.live,
                "ram_bytes": 0, "nvme_bytes": 0,
            }  # fmt: skip
            model_row(info.model)["owners"].update(info.owners)
        for tier in _TIERS:
            for (ns_id, model, tenant, tag), n in self._tier_acct(tier).items():
                ns_row = by_ns.get(ns_id.hex())
                if ns_row is not None:
                    ns_row[f"{tier}_bytes"] += n
                if self._dead(ns_id, tag):
                    dead[tier] += n
                    continue
                tenant_row(tenant)[f"{tier}_bytes"] += n
                model_row(model)[f"{tier}_bytes"] += n
        pairs = set(usage["ram"]) | set(usage["nvme"]) | set(self._pair_stats)
        for model, tenant in sorted(pairs):
            pair = (model, tenant)
            t_row = tenant_row(tenant)
            m_row: dict[str, Any] = {}
            model_row(model)["tenants"][tenant.hex()] = m_row
            for tier in _TIERS:
                got = share[tier].get(pair, 0.0)
                m_row[f"{tier}_bytes"] = usage[tier].get(pair, 0)
                m_row[f"{tier}_share"] = got
                t_row[f"{tier}_share"] += got
            ps = self._pair_stats.get(pair)
            m_row["ram_evictions"] = ps.ram_evictions if ps is not None else 0
            m_row["nvme_evictions"] = ps.nvme_evictions if ps is not None else 0
            m_row["evicted_by"] = (
                {k.hex(): v for k, v in ps.evicted_by.items()} if ps is not None else {}
            )
        for tenant, t in self._tenants.items():
            row = tenant_row(tenant)
            row["ram_evictions"] = t.ram_evictions
            row["nvme_evictions"] = t.nvme_evictions
            row["evicted_by"] = {k.hex(): v for k, v in t.evicted_by.items()}
            row["ram_tokens"] = t.ram_tokens
            row["nvme_tokens"] = t.nvme_tokens
        for row in by_model.values():
            row["owners"] = sorted(row["owners"])
        return {
            "dead_bytes": dead,
            "by_tenant": by_tenant,
            "by_model": by_model,
            "by_namespace": by_ns,
        }


def _listed(entries: Iterable[tuple[int, _Pos]]) -> Iterator[_Pos]:
    return (pos for _, pos in entries)


def _sorted_del(lst: list[tuple[int, _Pos]], e: tuple[int, _Pos]) -> None:
    i = bisect.bisect_left(lst, e)
    if i < len(lst) and lst[i] == e:
        del lst[i]


def _ns_from_dirname(name: str) -> bytes | None:
    if len(name) != 64:
        return None
    try:
        return bytes.fromhex(name)
    except ValueError:
        return None
