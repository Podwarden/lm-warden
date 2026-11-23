"""Benchmark: how long one ``fair`` victim choice takes (a script, not a test).

Two RAM cases, NVMe off:

- ``flat``: 10,000 single-block positions over 50 tenants and 2 models fill the pool,
  then 1,000 saves each need one victim.
- ``chains``: one long session of 3,000 positions saved root first, plus 10 short
  sessions of 100, fill a 4,000-slot pool; then 1,000 saves extend the short sessions.
  The long session is over its share, and only its last position is a leaf.

Each prints p50 and p99 of the victim choice alone (the store's lock held, as in
``_alloc``) and of the whole ``begin_save`` + ``commit_save``.

Two NVMe cases, ``segments``: 2,000 segment files of 8 blocks each and 30,000 of 2 (50
tenants, 2 models, a few heavy users), half of them written 8 days earlier and some of
those used again, so idle scoring takes its exact path for them. Each prints, lock held:
a full scoring pass over every segment (for reference; no reclaim does this), the
segment choice (``_choose_segment``: the cached ranking's bounded refresh plus the
winner's ``_plan_reclaim``), step 1's ``lru`` choice for reference (sorting the segments
by age plus the oldest's ``_plan_reclaim``), and the lock-held time of each fair reclaim
on the writer thread (``_reclaim_segment``; the lock is let go between reclaims).

Run it alone on the build host: ``python tests/unit/lmw_kv/bench_fair_victims.py``.
Gate G4 checks 1 ms p99 on the production node; nothing here asserts a number.
"""

from __future__ import annotations

import hashlib
import mmap
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "lmw_kv")]

from lmw_kv import store as store_mod  # noqa: E402
from lmw_kv.fair import Policy, TenantPolicy  # noqa: E402
from lmw_kv.format import HEADER_LEN, record_size  # noqa: E402
from lmw_kv.keys import Namespace  # noqa: E402
from lmw_kv.scopes import model_tag  # noqa: E402
from lmw_kv.store import Store, StoreConfig  # noqa: E402

CHOICES = 1_000
ROOT = b"\x00" * 32


def _ns(rev: str) -> Namespace:
    return Namespace(
        model_repo="org/model", revision_sha=rev, tokenizer_sha="tok", kv_dtype="fp8",
        block_size=16, attention_layout="hnd", kv_groups=("full",), tp=1, pp=1,
        engine_family="vllm", layout_version=1,
    )  # fmt: skip


def _tenant(i: int) -> bytes:
    return hashlib.sha256(b"t%d" % i).digest()[:16]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _store(tmp: Path, slots: int, policy: Policy) -> Store:
    cfg = StoreConfig(
        ram_path=tmp / "ram.pool", ram_bytes=slots * mmap.PAGESIZE, nvme_dir=None,
        nvme_cap_bytes=0,
    )  # fmt: skip
    s = Store(cfg)
    s.set_policy(policy)
    return s


def _measure(name: str, s: Store, save: Callable[[], None]) -> None:
    choose: list[float] = []
    with s._cond:  # the victim choice alone, without dropping it
        for _ in range(CHOICES):
            t0 = time.perf_counter()
            v = s._dead_victim() or s._idle_victim() or s._over_victim()[0]
            v = v or s._oldest_victim()
            choose.append(time.perf_counter() - t0)
            assert v is not None
    whole: list[float] = []
    for _ in range(CHOICES):
        t0 = time.perf_counter()
        save()
        whole.append(time.perf_counter() - t0)
    for what, xs in (("victim choice", choose), ("save with eviction", whole)):
        print(
            f"{name} {what}: p50 {_pct(xs, 0.5) * 1e3:.3f} ms, p99 {_pct(xs, 0.99) * 1e3:.3f} "
            f"ms, mean {statistics.fmean(xs) * 1e3:.3f} ms (n={len(xs)})"
        )


def flat(tmp: Path) -> None:
    positions, n_tenants = 10_000, 50
    rng = random.Random(1)
    tenants = [_tenant(i) for i in range(n_tenants)]
    # Mixed priorities, and a few heavy users that end up over their share.
    policy = Policy(
        mode="fair",
        owner_weights={"a": 1, "b": 2},
        tenants={t: TenantPolicy(weight=1 + i % 10) for i, t in enumerate(tenants)},
    )
    s = _store(tmp, positions, policy)
    try:
        nss = [
            s.register_namespace(_ns("a"), {"full": 1}, owner="a", model=model_tag("m", "a")),
            s.register_namespace(_ns("b"), {"full": 1}, owner="b", model=model_tag("m", "b")),
        ]
        weights = [8 if i < 5 else 1 for i in range(n_tenants)]
        n = 0

        def save() -> None:
            nonlocal n
            n += 1
            ns_id = nss[n % 2]
            key = hashlib.sha256(b"k%d" % n).digest()
            tenant = rng.choices(tenants, weights)[0]
            if s.begin_save(ns_id, key, ROOT, b"", tenant=tenant) is not None:
                s.commit_save(ns_id, key)

        for _ in range(positions):
            save()
        assert s.stats()["ram_used"] == positions * mmap.PAGESIZE
        _measure("flat", s, save)
    finally:
        s.close()


def chains(tmp: Path) -> None:
    long_len, short, short_len = 3_000, 10, 100
    slots = long_len + short * short_len
    s = _store(tmp, slots, Policy(mode="fair"))
    try:
        ns_id = s.register_namespace(_ns("c"), {"full": 1}, owner="a", model=model_tag("m", "c"))
        heads: dict[int, bytes] = {}
        n = 0

        def extend(i: int) -> None:
            nonlocal n
            n += 1
            key = hashlib.sha256(b"c%d" % n).digest()
            parent = heads.get(i, ROOT)
            if s.begin_save(ns_id, key, parent, b"", tenant=_tenant(i)) is not None:
                s.commit_save(ns_id, key)
                heads[i] = key

        for _ in range(long_len):
            extend(0)  # the long session, root first
        for _ in range(short_len):
            for i in range(1, short + 1):
                extend(i)
        assert s.stats()["ram_used"] == slots * mmap.PAGESIZE
        turn = 0

        def save() -> None:
            nonlocal turn
            turn += 1
            extend(1 + turn % short)

        _measure("chains", s, save)
    finally:
        s.close()


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def segments(tmp: Path, n_segs: int, per_seg: int) -> None:
    n_tenants = 50
    name = f"segments-{n_segs}"
    groups = {"full": 512}
    block = sum(record_size(g, n) for g, n in groups.items())
    store_mod.SEGMENT_BYTES = HEADER_LEN + per_seg * block
    seg_file = store_mod.SEGMENT_BYTES
    clock = _Clock()
    rng = random.Random(3)
    tenants = [_tenant(i) for i in range(n_tenants)]
    policy = Policy(
        mode="fair",
        owner_weights={"a": 1, "b": 2},
        tenants={t: TenantPolicy(weight=1 + i % 10) for i, t in enumerate(tenants)},
    )
    cfg = StoreConfig(
        ram_path=tmp / "ram.pool", ram_bytes=512 * mmap.PAGESIZE, nvme_dir=tmp / "nvme",
        nvme_cap_bytes=n_segs * seg_file, clock=clock,
    )  # fmt: skip
    s = Store(cfg)
    s.set_policy(policy)
    try:
        nss = [
            s.register_namespace(_ns(r), groups, owner=r, model=model_tag("m", r))
            for r in ("a", "b")
        ]
        weights = [8 if i < 5 else 1 for i in range(n_tenants)]
        n = 0
        old: list[tuple[bytes, bytes]] = []

        def write(count: int) -> None:
            nonlocal n
            for i in range(count):
                n += 1
                ns_id = nss[(n // per_seg) % 2]  # each namespace fills whole segments
                key = hashlib.sha256(b"s%d" % n).digest()
                tenant = rng.choices(tenants, weights)[0]
                if s.begin_save(ns_id, key, ROOT, b"", tenant=tenant) is not None:
                    s.commit_save(ns_id, key)
                    old.append((ns_id, key))
                if i % 256 == 255:
                    s.flush()
            s.flush()

        total = n_segs * per_seg - per_seg  # leave one segment of room
        write(total // 2)
        clock.now += 8 * 86_400.0
        for ns_id, key in rng.sample(old, 400):  # used again: those segments are mixed
            h = s.begin_load(ns_id, [key])
            s.end_load(h.id)
        write(total - total // 2)
        need = block + HEADER_LEN
        scoring: list[float] = []
        times: list[float] = []
        lru: list[float] = []
        with s._cond:
            print(f"{name}: {len(s._segs)} files, {len(s._nv)} records on NVMe")
            for _ in range(CHOICES):
                t0 = time.perf_counter()
                ctx = s._score_ctx()
                for ns_id, seg in s._segs:
                    s._segment_score(ns_id, seg, ctx=ctx)
                scoring.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                got = s._choose_segment(b"", need)
                times.append(time.perf_counter() - t0)
                assert got is not None
                t0 = time.perf_counter()
                for (ns_id, seg), _ in sorted(s._segs.items(), key=lambda kv: (kv[1][0], kv[0])):
                    if s._plan_reclaim(ns_id, seg) is not None:
                        break
                lru.append(time.perf_counter() - t0)
        whole: list[float] = []
        real = s._reclaim_segment

        def timed(for_tenant: bytes = b"", need: int = 0) -> bool:
            t0 = time.perf_counter()
            try:
                return real(for_tenant, need)
            finally:
                whole.append(time.perf_counter() - t0)

        s._reclaim_segment = timed  # type: ignore[method-assign]
        while len(whole) < CHOICES:
            write(per_seg)
        for what, xs in (
            ("full scoring pass", scoring),
            ("segment choice", times),
            ("lru choice (step 1)", lru),
            ("fair reclaim, lock held", whole),
        ):
            print(
                f"{name} {what}: p50 {_pct(xs, 0.5) * 1e3:.3f} ms, p99 "
                f"{_pct(xs, 0.99) * 1e3:.3f} ms, mean {statistics.fmean(xs) * 1e3:.3f} ms "
                f"(n={len(xs)})"
            )
    finally:
        s.close()


def main() -> None:
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    cases: list[Callable[[Path], None]] = [
        flat,
        chains,
        lambda tmp: segments(tmp, 2_000, 8),
        lambda tmp: segments(tmp, 30_000, 2),
    ]
    for case in cases:
        tmp = Path(tempfile.mkdtemp(prefix="lmw-bench-", dir=base))
        try:
            case(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
