"""Benchmark: the reuse counter work of ``end_load(used=True)`` (a script, not a test).

10,000 single-block positions over 50 tenants fill a RAM-only store. Then 100,000 loads
of one random position each run ``begin_load`` with a request tag (2,000 distinct tags,
reused, so the per-tag dedupe takes both paths; a quarter of the loads come from a
tenant other than the saver, so the holder sketches fill and evict) and
``end_load(used=True)``.

It prints p50 and p99 of the counter work alone (``ReuseBook.reloaded`` for the
handle's positions, timed inside ``end_load`` with the store's lock held) and of the
whole ``end_load``, for reference.

Run it alone on the build host: ``python tests/unit/lmw_kv/bench_reuse_counters.py``.
Gate G18 checks 50 us p99 for the counter work; nothing here asserts a number. The
file name keeps it out of pytest's collection (``test_*.py`` only).
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
from pathlib import Path
from typing import Any

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "lmw_kv")]

from lmw_kv.keys import Namespace  # noqa: E402
from lmw_kv.reuse import ReuseBook  # noqa: E402
from lmw_kv.scopes import model_tag  # noqa: E402
from lmw_kv.store import Store, StoreConfig  # noqa: E402

POSITIONS = 10_000
LOADS = 100_000
TENANTS = 50
TAGS = 2_000
ROOT = b"\x00" * 32


def _ns() -> Namespace:
    return Namespace(
        model_repo="org/model", revision_sha="r", tokenizer_sha="tok", kv_dtype="fp8",
        block_size=16, attention_layout="hnd", kv_groups=("full",), tp=1, pp=1,
        engine_family="vllm", layout_version=1,
    )  # fmt: skip


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _show(what: str, xs: list[float]) -> None:
    print(
        f"{what}: p50 {_pct(xs, 0.5) * 1e6:.2f} us, p99 {_pct(xs, 0.99) * 1e6:.2f} us, "
        f"mean {statistics.fmean(xs) * 1e6:.2f} us (n={len(xs)})"
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="lmw-bench-reuse-"))
    cfg = StoreConfig(
        ram_path=tmp / "ram.pool", ram_bytes=POSITIONS * mmap.PAGESIZE, nvme_dir=None,
        nvme_cap_bytes=0,
    )  # fmt: skip
    s = Store(cfg)
    work: list[float] = []
    real = ReuseBook.reloaded

    def timed(self: ReuseBook, *a: Any, **kw: Any) -> None:
        t0 = time.perf_counter()
        real(self, *a, **kw)
        work[-1] += time.perf_counter() - t0

    try:
        ns_id = s.register_namespace(_ns(), {"full": 1}, owner="a", model=model_tag("m", "r"))
        rng = random.Random(1)
        tenants = [hashlib.sha256(b"t%d" % i).digest()[:16] for i in range(TENANTS)]
        keys: list[tuple[bytes, bytes]] = []
        for i in range(POSITIONS):
            key = hashlib.sha256(b"k%d" % i).digest()
            saver = tenants[i % TENANTS]
            assert s.begin_save(ns_id, key, ROOT, b"", tenant=saver) is not None
            s.commit_save(ns_id, key)
            keys.append((key, saver))
        assert s.stats()["ram_used"] == POSITIONS * mmap.PAGESIZE
        ReuseBook.reloaded = timed  # type: ignore[method-assign]
        whole: list[float] = []
        for _ in range(LOADS):
            key, saver = rng.choice(keys)
            loader = saver if rng.random() < 0.75 else rng.choice(tenants)
            h = s.begin_load(ns_id, [key], tenant=loader, tag=f"req-{rng.randrange(TAGS)}")
            assert len(h.slots) == 1
            work.append(0.0)
            t0 = time.perf_counter()
            s.end_load(h.id, used=True)
            whole.append(time.perf_counter() - t0)
        reloads = sum(c.reloads for c in s._reuse.rows.values())
        print(f"{POSITIONS} positions, {LOADS} end_load(used=True), {reloads} reloads counted")
        _show("counter work (ReuseBook.reloaded)", work)
        _show("whole end_load", whole)
    finally:
        ReuseBook.reloaded = real  # type: ignore[method-assign]
        s.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
