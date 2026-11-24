"""Promotions from NVMe run in worker threads, finished in order (Task 13).

``begin_load`` starts each key's load in order under the store's rules, hands the
promotions (read, CRC, GCM decrypt straight into the slot) to ``PROMOTE_THREADS``
workers and finishes them in key order. A miss stops the load where the serial loop
stopped; loads started after it are undone (out of RAM, their records not judged, pins
released). A worker exception is a miss and drops nothing; nothing stays PROMOTING.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from lmw_kv.store import _PROMOTING, Store
from tests.unit.lmw_kv.test_store import (
    Conn,
    _cfg,
    _chain,
    _open,
    _save,
    opened,  # noqa: F401  (fixture)
)


def _nvme_chain(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    n: int,
) -> tuple[Store, bytes, Conn, list[bytes]]:
    """A fresh store whose chain of ``n`` blocks is on NVMe only."""
    cfg = _cfg(tmp_path, slots=n + 4)
    s, ns_id, c = _open(opened, cfg)
    chain = _chain([f"k{i}" for i in range(n)])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    for t in reversed(opened):
        t.close()
    opened.clear()
    s2, ns2, c2 = _open(opened, cfg)
    assert ns2 == ns_id
    return s2, ns_id, c2, [k for k, _ in chain]


def _damage(s: Store, ns_id: bytes, key: bytes) -> None:
    with s._cond:
        entries = s._nvme_entries(s._ns[ns_id], key)
    assert entries is not None
    loc = entries[0].loc
    seg = s._ns_dir(ns_id) / f"{loc.segment:06d}.seg"
    b = bytearray(seg.read_bytes())
    b[loc.offset + loc.length - 30] ^= 0x01
    seg.write_bytes(bytes(b))


def _no_promoting(s: Store) -> bool:
    return all(b.state != _PROMOTING for b in s._ram.values())


def test_promotions_run_in_workers_and_return_in_key_order(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s, ns_id, c, keys = _nvme_chain(tmp_path, opened, 6)
    threads: list[str] = []
    real = s._promote
    first_may_end = threading.Event()

    def promote(pool: Any, slot: int, info: Any, key: bytes, entries: Any) -> bool:
        threads.append(threading.current_thread().name)
        if key == keys[0]:
            assert first_may_end.wait(10)  # the first finishes last
        elif len(threads) == len(keys):
            first_may_end.set()
        return real(pool, slot, info, key, entries)

    monkeypatch.setattr(s, "_promote", promote)
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 6 and h.nvme_blocks == 6
    assert all(c.matches(k, off) for k, off in zip(keys, h.slots, strict=True))
    assert all(t.startswith("lmw-kv-promote") for t in threads)
    assert [ik[1] for ik, _ in s._loads[h.id]] == keys  # pins in key order
    s.end_load(h.id)
    assert not any(b.pins for b in s._ram.values())


def test_a_damaged_record_stops_the_load_and_undoes_the_rest(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, c, keys = _nvme_chain(tmp_path, opened, 5)
    _damage(s, ns_id, keys[2])
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 2  # stops at the damaged block, as the serial loop did
    assert all(c.matches(k, off) for k, off in zip(keys[:2], h.slots, strict=True))
    assert _no_promoting(s)
    assert set(s._ram) == {(ns_id, k) for k in keys[:2]}  # the rest left RAM
    s.end_load(h.id)
    assert not any(b.pins for b in s._ram.values())
    # The damaged block left the index; the two after it were never judged: still there.
    assert s.lookup(ns_id, keys[:2]) == 2
    assert s.lookup(ns_id, [keys[2]]) == 0
    with s._cond:
        info = s._ns[ns_id]
        assert s._nvme_entries(info, keys[3]) is not None
        assert s._nvme_entries(info, keys[4]) is not None


def test_a_worker_exception_is_a_miss_that_drops_nothing(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s, ns_id, c, keys = _nvme_chain(tmp_path, opened, 5)
    real = s._promote

    def promote(pool: Any, slot: int, info: Any, key: bytes, entries: Any) -> bool:
        if key == keys[1]:
            raise OSError("worker lost the disk")
        return real(pool, slot, info, key, entries)

    monkeypatch.setattr(s, "_promote", promote)
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 1
    assert c.matches(keys[0], h.slots[0])
    assert _no_promoting(s) and set(s._ram) == {(ns_id, keys[0])}
    s.end_load(h.id)
    assert s.lookup(ns_id, keys) == 5  # a transient error drops no NVMe copy


def test_a_base_exception_in_a_worker_leaves_nothing_behind(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s, ns_id, _, keys = _nvme_chain(tmp_path, opened, 4)
    real = s._promote

    class Abort(BaseException):
        pass

    def promote(pool: Any, slot: int, info: Any, key: bytes, entries: Any) -> bool:
        if key == keys[2]:
            raise Abort
        return real(pool, slot, info, key, entries)

    monkeypatch.setattr(s, "_promote", promote)
    with pytest.raises(Abort):
        s.begin_load(ns_id, keys)
    assert _no_promoting(s)
    assert not s._loads and not any(b.pins for b in s._ram.values())
    monkeypatch.setattr(s, "_promote", real)
    h = s.begin_load(ns_id, keys)  # nothing stuck: a fresh load gets everything
    assert len(h.slots) == 4
    s.end_load(h.id)


def test_ram_hits_settle_the_started_promotions_first(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, c, keys = _nvme_chain(tmp_path, opened, 6)
    h0 = s.begin_load(ns_id, [keys[2]])  # one key: promoted inline, now in RAM
    s.end_load(h0.id)
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 6 and h.nvme_blocks == 5
    assert all(c.matches(k, off) for k, off in zip(keys, h.slots, strict=True))
    s.end_load(h.id)


def test_a_staging_load_in_parallel_leaves_ram_at_end_load(tmp_path: Path) -> None:
    from tests.unit.lmw_kv.test_store import GROUPS, _ns

    cfg = _cfg(tmp_path, slots=12)
    s = Store(cfg)
    try:
        ns_id = s.register_namespace(_ns(), GROUPS, owner="a", staging=True)
        c = Conn(cfg.ram_path, GROUPS)
        chain = _chain([f"s{i}" for i in range(6)])
        for key, parent in chain:
            assert _save(s, c, ns_id, key, parent)
        s.flush()
        assert s.stats()["ram_used"] == 0  # S4: on disk, out of RAM
        keys = [k for k, _ in chain]
        h = s.begin_load(ns_id, keys)
        assert len(h.slots) == 6 and h.nvme_blocks == 6
        assert all(c.matches(k, off) for k, off in zip(keys, h.slots, strict=True))
        s.end_load(h.id, used=True)
        assert s.stats()["ram_used"] == 0  # S6: promotions leave at end_load
        c.close()
    finally:
        s.close()


def test_close_stops_the_workers(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, _, keys = _nvme_chain(tmp_path, opened, 3)
    s.end_load(s.begin_load(ns_id, keys).id)
    workers = s._promoters
    assert workers is not None
    s.close()
    assert s._promoters is None
    with pytest.raises(RuntimeError):
        workers.submit(lambda: None)


def test_fair_ram_loads_finish_each_promotion_before_the_next_allocation(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``fair`` allocation reads the tenant's charges: the earlier loads must be done."""
    from lmw_kv.fair import Policy
    from lmw_kv.store import _VALID

    s, ns_id, c, keys = _nvme_chain(tmp_path, opened, 4)
    s.set_policy(Policy(mode="fair"))
    seen: list[bool] = []
    real = s._promote

    def promote(pool: Any, slot: int, info: Any, key: bytes, entries: Any) -> bool:
        i = keys.index(key)
        with s._cond:
            seen.append(all(s._ram[(ns_id, k)].state == _VALID for k in keys[:i]))
        return real(pool, slot, info, key, entries)

    monkeypatch.setattr(s, "_promote", promote)
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 4
    assert all(c.matches(k, off) for k, off in zip(keys, h.slots, strict=True))
    assert seen == [True] * 4
    s.end_load(h.id)


def test_a_promotion_that_cannot_be_submitted_is_a_miss(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workers shut down between the start of a load and its promotion (close)."""
    s, ns_id, c, keys = _nvme_chain(tmp_path, opened, 4)
    s.end_load(s.begin_load(ns_id, keys[:2]).id)  # makes the workers
    workers = s._promoters
    assert workers is not None
    real = workers.submit
    n = [0]

    def submit(*a: Any, **kw: Any) -> Any:
        n[0] += 1
        if n[0] == 2:
            raise RuntimeError("cannot schedule new futures after shutdown")
        return real(*a, **kw)

    monkeypatch.setattr(workers, "submit", submit)
    with s._cond:  # drop the RAM copies so every key promotes again
        for ik, b in list(s._ram.items()):
            s._remove(ik, b)
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 1 and c.matches(keys[0], h.slots[0])
    assert _no_promoting(s)
    s.end_load(h.id)
    assert s.lookup(ns_id, keys) == 4  # a failed submit drops nothing
