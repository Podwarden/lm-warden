from __future__ import annotations

import errno
import hashlib
import logging
import mmap
import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import store as store_mod
from lmw_kv.format import HEADER_LEN, SegmentWriter, record_size
from lmw_kv.keys import Namespace
from lmw_kv.store import Store, StoreConfig

PAGE = mmap.PAGESIZE
GROUPS = {"full": 1000, "gdn": 300}
SLOT = 2 * PAGE  # each group rounded up to a page, packed in one slot
ROOT = b"\x00" * 32
SCOPE = b""


def _ns(revision: str = "r1", groups: Sequence[str] = ("full", "gdn")) -> Namespace:
    return Namespace(
        model_repo="org/model",
        revision_sha=revision,
        tokenizer_sha="tok",
        kv_dtype="fp8",
        block_size=16,
        attention_layout="hnd",
        kv_groups=tuple(groups),
        tp=1,
        pp=1,
        engine_family="vllm",
        layout_version=1,
    )


def _k(name: str) -> bytes:
    return hashlib.sha256(name.encode()).digest()


def _payload(key: bytes, group: str, size: int) -> bytes:
    seed = hashlib.sha256(key + group.encode()).digest()
    return (seed * (size // len(seed) + 1))[:size]


def _cfg(tmp_path: Path, slots: int = 8, nvme: bool = True, cap: int = 1 << 30) -> StoreConfig:
    return StoreConfig(
        ram_path=tmp_path / "ram.pool",
        ram_bytes=slots * SLOT,
        nvme_dir=tmp_path / "nvme" if nvme else None,
        nvme_cap_bytes=cap,
    )


class Conn:
    """Plays the connector: maps the RAM file and moves bytes by offset."""

    def __init__(self, path: Path, groups: dict[str, int]) -> None:
        self._groups = groups
        self._fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, os.path.getsize(path), mmap.MAP_SHARED)

    def write(self, key: bytes, offsets: dict[str, int]) -> None:
        for g, off in offsets.items():
            data = _payload(key, g, self._groups[g])
            self._mm[off : off + len(data)] = data

    def read(self, offsets: dict[str, int]) -> dict[str, bytes]:
        return {g: bytes(self._mm[off : off + self._groups[g]]) for g, off in offsets.items()}

    def matches(self, key: bytes, offsets: dict[str, int]) -> bool:
        got = self.read(offsets)
        return got == {g: _payload(key, g, n) for g, n in self._groups.items()}

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)


@pytest.fixture
def opened() -> Iterator[list[Store | Conn]]:
    things: list[Store | Conn] = []
    yield things
    for t in reversed(things):
        t.close()


def _open(
    opened: list[Store | Conn], cfg: StoreConfig, groups: dict[str, int] = GROUPS
) -> tuple[Store, bytes, Conn]:
    s = Store(cfg)
    opened.append(s)
    ns_id = s.register_namespace(_ns(groups=tuple(groups)), groups)
    c = Conn(cfg.ram_path, groups)
    opened.append(c)
    return s, ns_id, c


def _save(s: Store, c: Conn, ns_id: bytes, key: bytes, parent: bytes) -> bool:
    offsets = s.begin_save(ns_id, key, parent, SCOPE)
    if offsets is None:
        return False
    c.write(key, offsets)
    s.commit_save(ns_id, key)
    return True


def _disk(cfg: StoreConfig) -> int:
    """Bytes of every segment file on disk: what ``nvme_used`` and the cap count."""
    assert cfg.nvme_dir is not None
    return sum(p.stat().st_size for p in cfg.nvme_dir.rglob("*.seg"))


#: Bytes one block of GROUPS takes in a segment (records only, no header).
BLOCK_ON_DISK = sum(record_size(g, n) for g, n in GROUPS.items())


def _chain(names: Sequence[str]) -> list[tuple[bytes, bytes]]:
    out = []
    parent = ROOT
    for n in names:
        out.append((_k(n), parent))
        parent = _k(n)
    return out


def test_save_then_lookup_then_load_round_trips_bytes(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, nvme=False))
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    keys = [k for k, _ in chain]
    assert s.lookup(ns_id, keys) == 2
    assert s.lookup(ns_id, [keys[0], _k("missing"), keys[1]]) == 1
    assert s.lookup(ns_id, [_k("missing")]) == 0
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 2
    assert h.nvme_blocks == 0  # both were in RAM
    for key, offsets in zip(keys, h.slots, strict=True):
        assert set(offsets) == set(GROUPS)
        assert all(off % PAGE == 0 for off in offsets.values())
        assert c.matches(key, offsets)
    s.end_load(h.id)
    st = s.stats()
    assert st["blocks"] == 2
    assert st["ram_used"] == 2 * SLOT
    assert st["ram_bytes"] == 8 * SLOT
    assert st["nvme_enabled"] is False


def test_write_through_reaches_nvme_and_survives_a_new_store(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id, c = _open(opened, cfg)
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    assert s.stats()["nvme_used"] == _disk(cfg) == HEADER_LEN + 2 * BLOCK_ON_DISK
    c.close()
    s.close()
    opened.clear()

    s2, ns2, c2 = _open(opened, cfg)
    assert ns2 == ns_id
    keys = [k for k, _ in chain]
    assert s2.stats()["ram_used"] == 0
    assert s2.lookup(ns_id, keys) == 2
    assert s2.begin_save(ns_id, keys[0], ROOT, SCOPE) is None  # already stored on NVMe
    h = s2.begin_load(ns_id, keys)
    assert len(h.slots) == 2
    assert h.nvme_blocks == 2  # both promoted from NVMe by this call
    for key, offsets in zip(keys, h.slots, strict=True):
        assert c2.matches(key, offsets)
    s2.end_load(h.id)
    again = s2.begin_load(ns_id, keys)
    assert again.nvme_blocks == 0  # now in RAM
    s2.end_load(again.id)
    assert s2.stats()["ram_used"] == 2 * SLOT
    assert s2.stats()["blocks"] == 2


def test_namespace_change_makes_blocks_invisible(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id, c = _open(opened, cfg)
    keys = [k for k, _ in _chain(["a", "b"])]
    for key, parent in _chain(["a", "b"]):
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    new_id = s.register_namespace(_ns(revision="r2"), GROUPS)
    assert new_id != ns_id
    assert s.lookup(new_id, keys) == 0
    assert s.begin_load(new_id, keys).slots == []
    c.close()
    s.close()
    opened.clear()

    s2 = Store(cfg)
    opened.append(s2)
    new_id = s2.register_namespace(_ns(revision="r2"), GROUPS)
    assert s2.lookup(new_id, keys) == 0
    assert s2.begin_load(new_id, keys).slots == []
    assert s2.lookup(ns_id, keys) == 0  # unregistered namespace: a miss, never a load


def test_ram_full_evicts_lru_leaves_only(tmp_path: Path, opened: list[Store | Conn]) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=3))
    chain = _chain(["c0", "c1", "c2", "c3", "c4"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
        s.flush()  # the write-through lands, so the RAM copy becomes evictable
    keys = [k for k, _ in chain]
    st = s.stats()
    assert st["ram_used"] == 3 * SLOT
    assert st["evictions"] == 2
    assert st["blocks"] == 5
    assert s.lookup(ns_id, keys) == 5  # evicted blocks remain on NVMe
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 3  # RAM holds 3: the rest would need a pinned slot
    for key, offsets in zip(keys[:3], h.slots, strict=True):
        assert c.matches(key, offsets)
    s.end_load(h.id)
    assert s.lookup(ns_id, keys) == 5


def test_ram_only_eviction_takes_the_oldest_leaf_and_keeps_the_prefix_closed(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=3, nvme=False))
    a0, a1, b0, c0 = _k("a0"), _k("a1"), _k("b0"), _k("c0")
    assert _save(s, c, ns_id, a0, ROOT)
    assert _save(s, c, ns_id, a1, a0)
    assert _save(s, c, ns_id, b0, ROOT)
    assert _save(s, c, ns_id, c0, ROOT)  # a0 is oldest but not a leaf: a1 goes
    assert s.lookup(ns_id, [a0, a1]) == 1
    assert s.lookup(ns_id, [b0]) == 1
    assert s.lookup(ns_id, [c0]) == 1
    assert s.stats()["evictions"] == 1


def test_ram_only_chain_never_evicts_its_own_parent(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=3, nvme=False))
    chain = _chain(["c0", "c1", "c2", "c3"])
    for key, parent in chain[:3]:
        assert _save(s, c, ns_id, key, parent)
    key, parent = chain[3]
    assert s.begin_save(ns_id, key, parent, SCOPE) is None
    assert s.lookup(ns_id, [k for k, _ in chain]) == 3


def test_pinned_blocks_are_never_evicted_during_load(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=2, nvme=False))
    a, b, n = _k("a"), _k("b"), _k("new")
    assert _save(s, c, ns_id, a, ROOT)
    assert _save(s, c, ns_id, b, ROOT)
    ha = s.begin_load(ns_id, [a])
    hb = s.begin_load(ns_id, [b])
    assert s.begin_save(ns_id, n, ROOT, SCOPE) is None
    assert c.matches(a, ha.slots[0]) and c.matches(b, hb.slots[0])
    s.end_load(ha.id)
    assert _save(s, c, ns_id, n, ROOT)  # a was unpinned and is the LRU leaf
    assert c.matches(b, hb.slots[0])
    assert s.lookup(ns_id, [b]) == 1
    assert s.lookup(ns_id, [a]) == 0
    s.end_load(hb.id)


@pytest.mark.parametrize("failure", ["readonly", "missing", "enospc"])
def test_nvme_full_or_unwritable_turns_tier_off_and_ram_still_works(
    tmp_path: Path,
    opened: list[Store | Conn],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    cfg = _cfg(tmp_path, slots=3)
    if failure == "readonly":
        if os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        assert cfg.nvme_dir is not None
        cfg.nvme_dir.mkdir()
        cfg.nvme_dir.chmod(0o500)
    elif failure == "missing":
        (tmp_path / "file").write_bytes(b"")
        cfg.nvme_dir = tmp_path / "file" / "nvme"  # parent is a file: cannot be created
    else:

        def full(self: SegmentWriter, block_key: bytes, group: str, payload: bytes) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(SegmentWriter, "append", full)
    caplog.set_level(logging.WARNING, logger="lmw_kv.store")
    try:
        s, ns_id, c = _open(opened, cfg)
        chain = _chain(["a", "b", "c", "d"])
        for key, parent in chain[:2]:
            assert _save(s, c, ns_id, key, parent)
        s.flush()
        assert s.stats()["nvme_enabled"] is False
        warnings = [r for r in caplog.records if r.name == "lmw_kv.store"]
        assert len(warnings) == 1
        keys = [k for k, _ in chain]
        assert s.lookup(ns_id, keys) == 2
        h = s.begin_load(ns_id, keys[:2])
        assert all(c.matches(k, o) for k, o in zip(keys[:2], h.slots, strict=True))
        s.end_load(h.id)
        # RAM keeps working, and blocks that never reached NVMe are evictable as leaves
        assert _save(s, c, ns_id, *chain[2])
        assert s.begin_save(ns_id, *chain[3], SCOPE) is None
        x = _k("x")
        assert _save(s, c, ns_id, x, ROOT)  # evicts the leaf "c", never "a" or "b"
        assert s.lookup(ns_id, keys) == 2
        s.flush()
        assert len([r for r in caplog.records if r.name == "lmw_kv.store"]) == 1
    finally:
        if failure == "readonly" and cfg.nvme_dir is not None:
            cfg.nvme_dir.chmod(0o700)


def test_concurrent_same_key_saves_store_once(tmp_path: Path, opened: list[Store | Conn]) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=64))
    got_lock = threading.Lock()

    def save(key: bytes, barrier: threading.Barrier, got: list[dict[str, int] | None]) -> None:
        barrier.wait()
        r = s.begin_save(ns_id, key, ROOT, SCOPE)
        with got_lock:
            got.append(r)

    rounds = 40
    for i in range(rounds):
        key = _k(f"same-{i}")
        barrier = threading.Barrier(2)
        got: list[dict[str, int] | None] = []
        threads = [threading.Thread(target=save, args=(key, barrier, got)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        winners = [r for r in got if r is not None]
        assert len(winners) == 1
        assert s.lookup(ns_id, [key]) == 0  # not visible until committed
        c.write(key, winners[0])
        s.commit_save(ns_id, key)
        assert s.lookup(ns_id, [key]) == 1
        assert s.begin_save(ns_id, key, ROOT, SCOPE) is None
    s.flush()
    st = s.stats()
    assert st["blocks"] == rounds
    assert st["ram_used"] == rounds * SLOT
    assert st["nvme_used"] == HEADER_LEN + rounds * BLOCK_ON_DISK


def test_abort_save_frees_the_slot_and_the_key(tmp_path: Path, opened: list[Store | Conn]) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1, nvme=False))
    a = _k("a")
    assert s.begin_save(ns_id, a, ROOT, SCOPE) is not None
    s.abort_save(ns_id, a)
    assert s.stats()["ram_used"] == 0
    assert s.lookup(ns_id, [a]) == 0
    assert _save(s, c, ns_id, a, ROOT)


def test_hits_are_popped_once(tmp_path: Path, opened: list[Store | Conn]) -> None:
    s = Store(_cfg(tmp_path, nvme=False))
    opened.append(s)
    s.record_hits("req-1", 528, 1056, 0.25)
    s.record_hits("req-1", 528, 0, 0.5)
    assert s.pop_hits("req-1") == {"ram_tokens": 1056, "nvme_tokens": 1056, "load_s": 0.75}
    assert s.pop_hits("req-1") is None
    assert s.pop_hits("never") is None


def test_nvme_cap_reclaims_whole_segments_oldest_first(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8: the cap counts segment files on disk; over it the oldest segment goes whole."""
    monkeypatch.setattr(store_mod, "SEGMENT_BYTES", 1)  # one record per segment file
    seg_a, seg_b = (HEADER_LEN + record_size(g, n) for g, n in GROUPS.items())
    cfg = _cfg(tmp_path, cap=2 * (seg_a + seg_b) + HEADER_LEN)  # two blocks and a header
    s, ns_id, c = _open(opened, cfg)
    a, b, d = _k("a"), _k("b"), _k("d")
    for key in (a, b, d):
        assert _save(s, c, ns_id, key, ROOT)
        s.flush()
        assert _disk(cfg) <= cfg.nvme_cap_bytes
    st = s.stats()
    assert st["nvme_used"] == _disk(cfg) == 2 * (seg_a + seg_b)
    assert st["nvme_enabled"] is True
    assert cfg.nvme_dir is not None
    assert len(sorted(cfg.nvme_dir.rglob("*.seg"))) == 4  # a's two segments were deleted
    c.close()
    s.close()
    opened.clear()
    s2 = Store(cfg)
    opened.append(s2)
    s2.register_namespace(_ns(), GROUPS)
    assert s2.lookup(ns_id, [a]) == 0
    assert s2.lookup(ns_id, [b]) == 1
    assert s2.lookup(ns_id, [d]) == 1
    assert s2.stats()["nvme_used"] == _disk(cfg)  # counted from the files at start


def test_reclaiming_a_segment_drops_descendants_left_without_a_parent(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8: prefix closure. c0's segment goes; c0 is NVMe-only, so c1 and c2 go with it."""
    monkeypatch.setattr(store_mod, "SEGMENT_BYTES", 1)
    cfg = _cfg(tmp_path, cap=1 << 30)
    s, ns_id, c = _open(opened, cfg)
    chain = _chain(["c0", "c1", "c2"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    c.close()
    s.close()
    opened.clear()
    s2, _, c2 = _open(opened, cfg)  # all three on NVMe only now
    keys = [k for k, _ in chain]
    assert s2.lookup(ns_id, keys) == 3
    with s2._cond:
        assert s2._reclaim_oldest_segment()  # c0's first segment is the oldest
    assert s2.lookup(ns_id, keys) == 0
    assert s2.lookup(ns_id, keys[1:]) == 0  # no orphaned c1 / c2 left in the index
    assert s2.stats()["blocks"] == 0


def test_reclaiming_keeps_a_block_present_in_ram(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block with a valid RAM copy stays present when its segment goes: no descendant drops."""
    monkeypatch.setattr(store_mod, "SEGMENT_BYTES", 1)
    cfg = _cfg(tmp_path, cap=1 << 30)
    s, ns_id, c = _open(opened, cfg)
    chain = _chain(["c0", "c1"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    with s._cond:
        assert s._reclaim_oldest_segment()
    keys = [k for k, _ in chain]
    assert s.lookup(ns_id, keys) == 2  # c0 in RAM, c1 in RAM and on NVMe


def test_nvme_disk_use_stays_within_the_cap_under_scattered_eviction(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8: evicted blocks leave bytes in their segment; disk use must still stay capped.

    Blocks from many chains interleave in each segment, RAM is tiny (so loads promote
    and RAM evicts), and some blocks are loaded again (touched) while others age out.
    """
    monkeypatch.setattr(store_mod, "SEGMENT_BYTES", 4 * BLOCK_ON_DISK)  # a few blocks each
    cap = 12 * BLOCK_ON_DISK
    cfg = _cfg(tmp_path, slots=3, cap=cap)
    s, ns_id, c = _open(opened, cfg)
    chains = [_chain([f"x{i}-{j}" for j in range(4)]) for i in range(10)]
    for step in range(4):
        for i, chain in enumerate(chains):
            key, parent = chain[step]
            _save(s, c, ns_id, key, parent)
            s.flush()
            assert _disk(cfg) <= cap, (step, i)
            assert s.stats()["nvme_used"] == _disk(cfg)
            if i % 3 == 0:  # touch an older chain: its blocks are promoted and pinned briefly
                h = s.begin_load(ns_id, [k for k, _ in chains[(i + 5) % 10][: step + 1]])
                s.end_load(h.id)
    assert s.stats()["nvme_enabled"] is True  # never ENOSPC-like: room was made each time
    # every stored chain is still prefix closed: what lookup finds, loads
    for chain in chains:
        keys = [k for k, _ in chain]
        n = s.lookup(ns_id, keys)
        h = s.begin_load(ns_id, keys)
        assert len(h.slots) == n
        for key, offsets in zip(keys, h.slots, strict=False):
            assert c.matches(key, offsets)
        s.end_load(h.id)


def test_no_ram_tier_when_shm_is_too_small(
    tmp_path: Path,
    opened: list[Store | Conn],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(store_mod, "fit_shm", lambda path, wanted: 0)
    caplog.set_level(logging.WARNING, logger="lmw_kv.store")
    s = Store(_cfg(tmp_path))
    opened.append(s)
    ns_id = s.register_namespace(_ns(), GROUPS)
    assert s.stats()["ram_bytes"] == 0
    assert s.begin_save(ns_id, _k("a"), ROOT, SCOPE) is None
    assert s.lookup(ns_id, [_k("a")]) == 0
    assert s.begin_load(ns_id, [_k("a")]).slots == []
    assert len([r for r in caplog.records if r.name == "lmw_kv.store"]) == 1


def test_larger_namespace_reslabs_the_pool_when_idle(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    cfg = _cfg(tmp_path, slots=8, nvme=False)
    s, ns_id, c = _open(opened, cfg)
    assert _save(s, c, ns_id, _k("a"), ROOT)
    big = {"full": 3 * PAGE + 1}
    big_id = s.register_namespace(_ns(revision="big", groups=("full",)), big)
    assert s.lookup(ns_id, [_k("a")]) == 0  # RAM copies dropped by the re-slab
    cb = Conn(cfg.ram_path, big)
    opened.append(cb)
    offsets = s.begin_save(big_id, _k("b"), ROOT, SCOPE)
    assert offsets is not None and offsets["full"] % PAGE == 0
    cb.write(_k("b"), offsets)
    s.commit_save(big_id, _k("b"))
    h = s.begin_load(big_id, [_k("b")])
    assert cb.matches(_k("b"), h.slots[0])
    s.end_load(h.id)
    # the smaller namespace still fits in the larger slots
    assert _save(s, c, ns_id, _k("a"), ROOT)


def test_larger_namespace_gets_no_ram_while_blocks_are_pinned(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=8, nvme=False))
    a = _k("a")
    assert _save(s, c, ns_id, a, ROOT)
    h = s.begin_load(ns_id, [a])
    big_id = s.register_namespace(_ns(revision="big", groups=("full",)), {"full": 3 * PAGE})
    assert s.begin_save(big_id, _k("b"), ROOT, SCOPE) is None
    assert c.matches(a, h.slots[0])
    s.end_load(h.id)


def test_ram_block_is_not_evictable_until_its_nvme_write_lands(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    real_append = SegmentWriter.append

    def slow_append(self: SegmentWriter, block_key: bytes, group: str, payload: bytes) -> Any:
        started.set()
        release.wait()
        return real_append(self, block_key, group, payload)

    monkeypatch.setattr(SegmentWriter, "append", slow_append)
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1))
    a, b = _k("a"), _k("b")
    try:
        assert _save(s, c, ns_id, a, ROOT)
        assert started.wait(5)
        assert s.begin_save(ns_id, b, ROOT, SCOPE) is None  # a is still queued for NVMe
    finally:
        release.set()  # never leave the writer thread blocked
    s.flush()
    assert _save(s, c, ns_id, b, ROOT)  # a's RAM copy dropped; it stays on NVMe
    assert s.lookup(ns_id, [a]) == 1
    assert s.lookup(ns_id, [b]) == 1


def test_reslab_keeps_the_file_size(tmp_path: Path, opened: list[Store | Conn]) -> None:
    cfg = StoreConfig(tmp_path / "ram.pool", 16 * PAGE, None, 0)
    small = {"full": 3 * PAGE}  # 5 slots of 3 pages
    big = {"full": 4 * PAGE}  # 4 slots of 4 pages
    s, ns_id, c = _open(opened, cfg, small)
    size = os.path.getsize(cfg.ram_path)
    assert size == 16 * PAGE
    old = Conn(cfg.ram_path, big)  # a connector's mapping, made before the re-slab
    opened.append(old)
    big_id = s.register_namespace(_ns(revision="big", groups=("full",)), big)
    assert os.path.getsize(cfg.ram_path) == size
    assert s.stats()["ram_bytes"] == size
    keys = [_k(f"b{i}") for i in range(4)]
    for key in keys:
        offsets = s.begin_save(big_id, key, ROOT, SCOPE)
        assert offsets is not None
        assert offsets["full"] + big["full"] <= size
        old.write(key, offsets)
        s.commit_save(big_id, key)
    assert s.stats()["ram_used"] == 4 * big["full"]  # 4 whole slots; the tail stays unused
    h = s.begin_load(big_id, keys)
    assert len(h.slots) == 4
    for key, offsets in zip(keys, h.slots, strict=True):
        assert old.matches(key, offsets)
    s.end_load(h.id)
    assert s.lookup(ns_id, [_k("x")]) == 0


def test_end_load_releases_only_its_own_pins(tmp_path: Path, opened: list[Store | Conn]) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=3, nvme=False))
    roots = [_k(f"r{i}") for i in range(3)]
    for key in roots:
        assert _save(s, c, ns_id, key, ROOT)
    ha = s.begin_load(ns_id, roots)
    assert len(ha.slots) == 3
    hb = s.begin_load(ns_id, [*roots, _k("missing")])
    assert len(hb.slots) == 3
    s.end_load(hb.id)
    s.end_load(hb.id)  # ending twice changes nothing
    assert s.begin_save(ns_id, _k("new"), ROOT, SCOPE) is None  # A still pins all three
    for key, offsets in zip(roots, ha.slots, strict=True):
        assert c.matches(key, offsets)
    s.end_load(ha.id)
    assert _save(s, c, ns_id, _k("new"), ROOT)


def test_end_load_unknown_handle_is_a_noop(tmp_path: Path, opened: list[Store | Conn]) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1, nvme=False))
    assert _save(s, c, ns_id, _k("a"), ROOT)
    h = s.begin_load(ns_id, [_k("a")])
    s.end_load(h.id + 1000)
    assert s.begin_save(ns_id, _k("b"), ROOT, SCOPE) is None  # still pinned
    miss = s.begin_load(ns_id, [_k("missing")])
    assert miss.slots == [] and miss.id != h.id
    s.end_load(miss.id)
    s.end_load(h.id)
    assert _save(s, c, ns_id, _k("b"), ROOT)


def test_a_new_store_never_reuses_a_leftover_pool_file(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1: after an unclean exit a live engine may still map the old pool file.

    The next store must get a new inode, so the engine's ``_same_file`` check turns it
    off, and it must never resize or write the pages the engine still has mapped.
    """
    cfg = _cfg(tmp_path, nvme=False)
    cfg.ram_path.write_bytes(b"\xaa" * (8 * SLOT))
    old_ino = os.stat(cfg.ram_path).st_ino
    engine = Conn(cfg.ram_path, GROUPS)  # the old engine's mapping survives
    opened.append(engine)
    seen: list[bool] = []
    real_fit = store_mod.fit_shm

    def fit(path: Path, wanted: int) -> int:
        seen.append(path.exists())  # the stale file must not count as used
        return real_fit(path, wanted)

    monkeypatch.setattr(store_mod, "fit_shm", fit)
    s, ns_id, c = _open(opened, cfg)
    assert seen == [False]
    assert os.stat(cfg.ram_path).st_ino != old_ino
    for key, parent in _chain(["a", "b"]):
        assert _save(s, c, ns_id, key, parent)
    assert engine.read({"full": 0, "gdn": PAGE}) == {
        "full": b"\xaa" * GROUPS["full"],
        "gdn": b"\xaa" * GROUPS["gdn"],
    }
    first_ino = os.stat(cfg.ram_path).st_ino
    s.close()  # a clean exit too: the next store still gets its own file
    opened.remove(s)
    s2, _, _ = _open(opened, cfg)
    assert os.stat(cfg.ram_path).st_ino not in (old_ino, first_ino)  # c still maps the first
    assert s2.stats()["ram_used"] == 0


@pytest.mark.parametrize("nvme", [False, True])
def test_the_namespace_nvme_flag_decides_write_through_and_nvme_loads(
    tmp_path: Path, opened: list[Store | Conn], nvme: bool
) -> None:
    """I3: kv_store "ram" never writes or reads NVMe; "ram+nvme" does both."""
    cfg = _cfg(tmp_path)
    s = Store(cfg)
    opened.append(s)
    ns_id = s.register_namespace(_ns(), GROUPS, nvme=nvme)
    c = Conn(cfg.ram_path, GROUPS)
    opened.append(c)
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    assert s.stats()["nvme_used"] == (HEADER_LEN + 2 * BLOCK_ON_DISK if nvme else 0)
    assert (cfg.nvme_dir / ns_id.hex()).exists() is nvme  # type: ignore[operator]
    c.close()
    s.close()
    opened.clear()

    s2 = Store(cfg)
    opened.append(s2)
    s2.register_namespace(_ns(), GROUPS, nvme=nvme)
    keys = [k for k, _ in chain]
    assert s2.lookup(ns_id, keys) == (2 if nvme else 0)


def test_a_ram_only_namespace_never_loads_blocks_already_on_nvme(
    tmp_path: Path, opened: list[Store | Conn]
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id, c = _open(opened, cfg)  # ram+nvme first
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    c.close()
    s.close()
    opened.clear()

    s2 = Store(cfg)
    opened.append(s2)
    assert s2.register_namespace(_ns(), GROUPS, nvme=False) == ns_id
    keys = [k for k, _ in chain]
    assert s2.lookup(ns_id, keys) == 0
    assert s2.begin_load(ns_id, keys).slots == []
    c2 = Conn(cfg.ram_path, GROUPS)
    opened.append(c2)
    assert _save(s2, c2, ns_id, keys[0], ROOT)  # a RAM copy, not "already on NVMe"
    s2.flush()
    assert s2.stats()["nvme_used"] == HEADER_LEN + 2 * BLOCK_ON_DISK  # unchanged: no write


# ---- a raise never wedges a block (I5) ---------------------------------------------


def _on_nvme_only(tmp_path: Path, opened: list[Store | Conn]) -> tuple[Store, bytes, list[bytes]]:
    """A fresh store whose blocks a and b are on NVMe only."""
    cfg = _cfg(tmp_path)
    s, ns_id, c = _open(opened, cfg)
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    c.close()
    s.close()
    opened.clear()
    s2, _, _ = _open(opened, cfg)
    return s2, ns_id, [k for k, _ in chain]


def test_a_raising_promotion_drops_the_block_and_wakes_waiters(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    s, ns_id, keys = _on_nvme_only(tmp_path, opened)
    real = s._promote

    class Abort(BaseException):  # not an Exception: it escapes, the finally still runs
        pass

    def boom(*a: Any) -> bool:
        raise Abort

    monkeypatch.setattr(s, "_promote", boom)
    with pytest.raises(Abort):
        s.begin_load(ns_id, keys)
    assert not s._ram  # nothing left PROMOTING
    assert not s._loads
    assert s.stats()["ram_used"] == 0
    monkeypatch.setattr(s, "_promote", real)
    h = s.begin_load(ns_id, keys)  # never waits on a block stuck PROMOTING
    assert len(h.slots) == 2
    s.end_load(h.id)


def test_an_ordinary_error_in_a_promotion_is_a_miss(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    s, ns_id, keys = _on_nvme_only(tmp_path, opened)

    def boom(*a: Any) -> bool:
        raise ValueError("bad view")

    monkeypatch.setattr(s, "_promote", boom)
    assert s.begin_load(ns_id, keys).slots == []
    assert not s._ram and not s._loads
    assert s.lookup(ns_id, keys) == 2  # a transient error does not drop the NVMe copy


def test_waiting_on_a_promotion_is_bounded(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    s, ns_id, _ = _open(opened, _cfg(tmp_path, nvme=False))
    a = _k("a")
    with s._cond:
        slot = s._alloc()
        assert slot is not None
        s._ram[(ns_id, a)] = store_mod._Block(slot, ROOT, SCOPE, store_mod._PROMOTING, pins=1)
    monkeypatch.setattr(store_mod, "PROMOTE_WAIT_S", 0.2)
    t0 = time.monotonic()
    assert s.begin_load(ns_id, [a]).slots == []
    assert time.monotonic() - t0 < 2.0


def test_a_raise_mid_load_releases_the_pins_already_taken(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, nvme=False))
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    real = s._load_one
    calls = []

    def flaky(info: Any, key: bytes) -> Any:
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("index broken")
        return real(info, key)

    monkeypatch.setattr(s, "_load_one", flaky)
    with pytest.raises(RuntimeError):
        s.begin_load(ns_id, [k for k, _ in chain])
    assert all(b.pins == 0 for b in s._ram.values())
    assert not s._loads


def test_a_raising_write_through_never_leaves_a_block_queued(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: SegmentWriter, block_key: bytes, group: str, payload: bytes) -> Any:
        raise ValueError("not an NVMe error")

    monkeypatch.setattr(SegmentWriter, "append", boom)
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1))
    a, b = _k("a"), _k("b")
    assert _save(s, c, ns_id, a, ROOT)
    s.flush()
    assert not any(blk.queued for blk in s._ram.values())
    assert _save(s, c, ns_id, b, ROOT)  # a is evictable again (a RAM-only leaf)


# ---- the pool at start, sized to free memory (production gate, window 2) -----------


def test_the_pool_file_is_created_and_committed_when_the_store_starts(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pool exists before any namespace: register_ns must never allocate it.

    Window 2: allocating 16 GiB on the first register_ns took longer than the client's
    deadline while four TP ranks waited, and every rank turned its tier off for good.
    """
    cfg = _cfg(tmp_path, slots=8, nvme=False)
    s = Store(cfg)
    opened.append(s)
    st = os.stat(cfg.ram_path)
    assert st.st_size == 8 * SLOT == s.ram_bytes()
    assert st.st_blocks * 512 >= st.st_size  # every page committed
    ino = st.st_ino

    def boom(*_a: object) -> int:
        raise AssertionError("register_ns allocated the pool")

    monkeypatch.setattr(os, "posix_fallocate", boom, raising=False)
    monkeypatch.setattr(os, "pwrite", boom)
    ns_id = s.register_namespace(_ns(), GROUPS)
    assert os.stat(cfg.ram_path).st_ino == ino  # the same file, not a new one
    c = Conn(cfg.ram_path, GROUPS)
    opened.append(c)
    assert _save(s, c, ns_id, _k("a"), ROOT)


def test_close_removes_a_pool_no_namespace_ever_used(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, nvme=False)
    s = Store(cfg)
    assert cfg.ram_path.exists()
    s.close()
    assert not cfg.ram_path.exists()


def test_a_pool_larger_than_free_memory_is_shrunk_with_one_warning(
    tmp_path: Path,
    opened: list[Store | Conn],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(store_mod.ram, "MIN_POOL_BYTES", 2 * SLOT)
    monkeypatch.setattr(store_mod, "fit_memory", lambda wanted, **kw: 4 * SLOT + 7)
    caplog.set_level(logging.WARNING, logger="lmw_kv.store")
    cfg = _cfg(tmp_path, slots=8, nvme=False)
    s, ns_id, c = _open(opened, cfg)
    assert s.ram_bytes() == 4 * SLOT
    assert os.path.getsize(cfg.ram_path) == 4 * SLOT
    warns = [r.getMessage() for r in caplog.records if r.name == "lmw_kv.store"]
    assert len(warns) == 1
    assert "requested" in warns[0] and "granted" in warns[0] and "free memory" in warns[0]
    assert _save(s, c, ns_id, _k("a"), ROOT)


def test_the_ram_tier_is_off_when_free_memory_is_under_the_minimum(
    tmp_path: Path,
    opened: list[Store | Conn],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(store_mod.ram, "MIN_POOL_BYTES", 4 * SLOT)
    monkeypatch.setattr(store_mod, "fit_memory", lambda wanted, **kw: 3 * SLOT)
    caplog.set_level(logging.WARNING, logger="lmw_kv.store")
    cfg = _cfg(tmp_path, slots=8, nvme=False)
    s = Store(cfg)
    opened.append(s)
    assert s.ram_bytes() == 0
    assert not cfg.ram_path.exists()
    ns_id = s.register_namespace(_ns(), GROUPS)
    assert s.begin_save(ns_id, _k("a"), ROOT, SCOPE) is None
    warns = [r.getMessage() for r in caplog.records if r.name == "lmw_kv.store"]
    assert len(warns) == 1
    assert "off" in warns[0] and "requested" in warns[0] and "minimum" in warns[0]


def test_the_store_keeps_host_memory_for_the_engine_workers(
    tmp_path: Path, opened: list[Store | Conn], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def fit(wanted: int, **kw: Any) -> int:
        seen.append(kw)
        return wanted

    monkeypatch.setattr(store_mod, "fit_memory", fit)
    cfg = _cfg(tmp_path, nvme=False)
    cfg.engine_workers = 3
    opened.append(Store(cfg))
    assert seen == [{"engine_workers": 3}]
