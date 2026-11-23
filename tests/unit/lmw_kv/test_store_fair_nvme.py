"""NVMe under the ``fair`` policy: which whole segment to reclaim (spec 5.1 erratum, 5.3).

``SEGMENT_BYTES`` is shrunk so one segment file holds exactly four blocks of ``GROUPS``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import store as store_mod
from lmw_kv.fair import Policy, TenantPolicy
from lmw_kv.format import HEADER_LEN
from lmw_kv.manifest import Manifest
from lmw_kv.scopes import model_tag
from lmw_kv.store import Store, StoreConfig
from tests.unit.lmw_kv.test_store import (
    BLOCK_ON_DISK,
    GROUPS,
    ROOT,
    SCOPE,
    SLOT,
    Conn,
    _disk,
    _k,
    _ns,
    opened,  # noqa: F401  (fixture)
)

SEG = HEADER_LEN + 4 * BLOCK_ON_DISK  # one segment file: a header and four blocks
M1 = model_tag("org/model", "rev-1")
F = b"\x0f" * 16  # the flood
V = b"\x0e" * 16  # the victim
X, Y = b"\x01" * 16, b"\x02" * 16
C = b"\x0c" * 16  # a capped tenant
TAG = b"\x7a" * 16
STATE_GROUPS = {"state": 300}
FAIR = Policy(mode="fair")
DAY = 86_400.0


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _small_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "SEGMENT_BYTES", SEG)


def _cfg(path: Path, segs: int, *, slots: int = 8, clock: _Clock | None = None) -> StoreConfig:
    cfg = StoreConfig(
        ram_path=path / "ram.pool",
        ram_bytes=slots * SLOT,
        nvme_dir=path / "nvme",
        nvme_cap_bytes=segs * SEG,
    )
    if clock is not None:
        cfg.clock = clock
    return cfg


def _store(
    opened: list[Store | Conn],  # noqa: F811
    cfg: StoreConfig,
    policy: Policy = FAIR,
) -> Store:
    s = Store(cfg)
    opened.append(s)
    s.set_policy(policy)
    return s


def _reg(s: Store, rev: str = "r1", groups: dict[str, int] = GROUPS, **kw: Any) -> bytes:
    return s.register_namespace(_ns(rev, groups=tuple(groups)), groups, owner="a", model=M1, **kw)


def _w(
    s: Store,
    ns_id: bytes,
    key: bytes,
    tenant: bytes,
    *,
    parent: bytes = ROOT,
    pos: bytes = b"",
    tag: bytes = b"",
) -> None:
    """Save one block and wait for its NVMe write."""
    got = s.begin_save(ns_id, key, parent, SCOPE, tenant=tenant, pos=pos, scope_tag=tag)
    assert got is not None
    s.commit_save(ns_id, key)
    s.flush()


def _on_nvme(s: Store, ns_id: bytes, keys: list[bytes]) -> int:
    with s._cond:
        return sum((ns_id, k) in s._nv for k in keys)


def _close(s: Store, opened: list[Store | Conn]) -> None:  # noqa: F811
    s.close()
    opened.remove(s)


@pytest.mark.parametrize("mode", ["fair", "lru"])
def test_nvme_flood_keeps_victim_session(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    mode: str,
) -> None:
    cfg = _cfg(tmp_path, segs=4)
    s = _store(opened, cfg, Policy(mode=mode))  # type: ignore[arg-type]
    ns_id = _reg(s)
    v_keys = [_k(f"v{i}") for i in range(4)]
    parent = ROOT
    for key in v_keys:  # one chain, one segment of its own (the oldest)
        _w(s, ns_id, key, V, parent=parent)
        parent = key
    f_keys = [_k(f"f{i}") for i in range(200)]
    for key in f_keys:
        _w(s, ns_id, key, F)
        assert _disk(cfg) <= cfg.nvme_cap_bytes
    assert s.stats()["nvme_enabled"] is True
    _close(s, opened)

    s2 = _store(opened, cfg, Policy())
    ns_id = _reg(s2)
    assert s2.lookup(ns_id, v_keys) == (4 if mode == "fair" else 0)
    assert s2.lookup(ns_id, f_keys[-1:]) == 1


def test_dead_segment_goes_first(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, segs=3))
    ns_id = _reg(s)
    x_old = [_k(f"x{i}") for i in range(4)]
    y_keys = [_k(f"y{i}") for i in range(4)]
    for key in x_old:  # segment 1, the oldest, live
        _w(s, ns_id, key, X)
    for key in y_keys:  # segment 2, retired below
        _w(s, ns_id, key, Y, tag=TAG)
    assert s.retire_scopes([TAG]) == 1
    for i in range(4):  # segment 3; its last block needs room
        _w(s, ns_id, _k(f"x-new{i}"), X)
    assert _on_nvme(s, ns_id, y_keys) == 0
    assert _on_nvme(s, ns_id, x_old) == 4


def _two_namespaces(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> tuple[StoreConfig, bytes, bytes, list[bytes], list[bytes]]:
    """Namespace a's segment (the oldest), then b's; the store is closed.

    A reopened store ages a segment by its file mtime, and the two files are
    written within one filesystem timestamp tick often enough (Linux stamps
    mtime from a coarse, jiffy-granular clock) that "a is the oldest" was a
    race: on a tie the eviction heap falls back to comparing namespace ids,
    and b's sorts first, so b's segment went instead (flaky in CI). The mtimes
    are set here, a's a minute before b's, so the order is the one stated."""
    cfg = _cfg(tmp_path, segs=3)
    s = _store(opened, cfg)
    na, nb = _reg(s, "a"), _reg(s, "b")
    a_old = [_k(f"a{i}") for i in range(4)]
    b_keys = [_k(f"b{i}") for i in range(4)]
    for key in a_old:
        _w(s, na, key, X)
    for key in b_keys:
        _w(s, nb, key, X)
    _close(s, opened)
    now = time.time()
    for ns_id, t in ((na, now - 120.0), (nb, now - 60.0)):
        segs = list((tmp_path / "nvme" / ns_id.hex()).glob("*.seg"))
        assert segs
        for seg in segs:
            os.utime(seg, (t, t))
    return cfg, na, nb, a_old, b_keys


def test_unregistered_namespace_segment_is_dead(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """A namespace on NVMe that nobody registered since start is class dead, after the
    grace."""
    cfg, na, nb, a_old, b_keys = _two_namespaces(tmp_path, opened)
    clock = _Clock()
    cfg.clock = clock
    s2 = _store(opened, cfg)
    assert _reg(s2, "a") == na  # b never registers again
    assert s2.stats()["dead_bytes"]["nvme"] == 0  # inside the grace
    clock.now += store_mod.UNREGISTERED_GRACE_S
    assert s2.stats()["dead_bytes"]["nvme"] == 4 * BLOCK_ON_DISK
    for i in range(4):
        _w(s2, na, _k(f"a-new{i}"), X)
    assert _on_nvme(s2, nb, b_keys) == 0
    assert _on_nvme(s2, na, a_old) == 4


def test_unregistered_namespace_inside_the_grace_keeps_its_segment(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg, na, nb, a_old, b_keys = _two_namespaces(tmp_path, opened)
    clock = _Clock()
    cfg.clock = clock
    s2 = _store(opened, cfg)
    assert _reg(s2, "a") == na
    clock.now += store_mod.UNREGISTERED_GRACE_S - 1
    for i in range(4):
        _w(s2, na, _k(f"a-new{i}"), X)
    assert _on_nvme(s2, nb, b_keys) == 4  # not dead yet: the oldest segment went instead
    assert _on_nvme(s2, na, a_old) == 0
    assert s2.lookup(_reg(s2, "b"), b_keys) == 4  # b's model loads again: all there


def test_idle_segment_before_over_share(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    clock = _Clock()
    s = _store(opened, _cfg(tmp_path, segs=3, clock=clock))
    ns_id = _reg(s)
    f_old = [_k(f"f{i}") for i in range(4)]
    x_keys = [_k(f"x{i}") for i in range(4)]
    for key in f_old:  # segment 1, the oldest: F, over its share once it grows
        _w(s, ns_id, key, F)
    for key in x_keys:  # segment 2: X, under its share, then idle
        _w(s, ns_id, key, X)
    clock.now += 8 * DAY
    for key in f_old:  # F uses its old blocks again: not idle
        h = s.begin_load(ns_id, [key])
        assert len(h.slots) == 1
        s.end_load(h.id)
    for i in range(4):  # segment 3; its last block needs room
        _w(s, ns_id, _k(f"f-new{i}"), F)
    assert _on_nvme(s, ns_id, x_keys) == 0
    assert _on_nvme(s, ns_id, f_old) == 4


def test_position_drop_spans_namespaces(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path, segs=8)
    s = _store(opened, cfg)
    na = _reg(s, "attn")
    ns_state = _reg(s, "state", STATE_GROUPS, role="state")
    a_keys = [_k(f"a{i}") for i in range(4)]
    s_keys = [_k(f"s{i}") for i in range(4)]
    a_parent = s_parent = ROOT
    for i, (a, st) in enumerate(zip(a_keys, s_keys, strict=True)):
        pos = _k(f"p{i}")
        _w(s, na, a, X, parent=a_parent, pos=pos)
        _w(s, ns_state, st, X, parent=s_parent, pos=pos)
        a_parent, s_parent = a, st
    _close(s, opened)

    s2 = _store(opened, cfg)  # RAM empty: NVMe only
    assert _reg(s2, "attn") == na
    assert _reg(s2, "state", STATE_GROUPS, role="state") == ns_state
    assert s2.lookup(na, a_keys) == 4 and s2.lookup(ns_state, s_keys) == 4
    with s2._cond:
        (a_seg,) = [k for k in s2._segs if k[0] == na]
        s2._segs[a_seg][0] = 0.0  # the attention segment is the oldest
        assert s2._reclaim_segment(b"")
        assert a_seg not in s2._segs
    assert s2.lookup(na, a_keys) == 0
    assert s2.lookup(ns_state, s_keys) == 0
    assert _on_nvme(s2, ns_state, s_keys) == 0
    m = s2._manifests[ns_state]
    assert all(m.get(ns_state, key) == [] for key in s_keys)


def test_capped_tenant_reclaims_its_own_segment(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path, segs=10)  # no pressure on the whole tier
    policy = Policy(mode="fair", tenants={C: TenantPolicy(max_nvme_pct=10)})  # one segment
    s = _store(opened, cfg, policy)
    ns_id = _reg(s)
    x1 = [_k(f"x1-{i}") for i in range(4)]
    c_old = [_k(f"c{i}") for i in range(4)]
    x2 = [_k(f"x2-{i}") for i in range(4)]
    for key in x1:  # segment 1, the oldest
        _w(s, ns_id, key, X)
    for key in c_old:  # segment 2: C at its cap
        _w(s, ns_id, key, C)
    for key in x2:  # segment 3
        _w(s, ns_id, key, X)
    _w(s, ns_id, _k("c-new"), C)
    assert _on_nvme(s, ns_id, c_old) == 0
    assert _on_nvme(s, ns_id, [_k("c-new")]) == 1
    assert _on_nvme(s, ns_id, x1 + x2) == 8
    assert s.stats()["by_tenant"][C.hex()]["nvme_bytes"] == BLOCK_ON_DISK


def test_promoting_block_still_blocks_its_segment(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path, segs=3)
    s = _store(opened, cfg)
    ns_id = _reg(s)
    f_keys = [_k(f"f{i}") for i in range(4)]
    x_keys = [_k(f"x{i}") for i in range(4)]
    for key in f_keys:  # segment 1, the oldest
        _w(s, ns_id, key, F)
    for key in x_keys:
        _w(s, ns_id, key, X)
    _close(s, opened)

    s2 = _store(opened, cfg)
    assert _reg(s2) == ns_id
    started, release = threading.Event(), threading.Event()
    real = s2._promote

    def slow(*args: Any) -> bool:
        started.set()
        release.wait(10)
        return real(*args)

    s2._promote = slow  # type: ignore[method-assign,assignment]
    t = threading.Thread(target=lambda: s2.end_load(s2.begin_load(ns_id, f_keys[:1]).id))
    t.start()
    try:
        assert started.wait(5)
        with s2._cond:
            assert s2._make_nvme_room(cfg.nvme_cap_bytes - s2._nvme_used + 1)
    finally:
        release.set()
        t.join(10)
    assert _on_nvme(s2, ns_id, f_keys) == 4
    assert _on_nvme(s2, ns_id, x_keys) == 0


def test_no_leaf_or_table_scan_under_lock(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _store(opened, _cfg(tmp_path, segs=4, slots=4))
    na = _reg(s)
    calls = {"iter_blocks": 0, "keys_in_segment": 0, "children": 0, "plan": 0, "reclaim": 0}
    outside: list[str] = []
    in_plan: list[bool] = []

    def counted(name: str, real: Any) -> Any:
        def wrap(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            if name != "iter_blocks" and not in_plan:
                outside.append(name)
            return real(*args, **kwargs)

        return wrap

    for name in ("iter_blocks", "keys_in_segment", "children"):
        monkeypatch.setattr(Manifest, name, counted(name, getattr(Manifest, name)))
    real_plan, real_reclaim = Store._plan_reclaim, Store._reclaim_segment

    def plan(self: Store, *args: Any) -> Any:
        calls["plan"] += 1
        in_plan.append(True)
        try:
            return real_plan(self, *args)
        finally:
            in_plan.pop()

    def reclaim(self: Store, *args: Any, **kwargs: Any) -> Any:
        calls["reclaim"] += 1
        return real_reclaim(self, *args, **kwargs)

    monkeypatch.setattr(Store, "_plan_reclaim", plan)
    monkeypatch.setattr(Store, "_reclaim_segment", reclaim)
    i = 0
    parent = ROOT
    while calls["reclaim"] < 100 and i < 2000:
        key = _k(f"b{i}")
        _w(s, na, key, (X, Y)[i % 2], parent=parent)  # short chains: descendants to follow
        parent = ROOT if i % 3 == 2 else key
        i += 1
    assert calls["reclaim"] >= 100
    assert calls["iter_blocks"] == 0
    assert outside == []
    assert calls["keys_in_segment"] <= calls["plan"]
    assert calls["children"] > 0  # the prefix-closure walk did run, inside _plan_reclaim


def test_nvme_child_count_matches_the_manifest(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """The in-memory NVMe child count the fair leaf test reads agrees with SQLite."""
    cfg = _cfg(tmp_path, segs=8)
    s = _store(opened, cfg)
    ns_id = _reg(s)
    a, b, c, d = (_k(n) for n in "abcd")
    for key, parent in ((a, ROOT), (b, a), (c, b), (d, a)):  # a has two children
        _w(s, ns_id, key, X, parent=parent)

    def check(st: Store, kids: dict[bytes, int]) -> None:
        with st._cond:
            for key in (a, b, c, d):
                ik = (ns_id, key)
                assert st._nv_kids.get(ik, 0) == kids.get(key, 0), key
                if not st._children.get(ik, 0):
                    assert st._leaf_in_memory(ik) == st._is_leaf(ik), key

    check(s, {a: 2, b: 1})
    _close(s, opened)
    s2 = _store(opened, cfg)  # rebuilt from the manifest, RAM empty
    assert _reg(s2) == ns_id
    check(s2, {a: 2, b: 1})
    with s2._cond:
        s2._drop_nvme(ns_id, c)
    check(s2, {a: 2})
    with s2._cond:
        s2._drop_nvme(ns_id, d)
    check(s2, {a: 1})


def test_capped_tenant_interleaved_with_others_is_refused(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """C holds a quarter of every segment: reclaiming one would destroy 3 bytes of X's
    for each of its own, so C's write is refused and stays in RAM only."""
    cfg = _cfg(tmp_path, segs=10)
    policy = Policy(mode="fair", tenants={C: TenantPolicy(max_nvme_pct=10)})  # one segment
    s = _store(opened, cfg, policy)
    ns_id = _reg(s)
    x_keys: list[bytes] = []
    c_keys: list[bytes] = []
    for i in range(4):  # four segments, each X X X C
        for j in range(3):
            x_keys.append(_k(f"x{i}-{j}"))
            _w(s, ns_id, x_keys[-1], X)
        c_keys.append(_k(f"c{i}"))
        _w(s, ns_id, c_keys[-1], C)
    _w(s, ns_id, _k("c-new"), C)
    assert _on_nvme(s, ns_id, [_k("c-new")]) == 0
    assert (ns_id, _k("c-new")) in s._ram  # a RAM-only block
    assert _on_nvme(s, ns_id, x_keys) == 12
    assert _on_nvme(s, ns_id, c_keys) == 4


def test_garbage_only_segment_beats_a_live_one(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, segs=8))
    ns_id = _reg(s)
    x_keys = [_k(f"x{i}") for i in range(4)]
    y_keys = [_k(f"y{i}") for i in range(4)]
    for key in x_keys:  # segment 1, the oldest, live
        _w(s, ns_id, key, X)
    for key in y_keys:
        _w(s, ns_id, key, Y)
    _w(s, ns_id, _k("z"), X)  # the writer moves on to segment 3
    with s._cond:
        (seg2,) = [k for k in s._segs if k[0] == ns_id and k[1] == 2]
        for key in y_keys:  # every record dropped; the file stays until reclaimed
            s._drop_nvme(ns_id, key)
        assert s._segment_score(*seg2)[0] == 4 * BLOCK_ON_DISK
        assert s._reclaim_segment(b"")
        assert seg2 not in s._segs
    assert _on_nvme(s, ns_id, x_keys) == 4


def test_fair_reclaim_keeps_prefix_closure_across_segments(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path, segs=8)
    s = _store(opened, cfg)
    ns_id = _reg(s)
    keys = [_k(f"c{i}") for i in range(6)]  # c0-c3 in segment 1, c4-c5 in segment 2
    parent = ROOT
    for key in keys:
        _w(s, ns_id, key, X, parent=parent)
        parent = key
    _close(s, opened)
    s2 = _store(opened, cfg)  # NVMe only
    assert _reg(s2) == ns_id
    assert s2.lookup(ns_id, keys) == 6
    with s2._cond:
        s2._segs[(ns_id, 1)][0] = 0.0
        assert s2._reclaim_segment(b"")
        assert s2._segs == {}  # segment 2 lost every row: deleted with it
    assert s2.lookup(ns_id, keys) == 0
    assert all(s2.lookup(ns_id, [key]) == 0 for key in keys)  # no orphan left
    assert s2.stats()["blocks"] == 0


def _attn_and_state(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    state: list[tuple[str, str, str]],
) -> tuple[Store, bytes, bytes]:
    """Attention block a0 at position p0, then ``state`` blocks (name, parent, pos); the
    store is reopened (RAM empty) with the attention segment the oldest."""
    cfg = _cfg(tmp_path, segs=8)
    s = _store(opened, cfg)
    na = _reg(s, "attn")
    nst = _reg(s, "state", STATE_GROUPS, role="state")
    _w(s, na, _k("a0"), X, pos=_k("p0"))
    for name, parent, pos in state:
        _w(s, nst, _k(name), X, parent=_k(parent) if parent else ROOT, pos=_k(pos))
    _close(s, opened)
    s2 = _store(opened, cfg)
    _reg(s2, "attn")
    _reg(s2, "state", STATE_GROUPS, role="state")
    with s2._cond:
        (a_seg,) = [k for k in s2._segs if k[0] == na]
        s2._segs[a_seg][0] = 0.0
    return s2, na, nst


def test_position_peer_with_a_ram_copy_stays_present(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, na, nst = _attn_and_state(tmp_path, opened, [("s0", "", "p0")])
    h = s.begin_load(nst, [_k("s0")])  # promoted: a valid RAM copy
    assert len(h.slots) == 1
    s.end_load(h.id)
    with s._cond:
        assert s._reclaim_segment(b"")
    assert s.lookup(na, [_k("a0")]) == 0
    assert _on_nvme(s, nst, [_k("s0")]) == 0  # its NVMe row went with the position
    assert s.lookup(nst, [_k("s0")]) == 1  # the RAM copy keeps it present


def test_position_peer_with_a_child_elsewhere_stays(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """s0 is a member of p0, but its child s1 is at a position with no attention block:
    dropping s0 would orphan s1, so s0 stays (a miss on attention only)."""
    s, na, nst = _attn_and_state(tmp_path, opened, [("s0", "", "p0"), ("s1", "s0", "q1")])
    with s._cond:
        assert s._reclaim_segment(b"")
    assert s.lookup(na, [_k("a0")]) == 0
    assert s.lookup(nst, [_k("s0"), _k("s1")]) == 2


def test_retired_blocks_never_served_after_a_fair_reclaim(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path, segs=3)
    s = _store(opened, cfg)
    ns_id = _reg(s)
    t1 = [_k(f"t1-{i}") for i in range(4)]
    t2 = [_k(f"t2-{i}") for i in range(4)]
    for key in t1 + t2:  # segments 1 and 2, both isolated under TAG
        _w(s, ns_id, key, Y, tag=TAG)
    s.retire_scopes([TAG])
    for i in range(4):  # segment 3; its last block needs room: one dead segment goes
        _w(s, ns_id, _k(f"x{i}"), X)
    assert _on_nvme(s, ns_id, t1 + t2) == 4
    for key in t1 + t2:
        assert s.lookup(ns_id, [key]) == 0
        assert s.begin_load(ns_id, [key]).slots == []
    _close(s, opened)
    s2 = _store(opened, cfg)
    assert _reg(s2) == ns_id
    assert all(s2.lookup(ns_id, [key]) == 0 for key in t1 + t2)


W = b"\x0a" * 16  # the writer a reclaim makes room for
Z = b"\x0b" * 16  # another tenant, saving while the reclaim lets the lock go


def test_reclaims_stay_charged_to_the_writer_across_the_lock_release(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, segs=8, slots=4)
    s = _store(opened, cfg)
    ns_id = _reg(s)
    for t, name in ((X, "x"), (Y, "y"), (C, "c")):  # three segments of four blocks
        for i in range(4):
            _w(s, ns_id, _k(f"{name}{i}"), t)
    real_yield = os.sched_yield  # the store's os module
    saved: list[bool] = []

    def other_thread_saves() -> None:  # runs while _cond is let go
        if not saved:
            key = _k("z")
            saved.append(s.begin_save(ns_id, key, ROOT, SCOPE, tenant=Z) is not None)
            s.abort_save(ns_id, key)  # RAM was full: that save evicted one block
        real_yield()

    def charged() -> tuple[dict[str, int], int, int]:
        rows = s.stats()["by_tenant"]
        by_taker: dict[str, int] = {}
        for row in rows.values():
            for taker, n in row["evicted_by"].items():
                by_taker[taker] = by_taker.get(taker, 0) + n
        nvme = sum(r["nvme_evictions"] for r in rows.values())
        return by_taker, nvme, sum(r["ram_evictions"] for r in rows.values())

    taken0, nvme0, ram0 = charged()  # setup's own RAM evictions
    monkeypatch.setattr(os, "sched_yield", other_thread_saves)
    with s._cond:
        s._evict_for = W
        need = cfg.nvme_cap_bytes - s._nvme_used + SEG + 1  # two segments must go
        try:
            assert s._make_nvme_room(need, W)
        finally:
            s._evict_for = None
    assert saved == [True]
    taken, nvme, ram = charged()
    delta = {t: n - taken0.get(t, 0) for t, n in taken.items() if n != taken0.get(t, 0)}
    assert delta == {W.hex(): 8, Z.hex(): 1}
    assert (nvme - nvme0, ram - ram0) == (8, 1)


def test_capped_scan_moves_on_to_the_next_slice(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only two of C's segments are looked at per write: both mostly X's, so the first
    write is refused; the next looks further and finds C's half of segment 3."""
    monkeypatch.setattr(store_mod, "CAPPED_SCAN", 2)
    cfg = _cfg(tmp_path, segs=10)
    policy = Policy(mode="fair", tenants={C: TenantPolicy(max_nvme_pct=10)})  # one segment
    s = _store(opened, cfg, policy)
    ns_id = _reg(s)
    layout = ["xxxc", "xxxc", "ccxx", "xxxx"]
    keys: dict[int, list[tuple[bytes, bytes]]] = {}
    for si, who in enumerate(layout):
        for j, t in enumerate(who):
            key = _k(f"{si}-{j}")
            keys.setdefault(si, []).append((key, C if t == "c" else X))
            _w(s, ns_id, key, C if t == "c" else X)
    _w(s, ns_id, _k("c-new1"), C)
    assert _on_nvme(s, ns_id, [_k("c-new1")]) == 0  # refused: segments 1 and 2 only
    _w(s, ns_id, _k("c-new2"), C)
    assert _on_nvme(s, ns_id, [_k("c-new2")]) == 1  # segment 3 went
    assert _on_nvme(s, ns_id, [k for k, _ in keys[2]]) == 0
    assert _on_nvme(s, ns_id, [k for si in (0, 1, 3) for k, _ in keys[si]]) == 12
