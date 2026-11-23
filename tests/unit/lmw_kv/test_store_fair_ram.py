"""RAM victims under the ``fair`` policy: classes, shares, positions and caps (spec 5.3)."""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from lmw_kv.fair import Policy, TenantPolicy
from lmw_kv.scopes import model_tag
from lmw_kv.store import Store, StoreConfig
from tests.unit.lmw_kv.test_store import (
    GROUPS,
    ROOT,
    SCOPE,
    SLOT,
    Conn,
    _cfg,
    _k,
    _ns,
    opened,  # noqa: F401  (fixture)
)

M1 = model_tag("org/model", "rev-1")
MA = model_tag("org/model-a", "rev-1")
MB = model_tag("org/model-b", "rev-1")
F = b"\x0f" * 16  # the flood
V = b"\x0e" * 16  # the victim
X, Y, Z = b"\x01" * 16, b"\x02" * 16, b"\x03" * 16
C = b"\x0c" * 16  # a capped tenant
STATE_GROUPS = {"state": 300}
TAG = b"\x7a" * 16
FAIR = Policy(mode="fair")


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _store(
    opened: list[Store | Conn],  # noqa: F811
    path: Path,
    policy: Policy = FAIR,
    *,
    slots: int = 12,
    clock: _Clock | None = None,
) -> Store:
    path.mkdir(parents=True, exist_ok=True)
    cfg: StoreConfig = _cfg(path, slots=slots, nvme=False)
    if clock is not None:
        cfg.clock = clock
    s = Store(cfg)
    opened.append(s)
    s.set_policy(policy)
    return s


def _save(
    s: Store,
    ns_id: bytes,
    key: bytes,
    tenant: bytes,
    *,
    pos: bytes = b"",
    scope_tag: bytes = b"",
    parent: bytes = ROOT,
) -> bool:
    got = s.begin_save(ns_id, key, parent, SCOPE, tenant=tenant, pos=pos, scope_tag=scope_tag)
    if got is None:
        return False
    s.commit_save(ns_id, key)
    return True


def _in_ram(s: Store, ns_id: bytes, key: bytes) -> bool:
    return (ns_id, key) in s._ram


def test_fair_flood_does_not_evict_under_share_key(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    v_keys = [_k(f"v{i}") for i in range(3)]
    for key in v_keys:
        assert _save(s, ns_id, key, V)
    for i in range(30):
        assert _save(s, ns_id, _k(f"f{i}"), F)
    assert all(s.lookup(ns_id, [key]) == 1 for key in v_keys)
    assert s.usage("ram") == {(M1, V): 3 * SLOT, (M1, F): 9 * SLOT}
    st = s.stats()["by_tenant"]
    assert st[V.hex()]["ram_evictions"] == 0
    assert st[F.hex()]["ram_evictions"] == 21 and st[F.hex()]["evicted_by"] == {F.hex(): 21}


def test_pinned_over_key_falls_to_phase_two_not_lru(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    # Model A weighs 1 of 12: its part is one slot, and its one position takes two.
    s = _store(opened, tmp_path, Policy(mode="fair", owner_weights={"a": 1, "b": 11}))
    na1 = s.register_namespace(_ns("a1"), GROUPS, owner="a", model=MA)
    na2 = s.register_namespace(_ns("a2", groups=("state",)), STATE_GROUPS, owner="a", model=MA)
    nb = s.register_namespace(_ns("b1"), GROUPS, owner="b", model=MB)
    pa = _k("pa")
    assert _save(s, na1, _k("pa-attn"), X, pos=pa)
    assert _save(s, na2, _k("pa-state"), X, pos=pa)
    h1 = s.begin_load(na1, [_k("pa-attn")])
    h2 = s.begin_load(na2, [_k("pa-state")])
    assert len(h1.slots) == len(h2.slots) == 1
    for i in range(3):
        assert _save(s, nb, _k(f"z{i}"), Z)  # under its share in B, the oldest there
    for i in range(7):
        assert _save(s, nb, _k(f"y{i}"), Y)  # over its share in B; B is under its part
    assert s.stats()["ram_used"] == 12 * SLOT

    assert _save(s, nb, _k("z-new"), Z)
    assert not _in_ram(s, nb, _k("y0"))
    assert all(_in_ram(s, nb, _k(f"z{i}")) for i in range(3))
    assert _in_ram(s, na1, _k("pa-attn")) and _in_ram(s, na2, _k("pa-state"))
    s.end_load(h1.id)
    s.end_load(h2.id)


def test_an_over_share_saver_never_takes_an_under_share_key(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    v_keys = [_k(f"v{i}") for i in range(3)]
    f_keys = [_k(f"f{i}") for i in range(9)]
    for key in v_keys:
        assert _save(s, ns_id, key, V)
    for key in f_keys:
        assert _save(s, ns_id, key, F)
    h = s.begin_load(ns_id, f_keys[:1])  # every position of F pinned, one load each
    hs = [h] + [s.begin_load(ns_id, [key]) for key in f_keys[1:]]
    assert all(len(x.slots) == 1 for x in hs)

    assert s.begin_save(ns_id, _k("f-new"), ROOT, SCOPE, tenant=F) is None
    assert all(_in_ram(s, ns_id, key) for key in v_keys)
    # An under-share taker in the same state gets a slot from the global LRU.
    assert _save(s, ns_id, _k("z0"), Z)
    assert not _in_ram(s, ns_id, v_keys[0]) and _in_ram(s, ns_id, v_keys[1])
    for x in hs:
        s.end_load(x.id)


def test_the_refusal_is_per_model_and_tenant(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    na = s.register_namespace(_ns("a"), GROUPS, owner="a", model=MA)
    nb = s.register_namespace(_ns("b"), GROUPS, owner="b", model=MB)
    v_keys = [_k(f"v{i}") for i in range(3)]
    f_keys = [_k(f"f{i}") for i in range(9)]
    for key in v_keys:
        assert _save(s, na, key, V)
    for key in f_keys:
        assert _save(s, na, key, F)  # F is over its share on A, and fully pinned there
    hs = [s.begin_load(na, [key]) for key in f_keys]
    assert all(len(x.slots) == 1 for x in hs)

    assert s.begin_save(na, _k("fa-new"), ROOT, SCOPE, tenant=F) is None
    assert all(_in_ram(s, na, key) for key in v_keys)
    assert _save(s, nb, _k("fb-new"), F)  # F holds nothing on B: the global LRU serves it
    assert not _in_ram(s, na, v_keys[0])
    for x in hs:
        s.end_load(x.id)


def test_interior_positions_never_go_only_leaves(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, slots=4)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    parent = ROOT
    for name in "abcd":  # one RAM-only chain, root first: a is the oldest, d the leaf
        assert _save(s, ns_id, _k(name), X, parent=parent)
        parent = _k(name)
    assert _save(s, ns_id, _k("e"), X)
    assert not _in_ram(s, ns_id, _k("d"))
    assert _save(s, ns_id, _k("f"), X)  # c is a leaf now, and older than e
    assert not _in_ram(s, ns_id, _k("c"))
    assert all(_in_ram(s, ns_id, _k(k)) for k in "abef")
    assert s.lookup(ns_id, [_k("a"), _k("b")]) == 2


def test_a_ram_block_whose_child_is_on_nvme_is_not_a_leaf(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = Store(_cfg(tmp_path, slots=2))
    opened.append(s)
    s.set_policy(FAIR)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    real = s._queue.put_nowait
    dropped: list[Any] = []

    def full_once(item: Any) -> None:
        if not dropped:
            dropped.append(item)
            raise queue.Full  # the parent's NVMe write never happens
        real(item)

    monkeypatch.setattr(s._queue, "put_nowait", full_once)
    p, c = _k("p"), _k("c")
    assert _save(s, ns_id, p, X)
    s.flush()
    assert _save(s, ns_id, c, X, parent=p)
    s.flush()
    assert not s._ram[(ns_id, p)].durable and s._ram[(ns_id, c)].durable
    assert _save(s, ns_id, _k("x"), X)  # c goes from RAM: durable, so it may
    s.flush()
    assert not _in_ram(s, ns_id, c) and s.lookup(ns_id, [p, c]) == 2
    assert _save(s, ns_id, _k("y"), X)  # p is older, but its only child is on NVMe
    s.flush()
    assert _in_ram(s, ns_id, p) and not _in_ram(s, ns_id, _k("x"))
    assert s.lookup(ns_id, [p, c]) == 2


@pytest.mark.parametrize("how", ["released", "retired"])
def test_dead_goes_first(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    how: str,
) -> None:
    s = _store(opened, tmp_path)
    live = s.register_namespace(_ns("live"), GROUPS, owner="a", model=M1)
    if how == "released":
        dead = s.register_namespace(_ns("dead"), GROUPS, owner="d", model=MA)
        tag = b""
    else:
        dead, tag = live, TAG
    for i in range(3):
        assert _save(s, live, _k(f"l{i}"), X)
    for i in range(2):
        assert _save(s, dead, _k(f"d{i}"), Y, scope_tag=tag)
    for i in range(3, 10):
        assert _save(s, live, _k(f"l{i}"), X)
    if how == "released":
        assert s.release_owner("d") == 1
    else:
        assert s.retire_scopes([TAG]) == 1
    assert s.stats()["dead_bytes"]["ram"] == 2 * SLOT

    assert _save(s, live, _k("l-new"), X)
    assert not _in_ram(s, dead, _k("d0")) and _in_ram(s, dead, _k("d1"))
    assert _in_ram(s, live, _k("l0"))


@pytest.mark.parametrize(("idle_for", "idle_goes"), [(7201.0, True), (7199.0, False)])
def test_idle_loses_protection(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    idle_for: float,
    idle_goes: bool,
) -> None:
    clock = _Clock()
    s = _store(opened, tmp_path, clock=clock)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    for i in range(3):
        assert _save(s, ns_id, _k(f"v{i}"), V)
    clock.now += idle_for
    for i in range(9):
        assert _save(s, ns_id, _k(f"f{i}"), F)  # the pool is full; F is over its share
    assert _save(s, ns_id, _k("f-new"), F)
    assert _in_ram(s, ns_id, _k("v0")) is not idle_goes
    assert _in_ram(s, ns_id, _k("f0")) is idle_goes
    assert _in_ram(s, ns_id, _k("v1"))


def test_global_lru_when_nobody_over(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    for i in range(6):
        assert _save(s, ns_id, _k(f"y{i}"), Y)
        assert _save(s, ns_id, _k(f"x{i}"), X)
    h = s.begin_load(ns_id, [_k("y0")])  # y0 is used again: x0 is now the oldest
    s.end_load(h.id)
    assert _save(s, ns_id, _k("z0"), Z)
    assert not _in_ram(s, ns_id, _k("x0"))
    assert _in_ram(s, ns_id, _k("y0")) and _in_ram(s, ns_id, _k("y1"))


def _two_member_positions(s: Store) -> tuple[bytes, bytes]:
    attn = s.register_namespace(_ns("attn"), GROUPS, owner="a", model=M1)
    state = s.register_namespace(_ns("state", groups=("state",)), STATE_GROUPS, owner="a", model=M1)
    for p in ("p1", "p2"):
        assert _save(s, attn, _k(f"{p}-attn"), X, pos=_k(p))
        assert _save(s, state, _k(f"{p}-state"), X, pos=_k(p))
    return attn, state


def test_state_block_goes_with_its_position(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path / "a", slots=4)
    attn, state = _two_member_positions(s)
    assert _save(s, attn, _k("p3-attn"), X, pos=_k("p3"))
    assert not _in_ram(s, attn, _k("p1-attn")) and not _in_ram(s, state, _k("p1-state"))
    assert _in_ram(s, attn, _k("p2-attn")) and _in_ram(s, state, _k("p2-state"))

    # The attention member is pinned: its state member never goes alone.
    s2 = _store(opened, tmp_path / "b", slots=4)
    attn, state = _two_member_positions(s2)
    h = s2.begin_load(attn, [_k("p1-attn")])
    assert _save(s2, attn, _k("p3-attn"), X, pos=_k("p3"))
    assert _in_ram(s2, attn, _k("p1-attn")) and _in_ram(s2, state, _k("p1-state"))
    assert not _in_ram(s2, attn, _k("p2-attn")) and not _in_ram(s2, state, _k("p2-state"))
    s2.end_load(h.id)


@pytest.mark.parametrize("busy", ["pinned", "promoting", "queued", "saving"])
def test_pinned_promoting_or_queued_position_is_skipped(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    busy: str,
) -> None:
    s = _store(opened, tmp_path)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    for i in range(3):
        assert _save(s, ns_id, _k(f"v{i}"), V)
    f0 = _k("f0")  # the over tenant's oldest position, busy from the start
    h = None
    if busy == "saving":
        assert s.begin_save(ns_id, f0, ROOT, SCOPE, tenant=F) is not None
    else:
        assert _save(s, ns_id, f0, F)
    with s._cond:
        b = s._ram[(ns_id, f0)]
        if busy == "promoting":
            b.state = "promoting"
        elif busy == "queued":
            b.queued = True
    if busy == "pinned":
        h = s.begin_load(ns_id, [f0])
        assert len(h.slots) == 1
    for i in range(1, 9):
        assert _save(s, ns_id, _k(f"f{i}"), F)
    assert _save(s, ns_id, _k("f-new"), F)
    assert _in_ram(s, ns_id, f0)
    assert not _in_ram(s, ns_id, _k("f1"))
    assert all(_in_ram(s, ns_id, _k(f"v{i}")) for i in range(3))
    if h is not None:
        s.end_load(h.id)
    with s._cond:
        b.state, b.queued = ("valid" if busy != "saving" else "saving"), False


def test_cap_evicts_own_blocks_first(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    # 25% of 12 slots: three
    s = _store(opened, tmp_path, Policy(mode="fair", tenants={C: TenantPolicy(max_ram_pct=25)}))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    for i in range(2):
        assert _save(s, ns_id, _k(f"x{i}"), X)  # older than every block of C
    for i in range(3):
        assert _save(s, ns_id, _k(f"c{i}"), C)
    assert s.usage("ram")[(M1, C)] == 3 * SLOT
    assert _save(s, ns_id, _k("c3"), C)  # free slots remain, but C is at its cap
    assert not _in_ram(s, ns_id, _k("c0"))
    assert all(_in_ram(s, ns_id, _k(k)) for k in ("c1", "c2", "c3", "x0", "x1"))
    assert s.usage("ram")[(M1, C)] == 3 * SLOT
    assert s.stats()["ram_used"] == 5 * SLOT
    assert s.stats()["by_tenant"][C.hex()]["evicted_by"] == {C.hex(): 1}


def test_cap_with_nothing_evictable_refuses_the_save(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, Policy(mode="fair", tenants={C: TenantPolicy(max_ram_pct=25)}))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    keys = [_k(f"c{i}") for i in range(3)]
    for key in keys:
        assert _save(s, ns_id, key, C)
    h = s.begin_load(ns_id, keys)
    assert len(h.slots) == 3
    assert s.begin_save(ns_id, _k("c3"), ROOT, SCOPE, tenant=C) is None
    assert all(_in_ram(s, ns_id, key) for key in keys)
    assert _save(s, ns_id, _k("x0"), X)  # another tenant is not capped
    s.end_load(h.id)
    assert _save(s, ns_id, _k("c3"), C)  # unpinned: C evicts its own oldest again
    assert not _in_ram(s, ns_id, keys[0])


def test_cap_counts_blocks_still_being_saved(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, Policy(mode="fair", tenants={C: TenantPolicy(max_ram_pct=25)}))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    keys = [_k(f"c{i}") for i in range(3)]
    for key in keys:  # three saves in flight, none committed
        assert s.begin_save(ns_id, key, ROOT, SCOPE, tenant=C) is not None
    assert s.begin_save(ns_id, _k("c3"), ROOT, SCOPE, tenant=C) is None
    s.abort_save(ns_id, keys[0])  # an aborted save gives its room back
    assert s.begin_save(ns_id, _k("c3"), ROOT, SCOPE, tenant=C) is not None
    for key in [*keys[1:], _k("c3")]:
        s.commit_save(ns_id, key)
    assert s.usage("ram")[(M1, C)] == 3 * SLOT
    assert _save(s, ns_id, _k("c4"), C)  # committed now: C evicts its own oldest
    assert not _in_ram(s, ns_id, keys[1])


def test_cap_full_of_saves_in_flight_refuses_before_evicting(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, Policy(mode="fair", tenants={C: TenantPolicy(max_ram_pct=25)}))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    assert _save(s, ns_id, _k("c0"), C)
    for key in (_k("s1"), _k("s2")):
        assert s.begin_save(ns_id, key, ROOT, SCOPE, tenant=C) is not None
    # The cap drops below the saves in flight plus one more block (about 1.9 slots).
    s.set_policy(Policy(mode="fair", tenants={C: TenantPolicy(max_ram_pct=16)}))
    assert s.begin_save(ns_id, _k("c3"), ROOT, SCOPE, tenant=C) is None
    assert _in_ram(s, ns_id, _k("c0"))  # refused without evicting C's committed block
    assert s.stats()["by_tenant"][C.hex()]["ram_evictions"] == 0


def test_lru_mode_ignores_caps_and_classes(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    policy = Policy(mode="lru", tenants={C: TenantPolicy(max_ram_pct=25)})
    s = _store(opened, tmp_path, policy)
    live = s.register_namespace(_ns("live"), GROUPS, owner="a", model=M1)
    dead = s.register_namespace(_ns("dead"), GROUPS, owner="d", model=MA)
    for i in range(5):
        assert _save(s, live, _k(f"c{i}"), C)  # past its cap: lru has no caps
    assert all(_in_ram(s, live, _k(f"c{i}")) for i in range(5))
    for i in range(2):
        assert _save(s, dead, _k(f"d{i}"), Y)
    for i in range(5):
        assert _save(s, live, _k(f"x{i}"), X)
    assert s.release_owner("d") == 1
    assert _save(s, live, _k("x-new"), X)  # step 1 order: the oldest block, not the dead
    assert not _in_ram(s, live, _k("c0")) and _in_ram(s, dead, _k("d0"))
    assert s.stats()["policy"] == "lru"


@pytest.mark.parametrize("mode", ["lru", "fair"])
def test_stats_report_weights_parts_and_shares(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    mode: str,
) -> None:
    policy = Policy(
        mode=mode,  # type: ignore[arg-type]
        owner_weights={"a": 2, "a2": 1},
        tenants={X: TenantPolicy(weight=3)},
    )
    s = _store(opened, tmp_path, policy)
    na = s.register_namespace(_ns("a"), GROUPS, owner="a", model=MA)
    s.register_namespace(_ns("a"), GROUPS, owner="a2", model=MA)  # the largest weight wins
    nb = s.register_namespace(_ns("b"), GROUPS, owner="b", model=MB)  # default weight 1
    assert _save(s, na, _k("x0"), X)
    assert _save(s, na, _k("y0"), Y)
    assert _save(s, nb, _k("z0"), Z)
    st = s.stats()
    assert st["policy"] == mode
    a, b = st["by_model"][MA.hex()], st["by_model"][MB.hex()]
    assert a["weight"] == 2 and b["weight"] == 1
    assert a["ram_part"] == 8.0 * SLOT and b["ram_part"] == 4.0 * SLOT
    assert a["nvme_part"] == 0.0 and b["nvme_part"] == 0.0  # no NVMe tier
    # X weighs 3, Y the default 6: X takes a third of A's part
    assert a["tenants"][X.hex()] == {
        "ram_bytes": SLOT, "nvme_bytes": 0, "ram_share": 8.0 * SLOT / 3, "nvme_share": 0.0,
        "ram_evictions": 0, "nvme_evictions": 0, "evicted_by": {},
    }  # fmt: skip
    assert a["tenants"][Y.hex()]["ram_share"] == 16.0 * SLOT / 3
    assert b["tenants"] == {
        Z.hex(): {
            "ram_bytes": SLOT, "nvme_bytes": 0, "ram_share": 4.0 * SLOT, "nvme_share": 0.0,
            "ram_evictions": 0, "nvme_evictions": 0, "evicted_by": {},
        }
    }  # fmt: skip
    assert st["by_tenant"][Z.hex()]["ram_share"] == 4.0 * SLOT
    assert st["by_tenant"][X.hex()]["nvme_share"] == 0.0
