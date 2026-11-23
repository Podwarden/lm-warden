"""Loads under ``fair`` may borrow NVMe-backed RAM (ruling R14 revised, spec 5.3).

A key over its RAM share that resumes a session stored on NVMe promotes its blocks into
RAM. When nobody over its share has an evictable position (the loader's own are pinned
by this load), a save is refused (R4), but a load takes the least recently used position
whose every block is also on NVMe: its owner moves to the slower tier and loses nothing.
A block that is only in RAM is never taken this way. The loader pays for what it loaded.
"""

from __future__ import annotations

from pathlib import Path

from lmw_kv.fair import Policy, TenantPolicy
from lmw_kv.scopes import model_tag
from lmw_kv.store import Store
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
V = b"\x0e" * 16  # resumes a session above its RAM share
X, Y = b"\x01" * 16, b"\x02" * 16
# V weighs 1, X and Y 6 each: with 6 RAM slots V's share is under half a slot.
POLICY = Policy(
    mode="fair",
    tenants={V: TenantPolicy(weight=1), X: TenantPolicy(weight=6), Y: TenantPolicy(weight=6)},
)
SESSION = [_k(f"v{i}") for i in range(5)]


def _chain(s: Store, ns_id: bytes, keys: list[bytes], tenant: bytes) -> None:
    parent = ROOT
    for key in keys:
        assert s.begin_save(ns_id, key, parent, SCOPE, tenant=tenant) is not None
        s.commit_save(ns_id, key)
        s.flush()
        parent = key


def _resumed_store(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    *,
    y_on_nvme: bool,
) -> tuple[Store, bytes, bytes]:
    """V's session on NVMe only (a restart emptied RAM); X holds 2 slots that are also on
    NVMe, Y 2 slots (RAM only unless ``y_on_nvme``), saved before X's; 2 slots free."""
    cfg = _cfg(tmp_path, slots=6)
    first = Store(cfg)
    ns_id = first.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    first.set_policy(POLICY)
    _chain(first, ns_id, SESSION, V)
    first.close()
    s = Store(cfg)
    opened.append(s)
    s.set_policy(POLICY)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    ny = ns_id
    if not y_on_nvme:
        ny = s.register_namespace(_ns("r-ram"), GROUPS, owner="b", model=M1, nvme=False)
    _chain(s, ny, [_k("y0"), _k("y1")], Y)
    _chain(s, ns_id, [_k("x0"), _k("x1")], X)
    assert s.usage("ram") == {(M1, X): 2 * SLOT, (M1, Y): 2 * SLOT}
    return s, ns_id, ny


def test_an_over_share_resume_borrows_nvme_backed_slots(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, _ = _resumed_store(tmp_path, opened, y_on_nvme=True)
    h = s.begin_load(ns_id, SESSION, tenant=V)
    assert len(h.slots) == 5 and h.nvme_blocks == 5
    # X and Y were only demoted: every block is still found (on NVMe now).
    for key in ("x0", "x1", "y0", "y1"):
        assert s.lookup(ns_id, [_k(key)]) == 1
    assert s.usage("ram")[(M1, V)] == 5 * SLOT  # the loader pays for what it borrowed
    s.end_load(h.id)


def test_a_load_never_takes_a_block_that_is_only_in_ram(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, ny = _resumed_store(tmp_path, opened, y_on_nvme=False)
    h = s.begin_load(ns_id, SESSION, tenant=V)
    # Two free slots and X's two NVMe-backed ones; Y's RAM-only blocks are never taken.
    assert len(h.slots) == 4
    assert (ny, _k("y0")) in s._ram and (ny, _k("y1")) in s._ram
    assert s.lookup(ns_id, [_k("x0"), _k("x1")]) == 2
    s.end_load(h.id)


def test_a_save_is_still_refused_where_a_load_borrows(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, _ = _resumed_store(tmp_path, opened, y_on_nvme=True)
    h = s.begin_load(ns_id, SESSION[:3], tenant=V)  # V over its share, its slots pinned
    assert len(h.slots) == 3
    assert s.begin_save(ns_id, _k("v-new"), SESSION[2], SCOPE, tenant=V) is None
    assert s.usage("ram")[(M1, V)] == 3 * SLOT
    s.end_load(h.id)


def test_lru_loads_are_unchanged(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, ny = _resumed_store(tmp_path, opened, y_on_nvme=False)
    s.set_policy(Policy())
    h = s.begin_load(ns_id, SESSION, tenant=V)
    # Global LRU takes Y's oldest RAM-only leaves too, as step 1 did.
    assert len(h.slots) == 5
    assert (ny, _k("y0")) not in s._ram
    s.end_load(h.id)
