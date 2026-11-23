"""Who owns what in the store: owners (I3), models, positions, tenants, usage, stats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lmw_kv.format import SegmentWriter, record_size
from lmw_kv.scopes import model_tag
from lmw_kv.store import MANIFEST_NAME, Store, StoreConfig
from tests.unit.lmw_kv.test_store import (
    BLOCK_ON_DISK,
    GROUPS,
    ROOT,
    SCOPE,
    SLOT,
    Conn,
    _cfg,
    _chain,
    _k,
    _ns,
    opened,  # noqa: F401  (fixture)
)
from tests.unit.lmw_kv.test_store_manifest_upgrade import _rewrite_as_step1

M1 = model_tag("org/model", "rev-1")
M2 = model_tag("org/model", "rev-2")
T1 = b"\x01" * 16
T2 = b"\x02" * 16
STATE_GROUPS = {"state": 300}


def _store(opened: list[Store | Conn], cfg: StoreConfig) -> Store:  # noqa: F811
    s = Store(cfg)
    opened.append(s)
    return s


def _save(
    s: Store,
    ns_id: bytes,
    key: bytes,
    parent: bytes = ROOT,
    *,
    tenant: bytes = b"",
    pos: bytes = b"",
) -> bool:
    offsets = s.begin_save(ns_id, key, parent, SCOPE, tenant=tenant, pos=pos)
    if offsets is None:
        return False
    s.commit_save(ns_id, key)
    return True


def _segments(cfg: StoreConfig, ns_id: bytes) -> list[Path]:
    assert cfg.nvme_dir is not None
    return list((cfg.nvme_dir / ns_id.hex()).glob("*.seg"))


@pytest.mark.parametrize("a_first", [True, False])
def test_nvme_on_only_if_every_owner_wants_it(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    a_first: bool,
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    regs = [("a", False), ("b", True)]
    for owner, nvme in regs if a_first else reversed(regs):
        ns_id = s.register_namespace(_ns(), GROUPS, nvme=nvme, owner=owner, model=M1)
    for key, parent in _chain(["x", "y"]):
        assert _save(s, ns_id, key, parent)
    s.flush()
    assert _segments(cfg, ns_id) == []
    assert s.stats()["nvme_used"] == 0

    assert s.release_owner("a") == 0  # b still owns it
    assert _save(s, ns_id, _k("z"), _k("y"))
    s.flush()
    assert _segments(cfg, ns_id) != []
    assert s.stats()["nvme_enabled"] is True


def test_legacy_owner_keeps_last_registration_wins(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """Step 1 API only (runs on the step 1 store too, where it fails: KeyError, NVMe off)."""
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, nvme=False)
    s.register_namespace(_ns(), GROUPS, nvme=True)
    key = _k("x")
    assert s.begin_save(ns_id, key, ROOT, SCOPE) is not None
    s.commit_save(ns_id, key)
    s.flush()
    assert s.stats()["nvme_enabled"] is True
    assert _segments(cfg, ns_id) != []
    s.close()
    opened.clear()

    s2 = _store(opened, cfg)  # RAM empty: found only if the manifest row landed
    s2.register_namespace(_ns(), GROUPS)
    h = s2.begin_load(ns_id, [key])
    assert h.nvme_blocks == 1
    s2.end_load(h.id)
    assert s2.stats()["by_namespace"][ns_id.hex()]["owners"] == {"": True}


def test_a_ram_owner_registering_mid_write_leaves_no_row(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a")
    real = SegmentWriter.append

    def append(self: SegmentWriter, block_key: bytes, group: str, payload: bytes) -> Any:
        with s._cond:  # what a RAM-only owner's registration leaves behind
            info = s._ns[ns_id]
            info.owners["b"] = False
            info.nvme = False
        return real(self, block_key, group, payload)

    monkeypatch.setattr(SegmentWriter, "append", append)
    assert _save(s, ns_id, _k("x"))
    s.flush()
    assert s._manifests[ns_id].get(ns_id, _k("x")) == []
    assert not s._ram[(ns_id, _k("x"))].durable
    assert s.usage("nvme") == {}
    assert s.stats()["nvme_enabled"] is True


def test_a_load_writes_no_manifest_under_the_store_lock(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    chain = _chain(["x", "y", "z"])
    for key, parent in chain:
        assert _save(s, ns_id, key, parent, tenant=T1)
    s.flush()
    keys = [k for k, _ in chain]
    m = s._manifests[ns_id]
    writes: list[int] = []
    real_many = m.set_tenants

    def no_single(*args: Any) -> None:
        raise AssertionError("one UPDATE per block")

    def set_tenants(ns: bytes, changes: Any) -> None:
        assert not s._cond._is_owned()  # type: ignore[attr-defined]
        writes.append(len(changes))
        real_many(ns, changes)

    monkeypatch.setattr(m, "set_tenant", no_single)
    monkeypatch.setattr(m, "set_tenants", set_tenants)
    h = s.begin_load(ns_id, keys, tenant=T2)
    assert len(h.slots) == 3
    s.end_load(h.id)
    assert writes == [3]  # one batch, after the lock was let go
    assert {e.tenant for k in keys for e in m.get(ns_id, k)} == {T2}
    assert not s._tenant_dirty


def test_a_tenant_change_waits_while_the_namespace_is_ram_only(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    assert _save(s, ns_id, _k("x"), tenant=T1)
    s.flush()
    s.register_namespace(_ns(), GROUPS, owner="b", model=M1, nvme=False)
    h = s.begin_load(ns_id, [_k("x")], tenant=T2)  # the RAM copy
    s.end_load(h.id)
    m = s._manifests[ns_id]
    assert {e.tenant for e in m.get(ns_id, _k("x"))} == {T1}  # not written while off
    assert s.release_owner("b") == 0
    h = s.begin_load(ns_id, [_k("x")], tenant=T2)
    s.end_load(h.id)
    assert {e.tenant for e in m.get(ns_id, _k("x"))} == {T2}


def test_released_namespace_is_unregistered(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, nvme=False))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    chain = _chain(["x", "y"])
    for key, parent in chain:
        assert _save(s, ns_id, key, parent, tenant=T1)
    keys = [k for k, _ in chain]
    assert s.usage("ram") == {(M1, T1): 2 * SLOT}

    assert s.release_owner("a") == 1
    assert s.release_owner("a") == 0  # already gone
    assert s.lookup(ns_id, keys) == 0
    assert s.begin_load(ns_id, keys).slots == []
    assert s.begin_save(ns_id, _k("z"), keys[-1], SCOPE) is None
    st = s.stats()
    assert st["dead_bytes"]["ram"] == 2 * SLOT
    assert st["by_namespace"][ns_id.hex()]["live"] is False
    assert s.usage("ram") == {}  # dead blocks are not usage

    s.register_namespace(_ns(), GROUPS, owner="b", model=M1)
    assert s.lookup(ns_id, keys) == 2
    assert s.stats()["dead_bytes"]["ram"] == 0


def test_namespace_model_is_fixed(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    assert s.register_namespace(_ns(), GROUPS, owner="b", model=M2) == ns_id
    assert _save(s, ns_id, _k("x"), tenant=T1)
    assert s.release_owner("a") == 0
    assert _save(s, ns_id, _k("y"), _k("x"), tenant=T1)
    s.flush()
    assert set(s.usage("ram")) == {(M1, T1)}
    assert set(s.usage("nvme")) == {(M1, T1)}
    assert set(s._positions["ram"]) == {(M1, _k("x")), (M1, _k("y"))}
    st = s.stats()
    assert st["by_namespace"][ns_id.hex()]["model"] == M1.hex()
    assert st["by_namespace"][ns_id.hex()]["owners"] == {"b": True}
    assert set(st["by_model"]) == {M1.hex()}
    assert st["by_model"][M1.hex()]["owners"] == ["b"]
    rows = s._manifests[ns_id].get(ns_id, _k("y"))
    assert {e.model for e in rows} == {M1}


def test_last_user_pays_in_ram_and_on_nvme(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    chain = _chain(["x", "y"])
    for key, parent in chain:
        assert _save(s, ns_id, key, parent, tenant=T1)
    s.flush()
    keys = [k for k, _ in chain]
    assert s.usage("ram") == {(M1, T1): 2 * SLOT}
    assert s.usage("nvme") == {(M1, T1): 2 * BLOCK_ON_DISK}

    h = s.begin_load(ns_id, keys, tenant=T2)
    assert len(h.slots) == 2
    s.end_load(h.id)
    assert s.usage("ram") == {(M1, T2): 2 * SLOT}
    assert s.usage("nvme") == {(M1, T2): 2 * BLOCK_ON_DISK}
    m = s._manifests[ns_id]
    assert {e.tenant for k in keys for e in m.get(ns_id, k)} == {T2}

    h = s.begin_load(ns_id, keys)  # no tenant: the charge stays
    s.end_load(h.id)
    assert s.usage("ram") == {(M1, T2): 2 * SLOT}


def test_last_user_pays_for_a_promoted_block(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    assert _save(s, ns_id, _k("x"), tenant=T1)
    s.flush()
    s.close()
    opened.clear()

    s2 = _store(opened, cfg)
    s2.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    assert s2.usage("ram") == {}
    assert s2.usage("nvme") == {(M1, T1): BLOCK_ON_DISK}
    h = s2.begin_load(ns_id, [_k("x")], tenant=T2)
    assert h.nvme_blocks == 1
    s2.end_load(h.id)
    assert s2.usage("ram") == {(M1, T2): SLOT}
    assert s2.usage("nvme") == {(M1, T2): BLOCK_ON_DISK}


def test_positions_group_namespaces_of_one_model(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    att = s.register_namespace(_ns(), GROUPS, owner="a", model=M1, role="attention")
    st_ns = s.register_namespace(
        _ns(groups=("state",)), STATE_GROUPS, owner="a", model=M1, role="gdn_state"
    )
    key = _k("x")
    assert _save(s, att, key, tenant=T1)
    assert _save(s, st_ns, key, tenant=T1)
    s.flush()
    assert s._positions["ram"] == {(M1, key): {(att, key), (st_ns, key)}}
    assert s._positions["nvme"] == {(M1, key): {(att, key), (st_ns, key)}}
    assert s.usage("ram") == {(M1, T1): 2 * SLOT}
    state_on_disk = record_size("state", STATE_GROUPS["state"])
    assert s.usage("nvme") == {(M1, T1): BLOCK_ON_DISK + state_on_disk}
    by_ns = s.stats()["by_namespace"]
    assert by_ns[st_ns.hex()]["role"] == "gdn_state"
    assert by_ns[att.hex()]["role"] == "attention"


def test_explicit_position_key(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, nvme=False))
    a = s.register_namespace(_ns("r1"), GROUPS, owner="a", model=M1)
    b = s.register_namespace(_ns("r2"), GROUPS, owner="a", model=M1, rank=1)
    pos = b"\x09" * 32
    assert _save(s, a, _k("a0"), pos=pos)
    assert _save(s, b, _k("b0"), pos=pos)
    assert s._positions["ram"] == {(M1, pos): {(a, _k("a0")), (b, _k("b0"))}}
    assert s.stats()["by_namespace"][b.hex()]["rank"] == 1


def test_old_manifest_blocks_are_tenant_empty(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS)
    chain = _chain(["x", "y"])
    for key, parent in chain:
        assert _save(s, ns_id, key, parent, tenant=T1)
    s.flush()
    s.close()
    opened.clear()
    assert cfg.nvme_dir is not None
    _rewrite_as_step1(cfg.nvme_dir / ns_id.hex() / MANIFEST_NAME)

    s2 = _store(opened, cfg)
    old_model = ns_id[:16]
    assert s2.usage("nvme") == {(old_model, b""): 2 * BLOCK_ON_DISK}
    assert set(s2._positions["nvme"]) == {(old_model, k) for k, _ in chain}
    assert s2.stats()["by_tenant"][""]["nvme_bytes"] == 2 * BLOCK_ON_DISK


def test_ram_eviction_is_charged_to_the_victim_and_names_the_taker(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, slots=1, nvme=False))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    assert _save(s, ns_id, _k("x"), tenant=T1)
    assert _save(s, ns_id, _k("y"), tenant=T2)  # evicts x
    st = s.stats()
    t1 = st["by_tenant"][T1.hex()]
    assert t1["ram_evictions"] == 1
    assert t1["evicted_by"] == {T2.hex(): 1}
    assert t1["ram_bytes"] == 0
    assert st["by_tenant"][T2.hex()]["ram_bytes"] == SLOT
    assert st["evictions"] == 1


def test_hits_are_counted_per_tenant(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, nvme=False))
    s.record_hits("req-1", 32, 16, 0.01, tenant=T1)
    s.record_hits("req-2", 8, 0, 0.01, tenant=T1)
    s.record_hits("req-3", 5, 5, 0.01)
    t1 = s.stats()["by_tenant"][T1.hex()]
    assert (t1["ram_tokens"], t1["nvme_tokens"]) == (40, 16)
    assert s.pop_hits("req-1") == {"ram_tokens": 32, "nvme_tokens": 16, "load_s": 0.01}


def test_stats_reports_nvme_cap(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, _cfg(tmp_path, cap=123 << 20))
    st = s.stats()
    assert st["nvme_cap_bytes"] == 123 << 20
    assert st["policy"] == "lru"
    assert st["dead_bytes"] == {"ram": 0, "nvme": 0}
    assert st["by_tenant"] == {} and st["by_model"] == {} and st["by_namespace"] == {}
    s2 = _store(opened, _cfg(tmp_path / "ram-only", nvme=False))
    assert s2.stats()["nvme_cap_bytes"] == 0


def test_the_clock_is_injectable(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path, nvme=False)
    cfg.clock = lambda: 1234.5
    s = _store(opened, cfg)
    ns_id = s.register_namespace(_ns(), GROUPS)
    assert _save(s, ns_id, _k("x"))
    assert s._ram[(ns_id, _k("x"))].last_used == 1234.5


def test_stats_reports_when_the_store_started(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """Usage, evictions and evicted_by are counted since the store started: ``started_at``
    says when (wall time, the injectable clock), so a report can label them."""
    cfg = _cfg(tmp_path, nvme=False)
    now = [1000.0]
    cfg.clock = lambda: now[0]
    s = _store(opened, cfg)
    now[0] = 5000.0
    assert s.stats()["started_at"] == 1000.0


def test_evictions_are_split_per_model_and_tier(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """``by_model[m].tenants[t]`` carries the tenant's evictions inside that model, per
    tier, with who caused them; ``by_tenant`` keeps the sum over models."""
    s = _store(opened, _cfg(tmp_path, slots=2, nvme=False))
    ns1 = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    ns2 = s.register_namespace(_ns("other"), GROUPS, owner="b", model=M2)
    assert _save(s, ns1, _k("x"), tenant=T1)
    assert _save(s, ns2, _k("y"), tenant=T1)
    assert _save(s, ns1, _k("z"), tenant=T2)  # evicts x (M1, T1)
    st = s.stats()
    m1_t1 = st["by_model"][M1.hex()]["tenants"][T1.hex()]
    assert (m1_t1["ram_evictions"], m1_t1["nvme_evictions"]) == (1, 0)
    assert m1_t1["evicted_by"] == {T2.hex(): 1}
    assert m1_t1["ram_bytes"] == 0  # the row stays though its usage is gone
    m2_t1 = st["by_model"][M2.hex()]["tenants"][T1.hex()]
    assert (m2_t1["ram_evictions"], m2_t1["nvme_evictions"]) == (0, 0)
    assert m2_t1["evicted_by"] == {}
    assert st["by_model"][M1.hex()]["tenants"][T2.hex()]["ram_evictions"] == 0
    assert st["by_tenant"][T1.hex()]["ram_evictions"] == 1


def test_a_released_model_keeps_its_last_owner(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    """A model with no live owner left is still named: ``last_owner`` is the owner that
    registered it last, so the api does not report it as unknown."""
    s = _store(opened, _cfg(tmp_path, nvme=False))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    s.register_namespace(_ns(), GROUPS, owner="b", model=M1)
    assert _save(s, ns_id, _k("x"), tenant=T1)
    s.release_owner("a")
    s.release_owner("b")
    row = s.stats()["by_model"][M1.hex()]
    assert row["owners"] == [] and row["last_owner"] == "b"


def test_per_model_eviction_counters_are_capped(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (model, tenant) eviction counters keep the most recently counted pairs only, so
    the stats reply cannot grow for the life of the store; ``by_tenant`` keeps every count."""
    monkeypatch.setattr("lmw_kv.store.PAIR_STATS_MAX", 2)
    s = _store(opened, _cfg(tmp_path, slots=1, nvme=False))
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", model=M1)
    tenants = [bytes([i]) * 16 for i in range(1, 5)]
    for i, t in enumerate(tenants):
        assert _save(s, ns_id, _k(f"x{i}"), tenant=t)  # each evicts the one before
    st = s.stats()
    rows = st["by_model"][M1.hex()]["tenants"]
    assert tenants[0].hex() not in rows  # dropped: the oldest pair, which holds nothing
    assert rows[tenants[1].hex()]["ram_evictions"] == 1
    assert rows[tenants[2].hex()]["ram_evictions"] == 1
    assert st["by_tenant"][tenants[0].hex()]["ram_evictions"] == 1
    assert len(s._pair_stats) == 2
