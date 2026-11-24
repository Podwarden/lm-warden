"""Reuse counters wired into the store: ``stats.reuse``, ``top_reused`` (spec 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lmw_kv.reuse import HIST_BUCKETS, empty_report, label
from lmw_kv.scopes import model_tag
from lmw_kv.store import Store, StoreConfig
from tests.unit.lmw_kv.test_store import (
    BLOCK_ON_DISK,
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
A, B = b"\x0a" * 16, b"\x0b" * 16
TAG = b"\x7a" * 16  # an isolation scope


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _store(
    opened: list[Store | Conn],  # noqa: F811
    path: Path,
    *,
    slots: int = 8,
    nvme: bool = True,
    clock: _Clock | None = None,
) -> Store:
    path.mkdir(parents=True, exist_ok=True)
    cfg: StoreConfig = _cfg(path, slots=slots, nvme=nvme)
    cfg.clock = clock or _Clock()
    s = Store(cfg)
    opened.append(s)
    return s


def _reg(s: Store, rev: str = "r1", **kw: Any) -> bytes:
    return s.register_namespace(_ns(rev), GROUPS, owner="a", model=M1, **kw)


def _save(s: Store, ns_id: bytes, key: bytes, tenant: bytes = A, **kw: Any) -> None:
    parent = kw.pop("parent", ROOT)
    assert s.begin_save(ns_id, key, parent, SCOPE, tenant=tenant, **kw) is not None
    s.commit_save(ns_id, key)


def _use(s: Store, ns_id: bytes, keys: list[bytes], tag: str, tenant: bytes = A, **kw: Any) -> int:
    """One load of ``keys`` that the adapter used; how many slots it got."""
    h = s.begin_load(ns_id, keys, tenant=tenant, tag=tag)
    s.end_load(h.id, **({"used": True} | kw))
    return len(h.slots)


def _reuse(s: Store, tenant: bytes) -> dict[str, Any]:
    """The tenant's ``reuse`` map; zeros when it has no row (nothing stored or counted)."""
    row = s.stats()["by_tenant"].get(tenant.hex())
    r: dict[str, Any] = row["reuse"] if row is not None else empty_report()
    return r


def _row(s: Store, key: bytes) -> Any:
    return s._reuse.rows[(M1, key)]


def test_one_reload_per_request_across_groups_and_ranks(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, slots=8, nvme=False)
    nss = [_reg(s, f"rank{i}") for i in range(4)]
    key = _k("p0")
    for ns_id in nss:
        _save(s, ns_id, key)  # no pos: the position is (model, key) in every namespace
    for ns_id in nss:
        assert _use(s, ns_id, [key], "req-1") == 1
    row = _row(s, key)
    assert (row.loads, row.reloads, row.ram_reloads, row.saver_reloads) == (1, 1, 1, 1)
    r = _reuse(s, A)
    assert r["charged"]["reloads"] == 1 and r["saver"]["saver_reloads"] == 1
    assert r["charged"]["positions"] == {"ram": 1, "nvme": 0}
    assert s.stats()["by_model"][M1.hex()]["reuse"][A.hex()] == r


def test_used_false_or_missing_counts_a_load_only(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, nvme=False)
    ns_id = _reg(s)
    key = _k("p0")
    _save(s, ns_id, key)
    _use(s, ns_id, [key], "t1", used=False)
    h = s.begin_load(ns_id, [key], tenant=A, tag="t2")
    s.end_load(h.id)
    h = s.begin_load(ns_id, [key], tenant=A)  # no tag: a load per call, never a reload
    s.end_load(h.id, used=True)
    row = _row(s, key)
    assert row.loads == 3 and row.reloads == 0
    assert _reuse(s, A)["charged"]["positions_reloaded"] == {"ram": 0, "nvme": 0}


def _durable_on_nvme_only(
    opened: list[Store | Conn],  # noqa: F811
    tmp_path: Path,
) -> tuple[Store, bytes, bytes]:
    """A store of two slots whose first block's RAM copy was evicted (it is on NVMe)."""
    s = _store(opened, tmp_path, slots=2)
    ns_id = _reg(s)
    first = _k("first")
    _save(s, ns_id, first)
    s.flush()
    for name in ("x", "y"):
        _save(s, ns_id, _k(name))
        s.flush()
    assert (ns_id, first) not in s._ram and (ns_id, first) in s._nv
    return s, ns_id, first


def test_reload_tier_ram_or_nvme(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    assert _use(s, ns_id, [first], "promoted") == 1
    assert _use(s, ns_id, [first], "in-ram") == 1
    row = _row(s, first)
    assert (row.reloads, row.nvme_reloads, row.ram_reloads) == (2, 1, 1)
    r = _reuse(s, A)["charged"]
    assert (r["ram_reloads"], r["nvme_reloads"]) == (1, 1)
    assert r["positions_reloaded"] == {"ram": 1, "nvme": 1}


def test_nvme_bytes_written_and_read_match_the_records(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    st = s.stats()
    assert st["reuse"]["nvme_bytes_written"] == 3 * BLOCK_ON_DISK
    assert st["reuse"]["nvme_bytes_read"] == 0
    assert _reuse(s, A)["saver"]["nvme_bytes_written"] == 3 * BLOCK_ON_DISK
    assert _use(s, ns_id, [first], "t") == 1
    st = s.stats()
    assert st["reuse"]["nvme_bytes_read"] == BLOCK_ON_DISK
    assert _reuse(s, A)["saver"]["nvme_bytes_read"] == BLOCK_ON_DISK
    assert set(st["reuse"]) == {
        "reuse_persist_degraded", "half_life_s", "order", "nvme_bytes_written",
        "nvme_bytes_read",
    }  # fmt: skip
    assert st["reuse"]["order"] == "lru" and st["reuse"]["half_life_s"] == 21600


def _hist_total(r: dict[str, Any], view: str) -> int:
    return sum(c["positions"] for t in r[view]["evicted_hist"].values() for c in t.values())


def test_eviction_lands_in_the_histogram_and_frees_the_row(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, slots=2, nvme=False)
    ns_id = _reg(s)
    hot = _k("hot")
    _save(s, ns_id, hot)
    _use(s, ns_id, [hot], "t1")
    _save(s, ns_id, _k("b"))
    _save(s, ns_id, _k("c"))  # evicts the oldest leaf: hot (reloaded once)
    assert (M1, hot) not in s._reuse.rows
    r = _reuse(s, A)
    assert r["charged"]["evicted_hist"]["ram"]["1"] == {"positions": 1, "bytes": SLOT}
    assert r["saver"]["evicted_hist"]["ram"]["1"] == {"positions": 1, "bytes": SLOT}
    assert _hist_total(r, "charged") == _hist_total(r, "saver") == 1
    assert set(r["charged"]["evicted_hist"]["nvme"]) == set(HIST_BUCKETS)


def test_staging_release_is_not_an_eviction(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    ns_id = _reg(s, staging=True)
    key = _k("staged")
    _save(s, ns_id, key)
    s.flush()
    st = s.stats()
    assert st["staging"]["released"] == 1 and (ns_id, key) not in s._ram  # S4
    r = st["by_tenant"][A.hex()]["reuse"]
    assert _hist_total(r, "charged") == _hist_total(r, "saver") == 0
    assert r["charged"]["positions"] == {"ram": 0, "nvme": 1}
    assert r["saver"]["wasted_nvme_bytes"] == 0
    assert _use(s, ns_id, [key], "t") == 1  # promoted, then released at end_load (S6)
    r = _reuse(s, A)
    assert _hist_total(r, "charged") == 0
    assert r["charged"]["nvme_reloads"] == 1 and r["charged"]["positions"]["ram"] == 0


def test_a_dropped_staging_write_forgets_the_position(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    path = tmp_path / "s"
    path.mkdir()
    cfg = _cfg(path, cap=100)  # under one record: the write cannot land (S5)
    cfg.clock = _Clock()
    s = Store(cfg)
    opened.append(s)
    ns_id = _reg(s, staging=True)
    key = _k("lost")
    _save(s, ns_id, key)
    s.flush()
    st = s.stats()
    assert st["staging"]["write_dropped"] == 1 and st["evictions"] == 0
    assert (M1, key) not in s._reuse.rows  # forgotten, never tallied as an eviction
    assert _reuse(s, A) == empty_report()


def test_retired_positions_leave_every_report(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, nvme=False)
    ns_id = _reg(s)
    key = _k("isolated")
    _save(s, ns_id, key, scope_tag=TAG)
    _use(s, ns_id, [key], "t")
    assert _reuse(s, A)["charged"]["positions"]["ram"] == 1
    assert len(s.top_reused(None, None, "admin", 10)) == 1
    assert len(s.top_reused(M1, A, "holder", 10)) == 1
    assert s.retire_scopes([TAG]) == 1
    assert _reuse(s, A) == empty_report()  # a dead position appears nowhere
    assert s.top_reused(None, None, "admin", 10) == []
    assert s.top_reused(M1, A, "holder", 10) == []


def test_isolated_holder_report_ignores_other_keys(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    # Ruling R5: two sharing keys; B reloads a position A saved.
    s = _store(opened, tmp_path, nvme=False)
    ns_id = _reg(s)
    key = _k("shared-prefix")
    _save(s, ns_id, key, tenant=A)
    before_a = _reuse(s, A)["saver"]
    holder_a = s.top_reused(None, A, "holder", 10)
    assert _use(s, ns_id, [key], "b-req", tenant=B) == 1
    assert _reuse(s, A)["saver"] == before_a  # saver_reloads, reused, wasted: unchanged
    assert before_a["saver_reloads"] == 0 and before_a["positions_reloaded"] == 0
    assert s.top_reused(None, A, "holder", 10) == holder_a == []
    b = _reuse(s, B)["charged"]
    assert b["positions"] == {"ram": 1, "nvme": 0} and b["reloads"] == 1
    assert b["positions_reloaded"] == {"ram": 1, "nvme": 0}
    rows = s.top_reused(None, B, "holder", 10)
    assert [r["reloads"] for r in rows] == [1]


def test_another_key_filling_a_member_never_moves_the_holder_figures(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    # Two sharing keys: A saves one member of a position, B saves the missing member
    # (another rank) and reloads the whole position from NVMe (disk only mode).
    s = _store(opened, tmp_path, slots=8)
    ns_a, ns_b = _reg(s, "rank0", staging=True), _reg(s, "rank1", staging=True)
    key = _k("shared-prefix")
    _save(s, ns_a, key, tenant=A)
    s.flush()
    before_a = _reuse(s, A)["saver"]
    holder_a = s.top_reused(None, A, "holder", 10)
    assert before_a["nvme_bytes_written"] == BLOCK_ON_DISK
    _save(s, ns_b, key, tenant=B)
    s.flush()
    assert (ns_a, key) not in s._ram and (ns_b, key) not in s._ram  # on disk only
    for ns_id in (ns_a, ns_b):
        assert _use(s, ns_id, [key], "b-req", tenant=B) == 1
    after_a = _reuse(s, A)["saver"]
    for k in ("positions", "positions_reloaded", "saver_reloads", "nvme_bytes_written"):
        assert after_a[k] == before_a[k], k
    assert after_a["wasted_nvme_bytes"] == before_a["wasted_nvme_bytes"] == 0
    assert after_a["nvme_bytes_read"] == 0
    assert s.top_reused(None, A, "holder", 10) == holder_a == []
    assert _row(s, key).nvme_written == BLOCK_ON_DISK  # A's member only
    b = _reuse(s, B)["saver"]
    assert b["nvme_bytes_written"] == BLOCK_ON_DISK
    assert b["nvme_bytes_read"] == 2 * BLOCK_ON_DISK
    st = s.stats()["reuse"]
    assert st["nvme_bytes_written"] == 2 * BLOCK_ON_DISK
    assert st["nvme_bytes_read"] == 2 * BLOCK_ON_DISK


def test_isolated_positions_stay_out_of_other_keys_reports(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path, nvme=False)
    ns_id = _reg(s)
    mine = _k("isolated-a")
    _save(s, ns_id, mine, tenant=A, scope_tag=TAG)
    _use(s, ns_id, [mine], "a-req", tenant=A)
    # B (another scope derives other keys) sees nothing of A's position.
    assert s.top_reused(None, B, "holder", 10) == []
    assert B.hex() not in s.stats()["by_tenant"]
    rows = s.top_reused(None, A, "holder", 10)
    admin = s.top_reused(None, None, "admin", 10)
    assert rows[0]["label"] == label(s._label_key, A, M1, mine)
    assert admin[0]["label"] == label(s._label_key, b"", M1, mine)
    assert rows[0]["label"] != admin[0]["label"]
    for row in rows + admin:
        assert mine.hex() not in str(row) and TAG.hex() not in str(row)


def test_top_reused_rows(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    clock = _Clock()
    s = _store(opened, tmp_path, nvme=False, clock=clock)
    ns_id = _reg(s)
    keys = [_k(f"c{i}") for i in range(3)]
    parent = ROOT
    for depth, key in enumerate(keys):
        _save(s, ns_id, key, parent=parent, depth=depth)
        parent = key
    for i in range(3):
        clock.now += 10
        _use(s, ns_id, keys[: i + 1], f"r{i}")
    rows = s.top_reused(M1, None, "admin", 2)
    assert [(r["depth"], r["reloads"]) for r in rows] == [(0, 3), (1, 2)]
    assert rows[0]["last_reload"] == clock.now and rows[0]["first_saved"] == 1_000_000.0
    assert rows[0]["tiers"] == ["ram"] and rows[0]["charged"] == A.hex()
    assert rows[0]["model"] == M1.hex()
    assert s.top_reused(b"\x01" * 16, None, "admin", 10) == []
    assert len(s.top_reused(None, A, "admin", 10)) == 3
    assert s.top_reused(None, B, "admin", 10) == []


def test_label_key_is_the_data_key_with_nvme(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    on = _store(opened, tmp_path / "on")
    off = _store(opened, tmp_path / "off", nvme=False)
    assert on._label_key == on._data_key and len(on._data_key) == 32
    assert len(off._label_key) == 32 and off._label_key != on._label_key
