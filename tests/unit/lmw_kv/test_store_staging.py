"""Staging namespaces (spec 4.2, S1 to S9): RAM holds a block only while it is in flight.

A staging block leaves RAM once its NVMe write lands (S4), when its write cannot land
(S3, S5) or when the last load pinning it ends (S6). Such drops are counted in
``stats.staging``, never as evictions (S9). Saves may hold at most half the pool (S7);
loads may use any free slot. Under ``fair`` a staging allocation takes dead or durable
positions only (S8, ruling R9).
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import store as store_mod
from lmw_kv.fair import Policy, TenantPolicy
from lmw_kv.manifest import Manifest
from lmw_kv.scopes import model_tag, scope_tag
from lmw_kv.store import STAGING_SAVE_SHARE, Store, StoreConfig
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
from tests.unit.lmw_kv.test_store_isolation import SCOPE_A, _a_keys, _check_disjoint

M1 = model_tag("org/model", "rev-1")
M2 = model_tag("org/disk-only", "rev-1")
X, S = b"\x01" * 16, b"\x05" * 16
PHRASE = "disk-only models store nothing"


def _clocked(cfg: StoreConfig) -> StoreConfig:
    ticks = itertools.count(1000)
    return dataclasses.replace(cfg, clock=lambda: float(next(ticks)))


class Paused:
    """Holds the writer thread before each write until ``go`` is set (closed first)."""

    def __init__(self, s: Store) -> None:
        self.started, self.go = threading.Event(), threading.Event()
        real: Callable[[Any], None] = s._write_through_once

        def slow(ik: Any) -> None:
            self.started.set()
            self.go.wait(10)
            real(ik)

        s._write_through_once = slow

    def close(self) -> None:
        self.go.set()


def _staging_store(
    opened: list[Any],  # noqa: F811
    cfg: StoreConfig,
    *,
    pause: bool = False,
) -> tuple[Store, bytes, Conn, Paused | None]:
    s = Store(_clocked(cfg))
    opened.append(s)
    ns_id = s.register_namespace(_ns(), GROUPS, owner="a", staging=True)
    c = Conn(cfg.ram_path, GROUPS)
    opened.append(c)
    p = None
    if pause:
        p = Paused(s)
        opened.append(p)  # closed before the store: the writer is let go first
    return s, ns_id, c, p


def _save(s: Store, c: Conn, ns_id: bytes, key: bytes, parent: bytes = ROOT, **kw: Any) -> bool:
    offsets = s.begin_save(ns_id, key, parent, SCOPE, **kw)
    if offsets is None:
        return False
    c.write(key, offsets)
    s.commit_save(ns_id, key)
    return True


def _staging(s: Store) -> dict[str, int]:
    st: dict[str, int] = s.stats()["staging"]
    return st


def test_staged_block_leaves_ram_once_on_disk(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, c, _ = _staging_store(opened, _cfg(tmp_path))
    st = s.stats()
    assert st["by_namespace"][ns_id.hex()]["staging"] is True
    assert st["staging"] == {
        "namespaces": 1, "in_flight": 0, "save_refused": 0, "save_waited": 0,
        "write_dropped": 0, "released": 0,
    }  # fmt: skip
    a = _k("a")
    assert _save(s, c, ns_id, a)
    s.flush()
    st = s.stats()
    assert st["ram_used"] == 0
    assert s.lookup(ns_id, [a]) == 1  # on NVMe
    assert st["staging"]["released"] == 1 and st["staging"]["in_flight"] == 0
    assert st["evictions"] == 0


def test_pinned_staged_block_leaves_at_end_load(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, c, p = _staging_store(opened, _cfg(tmp_path), pause=True)
    assert p is not None
    a = _k("a")
    assert _save(s, c, ns_id, a)
    h = s.begin_load(ns_id, [a])
    assert len(h.slots) == 1 and h.nvme_blocks == 0  # pinned while its write waits
    p.go.set()
    s.flush()
    assert s.stats()["ram_used"] == SLOT  # durable now, but pinned: it stays
    assert c.matches(a, h.slots[0])
    s.end_load(h.id)
    st = s.stats()
    assert st["ram_used"] == 0 and st["staging"]["released"] == 1 and st["evictions"] == 0
    assert s.lookup(ns_id, [a]) == 1


def test_promoted_block_leaves_ram_at_end_load(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, c, _ = _staging_store(opened, _cfg(tmp_path))
    a = _k("a")
    assert _save(s, c, ns_id, a)
    s.flush()
    assert s.stats()["ram_used"] == 0
    h = s.begin_load(ns_id, [a])
    assert len(h.slots) == 1 and h.nvme_blocks == 1
    assert c.matches(a, h.slots[0])
    assert s.stats()["ram_used"] == SLOT and _staging(s)["in_flight"] == 1
    s.end_load(h.id)
    st = s.stats()
    assert st["ram_used"] == 0 and st["staging"]["released"] == 2
    assert st["staging"]["in_flight"] == 0 and st["evictions"] == 0


def test_full_write_queue_aborts_instead_of_keeping_ram(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_mod, "WRITE_QUEUE_MAX", 1)
    s, ns_id, c, p = _staging_store(opened, _cfg(tmp_path), pause=True)
    assert p is not None
    first, queued, full = _k("first"), _k("queued"), _k("full")
    assert _save(s, c, ns_id, first)
    assert p.started.wait(5)  # the writer holds the first, so the queue is empty again
    assert _save(s, c, ns_id, queued)  # fills the queue of one
    assert s.begin_save(ns_id, full, ROOT, SCOPE) is not None
    s.commit_save(ns_id, full)  # finds the queue full
    assert s.lookup(ns_id, [full]) == 0
    assert s.stats()["ram_used"] == 2 * SLOT
    assert _staging(s)["write_dropped"] == 1
    p.go.set()
    s.flush()
    st = s.stats()
    assert st["ram_used"] == 0 and st["evictions"] == 0
    assert st["staging"]["write_dropped"] == 1 and st["staging"]["released"] == 2
    assert s.lookup(ns_id, [full]) == 0 and s.lookup(ns_id, [queued]) == 1


def test_no_room_on_disk_drops_the_block(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, c, _ = _staging_store(opened, _cfg(tmp_path, cap=100))  # under one record
    a = _k("a")
    assert _save(s, c, ns_id, a)
    s.flush()
    st = s.stats()
    assert s.lookup(ns_id, [a]) == 0 and st["ram_used"] == 0
    assert st["staging"]["write_dropped"] == 1 and st["evictions"] == 0


def test_nvme_off_mid_run_empties_ram_and_refuses_saves(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="lmw_kv.store")
    s, ns_id, c, _ = _staging_store(opened, _cfg(tmp_path))
    real = Manifest.put
    failed: list[bool] = []

    def put_once(self: Manifest, *args: Any, **kw: Any) -> None:
        if not failed:
            failed.append(True)
            raise sqlite3.OperationalError("disk I/O error")
        real(self, *args, **kw)

    monkeypatch.setattr(Manifest, "put", put_once)
    a = _k("a")
    assert _save(s, c, ns_id, a)
    s.flush()
    st = s.stats()
    assert failed and st["nvme_enabled"] is False
    assert st["ram_used"] == 0 and s.lookup(ns_id, [a]) == 0
    assert st["staging"]["write_dropped"] == 1
    for name in ("b", "c", "d"):
        assert s.begin_save(ns_id, _k(name), ROOT, SCOPE) is None
    st = s.stats()
    assert st["staging"]["save_refused"] == 3 and st["ram_used"] == 0
    assert st["evictions"] == 0
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and PHRASE in r.message]
    assert len(warned) == 1


def test_saves_hold_at_most_half_the_pool(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, c, p = _staging_store(opened, _cfg(tmp_path, slots=8), pause=True)
    assert STAGING_SAVE_SHARE == 0.5
    for i in range(4):
        assert _save(s, c, ns_id, _k(f"s{i}"))
    assert s.begin_save(ns_id, _k("s4"), ROOT, SCOPE) is None
    st = s.stats()
    assert st["staging"]["save_refused"] == 1 and st["staging"]["in_flight"] == 4
    assert p is not None
    p.go.set()
    s.flush()
    assert s.stats()["ram_used"] == 0
    assert _save(s, c, ns_id, _k("s4"))  # room again once the writes landed


def test_loads_use_slots_saves_cannot(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, c, _ = _staging_store(opened, _cfg(tmp_path, slots=8))
    durable = [_k(f"d{i}") for i in range(4)]
    parent = ROOT
    for key in durable:
        assert _save(s, c, ns_id, key, parent)
        s.flush()
        parent = key
    assert s.stats()["ram_used"] == 0
    p = Paused(s)
    opened.append(p)
    for i in range(4):
        assert _save(s, c, ns_id, _k(f"s{i}"))
    assert s.begin_save(ns_id, _k("s4"), ROOT, SCOPE) is None  # S7
    h = s.begin_load(ns_id, durable)
    assert len(h.slots) == 4 and h.nvme_blocks == 4
    assert all(c.matches(k, o) for k, o in zip(durable, h.slots, strict=True))
    assert _staging(s)["in_flight"] == 8
    s.end_load(h.id)
    p.go.set()
    s.flush()
    st = s.stats()
    assert st["ram_used"] == 0 and st["evictions"] == 0 and st["staging"]["in_flight"] == 0


def test_staging_needs_every_owner(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id, c, _ = _staging_store(opened, cfg)
    assert s.register_namespace(_ns(), GROUPS, owner="b", nvme=False) == ns_id
    st = s.stats()
    assert st["by_namespace"][ns_id.hex()]["staging"] is False
    assert st["staging"]["namespaces"] == 0
    a = _k("a")
    assert _save(s, c, ns_id, a)
    s.flush()
    assert s.stats()["ram_used"] == SLOT and s.lookup(ns_id, [a]) == 1
    assert cfg.nvme_dir is not None and not list(cfg.nvme_dir.rglob("*.seg"))
    s.release_owner("b")
    assert s.stats()["by_namespace"][ns_id.hex()]["staging"] is True
    b = _k("b")
    assert _save(s, c, ns_id, b, a)
    s.flush()
    st = s.stats()
    assert st["ram_used"] == SLOT  # only the plain RAM block of before
    assert st["staging"]["released"] == 1 and s.lookup(ns_id, [a, b]) == 2


def test_staging_without_nvme_stores_nothing(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="lmw_kv.store")
    s, ns_id, _, _ = _staging_store(opened, _cfg(tmp_path, nvme=False))
    assert s.begin_save(ns_id, _k("a"), ROOT, SCOPE) is None
    assert s.begin_save(ns_id, _k("b"), ROOT, SCOPE) is None
    st = s.stats()
    assert st["staging"]["save_refused"] == 2 and st["ram_used"] == 0
    assert len([r for r in caplog.records if PHRASE in r.message]) == 1


def _fair_pool(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    ram_only: list[bytes],
    durable: list[bytes],
) -> tuple[Store, bytes, bytes, Conn]:
    """Model M1 (tenant X, ``ram+nvme``) fills all 8 slots: ``ram_only`` first (their
    writes never queued, so the oldest), then ``durable``; then a staging namespace of
    model M2 registers."""
    s = Store(_clocked(_cfg(tmp_path, slots=8)))
    opened.append(s)
    s.set_policy(Policy(mode="fair"))
    nx = s.register_namespace(_ns("x"), GROUPS, owner="x", model=M1)
    c = Conn(s.cfg.ram_path, GROUPS)
    opened.append(c)
    real = s._queue.put_nowait
    skip = {(nx, k) for k in ram_only}

    def put(item: Any) -> None:
        if item in skip:
            raise queue.Full
        real(item)

    monkeypatch.setattr(s._queue, "put_nowait", put)
    for key in [*ram_only, *durable]:
        assert _save(s, c, nx, key, tenant=X)
        s.flush()
    ns = s.register_namespace(_ns("stage"), GROUPS, owner="s", model=M2, staging=True)
    assert s.stats()["ram_used"] == 8 * SLOT
    return s, nx, ns, c


def _in_ram(s: Store, ns_id: bytes, key: bytes) -> bool:
    return (ns_id, key) in s._ram


def test_fair_staging_allocation_borrows_only_durable(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ram_only = [_k(f"r{i}") for i in range(4)]
    durable = [_k(f"d{i}") for i in range(4)]
    s, nx, ns, c = _fair_pool(tmp_path, opened, monkeypatch, ram_only, durable)
    staged = _k("staged")
    assert _save(s, c, ns, staged, tenant=S)  # never refused while one is durable
    # The RAM-only leaves are older than every durable position, and all stay.
    assert all(_in_ram(s, nx, k) for k in ram_only)
    assert sum(_in_ram(s, nx, k) for k in durable) == 3
    assert all(s.lookup(nx, [k]) == 1 for k in durable)  # the evicted one is on NVMe
    st = s.stats()
    assert st["by_tenant"][X.hex()]["ram_evictions"] == 1
    assert st["by_tenant"].get(S.hex(), {}).get("ram_evictions", 0) == 0
    s.flush()
    st = s.stats()
    assert st["staging"]["released"] == 1 and st["evictions"] == 1  # the release is not one
    assert st["by_tenant"][X.hex()]["ram_evictions"] == 1
    assert s.lookup(ns, [staged]) == 1


def test_fair_staging_save_with_nothing_durable_is_refused(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ram_only = [_k(f"r{i}") for i in range(8)]
    s, nx, ns, _ = _fair_pool(tmp_path, opened, monkeypatch, ram_only, [])
    assert s.begin_save(ns, _k("staged"), ROOT, SCOPE, tenant=S) is None
    st = s.stats()
    assert st["staging"]["save_refused"] == 1 and st["evictions"] == 0
    assert all(_in_ram(s, nx, k) for k in ram_only)


def test_staging_transfers_never_weigh_on_a_tenant_ram_cap(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    """S8: a capped tenant with a RAM model and a disk-only model. Its staging blocks in
    flight neither refuse its RAM model's save nor evict its stored RAM blocks."""
    s = Store(_clocked(_cfg(tmp_path, slots=8)))
    opened.append(s)
    s.set_policy(Policy(mode="fair", tenants={X: TenantPolicy(max_ram_pct=50)}))  # 4 slots
    nr = s.register_namespace(_ns("ram"), GROUPS, owner="r", model=M1, nvme=False)
    nd = s.register_namespace(_ns("disk"), GROUPS, owner="d", model=M2, staging=True)
    c = Conn(s.cfg.ram_path, GROUPS)
    opened.append(c)
    stored = [_k(f"m{i}") for i in range(3)]
    for key in stored:
        assert _save(s, c, nr, key, tenant=X)
    p = Paused(s)
    opened.append(p)
    for i in range(2):
        assert _save(s, c, nd, _k(f"t{i}"), tenant=X)  # committed, waiting for the disk
    saving = _k("t2")
    assert s.begin_save(nd, saving, ROOT, SCOPE, tenant=X) is not None  # still saving
    assert _staging(s)["in_flight"] == 3
    assert s.usage("ram") == {(M1, X): 3 * SLOT}  # transfers are not storage
    assert _save(s, c, nr, _k("m3"), tenant=X)  # within its cap: 4 stored slots
    st = s.stats()
    assert all(_in_ram(s, nr, k) for k in stored) and st["evictions"] == 0
    assert st["by_tenant"][X.hex()]["ram_evictions"] == 0
    s.abort_save(nd, saving)
    p.go.set()
    s.flush()
    st = s.stats()
    assert st["staging"]["in_flight"] == 0 and st["staging"]["released"] == 2
    assert s.usage("ram") == {(M1, X): 4 * SLOT} and st["evictions"] == 0


def test_staged_isolated_block_stays_invisible_to_other_scopes(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    s, ns_id, _, p = _staging_store(opened, _cfg(tmp_path), pause=True)
    assert p is not None
    a_keys = _a_keys(ns_id)
    tag = scope_tag(SCOPE_A)
    parent = ROOT
    for key in a_keys:
        assert s.begin_save(ns_id, key, parent, SCOPE_A, scope_tag=tag) is not None
        s.commit_save(ns_id, key)
        parent = key
    _check_disjoint(s, ns_id, a_keys)  # in RAM, in flight
    p.go.set()
    s.flush()
    assert s.stats()["ram_used"] == 0
    _check_disjoint(s, ns_id, a_keys)  # on NVMe
    # A block in flight when its scope is retired is never served, then dropped.
    p.go.clear()
    late = _k("late")
    assert s.begin_save(ns_id, late, a_keys[-1], SCOPE_A, scope_tag=tag) is not None
    s.commit_save(ns_id, late)
    assert s.lookup(ns_id, [*a_keys, late]) == 5
    assert s.retire_scopes([tag]) == 1
    assert s.lookup(ns_id, a_keys) == 0 and s.lookup(ns_id, [late]) == 0
    h = s.begin_load(ns_id, [*a_keys, late])
    assert h.slots == []
    s.end_load(h.id)
    p.go.set()
    s.flush()
    st = s.stats()
    assert st["ram_used"] == 0 and st["staging"]["write_dropped"] == 1
    assert s.lookup(ns_id, [late]) == 0


def test_a_save_at_the_cap_waits_for_a_write_to_land(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    """Back-pressure: a prefill step can complete more blocks than half the pool while
    the disk is still writing the last step's. The save waits (lock released) for a
    write to land instead of being refused, which would break the prompt's chain."""
    s, ns_id, c, p = _staging_store(opened, _cfg(tmp_path, slots=8), pause=True)
    assert p is not None
    for i in range(4):
        assert _save(s, c, ns_id, _k(f"s{i}"))
    got: list[Any] = []
    t = threading.Thread(target=lambda: got.append(s.begin_save(ns_id, _k("s4"), ROOT, SCOPE)))
    t.start()
    t.join(0.1)
    assert t.is_alive()  # waiting, not refused
    p.go.set()
    t.join(5)
    assert got and got[0] is not None
    st = _staging(s)
    assert st["save_refused"] == 0 and st["save_waited"] == 1


def test_a_save_waits_at_most_the_budget_then_refuses_at_once_until_room(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled disk costs one bounded wait, not one per save: after a wait runs out,
    saves are refused at once until a staging write lands again."""
    monkeypatch.setattr(store_mod, "STAGING_SAVE_WAIT_S", 0.2)
    s, ns_id, c, p = _staging_store(opened, _cfg(tmp_path, slots=8), pause=True)
    assert p is not None
    for i in range(4):
        assert _save(s, c, ns_id, _k(f"s{i}"))
    t0 = time.monotonic()
    assert s.begin_save(ns_id, _k("s4"), ROOT, SCOPE) is None
    assert 0.15 <= time.monotonic() - t0 < 2
    t0 = time.monotonic()
    assert s.begin_save(ns_id, _k("s5"), ROOT, SCOPE) is None
    assert time.monotonic() - t0 < 0.1  # stalled: no second wait
    assert _staging(s)["save_refused"] == 2
    p.go.set()
    s.flush()
    assert _save(s, c, ns_id, _k("s4"))  # a write landed: room and waits are back


def test_saves_wait_past_the_stuck_limit_while_writes_keep_landing(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk slower than the prefill is not a stuck disk: while writes land, a save
    waits up to the total budget, so the prefill slows down and nothing is lost."""
    monkeypatch.setattr(store_mod, "STAGING_SAVE_WAIT_S", 0.15)
    monkeypatch.setattr(store_mod, "STAGING_SAVE_WAIT_MAX_S", 5.0)
    s, ns_id, c, _ = _staging_store(opened, _cfg(tmp_path, slots=8))
    real = s._write_through_once

    def paced(ik: Any) -> None:
        time.sleep(0.1)  # one write lands every 0.1 s
        real(ik)

    s._write_through_once = paced  # type: ignore[method-assign]
    for i in range(4):
        assert _save(s, c, ns_id, _k(f"s{i}"))
    got: list[Any] = []

    def save(name: str) -> None:
        got.append(_save(s, c, ns_id, _k(name)))

    ts = [threading.Thread(target=save, args=(f"w{i}",)) for i in range(4)]
    t0 = time.monotonic()
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert time.monotonic() - t0 > 0.15  # the last one waited past the stuck limit
    assert got == [True] * 4
    st = _staging(s)
    assert st["save_refused"] == 0 and st["save_waited"] == 4
    s.flush()
