"""Reuse counters across a restart: ``reuse.db`` (spec 6, counters design 3.7)."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from lmw_kv.reuse import REUSE_NAME, PosCounters
from lmw_kv.reuse_db import ReuseDb, ReuseDbError
from lmw_kv.store import Store
from tests.unit.lmw_kv.test_store import Conn, _k, opened  # noqa: F401  (fixture)
from tests.unit.lmw_kv.test_store_reuse import (
    M1,
    TAG,
    A,
    _Clock,
    _durable_on_nvme_only,
    _reg,
    _row,
    _save,
    _store,
    _use,
)


def _file(tmp_path: Path) -> Path:
    return tmp_path / "nvme" / REUSE_NAME


def _on_disk(tmp_path: Path) -> dict[tuple[bytes, bytes], PosCounters]:
    db = ReuseDb(_file(tmp_path))
    try:
        return dict(db.rows())
    finally:
        db.close()


def _restart(opened: list[Store | Conn], s: Store, tmp_path: Path, **kw: object) -> Store:  # noqa: F811
    s.close()
    return _store(opened, tmp_path, **kw)  # type: ignore[arg-type]


def test_counters_survive_restart_for_nvme_positions(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    clock = _Clock()
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    s.cfg.clock = clock
    assert _use(s, ns_id, [first], "r1") == 1
    clock.now += 100.0
    assert _use(s, ns_id, [first], "r2") == 1
    before = _row(s, first)
    assert before.reloads == 2
    kept = PosCounters(**vars(before))
    s.persist_reuse()
    s2 = _restart(opened, s, tmp_path)
    after = _row(s2, first)
    assert after.reloads == 2
    assert (after.d, after.t) == (kept.d, kept.t)
    assert after == kept
    assert s2.stats()["reuse"]["reuse_persist_degraded"] is False


def test_close_persists_last(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    _use(s, ns_id, [first], "r1")
    s.close()  # no persist_reuse call
    assert _on_disk(tmp_path)[(M1, first)].reloads == 1


def test_ram_only_positions_are_not_kept(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    ns_id = _reg(s, nvme=False)
    key = _k("ram-only")
    _save(s, ns_id, key)
    _use(s, ns_id, [key], "t")
    assert (M1, key) in s._reuse.rows
    s.persist_reuse()
    assert _on_disk(tmp_path) == {}
    s2 = _restart(opened, s, tmp_path)
    assert (M1, key) not in s2._reuse.rows
    assert _on_disk(tmp_path) == {}


def test_row_without_an_nvme_member_is_deleted_on_open(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s = _store(opened, tmp_path)
    s.close()
    db = ReuseDb(_file(tmp_path))
    db.write([((M1, b"gone"), PosCounters(first_saved=1.0, saver=A, depth=0, reloads=4))], [])
    db.close()
    s2 = _store(opened, tmp_path)
    assert (M1, b"gone") not in s2._reuse.rows
    assert _on_disk(tmp_path) == {}


def test_nvme_position_without_a_row_is_seeded(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    s.close()
    con = sqlite3.connect(_file(tmp_path))
    con.execute("DELETE FROM reuse")
    con.commit()
    con.close()
    s2 = _store(opened, tmp_path)
    row = _row(s2, first)
    oldest = min(r.last_used for ik, r in s2._nv.items() if ik[1] == first)
    assert (row.reloads, row.d, row.depth, row.saver) == (0, 1.0, -1, A)
    assert row.first_saved == row.t == oldest
    assert (M1, first) in _on_disk(tmp_path)  # written at open


def test_unwritable_reuse_db_degrades_open(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "nvme").mkdir()
    _file(tmp_path).mkdir()
    with caplog.at_level(logging.WARNING):
        s = _store(opened, tmp_path)
    assert s.stats()["reuse"]["reuse_persist_degraded"] is True
    ns_id = _reg(s)
    mine, other = _k("mine"), _k("open")
    _save(s, ns_id, mine, scope_tag=TAG)
    _save(s, ns_id, other)
    assert _use(s, ns_id, [mine], "r1") == 1  # isolation does not depend on this file
    assert _use(s, ns_id, [other], "r2") == 1
    s.persist_reuse()  # no file: a no-op, never an error
    assert _row(s, mine).reloads == 1 and _row(s, other).reloads == 1
    assert s.stats()["isolation_refused"] is False
    assert sum("reuse" in r.getMessage() for r in caplog.records) == 1


def test_a_failed_write_degrades_keeps_rows_dirty_and_recovers(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    _use(s, ns_id, [first], "r1")
    real = ReuseDb.write
    fail = {"on": True}

    def flaky(self: ReuseDb, ups, dels):  # type: ignore[no-untyped-def]
        if fail["on"]:
            raise ReuseDbError("disk full")
        return real(self, ups, dels)

    monkeypatch.setattr(ReuseDb, "write", flaky)
    with caplog.at_level(logging.WARNING):
        s.persist_reuse()
        s.persist_reuse()
    assert s.stats()["reuse"]["reuse_persist_degraded"] is True
    assert sum("reuse" in r.getMessage() for r in caplog.records) == 1
    assert _row(s, first).reloads == 1  # counting goes on
    fail["on"] = False
    s.persist_reuse()
    assert s.stats()["reuse"]["reuse_persist_degraded"] is False
    assert _on_disk(tmp_path)[(M1, first)].reloads == 1


def test_persist_writes_outside_the_lock(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s, ns_id, first = _durable_on_nvme_only(opened, tmp_path)
    _use(s, ns_id, [first], "r1")
    inside, release = threading.Event(), threading.Event()
    real = ReuseDb.write

    def slow(self: ReuseDb, ups, dels):  # type: ignore[no-untyped-def]
        inside.set()
        assert release.wait(10)
        return real(self, ups, dels)

    monkeypatch.setattr(ReuseDb, "write", slow)
    t = threading.Thread(target=s.persist_reuse)
    t.start()
    try:
        assert inside.wait(10)
        got: list[int] = []
        th = threading.Thread(target=lambda: got.append(s.lookup(ns_id, [first])))
        th.start()
        th.join(5)
        assert not th.is_alive() and got == [1]  # returned while the write is waiting
        assert not release.is_set()
    finally:
        release.set()
        t.join(10)
    assert not t.is_alive()
