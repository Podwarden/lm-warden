"""``reuse.db``: one row per position, one transaction per write (spec 6, counters 3.7)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lmw_kv.reuse import REUSE_NAME, PosCounters
from lmw_kv.reuse_db import ReuseDb, ReuseDbError

M = b"\x01" * 16
A, B = b"\x0a" * 16, b"\x0b" * 16


def _c(**kw: object) -> PosCounters:
    base: dict[str, object] = {"first_saved": 5.0, "saver": A, "depth": 3}
    return PosCounters(**{**base, **kw})  # type: ignore[arg-type]


def _count(path: Path) -> int:
    db = sqlite3.connect(path)
    try:
        return int(db.execute("SELECT count(*) FROM reuse").fetchone()[0])
    finally:
        db.close()


def test_rows_round_trip(tmp_path: Path) -> None:
    path = tmp_path / REUSE_NAME
    full = _c(
        loads=7, reloads=5, saver_reloads=2, ram_reloads=3, nvme_reloads=2, last_reload=99.5,
        d=4.25, t=88.0, tenants_seen=(A, B), many=True, nvme_written=12345,
    )  # fmt: skip
    bare = _c(depth=-1)
    db = ReuseDb(path)
    assert db.rows() == []
    db.write([((M, b"p1"), full), ((M, b"p2"), bare)], [])
    db.close()
    db = ReuseDb(path)
    got = dict(db.rows())
    assert got == {(M, b"p1"): full, (M, b"p2"): bare}
    assert mode_of(path) == "wal"
    db.write([((M, b"p1"), _c(reloads=9))], [(M, b"p2")])
    assert dict(db.rows()) == {(M, b"p1"): _c(reloads=9)}
    db.close()


def mode_of(path: Path) -> str:
    db = sqlite3.connect(path)
    try:
        return str(db.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        db.close()


def test_write_is_one_transaction(tmp_path: Path) -> None:
    path = tmp_path / REUSE_NAME
    db = ReuseDb(path)
    db.write([((M, b"p1"), _c(reloads=1))], [])
    bad = _c()
    bad.saver = None  # type: ignore[assignment]  # cannot be stored
    with pytest.raises(ReuseDbError):
        db.write([((M, b"p1"), _c(reloads=2)), ((M, b"p2"), bad)], [(M, b"p1")])
    assert dict(db.rows()) == {(M, b"p1"): _c(reloads=1)}
    db.close()
    assert _count(path) == 1


def test_a_directory_is_an_error(tmp_path: Path) -> None:
    (tmp_path / REUSE_NAME).mkdir()
    with pytest.raises(ReuseDbError):
        ReuseDb(tmp_path / REUSE_NAME)
