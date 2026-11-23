from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from lmw_kv.retired import RETIRED_NAME, RetiredScopes, RetiredStoreError


def _t(i: int) -> bytes:
    return bytes([i]) * 16


def test_retired_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    r = RetiredScopes(path)
    assert r.add([_t(1), _t(2)]) == 2
    r.close()
    r2 = RetiredScopes(path)
    assert _t(1) in r2 and _t(2) in r2 and _t(3) not in r2
    assert len(r2) == 2
    mode = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    r2.close()


def test_retired_in_memory() -> None:
    r = RetiredScopes(None)
    assert len(r) == 0
    assert r.add([_t(1)]) == 1
    assert _t(1) in r and _t(2) not in r
    assert len(r) == 1
    assert "\\x01" not in repr(r)
    r.close()


def test_attach_merges_both_ways(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    disk = RetiredScopes(path)
    disk.add([_t(1), _t(2)])
    disk.close()

    r = RetiredScopes(None)
    r.add([_t(2), _t(3)])
    r.attach(path)
    assert {t for t in (_t(1), _t(2), _t(3), _t(4)) if t in r} == {_t(1), _t(2), _t(3)}
    assert r.add([_t(4)]) == 1  # written through once attached
    r.close()

    tags = {bytes(row[0]) for row in sqlite3.connect(path).execute("SELECT tag FROM retired")}
    assert tags == {_t(1), _t(2), _t(3), _t(4)}


def test_retired_rejects_bad_length(tmp_path: Path) -> None:
    r = RetiredScopes(tmp_path / RETIRED_NAME)
    for bad in (b"", b"x" * 15, b"x" * 17, b"x" * 32):
        with pytest.raises(ValueError):
            r.add([bad])
    with pytest.raises(ValueError):
        r.add([_t(1), b"short"])  # one bad tag retires none
    assert len(r) == 0
    r.close()


def test_add_counts_only_new(tmp_path: Path) -> None:
    r = RetiredScopes(tmp_path / RETIRED_NAME)
    assert r.add([_t(1), _t(1), _t(2)]) == 2
    assert r.add([_t(1), _t(2)]) == 0
    assert r.add([_t(2), _t(3)]) == 1
    assert r.add([]) == 0
    assert len(r) == 3
    r.close()


# ---- fail closed (fix round 1) ------------------------------------------------------

_BOOM = (
    "CREATE TRIGGER IF NOT EXISTS boom BEFORE INSERT ON retired "
    "BEGIN SELECT RAISE(ABORT, 'disk says no'); END"
)


def _sql(path: Path, stmt: str) -> None:
    db = sqlite3.connect(path, isolation_level=None)
    db.execute(stmt)
    db.close()


def _on_disk(path: Path) -> set[bytes]:
    db = sqlite3.connect(path)
    tags = {bytes(row[0]) for row in db.execute("SELECT tag FROM retired")}
    db.close()
    return tags


def test_failing_write_keeps_the_tag_retired_in_memory(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    r = RetiredScopes(path)
    assert not r.persist_degraded
    _sql(path, _BOOM)
    assert r.add([_t(1)]) == 1  # memory is authoritative: no exception for the caller
    assert _t(1) in r
    assert r.persist_degraded
    assert not r.isolation_refused  # still correct in this process
    r.close()


def test_retired_tag_survives_a_failing_persist_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    r = RetiredScopes(path)
    _sql(path, _BOOM)
    r.add([_t(1)])
    assert _on_disk(path) == set()
    _sql(path, "DROP TRIGGER boom")
    r.add([_t(2)])  # the next write also carries the tag that failed before
    assert _on_disk(path) == {_t(1), _t(2)}
    assert not r.persist_degraded
    r.close()
    again = RetiredScopes(path)
    assert _t(1) in again and _t(2) in again
    again.close()


def test_close_retries_a_failed_persist(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    r = RetiredScopes(path)
    _sql(path, _BOOM)
    r.add([_t(1)])
    _sql(path, "DROP TRIGGER boom")
    r.close()
    again = RetiredScopes(path)
    assert _t(1) in again
    again.close()


def test_attach_merges_disk_before_a_failing_insert(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    disk = RetiredScopes(path)
    disk.add([_t(1)])
    disk.close()
    _sql(path, _BOOM)
    r = RetiredScopes(None)
    r.add([_t(2)])
    with pytest.raises(RetiredStoreError):
        r.attach(path)
    assert _t(1) in r and _t(2) in r  # the disk's tags were merged first
    assert r.isolation_refused
    r.close()


def test_corrupt_store_refuses_isolation(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    path.write_bytes(os.urandom(8192))
    r = RetiredScopes(None)
    r.add([_t(1)])
    with pytest.raises(RetiredStoreError):
        r.attach(path)
    assert r.isolation_refused
    assert _t(1) in r
    with pytest.raises(RetiredStoreError):
        RetiredScopes(path)
    r.close()


def test_read_only_store_refuses_isolation(tmp_path: Path) -> None:
    path = tmp_path / RETIRED_NAME
    disk = RetiredScopes(path)
    disk.add([_t(1)])
    disk.close()
    path.chmod(0o444)
    try:
        r = RetiredScopes(None)
        with pytest.raises(RetiredStoreError):
            r.attach(path)
        assert r.isolation_refused
        r.close()
    finally:
        path.chmod(0o644)


def test_in_memory_set_is_not_degraded() -> None:
    r = RetiredScopes(None)
    r.add([_t(1)])
    assert not r.persist_degraded and not r.isolation_refused
    r.close()
