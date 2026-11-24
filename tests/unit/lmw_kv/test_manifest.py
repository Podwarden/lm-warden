from __future__ import annotations

import sqlite3
from pathlib import Path

from lmw_kv.format import Location
from lmw_kv.manifest import Manifest

NS = b"N" * 32
OTHER = b"O" * 32


def _k(i: int) -> bytes:
    return bytes([i + 1]) * 32


def _put(m: Manifest, i: int, parent: bytes, ns: bytes = NS, group: str = "full") -> None:
    m.put(ns, _k(i), parent, group, Location(1, i * 100, 90), 90, "tenant")


def test_longest_prefix(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    parent = b"\xff" * 32
    for i in range(5):
        _put(m, i, parent)
        parent = _k(i)
    assert m.longest_prefix(NS, [_k(0), _k(1), _k(2), _k(99), _k(4)]) == 3
    assert m.longest_prefix(NS, [_k(99)]) == 0
    assert m.longest_prefix(NS, []) == 0
    assert m.longest_prefix(OTHER, [_k(0)]) == 0


def test_put_get_delete_and_bytes_used(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    root = b"\xff" * 32
    _put(m, 1, root, group="full")
    _put(m, 1, root, group="swa")
    _put(m, 2, _k(1))
    entries = m.get(NS, _k(1))
    assert {e.group for e in entries} == {"full", "swa"}
    e = next(e for e in entries if e.group == "full")
    assert (e.loc, e.nbytes, e.scope, e.parent) == (Location(1, 100, 90), 90, "tenant", root)
    assert m.bytes_used() == 270
    m.delete(NS, _k(1))
    assert m.get(NS, _k(1)) == []
    assert m.bytes_used() == 90
    m.delete(NS, _k(1))  # idempotent


def test_leaves_lru_is_prefix_closed(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    _put(m, 0, b"\xff" * 32)
    _put(m, 1, _k(0))
    _put(m, 2, _k(1))
    assert m.leaves_lru(NS, 10) == [_k(2)]
    m.delete(NS, _k(2))
    assert m.leaves_lru(NS, 10) == [_k(1)]
    m.delete(NS, _k(1))
    assert m.leaves_lru(NS, 10) == [_k(0)]


def test_leaves_lru_orders_by_last_used_and_touch(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    root = b"\xff" * 32
    _put(m, 1, root)
    _put(m, 2, root)
    _put(m, 3, root)
    assert m.leaves_lru(NS, 10) == [_k(1), _k(2), _k(3)]
    m.touch(NS, _k(1))
    assert m.leaves_lru(NS, 10) == [_k(2), _k(3), _k(1)]
    assert m.leaves_lru(NS, 2) == [_k(2), _k(3)]


def test_persists_and_uses_wal(tmp_path: Path) -> None:
    path = tmp_path / "manifest.db"
    m = Manifest(path)
    _put(m, 1, b"\xff" * 32)
    m.close()
    m2 = Manifest(path)
    assert len(m2.get(NS, _k(1))) == 1
    assert m2.journal_mode() == "wal"


def test_has_child_segment_in_use_and_block_count(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    root = b"\xff" * 32
    _put(m, 1, root, group="full")
    _put(m, 1, root, group="swa")
    _put(m, 2, _k(1))
    m.put(NS, _k(3), _k(2), "full", Location(7, 0, 90), 90, "tenant")
    assert m.block_count() == 3
    assert m.has_child(NS, _k(1))
    assert not m.has_child(NS, _k(3))
    assert not m.has_child(OTHER, _k(1))
    assert m.segment_in_use(NS, 1)
    assert m.segment_in_use(NS, 7)
    assert not m.segment_in_use(NS, 2)
    m.delete(NS, _k(3))
    assert not m.segment_in_use(NS, 7)
    assert m.block_count() == 2


# Step 1's schema, copied literally: the manifest every step 1 store has on disk.
_STEP1_SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    ns BLOB NOT NULL,
    key BLOB NOT NULL,
    parent BLOB NOT NULL,
    grp TEXT NOT NULL,
    seg INTEGER NOT NULL,
    off INTEGER NOT NULL,
    len INTEGER NOT NULL,
    nbytes INTEGER NOT NULL,
    scope TEXT NOT NULL,
    last_used REAL NOT NULL,
    PRIMARY KEY (ns, key, grp)
);
CREATE INDEX IF NOT EXISTS blocks_parent ON blocks (ns, parent);
CREATE INDEX IF NOT EXISTS blocks_lru ON blocks (last_used);
CREATE INDEX IF NOT EXISTS blocks_seg ON blocks (ns, seg);
"""


def test_step1_manifest_upgrades_in_place(tmp_path: Path) -> None:
    path = tmp_path / "manifest.db"
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(_STEP1_SCHEMA)
    db.execute(
        "INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?)",
        (NS, _k(1), b"\xff" * 32, "full", 1, 0, 90, 90, "tenant", 123.0),
    )
    db.commit()
    db.close()

    m = Manifest(path)
    (old,) = m.get(NS, _k(1))
    assert (old.model, old.pos, old.tenant, old.scope_tag) == (b"", b"", b"", b"")
    assert (old.loc, old.nbytes, old.scope, old.last_used) == (
        Location(1, 0, 90), 90, "tenant", 123.0,
    )  # fmt: skip
    m.put(
        NS, _k(2), _k(1), "full", Location(1, 100, 90), 90, "tenant",
        model=b"M" * 16, pos=b"P" * 32, tenant=b"T" * 16, scope_tag=b"S" * 16,
    )  # fmt: skip
    (new,) = m.get(NS, _k(2))
    assert (new.model, new.pos, new.tenant, new.scope_tag) == (
        b"M" * 16, b"P" * 32, b"T" * 16, b"S" * 16,
    )  # fmt: skip
    names = {r[1] for r in m._db.execute("PRAGMA index_list(blocks)").fetchall()}
    assert {
        "blocks_parent", "blocks_lru", "blocks_seg", "blocks_tenant_lru", "blocks_scope_tag",
    } <= names  # fmt: skip
    m.close()

    again = Manifest(path)  # a second open of an upgraded file is a no-op
    assert len(again.get(NS, _k(1))) == 1
    assert again.get(NS, _k(2))[0].tenant == b"T" * 16
    again.close()


def test_entry_repr_hides_tenant_and_scope_tag(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    m.put(
        NS, _k(1), b"\xff" * 32, "full", Location(1, 0, 90), 90, "t",
        tenant=b"\xaa" * 16, scope_tag=b"\xbb" * 16,
    )  # fmt: skip
    (e,) = m.get(NS, _k(1))
    (row,) = m.iter_blocks(NS)
    for text in (repr(e), repr(row)):
        assert "\\xaa" not in text and "\\xbb" not in text


def test_set_tenant_moves_every_group(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    root = b"\xff" * 32
    for g in ("full", "swa"):
        m.put(NS, _k(1), root, g, Location(1, 0, 90), 90, "t", tenant=b"A" * 16)
    m.put(NS, _k(2), root, "full", Location(1, 0, 90), 90, "t", tenant=b"A" * 16)
    m.put(OTHER, _k(1), root, "full", Location(1, 0, 90), 90, "t", tenant=b"A" * 16)
    m.set_tenant(NS, _k(1), b"B" * 16)
    assert [e.tenant for e in m.get(NS, _k(1))] == [b"B" * 16, b"B" * 16]
    assert m.get(NS, _k(2))[0].tenant == b"A" * 16
    assert m.get(OTHER, _k(1))[0].tenant == b"A" * 16


def test_iter_blocks_sums_groups(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.db")
    root = b"\xff" * 32
    m.put(
        NS, _k(1), root, "full", Location(3, 0, 50), 10, "t",
        model=b"M" * 16, pos=b"P" * 32, tenant=b"T" * 16, scope_tag=b"S" * 16,
    )  # fmt: skip
    m.put(
        NS, _k(1), root, "swa", Location(4, 0, 70), 30, "t",
        model=b"M" * 16, pos=b"P" * 32, tenant=b"T" * 16, scope_tag=b"S" * 16,
    )  # fmt: skip
    m._db.execute("UPDATE blocks SET last_used=100.0 WHERE grp='full'")
    m._db.execute("UPDATE blocks SET last_used=200.0 WHERE grp='swa'")
    m.put(NS, _k(2), _k(1), "full", Location(4, 70, 50), 10, "t")
    m.put(OTHER, _k(9), root, "full", Location(1, 0, 50), 10, "t")
    rows = {r.key: r for r in m.iter_blocks(NS)}
    assert set(rows) == {_k(1), _k(2)}
    r = rows[_k(1)]
    assert (r.parent, r.model, r.pos, r.tenant, r.scope_tag) == (
        root, b"M" * 16, b"P" * 32, b"T" * 16, b"S" * 16,
    )  # fmt: skip
    assert r.nbytes == 40
    assert r.last_used == 100.0
    assert r.segments == ((3, 50), (4, 70))
    r2 = rows[_k(2)]
    assert (r2.nbytes, r2.segments, r2.tenant, r2.model) == (10, ((4, 50),), b"", b"")
    assert list(m.iter_blocks(b"Z" * 32)) == []


def test_half_upgraded_manifest_completes_its_upgrade(tmp_path: Path) -> None:
    """A crash between ALTERs leaves some new columns: the next open adds the rest."""
    path = tmp_path / "manifest.db"
    db = sqlite3.connect(path)
    db.executescript(_STEP1_SCHEMA)
    db.execute("INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?)",
               (NS, _k(1), b"\xff" * 32, "full", 1, 0, 90, 90, "tenant", 123.0))  # fmt: skip
    db.execute("ALTER TABLE blocks ADD COLUMN model BLOB NOT NULL DEFAULT x''")
    db.execute("ALTER TABLE blocks ADD COLUMN pos BLOB NOT NULL DEFAULT x''")
    db.commit()
    db.close()

    m = Manifest(path)
    cols = [r[1] for r in m._db.execute("PRAGMA table_info(blocks)").fetchall()]
    assert cols[-4:] == ["model", "pos", "tenant", "scope_tag"]
    names = {r[1] for r in m._db.execute("PRAGMA index_list(blocks)").fetchall()}
    assert {"blocks_tenant_lru", "blocks_scope_tag"} <= names
    (old,) = m.get(NS, _k(1))
    assert (old.loc, old.nbytes, old.last_used) == (Location(1, 0, 90), 90, 123.0)
    assert (old.model, old.pos, old.tenant, old.scope_tag) == (b"", b"", b"", b"")
    m.close()


def test_reclaim_queries_use_their_indexes(tmp_path: Path) -> None:
    """The reclaim walk runs under the store lock: no query may scan the namespace."""
    m = Manifest(tmp_path / "manifest.db")
    root = b"\xff" * 32
    _put(m, 1, root, group="full")
    _put(m, 1, root, group="swa")  # two groups of one key: folded to one
    _put(m, 2, _k(1))
    seen: list[str] = []
    m._db.set_trace_callback(seen.append)
    assert m.keys_in_segment(NS, 1) == [_k(1), _k(2)]
    assert m.children(NS, _k(1)) == [_k(2)]
    assert m.segment_in_use(NS, 1)
    assert m.has_child(NS, _k(1))
    m._db.set_trace_callback(None)
    want = ["blocks_seg", "blocks_parent", "blocks_seg", "blocks_parent"]
    assert len(seen) == len(want)
    for sql, index in zip(seen, want, strict=True):
        args = (b"x",) * sql.count("?")  # the trace may or may not expand parameters
        plan = " ".join(r[3] for r in m._db.execute("EXPLAIN QUERY PLAN " + sql, args))
        assert f"INDEX {index} " in plan, (sql, plan)


def test_a_row_write_does_not_wait_for_a_disk_flush(tmp_path: Path) -> None:
    """The writer puts a row while it holds the store lock. ``synchronous=FULL`` (SQLite's
    default) flushes the WAL to the disk on every row, which took 21 ms (up to 0.67 s)
    on a busy virtual disk and stalled every save behind it. In WAL mode ``NORMAL``
    survives a crash of the daemon; only a power loss can drop the last rows (a miss)."""
    m = Manifest(tmp_path / "manifest.db")
    assert m.journal_mode() == "wal"
    assert m.synchronous() == 1  # NORMAL
