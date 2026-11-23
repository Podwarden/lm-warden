"""SQLite manifest of the blocks held in NVMe segments."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lmw_kv.format import Location

# Step 1's table, unchanged: a step 1 manifest on disk already has exactly this.
_TABLE = """
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
)
"""

# Columns added after step 1, upgraded in place with ALTER TABLE. Old rows read as b"":
# unattributed tenant, no scope tag, the namespace's one model, position unknown.
_ADDED_COLUMNS = (
    ("model", "BLOB NOT NULL DEFAULT x''"),
    ("pos", "BLOB NOT NULL DEFAULT x''"),
    ("tenant", "BLOB NOT NULL DEFAULT x''"),
    ("scope_tag", "BLOB NOT NULL DEFAULT x''"),
)

# Created only after every ALTER: an index on a new column fails on a step 1 table.
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS blocks_parent ON blocks (ns, parent)",
    "CREATE INDEX IF NOT EXISTS blocks_lru ON blocks (last_used)",
    "CREATE INDEX IF NOT EXISTS blocks_seg ON blocks (ns, seg)",
    "CREATE INDEX IF NOT EXISTS blocks_tenant_lru ON blocks (model, tenant, last_used)",
    # Retiring a scope finds its blocks by tag (spec 4.5: a retired scope is class dead).
    "CREATE INDEX IF NOT EXISTS blocks_scope_tag ON blocks (ns, scope_tag)",
)


@dataclass(frozen=True)
class Entry:
    parent: bytes
    group: str
    loc: Location
    nbytes: int
    scope: str
    last_used: float
    model: bytes = b""
    pos: bytes = b""
    tenant: bytes = field(default=b"", repr=False)
    scope_tag: bytes = field(default=b"", repr=False)


@dataclass(frozen=True)
class PosRow:
    """One stored block, all its groups folded together."""

    key: bytes
    parent: bytes
    model: bytes
    pos: bytes
    tenant: bytes = field(repr=False)
    scope_tag: bytes = field(repr=False)
    nbytes: int  # summed over groups
    last_used: float  # the oldest group's
    segments: tuple[tuple[int, int], ...]  # (segment, record bytes of this block in it)


class Manifest:
    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(_TABLE)
            have = {r[1] for r in self._db.execute("PRAGMA table_info(blocks)").fetchall()}
            for name, decl in _ADDED_COLUMNS:
                if name not in have:
                    self._db.execute(f"ALTER TABLE blocks ADD COLUMN {name} {decl}")
            for stmt in _INDEXES:
                self._db.execute(stmt)
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def journal_mode(self) -> str:
        with self._lock:
            return str(self._db.execute("PRAGMA journal_mode").fetchone()[0])

    def put(
        self,
        ns_id: bytes,
        block_key: bytes,
        parent: bytes,
        group: str,
        loc: Location,
        nbytes: int,
        scope: str,
        *,
        model: bytes = b"",
        pos: bytes = b"",
        tenant: bytes = b"",
        scope_tag: bytes = b"",
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO blocks (ns, key, parent, grp, seg, off, len, nbytes, "
                "scope, last_used, model, pos, tenant, scope_tag) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ns_id, block_key, parent, group, loc.segment, loc.offset, loc.length,
                    nbytes, scope, time.time(), model, pos, tenant, scope_tag,
                ),
            )  # fmt: skip

    def get(self, ns_id: bytes, block_key: bytes) -> list[Entry]:
        with self._lock:
            rows = self._db.execute(
                "SELECT parent, grp, seg, off, len, nbytes, scope, last_used, "
                "model, pos, tenant, scope_tag "
                "FROM blocks WHERE ns=? AND key=? ORDER BY grp",
                (ns_id, block_key),
            ).fetchall()
        return [
            Entry(
                r[0], r[1], Location(r[2], r[3], r[4]), r[5], r[6], r[7],
                model=r[8], pos=r[9], tenant=r[10], scope_tag=r[11],
            )
            for r in rows
        ]  # fmt: skip

    def set_tenant(self, ns_id: bytes, block_key: bytes, tenant: bytes) -> None:
        """Charge every group of the block to ``tenant`` ("last user pays")."""
        with self._lock:
            self._db.execute(
                "UPDATE blocks SET tenant=? WHERE ns=? AND key=?", (tenant, ns_id, block_key)
            )

    def set_tenants(self, ns_id: bytes, changes: Sequence[tuple[bytes, bytes]]) -> None:
        """``set_tenant`` for many ``(block key, tenant)`` pairs in one transaction."""
        if not changes:
            return
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.executemany(
                    "UPDATE blocks SET tenant=? WHERE ns=? AND key=?",
                    [(tenant, ns_id, key) for key, tenant in changes],
                )
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def iter_blocks(self, ns_id: bytes) -> Iterator[PosRow]:
        """Every block of the namespace, its groups folded into one row (key order).

        The attribution columns come from the block's first group by name; they are
        written together for every group, so the groups agree.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT key, parent, model, pos, tenant, scope_tag, nbytes, last_used, seg, len "
                "FROM blocks WHERE ns=? ORDER BY key, grp",
                (ns_id,),
            ).fetchall()
        i = 0
        while i < len(rows):
            first = rows[i]
            key = first[0]
            nbytes = 0
            last_used = first[7]
            segs: dict[int, int] = {}
            while i < len(rows) and rows[i][0] == key:
                r = rows[i]
                nbytes += r[6]
                last_used = min(last_used, r[7])
                segs[r[8]] = segs.get(r[8], 0) + r[9]
                i += 1
            yield PosRow(
                key=key, parent=first[1], model=first[2], pos=first[3], tenant=first[4],
                scope_tag=first[5], nbytes=nbytes, last_used=last_used,
                segments=tuple(sorted(segs.items())),
            )  # fmt: skip

    def longest_prefix(self, ns_id: bytes, keys: Sequence[bytes]) -> int:
        """Number of leading keys that are present."""
        n = 0
        with self._lock:
            for k in keys:
                row = self._db.execute(
                    "SELECT 1 FROM blocks WHERE ns=? AND key=? LIMIT 1", (ns_id, k)
                ).fetchone()
                if row is None:
                    break
                n += 1
        return n

    def delete(self, ns_id: bytes, block_key: bytes) -> None:
        with self._lock:
            self._db.execute("DELETE FROM blocks WHERE ns=? AND key=?", (ns_id, block_key))

    def leaves_lru(self, ns_id: bytes, limit: int) -> list[bytes]:
        """Keys with no stored child, least recently used first (prefix-closed eviction)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT b.key FROM blocks b WHERE b.ns=? "
                "AND NOT EXISTS (SELECT 1 FROM blocks c WHERE c.ns=b.ns AND c.parent=b.key) "
                "GROUP BY b.key ORDER BY MIN(b.last_used), b.key LIMIT ?",
                (ns_id, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def touch(self, ns_id: bytes, block_key: bytes) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE blocks SET last_used=? WHERE ns=? AND key=?",
                (time.time(), ns_id, block_key),
            )

    def bytes_used(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COALESCE(SUM(nbytes),0) FROM blocks").fetchone()[0])

    def has_child(self, ns_id: bytes, block_key: bytes) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM blocks WHERE ns=? AND parent=? LIMIT 1", (ns_id, block_key)
            ).fetchone()
        return row is not None

    def segment_in_use(self, ns_id: bytes, segment: int) -> bool:
        """Whether any record still lives in ``segment`` (a dead segment can be deleted)."""
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM blocks WHERE ns=? AND seg=? LIMIT 1", (ns_id, segment)
            ).fetchone()
        return row is not None

    def keys_in_segment(self, ns_id: bytes, segment: int) -> list[bytes]:
        """Every block with at least one record in ``segment``.

        No ``DISTINCT`` in SQL: with it the planner walks the primary key (the whole
        namespace) instead of ``blocks_seg``; the groups of one key are folded here.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT key FROM blocks WHERE ns=? AND seg=?", (ns_id, segment)
            ).fetchall()
        return list(dict.fromkeys(r[0] for r in rows))

    def children(self, ns_id: bytes, block_key: bytes) -> list[bytes]:
        """The stored blocks whose parent is ``block_key`` (``blocks_parent``; see
        ``keys_in_segment`` for why there is no ``DISTINCT``)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key FROM blocks WHERE ns=? AND parent=?", (ns_id, block_key)
            ).fetchall()
        return list(dict.fromkeys(r[0] for r in rows))

    def block_count(self) -> int:
        with self._lock:
            return int(
                self._db.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM blocks GROUP BY ns, key)"
                ).fetchone()[0]
            )
