"""A store opens step 1 NVMe directories: their manifests upgrade in place."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from lmw_kv.store import MANIFEST_NAME
from tests.unit.lmw_kv.test_manifest import _STEP1_SCHEMA
from tests.unit.lmw_kv.test_store import (
    GROUPS,
    Conn,
    Store,
    _cfg,
    _chain,
    _open,
    _save,
    opened,  # noqa: F401  (fixture)
)

NEW_KEY = b"\x42" * 32
_STEP1_COLUMNS = "ns, key, parent, grp, seg, off, len, nbytes, scope, last_used"


def _rewrite_as_step1(manifest: Path) -> int:
    """Replace ``manifest`` with a step 1 file holding the same rows; returns the row count."""
    src = sqlite3.connect(manifest)
    rows = src.execute(f"SELECT {_STEP1_COLUMNS} FROM blocks").fetchall()
    src.close()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{manifest}{suffix}").unlink(missing_ok=True)
    db = sqlite3.connect(manifest)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(_STEP1_SCHEMA)
    db.executemany("INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()
    return len(rows)


def test_store_opens_step1_manifest_with_nvme_on(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id, c = _open(opened, cfg)
    chain = _chain(["a", "b"])
    for key, parent in chain:
        assert _save(s, c, ns_id, key, parent)
    s.flush()
    c.close()
    s.close()
    opened.clear()

    assert cfg.nvme_dir is not None
    manifest = cfg.nvme_dir / ns_id.hex() / MANIFEST_NAME
    assert _rewrite_as_step1(manifest) == 2 * len(GROUPS)
    cols = {r[1] for r in sqlite3.connect(manifest).execute("PRAGMA table_info(blocks)")}
    assert "tenant" not in cols  # really a step 1 file

    s2, ns2, c2 = _open(opened, cfg)
    assert ns2 == ns_id
    assert s2.stats()["nvme_enabled"] is True
    keys = [k for k, _ in chain]
    assert s2.lookup(ns_id, keys) == 2
    h = s2.begin_load(ns_id, keys)
    assert h.nvme_blocks == 2
    for key, offsets in zip(keys, h.slots, strict=True):
        assert c2.matches(key, offsets)
    s2.end_load(h.id)
    assert s2.stats()["nvme_enabled"] is True
    # Old rows read as unattributed: tenant "" and the namespace's one model.
    m = s2._manifests[ns_id]
    old = [e for k in keys for e in m.get(ns_id, k)]
    assert len(old) == 2 * len(GROUPS)
    assert {(e.model, e.pos, e.tenant, e.scope_tag) for e in old} == {(b"", b"", b"", b"")}
    assert _save(s2, c2, ns_id, NEW_KEY, keys[-1])  # new writes go to the upgraded file
    s2.flush()
    assert s2.stats()["nvme_enabled"] is True
    assert len(m.get(ns_id, NEW_KEY)) == len(GROUPS)


def test_damaged_manifest_switches_nvme_off(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id, c = _open(opened, cfg)
    c.close()
    s.close()
    opened.clear()
    assert cfg.nvme_dir is not None
    manifest = cfg.nvme_dir / ns_id.hex() / MANIFEST_NAME
    for suffix in ("-wal", "-shm"):
        Path(f"{manifest}{suffix}").unlink(missing_ok=True)
    manifest.write_bytes(os.urandom(8192))

    s2, ns2, c2 = _open(opened, cfg)
    assert s2.stats()["nvme_enabled"] is False
    chain = _chain(["x", "y"])
    for key, parent in chain:
        assert _save(s2, c2, ns2, key, parent)
    keys = [k for k, _ in chain]
    assert s2.lookup(ns2, keys) == 2
    h = s2.begin_load(ns2, keys)
    assert len(h.slots) == 2
    for key, offsets in zip(keys, h.slots, strict=True):
        assert c2.matches(key, offsets)
    s2.end_load(h.id)
