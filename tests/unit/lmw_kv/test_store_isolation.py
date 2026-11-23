"""Isolation in the RAM and NVMe tiers: disjoint keys, retired scopes, fail closed."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from lmw_kv.connector_logic import block_keys_for, fold_cache_salt
from lmw_kv.retired import RETIRED_NAME
from lmw_kv.scopes import gpu_cache_salt, isolation_scope, scope_tag
from lmw_kv.store import Store, StoreConfig
from tests.unit.lmw_kv.test_store import (
    GROUPS,
    ROOT,
    Conn,
    _cfg,
    _ns,
    opened,  # noqa: F401  (fixture)
)

NK = bytes(range(32))  # the network key
SECRET = b"\x5a" * 32  # the block-key secret
TOKENS = list(range(1000, 1064))
BLOCK = 16
SCOPE_A = isolation_scope(NK, "key-a", 0)
SCOPE_C = isolation_scope(NK, "key-c", 0)
A_SALT = gpu_cache_salt(NK, SCOPE_A, "")


def _keys(ns_id: bytes, salt: bytes | None) -> list[bytes]:
    assert salt is not None
    return block_keys_for(SECRET, salt, ns_id, TOKENS, BLOCK)


def _save_chain(
    s: Store, ns_id: bytes, keys: list[bytes], scope: bytes, *, tag: bytes = b""
) -> None:
    parent = ROOT
    for key in keys:
        assert s.begin_save(ns_id, key, parent, scope, scope_tag=tag) is not None
        s.commit_save(ns_id, key)
        parent = key


def _store(opened: list[Store | Conn], cfg: StoreConfig) -> tuple[Store, bytes]:  # noqa: F811
    s = Store(cfg)
    opened.append(s)
    return s, s.register_namespace(_ns(), GROUPS, owner="a")


def _check_disjoint(s: Store, ns_id: bytes, a_keys: list[bytes]) -> None:
    others = {
        "scope C": fold_cache_salt(SCOPE_C, gpu_cache_salt(NK, SCOPE_C, "")),
        "sharing, client salt empty": fold_cache_salt(b"", gpu_cache_salt(NK, b"", "")),
        "sharing, A's salt as client salt": fold_cache_salt(b"", gpu_cache_salt(NK, b"", A_SALT)),
        "sharing, no salt": fold_cache_salt(b"", None),
    }
    for name, salt in others.items():
        assert s.lookup(ns_id, _keys(ns_id, salt)) == 0, name
    assert s.lookup(ns_id, a_keys) == len(a_keys)


def _a_keys(ns_id: bytes) -> list[bytes]:
    return _keys(ns_id, fold_cache_salt(SCOPE_A, A_SALT))


def test_isolated_keys_disjoint_in_ram(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id = _store(opened, _cfg(tmp_path, nvme=False))
    a_keys = _a_keys(ns_id)
    assert len(a_keys) == 4
    _save_chain(s, ns_id, a_keys, SCOPE_A, tag=scope_tag(SCOPE_A))
    _check_disjoint(s, ns_id, a_keys)


def test_isolated_keys_disjoint_on_nvme(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id = _store(opened, cfg)
    a_keys = _a_keys(ns_id)
    _save_chain(s, ns_id, a_keys, SCOPE_A, tag=scope_tag(SCOPE_A))
    s.flush()
    s.close()
    opened.clear()
    s2, _ = _store(opened, cfg)  # RAM empty: only NVMe can answer
    _check_disjoint(s2, ns_id, a_keys)
    h = s2.begin_load(ns_id, a_keys)
    assert h.nvme_blocks == len(a_keys)
    s2.end_load(h.id)


def test_retired_scope_never_found(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id = _store(opened, cfg)
    a_keys = _a_keys(ns_id)
    tag = scope_tag(SCOPE_A)
    _save_chain(s, ns_id, a_keys, SCOPE_A, tag=tag)
    s.flush()
    assert s.retire_scopes([tag]) == 1
    assert s.retire_scopes([tag]) == 0
    assert s.lookup(ns_id, a_keys) == 0
    assert s.begin_load(ns_id, a_keys).slots == []
    assert s.begin_save(ns_id, a_keys[0], ROOT, SCOPE_A, scope_tag=tag) is None
    dead = s.stats()["dead_bytes"]
    assert dead["ram"] > 0 and dead["nvme"] > 0
    assert s.usage("ram") == {} and s.usage("nvme") == {}
    s.close()
    opened.clear()

    s2, _ = _store(opened, cfg)  # RAM empty, the retired set comes from disk
    assert s2.lookup(ns_id, a_keys) == 0
    assert s2.begin_load(ns_id, a_keys).slots == []
    assert s2.stats()["dead_bytes"]["nvme"] > 0


def test_a_new_save_under_a_retired_scope_is_refused(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id = _store(opened, _cfg(tmp_path, nvme=False))
    tag = scope_tag(SCOPE_A)
    s.retire_scopes([tag])
    a_keys = _a_keys(ns_id)
    assert s.begin_save(ns_id, a_keys[0], ROOT, SCOPE_A, scope_tag=tag) is None
    sharing = _keys(ns_id, b"")
    assert s.begin_save(ns_id, sharing[0], ROOT, b"") is not None  # sharing unaffected


def test_scope_retired_while_ram_only_is_persisted_later(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, _ = _store(opened, _cfg(tmp_path, nvme=False))
    tag = scope_tag(SCOPE_A)
    assert s.retire_scopes([tag]) == 1
    nvme_dir = tmp_path / "late-nvme"
    assert s.enable_nvme(nvme_dir) is True
    db = sqlite3.connect(nvme_dir / RETIRED_NAME)
    try:
        assert {bytes(r[0]) for r in db.execute("SELECT tag FROM retired")} == {tag}
    finally:
        db.close()


def test_unreadable_retired_store_refuses_isolation_only(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id = _store(opened, cfg)
    a_keys = _a_keys(ns_id)
    tag = scope_tag(SCOPE_A)
    _save_chain(s, ns_id, a_keys, SCOPE_A, tag=tag)
    s.flush()
    s.close()
    opened.clear()
    assert cfg.nvme_dir is not None
    retired = cfg.nvme_dir / RETIRED_NAME
    for suffix in ("-wal", "-shm"):
        Path(f"{retired}{suffix}").unlink(missing_ok=True)
    retired.write_bytes(os.urandom(8192))

    s2, _ = _store(opened, cfg)
    assert s2.stats()["nvme_enabled"] is True  # NVMe itself is fine
    assert s2.lookup(ns_id, a_keys) == 0  # fail closed: the retired set is unknown
    assert s2.begin_load(ns_id, a_keys).slots == []
    other = isolation_scope(NK, "key-d", 0)
    assert s2.begin_save(ns_id, b"\x07" * 32, ROOT, other, scope_tag=scope_tag(other)) is None
    sharing = _keys(ns_id, b"")
    _save_chain(s2, ns_id, sharing, b"")
    assert s2.lookup(ns_id, sharing) == len(sharing)


def test_a_scope_retired_mid_promotion_is_never_served(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    s, ns_id = _store(opened, cfg)
    a_keys = _a_keys(ns_id)
    tag = scope_tag(SCOPE_A)
    _save_chain(s, ns_id, a_keys, SCOPE_A, tag=tag)
    s.flush()
    s.close()
    opened.clear()
    s2, _ = _store(opened, cfg)  # NVMe only: the load promotes
    real = s2._promote

    def promote(*args: Any) -> bool:
        ok = real(*args)
        s2.retire_scopes([tag])  # lands while the bytes are copied, outside the lock
        return ok

    monkeypatch.setattr(s2, "_promote", promote)
    h = s2.begin_load(ns_id, a_keys)
    assert h.slots == [] and h.nvme_blocks == 0
    assert not s2._ram and not s2._loads
    assert s2.lookup(ns_id, a_keys) == 0
