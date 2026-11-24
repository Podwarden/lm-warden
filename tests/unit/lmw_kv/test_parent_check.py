"""The ``begin_save`` parent check (protocol minor 2): a block is refused when its
parent is not stored. In-memory indexes only; ``depth`` 0 and no ``depth`` skip it."""

from __future__ import annotations

from pathlib import Path

from lmw_kv.scopes import isolation_scope, scope_tag
from lmw_kv.store import Store
from tests.unit.lmw_kv.test_store import (
    GROUPS,
    ROOT,
    SCOPE,
    Conn,
    _cfg,
    _k,
    _ns,
    _open,
    _save,
    opened,  # noqa: F401  (fixture)
)

TAG = scope_tag(isolation_scope(bytes(range(32)), "key-a", 0))


def test_child_of_a_refused_parent_is_refused(tmp_path: Path, opened: list[Store | Conn]) -> None:  # noqa: F811
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1, nvme=False))
    k0, k1, other = _k("k0"), _k("k1"), _k("other")
    assert _save(s, c, ns_id, other, ROOT)
    h = s.begin_load(ns_id, [other])  # pins the only slot
    assert s.begin_save(ns_id, k0, ROOT, SCOPE, depth=0) is None  # no slot
    s.end_load(h.id)
    assert s.stats()["parent_refused"] == 0
    assert s.begin_save(ns_id, k1, k0, SCOPE, depth=1) is None
    assert s.stats()["parent_refused"] == 1
    assert s.begin_save(ns_id, k1, ROOT, SCOPE, depth=0) is not None  # the slot is free


def test_child_while_its_parent_is_being_saved_is_admitted(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, _ = _open(opened, _cfg(tmp_path, nvme=False))
    k0, k1 = _k("k0"), _k("k1")
    assert s.begin_save(ns_id, k0, ROOT, SCOPE, depth=0) is not None
    assert s.begin_save(ns_id, k1, k0, SCOPE, depth=1) is not None  # k0 not committed yet
    assert s.stats()["parent_refused"] == 0


def test_parent_only_on_nvme_admits(tmp_path: Path, opened: list[Store | Conn]) -> None:  # noqa: F811
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1))
    k0, k1, other = _k("k0"), _k("k1"), _k("other")
    assert _save(s, c, ns_id, k0, ROOT)
    s.flush()
    assert _save(s, c, ns_id, other, ROOT)  # evicts k0's RAM copy
    s.flush()
    assert (ns_id, k0) not in s._ram
    assert s.lookup(ns_id, [k0]) == 1  # still on NVMe
    assert s.begin_save(ns_id, k1, k0, SCOPE, depth=1) is not None


def test_parent_on_nvme_is_not_seen_when_nvme_is_off_for_the_namespace(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, _ = _open(opened, _cfg(tmp_path, nvme=False))
    assert s.begin_save(ns_id, _k("k1"), _k("k0"), SCOPE, depth=1) is None
    assert s.stats()["parent_refused"] == 1


def test_parent_under_a_retired_scope_refuses(tmp_path: Path, opened: list[Store | Conn]) -> None:  # noqa: F811
    s, ns_id, c = _open(opened, _cfg(tmp_path, slots=1))
    k0, k1, other = _k("k0"), _k("k1"), _k("other")
    assert s.begin_save(ns_id, k0, ROOT, SCOPE, scope_tag=TAG) is not None
    s.commit_save(ns_id, k0)
    s.flush()
    assert s.retire_scopes([TAG]) == 1
    assert s.begin_save(ns_id, k1, k0, SCOPE, depth=1) is None  # k0 in RAM, retired
    assert _save(s, c, ns_id, other, ROOT)  # evicts k0's RAM copy
    s.flush()
    assert (ns_id, k0) not in s._ram
    assert s.begin_save(ns_id, k1, k0, SCOPE, depth=1) is None  # k0 on NVMe, retired
    assert s.stats()["parent_refused"] == 2


def test_depth_zero_and_no_depth_skip_the_check(tmp_path: Path, opened: list[Store | Conn]) -> None:  # noqa: F811
    s, ns_id, _ = _open(opened, _cfg(tmp_path, nvme=False))
    unknown = _k("unknown")
    assert s.begin_save(ns_id, _k("a"), unknown, SCOPE, depth=0) is not None
    assert s.begin_save(ns_id, _k("b"), unknown, SCOPE) is not None
    assert s.stats()["parent_refused"] == 0


def test_a_parent_of_another_namespace_does_not_count(
    tmp_path: Path,
    opened: list[Store | Conn],  # noqa: F811
) -> None:
    s, ns_id, c = _open(opened, _cfg(tmp_path, nvme=False))
    other_ns = s.register_namespace(_ns("r2"), GROUPS)
    assert _save(s, c, other_ns, _k("k0"), ROOT)
    assert s.begin_save(ns_id, _k("k1"), _k("k0"), SCOPE, depth=1) is None
