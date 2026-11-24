"""Protocol 1, minor 1: tenants, owners, positions, retired scopes and the policy op.

Deadlines here are generous on purpose (the build host is loaded); nothing asserts on
wall-clock time.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from lmw_kv.client import Client
from lmw_kv.fair import Policy, TenantPolicy
from tests.unit.lmw_kv import test_daemon
from tests.unit.lmw_kv.test_daemon import GROUPS, NS, ROOT, Running, _k

running = test_daemon.running  # the daemon fixture, shared

T1 = bytes([1]) * 16
T2 = bytes([2]) * 16
TAG = bytes([7]) * 16
MODEL = bytes([9]) * 16
SLOW = 30.0


def _client(r: Running) -> Client:
    return Client(r.sock, timeout_s=SLOW, slow_timeout_s=SLOW)


class _Raw:
    """One raw connection: send a request line, read its reply line."""

    def __init__(self, path: str) -> None:
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(SLOW)
        self.s.connect(path)
        self.f = self.s.makefile("rwb")

    def ask(self, req: dict[str, Any] | bytes) -> dict[str, Any]:
        line = req if isinstance(req, bytes) else json.dumps(req).encode()
        self.f.write(line + b"\n")
        self.f.flush()
        reply: dict[str, Any] = json.loads(self.f.readline())
        return reply

    def close(self) -> None:
        self.f.close()
        self.s.close()


def _ns_doc() -> dict[str, Any]:
    return {**NS.__dict__, "kv_groups": list(NS.kv_groups)}


def test_hello_reports_minor_1(running: Running) -> None:
    r = _client(running).hello()
    assert r is not None and r["protocol"] == 1 and r["minor"] >= 1


def test_minor_1_fields_round_trip(running: Running) -> None:
    c = _client(running)
    ns = c.register_ns(NS, GROUPS, owner="row-1", model=MODEL, role="gdn_state", rank=3)
    assert ns is not None
    off = c.begin_save(ns, _k("a"), ROOT, b"\x05" * 32, tenant=T1, scope_tag=TAG, pos=_k("p"))
    assert off is not None
    assert c.commit_save(ns, _k("a")) is True
    assert c.lookup(ns, [_k("a")]) == 1
    loaded = c.begin_load(ns, [_k("a")], tenant=T2)  # last user pays
    assert loaded is not None and len(loaded[1]) == 1
    assert c.end_load(loaded[0]) is True
    assert c.record_hits("tag-1", 16, 0, 0.1, tenant=T2) is True
    st = c.stats()
    assert st is not None
    row = st["by_namespace"][ns.hex()]
    assert row["model"] == MODEL.hex() and row["role"] == "gdn_state" and row["rank"] == 3
    assert row["owners"] == {"row-1": True} and row["live"] is True
    assert row["ram_bytes"] > 0
    t2 = st["by_tenant"][T2.hex()]
    assert t2["ram_bytes"] > 0 and t2["ram_share"] > 0 and t2["ram_tokens"] == 16
    m = st["by_model"][MODEL.hex()]
    assert m["owners"] == ["row-1"] and m["weight"] >= 1 and m["ram_part"] > 0
    assert T2.hex() in m["tenants"]
    assert running.store._ram[(ns, _k("a"))].pos == _k("p")
    assert running.store._ram[(ns, _k("a"))].scope_tag == TAG
    assert st["isolation_refused"] is False and st["retired_scopes"] == 0


def test_v1_0_requests_still_work(running: Running) -> None:
    """The step 1 request lines, unchanged, give the step 1 replies."""
    raw = _Raw(running.sock)
    try:
        hello = raw.ask(b'{"op":"hello","id":1}')
        assert hello == {
            "id": 1,
            "ok": True,
            "protocol": 1,
            "minor": 2,  # the daemon is minor 2 now; the request line is step 1's
            "ram_path": str(running.ram_path),
            "ram_bytes": hello["ram_bytes"],
        }
        reg = raw.ask({"op": "register_ns", "id": 2, "namespace": _ns_doc(), "groups": GROUPS})
        assert set(reg) == {"id", "ok", "ns"} and reg["ok"] is True
        ns = reg["ns"]
        k, root = _k("a").hex(), ROOT.hex()
        save = {"op": "begin_save", "id": 3, "ns": ns, "key": k, "parent": root, "scope": ""}
        got = raw.ask(save)
        assert set(got) == {"id", "ok", "offsets"} and set(got["offsets"]) == set(GROUPS)
        assert raw.ask({"op": "commit_save", "id": 4, "ns": ns, "key": k}) == {"id": 4, "ok": True}
        assert raw.ask({"op": "lookup", "id": 5, "ns": ns, "keys": [k]}) == {
            "id": 5,
            "ok": True,
            "n": 1,
        }
        load = raw.ask({"op": "begin_load", "id": 6, "ns": ns, "keys": [k]})
        assert set(load) == {"id", "ok", "handle", "slots", "nvme"}
        assert load["slots"] == [got["offsets"]] and load["nvme"] == 0
        end = {"op": "end_load", "id": 7, "handle": load["handle"]}
        assert raw.ask(end) == {"id": 7, "ok": True}
        hits = {"op": "record_hits", "id": 8, "tag": "t", "ram_tokens": 16}
        assert raw.ask({**hits, "nvme_tokens": 0, "load_s": 0.5}) == {"id": 8, "ok": True}
        assert raw.ask({"op": "pop_hits", "id": 9, "tag": "t"}) == {
            "id": 9,
            "ok": True,
            "hits": {"ram_tokens": 16, "nvme_tokens": 0, "load_s": 0.5},
        }
        again = raw.ask(save | {"id": 10})
        assert again == {"id": 10, "ok": True, "offsets": None}  # already stored
        st = raw.ask({"op": "stats", "id": 11})["stats"]
        step1 = ("ram_bytes", "ram_used", "nvme_used", "nvme_enabled", "blocks", "evictions")
        assert all(f in st for f in (*step1, "nvme_evictions")) and st["blocks"] == 1
        # Unknown fields are ignored.
        r = raw.ask({"op": "lookup", "id": 12, "ns": ns, "keys": [k], "later": {"x": 1}})
        assert r == {"id": 12, "ok": True, "n": 1}
        # The legacy owner "" keeps last-registration-wins for nvme.
        reg_off = {"op": "register_ns", "id": 13, "namespace": _ns_doc(), "groups": GROUPS}
        assert raw.ask(reg_off | {"nvme": False})["ok"] is True
        assert running.store._ns[bytes.fromhex(ns)].nvme is False
        assert raw.ask(reg_off | {"id": 14})["ok"] is True
        assert running.store._ns[bytes.fromhex(ns)].nvme is True
    finally:
        raw.close()


@pytest.mark.parametrize(
    "op,extra",
    [
        ("begin_save", {"tenant": "01" * 15}),
        ("begin_save", {"scope_tag": "07" * 17}),
        ("begin_save", {"pos": "ab" * 31}),
        ("begin_save", {"tenant": 5}),
        ("register_ns", {"role": "r" * 40}),
        ("register_ns", {"owner": "o" * 256}),
        ("register_ns", {"model": "09" * 15}),
        ("register_ns", {"rank": -1}),
        ("register_ns", {"rank": True}),
        ("register_ns", {"owner": 3}),
        ("begin_load", {"tenant": "01" * 17}),
        ("record_hits", {"tenant": "zz"}),
    ],
)
def test_bad_field_lengths_are_bad_request(
    running: Running, op: str, extra: dict[str, Any]
) -> None:
    raw = _Raw(running.sock)
    try:
        reg = raw.ask({"op": "register_ns", "id": 1, "namespace": _ns_doc(), "groups": GROUPS})
        ns, k = reg["ns"], _k("a").hex()
        base: dict[str, dict[str, Any]] = {
            "register_ns": {"namespace": _ns_doc(), "groups": GROUPS},
            "begin_save": {"ns": ns, "key": k, "parent": ROOT.hex(), "scope": ""},
            "begin_load": {"ns": ns, "keys": [k]},
            "record_hits": {"tag": "t", "ram_tokens": 1, "nvme_tokens": 0, "load_s": 0.0},
        }
        r = raw.ask({"op": op, "id": 2, **base[op], **extra})
        assert r["id"] == 2 and r["ok"] is False and r["error"].startswith("bad_request")
        assert not running.store._saving  # nothing half-begun
        assert raw.ask({"op": "hello", "id": 3})["ok"] is True  # still open
    finally:
        raw.close()


def _wire(**over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "mode": "fair",
        "idle_ram_s": 60,
        "idle_nvme_s": 3600.5,
        "owners": {"row-1": 3, "row-2": 1},
        "tenants": {
            T1.hex(): {"weight": 10, "max_ram_pct": 50, "max_nvme_pct": None},
            T2.hex(): {"weight": 1},
        },
    }
    doc.update(over)
    return doc


def test_policy_from_wire() -> None:
    p = Policy.from_wire(_wire())
    assert p.mode == "fair" and p.idle_ram_s == 60.0 and p.idle_nvme_s == 3600.5
    assert dict(p.owner_weights) == {"row-1": 3, "row-2": 1}
    assert dict(p.tenants) == {
        T1: TenantPolicy(weight=10, max_ram_pct=50, max_nvme_pct=None),
        T2: TenantPolicy(weight=1),
    }
    assert Policy.from_wire({}) == Policy()  # every key is optional
    assert Policy.from_wire({"mode": "lru", "later": 1}).mode == "lru"  # unknown ignored
    with pytest.raises(TypeError):
        p.owner_weights["x"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "doc",
    [
        _wire(tenants={T1.hex(): {"weight": 0}}),
        _wire(tenants={T1.hex(): {"weight": 101}}),
        _wire(tenants={T1.hex(): {"max_ram_pct": 101}}),
        _wire(tenants={T1.hex(): {"max_nvme_pct": 0}}),
        _wire(tenants={T1.hex(): {"weight": True}}),
        _wire(tenants={T1.hex(): {"weight": 2.0}}),
        _wire(tenants={"01" * 15: {"weight": 1}}),
        _wire(tenants={"not hex": {"weight": 1}}),
        _wire(tenants={T1.hex(): 5}),
        _wire(tenants=[]),
        _wire(mode="fifo"),
        _wire(mode=None),
        _wire(idle_ram_s=0),
        _wire(idle_nvme_s=-1),
        _wire(idle_ram_s=float("nan")),
        _wire(idle_ram_s=float("inf")),
        _wire(idle_ram_s="60"),
        _wire(owners={"row-1": 0}),
        _wire(owners={"row-1": 101}),
        _wire(owners={"o" * 256: 1}),
        _wire(owners=[]),
    ],
)
def test_policy_from_wire_rejects(doc: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Policy.from_wire(doc)


def test_policy_wire_keys_are_the_ones_read() -> None:
    from lmw_kv.fair import WIRE_KEYS, WIRE_TENANT_KEYS

    doc = _wire()
    assert set(doc) == WIRE_KEYS
    assert WIRE_TENANT_KEYS == set(doc["tenants"][T1.hex()])
    # A key outside them changes nothing.
    assert Policy.from_wire({**doc, "extra": 1}) == Policy.from_wire(doc)


def test_policy_from_wire_errors_never_show_a_tenant_tag() -> None:
    with pytest.raises(ValueError) as e:
        Policy.from_wire(_wire(tenants={T1.hex(): {"weight": 0}}))
    assert T1.hex() not in str(e.value)


def test_set_policy_bad_keeps_previous(running: Running) -> None:
    c = _client(running)
    assert c.set_policy(_wire()) is True
    good = running.store._policy
    assert good.mode == "fair" and good.tenant(T1).weight == 10
    # One bad tenant among good fields: nothing of it is applied.
    bad = _wire(mode="lru", idle_ram_s=5, tenants={T2.hex(): {"weight": 7}, T1.hex(): {}})
    bad["tenants"][T1.hex()] = {"weight": 0}
    raw = _Raw(running.sock)
    try:
        r = raw.ask({"op": "set_policy", "id": 1, "policy": bad})
        assert r["ok"] is False and r["error"].startswith("bad_request")
        assert T1.hex() not in r["error"]
        r = raw.ask({"op": "set_policy", "id": 2, "policy": "fair"})
        assert r["ok"] is False and r["error"].startswith("bad_request")
        r = raw.ask({"op": "set_policy", "id": 3})
        assert r["ok"] is False and r["error"].startswith("bad_request")
    finally:
        raw.close()
    assert running.store._policy is good
    assert c.stats()["policy"] == "fair"  # type: ignore[index]
    assert c.set_policy({"mode": "lru"}) is True
    assert running.store._policy == Policy()


def test_retire_scopes_idempotent(running: Running) -> None:
    c = _client(running)
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("a"), ROOT, b"\x05" * 32, tenant=T1, scope_tag=TAG) is not None
    assert c.commit_save(ns, _k("a")) is True
    assert c.lookup(ns, [_k("a")]) == 1
    other = bytes([8]) * 16
    assert c.retire_scopes([TAG, other, TAG]) == 2
    assert c.retire_scopes([TAG, other]) == 0  # idempotent
    assert c.retire_scopes([]) == 0
    assert c.lookup(ns, [_k("a")]) == 0  # never found again
    assert c.begin_save(ns, _k("b"), ROOT, b"\x05" * 32, scope_tag=TAG) is None
    assert c.stats()["retired_scopes"] == 2  # type: ignore[index]
    running.store.flush()
    # Durable: the file next to the manifests holds both tags.
    import sqlite3

    path = running.store.cfg.nvme_dir / "retired.db"  # type: ignore[operator]
    with sqlite3.connect(path) as db:
        on_disk = {bytes(r[0]) for r in db.execute("SELECT tag FROM retired")}
    assert on_disk == {TAG, other}
    # A bad tag anywhere in the list retires none of them.
    raw = _Raw(running.sock)
    try:
        new = bytes([3]) * 16
        r = raw.ask({"op": "retire_scopes", "id": 1, "scope_tags": [new.hex(), "07" * 15]})
        assert r["ok"] is False and r["error"].startswith("bad_request")
        assert new not in running.store._retired
        r = raw.ask({"op": "retire_scopes", "id": 2, "scope_tags": new.hex()})
        assert r["ok"] is False and r["error"].startswith("bad_request")
        assert raw.ask({"op": "hello", "id": 3})["ok"] is True
    finally:
        raw.close()


def test_release_owner_counts(running: Running) -> None:
    c = _client(running)
    ns = c.register_ns(NS, GROUPS, owner="row-1")
    assert ns is not None
    assert c.register_ns(NS, GROUPS, owner="row-2") == ns
    assert c.begin_save(ns, _k("a"), ROOT) is not None
    assert c.commit_save(ns, _k("a")) is True
    assert c.release_owner("row-1") == 0  # row-2 still holds it
    assert c.lookup(ns, [_k("a")]) == 1
    assert c.release_owner("row-2") == 1
    assert c.release_owner("row-2") == 0
    assert c.release_owner("nobody") == 0
    assert c.lookup(ns, [_k("a")]) == 0  # unregistered until it registers again
    assert c.stats()["by_namespace"][ns.hex()]["live"] is False  # type: ignore[index]
    assert c.register_ns(NS, GROUPS, owner="row-1") == ns
    assert c.lookup(ns, [_k("a")]) == 1
    raw = _Raw(running.sock)
    try:
        # An empty owner would drop the legacy owner "" from every namespace.
        for bad in ({}, {"owner": ""}, {"owner": 1}, {"owner": "o" * 256}):
            r = raw.ask({"op": "release_owner", "id": 1, **bad})
            assert r["ok"] is False and r["error"].startswith("bad_request")
    finally:
        raw.close()


def test_release_owner_refuses_the_empty_owner(running: Running) -> None:
    """Releasing "" would unregister every namespace a minor 0 adapter registered."""
    c = _client(running)
    ns = c.register_ns(NS, GROUPS)  # the legacy owner ""
    assert ns is not None
    assert c.release_owner("") is None  # bad_request
    assert running.store._ns[ns].owners == {"": True} and running.store._ns[ns].live


def test_isolation_refused_is_reported_and_fails_closed(running: Running) -> None:
    """An unreadable retired store: stats say so, isolated work is refused, no crash."""
    c = _client(running)
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("iso"), ROOT, b"\x05" * 32, scope_tag=TAG) is not None
    assert c.commit_save(ns, _k("iso")) is True
    running.store._retired._refused = True  # as after a failed attach
    st = c.stats()
    assert st is not None and st["isolation_refused"] is True
    assert st["retired_persist_degraded"] is True
    assert c.lookup(ns, [_k("iso")]) == 0
    assert c.begin_save(ns, _k("iso2"), ROOT, b"\x05" * 32, scope_tag=TAG) is None
    loaded = c.begin_load(ns, [_k("iso")])
    assert loaded is not None and loaded[1] == []
    assert c.retire_scopes([bytes([4]) * 16]) == 1  # memory first, still answered
    # Sharing blocks are unaffected.
    assert c.begin_save(ns, _k("share"), ROOT) is not None
    assert c.commit_save(ns, _k("share")) is True
    assert c.lookup(ns, [_k("share")]) == 1
    assert c.hello() is not None


def test_client_new_ops_fail_open_when_daemon_down(tmp_path: Path) -> None:
    c = Client(str(tmp_path / "nothing.sock"), timeout_s=0.05, slow_timeout_s=0.05)
    ns = b"\x01" * 32
    assert c.register_ns(NS, GROUPS, owner="o", model=MODEL, role="mtp", rank=1) is None
    c._conn().retry_at = 0.0
    assert c.begin_save(ns, _k("a"), ROOT, tenant=T1, scope_tag=TAG, pos=_k("p")) is None
    c._conn().retry_at = 0.0
    assert c.begin_load(ns, [_k("a")], tenant=T1) is None
    c._conn().retry_at = 0.0
    assert c.record_hits("t", 1, 0, 0.0, tenant=T1) is None
    c._conn().retry_at = 0.0
    assert c.release_owner("o") is None
    c._conn().retry_at = 0.0
    assert c.retire_scopes([TAG]) is None
    c._conn().retry_at = 0.0
    assert c.set_policy({"mode": "fair"}) is None


def test_client_sends_pos_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def call(self: Client, req: dict[str, Any], timeout_s: float | None = None) -> None:
        sent.append(req)

    monkeypatch.setattr(Client, "_call", call)
    c = Client("/nonexistent.sock")
    ns = b"\x01" * 32
    c.begin_save(ns, _k("a"), ROOT)
    c.begin_save(ns, _k("a"), ROOT, tenant=T1, scope_tag=TAG, pos=_k("p"))
    assert "pos" not in sent[0] and sent[0]["tenant"] == "" and sent[0]["scope_tag"] == ""
    assert sent[1]["pos"] == _k("p").hex() and sent[1]["tenant"] == T1.hex()
    assert sent[1]["scope_tag"] == TAG.hex()


def test_client_new_ops_return_none_on_a_malformed_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def call(self: Client, req: dict[str, Any], timeout_s: float | None = None) -> Any:
        return {"ok": True, "released": "x", "added": True}

    monkeypatch.setattr(Client, "_call", call)
    c = Client("/nonexistent.sock")
    assert c.release_owner("o") is None
    assert c.retire_scopes([TAG]) is None
