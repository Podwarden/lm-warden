"""Protocol 1, minor 2. Each section covers one field; later tasks add theirs below."""

from __future__ import annotations

from typing import Any

import pytest

from lmw_kv.client import Client
from tests.unit.lmw_kv import test_daemon
from tests.unit.lmw_kv.test_daemon import GROUPS, NS, ROOT, Running, _k
from tests.unit.lmw_kv.test_protocol_minor1 import SLOW, _client, _ns_doc, _Raw

running = test_daemon.running  # the daemon fixture, shared


def _register(c: Client) -> bytes:
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    return ns


def test_hello_reports_minor_2(running: Running) -> None:
    r = _client(running).hello()
    assert r is not None and r["protocol"] == 1 and r["minor"] == 2


# ---- begin_save.depth ---------------------------------------------------------


def _save_req(ns: bytes, key: str, **extra: Any) -> dict[str, Any]:
    return {
        "op": "begin_save",
        "id": 2,
        "ns": ns.hex(),
        "key": _k(key).hex(),
        "parent": ROOT.hex(),
        "scope": "",
        **extra,
    }


@pytest.mark.parametrize("depth", [-1, True, False, "1", 1.5, [1]])
def test_begin_save_depth_must_be_a_non_negative_int(running: Running, depth: Any) -> None:
    ns = _register(_client(running))
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_save_req(ns, "bad", depth=depth))
        assert r["ok"] is False and r["error"].startswith("bad_request")
    finally:
        raw.close()


@pytest.mark.parametrize("depth", [0, 3])
def test_begin_save_accepts_a_good_depth(running: Running, depth: int) -> None:
    ns = _register(_client(running))
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_save_req(ns, "good", depth=depth))
        assert r["ok"] is True
        # depth 3 has no stored parent: answered null, not an error
        assert (r["offsets"] is None) == (depth > 0)
    finally:
        raw.close()


def test_a_minor1_begin_save_still_works(running: Running) -> None:
    c = Client(running.sock, timeout_s=SLOW, slow_timeout_s=SLOW)
    ns = _register(c)
    assert c.begin_save(ns, _k("a"), _k("unknown-parent")) is not None  # no depth: no check


def test_client_depth_drives_the_parent_check_and_the_stats(running: Running) -> None:
    c = Client(running.sock, timeout_s=SLOW, slow_timeout_s=SLOW)
    ns = _register(c)
    assert c.begin_save(ns, _k("a"), ROOT, depth=0) is not None
    assert c.begin_save(ns, _k("b"), _k("a"), depth=1) is not None
    assert c.begin_save(ns, _k("d"), _k("missing"), depth=1) is None
    st = c.stats()
    assert st is not None and st["parent_refused"] == 1


# ---- register_ns.staging --------------------------------------------------------


def _reg_req(**extra: Any) -> dict[str, Any]:
    return {"op": "register_ns", "id": 3, "namespace": _ns_doc(), "groups": GROUPS, **extra}


@pytest.mark.parametrize("staging", [1, 0, "true", None, [True]])
def test_register_ns_staging_must_be_a_bool(running: Running, staging: Any) -> None:
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_reg_req(staging=staging))
        assert r["ok"] is False and r["error"].startswith("bad_request")
    finally:
        raw.close()


def test_register_ns_staging_shows_in_stats(running: Running) -> None:
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_reg_req(staging=True))
        assert r["ok"] is True
        ns = r["ns"]
    finally:
        raw.close()
    st = _client(running).stats()
    assert st is not None
    assert st["by_namespace"][ns]["staging"] is True
    assert st["staging"] == {
        "namespaces": 1, "in_flight": 0, "save_refused": 0, "save_waited": 0,
        "write_dropped": 0, "released": 0,
    }  # fmt: skip


def test_a_minor1_register_ns_is_not_staging(running: Running) -> None:
    ns = _register(_client(running))
    st = _client(running).stats()
    assert st is not None and st["by_namespace"][ns.hex()]["staging"] is False
    assert st["staging"]["namespaces"] == 0


def test_client_sends_staging_only_when_true(
    running: Running, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _client(running)
    sent: list[dict[str, Any]] = []
    real = c._call_once

    def spy(req: dict[str, Any], timeout_s: float | None = None) -> Any:
        sent.append(req)
        return real(req, timeout_s)

    monkeypatch.setattr(c, "_call_once", spy)
    assert c.register_ns(NS, GROUPS, owner="plain") is not None
    ns = c.register_ns(NS, GROUPS, owner="disk", staging=True)
    assert ns is not None
    assert "staging" not in sent[0] and sent[1]["staging"] is True
    st = c.stats()
    assert st is not None
    # Owner "plain" did not ask for staging, so the namespace is not staging (S1).
    assert st["by_namespace"][ns.hex()]["staging"] is False
    assert c.release_owner("plain") == 0
    st = c.stats()
    assert st is not None and st["by_namespace"][ns.hex()]["staging"] is True


# ---- begin_load.tag, end_load.used, top_reused ----------------------------------

A = b"\x0a" * 16


def _saved(running: Running, c: Client, name: str = "p0") -> tuple[bytes, bytes]:
    ns = _register(c)
    key = _k(name)
    assert c.begin_save(ns, key, ROOT, tenant=A, pos=_k(f"pos-{name}")) is not None
    assert c.commit_save(ns, key)
    return ns, key


def _load_req(ns: bytes, key: bytes, **extra: Any) -> dict[str, Any]:
    return {"op": "begin_load", "id": 4, "ns": ns.hex(), "keys": [key.hex()], **extra}


@pytest.mark.parametrize("tag", [1, None, True, ["t"], "x" * 256, "é" * 128])
def test_begin_load_tag_must_be_a_short_string(running: Running, tag: Any) -> None:
    ns, key = _saved(running, _client(running))
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_load_req(ns, key, tag=tag))
        assert r["ok"] is False and r["error"].startswith("bad_request")
    finally:
        raw.close()


def test_begin_load_tag_and_end_load_used_types(running: Running) -> None:
    c = _client(running)
    ns, key = _saved(running, c)
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_load_req(ns, key, tag="x" * 255, tenant=A.hex()))
        assert r["ok"] is True and len(r["slots"]) == 1
        for used in (1, "true", 0, [True]):
            bad = raw.ask({"op": "end_load", "id": 5, "handle": r["handle"], "used": used})
            assert bad["ok"] is False and bad["error"].startswith("bad_request")
        assert raw.ask({"op": "end_load", "id": 6, "handle": r["handle"], "used": True})["ok"]
        r2 = raw.ask(_load_req(ns, key))  # minor 1: no tag, no used
        assert raw.ask({"op": "end_load", "id": 7, "handle": r2["handle"]})["ok"] is True
    finally:
        raw.close()
    st = c.stats()
    assert st is not None
    reuse = st["by_tenant"][A.hex()]["reuse"]
    assert reuse["charged"]["reloads"] == 1 and reuse["saver"]["saver_reloads"] == 1
    assert st["reuse"]["order"] == "lru"


def test_client_sends_tag_and_used_only_when_set(
    running: Running, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _client(running)
    ns, key = _saved(running, c)
    sent: list[dict[str, Any]] = []
    real = c._call_once

    def spy(req: dict[str, Any], timeout_s: float | None = None) -> Any:
        sent.append(req)
        return real(req, timeout_s)

    monkeypatch.setattr(c, "_call_once", spy)
    got = c.begin_load(ns, [key])
    assert got is not None and c.end_load(got[0])
    got = c.begin_load(ns, [key], tenant=A, tag="req-1")
    assert got is not None and c.end_load(got[0], used=True)
    assert "tag" not in sent[0] and "used" not in sent[1]
    assert sent[2]["tag"] == "req-1" and sent[3]["used"] is True
    rows = c.top_reused(view="holder", tenant=A)
    assert rows is not None and [r["reloads"] for r in rows] == [1]


def _top_req(**extra: Any) -> dict[str, Any]:
    return {"op": "top_reused", "id": 8, **extra}


@pytest.mark.parametrize(
    "req",
    [
        {"view": "x", "n": 5},
        {"n": 5},
        {"view": "holder", "n": 5},
        {"view": "admin", "n": 0},
        {"view": "admin", "n": 51},
        {"view": "admin", "n": True},
        {"view": "admin", "n": 2.0},
        {"view": "admin"},
        {"view": "admin", "n": 5, "model": "00"},
        {"view": "admin", "n": 5, "tenant": "zz"},
    ],
)
def test_top_reused_validation(running: Running, req: dict[str, Any]) -> None:
    raw = _Raw(running.sock)
    try:
        r = raw.ask(_top_req(**req))
        assert r["ok"] is False and r["error"].startswith("bad_request")
        assert raw.ask(_top_req(view="admin", n=50)) == {"id": 8, "ok": True, "rows": []}
    finally:
        raw.close()


def test_top_reused_rows_carry_no_position_key(running: Running) -> None:
    c = _client(running)
    ns, key = _saved(running, c)
    got = c.begin_load(ns, [key], tenant=A, tag="req-1")
    assert got is not None and c.end_load(got[0], used=True)
    admin = c.top_reused(view="admin", n=5)
    holder = c.top_reused(view="holder", tenant=A, model=_ns_model(running, ns))
    assert admin is not None and holder is not None and len(admin) == len(holder) == 1
    assert admin[0]["model"] == holder[0]["model"] == _ns_model(running, ns).hex()
    assert admin[0]["label"] != holder[0]["label"]
    text = repr(admin) + repr(holder)
    for secret in (key.hex(), _k("pos-p0").hex()):
        assert secret not in text and secret[:12] not in text


def _ns_model(running: Running, ns: bytes) -> bytes:
    st = running.store.stats()
    return bytes.fromhex(st["by_namespace"][ns.hex()]["model"])


def test_client_top_reused_fails_open(tmp_path: Any) -> None:
    c = Client(str(tmp_path / "nothing.sock"), timeout_s=0.05, slow_timeout_s=0.05)
    assert c.top_reused() is None
