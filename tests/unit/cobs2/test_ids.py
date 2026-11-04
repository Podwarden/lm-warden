"""Network key and content ids (spec section 6.3)."""

import errno
import hashlib
import os
import stat
from pathlib import Path

import pytest

from app.cobs2.canonical import canonicalize
from app.cobs2.ids import Ids, ids_for, ids_for_epochs, scope_salt
from app.cobs2.keys import NetworkKey, load_or_create_network_key

KEY = NetworkKey(epoch=1, key=b"k" * 32)


def _body(*texts: str, system: str = "be brief") -> dict:
    msgs = [{"role": "system", "content": system}]
    for i, t in enumerate(texts):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return {"messages": msgs}


def _canon(*texts: str, system: str = "be brief"):
    c = canonicalize(_body(*texts, system=system))
    assert c is not None
    return c


def test_key_file_created_0600_and_reused(tmp_path: Path) -> None:
    db = tmp_path / "vllm-warden.db"
    k1 = load_or_create_network_key(db)
    f = tmp_path / "cobs2_network_key"
    assert f.read_text().splitlines()[0] == "v1:" + k1.key.hex()
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert k1.epoch == 1 and len(k1.key) == 32
    assert load_or_create_network_key(db) == k1


def test_key_file_corrupt_or_empty_raises(tmp_path: Path) -> None:
    db = tmp_path / "vllm-warden.db"
    f = tmp_path / "cobs2_network_key"
    f.write_text("")
    with pytest.raises(RuntimeError, match="cobs2_network_key"):
        load_or_create_network_key(db)
    f.write_text("v1:nothex\n")
    with pytest.raises(RuntimeError, match="cobs2_network_key"):
        load_or_create_network_key(db)


def test_ids_stable_and_hex_34() -> None:
    a = ids_for(_canon("hi", "yo"), KEY, b"")
    b = ids_for(_canon("hi", "yo"), KEY, b"")
    assert a == b
    allids = [a.head, a.loose_head, *a.msgs, *a.nodes, *a.loose_msgs, *a.loose_nodes]
    assert all(len(i) == 34 and bytes.fromhex(i)[0] == 1 for i in allids)
    assert len(a.msgs) == len(a.nodes) == 2


def test_different_key_changes_every_id() -> None:
    a = ids_for(_canon("hi", "yo"), KEY, b"")
    b = ids_for(_canon("hi", "yo"), NetworkKey(1, b"z" * 32), b"")
    assert not set(a.msgs + a.nodes + (a.head,)) & set(b.msgs + b.nodes + (b.head,))


def test_salt_isolates_scopes() -> None:
    c = _canon("hi", "yo")
    a, b = ids_for(c, KEY, b""), ids_for(c, KEY, scope_salt({"cache_salt": "tenant"}))
    assert a.head != b.head and not set(a.msgs) & set(b.msgs) and not set(a.nodes) & set(b.nodes)
    assert scope_salt({}) == b"" and scope_salt({"cache_salt": ""}) == b""
    assert scope_salt({"cache_salt": 5}) == b""
    assert scope_salt({"cache_salt": "x"}) == hashlib.sha256(b"x").digest()


def test_node_chain_extends_by_one_per_message() -> None:
    short = ids_for(_canon("a", "b"), KEY, b"")
    long = ids_for(_canon("a", "b", "c"), KEY, b"")
    assert long.nodes[:2] == short.nodes
    assert long.loose_nodes[:2] == short.loose_nodes
    assert len(long.nodes) == 3 and long.msgs[:2] == short.msgs
    assert len(set(long.nodes)) == 3


def test_msg_id_is_position_and_head_independent() -> None:
    full = ids_for(_canon("hello", "reply", system="sys A"), KEY, b"")
    alone = canonicalize({"messages": [{"role": "assistant", "content": "reply"}]})
    assert alone is not None
    other = ids_for(alone, KEY, b"")
    assert other.msgs[0] == full.msgs[1]
    diff_head = ids_for(_canon("hello", "reply", system="sys B"), KEY, b"")
    assert diff_head.msgs == full.msgs and diff_head.head != full.head
    assert diff_head.nodes != full.nodes


def test_rotation_window_accepts_both_epochs() -> None:
    c = _canon("hi", "yo")
    old, new = NetworkKey(1, b"a" * 32), NetworkKey(2, b"b" * 32)
    r = ids_for_epochs(c, [new, old], b"")
    assert r == [ids_for(c, new, b""), ids_for(c, old, b"")]
    assert bytes.fromhex(r[0].head)[0] == 2 and bytes.fromhex(r[1].head)[0] == 1
    assert isinstance(r[0], Ids)


def test_key_wrong_length_binary_and_missing_dir_errors(tmp_path: Path) -> None:
    db = tmp_path / "vllm-warden.db"
    f = tmp_path / "cobs2_network_key"
    f.write_text("v1:" + "ab" * 16 + "\n")
    with pytest.raises(RuntimeError, match="cobs2_network_key"):
        load_or_create_network_key(db)
    f.write_bytes(b"\xff\xfe\x00\x80binary")
    with pytest.raises(RuntimeError, match="cobs2_network_key"):
        load_or_create_network_key(db)
    with pytest.raises(RuntimeError, match="data dir"):
        load_or_create_network_key(tmp_path / "nope" / "x.db")


def test_concurrent_first_start_agrees_and_leaves_no_tmp(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    db = tmp_path / "vllm-warden.db"
    with ThreadPoolExecutor(16) as ex:
        keys = list(ex.map(lambda _: load_or_create_network_key(db), range(32)))
    assert len({k.key for k in keys}) == 1
    assert load_or_create_network_key(db) == keys[0]
    assert [p.name for p in tmp_path.iterdir()] == ["cobs2_network_key"]


@pytest.mark.parametrize("exc", [PermissionError(1, "no"), OSError(errno.ENOTSUP, "no")])
def test_key_created_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: OSError
) -> None:
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(os, "link", boom)
    k = load_or_create_network_key(tmp_path / "vllm-warden.db")
    f = tmp_path / "cobs2_network_key"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["cobs2_network_key"]
    assert load_or_create_network_key(tmp_path / "vllm-warden.db") == k


def test_concurrent_first_start_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def boom(*a, **k):
        raise OSError(errno.EPERM, "no")

    monkeypatch.setattr(os, "link", boom)
    db = tmp_path / "vllm-warden.db"
    with ThreadPoolExecutor(16) as ex:
        keys = list(ex.map(lambda _: load_or_create_network_key(db), range(32)))
    assert len({k.key for k in keys}) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["cobs2_network_key"]


def test_loose_memo_is_exact_and_bounded(monkeypatch) -> None:
    from app.cobs2 import ids as ids_mod

    c = _canon("hi", "  yo\n", "more <system-reminder>r</system-reminder>")
    warm = ids_for(c, KEY, b"")
    ids_mod._LOOSE_MEMO.clear()
    cold = ids_for(c, KEY, b"")
    assert warm == cold == ids_for(c, KEY, b"")
    # a different key or salt never hits another's entries
    other = ids_for(c, NetworkKey(epoch=1, key=b"z" * 32), b"")
    assert set(other.loose_msgs).isdisjoint(cold.loose_msgs)
    salted = ids_for(c, KEY, b"s")
    assert set(salted.loose_nodes).isdisjoint(cold.loose_nodes)
    # only ids are held, and never more than the bound
    assert all(len(k) == 17 and len(v) == 17 for k, v in ids_mod._LOOSE_MEMO.items())
    monkeypatch.setattr(ids_mod, "_LOOSE_MEMO_MAX", 5)
    ids_for(_canon(*[f"m{i}" for i in range(10)]), KEY, b"")
    assert len(ids_mod._LOOSE_MEMO) <= 5


# -- sessions pages, task 1: conversation id --------------------------------

from app.cobs2.ids import conv_id  # noqa: E402

_K = NetworkKey(epoch=1, key=b"c" * 32)


def test_conv_id_is_stable_across_turns_and_reminders() -> None:
    a = ids_for(_canon("hi", system="S <system-reminder>Mon</system-reminder>"), _K, b"")
    b = ids_for(_canon("hi", "x", "y", system="S <system-reminder>Tue</system-reminder>"), _K, b"")
    ca = conv_id(a, _K, b"")
    assert ca is not None and ca == conv_id(b, _K, b"") and len(ca) == 34


def test_conv_id_differs_per_first_message_and_salt() -> None:
    a = ids_for(_canon("hi"), _K, b"")
    b = ids_for(_canon("ho"), _K, b"")
    assert conv_id(a, _K, b"") != conv_id(b, _K, b"")
    assert conv_id(a, _K, b"") != conv_id(a, _K, b"salt")


def test_conv_id_none_without_body() -> None:
    assert conv_id(ids_for(_canon(), _K, b""), _K, b"") is None
