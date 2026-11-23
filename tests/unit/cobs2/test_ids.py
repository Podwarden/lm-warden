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


# -- Prompt Lens dock, task 3: separate sys and tools ids (display only) --------

from app.cobs2.canonical import canon_json, loose_head  # noqa: E402
from app.cobs2.ids import HeadPartIds, _h, head_part_ids  # noqa: E402

_GOLDEN_BODY = {
    "messages": [
        {"role": "system", "content": "S\n2026-11-22 <system-reminder>r</system-reminder>"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a  b"},
    ],
    "tools": [{"type": "function", "function": {"name": "t1", "parameters": {"b": 1, "a": 2}}}],
}

#: Recorded at 721f308c (before head_part_ids existed): ids_for and conv_id
#: must stay byte-identical, so stored lineage and routing keep matching.
_GOLDEN = {
    b"": (
        Ids(
            head="015d9d887624bb81310e0de14c7743ef93",
            msgs=("015e41db7768a5a721549e278d5c769aa4", "014543dc3b0f79cab25465bc9402346a3d"),
            nodes=("01788487928e9c7204abbdfeb808d34aa7", "01d7d8be9556b062485552d84194dd64bd"),
            loose_head="01f1de4db7d5a74392fba521b60931523c",
            loose_msgs=(
                "01f9b32e7b33446a7d407a692ec84ecff6",
                "01862f55191495fe30e7847132dac6444e",
            ),
            loose_nodes=(
                "01c07904a352fdb2621bded6a1da990c17",
                "01f0540ad7bd099491cd5976155927276d",
            ),
        ),
        "0125f0d6a40585a4105e7a0aba2830592d",
    ),
    b"s": (
        Ids(
            head="011d48b05f98c5e2a0180592d3318b98ea",
            msgs=("0175784517d88771fe8eb3e39158ee17e4", "01f5c15ce9698eeecac8118f78678e7250"),
            nodes=("01e6392185eff754a80735ba465f89d4d3", "019ac7d6e823668eb2d90b92e5a239f0c5"),
            loose_head="01590c7727fe7c25eaedec7d9e226c5c8f",
            loose_msgs=(
                "0105d175db6eb5cfe1851bd534844f36fe",
                "011f902ddf256fe64ec456a3362c92d26d",
            ),
            loose_nodes=(
                "01a499d284965001b94407e99bf859d51a",
                "01d473c07ea1d1cebf1640b751172dc6a3",
            ),
        ),
        "01e09640da72510a15080c0591917aa9fb",
    ),
}


def _gcanon(body: dict):
    c = canonicalize(body)
    assert c is not None
    return c


def _parts(c, key: NetworkKey = KEY, salt: bytes = b"") -> HeadPartIds:
    return head_part_ids(c, key, salt, ids_for(c, key, salt).head)


def test_head_id_unchanged() -> None:
    from app.cobs2 import ids as ids_mod

    c = _gcanon(_GOLDEN_BODY)
    for salt, (want, want_conv) in _GOLDEN.items():
        ids_mod._LOOSE_MEMO.clear()
        cold = ids_for(c, KEY, salt)
        assert cold == want and conv_id(cold, KEY, salt) == want_conv
        # head part ids share the memo: computing them leaves ids_for as it was
        head_part_ids(c, KEY, salt, cold.head)
        assert ids_for(c, KEY, salt) == want
        assert conv_id(ids_for(c, KEY, salt), KEY, salt) == want_conv


def test_sys_and_tools_ids_are_independent() -> None:
    a = _gcanon(
        {"messages": [{"role": "system", "content": "S"}], "tools": [{"function": {"name": "t1"}}]}
    )
    b = _gcanon(
        {"messages": [{"role": "system", "content": "S"}], "tools": [{"function": {"name": "t2"}}]}
    )
    pa, pb = _parts(a), _parts(b)
    assert pa.sys == pb.sys and pa.tools != pb.tools and len(pa.sys) == 34
    assert pa.loose_sys == pb.loose_sys and pa.loose_tools != pb.loose_tools
    c = _gcanon(
        {"messages": [{"role": "system", "content": "T"}], "tools": [{"function": {"name": "t1"}}]}
    )
    pc = _parts(c)
    assert pc.sys != pa.sys and pc.tools == pa.tools and pc.loose_tools == pa.loose_tools
    allids = [pa.sys, pa.tools, pa.loose_sys, pa.loose_tools]
    assert all(len(i) == 34 and bytes.fromhex(i)[0] == KEY.epoch for i in allids)
    assert len(set(allids)) == 4  # each label is its own namespace


def test_loose_sys_ignores_a_date_line() -> None:
    a = _gcanon({"messages": [{"role": "system", "content": "S\n2026-11-22"}]})
    b = _gcanon({"messages": [{"role": "system", "content": "S\n2026-11-23"}]})
    pa, pb = _parts(a), _parts(b)
    assert pa.sys != pb.sys
    assert pa.loose_sys == pb.loose_sys
    assert pa.tools == pb.tools and pa.loose_tools == pb.loose_tools


def test_head_part_ids_follow_the_open_spec_labels_and_payloads() -> None:
    # kvsig decision 0002: sys over {"head": ...}, tools over {"tools": ...}, and
    # the loose halves over the same keys of the loose head. The construction is
    # this module's H (cobs2 prefix, canon_json), so the bytes differ from kvsig.
    c = _gcanon(_GOLDEN_BODY)
    for salt in (b"", b"s"):
        p = _parts(c, salt=salt)
        lh, lt = loose_head(c)
        assert p.sys == _h(KEY, b"sys", salt, canon_json({"head": c.head})).hex()
        assert p.tools == _h(KEY, b"tools", salt, canon_json({"tools": c.tools})).hex()
        assert p.loose_sys == _h(KEY, b"loose-sys", salt, canon_json({"head": lh})).hex()
        assert p.loose_tools == _h(KEY, b"loose-tools", salt, canon_json({"tools": lt})).hex()
        assert ids_for(c, KEY, salt).head not in (p.sys, p.tools)


def test_head_part_ids_isolated_by_key_and_salt() -> None:
    c = _gcanon(_GOLDEN_BODY)
    a = _parts(c)
    b = _parts(c, key=NetworkKey(1, b"z" * 32))
    s = _parts(c, salt=b"s")
    for x in (b, s):
        assert {a.sys, a.tools, a.loose_sys, a.loose_tools}.isdisjoint(
            {x.sys, x.tools, x.loose_sys, x.loose_tools}
        )


def test_head_part_ids_hash_once_per_head(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.cobs2 import ids as ids_mod

    ids_mod._LOOSE_MEMO.clear()
    c = _gcanon(_GOLDEN_BODY)
    head = ids_for(c, KEY, b"").head
    first = head_part_ids(c, KEY, b"", head)
    tagged = [k for k in ids_mod._LOOSE_MEMO if len(k) != 17]
    assert len(tagged) == 1 and len(ids_mod._LOOSE_MEMO[tagged[0]]) == 4 * 17
    assert tagged[0].endswith(bytes.fromhex(head))

    def boom(*a, **k):
        raise AssertionError("head parts hashed again")

    monkeypatch.setattr(ids_mod, "loose_head", boom)
    monkeypatch.setattr(ids_mod, "_h", boom)
    # a longer turn of the same conversation reuses the memo entry
    longer = _gcanon(
        {
            **_GOLDEN_BODY,
            "messages": [*_GOLDEN_BODY["messages"], {"role": "user", "content": "more"}],
        }
    )
    assert head_part_ids(longer, KEY, b"", head) == first
