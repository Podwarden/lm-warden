"""Namespace and block key derivation."""

import dataclasses

from lmw_kv.keys import Namespace, block_key, namespace_id, root_key

BASE = Namespace(
    "Qwen/Qwen3.5-4B",
    "851bf6e8",
    "tok1",
    "bfloat16",
    528,
    "hybrid-gdn",
    ("full", "gdn0", "gdn1", "gdn2"),
    1,
    1,
    "vllm-0.32",
    1,
)


def test_namespace_id_changes_with_every_field() -> None:
    seen = {namespace_id(BASE)}
    for f in dataclasses.fields(BASE):
        v = getattr(BASE, f.name)
        other = v + 1 if isinstance(v, int) else (v + ("x",) if isinstance(v, tuple) else v + "x")
        seen.add(namespace_id(dataclasses.replace(BASE, **{f.name: other})))
    assert len(seen) == len(dataclasses.fields(BASE)) + 1
    assert all(len(i) == 32 for i in seen)


def test_block_key_chains_and_depends_on_salt_parent_tokens() -> None:
    s, ns = bytes(32), namespace_id(BASE)
    r = root_key(s, b"", ns)
    assert len(r) == 32
    k1 = block_key(s, b"", ns, r, [1, 2, 3])
    assert len(k1) == 32
    assert block_key(s, b"", ns, r, [1, 2, 3]) == k1
    assert block_key(s, b"scope", ns, r, [1, 2, 3]) != k1
    assert block_key(s, b"", ns, k1, [1, 2, 3]) != k1
    assert block_key(s, b"", ns, r, [1, 2, 4]) != k1
    assert block_key(bytes([1]) * 32, b"", ns, r, [1, 2, 3]) != k1


# ---- known-answer vectors (T1) ----------------------------------------------------
# These bytes are a persisted contract: blocks on NVMe are found by these keys, and
# other engines' adapters derive them from PROTOCOL.md. A change here orphans every
# stored block. Never "fix" a vector to make a test pass; bump layout_version instead.

KAT_NS = Namespace("org/model", "r1", "tok", "fp8", 16, "hnd", ("full", "gdn"), 1, 1, "test", 1)
KAT_JSON = (
    b'{"attention_layout":"hnd","block_size":16,"engine_family":"test","kv_dtype":"fp8",'
    b'"kv_groups":["full","gdn"],"layout_version":1,"model_repo":"org/model","pp":1,'
    b'"revision_sha":"r1","tokenizer_sha":"tok","tp":1}'
)
KAT_SECRET = bytes(range(32))


def test_known_answer_namespace_id() -> None:
    import hashlib

    want = "1040228e1c4af1f3270767fb15964b9f3e83c48b22525ea45659c674c90221d4"
    assert hashlib.sha256(b"ns\x00" + KAT_JSON).hexdigest() == want  # from the spec alone
    assert namespace_id(KAT_NS).hex() == want


def test_known_answer_root_and_block_keys_without_salt() -> None:
    import hashlib
    import hmac
    import struct

    ns = namespace_id(KAT_NS)
    root = root_key(KAT_SECRET, b"", ns)
    assert root.hex() == "ffd19c51b3df702ea5c0e1d49ccbb1514127ed54fc13d77b3e1a631c680435e8"
    assert root == hmac.new(KAT_SECRET, b"root\x00\x00" + ns, hashlib.sha256).digest()
    b0 = block_key(KAT_SECRET, b"", ns, root, list(range(16)))
    assert b0.hex() == "d51d2dfcc1c7d9cfe31a2c1d11a05b24bf749635fad595ebf4247b0939b8ebfa"
    toks = b"".join(struct.pack("<I", t) for t in range(16))
    assert b0 == hmac.new(KAT_SECRET, b"blk\x00\x00" + ns + root + toks, hashlib.sha256).digest()
    b1 = block_key(KAT_SECRET, b"", ns, b0, list(range(16, 32)))
    assert b1.hex() == "1598d74a71e78da4932e9d20b94f2395f03a22cba058ed7be646d9684a2771df"


def test_known_answer_with_a_salt_and_wide_token_ids() -> None:
    ns = namespace_id(KAT_NS)
    salt = b"\x01\x02\x03"
    root = root_key(KAT_SECRET, salt, ns)
    assert root.hex() == "9f64fe24ab4132d5d55425ae61ccb39a7c4c62cd39800611800e7cfae741d7fe"
    toks = [0, 1, 2**32 - 1, 70000, *range(12)]
    assert (
        block_key(KAT_SECRET, salt, ns, root, toks).hex()
        == "00f79be71f9f3e89fc35338fe8bfeceeb74a53445e61b06b6d649aa95143ae89"
    )


def test_known_answer_of_the_batched_adapter_keys_and_the_cache_salt_fold() -> None:
    from lmw_kv import connector_logic as cl

    ns = namespace_id(KAT_NS)
    got = cl.block_keys_for(KAT_SECRET, b"", ns, list(range(33)), 16)  # 2 whole blocks
    assert [k.hex() for k in got] == [
        "d51d2dfcc1c7d9cfe31a2c1d11a05b24bf749635fad595ebf4247b0939b8ebfa",
        "1598d74a71e78da4932e9d20b94f2395f03a22cba058ed7be646d9684a2771df",
    ]
    salt = cl.fold_cache_salt(b"", "tenant-a")
    assert salt is not None
    assert salt.hex() == "3bd5cbf832e38e1220908cc49ed4cf783d9c801d61898daa720e5955ec09249a"
    assert (
        cl.block_keys_for(KAT_SECRET, salt, ns, list(range(16)), 16)[0].hex()
        == "4f6a265a42e8c87a5128f5f4ffb9b3894cb56662cd48ae1d700660b7dac4008d"
    )
    assert cl.fold_cache_salt(b"\xaa", None) == b"\xaa"
