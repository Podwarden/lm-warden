"""The connector's vLLM-free logic. Never imports ``lmw_kv.connector`` (that needs vLLM)."""

from __future__ import annotations

import logging
import os
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import connector_logic as cl
from lmw_kv import scopes
from lmw_kv.client import Client
from lmw_kv.keys import Namespace, block_key, namespace_id, root_key
from lmw_kv.store import Store
from tests.unit.lmw_kv.test_store import _cfg as _store_cfg
from tests.unit.lmw_kv.test_store import opened  # noqa: F401  (fixture)

SECRET = bytes(range(32))
BS = 4  # tokens per block in the small tests

GROUPS = [
    cl.GroupDesc(gid=0, kind="MambaSpec", n_layers=8, page_bytes=100, is_state=True),
    cl.GroupDesc(gid=1, kind="FullAttentionSpec", n_layers=8, page_bytes=100, is_state=False),
]


def _model_ns(**over: Any) -> Any:
    kw: dict[str, Any] = dict(
        model_repo="org/model",
        revision="r1",
        tokenizer="org/model@r1",
        kv_dtype="auto/bfloat16",
        block_size=BS,
        groups=GROUPS,
        tp=1,
        pp=1,
        engine_family="vllm-0.32.0",
    )
    kw.update(over)
    return cl.model_namespace(**kw)


MODEL_NS = _model_ns()
MODEL_ID = namespace_id(MODEL_NS)
STORE_IDS = {g.gid: namespace_id(cl.store_namespace(MODEL_NS, g, 0)) for g in GROUPS}
CFG = cl.ConnectorConfig(socket="/x.sock", tiers="ram", min_tokens=8)


class FakeClient:
    """In-memory stand-in for ``Client``: (ns, key) -> stored."""

    def __init__(self) -> None:
        self.stored: set[tuple[bytes, bytes]] = set()
        self.calls: list[tuple[Any, ...]] = []
        self.saving: dict[tuple[bytes, bytes], int] = {}
        self.next_off = 0
        self.ended: list[int] = []
        self.load_limit: int | None = None  # emulate NVMe eviction after lookup
        self.loaded: dict[bytes, int] = {}  # blocks handed out per namespace (load_limit)
        self.offsets_override: dict[str, Any] | None = None
        self.nvme = 0  # blocks each begin_load reports as promoted from NVMe
        self.depths: dict[tuple[bytes, bytes], int | None] = {}  # begin_save's depth

    def lookup(self, ns_id: bytes, keys: list[bytes]) -> int:
        self.calls.append(("lookup", ns_id, len(keys)))
        n = 0
        for k in keys:
            if (ns_id, k) not in self.stored:
                break
            n += 1
        return n

    def begin_load(
        self, ns_id: bytes, keys: list[bytes], *, tenant: bytes = b"", tag: str = ""
    ) -> tuple[int, list[dict[str, int]], int]:
        self.calls.append(("begin_load", ns_id, list(keys), tenant, tag))
        n = self.lookup(ns_id, keys)
        if self.load_limit is not None:  # only load_limit blocks of a namespace remain
            n = max(0, min(n, self.load_limit - self.loaded.get(ns_id, 0)))
        self.loaded[ns_id] = self.loaded.get(ns_id, 0) + n
        return 7, [self.offsets_override or {"*": 4096 * i} for i in range(n)], min(self.nvme, n)

    def end_load(self, handle: int, *, used: bool | None = None) -> bool:
        self.ended.append(handle)
        self.calls.append(("end_load", handle, used))
        return True

    def begin_save(
        self,
        ns_id: bytes,
        key: bytes,
        parent: bytes,
        scope: bytes = b"",
        *,
        tenant: bytes = b"",
        scope_tag: bytes = b"",
        pos: bytes = b"",
        depth: int | None = None,
    ) -> dict[str, int] | None:
        self.calls.append(("begin_save", ns_id, key, parent, scope, tenant, scope_tag, pos))
        self.depths[(ns_id, key)] = depth
        if (ns_id, key) in self.stored:
            return None
        if self.offsets_override is not None:
            return self.offsets_override
        self.next_off += 4096
        return {"*": self.next_off}

    def commit_save(self, ns_id: bytes, key: bytes) -> bool:
        self.calls.append(("commit_save", ns_id, key))
        self.stored.add((ns_id, key))
        return True

    def abort_save(self, ns_id: bytes, key: bytes) -> bool:
        self.calls.append(("abort_save", ns_id, key))
        return True

    def record_hits(
        self, tag: str, ram: int, nvme: int, load_s: float, *, tenant: bytes = b""
    ) -> bool:
        self.calls.append(("record_hits", tag, ram, nvme, tenant))
        return True

    def ops(self, name: str) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == name]


class AnyGroup(dict):  # offsets for whatever group name is asked
    def get(self, k: Any, default: Any = None) -> Any:
        return super().get("*", default)


# ---- pure helpers ---------------------------------------------------------------


def test_whole_blocks_only_1000_tokens_at_528_gives_one_key() -> None:
    keys = cl.block_keys_for(SECRET, b"", MODEL_ID, list(range(1000)), 528)
    assert len(keys) == 1
    assert cl.block_keys_for(SECRET, b"", MODEL_ID, list(range(1056)), 528)[0] == keys[0]
    assert len(cl.block_keys_for(SECRET, b"", MODEL_ID, list(range(1056)), 528)) == 2
    assert cl.block_keys_for(SECRET, b"", MODEL_ID, list(range(527)), 528) == []


def test_block_keys_match_the_reference_chain_and_the_protocol_known_answer() -> None:
    toks = [5, 70000, 2**32 - 1, 0, 9, 9, 9, 9, 1]
    keys = cl.block_keys_for(SECRET, b"\x01\x02", MODEL_ID, toks, BS)
    parent = root_key(SECRET, b"\x01\x02", MODEL_ID)
    want = []
    for i in range(2):
        parent = block_key(SECRET, b"\x01\x02", MODEL_ID, parent, toks[i * BS : (i + 1) * BS])
        want.append(parent)
    assert keys == want
    ns = bytes.fromhex("1040228e1c4af1f3270767fb15964b9f3e83c48b22525ea45659c674c90221d4")
    got = cl.block_keys_for(SECRET, b"", ns, list(range(32)), 16)
    assert [k.hex() for k in got] == [
        "d51d2dfcc1c7d9cfe31a2c1d11a05b24bf749635fad595ebf4247b0939b8ebfa",
        "1598d74a71e78da4932e9d20b94f2395f03a22cba058ed7be646d9684a2771df",
    ]


def test_matched_beyond_gpu_is_never_negative() -> None:
    assert cl.matched_beyond_gpu(0, 3, 528) == 3 * 528
    assert cl.matched_beyond_gpu(528, 3, 528) == 2 * 528
    assert cl.matched_beyond_gpu(3 * 528, 3, 528) == 0
    assert cl.matched_beyond_gpu(10 * 528, 3, 528) == 0
    assert cl.matched_beyond_gpu(0, 0, 528) == 0


def test_never_load_the_last_prompt_token() -> None:
    assert cl.max_load_blocks(1056, 528) == 1
    assert cl.max_load_blocks(1057, 528) == 2
    assert cl.max_load_blocks(1, 528) == 0


def test_read_secret(tmp_path: Path) -> None:
    good = tmp_path / "k"
    good.write_text("v1:" + SECRET.hex() + "\n")
    assert cl.read_secret(str(good)) == SECRET
    assert cl.read_secret(None) is None
    assert cl.read_secret(str(tmp_path / "missing")) is None
    for bad in ["", "v2:" + SECRET.hex(), "v1:" + SECRET.hex()[:-2], "v1:zz" + "0" * 62, "junk"]:
        good.write_text(bad + "\n")
        assert cl.read_secret(str(good)) is None, bad


def test_parse_scope() -> None:
    assert cl.parse_scope(None) == b""
    assert cl.parse_scope("") == b""
    assert cl.parse_scope("0aff") == b"\x0a\xff"
    assert cl.parse_scope("xyz") is None
    assert cl.parse_scope("0" * 512) is None  # 256 bytes, longer than a salt may be
    assert cl.parse_scope(5) is None


def test_parse_config() -> None:
    c = cl.parse_config({"socket": "/s", "tiers": "ram+nvme"})
    assert c == cl.ConnectorConfig("/s", "ram+nvme", cl.DEFAULT_MIN_TOKENS)
    assert cl.DEFAULT_MIN_TOKENS == 2048
    assert cl.parse_config({"socket": "/s", "tiers": "ram", "min_tokens": "100"}).min_tokens == 100
    assert cl.parse_config(None) is None
    assert cl.parse_config({"tiers": "ram"}) is None
    assert cl.parse_config({"socket": "/s", "tiers": "disk"}) is None
    assert cl.parse_config({"socket": "/s", "tiers": "ram", "min_tokens": "lots"}) is None
    assert cl.parse_config({"socket": "/s", "tiers": "ram", "min_tokens": -1}) is None


def test_namespaces() -> None:
    assert MODEL_NS.block_size == BS
    assert MODEL_NS.kv_groups == ("g0.MambaSpec", "g1.FullAttentionSpec")
    assert MODEL_NS.layout_version == 1
    ids = {namespace_id(cl.store_namespace(MODEL_NS, g, r)) for g in GROUPS for r in (0, 1)}
    assert len(ids) == 4 and MODEL_ID not in ids
    # the store namespace still commits to the whole model, revision included
    other = _model_ns(revision="r2")
    assert namespace_id(cl.store_namespace(other, GROUPS[0], 0)) != STORE_IDS[0]
    assert cl.store_group_name(GROUPS[1], 1) == "g1.FullAttentionSpec@tp1"


def test_in_bounds() -> None:
    assert cl.in_bounds(0, 10, 10)
    assert not cl.in_bounds(1, 10, 10)
    assert not cl.in_bounds(-1, 1, 10)
    assert not cl.in_bounds(True, 1, 10)
    assert not cl.in_bounds("0", 1, 10)


# ---- matching -------------------------------------------------------------------


def _store(fc: FakeClient, gid: int, keys: list[bytes]) -> None:
    fc.stored.update((STORE_IDS[gid], k) for k in keys)


def _keys(n_tokens: int, salt: bytes = b"") -> list[bytes]:
    return cl.block_keys_for(SECRET, salt, MODEL_ID, list(range(n_tokens)), BS)


def test_matched_blocks_needs_state_at_the_match_point() -> None:
    fc = FakeClient()
    keys = _keys(40)
    _store(fc, 1, keys[:6])
    _store(fc, 0, [keys[2]])  # state only at block 2
    m = cl.matched_blocks(fc.lookup, [STORE_IDS[1]], [STORE_IDS[0]], keys)
    assert m == 3
    _store(fc, 0, [keys[5]])
    assert cl.matched_blocks(fc.lookup, [STORE_IDS[1]], [STORE_IDS[0]], keys) == 6
    assert cl.matched_blocks(fc.lookup, [STORE_IDS[1]], [], keys) == 6


def test_matched_blocks_without_any_state_is_a_miss_and_probes_are_capped() -> None:
    fc = FakeClient()
    keys = _keys(400)
    _store(fc, 1, keys)
    _store(fc, 0, [keys[0]])
    assert cl.matched_blocks(fc.lookup, [STORE_IDS[1]], [STORE_IDS[0]], keys, probes=16) == 0
    assert len(fc.ops("lookup")) == 1 + 16
    assert cl.matched_blocks(fc.lookup, [STORE_IDS[1]], [STORE_IDS[0]], keys, probes=100) == 1


def test_matched_blocks_fails_open_on_a_raising_lookup() -> None:
    def boom(ns: bytes, keys: list[bytes]) -> int:
        raise RuntimeError("daemon gone")

    assert cl.matched_blocks(boom, [STORE_IDS[1]], [STORE_IDS[0]], _keys(40)) == 0


# ---- scheduler logic ------------------------------------------------------------


def _sched(fc: Any, cfg: cl.ConnectorConfig = CFG) -> cl.SchedulerLogic:
    return cl.SchedulerLogic(fc, SECRET, cfg, MODEL_ID, GROUPS, STORE_IDS, BS)


def test_short_prompt_gives_no_lookup_and_no_save_keys() -> None:
    fc = FakeClient()
    s = _sched(fc)
    toks = list(range(7))  # below min_tokens = 8
    assert s.match("r", toks, {}, 0) == (0, False)
    assert fc.calls == []
    plan = s.build([("r", 0, 7)], lambda rid: [[1, 2], [3, 4]])
    assert plan.saves == [] and plan.loads == []


def test_saves_whole_prompt_blocks_and_state_only_at_an_aligned_step_end() -> None:
    fc = FakeClient()
    s = _sched(fc)
    toks = list(range(18))  # 4 whole blocks + 2 tokens
    assert s.match("r", toks, {"lmw_scope": "aa"}, 0) == (0, False)
    blocks = [[None, None, 10, None, None], [20, 21, 22, 23, 24]]
    keys = _keys(18, b"\xaa")
    root = root_key(SECRET, b"\xaa", MODEL_ID)
    # step 1 computes 12 tokens (3 whole blocks, aligned): attention 0..2, state at block 2
    plan = s.build([("r", 0, 12)], lambda rid: blocks)
    got = {(e.gid, e.block_id, e.key, e.parent, e.scope) for e in plan.saves}
    assert got == {
        (1, 20, keys[0], root, b"\xaa"),
        (1, 21, keys[1], keys[0], b"\xaa"),
        (1, 22, keys[2], keys[1], b"\xaa"),
        (0, 10, keys[2], keys[1], b"\xaa"),
    }
    # step 2 ends at the prompt end (18, not aligned): attention block 3 only, no state
    plan = s.build([("r", 12, 6)], lambda rid: blocks)
    assert [(e.gid, e.block_id) for e in plan.saves] == [(1, 23)]
    # decode steps save nothing more
    assert s.build([("r", 18, 1)], lambda rid: blocks).saves == []


def test_null_or_missing_blocks_are_not_saved() -> None:
    fc = FakeClient()
    s = _sched(fc)
    s.match("r", list(range(8)), {}, 0)
    plan = s.build([("r", 0, 8)], lambda rid: [[None, None], [None, 31]])
    assert [(e.gid, e.block_id) for e in plan.saves] == [(1, 31)]
    s.match("q", list(range(8)), {}, 0)
    assert s.build([("q", 0, 8)], lambda rid: None).saves == []


def test_bad_scope_is_never_cached() -> None:
    fc = FakeClient()
    s = _sched(fc)
    assert s.match("r", list(range(40)), {"lmw_scope": "not-hex"}, 0) == (0, False)
    assert s.build([("r", 0, 40)], lambda rid: [[1] * 10, [2] * 10]).saves == []


def test_match_loads_async_beyond_the_gpu_hit_and_plans_the_load() -> None:
    fc = FakeClient()
    s = _sched(fc)
    toks = list(range(21))  # 5 whole blocks; at most 5 loadable ((21-1)//4)
    keys = _keys(21)
    _store(fc, 1, keys[:4])
    _store(fc, 0, [keys[3]])
    assert s.match("r", toks, {"lmw_req": "req-1"}, 4) == (12, True)
    blocks = [[None, None, None, 40, None, None], [50, 51, 52, 53, 54, 55]]
    s.after_alloc("r", blocks, 12)
    plan = s.build([], lambda rid: blocks)
    assert len(plan.loads) == 1
    job = plan.loads[0]
    assert (job.req_id, job.tag, job.tokens, job.doomed) == ("r", "req-1", 12, False)
    ents = {e.gid: (e.keys, e.block_ids) for e in job.entries}
    assert ents[1] == (keys[1:4], [51, 52, 53])
    assert ents[0] == ([keys[3]], [40])
    # the loaded blocks are not saved again once the request runs
    plan = s.build([("r", 16, 5)], lambda rid: blocks)
    assert [(e.gid, e.block_id) for e in plan.saves] == [(1, 54)]


def test_tag_defaults_to_the_engine_request_id() -> None:
    fc = FakeClient()
    s = _sched(fc)
    keys = _keys(12)
    _store(fc, 1, keys[:2])
    _store(fc, 0, [keys[1]])
    assert s.match("r9", list(range(12)), None, 0) == (8, True)
    s.after_alloc("r9", [[None, 1, None], [2, 3, 4]], 8)
    assert s.build([], lambda rid: None).loads[0].tag == "r9"


def test_a_tag_over_the_daemon_limit_becomes_its_hash() -> None:
    import hashlib

    long_id = "r" * 300
    wide = "\u00e9" * 128  # 128 characters, 256 UTF-8 bytes
    fits = "\u00e9" * 127 + "x"  # 255 bytes: kept as it is
    assert cl.load_tag(fits) == fits
    for v in (long_id, wide, "\ud800" + "x"):  # the last one is not valid UTF-8
        t = cl.load_tag(v)
        assert t == cl.load_tag(v) and len(t) == 64 and t != v
        assert len(t.encode("utf-8")) <= cl.LOAD_TAG_MAX_BYTES
    assert cl.load_tag(long_id) == hashlib.sha256(long_id.encode()).hexdigest()
    assert cl.load_tag(long_id) != cl.load_tag(long_id + "2")
    for rid, params in ((long_id, None), ("r", {"lmw_req": wide})):
        fc = FakeClient()
        s = _sched(fc)
        keys = _keys(12)
        _store(fc, 1, keys[:2])
        _store(fc, 0, [keys[1]])
        assert s.match(rid, list(range(12)), params, 0) == (8, True)
        s.after_alloc(rid, [[None, 1, None], [2, 3, 4]], 8)
        tag = s.build([], lambda r: None).loads[0].tag
        assert tag == cl.load_tag(long_id if params is None else wide) and len(tag) == 64


def test_a_null_block_where_a_load_must_land_dooms_the_job() -> None:
    fc = FakeClient()
    s = _sched(fc)
    keys = _keys(12)
    _store(fc, 1, keys[:2])
    _store(fc, 0, [keys[1]])
    assert s.match("r", list(range(12)), {}, 0) == (8, True)
    s.after_alloc("r", [[None, None, None], [2, 3, 4]], 8)  # no state block allocated
    assert s.build([], lambda rid: None).loads[0].doomed


def test_zero_external_tokens_plans_nothing() -> None:
    s = _sched(FakeClient())
    s.match("r", list(range(40)), {}, 0)
    s.after_alloc("r", [[1], [2]], 0)
    assert s.build([], lambda rid: None).loads == []


def test_raising_client_fails_open() -> None:
    class Boom:
        def __getattr__(self, name: str) -> Any:
            def f(*a: Any, **k: Any) -> Any:
                raise RuntimeError("boom")

            return f

    s = _sched(Boom())
    assert s.match("r", list(range(40)), {}, 0) == (0, False)


@pytest.fixture
def hung_socket() -> Iterator[str]:
    """A Unix socket that accepts and never answers (a hung daemon)."""
    d = tempfile.mkdtemp(prefix="lmwkv")
    path = os.path.join(d, "s")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        srv.settimeout(0.05)
        while not stop.is_set():
            try:
                held.append(srv.accept()[0])
            except OSError:
                pass

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    yield path
    stop.set()
    t.join()
    for c in held:
        c.close()
    srv.close()
    os.unlink(path)
    os.rmdir(d)


def test_hung_daemon_gives_a_miss_within_the_deadline(hung_socket: str) -> None:
    s = _sched(Client(hung_socket, timeout_s=0.05))
    t0 = time.monotonic()
    assert s.match("r", list(range(400)), {}, 0) == (0, False)
    assert time.monotonic() - t0 < 0.5


def test_missing_daemon_gives_a_miss() -> None:
    s = _sched(Client("/nonexistent/lmw-kv.sock"))
    assert s.match("r", list(range(400)), {}, 0) == (0, False)


def test_finishing_never_delays_the_free_and_forgets_the_request() -> None:
    # vLLM 0.32 holds an aborted WAITING_FOR_REMOTE_KVS request's blocks itself until
    # the load is reported in finished_recving; the connector must not ask for more.
    fc = FakeClient()
    s = _sched(fc)
    keys = _keys(12)
    _store(fc, 1, keys[:2])
    _store(fc, 0, [keys[1]])
    s.match("r", list(range(12)), {}, 0)
    s.after_alloc("r", [[None, 1, None], [2, 3, 4]], 8)
    s.build([], lambda rid: None)
    assert s.finished("r") is None
    assert s.build([("r", 8, 4)], lambda rid: [[None, 1, None], [2, 3, 4]]).saves == []
    assert s.finished("never-seen") is None


def test_a_request_gets_at_most_one_load_attempt() -> None:
    fc = FakeClient()
    s = _sched(fc)
    keys = _keys(12)
    _store(fc, 1, keys[:2])
    _store(fc, 0, [keys[1]])
    toks = list(range(12))
    assert s.match("r", toks, {}, 0) == (8, True)
    s.after_alloc("r", [[None, 1, None], [2, 3, 4]], 8)
    s.build([], lambda rid: None)
    t = cl.LoadTracker()
    t.start("r")
    t.finish("r", False)
    assert t.poll() == ({"r"}, {"r"})
    # recompute brings r back with num_computed_tokens = 0: no second attempt
    assert s.match("r", toks, {}, 0) == (0, False)
    assert s.build([], lambda rid: None).loads == []
    # another request with the same prompt still matches
    assert s.match("q", toks, {}, 0) == (8, True)
    # once r is forgotten, the id may be reused
    s.finished("r")
    assert s.match("r", toks, {}, 0) == (8, True)


def test_a_load_planned_for_an_unknown_request_is_doomed() -> None:
    s = _sched(FakeClient())
    s.after_alloc("ghost", [[None, 1, None], [2, 3, 4]], 8)
    job = s.build([], lambda rid: None).loads[0]
    assert job.doomed and job.entries == []


def test_a_load_that_lists_no_blocks_is_doomed() -> None:
    fc = FakeClient()
    s = _sched(fc)
    keys = _keys(12)
    _store(fc, 1, keys[:2])
    _store(fc, 0, [keys[1]])
    s.match("r", list(range(12)), {}, 0)
    s.after_alloc("r", [], 8)  # e.g. vLLM's block lists could not be read
    job = s.build([], lambda rid: None).loads[0]
    assert job.doomed and job.entries == []
    assert (
        cl.run_load_job(
            fc,
            job,
            STORE_IDS,
            NAMES,
            SIZES,
            1 << 20,
            lambda g, b, o: None,
            lambda: None,
            lambda: True,
            window=1000,
        )
        is None
    )


# ---- worker: load tracking ------------------------------------------------------


def test_tracker_reports_every_load_once_and_failures_at_request_level() -> None:
    t = cl.LoadTracker()
    t.start("a")
    t.start("b")
    assert t.poll() == (set(), set())
    t.finish("a", True)
    t.finish("b", False)
    assert t.poll() == ({"a", "b"}, {"b"})  # failed is in finished_recving AND failed_recving
    assert t.poll() == (set(), set())


def test_tracker_never_reports_a_load_as_sending() -> None:
    # Abort while loading: vLLM frees the blocks on finished_recving, so the load is
    # reported there exactly once, never as finished_sending (that would trip the
    # scheduler's "req_id in self.requests" assert once the request is gone).
    t = cl.LoadTracker()
    t.start("r")
    t.finish("r", False)
    recving, failed = t.poll()
    assert recving == {"r"} and failed == {"r"}
    assert not hasattr(t, "abort")


# ---- worker: running loads and saves --------------------------------------------

NAMES = {g.gid: f"g{g.gid}" for g in GROUPS}
SIZES = {0: 1000, 1: 1000}


def _job(fc: FakeClient, n: int = 3) -> cl.LoadJob:
    keys = _keys(40)
    _store(fc, 1, keys[:n])
    _store(fc, 0, [keys[n - 1]])
    return cl.LoadJob(
        req_id="r",
        tag="t",
        tokens=n * BS,
        entries=[
            cl.LoadEntry(1, keys[:n], list(range(100, 100 + n))),
            cl.LoadEntry(0, [keys[n - 1]], [200], state=True),
        ],
        doomed=False,
    )


def _run_load(fc: FakeClient, job: cl.LoadJob, limit: int = 1 << 20, same: bool = True) -> Any:
    copies: list[tuple[int, int, int]] = []
    syncs: list[int] = []
    ok = cl.run_load_job(
        fc,
        job,
        STORE_IDS,
        NAMES,
        SIZES,
        limit,
        copy=lambda g, b, o: copies.append((g, b, o)),
        sync=lambda: syncs.append(1),
        same_file=lambda: same,
        window=1000,
        sleep=lambda _s: None,
    )
    return ok, copies, syncs


def test_load_job_copies_every_slot_then_ends_the_load() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 8192})
    got, copies, syncs = _run_load(fc, _job(fc))
    assert got == [0, 0]
    assert sorted(copies) == [(0, 200, 8192), (1, 100, 8192), (1, 101, 8192), (1, 102, 8192)]
    assert syncs and fc.ended == [7, 7]


def test_load_job_shorter_than_planned_fails_but_still_releases_pins() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    fc.load_limit = 2  # NVMe evicted a block between lookup and load
    got, _copies, syncs = _run_load(fc, _job(fc))
    assert got is None
    assert fc.ended  # every handle taken is ended
    assert syncs  # copies already issued are waited for before the pins go


def test_load_job_out_of_range_offset_is_a_miss_never_a_copy() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 4096})
    got, copies, _ = _run_load(fc, _job(fc), limit=4096 + 999)
    assert got is None and copies == []
    fc2 = FakeClient()
    fc2.offsets_override = {"other-group": 0}
    got, copies, _ = _run_load(fc2, _job(fc2))
    assert got is None and copies == []


def test_load_job_doomed_or_daemon_restarted_fails() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    job = _job(fc)
    assert _run_load(fc, job, same=False)[0] is None
    job.doomed = True
    got, copies, _ = _run_load(fc, job)
    assert got is None and copies == []


def test_load_job_with_unreachable_daemon_fails() -> None:
    fc = FakeClient()
    job = _job(fc)
    ok = cl.run_load_job(
        Client("/nonexistent/x.sock"),
        job,
        STORE_IDS,
        NAMES,
        SIZES,
        1 << 20,
        copy=lambda g, b, o: None,
        sync=lambda: None,
        same_file=lambda: True,
        window=1000,
    )
    assert ok is None


def test_load_job_reports_nvme_blocks_and_the_hit_split() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    fc.nvme = 2
    job = _job(fc)  # 3 attention blocks + 1 state snapshot, 12 tokens
    got, _, _ = _run_load(fc, job)
    assert got == [2, 1]
    assert cl.hit_split(job, got, BS) == (4, 8)  # tokens once per block, not per group
    assert cl.hit_split(job, [0, 0], BS) == (12, 0)
    assert cl.hit_split(job, [3, 1], BS) == (0, 12)
    state_only = cl.LoadJob("r", "t", 12, [cl.LoadEntry(0, [b"k" * 32], [1], state=True)])
    assert cl.hit_split(state_only, [1], BS) == (0, 12)
    assert cl.hit_split(state_only, [0], BS) == (12, 0)


def _saves() -> list[cl.SaveEntry]:
    keys = _keys(12)
    return [
        cl.SaveEntry(keys[0], b"r" * 32, cl.BlockScope(b""), 1, 10),
        cl.SaveEntry(keys[1], keys[0], cl.BlockScope(b""), 1, 11),
        cl.SaveEntry(keys[1], keys[0], cl.BlockScope(b""), 0, 12),
    ]


def _run_saves(fc: FakeClient, limit: int = 1 << 20, same: bool = True) -> Any:
    copies: list[tuple[int, int, int]] = []
    n = cl.run_saves(
        fc,
        _saves(),
        STORE_IDS,
        NAMES,
        SIZES,
        limit,
        copy=lambda g, b, o: copies.append((g, b, o)),
        sync=lambda: None,
        same_file=lambda: same,
    )
    return n, copies


def test_saves_copy_into_begun_slots_then_commit() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    n, copies = _run_saves(fc)
    assert n == 3 and len(copies) == 3
    assert len(fc.ops("commit_save")) == 3
    assert fc.ops("begin_save")[0][3] == b"r" * 32


def test_saves_skip_existing_blocks() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    _store(fc, 1, [_saves()[0].key])
    n, copies = _run_saves(fc)
    assert n == 2 and len(copies) == 2


def test_saves_out_of_range_are_aborted_not_copied() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 1 << 20})
    n, copies = _run_saves(fc, limit=1 << 20)
    assert n == 0 and copies == []
    assert len(fc.ops("abort_save")) == 3


def test_saves_abort_when_the_ram_file_changed() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    n, _ = _run_saves(fc, same=False)
    assert n == 0
    assert fc.ops("commit_save") == [] and len(fc.ops("abort_save")) == 3


def test_saves_abort_everything_when_a_copy_raises() -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})

    def bad_copy(g: int, b: int, o: int) -> None:
        raise RuntimeError("cuda")

    n = cl.run_saves(
        fc, _saves(), STORE_IDS, NAMES, SIZES, 1 << 20, bad_copy, lambda: None, lambda: True
    )
    assert n == 0 and fc.ops("commit_save") == []
    assert len(fc.ops("abort_save")) >= 1


# ---- failed loads are logged, once, by run_load_job --------------------------------


def test_a_failed_load_logs_one_warning_naming_the_request(caplog: Any) -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    fc.load_limit = 2  # a block went missing between lookup and load
    with caplog.at_level(logging.WARNING, logger="lmw_kv.connector"):
        got, _copies, _syncs = _run_load(fc, _job(fc))
    assert got is None
    lines = [r.getMessage() for r in caplog.records if "load failed" in r.getMessage()]
    assert lines == ["LM Warden KV load failed for r (incomplete); vLLM recomputes it"]


def test_a_load_that_raises_logs_the_reason(caplog: Any) -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})

    def boom(g: int, b: int, o: int) -> None:
        raise RuntimeError("copy broke")

    with caplog.at_level(logging.WARNING, logger="lmw_kv.connector"):
        got = cl.run_load_job(
            fc,
            _job(fc),
            STORE_IDS,
            NAMES,
            SIZES,
            1 << 20,
            boom,
            lambda: None,
            lambda: True,
            window=1000,
        )
    assert got is None and fc.ended  # pins still released
    (line,) = [r.getMessage() for r in caplog.records if "load failed" in r.getMessage()]
    assert "copy broke" in line and line.startswith("LM Warden KV load failed for r")


def test_a_good_load_logs_nothing(caplog: Any) -> None:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    with caplog.at_level(logging.WARNING, logger="lmw_kv.connector"):
        got, _c, _s = _run_load(fc, _job(fc))
    assert got == [0, 0]
    assert not [r for r in caplog.records if "load failed" in r.getMessage()]


# ---- attaching a worker to the daemon: register, map, register again -------------


class FreshDaemon:
    """Like a fresh daemon: the RAM file only exists after the first register_ns."""

    def __init__(self, ram: Any) -> None:
        self.ram = ram
        self.calls: list[str] = []
        self.nvme: list[bool] = []
        self.staging: list[bool] = []
        self.minor1: list[dict[str, Any]] = []

    def register_ns(
        self,
        ns: Namespace,
        groups: dict[str, int],
        nvme: bool = True,
        staging: bool = False,
        **minor1: Any,
    ) -> bytes:
        self.calls.append("register_ns")
        self.nvme.append(nvme)
        self.staging.append(staging)
        self.minor1.append(minor1)
        self.ram.touch()
        return namespace_id(ns)


_WHO: dict[str, Any] = dict(owner="m1", model=b"\x07" * 16, rank=0, groups=GROUPS)


def _worker_namespaces() -> list[tuple[int, Namespace, dict[str, int]]]:
    return [(g.gid, cl.store_namespace(MODEL_NS, g, 0), {f"g{g.gid}": 1000}) for g in GROUPS]


def test_attach_maps_the_ram_file_only_after_registering(tmp_path: Any) -> None:
    ram = tmp_path / "lmw-kv.pool"
    d = FreshDaemon(ram)

    def map_ram() -> None:
        d.calls.append("map")
        open(ram, "rb").close()  # FileNotFoundError if mapped before register_ns

    ids = cl.attach_store(d, _worker_namespaces(), map_ram, lambda: True, **_WHO)
    assert ids == STORE_IDS
    n = len(GROUPS)
    assert d.calls == ["register_ns"] * n + ["map"] + ["register_ns"] * n


def test_attach_refuses_a_daemon_that_disagrees_or_restarted(tmp_path: Any) -> None:
    ram = tmp_path / "lmw-kv.pool"

    class Wrong(FreshDaemon):
        def register_ns(
            self, ns: Namespace, groups: dict[str, int], nvme: bool = True, **minor1: Any
        ) -> bytes:
            super().register_ns(ns, groups, nvme, **minor1)
            return b"\0" * 32

    with pytest.raises(cl.SetupError, match="register_ns for group"):
        cl.attach_store(Wrong(ram), _worker_namespaces(), lambda: None, lambda: True, **_WHO)
    with pytest.raises(cl.SetupError, match="restarted"):
        cl.attach_store(FreshDaemon(ram), _worker_namespaces(), lambda: None, lambda: False, **_WHO)


@pytest.mark.parametrize(("tiers", "nvme"), [("ram", False), ("ram+nvme", True)])
def test_attach_registers_the_nvme_flag_from_the_tiers(
    tmp_path: Any, tiers: str, nvme: bool
) -> None:
    """I3: kv_store "ram" registers its namespaces with NVMe off."""
    d = FreshDaemon(tmp_path / "lmw-kv.pool")
    assert cl.uses_nvme(tiers) is nvme
    cl.attach_store(
        d, _worker_namespaces(), lambda: None, lambda: True, nvme=cl.uses_nvme(tiers), **_WHO
    )
    assert d.nvme == [nvme] * (2 * len(GROUPS))


# ---- cache_salt isolation (I1) ----------------------------------------------------


def _saved_keys(s: cl.SchedulerLogic, rid: str) -> set[bytes]:
    plan = s.build([(rid, 0, 12)], lambda r: [[None, None, 10], [20, 21, 22]])
    return {e.key for e in plan.saves}


def test_two_cache_salts_on_one_prefix_never_share_blocks() -> None:
    fc = FakeClient()
    s = _sched(fc)
    toks = list(range(13))
    s.match("a", toks, {}, 0, cache_salt="tenant-a")
    s.match("b", toks, {}, 0, cache_salt="tenant-b")
    a, b = _saved_keys(s, "a"), _saved_keys(s, "b")
    assert a and b and not (a & b)
    # what tenant a stored is a miss for tenant b
    _store(fc, 1, list(a))
    _store(fc, 0, list(a))
    assert s.match("b2", toks, {}, 0, cache_salt="tenant-b") == (0, False)
    assert s.match("a2", toks, {}, 0, cache_salt="tenant-a") == (12, True)


def test_no_cache_salt_keeps_the_keys_unchanged() -> None:
    fc = FakeClient()
    s = _sched(fc)
    toks = list(range(13))
    s.match("n", toks, {}, 0, cache_salt=None)
    s.match("e", toks, {}, 0, cache_salt="")
    want = set(_keys(13)[:3])
    assert _saved_keys(s, "n") == want
    assert _saved_keys(s, "e") == want
    s.match("x", toks, {}, 0)  # the default: no salt
    assert _saved_keys(s, "x") == want


def test_cache_salt_folds_into_the_scope_salt() -> None:
    fc = FakeClient()
    s = _sched(fc)
    toks = list(range(13))
    s.match("a", toks, {"lmw_scope": "aa"}, 0, cache_salt="t")
    s.match("b", toks, {"lmw_scope": "bb"}, 0, cache_salt="t")
    s.match("c", toks, {"lmw_scope": "aa"}, 0)
    a, b, c = _saved_keys(s, "a"), _saved_keys(s, "b"), _saved_keys(s, "c")
    assert not (a & b) and not (a & c)
    assert c == set(_keys(13, b"\xaa")[:3])


def test_a_cache_salt_that_is_not_a_string_is_never_cached() -> None:
    fc = FakeClient()
    s = _sched(fc)
    assert s.match("r", list(range(40)), {}, 0, cache_salt=b"raw") == (0, False)
    assert s.build([("r", 0, 40)], lambda rid: [[1] * 10, [2] * 10]).saves == []


# ---- the loader thread (I4, M1) ---------------------------------------------------


def _jobq(*jobs: cl.LoadJob | None) -> Any:
    import queue

    q: queue.Queue[cl.LoadJob | None] = queue.Queue()
    for j in jobs:
        q.put(j)
    return q


def _lj(rid: str) -> cl.LoadJob:
    return cl.LoadJob(rid, rid, 4, [cl.LoadEntry(1, [b"k" * 32], [1])])


def test_a_loader_that_cannot_start_fails_every_load_and_turns_off() -> None:
    tr = cl.LoadTracker()
    off: list[str] = []
    for rid in ("a", "b"):
        tr.start(rid)

    def setup() -> Any:
        raise RuntimeError("CUDA error: no device")

    def run(job: cl.LoadJob, ctx: Any) -> list[int] | None:
        raise AssertionError("never runs without a working setup")

    cl.serve_loads(
        _jobq(_lj("a"), _lj("b"), None),
        tr,
        setup,
        run,
        lambda *a: None,
        lambda e: off.append(str(e)),
    )
    assert off == ["CUDA error: no device"]
    assert tr.poll() == ({"a", "b"}, {"a", "b"})  # finished, as failures: vLLM recomputes


def test_a_raising_load_fails_only_that_request_and_the_loop_goes_on() -> None:
    tr = cl.LoadTracker()

    def run(job: cl.LoadJob, ctx: Any) -> list[int] | None:
        if job.req_id == "bad":
            raise RuntimeError("boom")
        return [0]

    cl.serve_loads(
        _jobq(_lj("bad"), _lj("good"), None), tr, lambda: "ctx", run, lambda *a: None, print
    )
    assert tr.poll() == ({"bad", "good"}, {"bad"})


def test_hits_are_recorded_before_the_load_is_reported_finished() -> None:
    """M1: the api pops the hits once the turn ends; they must be there by then."""
    tr = cl.LoadTracker()
    order: list[str] = []

    def record(job: cl.LoadJob, nvme: list[int], elapsed: float) -> None:
        order.append("record")
        assert tr.poll() == (set(), set())  # not reported yet
        assert elapsed >= 0

    real_finish = tr.finish

    def finish(rid: str, ok: bool) -> None:
        order.append("finish")
        real_finish(rid, ok)

    tr.finish = finish  # type: ignore[method-assign]
    cl.serve_loads(_jobq(_lj("a"), None), tr, lambda: None, lambda j, c: [0], record, print)
    assert order == ["record", "finish"]
    assert tr.poll() == ({"a"}, set())


def test_a_raising_record_still_reports_the_load() -> None:
    tr = cl.LoadTracker()

    def record(job: cl.LoadJob, nvme: list[int], elapsed: float) -> None:
        raise OSError("daemon gone")

    cl.serve_loads(_jobq(_lj("a"), None), tr, lambda: None, lambda j, c: [0], record, print)
    assert tr.poll() == ({"a"}, set())


def test_submit_load_fails_at_once_when_the_loader_is_dead_or_off() -> None:
    tr = cl.LoadTracker()
    queued: list[cl.LoadJob] = []
    assert cl.submit_load(tr, _lj("a"), True, True, queued.append) is True
    assert [j.req_id for j in queued] == ["a"]
    assert tr.poll() == (set(), set())  # in flight
    assert cl.submit_load(tr, _lj("b"), True, False, queued.append) is False  # dead thread
    assert cl.submit_load(tr, _lj("c"), False, True, queued.append) is False  # worker off
    assert [j.req_id for j in queued] == ["a"]
    assert tr.poll() == ({"b", "c"}, {"b", "c"})


# ---- F6: the namespace is keyed by commits, never a branch name -----------------------

SHA_A, SHA_B = "a" * 40, "b" * 40


def test_parse_config_reads_the_resolved_revisions() -> None:
    c = cl.parse_config(
        {"socket": "/s", "tiers": "ram", "revision": SHA_A, "tokenizer_revision": SHA_B}
    )
    assert c is not None and (c.revision, c.tokenizer_revision) == (SHA_A, SHA_B)
    c = cl.parse_config({"socket": "/s", "tiers": "ram"})
    assert c is not None and c.revision is None and c.tokenizer_revision is None


def _cfg(rev: Any, tok: Any) -> cl.ConnectorConfig:
    return cl.ConnectorConfig("/s", "ram", 8, rev, tok)


def test_namespace_revisions_use_the_resolved_commits_not_the_branch() -> None:
    assert cl.namespace_revisions(_cfg(SHA_A, SHA_B), "main", None) == (SHA_A, SHA_B)
    assert cl.namespace_revisions(_cfg(SHA_A, SHA_A), SHA_A, SHA_A) == (SHA_A, SHA_A)


@pytest.mark.parametrize(
    ("rev", "tok", "engine_rev", "engine_tok"),
    [
        (None, None, "main", None),  # nothing resolved: the old "non-empty" check passed this
        ("main", "main", "main", None),  # a branch name is never a namespace
        (SHA_A, None, "main", None),
        (SHA_A.upper(), SHA_A, "main", None),
        (SHA_A, SHA_A, SHA_B, None),  # the engine was started on another commit
        (SHA_A, SHA_A, "main", SHA_B),
    ],
)
def test_namespace_revisions_refuse_anything_but_agreeing_commits(
    rev: Any, tok: Any, engine_rev: Any, engine_tok: Any
) -> None:
    with pytest.raises(ValueError):
        cl.namespace_revisions(_cfg(rev, tok), engine_rev, engine_tok)


def test_a_new_commit_is_a_new_namespace() -> None:
    a = namespace_id(_model_ns(revision=SHA_A, tokenizer=f"org/model@{SHA_A}"))
    b = namespace_id(_model_ns(revision=SHA_B, tokenizer=f"org/model@{SHA_B}"))
    assert a != b


# ---- Plan 2: owner, model, role, rank, tenant and scope tag (Task 7) ---------------

NK = bytes(range(32))
S = scopes.isolation_scope(NK, "key-1", 0)  # an isolated key's raw scope
T = scopes.tenant_tag(NK, "key-1")
M = scopes.model_tag("m1", "r1")


def test_parse_config_reads_the_owner() -> None:
    base = {"socket": "/s", "tiers": "ram"}
    assert cl.parse_config(base).owner == ""  # type: ignore[union-attr]
    assert cl.parse_config({**base, "owner": "m7"}).owner == "m7"  # type: ignore[union-attr]
    for bad in (7, None, b"m7", ["m7"]):
        assert cl.parse_config({**base, "owner": bad}) is None, bad


def test_role_of() -> None:
    assert cl.role_of(GROUPS[0]) == "gdn_state"
    assert cl.role_of(GROUPS[1]) == "attention"
    mtp = cl.GroupDesc(
        gid=2, kind="FullAttentionSpec", n_layers=1, page_bytes=1, is_state=False, eagle=True
    )
    assert cl.role_of(mtp) == "mtp"


def test_parse_tenant() -> None:
    assert cl.parse_tenant(None) == b""
    assert cl.parse_tenant("") == b""
    assert cl.parse_tenant(T.hex()) == T
    assert cl.parse_tenant(T.hex().upper()) == T
    for bad in ("xyz", T.hex()[:-2], T.hex() + "00", " " + T.hex()[1:], 5, b"", T, "00 " * 16):
        assert cl.parse_tenant(bad) is None, bad


def test_register_carries_owner_model_role_rank(tmp_path: Any) -> None:
    d = FreshDaemon(tmp_path / "lmw-kv.pool")
    nss = [(g.gid, cl.store_namespace(MODEL_NS, g, 3), {f"g{g.gid}": 1000}) for g in GROUPS]
    cl.attach_store(d, nss, lambda: None, lambda: True, owner="m1", model=M, rank=3, groups=GROUPS)
    want = [
        {"owner": "m1", "model": M, "role": "gdn_state", "rank": 3},
        {"owner": "m1", "model": M, "role": "attention", "rank": 3},
    ]
    assert d.minor1 == want * 2  # register, map, register again


def _plan(params: dict[str, Any], cache_salt: Any = None) -> cl.StepPlan:
    s = _sched(FakeClient())
    s.match("r", list(range(13)), params, 0, cache_salt=cache_salt)
    return s.build([("r", 0, 12)], lambda rid: [[None, None, 10], [20, 21, 22]])


def _saved(plan: cl.StepPlan) -> FakeClient:
    fc = FakeClient()
    fc.offsets_override = AnyGroup({"*": 0})
    n = cl.run_saves(
        fc,
        plan.saves,
        STORE_IDS,
        NAMES,
        SIZES,
        1 << 20,
        lambda g, b, o: None,
        lambda: None,
        lambda: True,
    )
    assert n == len(plan.saves) == 4
    return fc


def test_tenant_and_scope_tag_reach_saves_and_loads() -> None:
    params = {"lmw_scope": S.hex(), "lmw_tenant": T.hex()}
    fc = _saved(_plan(params))
    begins = fc.ops("begin_save")
    assert begins and all(c[5] == T and c[6] == scopes.scope_tag(S) for c in begins)
    assert all(c[7] == b"" for c in begins)  # no pos: the daemon's default (key) groups them
    # the load job carries the tenant into begin_load and record_hits
    fc2 = FakeClient()
    s = _sched(fc2)
    keys = _keys(12, S)
    _store(fc2, 1, keys[:2])
    _store(fc2, 0, [keys[1]])
    assert s.match("q", list(range(12)), params, 0) == (8, True)
    s.after_alloc("q", [[None, 1, None], [2, 3, 4]], 8)
    (job,) = s.build([], lambda rid: None).loads
    assert job.tenant == T
    fc2.offsets_override = AnyGroup({"*": 0})
    got, _c, _s = _run_load(fc2, job)
    assert got == [0, 0]
    assert {c[3] for c in fc2.ops("begin_load")} == {T}
    cl.record_load_hits(fc2, job, got, 0.5, BS, rank=0)
    assert fc2.ops("record_hits") == [("record_hits", "q", 8, 0, T)]
    cl.record_load_hits(fc2, job, got, 0.5, BS, rank=1)  # only rank 0 records
    assert len(fc2.ops("record_hits")) == 1


@pytest.mark.parametrize("cache_salt", [None, "client-salt"])
def test_every_isolated_save_carries_the_scope_tag_of_its_raw_scope(cache_salt: Any) -> None:
    fc = _saved(_plan({"lmw_scope": S.hex()}, cache_salt))
    assert {c[6] for c in fc.ops("begin_save")} == {scopes.scope_tag(S)}


def test_a_sharing_request_stays_untagged_and_unattributed() -> None:
    """Step 1 on the wire: no tenant, no scope tag, the scope bytes as before."""
    fc = _saved(_plan({}))
    assert {c[4:] for c in fc.ops("begin_save")} == {(b"", b"", b"", b"")}
    fc = _saved(_plan({}, cache_salt="client"))  # a folded client salt is legitimately untagged
    salt = cl.fold_cache_salt(b"", "client")
    assert {c[4:] for c in fc.ops("begin_save")} == {(salt, b"", b"", b"")}


def test_raw_scope_never_reaches_the_daemon_given_the_proxy_gpu_salt() -> None:
    """Pinned by ruling R7. The connector folds the raw ``lmw_scope`` with the request's
    ``cache_salt``; the raw scope stays out of the daemon ONLY because the proxy gives every
    isolated request its derived GPU salt. That half is proven in
    tests/unit/proxy/test_kv_isolation.py
    (``test_no_isolated_request_reaches_the_engine_without_its_salt``). Do not rename one
    without the other."""
    gpu_salt = scopes.gpu_cache_salt(NK, S, "")  # what the proxy derives for this key
    fc = _saved(_plan({"lmw_scope": S.hex(), "lmw_tenant": T.hex()}, gpu_salt))
    folded = cl.fold_cache_salt(S, gpu_salt)
    for c in fc.ops("begin_save"):
        assert S not in c[1:]
        assert c[4] == folded  # the stored scope field is the folded salt
        assert c[6] == scopes.scope_tag(S)  # the daemon gets the public handle, not S


def test_block_scope_ties_the_tag_to_the_salt_and_redacts_itself() -> None:
    import pickle

    b = cl.BlockScope(S, "x")
    assert b.salt == cl.fold_cache_salt(S, "x") and b.tag == scopes.scope_tag(S)
    assert S.hex() not in repr(b) and b.salt.hex() not in repr(b) and b.tag.hex() not in repr(b)
    assert pickle.loads(pickle.dumps(b)) == b  # crosses to the workers in the step metadata
    with pytest.raises(ValueError):
        cl.BlockScope(S, b"not a str")
    with pytest.raises(ValueError):
        cl.BlockScope(S, "\ud800")  # a lone surrogate cannot be folded
    e = cl.SaveEntry(b"k" * 32, b"p" * 32, b, 1, 2, T)
    assert e.scope == b.salt and e.scope_tag == b.tag and e.tenant == T
    assert T.hex() not in repr(e) and S.hex() not in repr(e)
    job = cl.LoadJob("r", "t", 4, [], tenant=T)
    assert T.hex() not in repr(job)


def test_malformed_tenant_is_not_cached(caplog: Any) -> None:
    fc = FakeClient()
    s = _sched(fc)
    with caplog.at_level(logging.WARNING, logger="lmw_kv.connector"):
        for rid in ("a", "b"):
            assert s.match(rid, list(range(40)), {"lmw_tenant": "zz" * 16}, 0) == (0, False)
            assert s.build([(rid, 0, 40)], lambda r: [[1] * 10, [2] * 10]).saves == []
    assert fc.calls == []
    lines = [r.getMessage() for r in caplog.records if "lmw_tenant" in r.getMessage()]
    assert len(lines) == 1 and "zz" not in lines[0]  # once, without the value


def test_a_bad_cache_salt_turns_the_tier_off_for_that_request_only(caplog: Any) -> None:
    fc = FakeClient()
    s = _sched(fc)
    params = {"lmw_scope": S.hex(), "lmw_tenant": T.hex()}
    with caplog.at_level(logging.WARNING, logger="lmw_kv.connector"):
        assert s.match("bad", list(range(13)), params, 0, cache_salt="\ud800") == (0, False)
    assert s.build([("bad", 0, 12)], lambda r: [[None, None, 10], [20, 21, 22]]).saves == []
    assert not any(S.hex() in r.getMessage() for r in caplog.records)
    assert len(_plan(params, "ok").saves) == 4  # the next request is cached as usual


def test_a_bad_cache_salt_warns_once_per_request(caplog: Any) -> None:
    fc = FakeClient()
    s = _sched(fc)
    params = {"lmw_scope": S.hex(), "lmw_tenant": T.hex()}
    with caplog.at_level(logging.WARNING, logger="lmw_kv.connector"):
        for _ in range(3):  # the scheduler may ask again while the request waits
            assert s.match("bad", list(range(13)), params, 0, cache_salt=7) == (0, False)
        assert s.match("bad2", list(range(13)), params, 0, cache_salt=7) == (0, False)
        s.finished("bad")
        s.finished("bad2")
    lines = [r.getMessage() for r in caplog.records if "cache_salt" in r.getMessage()]
    assert len(lines) == 2


def test_missing_tenant_is_shared() -> None:
    for params in ({}, {"lmw_tenant": ""}, {"lmw_tenant": None}):
        plan = _plan(params)
        assert plan.saves and all(e.tenant == b"" for e in plan.saves)


class _StoreClient:
    """The in-process store behind the client calls the connector makes (no daemon)."""

    def __init__(self, store: Store) -> None:
        self.s = store

    def register_ns(
        self, ns: Namespace, groups: dict[str, int], nvme: bool = True, **kw: Any
    ) -> bytes:
        return self.s.register_namespace(ns, groups, nvme=nvme, **kw)

    def begin_save(
        self, ns_id: bytes, key: bytes, parent: bytes, scope: bytes = b"", **kw: Any
    ) -> dict[str, int] | None:
        return self.s.begin_save(ns_id, key, parent, scope, **kw)

    def commit_save(self, ns_id: bytes, key: bytes) -> bool:
        self.s.commit_save(ns_id, key)
        return True

    def abort_save(self, ns_id: bytes, key: bytes) -> bool:
        self.s.abort_save(ns_id, key)
        return True


def test_groups_and_ranks_form_one_position(
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
) -> None:
    ns2 = _model_ns(tp=2)
    # the scheduler keys every block with the group-independent model namespace
    s = cl.SchedulerLogic(FakeClient(), SECRET, CFG, namespace_id(ns2), GROUPS, STORE_IDS, BS)
    params = {"lmw_tenant": T.hex()}
    s.match("r", list(range(13)), params, 0)
    plan = s.build([("r", 0, 12)], lambda rid: [[None, None, 10], [20, 21, 22]])
    keys = cl.block_keys_for(SECRET, b"", namespace_id(ns2), list(range(13)), BS)
    store = Store(_store_cfg(tmp_path, slots=8, nvme=False))
    opened.append(store)
    c = _StoreClient(store)
    want: dict[tuple[bytes, bytes], set[tuple[bytes, bytes]]] = {}
    for rank in (0, 1):  # each TP rank is its own worker process
        names = {g.gid: cl.store_group_name(g, rank) for g in GROUPS}
        nss = [(g.gid, cl.store_namespace(ns2, g, rank), {names[g.gid]: 1000}) for g in GROUPS]
        ids = cl.attach_store(
            c,
            nss,
            lambda: None,
            lambda: True,
            nvme=False,
            owner="m1",
            model=M,
            rank=rank,
            groups=GROUPS,
        )
        n = cl.run_saves(
            c,
            plan.saves,
            ids,
            names,
            SIZES,
            store.cfg.ram_bytes,
            lambda g, b, o: None,
            lambda: None,
            lambda: True,
        )
        assert n == 4
        for e in plan.saves:
            want.setdefault((M, e.key), set()).add((ids[e.gid], e.key))
    assert store._positions["ram"] == want
    assert [len(want[(M, k)]) for k in keys[:3]] == [2, 2, 4]  # the state rides with block 2
