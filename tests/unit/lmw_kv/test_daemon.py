from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import mmap
import os
import shutil
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lmw_kv import protocol
from lmw_kv.client import Client
from lmw_kv.daemon import NVME_LOCK_NAME, Daemon, store_config
from lmw_kv.keys import Namespace
from lmw_kv.store import Store, StoreConfig

PAGE = mmap.PAGESIZE
GROUPS = {"full": 1000, "gdn": 300}
ROOT = b"\x00" * 32
NS = Namespace(
    model_repo="org/model",
    revision_sha="r1",
    tokenizer_sha="tok",
    kv_dtype="fp8",
    block_size=16,
    attention_layout="hnd",
    kv_groups=("full", "gdn"),
    tp=1,
    pp=1,
    engine_family="test",
    layout_version=1,
)


#: The slow-call deadline of the test clients (register_ns, begin_load): generous, the
#: tests check behaviour, never speed.
SLOW_S = 30.0


def _k(name: str) -> bytes:
    return hashlib.sha256(name.encode()).digest()


class Running:
    def __init__(
        self,
        store: Store,
        daemon: Daemon,
        sock: str,
        ram_path: Path,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.store, self.daemon, self.sock, self.ram_path = store, daemon, sock, ram_path
        self.loop = loop

    def client(self, timeout_s: float = 1.0) -> Client:
        # register_ns waits for the NVMe write queue to drain (Store.flush): on a loaded
        # build host that took longer than the 2 s default, so the slow deadline is long.
        return Client(self.sock, timeout_s=timeout_s, slow_timeout_s=SLOW_S)


@pytest.fixture
def running(tmp_path: Path) -> Iterator[Running]:
    sockdir = tempfile.mkdtemp(dir="/tmp")  # AF_UNIX paths are short on macOS
    sock = os.path.join(sockdir, "kv.sock")
    cfg = StoreConfig(
        ram_path=tmp_path / "ram.pool",
        ram_bytes=8 * 2 * PAGE,
        nvme_dir=tmp_path / "nvme",
        nvme_cap_bytes=1 << 30,
    )
    store = Store(cfg)
    daemon = Daemon(store, sock)
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(daemon.start())
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert ready.wait(5)
    try:
        yield Running(store, daemon, sock, cfg.ram_path, loop)
    finally:
        asyncio.run_coroutine_threadsafe(daemon.stop(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()
        store.close()
        shutil.rmtree(sockdir, ignore_errors=True)


def test_protocol_has_no_engine_specific_concepts() -> None:
    names = {*protocol.OPS, *protocol.REPLY_FIELDS, *protocol.ENVELOPE_FIELDS}
    names.update(f for fields in protocol.OPS.values() for f in fields)
    names.update(NS.__dataclass_fields__)
    for name in names:
        assert not any(bad in name for bad in ("vllm", "request", "scheduler", "worker")), name
    assert "tag" in protocol.OPS["record_hits"]
    assert "tag" in protocol.OPS["pop_hits"]


def test_hello_reports_protocol_and_ram_file(running: Running) -> None:
    r = running.client().hello()
    assert r is not None
    assert r["protocol"] == 1
    assert r["ram_path"] == str(running.ram_path)
    assert isinstance(r["ram_bytes"], int)


def test_ram_file_exists_before_the_first_register_ns(running: Running) -> None:
    """The daemon creates its pool at start; register_ns only carves it into slots.

    The connector still registers, maps and registers again (connector._setup_worker);
    register_ns is idempotent.
    """
    c = running.client()
    assert c.hello() is not None
    assert running.ram_path.exists()
    ino = running.ram_path.stat().st_ino
    assert c.register_ns(NS, GROUPS) is not None
    assert c.register_ns(NS, GROUPS) is not None
    assert running.ram_path.stat().st_ino == ino


def test_socket_is_private(running: Running) -> None:
    assert stat.S_IMODE(os.stat(running.sock).st_mode) == 0o600


def test_save_lookup_load_cycle(running: Running) -> None:
    c = running.client()
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None and len(ns) == 32
    k1, k2 = _k("a"), _k("b")
    assert c.lookup(ns, [k1, k2]) == 0

    offsets = c.begin_save(ns, k1, ROOT, b"")
    assert offsets is not None and set(offsets) == set(GROUPS)
    with open(running.ram_path, "r+b") as f:
        for g, off in offsets.items():
            f.seek(off)
            f.write(bytes([len(g)]) * GROUPS[g])
    assert c.lookup(ns, [k1]) == 0  # not visible until committed
    assert c.commit_save(ns, k1) is True
    assert c.lookup(ns, [k1, k2]) == 1

    loaded = c.begin_load(ns, [k1, k2])
    assert loaded is not None
    handle, slots, nvme = loaded
    assert len(slots) == 1 and slots[0] == offsets
    assert nvme == 0  # served from RAM
    with open(running.ram_path, "rb") as f:
        for g, off in slots[0].items():
            f.seek(off)
            assert f.read(GROUPS[g]) == bytes([len(g)]) * GROUPS[g]
    assert c.end_load(handle) is True

    assert c.record_hits("t1", 100, 0, 0.5) is True
    assert c.record_hits("t1", 0, 50, 0.25) is True
    assert c.pop_hits("t1") == {"ram_tokens": 100, "nvme_tokens": 50, "load_s": 0.75}
    # R23: the daemon answered and nothing is recorded (popped already, or never) -> {}.
    assert c.pop_hits("t1") == {}
    assert c.pop_hits("never") == {}
    st = c.stats()
    assert st is not None and st["blocks"] >= 1


def test_abort_save_frees_the_key(running: Running) -> None:
    c = running.client()
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("a"), ROOT) is not None
    assert c.begin_save(ns, _k("a"), ROOT) is None  # already being saved
    assert c.abort_save(ns, _k("a")) is True
    assert c.begin_save(ns, _k("a"), ROOT) is not None


def test_dropped_connection_aborts_saves_and_ends_loads(running: Running) -> None:
    keep = running.client()
    ns = keep.register_ns(NS, GROUPS)
    assert ns is not None
    assert keep.begin_save(ns, _k("done"), ROOT) is not None
    assert keep.commit_save(ns, _k("done")) is True

    crashed = running.client()
    assert crashed.begin_save(ns, _k("half"), _k("done")) is not None  # never committed
    loaded = crashed.begin_load(ns, [_k("done")])
    assert loaded is not None and len(loaded[1]) == 1  # never ended
    assert running.store._loads

    crashed.close()
    assert running.daemon.released.acquire(timeout=5)

    assert not running.store._loads  # the pin is gone
    assert keep.begin_save(ns, _k("half"), _k("done")) is not None  # the slot came back
    assert keep.lookup(ns, [_k("done")]) == 1


def test_committed_saves_survive_a_dropped_connection(running: Running) -> None:
    a = running.client()
    ns = a.register_ns(NS, GROUPS)
    assert ns is not None
    assert a.begin_save(ns, _k("a"), ROOT) is not None
    assert a.commit_save(ns, _k("a")) is True
    a.close()
    assert running.daemon.released.acquire(timeout=5)
    assert running.client().lookup(ns, [_k("a")]) == 1


def test_errors_keep_the_connection_open(running: Running) -> None:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(5)
    s.connect(running.sock)
    f = s.makefile("rwb")

    def ask(raw: bytes) -> dict[str, object]:
        f.write(raw + b"\n")
        f.flush()
        return json.loads(f.readline())

    r = ask(b'{"op":"nope","id":1}')
    assert r["id"] == 1 and r["ok"] is False and str(r["error"]).startswith("unknown_op")
    r = ask(b"not json")
    assert r["ok"] is False and str(r["error"]).startswith("bad_request")
    r = ask(b'{"op":"lookup","id":3,"ns":"zz","keys":[]}')
    assert r["id"] == 3 and r["ok"] is False and str(r["error"]).startswith("bad_request")
    r = ask(b'{"op":"register_ns","id":4,"namespace":{"model_repo":"x"},"groups":{"g":1}}')
    assert r["ok"] is False and str(r["error"]).startswith("bad_request")
    r = ask(b'{"op":"hello","id":5}')
    assert r["ok"] is True and r["protocol"] == 1
    s.close()


def test_client_survives_a_refused_request(running: Running) -> None:
    c = running.client()
    assert c.register_ns(NS, {"full": 0}) is None  # the store rejects it: ok false
    assert c.hello() is not None  # same connection, still fine


def test_slow_store_call_does_not_stall_other_connections(running: Running) -> None:
    c1, c2 = running.client(timeout_s=5), running.client()
    ns = c1.register_ns(NS, GROUPS)
    assert ns is not None
    entered, release = threading.Event(), threading.Event()
    real = running.store.begin_load

    def slow(ns_id: bytes, keys: list[bytes]):  # type: ignore[no-untyped-def]
        entered.set()
        release.wait(10)
        return real(ns_id, keys)

    running.store.begin_load = slow  # type: ignore[method-assign]
    out: list[object] = []
    t = threading.Thread(target=lambda: out.append(c1.begin_load(ns, [_k("a")])))
    t.start()
    try:
        assert entered.wait(5)
        assert c2.hello() is not None  # answered while c1's call is stuck
    finally:
        release.set()
        t.join(5)
    assert out == [(out[0][0], [], 0)]  # type: ignore[index]


def test_client_fails_open_when_daemon_is_down(tmp_path: Path) -> None:
    c = Client(str(tmp_path / "nothing.sock"), timeout_s=0.05)
    t0 = time.monotonic()
    assert c.lookup(b"\x01" * 32, [_k("a")]) == 0
    assert c.hello() is None
    assert c.begin_save(b"\x01" * 32, _k("a"), ROOT) is None
    assert c.begin_load(b"\x01" * 32, [_k("a")]) is None
    assert c.commit_save(b"\x01" * 32, _k("a")) is None
    assert c.pop_hits("t") is None
    assert time.monotonic() - t0 < 0.5


def test_client_fails_open_on_a_hung_daemon() -> None:
    sockdir = tempfile.mkdtemp(dir="/tmp")
    path = os.path.join(sockdir, "h.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(8)  # accepts into the backlog, never reads or answers
    try:
        c = Client(path, timeout_s=0.1)
        t0 = time.monotonic()
        assert c.lookup(b"\x01" * 32, [_k("a")]) == 0
        took = time.monotonic() - t0
        assert 0.08 <= took < 0.5
        t0 = time.monotonic()  # while backing off, further calls return at once
        assert c.lookup(b"\x01" * 32, [_k("a")]) == 0
        assert time.monotonic() - t0 < 0.05
    finally:
        srv.close()
        shutil.rmtree(sockdir, ignore_errors=True)


def test_client_recovers_after_the_daemon_restarts(running: Running) -> None:
    c = running.client()
    assert c.hello() is not None
    c._conn().close()  # as after a broken connection
    assert c.hello() is not None


def test_client_logs_failures_at_most_once_a_minute(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    c = Client(str(tmp_path / "nothing.sock"), timeout_s=0.05)
    with caplog.at_level("WARNING", logger="lmw_kv.client"):
        for _ in range(5):
            c._conn().retry_at = 0.0
            c.hello()
    assert len(caplog.records) == 1


def test_threads_do_not_block_each_other(running: Running) -> None:
    c = Client(running.sock, timeout_s=1.0, slow_timeout_s=10.0)
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("a"), ROOT) is not None
    assert c.commit_save(ns, _k("a")) is True
    loaded = c.begin_load(ns, [_k("a")])
    assert loaded is not None
    handle = loaded[0]

    entered, release = threading.Event(), threading.Event()
    real = running.store.begin_load

    def slow(ns_id: bytes, keys: list[bytes]):  # type: ignore[no-untyped-def]
        entered.set()
        release.wait(10)
        return real(ns_id, keys)

    running.store.begin_load = slow  # type: ignore[method-assign]
    t = threading.Thread(target=lambda: c.begin_load(ns, [_k("a")]))
    t.start()
    try:
        assert entered.wait(5)
        t0 = time.monotonic()
        assert c.lookup(ns, [_k("a")]) == 1
        assert c.end_load(handle) is True
        assert time.monotonic() - t0 < 0.5
    finally:
        release.set()
        t.join(5)
    assert not running.store._loads or handle not in running.store._loads


def test_stop_with_a_connected_client_returns(running: Running) -> None:
    c = running.client()
    assert c.hello() is not None  # connected and now idle
    # the fixture calls stop() again afterwards, so stopping twice must be safe too
    fut = asyncio.run_coroutine_threadsafe(running.daemon.stop(), running.loop)
    fut.result(2)
    assert c.hello() is None  # the daemon is gone


def test_protocol_doc_known_answer() -> None:
    import re

    from lmw_kv.keys import block_key, namespace_id, root_key

    doc = (Path(__file__).parents[3] / "lmw_kv" / "PROTOCOL.md").read_text()
    got = dict(re.findall(r"^(ns|secret|root|block0|block1): +([0-9a-f]{64})$", doc, re.M))
    assert set(got) == {"ns", "secret", "root", "block0", "block1"}
    secret = bytes(range(32))
    assert got["secret"] == secret.hex()
    ns = namespace_id(NS)
    assert got["ns"] == ns.hex()
    root = root_key(secret, b"", ns)
    assert got["root"] == root.hex()
    b0 = block_key(secret, b"", ns, root, list(range(16)))
    assert got["block0"] == b0.hex()
    assert got["block1"] == block_key(secret, b"", ns, b0, list(range(16, 32))).hex()


def _fail_on_a_thread(c: Client) -> None:
    """Thread A: break its own connection and make a call that fails (and backs off)."""

    def run() -> None:
        assert c.hello() is not None
        c._conn().sock.close()  # type: ignore[union-attr]
        assert c.hello() is None

    t = threading.Thread(target=run)
    t.start()
    t.join(5)


def test_one_threads_failure_does_not_skip_another_threads_release(running: Running) -> None:
    c = Client(running.sock, timeout_s=1.0, slow_timeout_s=SLOW_S)
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("a"), ROOT) is not None
    _fail_on_a_thread(c)
    assert c.commit_save(ns, _k("a")) is True  # sent at once, not skipped
    assert running.store.lookup(ns, [_k("a")]) == 1


def test_one_threads_failure_does_not_skip_end_load(running: Running) -> None:
    c = Client(running.sock, timeout_s=1.0, slow_timeout_s=SLOW_S)
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("a"), ROOT) is not None
    assert c.commit_save(ns, _k("a")) is True
    loaded = c.begin_load(ns, [_k("a")])
    assert loaded is not None and running.store._loads

    _fail_on_a_thread(c)
    assert c.end_load(loaded[0]) is True  # B is sent at once
    assert not running.store._loads


def test_exited_thread_socket_is_closed(running: Running) -> None:
    import gc

    # Generous timeouts: on a loaded test host a 1 s client timeout made the
    # worker's begin_load give up before the daemon answered.
    c = Client(running.sock, timeout_s=10.0, slow_timeout_s=SLOW_S)
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    assert c.begin_save(ns, _k("a"), ROOT) is not None
    assert c.commit_save(ns, _k("a")) is True
    c.close()  # forget the main thread's connection
    assert running.daemon.released.acquire(timeout=10)

    loaded: list[object] = []

    def worker() -> None:
        loaded.append(c.begin_load(ns, [_k("a")]))  # never ended
        # Checked while the thread lives: once it ends its socket may close (and the
        # daemon end the load) before the main thread looks.
        loaded.append(bool(running.store._loads))

    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    assert not t.is_alive()
    assert loaded and loaded[0] is not None
    assert loaded[1] is True  # the load was pinned
    # The thread's socket closes when its thread-local state is collected;
    # collect repeatedly rather than relying on one pass under load.
    deadline = time.monotonic() + 10
    released = False
    while not released and time.monotonic() < deadline:
        gc.collect()
        released = running.daemon.released.acquire(timeout=0.5)
    assert released
    assert not running.store._loads


def test_begin_load_reports_blocks_promoted_from_nvme(running: Running) -> None:
    from lmw_kv.store import LoadHandle

    c = running.client()
    ns = c.register_ns(NS, GROUPS)
    assert ns is not None
    real = running.store.begin_load

    def promoted(ns_id: bytes, keys: list[bytes]) -> LoadHandle:
        h = real(ns_id, keys)
        return LoadHandle(h.id, [{"full": 0, "gdn": PAGE}] * 3, nvme_blocks=2)

    running.store.begin_load = promoted  # type: ignore[method-assign]
    loaded = c.begin_load(ns, [_k("a"), _k("b"), _k("c")])
    assert loaded is not None and len(loaded[1]) == 3 and loaded[2] == 2


def _args(tmp_path: Path, nvme: Path) -> argparse.Namespace:
    return argparse.Namespace(
        ram_path=str(tmp_path / "ram.pool"), ram_gib=0.01, nvme_dir=str(nvme), nvme_cap_gb=1.0
    )


def test_a_second_daemon_on_the_same_nvme_dir_runs_ram_only(tmp_path: Path, caplog: Any) -> None:
    """R25: during a rolling update two api pods share the NVMe volume; one writer only."""
    nvme = tmp_path / "nvme"
    first, fd1 = store_config(_args(tmp_path, nvme))
    assert first.nvme_dir == nvme and fd1 is not None
    with caplog.at_level(logging.WARNING, logger="lmw_kv.daemon"):
        second, fd2 = store_config(_args(tmp_path, nvme))
    assert second.nvme_dir is None and fd2 is None
    warns = [r.getMessage() for r in caplog.records if "RAM-only" in r.getMessage()]
    assert len(warns) == 1 and "locked by another KV store daemon" in warns[0]
    store = Store(second)
    try:
        assert store.stats()["nvme_enabled"] is False
    finally:
        store.close()
    os.close(fd1)  # the first daemon exits: the lock is free again
    third, fd3 = store_config(_args(tmp_path, nvme))
    assert third.nvme_dir == nvme and fd3 is not None
    os.close(fd3)


def test_the_lock_file_is_not_taken_for_a_namespace(tmp_path: Path) -> None:
    nvme = tmp_path / "nvme"
    cfg, fd = store_config(_args(tmp_path, nvme))
    assert (nvme / NVME_LOCK_NAME).exists()
    store = Store(cfg)
    try:
        assert store.stats()["nvme_enabled"] is True
    finally:
        store.close()
        if fd is not None:
            os.close(fd)


def test_run_closes_the_store_and_frees_the_lock_when_it_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1: an exception in the daemon's main coroutine still closes the store."""
    from lmw_kv import daemon as daemon_mod

    closed: list[bool] = []
    real_close = Store.close

    def close(self: Store) -> None:
        closed.append(True)
        real_close(self)

    async def boom(self: Daemon) -> None:
        raise RuntimeError("socket bind failed")

    monkeypatch.setattr(Store, "close", close)
    monkeypatch.setattr(Daemon, "start", boom)
    nvme = tmp_path / "nvme"
    args = _args(tmp_path, nvme)
    args.socket = str(tmp_path / "kv.sock")
    with pytest.raises(RuntimeError, match="socket bind failed"):
        asyncio.run(daemon_mod._run(args))
    assert closed == [True]
    cfg, fd = store_config(_args(tmp_path, nvme))  # the NVMe lock was released
    assert fd is not None
    os.close(fd)


def test_begin_load_deadline_grows_with_the_key_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """I2: a long NVMe restore must not time out at a fixed 2 s."""
    from lmw_kv import client as client_mod

    seen: list[float | None] = []

    def call(self: Client, req: dict[str, Any], timeout_s: float | None = None) -> None:
        seen.append(timeout_s)

    monkeypatch.setattr(Client, "_call", call)
    c = Client("/nonexistent.sock")
    ns = b"\x00" * 32
    c.begin_load(ns, [_k("a")])
    c.begin_load(ns, [_k(str(i)) for i in range(200)])
    c.begin_load(ns, [_k(str(i)) for i in range(100_000)])
    one, many, huge = seen
    assert one == pytest.approx(2.0 + client_mod.LOAD_PER_KEY_S)
    assert many == pytest.approx(2.0 + 200 * client_mod.LOAD_PER_KEY_S)
    assert many > 10.0  # 200 NVMe blocks get real time, not 2 s
    assert huge == client_mod.LOAD_MAX_S


def test_begin_load_waits_past_the_base_deadline_for_a_slow_promotion(
    running: Running,
) -> None:
    # Registered by another client: with a 0.1 s slow deadline register_ns itself timed
    # out on a loaded host. The per-key allowance is large so the load's deadline
    # (0.1 + 2 * 15 s) is far past the 0.3 s it takes, however loaded the host is; the
    # 0.1 s base alone would still fail it.
    ns = running.client().register_ns(NS, GROUPS)
    assert ns is not None
    c = Client(running.sock, timeout_s=1.0, slow_timeout_s=0.1, load_per_key_s=SLOW_S / 2)
    real = running.store.begin_load

    def slow(ns_id: bytes, keys: list[bytes]) -> Any:
        time.sleep(0.3)  # longer than the 0.1 s base
        return real(ns_id, keys)

    running.store.begin_load = slow  # type: ignore[method-assign]
    assert c.begin_load(ns, [_k("a"), _k("b")]) is not None


def test_register_ns_carries_the_nvme_flag(running: Running) -> None:
    """I3: a namespace registered with nvme false is never written to NVMe."""
    c = running.client()
    ns = c.register_ns(NS, GROUPS, nvme=False)
    assert ns is not None
    assert running.store._ns[ns].nvme is False
    assert c.register_ns(NS, GROUPS) == ns  # the default is true
    assert running.store._ns[ns].nvme is True


def test_register_ns_refuses_a_non_boolean_nvme_flag(running: Running) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(running.sock)
        req = {
            "op": "register_ns",
            "id": 1,
            "namespace": {**NS.__dict__, "kv_groups": list(NS.kv_groups)},
            "groups": GROUPS,
            "nvme": "yes",
        }
        s.sendall(json.dumps(req).encode() + b"\n")
        reply = json.loads(s.makefile().readline())
    assert reply["ok"] is False and reply["error"].startswith("bad_request")


def test_a_ram_only_daemon_turns_nvme_on_once_it_gets_the_lock(tmp_path: Path) -> None:
    """M4: the second daemon of a rolling update retries the lock and then uses NVMe."""
    from lmw_kv.daemon import try_take_nvme

    nvme = tmp_path / "nvme"
    first_cfg, fd1 = store_config(_args(tmp_path, nvme))
    assert fd1 is not None
    first_cfg.ram_path = tmp_path / "first.pool"
    first = Store(first_cfg)
    ns = first.register_namespace(NS, GROUPS)
    off = first.begin_save(ns, _k("old"), ROOT, b"")
    assert off is not None
    first.commit_save(ns, _k("old"))
    first.flush()

    second_cfg, fd2 = store_config(_args(tmp_path, nvme))
    assert fd2 is None and second_cfg.nvme_dir is None
    second = Store(second_cfg)
    try:
        assert second.register_namespace(NS, GROUPS) == ns
        assert try_take_nvme(second, nvme) is None  # still held by the first daemon
        assert second.stats()["nvme_enabled"] is False
        assert second.lookup(ns, [_k("old")]) == 0
        first.close()
        os.close(fd1)  # the first daemon exits
        fd = try_take_nvme(second, nvme)
        assert fd is not None
        assert second.stats()["nvme_enabled"] is True
        assert second.lookup(ns, [_k("old")]) == 1  # the first daemon's blocks are visible
        assert second.begin_save(ns, _k("new"), ROOT, b"") is not None
        second.commit_save(ns, _k("new"))
        second.flush()
        on_disk = sum(f.stat().st_size for f in nvme.rglob("*.seg"))
        assert second.stats()["nvme_used"] == on_disk > 0  # written through
        os.close(fd)
    finally:
        second.close()


def test_the_daemon_retries_the_nvme_lock_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4: the retry runs inside the daemon's main coroutine, not only in a helper."""
    from lmw_kv import daemon as daemon_mod

    nvme = tmp_path / "nvme"
    _, fd1 = store_config(_args(tmp_path, nvme))
    assert fd1 is not None
    monkeypatch.setattr(daemon_mod, "NVME_LOCK_RETRY_S", 0.05)
    sockdir = tempfile.mkdtemp(dir="/tmp")
    args = _args(tmp_path, nvme)
    args.socket = os.path.join(sockdir, "kv.sock")
    seen: list[Store] = []
    real_init = Store.__init__

    def init(self: Store, cfg: StoreConfig) -> None:
        real_init(self, cfg)
        seen.append(self)

    monkeypatch.setattr(Store, "__init__", init)

    async def main() -> bool:
        task = asyncio.create_task(daemon_mod._run(args))
        for _ in range(100):
            await asyncio.sleep(0.02)
            if seen:
                break
        assert seen and seen[0].stats()["nvme_enabled"] is False
        os.close(fd1)  # the other daemon exits
        for _ in range(200):
            await asyncio.sleep(0.02)
            if seen[0].stats()["nvme_enabled"]:
                break
        on = bool(seen[0].stats()["nvme_enabled"])
        os.kill(os.getpid(), signal.SIGTERM)  # the daemon's own stop handler
        await asyncio.wait_for(task, 5)
        return on

    try:
        assert asyncio.run(main()) is True
    finally:
        shutil.rmtree(sockdir, ignore_errors=True)
    _, fd = store_config(_args(tmp_path, nvme))  # the late lock was released at exit
    assert fd is not None
    os.close(fd)


def test_cancelling_the_nvme_retry_mid_take_keeps_the_lock_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop that cancels the lock retry while a take runs must not leak the lock fd."""
    from lmw_kv import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "NVME_LOCK_RETRY_S", 0.0)
    entered, release = threading.Event(), threading.Event()
    fd = os.open(tmp_path / "lock", os.O_RDWR | os.O_CREAT, 0o600)

    def take(store: Store, nvme_dir: Path) -> int:
        entered.set()
        release.wait(30)
        return fd  # the take won the lock after all

    monkeypatch.setattr(daemon_mod, "try_take_nvme", take)
    fds: list[int] = []

    async def main() -> bool:
        task = asyncio.create_task(daemon_mod._retry_nvme(None, tmp_path, fds))  # type: ignore[arg-type]
        assert await asyncio.to_thread(entered.wait, 30)
        task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        return task.cancelled()

    try:
        assert asyncio.run(main()) is True  # the cancel still goes through
        assert fds == [fd]  # the caller closes it at exit
    finally:
        os.close(fd)


def test_hello_answers_while_every_worker_thread_is_busy(running: Running) -> None:
    """I5: the api's liveness probe must not read a busy daemon as a wedged one."""
    import concurrent.futures

    gate = threading.Event()
    real = running.store.lookup

    def stuck(ns_id: bytes, keys: list[bytes]) -> int:
        gate.wait(10)
        return real(ns_id, keys)

    running.store.lookup = stuck  # type: ignore[method-assign]
    pool = concurrent.futures.ThreadPoolExecutor(64)
    try:
        c = Client(running.sock, timeout_s=15.0)
        busy = [pool.submit(c.lookup, b"\x00" * 32, [_k("a")]) for _ in range(48)]
        time.sleep(0.3)  # the daemon's default executor is now full of stuck lookups
        t0 = time.monotonic()
        assert Client(running.sock, timeout_s=2.0).hello() is not None
        assert time.monotonic() - t0 < 1.0
    finally:
        gate.set()
        for f in busy:
            f.result(15)
        pool.shutdown()


# ---- F4: parent-death signal ---------------------------------------------------


def test_die_with_parent_is_a_no_op_without_a_parent_pid_or_off_linux(monkeypatch):
    from lmw_kv import daemon as d

    assert d.die_with_parent(None) is False
    monkeypatch.setattr(d.sys, "platform", "darwin")
    assert d.die_with_parent(os.getppid()) is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux only")
def test_daemon_gets_sigterm_when_the_api_that_started_it_dies(tmp_path):
    """F4: an api killed with SIGKILL (the OOM killer) must not leave its daemon running."""
    import subprocess

    child_code = (
        "import os, time\n"
        "from lmw_kv.daemon import die_with_parent\n"
        "assert die_with_parent(os.getppid())\n"
        "print('armed', flush=True)\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import subprocess, sys\n"
        f"c = subprocess.Popen([sys.executable, '-c', {child_code!r}], stdout=subprocess.PIPE)\n"
        "assert c.stdout.readline().strip() == b'armed'\n"
        "print(c.pid, flush=True)\n"
        "import time; time.sleep(60)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, env=env)
    try:
        child = int(parent.stdout.readline())
        parent.kill()  # SIGKILL: no cleanup runs in the parent
        parent.wait()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            proc_stat = Path(f"/proc/{child}/stat").read_text()
            if proc_stat.split(") ", 1)[1].startswith("Z"):
                break  # dead, waiting for init to reap it
            time.sleep(0.05)
        else:
            os.kill(child, signal.SIGKILL)
            raise AssertionError("the daemon outlived its parent")
    finally:
        if parent.poll() is None:
            parent.kill()


# ---- the pool at start; register_ns fast and retried (production gate, window 2) ----


def _run_args(tmp_path: Path, sockdir: str, ram_gib: float = 0.01) -> argparse.Namespace:
    return argparse.Namespace(
        socket=os.path.join(sockdir, "kv.sock"),
        ram_path=str(tmp_path / "ram.pool"),
        ram_gib=ram_gib,
        nvme_dir=None,
        nvme_cap_gb=0.0,
    )


def _wait_hello(sock: str, timeout_s: float) -> float | None:
    """Seconds until the daemon first answers ``hello``, or None."""
    c = Client(sock, timeout_s=0.2)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        c._conn().retry_at = 0.0
        if c.hello() is not None:
            return time.monotonic() - t0
        time.sleep(0.02)
    return None


def _with_daemon(args: argparse.Namespace, body: Any) -> Any:
    """Run the real daemon main coroutine on this (main) thread while ``body`` runs in
    a worker thread; stop the daemon when ``body`` returns."""
    from lmw_kv import daemon as daemon_mod

    async def main() -> Any:
        task = asyncio.create_task(daemon_mod._run(args))
        try:
            return await asyncio.to_thread(body)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return asyncio.run(main())


def test_the_daemon_answers_only_after_its_pool_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Healthy (``hello`` answers) means the pool already exists and is committed.

    The api waits for a healthy daemon before it loads a model, so the allocation is
    over before the engine reads its weights or registers.
    """
    from lmw_kv import store as store_mod

    gate = threading.Event()
    entered = threading.Event()
    real = store_mod.create_pool_file

    def slow(path: Path, size: int) -> None:
        entered.set()
        assert gate.wait(10)
        real(path, size)

    monkeypatch.setattr(store_mod, "create_pool_file", slow)
    sockdir = tempfile.mkdtemp(dir="/tmp")
    args = _run_args(tmp_path, sockdir)

    def body() -> None:
        try:
            assert entered.wait(5)
            assert _wait_hello(args.socket, 0.5) is None  # still allocating: not healthy
            assert not Path(args.ram_path).exists()
            gate.set()
            assert _wait_hello(args.socket, 5) is not None
            st = os.stat(args.ram_path)
            assert st.st_size > 0 and st.st_blocks * 512 >= st.st_size
        finally:
            gate.set()

    try:
        _with_daemon(args, body)
    finally:
        shutil.rmtree(sockdir, ignore_errors=True)
    assert not Path(args.ram_path).exists()  # the pool goes with the daemon


def test_four_ranks_register_at_once_on_a_fresh_daemon_and_all_succeed_quickly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Window 2: four TP ranks called register_ns together while the pool was being
    allocated; all four timed out at 2 s. A slow allocation now delays only the
    daemon's start, and every register_ns answers at once."""
    real = os.posix_fallocate

    def slow_fallocate(fd: int, off: int, n: int) -> None:
        time.sleep(2.5)  # longer than the old 2 s register_ns deadline
        real(fd, off, n)

    monkeypatch.setattr(os, "posix_fallocate", slow_fallocate, raising=False)
    sockdir = tempfile.mkdtemp(dir="/tmp")
    args = _run_args(tmp_path, sockdir)
    results: list[tuple[bytes | None, float]] = [(None, 0.0)] * 4

    def body() -> None:
        assert _wait_hello(args.socket, 10) is not None
        barrier = threading.Barrier(4)

        def rank(i: int) -> None:
            c = Client(args.socket)  # the connector's own defaults
            ns = dataclasses.replace(NS, tp=4, revision_sha=f"rank{i}")
            barrier.wait()
            t0 = time.monotonic()
            results[i] = (c.register_ns(ns, GROUPS, nvme=False), time.monotonic() - t0)

        threads = [threading.Thread(target=rank, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(30)

    try:
        _with_daemon(args, body)
    finally:
        shutil.rmtree(sockdir, ignore_errors=True)
    assert all(ns is not None for ns, _ in results), results
    assert max(took for _, took in results) < 1.0, results


def test_register_ns_has_a_generous_deadline() -> None:
    c = Client("/nonexistent.sock")
    assert c.register_timeout_s >= 30.0
    assert c.slow_timeout_s == 2.0  # begin_load's base deadline is unchanged


class _FlakyServer:
    """Answers register_ns, but never the first one: that call times out."""

    def __init__(self, refuse: bool = False) -> None:
        self.dir = tempfile.mkdtemp(dir="/tmp")
        self.path = os.path.join(self.dir, "f.sock")
        self.srv = socket.socket(socket.AF_UNIX)
        self.srv.bind(self.path)
        self.srv.listen(8)
        self.requests: list[dict[str, Any]] = []
        self.refuse = refuse
        self._held: list[socket.socket] = []
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        for line in f:
            req = json.loads(line)
            self.requests.append(req)
            if len(self.requests) == 1 and not self.refuse:
                self._held.append(conn)  # never answer the first call
                return
            if self.refuse:
                reply = {"id": req["id"], "ok": False, "error": "bad_request: no"}
            else:
                ns_hex = hashlib.sha256(b"ns").hexdigest()
                reply = {"id": req["id"], "ok": True, "ns": ns_hex}
            conn.sendall(json.dumps(reply).encode() + b"\n")

    def close(self) -> None:
        self.srv.close()
        for c in self._held:
            c.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def test_register_ns_retries_once_after_a_timeout(caplog: pytest.LogCaptureFixture) -> None:
    srv = _FlakyServer()
    try:
        c = Client(srv.path, register_timeout_s=0.3)
        t0 = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="lmw_kv.client"):
            got = c.register_ns(NS, GROUPS)
        took = time.monotonic() - t0
        assert got == hashlib.sha256(b"ns").digest()
        assert [r["op"] for r in srv.requests] == ["register_ns", "register_ns"]
        assert took < 0.3 + 0.5 + 0.3 + 0.5
        assert any("retrying" in r.getMessage() for r in caplog.records)
    finally:
        srv.close()


def test_register_ns_does_not_retry_a_refusal() -> None:
    srv = _FlakyServer(refuse=True)
    try:
        c = Client(srv.path, register_timeout_s=0.3)
        assert c.register_ns(NS, GROUPS) is None
        assert len(srv.requests) == 1
    finally:
        srv.close()


def test_register_ns_gives_up_after_one_retry(tmp_path: Path) -> None:
    c = Client(str(tmp_path / "nothing.sock"), register_timeout_s=0.1)
    t0 = time.monotonic()
    assert c.register_ns(NS, GROUPS) is None
    assert time.monotonic() - t0 < 2.0


# ---- review of !683: engine workers, SIGTERM during the pool commit ----------------


def test_the_engine_worker_count_reaches_the_store_config(tmp_path: Path) -> None:
    from lmw_kv import daemon as daemon_mod

    args = _run_args(tmp_path, "/tmp")
    args.engine_workers = 3
    cfg, fd = daemon_mod.store_config(args)
    assert fd is None and cfg.engine_workers == 3
    assert (
        daemon_mod.parse_args(
            ["--socket", "s", "--ram-path", "p", "--ram-gib", "1", "--engine-workers", "2"]
        ).engine_workers
        == 2
    )
    assert (
        daemon_mod.parse_args(["--socket", "s", "--ram-path", "p", "--ram-gib", "1"]).engine_workers
        == 0
    )


def test_sigterm_while_the_pool_is_committed_removes_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop that lands during the (seconds long) commit must not leave the pool file in
    /dev/shm: the handler is armed before the store is built, and the store is closed."""
    from lmw_kv import daemon as daemon_mod
    from lmw_kv import store as store_mod

    gate = threading.Event()
    entered = threading.Event()
    real = store_mod.create_pool_file

    def slow(path: Path, size: int) -> None:
        real(path, size)
        entered.set()
        assert gate.wait(10)

    monkeypatch.setattr(store_mod, "create_pool_file", slow)
    sockdir = tempfile.mkdtemp(dir="/tmp")
    args = _run_args(tmp_path, sockdir)

    async def main() -> None:
        task = asyncio.create_task(daemon_mod._run(args))
        assert await asyncio.to_thread(entered.wait, 5)
        assert Path(args.ram_path).exists()
        os.kill(os.getpid(), signal.SIGTERM)  # mid-commit
        await asyncio.sleep(0.2)
        gate.set()
        await asyncio.wait_for(task, 10)

    try:
        asyncio.run(main())
    finally:
        gate.set()
        shutil.rmtree(sockdir, ignore_errors=True)
    assert not Path(args.ram_path).exists()
    assert not os.path.exists(args.socket)  # it never started serving
