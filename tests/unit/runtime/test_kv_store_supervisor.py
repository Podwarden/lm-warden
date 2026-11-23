"""KvStoreSupervisor: starts the daemon only when a model wants it, keeps it up.

A tiny fake daemon (a Python script) stands in for ``lmw_kv.daemon``: it records
its argv and answers ``hello`` on the socket, so no test needs /dev/shm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.settings import SettingsRepo
from app.runtime.kv_store import KvStoreSupervisor

FAKE = textwrap.dedent(
    """
    import json, os, socketserver, sys
    a = sys.argv[1:]
    d = dict(zip(a[::2], a[1::2]))
    with open(os.environ["FAKE_ARGS"], "a") as f:
        f.write(json.dumps({"pid": os.getpid(), "args": d}) + "\\n")

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                r = json.loads(line)
                if r["op"] != "hello" and os.environ.get("FAKE_OPS"):
                    with open(os.environ["FAKE_OPS"], "a") as f:
                        f.write(json.dumps({"pid": os.getpid(), "req": r}) + "\\n")
                out = {"op": r["op"], "id": r["id"], "ok": True, "protocol": 1, "minor": 1,
                       "ram_path": d["--ram-path"], "added": 0, "released": 0,
                       "ram_bytes": int(os.environ.get("FAKE_RAM_BYTES", "0"))}
                self.wfile.write(json.dumps(out).encode() + b"\\n")

    socketserver.ThreadingUnixStreamServer(d["--socket"], H).serve_forever()
    """
)


@pytest.fixture(autouse=True)
def _ample_memory(monkeypatch):
    """Sizing against free memory must not depend on the test host: every test sees
    ample memory unless it patches ``fit_memory`` itself."""
    monkeypatch.setattr("lmw_kv.ram.fit_memory", lambda wanted, **kw: wanted)


@pytest.fixture
def env(tmp_path, monkeypatch):
    # AF_UNIX paths are limited to ~104 bytes on macOS: keep the data dir short.
    data = Path(tempfile.mkdtemp(prefix="kv"))
    script = tmp_path / "fake_daemon.py"
    script.write_text(FAKE)
    args_file = tmp_path / "args.jsonl"
    monkeypatch.setenv("FAKE_ARGS", str(args_file))
    settings = SimpleNamespace(data_dir=data, db_path=data / "w.db")
    yield SimpleNamespace(settings=settings, script=script, args_file=args_file)
    import shutil

    shutil.rmtree(data, ignore_errors=True)


def _sup(env, **kw) -> KvStoreSupervisor:
    kw.setdefault("poll_s", 0.05)
    kw.setdefault("backoff_s", (0.05, 0.2))
    kw.setdefault("stop_grace_s", 2.0)
    kw.setdefault("shm_path", str(env.settings.data_dir))  # short and pod-local, like /dev/shm
    return KvStoreSupervisor(env.settings, command=[sys.executable, str(env.script)], **kw)


async def _seed(env, gpus: dict[str, list[int]] | None = None, **kv_by_id: str) -> None:
    async with open_db(env.settings.db_path) as db:
        await apply_migrations(db)
        for mid, kv in kv_by_id.items():
            await ModelRepo(db).insert(
                ModelRow(
                    id=mid,
                    served_model_name=mid,
                    hf_repo="org/m",
                    hf_revision="main",
                    gpu_indices=(gpus or {}).get(mid, [0]),
                    tensor_parallel_size=1,
                    dtype=None,
                    max_model_len=None,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=False,
                    extra_args=[],
                    extra_env={},
                    status="registered",
                    pulled_bytes=0,
                    pulled_total=None,
                    last_error=None,
                    kv_store=kv,
                )
            )


async def _until(cond, limit_s=10.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + limit_s
    while not cond():
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(0.02)


def _launches(env) -> list[dict]:
    if not env.args_file.exists():
        return []
    return [json.loads(x) for x in env.args_file.read_text().splitlines()]


async def test_not_started_when_every_model_is_off(env):
    await _seed(env, a="off", b="off")
    sup = _sup(env)
    await sup.start()
    await asyncio.sleep(0.3)  # negative check: give the poll loop several turns
    assert _launches(env) == []
    assert not sup.healthy()
    assert sup.client() is None
    await sup.stop()


async def test_started_and_healthy_for_a_ram_model(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="off", b="ram")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    client = sup.client()
    assert client is not None
    assert await asyncio.to_thread(client.hello) is not None
    (launch,) = _launches(env)
    assert launch["args"]["--socket"] == str(Path(env.settings.data_dir) / "lmw-kv.sock")
    assert launch["args"]["--ram-gib"] == "16.0"
    assert "--nvme-dir" not in launch["args"]
    await sup.stop()


async def test_started_after_a_model_is_turned_on_without_restart(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="off")
    sup = _sup(env)
    await sup.start()
    await asyncio.sleep(0.2)
    assert _launches(env) == []
    async with open_db(env.settings.db_path) as db:
        await db.execute("UPDATE models SET kv_store = 'ram+nvme' WHERE id = 'a'")
        await db.commit()
    await _until(sup.healthy)
    await sup.stop()


async def test_restarted_after_it_is_killed(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    first = _launches(env)[0]["pid"]
    os.kill(first, signal.SIGKILL)
    await _until(lambda: len(_launches(env)) == 2 and sup.healthy())
    assert _launches(env)[1]["pid"] != first
    await sup.stop()


async def test_stop_removes_the_socket_and_the_process(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    sock = Path(env.settings.data_dir) / "lmw-kv.sock"
    assert sock.exists()
    pid = _launches(env)[0]["pid"]
    await sup.stop()
    assert not sock.exists()
    assert not sup.healthy() and sup.client() is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_the_daemon_gets_the_requested_size_and_owns_the_one_warning(
    env, monkeypatch, caplog
):
    """The daemon sizes the pool (``/dev/shm`` and free memory) right before it commits
    it, and logs the one WARNING naming requested vs granted. The api passes the
    operator's request unchanged and logs nothing about sizing, so the warning names
    the number the operator set, once."""
    seen = []

    def fit(path, wanted):
        seen.append((str(path), wanted))
        return 3 << 30  # a 5 GiB /dev/shm minus the 2 GiB reserve

    monkeypatch.setattr("lmw_kv.ram.fit_shm", fit)
    await _seed(env, a="ram")
    async with open_db(env.settings.db_path) as db:
        await SettingsRepo(db).set("kv_store_ram_gib", "16")
    caplog.set_level(logging.WARNING, logger="app.runtime.kv_store")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    assert _launches(env)[0]["args"]["--ram-gib"] == "16.0"
    assert seen[0] == (str(Path(env.settings.data_dir) / "lmw-kv.pool"), 16 << 30)
    assert not [r for r in caplog.records if r.name == "app.runtime.kv_store"]
    await sup.stop()


async def test_no_daemon_when_free_memory_leaves_no_ram_tier(env, monkeypatch, caplog):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    monkeypatch.setattr("lmw_kv.ram.fit_memory", lambda wanted, **kw: 1 << 20)  # under 1 GiB
    await _seed(env, a="ram")
    caplog.set_level(logging.WARNING, logger="app.runtime.kv_store")
    sup = _sup(env)
    await sup.start()
    await _until(lambda: any("not starting" in r.getMessage() for r in caplog.records))
    assert _launches(env) == []
    await sup.stop()


async def test_nvme_dir_and_cap_are_passed_only_when_set(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram+nvme")
    async with open_db(env.settings.db_path) as db:
        repo = SettingsRepo(db)
        await repo.set("kv_store_nvme_dir", "/tmp/nv")
        await repo.set("kv_store_nvme_cap_gb", "100")
        await repo.set("kv_store_ram_gib", "2")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    args = _launches(env)[0]["args"]
    assert args["--nvme-dir"] == "/tmp/nv"
    assert args["--nvme-cap-gb"] == "100.0"
    assert args["--ram-gib"] == "2.0"
    await sup.stop()


async def test_no_tier_at_all_means_no_daemon(env, monkeypatch, caplog):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: 0)
    await _seed(env, a="ram")
    caplog.set_level(logging.WARNING, logger="app.runtime.kv_store")
    sup = _sup(env)
    await sup.start()
    await _until(lambda: any("not starting" in r.getMessage() for r in caplog.records))
    assert _launches(env) == []
    await sup.stop()


async def test_poke_starts_the_daemon_without_waiting_for_the_poll(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="off")
    sup = _sup(env, poll_s=3600.0)
    await sup.start()
    await asyncio.sleep(0.2)  # let the loop reach its long nap
    async with open_db(env.settings.db_path) as db:
        await db.execute("UPDATE models SET kv_store = 'ram' WHERE id = 'a'")
        await db.commit()
    assert await sup.ensure_running(timeout_s=10.0)
    assert sup.healthy()
    await sup.stop()


async def test_ensure_running_returns_false_after_timeout_when_the_daemon_cannot_start(
    env, monkeypatch
):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    sup = KvStoreSupervisor(
        env.settings,
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
        poll_s=0.05,
        backoff_s=(0.05, 0.1),
    )
    await sup.start()
    assert await sup.ensure_running(timeout_s=0.5) is False
    await sup.stop()


async def test_socket_is_pod_local_next_to_the_pool_never_on_the_data_volume(env, monkeypatch):
    """R25: two api pods share the data volume during a rolling update; the socket must not."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    shm = Path(tempfile.mkdtemp(prefix="shm"))
    sup = _sup(env, shm_path=str(shm))
    await sup.start()
    await _until(sup.healthy)
    (launch,) = _launches(env)
    assert launch["args"]["--socket"] == str(shm / "lmw-kv.sock")
    assert launch["args"]["--ram-path"] == str(shm / "lmw-kv.pool")
    assert not (Path(env.settings.data_dir) / "kv").exists()
    await sup.stop()
    import shutil

    shutil.rmtree(shm, ignore_errors=True)


async def test_a_leftover_pool_file_is_not_counted_when_sizing(env, monkeypatch):
    """C1: a pool file left by a daemon that died must not shrink the next daemon's pool."""
    seen = []

    def fit(path, wanted):
        seen.append(Path(path).exists())
        return wanted

    monkeypatch.setattr("lmw_kv.ram.fit_shm", fit)
    pool = Path(env.settings.data_dir) / "lmw-kv.pool"
    pool.write_bytes(b"x" * 4096)
    await _seed(env, a="ram")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    assert seen and seen[0] is False
    assert not pool.exists()
    await sup.stop()


WEDGING = FAKE.replace(
    "        for line in self.rfile:\n",
    "        for line in self.rfile:\n"
    "            if os.path.exists(os.environ['FAKE_WEDGE']):\n"
    "                import time; time.sleep(3600)\n",
)
assert WEDGING != FAKE


async def test_a_daemon_that_stops_answering_is_restarted(env, monkeypatch, tmp_path):
    """I5: after the first healthy reply the supervisor keeps probing and kills a wedged daemon."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    wedge = tmp_path / "wedge"
    monkeypatch.setenv("FAKE_WEDGE", str(wedge))
    env.script.write_text(WEDGING)
    await _seed(env, a="ram")
    sup = _sup(env, live_s=0.05, live_timeout_s=0.2, live_misses=2)
    await sup.start()
    await _until(sup.healthy)
    first = _launches(env)[0]["pid"]
    await asyncio.sleep(0.3)  # answering daemons are left alone
    assert len(_launches(env)) == 1
    wedge.touch()  # from now on every hello hangs
    await _until(lambda: not sup.healthy())
    wedge.unlink()
    await _until(lambda: len(_launches(env)) == 2 and sup.healthy())
    with pytest.raises(ProcessLookupError):
        os.kill(first, 0)
    await sup.stop()


async def test_a_daemon_that_never_answers_is_restarted(env, monkeypatch, tmp_path):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    wedge = tmp_path / "wedge"
    wedge.touch()
    monkeypatch.setenv("FAKE_WEDGE", str(wedge))
    env.script.write_text(WEDGING)
    await _seed(env, a="ram")
    sup = _sup(env, start_timeout_s=0.5)
    await sup.start()
    await _until(lambda: len(_launches(env)) >= 2)
    assert not sup.healthy()
    wedge.unlink()
    await _until(sup.healthy)
    await sup.stop()


async def test_pool_sizing_runs_off_the_event_loop(env, monkeypatch):
    """F9: discard_stale and fit_shm (unlink, statvfs) must not block the api's loop."""
    import threading

    seen: list[threading.Thread] = []

    def fit(path, wanted):
        seen.append(threading.current_thread())
        return wanted

    def discard(path):
        seen.append(threading.current_thread())

    monkeypatch.setattr("lmw_kv.ram.fit_shm", fit)
    monkeypatch.setattr("lmw_kv.ram.discard_stale", discard)
    await _seed(env, a="ram")
    loop_thread = threading.current_thread()
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    await sup.stop()
    assert len(seen) >= 2
    assert all(t is not loop_thread for t in seen)


SLOW = FAKE.replace(
    "socketserver.ThreadingUnixStreamServer(",
    "import time; time.sleep(float(os.environ['FAKE_DELAY']))\n"
    "socketserver.ThreadingUnixStreamServer(",
)
assert SLOW != FAKE


async def test_ensure_running_waits_for_a_slow_first_start(env, monkeypatch):
    """F3: a first start commits the whole pool; a load must wait for it, not give up early."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    monkeypatch.setenv("FAKE_DELAY", "1.0")
    env.script.write_text(SLOW)
    await _seed(env, a="ram")
    sup = _sup(env, start_timeout_s=30.0)
    await sup.start()
    assert await sup.ensure_running() is True
    await sup.stop()


async def test_ensure_running_waits_up_to_the_daemon_start_timeout(env, monkeypatch, tmp_path):
    """F3: the wait is the daemon's own start timeout, not a fixed 10 s."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    wedge = tmp_path / "wedge"
    wedge.touch()
    monkeypatch.setenv("FAKE_WEDGE", str(wedge))
    env.script.write_text(WEDGING)
    await _seed(env, a="ram")
    sup = _sup(env, start_timeout_s=0.5)
    await sup.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await sup.ensure_running() is False
    assert 0.4 <= loop.time() - t0 < 5.0
    await sup.stop()


async def test_ensure_running_returns_at_once_when_there_is_nothing_to_start(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: 0)  # no RAM, no NVMe
    await _seed(env, a="ram")
    sup = _sup(env, start_timeout_s=60.0, poll_s=3600.0)
    await sup.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await sup.ensure_running() is False
    assert loop.time() - t0 < 5.0
    await sup.stop()


async def test_a_restarted_daemon_is_a_new_generation_and_is_reported(env, monkeypatch, caplog):
    """F3: engines loaded under the old daemon keep their tier off; the restart is visible."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    caplog.set_level(logging.WARNING, logger="app.runtime.kv_store")
    heard: list[int] = []
    sup = _sup(env)
    sup.on_restart(heard.append)
    assert sup.generation() == 0
    await sup.start()
    await _until(sup.healthy)
    assert sup.generation() == 1 and heard == []
    os.kill(_launches(env)[0]["pid"], signal.SIGKILL)
    await _until(lambda: sup.generation() == 2 and sup.healthy())
    assert heard == [2]
    assert any("until they are reloaded" in r.getMessage() for r in caplog.records)
    await sup.stop()


# ---- F4: a daemon an earlier api left behind ------------------------------------


def _fake_proc(root: Path, pid: int, argv: list[str]) -> None:
    (root / str(pid)).mkdir(parents=True, exist_ok=True)
    (root / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_reap_stops_a_stale_daemon_that_serves_our_socket(tmp_path):
    from app.runtime.kv_store import reap_stale_daemon

    proc = tmp_path / "proc"
    sock = "/dev/shm/lmw-kv.sock"
    _fake_proc(proc, 4242, ["python", "-m", "lmw_kv.daemon", "--socket", sock, "--ram-gib", "16"])
    pidfile = tmp_path / "lmw-kv.pid"
    pidfile.write_text("4242\n")
    sent: list[tuple[int, int]] = []

    def kill(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGTERM:  # the daemon exits on SIGTERM
            import shutil

            shutil.rmtree(proc / str(pid))

    assert reap_stale_daemon(pidfile, sock, proc_root=proc, kill=kill, sleep=lambda s: None) == 4242
    assert sent == [(4242, signal.SIGTERM)]
    assert not pidfile.exists()


def test_reap_sigkills_a_stale_daemon_that_ignores_sigterm(tmp_path):
    from app.runtime.kv_store import reap_stale_daemon

    proc = tmp_path / "proc"
    sock = "/x/lmw-kv.sock"
    _fake_proc(proc, 77, ["python", "-m", "lmw_kv.daemon", "--socket", sock])
    pidfile = tmp_path / "lmw-kv.pid"
    pidfile.write_text("77")
    sent: list[int] = []
    reap_stale_daemon(
        pidfile, sock, proc_root=proc, kill=lambda p, s: sent.append(s), grace_s=0.3,
        sleep=lambda s: None,
    )  # fmt: skip
    assert sent == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "lmw_kv.daemon", "--socket", "/other/lmw-kv.sock"],  # another pod's
        ["sleep", "100"],  # the pid was reused by something else
        [],  # gone (or a zombie: empty cmdline)
    ],
)
def test_reap_never_touches_a_process_that_is_not_our_daemon(tmp_path, argv):
    from app.runtime.kv_store import reap_stale_daemon

    proc = tmp_path / "proc"
    if argv:
        _fake_proc(proc, 99, argv)
    else:
        proc.mkdir()
    pidfile = tmp_path / "lmw-kv.pid"
    pidfile.write_text("99")
    sent: list[int] = []
    found = reap_stale_daemon(
        pidfile, "/dev/shm/lmw-kv.sock", proc_root=proc, kill=lambda p, s: sent.append(s)
    )
    assert found is None and sent == []
    assert not pidfile.exists()


async def test_supervisor_records_the_daemon_pid_and_passes_its_own_pid(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    (launch,) = _launches(env)
    assert launch["args"]["--parent-pid"] == str(os.getpid())
    pidfile = Path(env.settings.data_dir) / "lmw-kv.pid"
    assert pidfile.read_text().strip() == str(launch["pid"])
    await sup.stop()
    assert not pidfile.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
async def test_supervisor_start_stops_a_daemon_left_by_an_earlier_api(env, monkeypatch):
    """F4: an OOM-killed api leaves its daemon; the next api stops it before it starts its own."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    shm = Path(env.settings.data_dir)
    sock = shm / "lmw-kv.sock"
    stale = await asyncio.create_subprocess_exec(
        sys.executable, str(env.script), "--socket", str(sock), "--ram-path", str(shm / "p"),
        start_new_session=True,
    )  # fmt: skip
    try:
        await _until(sock.exists)
        (shm / "lmw-kv.pid").write_text(str(stale.pid))
        sup = _sup(env)
        await sup.start()
        assert await asyncio.wait_for(stale.wait(), 10) == -signal.SIGTERM
        await _until(sup.healthy)
        assert _launches(env)[-1]["pid"] != stale.pid
        await sup.stop()
    finally:
        if stale.returncode is None:
            stale.kill()
            await stale.wait()


def test_reap_waits_for_a_sigkilled_daemon_to_be_gone(tmp_path):
    """M4: after SIGKILL the stale daemon may hold the NVMe lock a moment longer; wait."""
    from app.runtime.kv_store import reap_stale_daemon

    proc = tmp_path / "proc"
    sock = "/x/lmw-kv.sock"
    _fake_proc(proc, 55, ["python", "-m", "lmw_kv.daemon", "--socket", sock])
    pidfile = tmp_path / "lmw-kv.pid"
    pidfile.write_text("55")
    events: list[str] = []
    after_kill = {"n": 0}

    def kill(pid, sig):
        events.append("TERM" if sig == signal.SIGTERM else "KILL")

    def sleep(s):
        if "KILL" in events:
            after_kill["n"] += 1
            if after_kill["n"] == 3:  # gone three polls after SIGKILL
                import shutil

                shutil.rmtree(proc / "55")
                events.append("gone")

    reap_stale_daemon(pidfile, sock, proc_root=proc, kill=kill, grace_s=0.2, sleep=sleep)
    assert events == ["TERM", "KILL", "gone"]  # returned only once it was gone
    assert after_kill["n"] == 3


# ---- review of !683 ---------------------------------------------------------------


async def test_the_daemon_keeps_memory_for_one_engine_worker_per_gpu(env, monkeypatch):
    """The pool is sized before any engine runs: every GPU a model can use may hold one
    engine worker, whose host memory the pool must leave free."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    seen = []

    def fit(wanted, **kw):
        seen.append(kw)
        return wanted

    monkeypatch.setattr("lmw_kv.ram.fit_memory", fit)
    await _seed(env, gpus={"a": [0, 1], "b": [1, 2]}, a="ram", b="off")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    assert _launches(env)[0]["args"]["--engine-workers"] == "3"
    assert seen[0] == {"engine_workers": 3}
    await sup.stop()


async def test_requested_and_granted_ram_are_reported(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    monkeypatch.setenv("FAKE_RAM_BYTES", str(3 << 30))
    await _seed(env, a="ram")
    async with open_db(env.settings.db_path) as db:
        await SettingsRepo(db).set("kv_store_ram_gib", "16")
    sup = _sup(env)
    assert sup.ram() == {"requested_bytes": None, "granted_bytes": None}
    await sup.start()
    await _until(sup.healthy)
    assert sup.ram() == {"requested_bytes": 16 << 30, "granted_bytes": 3 << 30}
    await sup.stop()


def test_the_start_timeout_grows_with_the_pool(env):
    sup = _sup(env)  # start_timeout_s 60
    assert sup.start_timeout_for(8.0) == 60.0
    assert sup.start_timeout_for(30.0) == 60.0
    assert sup.start_timeout_for(90.0) == 180.0  # 2 s per GiB above 30 GiB
    assert _sup(env, start_timeout_s=0.5).start_timeout_for(16.0) == 0.5  # a set value holds


async def test_ensure_running_waits_for_the_scaled_start_timeout(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    async with open_db(env.settings.db_path) as db:
        await SettingsRepo(db).set("kv_store_ram_gib", "100")
    sup = _sup(env)
    await sup.start()
    await _until(sup.healthy)
    assert sup.start_timeout() == 200.0
    await sup.stop()


async def test_the_no_ram_tier_warning_is_logged_once(env, monkeypatch, caplog):
    """The poll runs every few seconds; the operator gets the warning once, not each poll."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: 0)
    await _seed(env, a="ram")
    caplog.set_level(logging.WARNING, logger="app.runtime.kv_store")
    sup = _sup(env)
    await sup.start()
    await _until(lambda: sup._poll_seq >= 6)
    await sup.stop()
    assert len([r for r in caplog.records if "not starting" in r.getMessage()]) == 1


# ---- Task 12: every healthy generation is synced ----------------------------------------


async def test_on_healthy_fires_for_generation_1_and_again_for_2(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    heard: list[tuple[int, bool]] = []
    sup = _sup(env)
    sup.on_healthy(lambda gen: heard.append((gen, sup.healthy())))
    sup.on_healthy(lambda gen: 1 / 0)  # a failing listener stops nothing
    await sup.start()
    await _until(sup.healthy)
    assert heard == [(1, True)]
    os.kill(_launches(env)[0]["pid"], signal.SIGKILL)
    await _until(lambda: sup.generation() == 2 and sup.healthy())
    assert heard == [(1, True), (2, True)]
    await sup.stop()


def _ops(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines()]


async def _seed_retired_key(env) -> bytes:
    """An isolated key that was reset once: its epoch 0 tag is retired. Returns the tag."""
    from app.db.repos.tokens import TokenRepo, sqlite_utc_now
    from app.kv.isolation import tag_for

    tf = tag_for(NK)
    async with open_db(env.settings.db_path) as db:
        repo = TokenRepo(db)
        await repo.create("k", "k", "vw_k_" + "x" * 30, priority=4, kv_isolation="isolated")
        await repo.update_kv("k", isolation=None, reset=True, tag_for=tf, now=sqlite_utc_now())
    return tf("k", 0)


NK = bytes(range(32))


async def _reap_then_sync(env, monkeypatch, tmp_path, stale_first: bool):
    from app.kv.policy_sync import PolicySync, request_sync

    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    ops_file = tmp_path / "ops.jsonl"
    monkeypatch.setenv("FAKE_OPS", str(ops_file))
    await _seed(env, a="ram")
    tag = await _seed_retired_key(env)
    shm = Path(env.settings.data_dir)
    stale = None
    if stale_first:
        sock = shm / "lmw-kv.sock"
        stale = await asyncio.create_subprocess_exec(
            sys.executable, str(env.script), "--socket", str(sock), "--ram-path", str(shm / "p"),
            start_new_session=True,
        )  # fmt: skip
        await _until(sock.exists)
        (shm / "lmw-kv.pid").write_text(str(stale.pid))
    sup = _sup(env)
    state = SimpleNamespace(
        kv_store=sup,
        settings=env.settings,
        cobs2_key=SimpleNamespace(key=NK),
        kv_sync=PolicySync(window_s=0.05),
    )
    sup.on_healthy(lambda gen: request_sync(state))
    sup.sync_before_loads()
    try:
        await sup.start()
        if stale is not None:
            assert await asyncio.wait_for(stale.wait(), 10) == -signal.SIGTERM
        # What a model load does first: once this returns, the daemon has both.
        assert await sup.ensure_running(timeout_s=20) is True
        first = _launches(env)[-1]["pid"]
        assert stale is None or first != stale.pid
        mine = [o["req"] for o in _ops(ops_file) if o["pid"] == first]
        assert [r["op"] for r in mine] == ["set_policy", "retire_scopes"]
        assert mine[0]["policy"]["owners"] == {"a": 1}
        assert len(mine[0]["policy"]["tenants"]) == 1
        assert mine[1]["scope_tags"] == [tag.hex()]
        # A restarted daemon (generation 2) gets both again before a load goes ahead.
        os.kill(first, signal.SIGKILL)
        await _until(lambda: sup.generation() == 2 and sup.healthy())
        assert await sup.ensure_running(timeout_s=20) is True
        second = _launches(env)[-1]["pid"]
        mine = [o["req"] for o in _ops(ops_file) if o["pid"] == second]
        assert [r["op"] for r in mine] == ["set_policy", "retire_scopes"]
        assert mine[1]["scope_tags"] == [tag.hex()]
    finally:
        await sup.stop()
        await state.kv_sync.aclose()
        if stale is not None and stale.returncode is None:
            stale.kill()
            await stale.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
async def test_reap_then_sync(env, monkeypatch, tmp_path):
    """Review Focus 3: a stale daemon is reaped; generation 1 gets the policy and the full
    retired list before the first model load begins, and generation 2 gets both again."""
    await _reap_then_sync(env, monkeypatch, tmp_path, stale_first=True)


async def test_every_generation_is_synced_before_loads(env, monkeypatch, tmp_path):
    await _reap_then_sync(env, monkeypatch, tmp_path, stale_first=False)


async def test_a_load_does_not_wait_for_a_sync_that_never_comes(env, monkeypatch):
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, a="ram")
    sup = _sup(env, sync_wait_s=0.3)
    sup.sync_before_loads()  # but nobody syncs
    await sup.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await sup.ensure_running(timeout_s=20) is True
    assert loop.time() - t0 < 3
    sup.mark_synced(sup.generation())
    t0 = loop.time()
    assert await sup.ensure_running() is True
    assert loop.time() - t0 < 0.2
    await sup.stop()


async def test_stats_have_their_own_client_with_a_slow_timeout(env, monkeypatch):
    """Reports call ``stats`` on a client of their own: a slow reply never closes the
    hot-path client's connection (whose 50 ms timeout guards every turn's tier hits)."""
    monkeypatch.setattr("lmw_kv.ram.fit_shm", lambda path, wanted: wanted)
    await _seed(env, b="ram")
    sup = _sup(env)
    assert sup.stats_client() is None
    await sup.start()
    await _until(sup.healthy)
    stats, hot = sup.stats_client(), sup.client()
    assert stats is not None and hot is not None and stats is not hot
    assert stats.timeout_s == 2.0 and hot.timeout_s == 0.05
    await sup.stop()
    assert sup.stats_client() is None
