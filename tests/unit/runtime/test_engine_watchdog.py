"""#218 — the orphan GPU reaper must kill only processes this warden fathered.

Before this, ``reap_orphan_gpu_holders`` decided ownership with
``"vllm" in cmdline.lower()``. That matched a second vllm-warden on the same
box, another team's vLLM container and a researcher running
``python -m vllm.entrypoints.openai.api_server`` — and SIGKILLed all of them the
moment this warden recovered a crashed model. The ownership signal is now
session membership: ``LocalSubprocessDriver.spawn`` uses
``start_new_session=True``, so a wrapper's session id is its own pid and every
worker it forks inherits that id and keeps it after the wrapper dies.

These tests pin both halves of the trade: a stranger is never killed even when
its cmdline is indistinguishable from ours, and a genuine orphan of ours still
is.
"""

from __future__ import annotations

import asyncio
import errno
import time
from types import SimpleNamespace

import pytest

from app.runtime import watchdog as wd
from app.runtime.watchdog import EngineOwnership
from tests.conftest import wait_until_async

# A real foreign command line, of the shape the old substring test could not
# tell from our own workers.
FOREIGN_CMD = (
    "/home/researcher/.venv/bin/python -m vllm.entrypoints.openai.api_server "
    "--model mistralai/Mistral-7B-v0.3 --port 8000"
)


def _patch_procs(monkeypatch, *, apps, cmds, sids, parents=None, killed=None):
    """Fake out every /proc + nvidia-smi read the reaper makes.

    ``sids`` is the piece that matters: it is the process table's answer to
    "which session does this pid belong to", which is what ownership now keys
    on. A value that is an Exception is RAISED by the fake ``_sid``, standing
    in for ``os.getsid`` failing on a pid that is live but unreadable.
    ``apps`` may be a callable so a test can model the table changing between
    sweeps (a kill removing its holder), and ``killed`` may name the list the
    fake ``os.kill`` appends to so a callable ``apps`` can close over it.
    """
    parents = parents or {}
    monkeypatch.setattr(wd, "_gpu_compute_apps", apps if callable(apps) else (lambda: apps))
    monkeypatch.setattr(wd, "_cmdline", lambda pid: cmds.get(pid, ""))

    def _sid(pid):
        value = sids.get(pid)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(wd, "_sid", _sid)
    monkeypatch.setattr(wd, "_ppid", lambda pid: parents.get(pid))
    killed = killed if killed is not None else []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))
    return killed


# --- the stranger ------------------------------------------------------------


def test_never_kills_a_foreign_vllm_process(monkeypatch):
    """The exact process #218 is about: another tenant's vLLM, same cmdline."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(31337, 10550)],
        cmds={31337: FOREIGN_CMD},
        sids={31337: 31000},  # a session this warden never started
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == [], "a cmdline substring is not an ownership signal"


def test_never_kills_a_second_wardens_orphaned_workers(monkeypatch):
    """Two vllm-wardens on one box. Their orphans look exactly like ours —
    same VLLM::Worker_TP0 comm, same dead-wrapper shape — and are still theirs."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(700001, 10550), (700002, 10550)],
        cmds={700001: "VLLM::Worker_TP0", 700002: "VLLM::Worker_TP1"},
        sids={700001: 690000, 700002: 690000},  # the other warden's wrapper
        parents={700001: 1, 700002: 1},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


def test_kills_nothing_when_it_can_attribute_nothing(monkeypatch):
    """Empty ownership — e.g. straight after a warden restart — must be inert.

    This is the fail-safe: an un-reaped orphan costs one failed reload, a
    wrongly reaped stranger costs somebody their job.
    """
    killed = _patch_procs(
        monkeypatch,
        apps=[(801467, 10550), (31337, 4000)],
        cmds={801467: "VLLM::Worker_TP0", 31337: FOREIGN_CMD},
        sids={801467: 800137, 31337: 31000},
    )

    assert wd.reap_orphan_gpu_holders(EngineOwnership()) == []
    assert killed == []


def test_declined_kills_are_logged_with_a_reason(monkeypatch, caplog):
    """An operator looking at "Free memory on device cuda:0" after a failed
    reload has to be able to see that something we refused to touch is sitting
    on the VRAM. Silence here turns a diagnosable failure into a mystery.

    Both decline reasons are exercised, because both have to carry the MiB:
    31337 is a stranger's session, 555 is in a session of ours but fails the
    pid-wraparound cmdline check. Whichever one the operator hits, the number
    they need is the one in the CUDA error they are staring at.
    """
    _patch_procs(
        monkeypatch,
        apps=[(31337, 10550), (555, 8000)],
        cmds={31337: FOREIGN_CMD, 555: "/usr/bin/python3 train_resnet.py"},
        sids={31337: 31000, 555: 800137},
        parents={555: 1},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())
    with caplog.at_level("WARNING", logger="app.runtime.watchdog"):
        assert wd.reap_orphan_gpu_holders(owned) == []

    text = caplog.text
    assert "31337" in text, "the pid we declined to kill must be named"
    assert "10550" in text, "how much VRAM it holds is the operator's whole question"
    assert "555" in text
    assert "8000 MiB" in text, "the second decline reason must carry the VRAM too"
    assert "#218" in text


def test_unreadable_session_is_not_attributed(monkeypatch):
    """_sid returns None for a process in another PID namespace, or one that
    exited between the nvidia-smi read and the /proc read. Unknown is not ours."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(801467, 10550)],
        cmds={801467: "VLLM::Worker_TP0"},
        sids={801467: None},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (PermissionError(errno.EPERM, "Operation not permitted"), "EPERM"),
        (ProcessLookupError(errno.ESRCH, "No such process"), "ESRCH"),
    ],
)
def test_unreadable_session_says_why_it_refused(monkeypatch, caplog, error, code):
    """#278 slice 3 — the 2026-09-30 refusal read 'session=None not one of
    ours', and an operator could not tell a pid that raced away (ESRCH) from a
    pid alive in another PID namespace (EPERM). The refusal must name the errno.
    The process is still NOT killed: unreadable is not ours."""

    def _getsid(pid):
        raise error

    monkeypatch.setattr(wd, "_gpu_compute_apps", lambda: [(801467, 24604)])
    monkeypatch.setattr(wd, "_cmdline", lambda pid: "VLLM::Worker_TP0")
    monkeypatch.setattr(wd, "_ppid", lambda pid: None)
    killed = []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(wd.os, "getsid", _getsid)
    owned = EngineOwnership(sessions=frozenset({801467}), live_pids=frozenset())

    with caplog.at_level("WARNING", logger="app.runtime.watchdog"):
        assert wd.reap_orphan_gpu_holders(owned) == []

    assert killed == [], "an unreadable session must never become a kill"
    assert code in caplog.text, "the refusal must say WHY the session was unreadable"


def test_pid_reuse_in_an_owned_session_still_needs_to_look_like_vllm(monkeypatch):
    """Defence in depth: session ids are pids and pid numbers wrap. If a session
    we recorded is re-created by an unrelated leader, the second test saves us."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(555, 8000)],
        cmds={555: "/usr/bin/python3 train_resnet.py"},
        sids={555: 800137},  # collides with a wrapper pid we once spawned
        parents={555: 1},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


# --- our own orphans ---------------------------------------------------------


def test_reaps_our_own_orphaned_workers(monkeypatch):
    """2026-08-17: four VLLM::Worker_TP* held 10550 MiB each after EngineCore
    and the wrapper were both gone, and every reload then failed on free VRAM."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(801467, 10550), (801590, 10550)],
        cmds={801467: "VLLM::Worker_TP0", 801590: "VLLM::Worker_TP1"},
        sids={801467: 800137, 801590: 800137},  # 800137 = our dead wrapper
        # Reparented to the container shim, NOT pid 1 — the host PID namespace
        # is shared, so an "orphan == ppid 1" test would miss these entirely.
        parents={801467: 794190, 801590: 794190, 794190: 0},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert sorted(wd.reap_orphan_gpu_holders(owned)) == [801467, 801590]
    assert sorted(killed) == [801467, 801590]


def test_never_kills_workers_of_a_live_engine(monkeypatch):
    """A second, healthy model must survive the first one's recovery — even
    though its wrapper's session IS one of ours."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(900001, 10550)],
        cmds={900001: "VLLM::Worker_TP0"},
        sids={900001: 888000},
        parents={900001: 888000, 888000: 794190},
    )
    owned = EngineOwnership(sessions=frozenset({800137, 888000}), live_pids=frozenset({888000}))

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


def test_live_engine_survives_a_broken_parent_chain(monkeypatch):
    """The ancestor walk fails when an intermediate process is gone or /proc
    is unreadable. Session membership of a LIVE wrapper still protects the
    workers — which is the difference between a spurious kill and a no-op."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(900001, 10550)],
        cmds={900001: "VLLM::Worker_TP0"},
        sids={900001: 888000},
        parents={},  # no ancestry available at all
    )
    owned = EngineOwnership(sessions=frozenset({888000}), live_pids=frozenset({888000}))

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


# --- building the ownership set ---------------------------------------------


class _Handle:
    def __init__(self, pid, returncode=None):
        self.pid = pid
        self.returncode = returncode


class _Sup:
    def __init__(self, handles):
        self._handles = dict(handles)

    def get_pid(self, model_id):
        h = self._handles.get(model_id)
        return h.pid if h else None


def test_ownership_separates_live_wrappers_from_exited_ones():
    """An exited wrapper is still an owned SESSION (its workers may be holding
    VRAM) but must not be in live_pids, or its orphans are protected forever."""
    sup = _Sup({"a": _Handle(800137, returncode=1), "b": _Handle(888000)})

    owned = wd.engine_ownership(sup)

    assert owned.sessions == frozenset({800137, 888000})
    assert owned.live_pids == frozenset({888000})


def test_ownership_carries_the_pid_the_unload_is_about_to_erase():
    """Supervisor.unload pops the handle in a finally, so _restart has to hand
    the doomed pid in explicitly or the reaper is a no-op on its only case."""
    owned = wd.engine_ownership(_Sup({}), also_sessions=(800137, None))

    assert owned.sessions == frozenset({800137})
    assert owned.live_pids == frozenset()


# --- the wrapper-pid ledger --------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_ledger():
    wd._WRAPPER_PIDS.clear()
    yield
    wd._WRAPPER_PIDS.clear()


def test_ledger_survives_the_supervisor_dropping_the_handle():
    """_watch_exit pops the handle the instant the wrapper exits, which is
    BEFORE the sweep gets round to restarting the model. Without the ledger the
    exit path — the commonest crash shape — could never attribute its orphans."""
    sup = _Sup({"m1": _Handle(800137)})
    wd.remember_wrapper_pids(sup)

    sup._handles.clear()  # what _watch_exit does on exit

    assert sup.get_pid("m1") is None
    assert wd.last_wrapper_pid("m1") == 800137


def test_ledger_entry_expires(monkeypatch):
    """A model that failed and was never restarted must not leave a pid lying
    around until the kernel reissues that number to a stranger."""
    wd.remember_wrapper_pids(_Sup({"m1": _Handle(800137)}))
    now = time.monotonic()
    monkeypatch.setattr(wd.time, "monotonic", lambda: now + wd._WRAPPER_PID_TTL_S + 1)

    assert wd.last_wrapper_pid("m1") is None
    assert "m1" not in wd._WRAPPER_PIDS, "an expired entry must be dropped, not re-read"


def test_ledger_forgets_deleted_models():
    wd.remember_wrapper_pids(_Sup({"m1": _Handle(800137), "m2": _Handle(888000)}))

    wd.forget_wrapper_pids({"m1"})

    assert wd.last_wrapper_pid("m1") == 800137
    assert wd.last_wrapper_pid("m2") is None


def test_ledger_keeps_every_wrapper_pid_a_model_spawns():
    """#278 slice 3 — a double spawn leaves TWO wrapper pids in the process
    table (the first already dead, the second live), and the old one-slot
    ledger overwrote the first the moment the second was sampled. Both
    sessions must stay attributable, and re-recording a pid — what the tick
    does with the live handle every interval — must refresh it, not duplicate."""
    wd.record_wrapper_pid("m1", 1001)
    wd.record_wrapper_pid("m1", 1002)

    assert wd.fresh_wrapper_pids("m1") == [1002, 1001]
    assert wd.last_wrapper_pid("m1") == 1002

    wd.record_wrapper_pid("m1", 1002)

    assert wd.fresh_wrapper_pids("m1") == [1002, 1001], "re-recording must refresh, not duplicate"


def test_spawn_recorded_pids_expire_with_the_ledger_ttl(monkeypatch):
    """#278 slice 3 — a spawn-time record is only as trustworthy as a
    tick-sampled one: past the TTL the kernel may have reissued that pid to a
    stranger, so it must stop being attributed."""
    wd.record_wrapper_pid("m1", 1001)
    wd.record_wrapper_pid("m1", 1002)
    now = time.monotonic()
    monkeypatch.setattr(wd.time, "monotonic", lambda: now + wd._WRAPPER_PID_TTL_S + 1)

    assert wd.fresh_wrapper_pids("m1") == []
    assert wd.last_wrapper_pid("m1") is None
    assert "m1" not in wd._WRAPPER_PIDS, "an expired entry must be dropped, not re-read"


# --- waiting for the VRAM to come back ---------------------------------------


@pytest.mark.asyncio
async def test_wait_for_gpu_release_returns_when_our_orphans_are_gone(monkeypatch):
    _patch_procs(monkeypatch, apps=[], cmds={}, sids={})
    owned = EngineOwnership(sessions=frozenset({800137}))

    assert await wd.wait_for_gpu_release(owned, timeout_s=1, interval_s=0.01)


@pytest.mark.asyncio
async def test_wait_for_gpu_release_times_out_while_ours_still_holds(monkeypatch):
    """Freeing is not instant after SIGKILL; reloading too early reproduces the
    very failure being recovered from."""
    _patch_procs(
        monkeypatch,
        apps=[(801467, 10550)],
        cmds={801467: "VLLM::Worker_TP0"},
        sids={801467: 800137},
        parents={801467: 794190, 794190: 0},
    )
    owned = EngineOwnership(sessions=frozenset({800137}))

    assert not await wd.wait_for_gpu_release(owned, timeout_s=0.2, interval_s=0.05)


@pytest.mark.asyncio
async def test_wait_does_not_block_on_a_foreign_holder(monkeypatch):
    """We are never going to free a stranger's VRAM, so waiting on it just adds
    30s to a reload that will fail anyway. The declined-kill log is the signal."""
    _patch_procs(
        monkeypatch,
        apps=[(31337, 10550)],
        cmds={31337: FOREIGN_CMD},
        sids={31337: 31000},
    )
    owned = EngineOwnership(sessions=frozenset({800137}))

    assert await wd.wait_for_gpu_release(owned, timeout_s=1, interval_s=0.01)


# --- the restart path wires it together --------------------------------------


@pytest.mark.asyncio
async def test_restart_samples_the_wrapper_pid_before_unloading(monkeypatch, tmp_path):
    """The whole fix hinges on reading get_pid() BEFORE unload() pops the
    handle. Sampling it afterwards yields None and reaps nothing."""
    seen: dict = {}

    class Sup:
        def __init__(self):
            self._handles = {"m1": _Handle(800137)}
            self._generation = {}
            self._load_runners = {}
            self._claims = {}

        def claim(self, model_id, holder):
            current = self._claims.get(model_id)
            if current is None or current == holder:
                self._claims[model_id] = holder
                return None
            return current

        def release_claim(self, model_id, holder):
            if self._claims.get(model_id) == holder:
                del self._claims[model_id]

        def get_pid(self, model_id):
            h = self._handles.get(model_id)
            return h.pid if h else None

        def get_overrides(self, model_id):
            return None

        def get_generation(self, model_id):
            return self._generation.get(model_id, 0)

        def register_load_runner(self, model_id, task):
            self._load_runners[model_id] = task

        def is_running(self, model_id):
            h = self._handles.get(model_id)
            return h is not None and h.returncode is None

        async def unload(self, model_id, *, force=False):
            self._handles.pop(model_id, None)  # what Supervisor.unload does

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    class FakeRepo:
        def __init__(self, db):
            pass

        async def get(self, mid):
            return SimpleNamespace(id=mid, prior_status="loaded", status="loaded")

        async def update_status(self, mid, status, last_error=None):
            pass

        async def set_prior_status(self, mid, prior):
            pass

    async def fake_start_engine(*a, **k):
        return None

    def spy_reap(owned):
        seen["sessions"] = owned.sessions
        return []

    monkeypatch.setattr(wd, "open_db", lambda *a, **k: _NullDB())
    monkeypatch.setattr(wd, "ModelRepo", FakeRepo)
    monkeypatch.setattr(wd, "reap_orphan_gpu_holders", spy_reap)
    monkeypatch.setattr("app.models.routes_api.start_engine", fake_start_engine)

    app_state = SimpleNamespace(
        supervisor=Sup(),
        port_allocator=SimpleNamespace(allocate=lambda: 10000, release=lambda p: None),
    )
    settings = SimpleNamespace(db_path=tmp_path / "x.db")

    await wd._restart(settings, app_state, "m1", None)

    assert 800137 in seen["sessions"], (
        "the doomed wrapper's pid is the session id of the orphans it left "
        "behind; losing it makes the reaper a no-op"
    )


@pytest.mark.asyncio
async def test_restart_reaps_an_orphan_from_a_spawn_the_tick_never_sampled(monkeypatch, tmp_path):
    """#278 slice 3 — the 2026-09-30 double spawn: the first wrapper (pid 1001)
    is replaced by a second spawn (pid 1002) before any watchdog tick samples
    it, so ``remember_wrapper_pids`` never saw 1001. The spawn-time record is
    the only copy of that pid, and the worker the first wrapper fathered still
    holds its session — 24604 MiB the reaper used to decline as 'not one of
    ours' and leave to OOM the reload."""
    dead_wrapper, live_wrapper, worker = 1001, 1002, 4143676
    killed: list[int] = []

    def apps():
        return [] if killed else [(worker, 24604)]

    _patch_procs(
        monkeypatch,
        apps=apps,
        cmds={worker: "VLLM::Worker_TP0"},
        sids={worker: dead_wrapper},
        parents={worker: 0},
        killed=killed,
    )
    wd.record_wrapper_pid("m1", dead_wrapper)  # what the spawn-time hook records
    wd.record_wrapper_pid("m1", live_wrapper)

    async def no_start_engine(*a, **k):
        return None

    real_reap = wd.reap_orphan_gpu_holders
    sup = _LoadSup(pid=live_wrapper)
    _writes, app_state, settings = _wire_restart(monkeypatch, sup=sup, start_engine=no_start_engine)
    monkeypatch.setattr(wd, "reap_orphan_gpu_holders", real_reap)

    out = await wd._restart(settings, app_state, "m1", None)

    assert killed == [worker], (
        "the orphan sits in the FIRST wrapper's session; only the spawn-time "
        "ledger record names it, and the reaper must kill what it fathered"
    )
    assert out == "loaded"


# --- #278 slice 2: a restart never clobbers a load it did not start ----------
#
# _restart must (a) run its spawn as a REGISTERED load runner, so an operator's
# force-unload can cancel it the way it cancels a /load, and (b) drop its
# terminal 'failed' write whenever a newer load has since owned the model (the
# generation moved, or an engine this attempt did not spawn is running). The
# _LoadSup stub plays the real Supervisor's relevant roles: it tracks the
# generation, the running handle, and the registered runner — whose cancel() is
# exactly what Supervisor.unload does to a force-unloaded load's runner.


class _LoadSup:
    def __init__(self, pid=None):
        self._handles = {}
        self._generation = {}
        self._load_runners = {}
        self._claims = {}
        self.unloads = []
        if pid is not None:
            self._handles["m1"] = _Handle(pid)

    def claim(self, model_id, holder):
        current = self._claims.get(model_id)
        if current is None or current == holder:
            self._claims[model_id] = holder
            return None
        return current

    def release_claim(self, model_id, holder):
        if self._claims.get(model_id) == holder:
            del self._claims[model_id]

    def get_pid(self, model_id):
        h = self._handles.get(model_id)
        return h.pid if h else None

    def get_generation(self, model_id):
        return self._generation.get(model_id, 0)

    def is_running(self, model_id):
        h = self._handles.get(model_id)
        return h is not None and h.returncode is None

    def register_load_runner(self, model_id, task):
        self._load_runners[model_id] = task

    async def unload(self, model_id, *, force=False):
        self.unloads.append((model_id, force))
        self._handles.pop(model_id, None)
        runner = self._load_runners.pop(model_id, None)
        if runner is not None and not runner.done():
            runner.cancel()


class _RestartNullDB:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False


class _RestartAlloc:
    def __init__(self):
        self.released = []

    def allocate(self):
        return 10000

    def release(self, p):
        self.released.append(p)


def _wire_restart(monkeypatch, *, sup, start_engine, row=None):
    """Point every _restart collaborator at fakes so the ONLY real code running
    is _restart itself. Returns (writes, app_state, settings) where writes
    records every (kind, value, last_error) update_status/set_prior_status call.

    ``row`` overrides what the (single) FakeRepo read returns — the default is
    a loaded row with prior_status='loaded'; pass e.g. {'status': 'loading',
    'prior_status': None} to model a row that moved under _restart's feet."""
    from app.runtime import watchdog

    writes = []

    class FakeRepo:
        def __init__(self, db):
            pass

        async def get(self, mid):
            if row is not None:
                return SimpleNamespace(id=mid, **row)
            return SimpleNamespace(id=mid, prior_status="loaded", status="loaded")

        async def update_status(self, mid, status, last_error=None):
            writes.append(("status", status, last_error))

        async def set_prior_status(self, mid, prior):
            writes.append(("prior", prior, None))

    monkeypatch.setattr(watchdog, "open_db", lambda *a, **k: _RestartNullDB())
    monkeypatch.setattr(watchdog, "ModelRepo", FakeRepo)
    monkeypatch.setattr(watchdog, "reap_orphan_gpu_holders", lambda owned: [])
    monkeypatch.setattr("app.models.routes_api.start_engine", start_engine)
    app_state = SimpleNamespace(supervisor=sup, port_allocator=_RestartAlloc())
    settings = SimpleNamespace(db_path="x.db")
    return writes, app_state, settings


@pytest.mark.asyncio
async def test_restart_does_not_write_failed_when_a_newer_load_supersedes_the_spawn(monkeypatch):
    """#278 — a concurrent operator load wins: it bumps the generation and
    starts the engine while our respawn is still running. The mark_warming
    RuntimeError our attempt then hits must not write 'failed' over a load we
    did not start (the exact 2026-09-30 production row)."""

    async def superseded_start_engine(*a, **k):
        # The newer load owns the model by the time our spawn errors out:
        # generation moved, and an engine we did not spawn is running.
        sup._generation["m1"] = sup.get_generation("m1") + 1
        sup._handles["m1"] = _Handle(999999)
        raise RuntimeError("cannot mark warming from state ModelState.WARMING: expected LOADING")

    sup = _LoadSup(pid=800137)
    writes, app_state, settings = _wire_restart(
        monkeypatch, sup=sup, start_engine=superseded_start_engine
    )

    out = await wd._restart(settings, app_state, "m1", None)

    assert out == "superseded", "a superseded restart must say so, not pretend it loaded"
    assert ("status", "loading", None) in writes, "it still marks 'loading' before spawning"
    failed = [w for w in writes if w[:2] == ("status", "failed")]
    assert not failed, (
        f"a restart that did not start the winning load must not write 'failed' over it "
        f"(writes={writes})"
    )
    # Only the pre-spawn set_prior_status (the 'loading' write's flag restore)
    # may happen; the failure path's restore must be dropped too.
    assert [w for w in writes if w[0] == "prior"] == [("prior", "loaded", None)]


@pytest.mark.asyncio
async def test_restart_still_repairs_state_when_the_spawn_fails_on_its_own(monkeypatch):
    """Control — the SAME mark_warming RuntimeError with NO concurrent load
    (generation unchanged, nothing running) still writes 'failed' and restores
    prior_status. This pins the 2026-08-18 11.5-hour fix: the generation guard
    must not swallow a respawn that genuinely failed on its own merits."""

    async def failing_start_engine(*a, **k):
        # No concurrent load: generation unchanged, no engine running.
        raise RuntimeError("cannot mark warming from state ModelState.WARMING: expected LOADING")

    sup = _LoadSup(pid=800137)
    writes, app_state, settings = _wire_restart(
        monkeypatch, sup=sup, start_engine=failing_start_engine
    )

    with pytest.raises(RuntimeError, match="cannot mark warming"):
        await wd._restart(settings, app_state, "m1", None)

    failed = [w for w in writes if w[:2] == ("status", "failed")]
    assert failed, "a respawn that failed on its own must still repair the row back to 'failed'"
    assert "cannot mark warming" in (failed[-1][2] or ""), "the spawn error must survive"
    assert ("prior", "loaded", None) in writes, "the restart flag must be restored"


@pytest.mark.asyncio
async def test_restart_returns_superseded_when_a_force_unload_cancels_the_restart(monkeypatch):
    """#278 — an operator force-unload during an in-flight restart must cancel
    the restart's spawn (as it does to a /load's runner) and leave the model
    alone: no 'failed' write, no exception, just 'superseded'."""
    release = asyncio.Event()

    async def blocking_start_engine(*a, **k):
        await release.wait()  # the spawn is in flight when the operator unloads

    sup = _LoadSup(pid=800137)
    writes, app_state, settings = _wire_restart(
        monkeypatch, sup=sup, start_engine=blocking_start_engine
    )

    restart_task = asyncio.create_task(wd._restart(settings, app_state, "m1", None))
    await wait_until_async(
        lambda: "m1" in sup._load_runners,
        what="the restart to register its load runner",
    )
    # The operator's force-unload cancels the registered spawn, exactly as
    # Supervisor.unload does to a force-unloaded load's runner.
    await sup.unload("m1", force=True)
    await wait_until_async(lambda: restart_task.done(), what="the restart to settle")

    assert not restart_task.cancelled(), "a superseded restart must not look cancelled"
    assert restart_task.exception() is None, "a superseded restart must not raise"
    assert restart_task.result() == "superseded"
    assert not [
        w for w in writes if w[:2] == ("status", "failed")
    ], f"a force-unloaded restart must not write 'failed' (writes={writes})"


@pytest.mark.asyncio
async def test_restart_reraises_its_own_cancellation(monkeypatch):
    """Guard — cancelling _restart ITSELF (a warden shutdown) must still raise
    CancelledError. Swallowing it would let it escape into
    restore_after_warden_restart/restart_crashed_models' `except Exception` and
    run_watchdog_forever would then re-raise, killing the watchdog loop. It must
    also not leave the half-started spawn running."""
    release = asyncio.Event()

    async def blocking_start_engine(*a, **k):
        await release.wait()

    sup = _LoadSup(pid=800137)
    _writes, app_state, settings = _wire_restart(
        monkeypatch, sup=sup, start_engine=blocking_start_engine
    )

    restart_task = asyncio.create_task(wd._restart(settings, app_state, "m1", None))
    await wait_until_async(
        lambda: "m1" in sup._load_runners,
        what="the restart to register its load runner",
    )
    # The watchdog cancels _restart itself, not the load runner.
    restart_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await restart_task

    runner = sup._load_runners.get("m1")
    assert runner is not None, "the restart must have registered its spawn before being cancelled"
    await wait_until_async(lambda: runner.done(), what="the abandoned spawn to be cancelled")
    assert runner.cancelled(), "a cancelled restart must not leave its spawn running"


# --- #278 slice 1: a restart claims the model and re-checks the fresh row ----
#
# The claim stops a concurrent /load or a second restart cold. The fresh-row
# recheck is the second line of defence for movement the claim cannot cover
# (an operator force-unload can still rewrite the row under a restart that
# already holds the claim) — restore/sweep/check_once hand in their own
# predicate so a stale trigger snapshot can never drive a respawn that is no
# longer warranted.


@pytest.mark.asyncio
async def test_restart_backs_off_when_another_writer_already_holds_the_claim(monkeypatch):
    """#278 slice 1 — an operator /load already holds the claim. _restart must
    not unload a load it did not start: it backs off with 'skipped: load in
    flight' and touches neither the engine nor the row."""

    async def no_start_engine(*a, **k):
        raise AssertionError("the refused restart must never reach the spawn")

    sup = _LoadSup(pid=None)
    sup.claim("m1", "load")  # an operator load is in flight
    writes, app_state, settings = _wire_restart(monkeypatch, sup=sup, start_engine=no_start_engine)

    out = await wd._restart(settings, app_state, "m1", None)

    assert out == "skipped: load in flight"
    assert not sup.unloads, "a refused restart must not unload the load it did not start"
    assert not writes, "a refused restart must not write to the row"
    assert "m1" not in sup._load_runners, "a refused restart must not register a runner"


@pytest.mark.asyncio
async def test_restart_skips_when_the_fresh_row_is_no_longer_restartable(monkeypatch):
    """#278 slice 1 — by the time _restart acts, the row is 'loading' with no
    prior_status: a /load is in flight that the trigger's snapshot could not
    see. The fresh-row recheck must veto the unload — no engine touch, no
    write."""

    async def no_start_engine(*a, **k):
        raise AssertionError("a recheck-vetoed restart must never reach the spawn")

    def recheck(row):
        return row.status == "loaded"

    sup = _LoadSup(pid=800137)
    writes, app_state, settings = _wire_restart(
        monkeypatch,
        sup=sup,
        start_engine=no_start_engine,
        row={"prior_status": None, "status": "loading"},
    )

    out = await wd._restart(settings, app_state, "m1", None, recheck=recheck)

    assert out == "skipped: row changed"
    assert not sup.unloads
    assert not writes


@pytest.mark.asyncio
async def test_restart_recheck_uses_the_callers_wants_restart_predicate(monkeypatch):
    """#278 slice 1 — the recheck is the caller's own wants_restart predicate,
    run against a FRESH read. A fresh 'loaded' row with no prior_status is not
    restartable (there is no crash to repair), so the restart skips it: the
    predicate is the authority, not the stale snapshot that triggered the
    restart."""

    async def no_start_engine(*a, **k):
        raise AssertionError("a recheck-vetoed restart must never reach the spawn")

    sup = _LoadSup(pid=800137)
    writes, app_state, settings = _wire_restart(
        monkeypatch,
        sup=sup,
        start_engine=no_start_engine,
        row={"prior_status": None, "status": "loaded"},
    )

    out = await wd._restart(settings, app_state, "m1", None, recheck=wd.wants_restart)

    assert out == "skipped: row changed"
    assert not sup.unloads
    assert not writes
