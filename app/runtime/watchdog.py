"""Engine watchdog: detect a dead engine under a live wrapper, preserve the
evidence, and restore the model to its pre-crash condition.

Why this exists (2026-08-17). ``RuntimeSupervisor._watch_exit`` awaits
``handle.wait()`` on the process the driver spawned — for the local driver that
is the ``vllm serve`` **wrapper**. vLLM v1 runs ``EngineCore`` and the TP workers
underneath it. Four times in one day EngineCore died while the wrapper stayed
alive, so the wait never returned, ``on_exit`` never fired, the row kept
``status='loaded'`` and the proxy forwarded to a dead engine indefinitely:

    794285  uvicorn (warden api)
    800137  └─ vllm serve          <-- ALIVE, RSS 2.1 GB
    801311     └─ resource_tracker   <-- only child left

The wrapper is also an unreliable signal in the other direction: at 07:39 it did
exit, at 09:03 it was still alive seven minutes later. So this watchdog does not
watch processes at all — it asks the engine's own /health endpoint, which is the
same signal the load path already trusts.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from app.config import Settings
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.settings import SettingsRepo

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Imported for annotations only. watchdog is imported BY the
    # supervisor's callers, not the other way round, so a runtime
    # import here would risk a cycle for no benefit.
    from app.runtime.supervisor import Supervisor

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 5.0


# Runtime-tunable knobs, read from the settings KV on every tick so an operator's
# change in the Settings UI takes effect on the next pass rather than at the next
# warden restart. The env-derived Settings values are the fallback/default.
WATCHDOG_KEYS = (
    "watchdog_enabled",
    "watchdog_restore_on_boot",
    "watchdog_interval_s",
    "watchdog_failure_threshold",
    "watchdog_max_restarts",
)


async def live_config(settings) -> dict:
    """Effective watchdog config: KV overrides on top of the env defaults."""
    cfg = {
        "enabled": settings.watchdog_enabled,
        "restore_on_boot": True,
        "interval_s": settings.watchdog_interval_s,
        "failure_threshold": settings.watchdog_failure_threshold,
        "max_restarts": settings.watchdog_max_restarts,
    }
    try:
        async with open_db(settings.db_path) as db:
            kv = await SettingsRepo(db).get_many(list(WATCHDOG_KEYS))
    except Exception:  # noqa: BLE001 — a KV read must never stop the watchdog
        logger.warning("watchdog: could not read settings KV; using defaults")
        return cfg
    if (v := kv.get("watchdog_enabled")) is not None:
        cfg["enabled"] = v in ("1", "true", "True")
    if (v := kv.get("watchdog_restore_on_boot")) is not None:
        cfg["restore_on_boot"] = v in ("1", "true", "True")
    for key, name in (
        ("watchdog_interval_s", "interval_s"),
        ("watchdog_failure_threshold", "failure_threshold"),
        ("watchdog_max_restarts", "max_restarts"),
    ):
        if (v := kv.get(key)) is not None:
            try:
                cfg[name] = int(v)
            except ValueError:
                logger.warning("watchdog: ignoring non-numeric %s=%r", key, v)
    return cfg


async def probe_health(host: str, port: int) -> tuple[bool, str]:
    """Probe the engine's own /health. Returns (healthy, detail-for-the-record).

    Both a non-200 and a connection error count as unhealthy. The detail string
    is stored in the crash report: if a dead engine ever answers 200 here, that
    record is what proves this probe is too weak and must escalate to a real
    generation.
    """
    url = f"http://{host}:{port}/health"
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as c:
            r = await c.get(url)
        return r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:  # noqa: BLE001 — any failure to reach it is unhealthy
        return False, f"{type(exc).__name__}: {exc}"


def _write(path: Path, text: str) -> None:
    path.write_text(text, errors="replace")


def _capture_command(path: Path, argv: list[str]) -> None:
    out = subprocess.run(  # noqa: S603
        argv, capture_output=True, text=True, timeout=15, check=False
    )
    _write(path, out.stdout or out.stderr)


def _copy_tail(src: Path, dst: Path, max_bytes: int) -> None:
    """Copy at most the last ``max_bytes`` of ``src`` into ``dst``.

    The crash is at the end of the file, so the tail is the part anyone reads.
    Copying the whole log made /data occupancy scale with uptime rather than
    with a fixed retention count (see Settings.watchdog_crash_log_bytes).
    """
    size = src.stat().st_size
    with src.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        with dst.open("wb") as out:
            if size > max_bytes:
                out.write(b"[truncated: copied the last %d of %d bytes]\n" % (max_bytes, size))
            shutil.copyfileobj(fh, out)


def capture_evidence(settings, model_id: str, *, report: dict) -> Path | None:
    """Write the crash scene to disk BEFORE recovery destroys it.

    Every step is individually guarded: evidence collection must never prevent a
    restart. A missing artefact is logged, not raised.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    crash_dir = settings.crashes_dir / model_id / stamp
    try:
        crash_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("watchdog: cannot create crash dir for %s", model_id)
        return None

    for name, fn in (
        # The live log is reused by the next load, so the crash tail would
        # otherwise be buried under a fresh startup.
        (
            "engine.log",
            lambda: _copy_tail(
                settings.logs_dir / f"{model_id}.log",
                crash_dir / "engine.log",
                settings.watchdog_crash_log_bytes,
            ),
        ),
        # Shows whether the wrapper outlived the engine — the signature failure.
        (
            "processes.txt",
            lambda: _capture_command(
                crash_dir / "processes.txt", ["ps", "-eo", "pid,ppid,stat,etime,rss,args"]
            ),
        ),
        (
            "meminfo.txt",
            lambda: _write(
                crash_dir / "meminfo.txt",
                Path("/proc/meminfo").read_text()
                # Node-wide and NOT namespaced: settles OOM questions with no node
                # access. It read 0 across all four deaths, refuting the OOM theory.
                + "\n--- /proc/vmstat oom ---\n"
                + "\n".join(
                    ln for ln in Path("/proc/vmstat").read_text().splitlines() if "oom" in ln
                ),
            ),
        ),
        ("gpu.txt", lambda: _capture_command(crash_dir / "gpu.txt", ["nvidia-smi"])),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — best effort, never fatal
            logger.warning("watchdog: could not capture %s: %s", name, exc)

    try:
        _write(crash_dir / "report.json", json.dumps(report, indent=2, default=str))
    except Exception:  # noqa: BLE001
        logger.exception("watchdog: could not write report.json")
    return crash_dir


def prune_crash_dirs(settings, model_id: str) -> None:
    """Keep only the newest N crash dirs so a crash loop cannot fill the volume."""
    root = settings.crashes_dir / model_id
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    except OSError:
        return
    for stale in dirs[: max(0, len(dirs) - settings.watchdog_crash_keep)]:
        try:
            shutil.rmtree(stale)
        except OSError:
            logger.warning("watchdog: could not prune %s", stale)


class RestartBudget:
    """Crash-loop guard. An engine that dies on startup must not be restarted
    forever: that burns GPU cycles and buries the first crash under identical
    reports."""

    def __init__(self, max_restarts: int, window_s: float) -> None:
        self._max = max_restarts
        self._window = window_s
        self._events: dict[str, list[float]] = {}

    def allow(self, model_id: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        events = [t for t in self._events.get(model_id, []) if now - t < self._window]
        self._events[model_id] = events
        return len(events) < self._max

    def record(self, model_id: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._events.setdefault(model_id, []).append(now)

    def reset(self, model_id: str) -> None:
        self._events.pop(model_id, None)

    def reconfigure(self, max_restarts: int) -> None:
        """Apply a live settings change without losing the crash history."""
        self._max = max_restarts


def _gpu_compute_apps() -> list[tuple[int, int]]:
    """[(pid, used_MiB)] from nvidia-smi. Empty on any failure."""
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return []
    apps = []
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            apps.append((int(parts[0]), int(parts[1])))
    return apps


def _cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        )
    except OSError:
        return ""


def _ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _sid(pid: int) -> int:
    """Session id of ``pid``; raises ``OSError`` when it cannot be read.

    #218 — this is the one fact about an orphaned worker that outlives its
    process tree. ``LocalSubprocessDriver.spawn`` starts every engine with
    ``start_new_session=True``, so the ``vllm serve`` wrapper is a session
    LEADER and its session id equals its own pid. EngineCore and the
    ``VLLM::Worker_TP*`` processes are forked underneath it, inherit that
    session id, and keep it after the wrapper is dead — which is exactly the
    state the reaper has to reason about. Nothing else in the corpse still
    names the warden that started it: the parent chain is broken (that is what
    makes it an orphan) and the cmdline is just text anyone can produce.

    ``app.system.pid_attribution.read_pgid`` is a second, independent read of
    the same ``start_new_session=True`` fact via the process GROUP id. Session
    id is the stronger of the two — a ``setpgid`` moves a process out of its
    group but nothing short of ``setsid`` moves it out of its session — so the
    two want unifying onto this one; not in #218's scope, but do not add a
    third.

    The read fails in two ways the caller must tell apart in its log:
    ``ProcessLookupError`` (ESRCH) means the pid exited between the nvidia-smi
    read and here — a harmless race; ``PermissionError`` (EPERM) means the pid
    is alive in another PID namespace — a boundary the warden cannot cross.
    Both mean "not attributable", so the caller declines either way — but the
    2026-09-30 incident log said only ``session=None`` and an operator could
    not tell which of the two they were looking at (#278 slice 3).
    """
    return os.getsid(pid)


def _has_live_ancestor(pid: int, live_pids: set[int]) -> bool:
    """Walk up to see whether this process still belongs to a running engine.

    The container shares the host PID namespace, so orphans are reparented to the
    container shim rather than PID 1 — "ppid == 1" is NOT a valid orphan test here.
    Ancestry against the supervisor's live handles is.
    """
    seen = 0
    cur = _ppid(pid)
    while cur and cur > 1 and seen < 20:
        if cur in live_pids:
            return True
        cur = _ppid(cur)
        seen += 1
    return False


@dataclass(frozen=True)
class EngineOwnership:
    """What this warden can *prove* about which GPU processes are its own (#218).

    ``sessions`` holds the session ids of engines this warden started — i.e.
    the pids of the ``vllm serve`` wrappers it spawned, live or already exited,
    because the local driver makes every wrapper a session leader (see
    :func:`_sid`). Membership of one of those sessions is the ownership signal:
    a process is ours only if it was forked underneath a wrapper we started.

    ``live_pids`` holds the wrapper pids the supervisor still has a running
    handle for. Those are the "do not touch" set — their workers belong to a
    model that is serving right now.

    An EMPTY ``sessions`` set means "this warden cannot attribute anything",
    and the reaper then kills nothing at all. That is the deliberate choice
    #218 asks for. After a warden restart the in-memory handle table is gone
    and ``RuntimeRepo.clear_all()`` has already wiped the persisted pids, so
    the previous generation's session ids are unrecoverable — and a pid read
    back from disk after a restart names whatever the kernel has since handed
    that number to, which is how you kill a stranger. Missing an orphan costs
    one failed reload and an operator-visible "Free memory on device cuda:0"
    error; killing another tenant's engine costs them their job with no
    warning and no way back. We take the recoverable failure.
    """

    sessions: frozenset[int] = frozenset()
    live_pids: frozenset[int] = frozenset()


# model_id -> [(wrapper pid, monotonic timestamp it was recorded), ...].
#
# #218 — the ownership signal for an orphaned worker is its session id, which
# equals the pid of the wrapper that fathered it. But the supervisor drops the
# handle the instant the wrapper exits (``_watch_exit`` pops it in the same
# breath as flipping the row to 'failed'), so on the exit-path sweep —
# the commonest crash shape here — ``sup.get_pid`` already returns None and
# that pid is gone before anything gets a chance to reap. This ledger keeps it.
#
# A model may hold MORE THAN ONE wrapper pid at once. A crash, a fast
# replacement spawn, and a slow exit-watcher leave the first wrapper dead
# with its workers still holding VRAM while the second wrapper is already
# registered — and a wrapper that dies within a single watchdog interval is
# never seen by a tick-based ``remember_wrapper_pids`` sample at all (the
# 2026-09-30 double spawn). So pids are ALSO recorded at spawn time, through
# the supervisor's ``on_wrapper_spawn`` hook, and EVERY fresh pid for a model
# goes into ``engine_ownership`` on a restart. The ledger is the only copy of
# a pid the handle table has already forgotten.
#
# Deliberately in-process and NOT persisted: a pid is only meaningful for the
# lifetime of the process table that issued it.
_WRAPPER_PIDS: dict[str, list[tuple[int, float]]] = {}

# How long a remembered wrapper pid may be trusted. Pid numbers wrap, and a
# model that failed and was never restarted would otherwise leave its pid in
# the ledger indefinitely, until the kernel hands that number to a stranger who
# then looks like one of ours. A restart happens within a tick or two of the
# crash, so a quarter of an hour is generous for the legitimate use and short
# enough that wraparound is not a realistic worry.
_WRAPPER_PID_TTL_S = 900.0

# A model is (re)spawned at most once per load, and a load needs a crash plus
# a sweep interval, so a handful of pids per model covers every shape. The cap
# only guards a churning model from pinning ledger memory.
_WRAPPER_PID_CAP = 8


def _fresh_entries(model_id: str) -> list[tuple[int, float]] | None:
    """The model's ledger entries still inside the TTL, or None if it has none.

    Expired entries are pruned on the way out and the model's key dropped when
    its last entry expires, so a never-restarted model leaves no pid behind.
    """
    entries = _WRAPPER_PIDS.get(model_id)
    if not entries:
        return None
    now = time.monotonic()
    fresh = [e for e in entries if now - e[1] <= _WRAPPER_PID_TTL_S]
    if not fresh:
        _WRAPPER_PIDS.pop(model_id, None)
        return None
    if len(fresh) < len(entries):
        _WRAPPER_PIDS[model_id] = fresh
    return fresh


def record_wrapper_pid(model_id: str, pid: int | None) -> None:
    """Remember that this warden spawned wrapper ``pid`` for ``model_id``.

    Called at spawn time by the supervisor (``on_wrapper_spawn``) so a wrapper
    that dies before the next tick is still in the ledger, and on every tick
    by ``remember_wrapper_pids`` for the pid the supervisor still holds.
    Re-recording a pid refreshes its stamp instead of duplicating it.
    """
    if not pid:
        return
    now = time.monotonic()
    entries = _WRAPPER_PIDS.get(model_id, [])
    kept = [(p, t) for p, t in entries if p != pid and now - t <= _WRAPPER_PID_TTL_S]
    kept.append((pid, now))
    _WRAPPER_PIDS[model_id] = kept[-_WRAPPER_PID_CAP:]


def remember_wrapper_pids(sup: Supervisor) -> None:
    """Snapshot the supervisor's live wrapper pids into the session ledger."""
    for model_id in list(sup._handles):  # noqa: SLF001 — no public "all handles" read
        record_wrapper_pid(model_id, sup.get_pid(model_id))


def forget_wrapper_pids(keep: set[str]) -> None:
    """Drop ledger entries for models that no longer exist."""
    for gone in [m for m in _WRAPPER_PIDS if m not in keep]:
        _WRAPPER_PIDS.pop(gone, None)


def fresh_wrapper_pids(model_id: str) -> list[int]:
    """Every wrapper pid still fresh enough to attribute sessions to, newest first."""
    entries = _fresh_entries(model_id)
    return [pid for pid, _ in reversed(entries)] if entries else []


def last_wrapper_pid(model_id: str) -> int | None:
    """Last wrapper pid observed for ``model_id``, if still fresh enough to act on."""
    entries = _fresh_entries(model_id)
    return entries[-1][0] if entries else None


def engine_ownership(
    sup: Supervisor, *, also_sessions: Iterable[int | None] = ()
) -> EngineOwnership:
    """Read the supervisor's handle table into an :class:`EngineOwnership`.

    ``also_sessions`` carries wrapper pids the supervisor no longer holds a
    handle for — the engine ``_restart`` just force-unloaded, or one recovered
    from the ledger. Those are precisely the sessions whose workers are still
    holding VRAM, so leaving them out would make the reaper a no-op on the only
    case it exists for. Callers must sample them BEFORE the unload:
    ``Supervisor.unload`` pops the handle in a ``finally``, after which
    ``get_pid`` returns None and the session id is unrecoverable.
    """
    sessions: set[int] = {p for p in also_sessions if p}
    live: set[int] = set()
    # Private read: there is no public accessor that also includes handles
    # whose process has exited, and those are exactly the ones whose workers
    # may still be holding VRAM. ``parent_pid_to_model()`` filters them out.
    for model_id, handle in sup._handles.items():  # noqa: SLF001
        pid = sup.get_pid(model_id)
        if not pid:
            continue
        sessions.add(pid)
        if getattr(handle, "returncode", None) is None:
            live.add(pid)
    return EngineOwnership(frozenset(sessions), frozenset(live))


def _classify_gpu_holders(
    owned: EngineOwnership,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Partition the GPU compute apps into (reapable, declined).

    *Reapable*: forked under a wrapper this warden spawned, and that wrapper is
    gone — a true orphan of ours.
    *Declined*: holds VRAM and is NOT provably ours. Returned, rather than
    silently dropped, so the caller can say out loud what it left alone and why;
    an operator staring at a failed reload needs to see that something else on
    the box is sitting on the memory.
    Processes belonging to a live engine of ours are in neither list — they are
    simply fine, and reporting them as "declined" would bury the interesting case.
    """
    reapable: list[tuple[int, str]] = []
    declined: list[tuple[int, str]] = []
    live = set(owned.live_pids)
    for pid, used in _gpu_compute_apps():
        cmd = _cmdline(pid)
        try:
            sid = _sid(pid)
        except OSError as exc:
            # Gone (ESRCH) or unreadable (EPERM): not attributable, and never
            # a kill — but say WHICH, because a bare session=None after a
            # multi-gigabyte refusal sent the 2026-09-30 operator guessing
            # between a pid-namespace boundary and a race (#278 slice 3).
            code = errno.errorcode.get(exc.errno, str(exc.errno))
            declined.append(
                (pid, f"session=unreadable ({code}) not one of ours; {used} MiB; {cmd[:80]}")
            )
            continue
        if sid is None or sid not in owned.sessions:
            # #218 — this test used to be ``"vllm" in cmd.lower()``, which is
            # not an ownership signal at all. A second vllm-warden on the same
            # box, another team's vLLM container, and a researcher's
            # ``python -m vllm.entrypoints.openai.api_server`` all matched it,
            # and all were SIGKILLed the moment this warden recovered a crashed
            # model. Session membership is a fact about who forked the process;
            # a substring of argv is a fact about nothing.
            declined.append((pid, f"session={sid} not one of ours; {used} MiB; {cmd[:80]}"))
            continue
        if sid in live or _has_live_ancestor(pid, live):
            continue  # belongs to an engine that is still running
        if "VLLM::" not in cmd and "vllm" not in cmd.lower():
            # Defence in depth, NOT the ownership test. Session ids are pids and
            # pid numbers wrap, so a session we recorded could in principle be
            # re-created by an unrelated leader. Requiring the process to also
            # look like a vLLM engine keeps that rare collision from becoming a
            # kill.
            declined.append(
                (
                    pid,
                    f"in our session {sid} but does not look like vLLM; " f"{used} MiB; {cmd[:80]}",
                )
            )
            continue
        reapable.append((pid, cmd))
    return reapable, declined


def reap_orphan_gpu_holders(owned: EngineOwnership) -> list[int]:
    """SIGKILL *this warden's* vLLM workers that still hold VRAM with no live engine.

    ``unload(force=True)`` only reaps the process the driver spawned — the
    ``vllm serve`` wrapper. The TP workers are its grandchildren and they SURVIVE
    it: observed 2026-08-17 with four ``VLLM::Worker_TP*`` processes holding
    10550 MiB each long after EngineCore and the wrapper were gone. Any reload
    then dies with "Free memory on device cuda:0 (4.98/15.6 GiB) is less than
    desired (14.82 GiB)", which is a wasted restart and a wasted crash budget.

    A process is killed only when all three hold: it holds GPU memory, it lives
    in a session rooted at a wrapper THIS warden spawned (``owned.sessions``),
    and that wrapper is no longer running. Anything it cannot positively
    attribute it leaves alone and logs — see :class:`EngineOwnership` for why
    that trade is the right way round.
    """
    reapable, declined = _classify_gpu_holders(owned)
    if declined:
        logger.warning(
            "watchdog: leaving %d GPU process(es) alone — not provably ours " "(#218): %s",
            len(declined),
            "; ".join(f"pid {pid}: {why}" for pid, why in declined),
        )
    killed = []
    for pid, cmd in reapable:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            logger.warning("watchdog: killed orphaned GPU holder %s (%s)", pid, cmd[:60])
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("watchdog: could not kill orphan %s: %s", pid, exc)
    return killed


async def wait_for_gpu_release(
    owned: EngineOwnership, *, timeout_s: float = 30.0, interval_s: float = 2.0
) -> bool:
    """Wait until none of *our* orphaned processes still holds VRAM. Freeing is
    not instant after SIGKILL, and reloading too early reproduces the very
    failure we are recovering from.

    Only processes the reaper would kill are waited on. A foreign GPU holder is
    not going to release anything because we asked nicely, so blocking on one
    would just add 30s to a reload that is going to fail either way — the
    declined-kill log line is what tells the operator about it.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        reapable, _ = _classify_gpu_holders(owned)
        if not reapable:
            return True
        await asyncio.sleep(interval_s)
    return False


def _restart_superseded(
    sup: Supervisor, model_id: str, generation_at_start: int, spawned_generation: int | None
) -> bool:
    """#278 — True when a newer load has since owned ``model_id`` so this
    restart's terminal status write must be dropped, not applied.

    A restart that fails (or is force-unloaded) must never write 'failed' over a
    load it did not start. Two shapes of "lost the race":

    * The spawn actually ran and bumped the generation (``spawned_generation``
      is set by start_engine's on_spawn). Then we own that generation and a
      mismatch against get_generation() means someone loaded again on top of us.
    * The spawn never completed (it raised before spawning, or was cancelled
      mid-flight). Then ``spawned_generation`` is None and we fall back to the
      start-from generation: it moved, OR an engine we did not spawn is running.
    """
    if spawned_generation is not None:
        return sup.get_generation(model_id) != spawned_generation
    return sup.get_generation(model_id) != generation_at_start or sup.is_running(model_id)


async def _restart(
    settings: Settings,
    app_state: Any,
    model_id: str,
    overrides: dict[str, Any] | None,
    *,
    recheck: Callable[[Any], bool] | None = None,
) -> str:
    """Claim the model, then force-unload the corpse and reload through the
    shared load path.

    The claim (#278 slice 1) is what stops this from unloading an operator's
    in-flight /load (or a sibling restart): it is taken before ANY engine or
    row touch, and a refused claim returns without doing either. ``recheck``,
    when the caller passes one, re-runs the caller's restart predicate against
    a FRESH row read after the claim, so a trigger snapshot that has gone stale
    (an operator force-unload rewrote the row under us) cannot drive a respawn
    that is no longer warranted. ``None`` — the default, used by the stress
    harness — skips the re-read and acts on the row as it was when the caller
    saw it.
    """
    sup = app_state.supervisor
    held = sup.claim(model_id, "watchdog")
    if held is not None:
        logger.warning("watchdog: restart of %s skipped — %s in flight", model_id, held)
        return f"skipped: {held} in flight"
    try:
        if recheck is not None:
            async with open_db(settings.db_path) as db:
                fresh = await ModelRepo(db).get(model_id)
            if fresh is None:
                return "model row vanished"
            if not recheck(fresh):
                logger.info(
                    "watchdog: restart of %s skipped — the row no longer needs it",
                    model_id,
                )
                return "skipped: row changed"
        # Sampled BEFORE the unload, and falling back to the ledger when the
        # wrapper already exited on its own: these pids are the session ids of
        # the very workers the reaper has to find (#218), and ``unload`` pops
        # the handle in a ``finally``, after which they are unrecoverable.
        # EVERY fresh ledger entry goes in, not just the newest: a double
        # spawn leaves the first wrapper's workers in the FIRST wrapper's
        # session, and that pid is only in the ledger (#278 slice 3).
        doomed_pids = [sup.get_pid(model_id), *fresh_wrapper_pids(model_id)]
        # force=True is the step a plain reload misses: the stale wrapper still
        # holds its port and GPU claims, so a fresh spawn would fail on both.
        await sup.unload(model_id, force=True)

        # force-unload kills only the wrapper the driver spawned; the TP workers
        # are its grandchildren and survive, still holding all the VRAM.
        owned = engine_ownership(sup, also_sessions=tuple(doomed_pids))
        killed = reap_orphan_gpu_holders(owned)
        if killed and not await wait_for_gpu_release(owned):
            logger.error("watchdog: VRAM still held after killing %s; reloading anyway", killed)

        async with open_db(settings.db_path) as db:
            model = await ModelRepo(db).get(model_id)
        if model is None:
            return "model row vanished"

        # Imported here: routes_api imports heavy siblings, and a module-level
        # import would make the runtime package depend on the API package.
        from app.models.routes_api import start_engine  # noqa: PLC0415

        # The load ROUTE marks the row 'loading' before calling start_engine;
        # this path must do the same or the row reads 'loaded' for the whole
        # reload and lies to the UI. Deliberately here and not inside
        # start_engine: an extra write at the top of the shared path serializes
        # against SQLite's 30s busy_timeout and can delay every engine spawn.
        # Captured BEFORE the 'loading' write below, which clears prior_status
        # by design (see ModelRepo.update_status). Without this the failure
        # path cannot put the flag back and the row loses its only claim to a
        # retry.
        was = model.prior_status or "loaded"

        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(model_id, "loading")
            # ...and put it straight back, so the flag is true for the WHOLE
            # restore attempt rather than only at its edges (#236). While this
            # await runs the row reads 'loading' with no prior_status, which is
            # indistinguishable from an operator's first-ever load of a model —
            # and boot reconciliation now demotes THAT to 'pulled' rather than
            # retrying it forever. Keeping the flag lit is what lets a warden
            # restart mid-restore still restore, instead of quietly leaving a
            # model that was serving parked at 'pulled'.
            await ModelRepo(db).set_prior_status(model_id, was)

        port = app_state.port_allocator.allocate()
        # #278 — run the respawn as a REGISTERED load runner, exactly as the
        # load route does, so an operator's force-unload cancels it the way it
        # cancels a manual load instead of leaving it to run its health timeout
        # against an engine the unload already tore down. The generation it
        # started from (read before the spawn) and the one it actually spawned
        # as (reported by start_engine's on_spawn) let us tell "this attempt
        # lost the race to a newer load" apart from "this attempt genuinely
        # failed", so a restart never writes 'failed' over a load it did not
        # start.
        generation_at_start = sup.get_generation(model_id)
        spawned_generation: list[int | None] = [None]

        def _on_spawn(gen: int) -> None:
            spawned_generation[0] = gen

        load_task = asyncio.create_task(
            start_engine(
                settings,
                sup,
                app_state.port_allocator,
                model,
                port,
                overrides,
                on_spawn=_on_spawn,
            )
        )
        sup.register_load_runner(model_id, load_task)
        try:
            await load_task
        except asyncio.CancelledError:
            # Two very different cancellations reach here. If our spawn task
            # was cancelled (an operator force-unloaded the model) while
            # _restart ITSELF is not being torn down, the model is now owned by
            # someone else: write no terminal status, just report that we were
            # superseded. If _restart itself is being cancelled (a warden
            # shutdown), that cancellation must keep propagating — swallowing
            # it would escape into the callers' `except Exception` and kill the
            # watchdog loop.
            current = asyncio.current_task()
            if load_task.cancelled() and current is not None and current.cancelling() == 0:
                logger.warning(
                    "watchdog: restart of %s superseded by a newer load; not writing state",
                    model_id,
                )
                app_state.port_allocator.release(port)
                return "superseded"
            if not load_task.done():
                load_task.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised after repairing state
            if _restart_superseded(sup, model_id, generation_at_start, spawned_generation[0]):
                # A newer load owns model_id by the time our spawn errors: it
                # moved the generation (or left an engine we did not spawn
                # running). A 'failed' write here would clobber its row with
                # this superseded attempt's stale result, so drop it — and the
                # flag restore with it, since that belongs to the failure path
                # we are abandoning.
                logger.warning(
                    "watchdog: restart of %s superseded by a newer load (%s); not writing state",
                    model_id,
                    exc,
                )
                app_state.port_allocator.release(port)
                return "superseded"
            # A spawn that fails here would otherwise strand the row in
            # 'loading' FOREVER: nothing resets it, on_exit never fires (no
            # process was ever supervised long enough to exit), and no recovery
            # path inspects 'loading'. Prod sat dead 11.5h in exactly this
            # state on 2026-08-18 -- the sweep restarted a crashed engine, the
            # respawn produced no engine log at all, and the row was still
            # 'loading' when the pod restarted hours later.
            #
            # Put it back to 'failed' and restore the restart flag, so the next
            # sweep retries it under the RestartBudget rather than the row
            # silently leaving the recoverable set.
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(
                    model_id, "failed", last_error=f"restart failed to spawn: {exc}"
                )
                await ModelRepo(db).set_prior_status(model_id, was)
            app_state.port_allocator.release(port)
            raise

        async with open_db(settings.db_path) as db:
            row = await ModelRepo(db).get(model_id)
        return row.status if row else "unknown"
    finally:
        # The claim is ours from the top of this function, however it ended —
        # success, a refused row, a superseded spawn, or a warden shutdown.
        sup.release_claim(model_id, "watchdog")


def wants_restart(row, leases=None) -> bool:
    """True when a row is a crashed engine that WAS serving, so restart it.

    One predicate, deliberately, for both ways a serving engine can end up
    dead: the warden restarted underneath it (boot reconciliation sets
    prior_status), or the engine died while the warden kept running
    (``on_exit`` sets it). Those used to be handled by different mechanisms
    and only the first one worked.

    This reads ``prior_status``, NOT ``last_error``. The old code compared
    last_error against an exact sentence, so any diagnosis written on the
    crash path displaced the sentinel and silently disabled recovery -- the
    2026-08-18 outage, where the model sat dead for 1h43m advertising a
    completely unrelated "requires trust_remote_code" message.

    ``loading`` counts too, and that is safe for a non-obvious reason: the
    ONLY writer of ``prior_status='loading'`` is boot reconciliation, i.e.
    "the warden restarted while this model was mid-load". ``on_exit`` sets
    ``prior_status`` exclusively for rows that were ``loaded``, so a model
    that failed on its own merits during a user-initiated load never gets the
    flag and is never retried here. Bad configs still stay down for a human.

    Without ``loading``, a model interrupted while it was being auto-restarted
    became permanently unrecoverable -- which is exactly how prod stayed dead
    for 11.5 hours on 2026-08-18/19: the engine crashed, the sweep restarted
    it, the respawn stalled in ``loading``, the pod later restarted, and the
    row was then ineligible forever.

    ``pulling`` and ``unloading`` are deliberately excluded: a half-finished
    pull is not a serving model, and an unload is an operator's stated intent.

    ``leases`` is the stress-test registry (``app/stress/lease.py``), optional
    so every existing caller is unaffected. A model under an unexpired stress
    lease is skipped: its run crashed the engine on purpose, is already probing
    far more tightly than this loop's ~90s detection latency, and will restart
    it itself. Without this, the sweep and the runner both call ``_restart``,
    which force-unloads unconditionally, and whichever loses the race destroys
    the other's freshly loaded engine -- while both charge the restart budget
    that leaves the model dead after three.

    The check belongs HERE rather than in ``check_once`` because this predicate
    is what both recovery paths consult. ``check_once`` only ever sees rows
    whose status is ``loaded``, and a crash sets ``failed`` first, so guarding
    it would guard the path that does not fire.
    """
    if leases is not None and leases.is_held(row.id):
        return False
    return row.status == "failed" and (row.prior_status or "") in (
        "loaded",
        "loading",
    )


async def restore_after_warden_restart(settings, app_state) -> list[str]:
    """Reload models that were serving when the warden itself went down.

    The engine is a child of the warden process, so a warden restart takes the
    engine with it and boot reconciliation marks the row `failed`. Nothing then
    brings it back: observed 2026-08-17, when a routine redeploy left the model
    unloaded and every request 404'd until a human loaded it by hand.

    Only rows marked as having been *loaded* are restored. A model that was
    mid-pull, or that a human deliberately unloaded, is left alone.

    Note (#218): a restart also wipes the wrapper-pid ledger, so any VRAM still
    held by the PREVIOUS generation's workers cannot be attributed to us and is
    therefore not reaped here — the reload may fail on free memory and say so.
    That is the intended trade: a fresh warden has no way to tell its own
    predecessor's orphans from another tenant's engine, and guessing means
    killing strangers.
    """
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    leases = getattr(app_state, "stress_leases", None)
    candidates = [r for r in rows if wants_restart(r, leases=leases)]
    restored = []
    for row in candidates:
        logger.warning("watchdog: %s was loaded before the warden restarted — restoring", row.id)
        try:
            outcome = await _restart(
                settings,
                app_state,
                row.id,
                None,
                # The candidate list is a snapshot; re-veto against the fresh
                # row so a row that moved under us is not restarted.
                recheck=lambda r: wants_restart(r, leases=leases),
            )
            restored.append(f"{row.id}:{outcome}")
        except Exception:  # noqa: BLE001 — one bad model must not block the others
            logger.exception("watchdog: could not restore %s", row.id)
    return restored


async def restart_crashed_models(settings, app_state, budget) -> list[str]:
    """Restart engines that died while the warden stayed up.

    ``check_once`` only probes rows the DB believes are ``loaded``. When the
    engine's own wrapper exits, the supervisor's ``_watch_exit`` gets there
    first and flips the row to ``failed`` -- at which point the health-probe
    loop stops looking at it forever and NOTHING restarts it. That is the gap
    that kept prod down twice on 2026-08-18 (09:15 and 13:11), both times
    until a human noticed.

    Shares the SAME ``RestartBudget`` as the probe path on purpose: a model
    flapping between the two paths must still hit one combined ceiling rather
    than getting a fresh allowance from each.

    Evidence is captured BEFORE the restart, because the restart overwrites
    the live engine log with a fresh startup. The probe path already did this;
    the exit path did not, which is why /data/crashes was empty after a real
    crash.
    """
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    # NB the wrapper-pid ledger (#218) is refreshed by ``check_once``, which
    # ``run_watchdog_forever`` always runs immediately before this sweep. It is
    # not refreshed here on purpose: by the time a row reaches this path its
    # wrapper has usually already exited and the supervisor has dropped the
    # handle, so re-reading it now would find nothing the earlier pass did not.
    out = []
    leases = getattr(app_state, "stress_leases", None)
    for row in (r for r in rows if wants_restart(r, leases=leases)):
        if not budget.allow(row.id):
            logger.error("watchdog: %s exceeded the restart budget — leaving it failed", row.id)
            # Drop the flag so the sweep stops retrying it every tick and the
            # row settles into a state an operator has to look at.
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).set_prior_status(row.id, None)
            out.append(f"{row.id}:budget-exhausted")
            continue
        logger.warning("watchdog: %s died while serving — restarting", row.id)
        capture_evidence(
            settings,
            row.id,
            report={
                "model_id": row.id,
                "detected_by": "exit-path sweep",
                "last_error": row.last_error,
                "prior_status": row.prior_status,
            },
        )
        prune_crash_dirs(settings, row.id)
        budget.record(row.id)
        try:
            outcome = await _restart(
                settings,
                app_state,
                row.id,
                None,
                # The candidate list is a snapshot; re-veto against the fresh
                # row so a row that moved under us is not restarted.
                recheck=lambda r: wants_restart(r, leases=leases),
            )
            out.append(f"{row.id}:{outcome}")
        except Exception:  # noqa: BLE001 — one bad model must not block the others
            logger.exception("watchdog: could not restart %s", row.id)
    return out


async def check_once(
    settings, app_state, state: dict, budget: RestartBudget, cfg: dict | None = None
) -> None:
    """One watchdog pass over every model the DB believes is loaded."""
    cfg = cfg or {
        "failure_threshold": settings.watchdog_failure_threshold,
        "max_restarts": settings.watchdog_max_restarts,
    }
    threshold = cfg["failure_threshold"]
    sup = app_state.supervisor
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()

    # Refresh the wrapper-pid ledger while the handles are still there — after
    # a crash the supervisor has already dropped them, and that pid is the only
    # ownership signal an orphaned worker carries (#218). Pruned against the
    # full row set so a deleted model does not leave a pid behind to be
    # confused with whatever the kernel reissues that number to.
    remember_wrapper_pids(sup)
    forget_wrapper_pids({r.id for r in rows})

    # 'loading' is deliberately skipped: the load path owns that window and has
    # its own timeout. Probing it would restart a model that is merely slow.
    loaded = [r for r in rows if r.status == "loaded"]
    live = {r.id for r in loaded}
    for gone in [k for k in state if k not in live]:
        state.pop(gone, None)

    leases = getattr(app_state, "stress_leases", None)
    for row in loaded:
        # A stress run deliberately provokes exactly what this loop hunts: an
        # engine that stops answering. THIS is the path that fires for it --
        # the row still reads 'loaded' because nothing set it 'failed', so
        # wants_restart (which both recovery paths consult) is never reached
        # here and its lease guard never applies. Without this check the run
        # and the watchdog both call _restart, which force-unloads
        # unconditionally, and whichever loses the race destroys the other's
        # engine while charging a restart budget that leaves the model dead
        # after three.
        if leases is not None and leases.is_held(row.id):
            continue
        port = sup.get_port(row.id)
        if port is None:
            continue  # no runtime record yet; nothing to probe
        host = sup.get_host(row.id) or "127.0.0.1"
        healthy, detail = await probe_health(host, port)
        if healthy:
            if state.get(row.id):
                logger.info("watchdog: %s recovered on its own", row.id)
            state[row.id] = 0
            budget.reset(row.id)
            continue

        state[row.id] = state.get(row.id, 0) + 1
        fails = state[row.id]
        logger.warning(
            "watchdog: %s health probe failed (%d/%d): %s",
            row.id,
            fails,
            threshold,
            detail,
        )
        if fails < threshold:
            continue

        state[row.id] = 0
        handle = sup._handles.get(row.id)  # noqa: SLF001 — exit code is the evidence
        overrides = sup.get_overrides(row.id)  # preserves an auto-capped context
        report = {
            "model_id": row.id,
            "served_model_name": row.served_model_name,
            "detected_at": datetime.now(UTC).isoformat(),
            "consecutive_failures": fails,
            # If a dead engine ever answers 200 here, this field is the proof that
            # /health is too weak a probe and it must escalate to a generation.
            "health_probe": detail,
            # Negative => killed by that signal. This is the discriminator between
            # an external kill and a native crash, and it is exactly what was
            # missing when the root cause could not be determined.
            "wrapper_returncode": getattr(handle, "returncode", None),
            "wrapper_pid": sup.get_pid(row.id),
            "config": {
                "hf_repo": row.hf_repo,
                "gpu_indices": row.gpu_indices,
                "tensor_parallel_size": row.tensor_parallel_size,
                "max_model_len": row.max_model_len,
                "gpu_memory_utilization": row.gpu_memory_utilization,
                "extra_args": row.extra_args,
            },
            "effective_overrides": overrides,
        }

        if not budget.allow(row.id):
            report["restart"] = "refused: crash-loop guard"
            capture_evidence(settings, row.id, report=report)
            prune_crash_dirs(settings, row.id)
            msg = (
                f"engine died {cfg['max_restarts']}+ times within "
                f"{settings.watchdog_restart_window_s:.0f}s — automatic restart "
                f"disabled; see {settings.crashes_dir / row.id}"
            )
            logger.error("watchdog: %s", msg)
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(row.id, "failed", last_error=msg)
            continue

        # Evidence BEFORE recovery — the restart destroys the scene.
        crash_dir = capture_evidence(settings, row.id, report=report)
        budget.record(row.id)
        logger.error(
            "watchdog: %s engine is dead (%s); evidence in %s — restarting",
            row.id,
            detail,
            crash_dir,
        )
        try:
            outcome = await _restart(
                settings,
                app_state,
                row.id,
                overrides,
                # The trigger was 'this loaded row is not answering'; if the
                # row is no longer 'loaded' the respawn is no longer owed.
                recheck=lambda r: r.status == "loaded",
            )
        except Exception as exc:  # noqa: BLE001 — a failed restart must not kill the loop
            logger.exception("watchdog: restart of %s failed", row.id)
            outcome = f"restart raised: {type(exc).__name__}: {exc}"
        report["restart"] = outcome
        if crash_dir is not None:
            try:
                (crash_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
            except OSError:
                logger.warning("watchdog: could not update report.json")
        prune_crash_dirs(settings, row.id)
        logger.warning("watchdog: %s restart finished with status=%s", row.id, outcome)


async def _sweep_engine_logs(app_state: Any) -> None:
    """Rotate engine logs that grew past the cap mid-run (#216).

    The in-container driver's spawn-time rename only works while no writer
    holds the file; a long-lived engine appends to the same O_APPEND fd for
    weeks, so the sweep copy-truncates instead. It is housekeeping, not
    health probing — it runs even when the watchdog is disabled — and it
    must never stop the loop: a failure is logged and retried next tick.
    """
    try:
        supervisor = app_state.supervisor
        # Build the live set on the event loop: the handle table is owned
        # by the loop and a load or unload racing a worker-thread read
        # would raise "dictionary changed size during iteration". Only
        # the driver's file work goes to the thread.
        live = supervisor.live_engine_ids()
        await asyncio.to_thread(supervisor.sweep_engine_logs, live)
    except Exception:
        logger.exception("watchdog: engine log sweep failed; continuing")


async def run_watchdog_forever(settings, app_state) -> None:
    state: dict[str, int] = {}
    cfg = await live_config(settings)
    budget = RestartBudget(cfg["max_restarts"], settings.watchdog_restart_window_s)
    logger.info(
        "engine watchdog started (every %ss, %s consecutive failures to act, "
        "restore-on-boot=%s)",
        cfg["interval_s"],
        cfg["failure_threshold"],
        cfg["restore_on_boot"],
    )
    if cfg["restore_on_boot"]:
        try:
            restored = await restore_after_warden_restart(settings, app_state)
            if restored:
                logger.warning("watchdog: restored after warden restart: %s", restored)
        except Exception:
            logger.exception("watchdog: post-restart restore failed; continuing")
    while True:
        try:
            cfg = await live_config(settings)
            budget.reconfigure(cfg["max_restarts"])
            # In-run log rotation runs on EVERY tick, even with health
            # probing disabled: a disabled watchdog does not mean the
            # engines stop logging.
            await _sweep_engine_logs(app_state)
            if cfg["enabled"]:
                await check_once(settings, app_state, state, budget, cfg)
                # Runs AFTER check_once so a model the probe path just gave up
                # on is picked up on the next tick rather than restarted twice
                # in the same pass. Covers the engines that died with their
                # wrapper, which check_once can never see (it only looks at
                # rows the DB believes are 'loaded').
                swept = await restart_crashed_models(settings, app_state, budget)
                if swept:
                    logger.warning("watchdog: restarted crashed engines: %s", swept)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("watchdog iteration failed; continuing")
        await asyncio.sleep(cfg["interval_s"])
