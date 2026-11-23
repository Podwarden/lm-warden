"""KV-cache tier daemon: paths shared by the engine argv and the supervisor."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import logging
import os
import re
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.db.database import open_db
from app.db.repos.settings import SettingsRepo

if TYPE_CHECKING:
    from lmw_kv.client import Client

log = logging.getLogger(__name__)

#: Runtime-setting defaults (see ``RUNTIME_KEYS``); the DB stores text.
DEFAULT_RAM_GIB = 16.0
DEFAULT_NVME_CAP_GB = 250.0

#: The ``kv_store`` values that turn the connector on (``off`` is everything else).
KV_TIERS = ("ram", "ram+nvme")


#: The only engine driver that can run the KV connector. The daemon's socket, its RAM
#: pool and the key file live in the api container (``/dev/shm`` and the data volume);
#: only an engine that is a subprocess of the api shares them. A docker-driver engine is
#: a sibling container and cannot reach them.
KV_ENGINE_DRIVER = "local"


class KvTierUnsupported(ValueError):
    """``kv_store`` is on but the engine driver cannot run the KV connector."""


def kv_enabled(model: Any) -> bool:
    """True when the row asks for a KV tier. Strict, so a stand-in row never trips it."""
    return getattr(model, "kv_store", "off") in KV_TIERS


def kv_driver_refusal(driver: str) -> str:
    """Why ``kv_store`` cannot be on under ``driver`` (anything but ``local``)."""
    return (
        f"kv_store needs the in-container engine driver (VW_ENGINE_DRIVER={KV_ENGINE_DRIVER}); "
        f"this deployment runs VW_ENGINE_DRIVER={driver}, whose engine containers cannot "
        "reach the KV store daemon's socket, RAM pool or key file. Set kv_store to off."
    )


def configured_engine_driver() -> str:
    """``VW_ENGINE_DRIVER`` as ``app.config.load_settings`` reads it, for a caller with no
    settings at hand (the model writer)."""
    return os.environ.get("VW_ENGINE_DRIVER", "local")


def check_kv_driver(model: Any, driver: str) -> None:
    """Raise ``KvTierUnsupported`` when ``model`` asks for a KV tier ``driver`` cannot run."""
    if kv_enabled(model) and driver != KV_ENGINE_DRIVER:
        raise KvTierUnsupported(kv_driver_refusal(driver))


def socket_path(settings: Any, shm_dir: str | None = None) -> str:
    """Unix socket the engine's ``LmwKVConnector`` talks to: ``<shm dir>/lmw-kv.sock``.

    Pod-local, next to the RAM pool (ruling R25). Never on the data volume: during a
    rolling update two api pods share that volume, and one pod's engine could reach the
    other pod's daemon, whose RAM file it cannot see, or one pod's shutdown could unlink
    the other's live socket. ``settings`` is kept so callers need not change.
    """
    if shm_dir is None:
        # Late: app.runtime.backends imports this module at load time.
        from app.runtime.backends.vllm.env import ENGINE_SHM_PATH

        shm_dir = ENGINE_SHM_PATH
    return str(Path(shm_dir) / "lmw-kv.sock")


PIDFILE_NAME = "lmw-kv.pid"
#: Deadline of the reports' ``stats`` call (``KvStoreSupervisor.stats_client``).
STATS_TIMEOUT_S = 2.0


def _cmdline(proc_root: Path, pid: int) -> list[str]:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def _is_our_daemon(cmdline: list[str], socket: str) -> bool:
    """The process serves ``socket``: its argv has ``--socket <socket>``."""
    return any(a == "--socket" and b == socket for a, b in zip(cmdline, cmdline[1:], strict=False))


def reap_stale_daemon(
    pidfile: Path,
    socket: str,
    *,
    proc_root: Path = Path("/proc"),
    kill: Callable[[int, int], None] = os.kill,
    grace_s: float = 5.0,
    kill_wait_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Stop a daemon an earlier api left running; its pid, or None. Blocking.

    ``pidfile`` (next to the socket in the pod-local shm dir) names the last daemon's
    pid. It is terminated only if ``/proc/<pid>/cmdline`` shows it serves ``socket``, so
    a reused pid is never touched: SIGTERM, then SIGKILL after ``grace_s``, then a wait of
    up to ``kill_wait_s`` for it to be gone. The pidfile
    is removed either way. Without ``/proc`` (macOS) nothing is checked or killed.
    """
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        pidfile.unlink(missing_ok=True)
        return None
    found = None
    if pid > 1 and pid != os.getpid() and proc_root.is_dir():
        if _is_our_daemon(_cmdline(proc_root, pid), socket):
            found = pid
            log.warning("kv store: stopping a daemon (pid %d) an earlier api left running", pid)
            with contextlib.suppress(ProcessLookupError):
                kill(pid, signal.SIGTERM)
            waited = 0.0
            while _is_our_daemon(_cmdline(proc_root, pid), socket) and waited < grace_s:
                sleep(0.1)
                waited += 0.1
            if _is_our_daemon(_cmdline(proc_root, pid), socket):
                with contextlib.suppress(ProcessLookupError):
                    kill(pid, signal.SIGKILL)
                # Until it is gone it still holds the NVMe lock and the pool pages.
                waited = 0.0
                while _is_our_daemon(_cmdline(proc_root, pid), socket) and waited < kill_wait_s:
                    sleep(0.1)
                    waited += 0.1
                if _is_our_daemon(_cmdline(proc_root, pid), socket):
                    log.warning("kv store: stale daemon (pid %d) still runs after SIGKILL", pid)
    pidfile.unlink(missing_ok=True)
    return found


_COMMIT = re.compile(r"[0-9a-f]{40}")


class KvRevisionUnresolved(Exception):
    """The model's revision is not known as a commit: this load runs with the tier off."""


#: ``extra_args`` flags that change which commit or tokenizer the engine loads, so the
#: commits LM Warden resolved would not be the ones in use (with vLLM's abbreviation
#: floors; a floor set too low only turns the tier off for one more spelling).
_REVISION_FLAGS = {
    "--revision": "--rev",
    "--tokenizer-revision": "--tokenizer-r",
    "--tokenizer": "--tokenizer",
    "--code-revision": "--cod",
}


def _has_snapshot(cache: Any, repo: str, sha: str | None) -> bool:
    """``models--org--name/snapshots/<sha>`` exists, at the cache root or under ``hub/``."""
    if not cache or not sha:
        return False
    slug = f"models--{repo.replace('/', '--')}"
    root = Path(cache)
    return any((base / slug / "snapshots" / sha).is_dir() for base in (root, root / "hub"))


def resolve_kv_revisions(settings: Any, model: Any) -> dict[str, str]:
    """The commits (40 hex) the engine loads, for the KV namespace. Blocking (reads the cache).

    The row's ``hf_revision`` (vLLM's default branch ``main`` when empty) is resolved
    through the HF cache (``models--org--name/refs/<branch>``) to the commit the engine
    will load. The tokenizer comes from ``tokenizer_repo`` when set, else the model repo,
    at the same revision (vLLM's default for ``--tokenizer-revision``). Never falls back
    to a branch name: ``main`` moves, and a re-pull would then serve KV from old weights.
    Raises ``KvRevisionUnresolved`` (the tier is off for this load) when a commit cannot
    be found, or when ``extra_args`` change the revision or tokenizer.
    """
    from app.models.model_id_flags import long_flag_in
    from app.runtime.variants import resolve_hf_commit

    def is_commit(rev: str | None) -> bool:
        return isinstance(rev, str) and _COMMIT.fullmatch(rev) is not None

    flag = long_flag_in("vllm", list(getattr(model, "extra_args", []) or []), _REVISION_FLAGS)
    if flag is not None:
        raise KvRevisionUnresolved(
            f"extra_args set {flag}, so the commit the engine loads is not known"
        )
    cache = getattr(settings, "hf_cache_dir", None)
    revision = getattr(model, "hf_revision", None) or "main"
    repo = model.hf_repo
    sha = resolve_hf_commit(cache, repo, revision)
    if not is_commit(sha):
        raise KvRevisionUnresolved(
            f"revision {revision!r} of {repo} is not in the model cache as a commit"
        )
    tok_repo = getattr(model, "tokenizer_repo", None) or repo
    if tok_repo == repo:
        tok_sha: str | None = sha
    else:
        # Another repository has its own commits: never the model's. Its own ref, the
        # row's branch when it is one, else main; and the snapshot must be in the cache.
        tok_ref = revision if not is_commit(revision) else "main"
        tok_sha = resolve_hf_commit(cache, tok_repo, tok_ref)
        if not is_commit(tok_sha) or not _has_snapshot(cache, tok_repo, tok_sha):
            raise KvRevisionUnresolved(
                f"ref {tok_ref!r} of the tokenizer {tok_repo} is not in the model cache "
                "as a commit"
            )
    assert sha is not None and tok_sha is not None
    return {"revision": sha, "tokenizer_revision": tok_sha}


def kv_launch_kwargs(settings: Any, model: Any) -> dict[str, Any]:
    """``Backend.plan`` kwargs wiring a model's KV tier in; empty while ``kv_store`` is off.

    ``kv_key_file`` is the PATH of the cobs2 network key file, never the key. Blocking
    (reads the HF cache). Raises ``KvTierUnsupported`` when the engine driver cannot run
    the connector, and ``KvRevisionUnresolved`` when this load must run with the tier
    off (plan it with :func:`without_kv`).
    """
    if not kv_enabled(model):
        return {}
    check_kv_driver(model, getattr(settings, "engine_driver", KV_ENGINE_DRIVER))
    from app.cobs2.keys import network_key_path

    return {
        "kv_socket": socket_path(settings),
        "kv_key_file": str(network_key_path(settings.db_path)),
        "kv_revisions": resolve_kv_revisions(settings, model),
    }


def without_kv(model: Any) -> Any:
    """A copy of ``model`` with ``kv_store`` off, to plan a load whose tier is refused."""
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        return dataclasses.replace(model, kv_store="off")
    clone = copy.copy(model)
    clone.kv_store = "off"
    return clone


def _num(raw: str | None, default: float) -> float:
    try:
        v = float(raw) if raw else default
    except ValueError:
        return default
    return v if v >= 0 else default


#: A daemon commits its whole RAM pool before it answers, so its start timeout grows
#: with the pool (``KvStoreSupervisor.start_timeout_for``): ``start_timeout_s`` covers a
#: pool of up to this many GiB and scales in proportion above it. The default 60 s over
#: 30 GiB allows 2 s per GiB; about 0.3 s per GiB was measured on an idle host.
START_TIMEOUT_COVERS_GIB = 30.0


class KvStoreSupervisor:
    """Runs the ``lmw_kv`` store daemon as a subprocess of the api.

    The daemon runs while at least one model row has ``kv_store != "off"``. A
    background loop (never awaited by api startup) polls for that every
    ``poll_s``, starts the daemon, health-checks it with ``hello`` and restarts
    it with back-off when it exits. Once started it keeps running until
    ``stop()``, even if the last model turns ``kv_store`` off.

    Liveness: after the first healthy reply it keeps sending ``hello`` every
    ``live_s`` (deadline ``live_timeout_s``). After ``live_misses`` misses in a
    row the daemon counts as wedged: it is killed and restarted. A daemon that
    never answers within ``start_timeout_s`` of starting is restarted too.

    Generations: each daemon that turns healthy gets the next ``generation()``
    (1, 2, ...). A restarted daemon has a new RAM pool file, and an engine's
    connector turns itself off for good once it sees that, so an engine loaded
    under an older generation runs without its tier until it is reloaded.
    ``on_restart`` callbacks hear about every generation after the first; ``on_healthy``
    callbacks hear about every generation, the first included (the api syncs the
    policy and the retired scopes from there). After ``sync_before_loads()``,
    ``ensure_running`` also waits (up to ``sync_wait_s``) until ``mark_synced`` reports
    the current generation, so a model load starts against a synced daemon.
    """

    def __init__(
        self,
        settings: Any,
        *,
        shm_path: str | None = None,
        command: list[str] | None = None,
        poll_s: float = 5.0,
        backoff_s: tuple[float, float] = (1.0, 30.0),
        stop_grace_s: float = 5.0,
        live_s: float = 5.0,
        live_timeout_s: float = 2.0,
        live_misses: int = 3,
        start_timeout_s: float = 60.0,
        sync_wait_s: float = 10.0,
    ) -> None:
        self._settings = settings
        if shm_path is None:
            # Late: app.runtime.backends imports this module at load time.
            from app.runtime.backends.vllm.env import ENGINE_SHM_PATH

            shm_path = ENGINE_SHM_PATH
        self._shm_path = shm_path
        self._command = command or [sys.executable, "-m", "lmw_kv.daemon"]
        self._poll_s = poll_s
        self._backoff = backoff_s
        self._grace_s = stop_grace_s
        self._live_s = live_s
        self._live_timeout_s = live_timeout_s
        self._live_misses = max(1, live_misses)
        self._start_timeout_s = start_timeout_s
        self._start_timeout_now = start_timeout_s  # scaled with the requested pool
        self._ram_requested: int | None = None  # bytes asked of the running daemon
        self._ram_granted: int | None = None  # bytes it committed (its hello ram_bytes)
        self._logged_no_tier = False
        self._live_client: Client | None = None
        self._stats_client: Client | None = None
        self._socket = socket_path(settings, shm_path)
        self._pidfile = Path(shm_path) / PIDFILE_NAME
        self._task: asyncio.Task[None] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._healthy = False
        self._generation = 0  # bumped each time a daemon turns healthy
        self._restart_cbs: list[Callable[[int], None]] = []
        self._healthy_cbs: list[Callable[[int], None]] = []
        self._sync_gate = False  # sync_before_loads() was called
        self._sync_wait_s = sync_wait_s
        self._synced = 0  # the last generation a sync attempt finished for
        # ensure_running waits on this for "healthy" or "a poll after my poke chose no daemon".
        # Set and replaced on every change (one-shot): no lock, so a cancelled task can
        # never leave it held.
        self._changed = asyncio.Event()
        self._poll_seq = 0  # bumped as each poll starts
        self._decided_seq = 0  # the poll whose decision ``_idle`` holds
        self._idle = False  # that poll found nothing to start
        self._client: Client | None = None
        self._wake = asyncio.Event()

    async def start(self) -> None:
        """Begin supervising in the background; returns at once."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="kv-store-supervisor")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._terminate()
        self._healthy = False
        self._notify()
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
        if self._live_client is not None:
            await asyncio.to_thread(self._live_client.close)
            self._live_client = None
        if self._stats_client is not None:
            await asyncio.to_thread(self._stats_client.close)
            self._stats_client = None
        with contextlib.suppress(OSError):
            os.unlink(self._socket)
        with contextlib.suppress(OSError):
            os.unlink(self._pidfile)

    def poke(self) -> None:
        """Re-check at once whether the daemon is wanted, instead of at the next poll."""
        self._wake.set()

    async def ensure_running(self, timeout_s: float | None = None) -> bool:
        """``poke()``, then wait for a healthy daemon; True if healthy.

        Waits up to ``timeout_s``, by default the daemon's own ``start_timeout_s``: a first
        start commits the whole RAM pool (``fallocate`` of 16 GiB or more) and an engine
        loaded before the daemon answers keeps its tier off for its whole life. Returns
        False at once when a poll after this poke finds there is nothing to start.
        """
        if not self._healthy:
            timeout = self._start_timeout_now if timeout_s is None else timeout_s
            seen = self._poll_seq
            self.poke()
            await self._wait_for(
                lambda: self._healthy or (self._decided_seq > seen and self._idle), timeout
            )
        if self._healthy and self._sync_gate and self._synced < self._generation:
            # Fail open: a sync that does not finish in time only means the daemon runs
            # its defaults (lru) a little longer.
            await self._wait_for(
                lambda: not self._healthy or self._synced >= self._generation, self._sync_wait_s
            )
        return self._healthy

    async def _wait_for(self, cond: Callable[[], bool], limit_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + limit_s
        while not cond():
            changed = self._changed
            left = deadline - loop.time()
            if left <= 0:
                return
            try:
                async with asyncio.timeout(left):
                    await changed.wait()
            except TimeoutError:
                return

    def healthy(self) -> bool:
        return self._healthy

    def start_timeout_for(self, ram_gib: float) -> float:
        """How long a daemon asked for ``ram_gib`` may take to answer: it commits the whole
        pool first. ``start_timeout_s`` at least, in proportion to the pool above
        ``START_TIMEOUT_COVERS_GIB``."""
        return self._start_timeout_s * max(1.0, ram_gib / START_TIMEOUT_COVERS_GIB)

    def start_timeout(self) -> float:
        """The start timeout of the current (or next) daemon."""
        return self._start_timeout_now

    def ram(self) -> dict[str, int | None]:
        """The RAM pool asked of the daemon (``kv_store_ram_gib``) and the one it committed,
        which is smaller when ``/dev/shm`` or free memory did not fit the request; None
        until a daemon has answered."""
        return {"requested_bytes": self._ram_requested, "granted_bytes": self._ram_granted}

    def generation(self) -> int:
        """The generation of the current (or last) healthy daemon; 0 before the first."""
        return self._generation

    def on_restart(self, cb: Callable[[int], None]) -> None:
        """Call ``cb(generation)`` each time a restarted daemon turns healthy."""
        self._restart_cbs.append(cb)

    def on_healthy(self, cb: Callable[[int], None]) -> None:
        """Call ``cb(generation)`` each time a daemon turns healthy, the first included.

        Synchronous, on the event loop, right after the daemon counts as healthy; a
        failing callback is logged and stops nothing."""
        self._healthy_cbs.append(cb)

    def sync_before_loads(self) -> None:
        """From now on ``ensure_running`` waits for ``mark_synced`` of the current
        generation (at most ``sync_wait_s``) before it reports a healthy daemon."""
        self._sync_gate = True

    def mark_synced(self, generation: int) -> None:
        """A sync attempt for ``generation`` finished (whatever its outcome)."""
        if generation > self._synced:
            self._synced = generation
            self._notify()

    def client(self) -> Client | None:
        """The shared (thread-safe) client while the daemon is healthy, else None."""
        return self._client if self._healthy else None

    def stats_client(self) -> Client | None:
        """The reports' client for ``stats`` while the daemon is healthy, else None.

        A connection of its own with a ``STATS_TIMEOUT_S`` deadline: a slow stats reply
        never closes the hot-path client's connection, whose 50 ms timeout guards every
        turn's tier hits on the same worker threads.
        """
        return self._stats_client if self._healthy else None

    # ---- internals -----------------------------------------------------------

    async def _read_config(self) -> tuple[bool, dict[str, float | str]]:
        from app.db.repos.models import ModelRepo

        async with open_db(self._settings.db_path) as db:
            models = await ModelRepo(db).list_all()
            wanted = any(m.kv_store in KV_TIERS for m in models)
            kv = await SettingsRepo(db).get_many(
                ["kv_store_ram_gib", "kv_store_nvme_dir", "kv_store_nvme_cap_gb"]
            )
        # One engine worker per GPU at most runs beside the pool (a model's TP/DP workers
        # each hold one GPU); every GPU a registered model may use counts.
        gpus = {g for m in models for g in (m.gpu_indices or [])}
        return wanted, {
            "ram_gib": _num(kv.get("kv_store_ram_gib"), DEFAULT_RAM_GIB),
            "nvme_dir": (kv.get("kv_store_nvme_dir") or "").strip(),
            "nvme_cap_gb": _num(kv.get("kv_store_nvme_cap_gb"), DEFAULT_NVME_CAP_GB),
            "engine_workers": float(max(1, len(gpus))),
        }

    def _argv(self, cfg: dict[str, float | str]) -> list[str] | None:
        """The daemon argv, or None when there would be no tier at all. Blocking (statvfs,
        unlink, /proc reads): the loop runs it in a worker thread.

        The daemon gets the operator's ``kv_store_ram_gib`` unchanged: it sizes the pool
        itself (``/dev/shm`` and free memory) right before it commits it, and logs the one
        WARNING naming requested and granted. The fit here only decides whether there is
        anything to start."""
        import lmw_kv.ram as ram

        wanted = int(float(cfg["ram_gib"]) * (1 << 30))
        workers = int(cfg.get("engine_workers", 0) or 0)
        pool = Path(self._shm_path) / "lmw-kv.pool"
        try:
            # Our daemon is not running here: a pool file is a leftover of one that died.
            # The daemon unlinks it too; doing it first keeps it out of the free space.
            ram.discard_stale(pool)
            fitted = ram.grant(
                wanted,
                ram.fit_shm(pool, wanted),
                ram.fit_memory(wanted, engine_workers=workers),
            )
        except OSError as exc:
            log.warning("kv store: cannot size %s (%s); no RAM tier", pool, exc)
            fitted = 0
        nvme = str(cfg["nvme_dir"])
        if fitted == 0 and not nvme:
            if not self._logged_no_tier:  # the poll runs every few seconds: say it once
                self._logged_no_tier = True
                log.warning(
                    "kv store: no RAM tier fits (%s, free memory) and no NVMe dir; not starting",
                    pool,
                )
            return None
        self._logged_no_tier = False
        self._ram_requested = wanted
        self._start_timeout_now = self.start_timeout_for(float(cfg["ram_gib"]))
        argv = [
            *self._command,
            "--parent-pid",
            str(os.getpid()),
            "--socket",
            self._socket,
            "--ram-path",
            str(pool),
            "--ram-gib",
            str(wanted / (1 << 30)),
            "--engine-workers",
            str(workers),
        ]
        if nvme:
            argv += ["--nvme-dir", nvme, "--nvme-cap-gb", str(cfg["nvme_cap_gb"])]
        return argv

    def _notify(self) -> None:
        """Wake every ``ensure_running`` waiter to re-check its condition."""
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def _loop(self) -> None:
        delay = self._backoff[0]
        # A daemon an earlier api left behind (an OOM-killed api has no chance to stop it)
        # holds the old pool's pages and the NVMe lock: stop it before anything is sized.
        try:
            await asyncio.to_thread(
                reap_stale_daemon, self._pidfile, self._socket, grace_s=self._grace_s
            )
        except Exception:  # noqa: BLE001 -- never a reason not to supervise
            log.warning("kv store: checking for a stale daemon failed", exc_info=True)
        while True:
            self._poll_seq += 1
            seq = self._poll_seq
            try:
                wanted, cfg = await self._read_config()
                # Off the loop: statvfs and unlink on /dev/shm are blocking calls.
                argv = await asyncio.to_thread(self._argv, cfg) if wanted else None
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- a bad poll must not end supervision
                log.warning("kv store: reading config failed", exc_info=True)
                argv = None
            self._decided_seq, self._idle = seq, argv is None
            self._notify()
            if argv is None:
                # asyncio.timeout, not wait_for: Python 3.11's wait_for can hang a cancel
                # that lands while the wake event is being set (a poke during stop()).
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(self._poll_s):
                        await self._wake.wait()
                self._wake.clear()
                continue
            try:
                was_healthy = await self._run_once(argv)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.warning("kv store: daemon failed to run", exc_info=True)
                was_healthy = False
            delay = self._backoff[0] if was_healthy else min(delay * 2, self._backoff[1])
            log.warning("kv store: daemon exited; restarting in %.1fs", delay)
            await asyncio.sleep(delay)

    async def _run_once(self, argv: list[str]) -> bool:
        """Run the daemon until it exits; True when it ever became healthy."""
        from lmw_kv.client import Client

        Path(self._socket).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.unlink(self._socket)
        self._proc = await asyncio.create_subprocess_exec(*argv, start_new_session=True)
        proc = self._proc
        try:
            self._pidfile.write_text(f"{proc.pid}\n")
        except OSError as exc:
            log.warning("kv store: cannot write %s: %s", self._pidfile, exc)
        if self._client is None:
            self._client = Client(self._socket)
        if self._live_client is None:
            self._live_client = Client(self._socket, timeout_s=self._live_timeout_s)
        if self._stats_client is None:
            self._stats_client = Client(self._socket, timeout_s=STATS_TIMEOUT_S)
        client = self._client
        became_healthy = False
        probe = asyncio.create_task(self._probe(client, proc))
        try:
            await proc.wait()
        finally:
            probe.cancel()
            await asyncio.gather(probe, return_exceptions=True)
            became_healthy = self._healthy
            self._healthy = False
            await self._terminate()
        return became_healthy

    async def _probe(self, client: Client, proc: asyncio.subprocess.Process) -> None:
        """Wait for the first healthy reply, then keep checking that the daemon answers."""
        loop = asyncio.get_running_loop()
        give_up = loop.time() + self._start_timeout_now
        while True:
            if proc.returncode is not None:
                return
            h = await asyncio.to_thread(client.hello)
            if h is not None:
                rb = h.get("ram_bytes")
                self._ram_granted = rb if isinstance(rb, int) and not isinstance(rb, bool) else None
                break
            if loop.time() >= give_up:
                log.warning(
                    "kv store: daemon did not answer within %.0fs; restarting it",
                    self._start_timeout_now,
                )
                await self._kill(proc)
                return
            await asyncio.sleep(0.1)
        self._generation += 1
        self._healthy = True
        for cb in self._healthy_cbs:
            try:
                cb(self._generation)
            except Exception:  # noqa: BLE001 -- a listener must not stop supervision
                log.debug("kv store: healthy listener failed", exc_info=True)
        self._notify()
        log.info("kv store: daemon healthy on %s (generation %d)", self._socket, self._generation)
        if self._generation > 1:
            log.warning(
                "kv store: the daemon restarted (generation %d); engines loaded before keep "
                "running with their KV tier off until they are reloaded",
                self._generation,
            )
            for cb in self._restart_cbs:
                try:
                    cb(self._generation)
                except Exception:  # noqa: BLE001 -- a listener must not stop supervision
                    log.debug("kv store: restart listener failed", exc_info=True)
        live = self._live_client or client
        misses = 0
        while proc.returncode is None:
            await asyncio.sleep(self._live_s)
            if proc.returncode is not None:
                return
            if await asyncio.to_thread(live.hello) is not None:
                misses = 0
                continue
            misses += 1
            if misses >= self._live_misses:
                log.warning(
                    "kv store: daemon stopped answering (%d missed checks); restarting it",
                    misses,
                )
                self._healthy = False
                await self._kill(proc)
                return

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        """SIGTERM a wedged daemon's process group, SIGKILL after the grace period.

        ``_run_once`` is waiting on the process and restarts it once it exits.
        """
        if proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=self._grace_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

    async def _terminate(self) -> None:
        """SIGTERM the daemon's process group, SIGKILL after the grace period."""
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._grace_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
