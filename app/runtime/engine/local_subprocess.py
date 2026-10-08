"""In-container subprocess engine driver — the default, identical in
behaviour to the pre-#160 inline Supervisor logic. Spawns the backend's argv
in a new session, logs to a per-model file, and stops it via process-group
SIGTERM->SIGKILL.

Behaviour note (deviation from plan): ``spec.env`` is passed to the child
*verbatim*, NOT merged over ``os.environ``. This preserves the pre-#160
contract — Supervisor builds a curated env via ``build_subprocess_env``
(which deliberately omits secrets like ``VW_ADMIN_PASSWORD``); merging
``os.environ`` back in would leak those into the engine and break the
existing ``test_supervisor_load`` assertions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
from collections.abc import Iterable
from pathlib import Path

from app.runtime.engine import EngineSpec
from app.runtime.engine.run_marker import format_run_sentinel

log = logging.getLogger(__name__)


class LocalHandle:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def wait(self) -> int:
        return await self._proc.wait()


class LocalSubprocessDriver:
    # The engine runs as a subprocess of the warden container, so its vLLM
    # version is whatever is baked into the warden image — ``EngineSpec.image``
    # is meaningless here and silently ignored. Surfaced so Supervisor.load can
    # refuse an engine-version pin instead of launching the wrong version.
    supports_engine_image = False

    def __init__(self, *, log_dir: str, log_max_bytes: int = 0) -> None:
        self._log_dir = Path(log_dir)
        self._log_max_bytes = log_max_bytes

    def _rotate(self, log_path: Path) -> None:
        """Keep one previous generation, then start fresh.

        The log is opened O_APPEND and reused by every load, so without this it
        only ever grows. It shares /data with the SQLite DB, and filling that
        volume takes the database down with it (the 2026-06-15 ENOSPC
        incident). Rotation happens at spawn, never mid-run, so no writer holds
        the file we are replacing.
        """
        if self._log_max_bytes <= 0:
            return
        try:
            size = log_path.stat().st_size
        except FileNotFoundError:
            # The FIRST load of a model has no log yet, so there is nothing to
            # rotate. This used to fall into the OSError handler below and
            # print a full FileNotFoundError traceback on every first load --
            # harmless (spawn creates the file a moment later) but it reads
            # like a fault at exactly the moment a new operator is watching.
            return
        except OSError:
            # Rotation is housekeeping; never let it stop an engine starting.
            log.warning("could not stat %s", log_path, exc_info=True)
            return
        if size < self._log_max_bytes:
            return
        try:
            log_path.replace(log_path.with_suffix(".log.1"))
        except OSError:
            log.warning("could not rotate %s", log_path, exc_info=True)

    def _copy_tail_capped(self, src: Path, dst: Path, max_bytes: int) -> None:
        """Copy at most the last ``max_bytes`` of ``src`` into ``dst``,
        marking the dropped head. Same convention as the watchdog's crash
        log tail: the interesting content is at the end of the file."""
        size = src.stat().st_size
        with src.open("rb") as fh, dst.open("wb") as out:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                out.write(b"[truncated: copied the last %d of %d bytes]\n" % (max_bytes, size))
            shutil.copyfileobj(fh, out)

    def _rotate_live(self, log_path: Path) -> bool:
        """In-run rotation for a log whose writer still holds an O_APPEND fd.

        ``_rotate`` renames the file at spawn, which only works while NO
        writer exists: renamed under a live fd, the file keeps growing and
        nothing is reclaimed. Copy-truncate works WITH a live writer: the
        newest content is copied to the rotated name and the live file is
        zeroed in place; the writer's next write lands at the new end
        (offset 0) with no reopen. The copy and the truncate are not
        atomic, so a writer racing the copy can lose a few boundary lines —
        the same documented trade as logrotate's copytruncate. Engine logs
        are diagnostic, not transactional.
        """
        if self._log_max_bytes <= 0:
            return False
        try:
            size = log_path.stat().st_size
        except FileNotFoundError:
            return False
        except OSError:
            log.warning("could not stat %s", log_path, exc_info=True)
            return False
        if size < self._log_max_bytes:
            return False
        rotated = log_path.with_suffix(".log.1")
        try:
            self._copy_tail_capped(log_path, rotated, self._log_max_bytes)
            os.truncate(log_path, 0)
        except OSError:
            log.warning("could not rotate live log %s", log_path, exc_info=True)
            return False
        log.info(
            "engine log %s rotated in-run: kept the last %d of %d bytes in %s",
            log_path,
            self._log_max_bytes,
            size,
            rotated,
        )
        return True

    def sweep_engine_logs(self, live_model_ids: Iterable[str]) -> int:
        """Rotate the over-cap logs of engines running RIGHT NOW.

        Only live models' logs are rotated: a stopped model's log has no
        writer, so it cannot grow, and rotating it would move the tail the
        Live Logs backlog reads out of the live file — which the next
        spawn-time rename would then clobber. Returns the number of files
        rotated.
        """
        rotated = 0
        for model_id in sorted(live_model_ids):
            if self._rotate_live(self._log_dir / f"{model_id}.log"):
                rotated += 1
        return rotated

    def _write_run_sentinel(self, log_fd: int) -> None:
        """Delimit this spawn in the append-only log (#234).

        Without a boundary the diagnosis reader's 200-line window straddles
        runs and reports the PREVIOUS attempt's failure as this one's cause.
        Written before the child is exec'd so it is strictly the first byte of
        the run, and preceded by a newline when the file already has content so
        an un-terminated final chunk from the last run cannot swallow it.

        Best-effort: a log we cannot annotate must never stop an engine
        starting -- the reader falls back to the old whole-tail behaviour.
        """
        try:
            prefix = "" if os.fstat(log_fd).st_size == 0 else "\n"
            os.write(log_fd, f"{prefix}{format_run_sentinel()}\n".encode())
        except OSError:
            log.warning("could not write run sentinel to engine log", exc_info=True)

    async def spawn(self, spec: EngineSpec) -> LocalHandle:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{spec.model_id}.log"
        self._rotate(log_path)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            self._write_run_sentinel(log_fd)
            # argv[0] comes from the backend's LaunchPlan. This driver decides
            # WHERE a process runs, never WHICH program it is.
            proc = await asyncio.create_subprocess_exec(
                *spec.argv,
                env=dict(spec.env),
                stdout=log_fd,
                stderr=log_fd,
                start_new_session=True,
            )
        finally:
            os.close(log_fd)
        return LocalHandle(proc)

    def engine_host(self, model_id: str) -> str:
        # In-container subprocess shares the control-plane's network
        # namespace, so the engine is reachable on loopback.
        return "127.0.0.1"

    async def terminate(self, handle: LocalHandle, *, grace_s: float) -> None:
        pid = handle.pid
        if pid is None or handle.returncode is not None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(handle.wait(), timeout=grace_s)
        except TimeoutError:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await handle.wait()
