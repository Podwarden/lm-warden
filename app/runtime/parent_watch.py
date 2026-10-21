"""VW_PARENT_PID belt: watch for the launcher process dying.

Metal design §2: a force-quit must not leave a headless uvicorn behind. The
Swift launcher sets VW_PARENT_PID to its own pid; on parent death we SIGTERM
ourselves so uvicorn runs its normal graceful shutdown (engines in the same
process group are covered by the separate group-kill belt).
"""

import asyncio
import logging
import os
import signal
from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)

_ENV = "VW_PARENT_PID"


def parse_parent_pid(env: Mapping[str, str]) -> int | None:
    """VW_PARENT_PID -> launcher pid. Unset or empty means no watcher; any
    other malformed value fails boot loud — a silently disabled belt is the
    orphan hazard this exists to prevent."""
    raw = env.get(_ENV, "")
    if raw == "":
        return None
    if raw[0] in "+-" or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{_ENV} must be a plain decimal pid, got {raw!r}")
    value = int(raw)
    if value < 2:
        raise ValueError(f"{_ENV} must be >= 2, got {raw!r}")
    return value


def parent_alive(
    pid: int,
    *,
    start_ppid: int,
    getppid: Callable[[], int] = os.getppid,
    kill: Callable[[int, int], object] = os.kill,
) -> bool:
    """The parent is gone when kill(pid, 0) says no such process, or when we
    have been reparented (getppid moved on). Any other signal is alive:
    never a spurious shutdown."""
    if getppid() != start_ppid:
        return False
    try:
        kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


async def watch_parent(
    pid: int,
    *,
    interval: float = 2.0,
    getppid: Callable[[], int] = os.getppid,
    kill: Callable[[int, int], object] = os.kill,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_death: Callable[[], object] | None = None,
) -> None:
    """Poll for the parent's death, checking before the first sleep and after
    each one. On the first dead check: log one WARNING naming the pid, call
    on_death once (default: SIGTERM ourselves), and return."""
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got {interval!r}")
    start_ppid = getppid()
    while True:
        if not parent_alive(pid, start_ppid=start_ppid, getppid=getppid, kill=kill):
            logger.warning("parent pid %s is gone; triggering graceful shutdown", pid)
            if on_death is None:
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                on_death()
            return
        await sleep(interval)
