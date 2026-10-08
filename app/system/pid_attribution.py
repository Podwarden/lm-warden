"""PID → model_id attribution for the live GPU probe.

Strategy
--------
The supervisor spawns each vLLM model with ``start_new_session=True`` (see
``app/runtime/supervisor.py``). That means every model gets its own POSIX
session, and the parent PID becomes the session-group leader: PGID == parent
PID. Worker processes spawned by vLLM under tensor parallelism inherit that
session, so they share the same pgrp.

Given the set of supervisor-tracked parent PIDs, we can attribute any process
PID to a model by reading ``/proc/<pid>/stat`` field 5 (pgrp) and looking it
up in the parent-pid → model_id map.

If ``/proc`` is unavailable (e.g. non-Linux dev box), or a PID has already
exited, attribution returns ``None`` and the caller treats the holder as
external.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROC = Path("/proc")


def read_pgid(pid: int, *, proc_root: Path = PROC) -> int | None:
    """Return the process group id for ``pid``, or None if unreadable.

    Reads /proc/<pid>/stat. Field 5 (pgrp) is the process group id. The 2nd
    field is the executable name in parentheses and may itself contain
    parentheses or spaces, so we slice from the last ')' instead of splitting
    naively.
    """
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError) as exc:
        logger.debug("read_pgid(%d) failed: %s", pid, exc)
        return None
    rparen = raw.rfind(")")
    if rparen == -1:
        return None
    rest = raw[rparen + 1 :].strip().split()
    # After ')' the fields are: state ppid pgrp ... → indices 0,1,2.
    if len(rest) < 3:
        return None
    try:
        return int(rest[2])
    except ValueError:
        return None


def attribute_pid_to_model(
    pid: int,
    parent_pid_to_model: dict[int, str],
    *,
    proc_root: Path = PROC,
) -> str | None:
    """Map a PID seen by nvidia-smi to a model_id (or None).

    Two-step lookup so we tolerate vLLM tensor-parallel workers:

    1. If ``pid`` IS one of our tracked parent PIDs, return that model_id.
    2. Otherwise read its pgrp; if pgrp matches a tracked parent PID, return
       that model_id. (Parent was spawned with ``start_new_session=True``, so
       pgrp == parent_pid for the entire process group.)
    """
    if pid in parent_pid_to_model:
        return parent_pid_to_model[pid]
    pgid = read_pgid(pid, proc_root=proc_root)
    if pgid is None:
        return None
    return parent_pid_to_model.get(pgid)
