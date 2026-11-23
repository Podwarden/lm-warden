"""RAM slab pool: a shared-memory file carved into fixed-size slots.

The store daemon owns the file; a connector in another process maps the same
file and pins it, so every slot is addressable by byte offset.
"""

from __future__ import annotations

import mmap
import os
import threading
from pathlib import Path

#: Free space ``fit_shm`` leaves in ``/dev/shm`` for the engines. Tensor-parallel vLLM
#: engines need at least 2 GiB of shared memory there (docs/operating.md, "Engine
#: shared memory"), and the pool is sized before they start, so their share is kept.
SHM_RESERVE_BYTES = 2 << 30


#: Memory the pool never takes, so the engine (its CPU-side load buffers, Python
#: heap, CUDA host allocations) and the OS keep room: ``max`` of this and
#: ``MEMORY_MARGIN_FRACTION`` of total memory.
MEMORY_MARGIN_MIN_BYTES = 4 << 30
MEMORY_MARGIN_FRACTION = 0.10

#: A pool shrunk below this is not worth its cost: the RAM tier is turned off instead.
#: A requested pool smaller than this that fits as asked is kept.
MIN_POOL_BYTES = 1 << 30

#: Host memory kept free for each engine worker that may run alongside the pool (its
#: Python and CUDA runtime, pinned and CUDA host buffers, vLLM's CPU swap space). The
#: pool is sized when the daemon starts, usually before any engine exists, so the margin
#: above cannot see these yet. The api passes the number of workers (``--engine-workers``).
ENGINE_WORKER_RESERVE_BYTES = 4 << 30

MEMINFO = Path("/proc/meminfo")
CGROUP = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")


def memory_margin(total: int) -> int:
    """Bytes kept free for the engine and the OS out of ``total`` bytes of memory."""
    return max(MEMORY_MARGIN_MIN_BYTES, int(total * MEMORY_MARGIN_FRACTION))


def _meminfo(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[name.strip()] = int(parts[0]) * (1024 if parts[1:2] == ["kB"] else 1)
    return out


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None  # "max" (no limit) is None


def _stat_field(path: Path, field: str) -> int:
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == field and value.strip().isdigit():
                return int(value)
    except OSError:
        pass
    return 0


def _cgroup_dirs(root: Path, proc_cgroup: Path) -> tuple[Path, Path]:
    """This process's (v2 dir, v1 memory dir), from ``/proc/self/cgroup``.

    A line ``0::/a/b`` is the v2 group, ``N:...memory...:/a/b`` the v1 memory group. In a
    container with its own cgroup namespace the path is ``/`` and the group is the root
    of the mount. A path whose directory is not mounted here falls back to the root.
    """
    v2, v1 = root, root / "memory"
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return v2, v1
    for line in lines:
        hid, _, rest = line.partition(":")
        ctrls, _, path = rest.partition(":")
        rel = path.strip().lstrip("/")
        if hid == "0" and ctrls == "":
            cand = root / rel
            v2 = cand if (cand / "memory.max").exists() else root
        elif "memory" in ctrls.split(","):
            cand = root / "memory" / rel
            v1 = cand if (cand / "memory.limit_in_bytes").exists() else root / "memory"
    return v2, v1


def _cgroup_memory(root: Path, proc_cgroup: Path) -> tuple[int, int] | None:
    """(available, limit) of this process's cgroup memory, or None when it has no limit.

    cgroup v2 (``memory.max``) or v1 (``memory.limit_in_bytes``). Inactive file pages count
    as available: the kernel reclaims them before the limit bites. Active file pages (model
    weights read moments ago) do not, so the pool never pushes them out.
    """
    v2_dir, v1_dir = _cgroup_dirs(root, proc_cgroup)
    v2 = _read_int(v2_dir / "memory.max")
    if v2 is not None:
        used = _read_int(v2_dir / "memory.current") or 0
        inactive = _stat_field(v2_dir / "memory.stat", "inactive_file")
        return max(0, v2 - used + inactive), v2
    v1 = _read_int(v1_dir / "memory.limit_in_bytes")
    if v1 is not None and v1 < 1 << 60:  # v1 writes a huge number for "no limit"
        used = _read_int(v1_dir / "memory.usage_in_bytes") or 0
        inactive = _stat_field(v1_dir / "memory.stat", "total_inactive_file")
        return max(0, v1 - used + inactive), v1
    return None


def memory_room(
    *, meminfo: Path = MEMINFO, cgroup: Path = CGROUP, proc_cgroup: Path = PROC_CGROUP
) -> tuple[int, int] | None:
    """(available, total) bytes of memory for this process, or None when unknown.

    ``MemAvailable`` and ``MemTotal`` from ``/proc/meminfo``, lowered to the cgroup's
    room and limit when the process's cgroup has one (a pod with a memory limit).
    """
    try:
        info = _meminfo(meminfo)
        avail, total = info["MemAvailable"], info["MemTotal"]
    except (OSError, KeyError, ValueError):
        return None
    cg = _cgroup_memory(cgroup, proc_cgroup)
    if cg is not None:
        avail, total = min(avail, cg[0]), min(total, cg[1])
    return avail, total


def fit_memory(
    wanted: int,
    *,
    engine_workers: int = 0,
    meminfo: Path = MEMINFO,
    cgroup: Path = CGROUP,
    proc_cgroup: Path = PROC_CGROUP,
) -> int:
    """Bytes of ``wanted`` that fit in available memory less ``memory_margin`` and
    ``ENGINE_WORKER_RESERVE_BYTES`` for each of ``engine_workers``.

    ``wanted`` when memory cannot be read (not Linux). Call it before the pool exists:
    a tmpfs pool's pages are no longer available.
    """
    room = memory_room(meminfo=meminfo, cgroup=cgroup, proc_cgroup=proc_cgroup)
    if room is None:
        return wanted
    avail, total = room
    keep = memory_margin(total) + max(0, engine_workers) * ENGINE_WORKER_RESERVE_BYTES
    return max(0, min(wanted, avail - keep))


def grant(wanted: int, shm_fit: int, mem_fit: int) -> int:
    """The pool size: the smallest of the three, whole pages, or 0 when it had to be
    shrunk under ``MIN_POOL_BYTES``."""
    want = wanted // mmap.PAGESIZE * mmap.PAGESIZE
    got = max(0, min(want, shm_fit, mem_fit)) // mmap.PAGESIZE * mmap.PAGESIZE
    if got < want and got < MIN_POOL_BYTES:
        return 0
    return got


def create_pool_file(path: Path, size_bytes: int) -> None:
    """Create the pool file (a new inode, mode 0600) and commit every page now.

    Fails with FileExistsError if ``path`` exists (call ``discard_stale`` first). On
    any failure the file is removed. Committing up front is what makes a daemon's start
    slow (seconds for 16 GiB), so it runs before the daemon answers anyone; a
    ``RamPool`` made later with ``precommitted`` only maps it.
    """
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size_bytes)
        _commit(fd, size_bytes)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    os.close(fd)


def _commit(fd: int, size_bytes: int) -> None:
    """Commit every page now so an undersized tmpfs fails here (ENOSPC), not with SIGBUS."""
    if hasattr(os, "posix_fallocate"):
        os.posix_fallocate(fd, 0, size_bytes)
        return
    for off in range(0, size_bytes, mmap.PAGESIZE):
        os.pwrite(fd, b"\0", off)


def discard_stale(path: Path) -> None:
    """Unlink a pool file left by an earlier daemon (an unclean exit leaves one).

    Call it once, before sizing and creating a new pool. An engine that still maps
    the old file keeps its pages, but the new pool is a new inode: the engine sees
    the change (its ``_same_file`` check) and turns its store off. Resizing the old
    file instead could leave the engine's pinned pages orphaned and its copies
    landing in memory the daemon no longer uses.
    """
    path.unlink(missing_ok=True)


def fit_shm(path: Path, wanted: int) -> int:
    """Bytes that fit on the filesystem holding `path`, minus the reserve, at most `wanted`.

    Call ``discard_stale`` first: a leftover file at `path` counts as used space.
    """
    st = os.statvfs(path if path.exists() else path.parent)
    free = st.f_bavail * st.f_frsize
    return max(0, min(wanted, free - SHM_RESERVE_BYTES))


class RamPool:
    def __init__(
        self,
        path: Path,
        size_bytes: int,
        slot_bytes: int,
        *,
        fixed_size: bool = False,
        precommitted: bool = False,
    ) -> None:
        """Carve ``size_bytes`` into ``slot_bytes`` slots.

        By default the file is trimmed to the whole slots. With ``fixed_size`` the
        file is exactly ``size_bytes`` and the tail past the last whole slot stays
        unused, so pools with different slot sizes keep the same file size (a
        connector's mapping of the file stays valid across a re-slab).
        ``precommitted``: the file was made by ``create_pool_file`` with this size, so
        it is only mapped (no truncate, no allocation); a file of another size, or a
        missing one, is still sized and committed.
        """
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        self.slot_bytes = slot_bytes
        self.slots_total = size_bytes // slot_bytes
        if self.slots_total == 0:
            raise ValueError("pool smaller than one slot")
        self.size_bytes = size_bytes if fixed_size else self.slots_total * slot_bytes
        self.path = path
        self._lock = threading.Lock()
        self._free = list(range(self.slots_total - 1, -1, -1))
        created = not path.exists()
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if not (precommitted and os.fstat(self._fd).st_size == self.size_bytes):
                os.ftruncate(self._fd, self.size_bytes)
                _commit(self._fd, self.size_bytes)
            self._mm = mmap.mmap(self._fd, self.size_bytes, mmap.MAP_SHARED)
        except BaseException:
            os.close(self._fd)
            if created:
                path.unlink(missing_ok=True)
            raise

    def alloc(self) -> int | None:
        with self._lock:
            return self._free.pop() if self._free else None

    def free(self, slot: int) -> None:
        self._check(slot)
        with self._lock:
            if slot in self._free:
                raise ValueError(f"slot {slot} already free")
            self._free.append(slot)

    def slots_free(self) -> int:
        with self._lock:
            return len(self._free)

    def offset(self, slot: int) -> int:
        self._check(slot)
        return slot * self.slot_bytes

    def view(self, slot: int) -> memoryview:
        off = self.offset(slot)
        return memoryview(self._mm)[off : off + self.slot_bytes]

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)

    def _check(self, slot: int) -> None:
        if not 0 <= slot < self.slots_total:
            raise IndexError(f"slot {slot} out of range")
