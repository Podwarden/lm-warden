"""Page-cache residency helpers (Linux only), used by the unit test and the GPU-host check."""

from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path


def resident_bytes(path: Path) -> int:
    """Bytes of ``path`` in the page cache (``mincore``); 0 for an empty file."""
    size = path.stat().st_size
    if size == 0:
        return 0
    page = mmap.PAGESIZE
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed", str(path))
        try:
            vec = ctypes.create_string_buffer((size + page - 1) // page)
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed", str(path))
            return sum(b & 1 for b in vec.raw) * page
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def fs_type(path: Path) -> str:
    """Filesystem type of the mount holding ``path``, from ``/proc/self/mountinfo``."""
    target = str(path.resolve())
    best, best_len = "", -1
    with open("/proc/self/mountinfo") as f:
        for line in f:
            left, _, right = line.partition(" - ")
            mount_point = left.split()[4].replace("\\040", " ")
            if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) and len(
                mount_point
            ) > best_len:
                best, best_len = right.split()[0], len(mount_point)
    return best
