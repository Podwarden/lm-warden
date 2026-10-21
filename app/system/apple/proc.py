"""Apple per-process footprint (Metal design s8): ``libproc``'s
``proc_pid_rusage`` -> ``ri_phys_footprint``.

Same contract as ``app/system/gpu.py``: degrade every unreadable footprint to
None, never raise. None means "not reported", never 0. The ``libproc`` handle
is injectable so the whole contract is unit-tested off darwin; on darwin it is
the real ctypes handle to ``/usr/lib/libproc.dylib``."""

import ctypes
import logging
import sys
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# RUSAGE_INFO_V2 is the rusage_info flavour that carries ri_phys_footprint.
_RUSAGE_INFO_V2 = 2
_LIBPROC_PATH = "/usr/lib/libproc.dylib"


class _RusageInfoV2(ctypes.Structure):
    """``struct rusage_info_v2`` from sys/resource.h, field for field:
    ``ri_uuid`` then 18 ``unsigned long long`` fields, the 8th of which is
    ``ri_phys_footprint`` (bytes), the per-process memory footprint. The
    layout is pinned by a test to ``sizeof == 16`` + ``18 * 8 == 160`` bytes
    with ``ri_phys_footprint`` at byte offset 72 — a deviating field list or
    an assumed alignment would read the wrong bytes on darwin."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_ulonglong),
        ("ri_system_time", ctypes.c_ulonglong),
        ("ri_pkg_idle_wkups", ctypes.c_ulonglong),
        ("ri_interrupt_wkups", ctypes.c_ulonglong),
        ("ri_pageins", ctypes.c_ulonglong),
        ("ri_wired_size", ctypes.c_ulonglong),
        ("ri_resident_size", ctypes.c_ulonglong),
        ("ri_phys_footprint", ctypes.c_ulonglong),
        ("ri_proc_start_abstime", ctypes.c_ulonglong),
        ("ri_proc_exit_abstime", ctypes.c_ulonglong),
        ("ri_child_user_time", ctypes.c_ulonglong),
        ("ri_child_system_time", ctypes.c_ulonglong),
        ("ri_child_pkg_idle_wkups", ctypes.c_ulonglong),
        ("ri_child_interrupt_wkups", ctypes.c_ulonglong),
        ("ri_child_pageins", ctypes.c_ulonglong),
        ("ri_child_elapsed_abstime", ctypes.c_ulonglong),
        ("ri_diskio_bytesread", ctypes.c_ulonglong),
        ("ri_diskio_byteswritten", ctypes.c_ulonglong),
    ]


def _load_libproc() -> ctypes.CDLL | None:
    """The real libproc ctypes handle on darwin, else None. Loading never
    raises: a missing library or a failed dlopen degrades to None."""
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL(_LIBPROC_PATH)
    except Exception as error:  # a broken dlopen degrades to None, never raises
        logger.debug("apple proc libproc load failed: %s", error)
        return None
    # proc_pid_rusage(pid, flavor, rusage_info_t *); ctypes would otherwise
    # pass the struct by value (truncated) and read the return as c_int.
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    return lib


def phys_footprint(pid: int, *, libproc: ctypes.CDLL | None = None) -> int | None:
    """``proc_pid_rusage`` for ``pid`` -> ``ri_phys_footprint`` in bytes, or
    None.

    ``libproc`` is the injectable handle (a fake in tests, typed as ``CDLL``
    because that is what the real path passes). With no injection, it is the
    real libproc, loaded only on darwin. None is returned — and the libproc
    never invoked — when ``pid`` is not positive, when there is no libproc
    (off darwin with no injection), when the call returns nonzero, or on any
    exception. This function never raises."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if libproc is None:
        libproc = _load_libproc()
    if libproc is None:
        return None
    buf = _RusageInfoV2()
    try:
        rc = libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(buf))
    except Exception as error:  # a broken call degrades to None, never raises
        logger.debug("apple proc phys_footprint(%s) failed: %s", pid, error)
        return None
    if rc != 0:
        return None
    return int(buf.ri_phys_footprint)


def footprints(pids: Sequence[int]) -> dict[int, int | None]:
    """``{pid: phys_footprint(pid)}`` for each pid, in input order.

    Delegates per pid, so every None/never-raise guarantee of
    ``phys_footprint`` holds for each entry; an unparseable pid is None."""
    out: dict[int, int | None] = {}
    for pid in pids:
        out[pid] = phys_footprint(pid)
    return out
