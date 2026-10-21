"""Apple Metal device probe (Metal design s7): the device ceiling
``MTLDevice.recommendedMaxWorkingSetSize`` — Apple's public, documented
answer to "how many bytes may the GPU working set hold" (12713115648 on a
16 GiB M1, about 74%).

Read through pyobjc's ``Metal`` bridge (``pyobjc-framework-Metal``, locked in
requirements-macos.txt). Same contract as the other Apple probes: never
raise; None when Metal is unavailable — Linux, no pyobjc, no Metal device —
or the reading is not a positive integer. None means "unknown", and the
capacity model falls back to its conservative hw.memsize behaviour on it.

The default device (or the fact that there is none) is resolved once per
process: importing Metal and creating the device costs tens of ms, and the
GPU table polls the budget. The ceiling itself is re-read on every call, so
a change the OS makes to it (e.g. if it follows ``iogpu.wired_limit_mb``) is
seen without a restart.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_UINT_MAX = 2**63 - 1

_lock = Lock()
_resolved = False
_device: Any = None


def _reset_cache() -> None:
    """Forget the resolved device (tests only)."""
    global _resolved, _device
    with _lock:
        _resolved = False
        _device = None


def _default_device() -> Any:
    """The system default MTLDevice, or None; resolved once, never raises."""
    global _resolved, _device
    with _lock:
        if not _resolved:
            try:
                import Metal  # pyobjc-framework-Metal; absent on Linux

                _device = Metal.MTLCreateSystemDefaultDevice()
            except Exception as error:  # ImportError, or a bridge failure
                logger.debug("Metal default device unavailable: %s", error)
                _device = None
            _resolved = True
        return _device


def device_ceiling() -> int | None:
    """``recommendedMaxWorkingSetSize`` of the default Metal device in bytes,
    or None when Metal is unavailable or the reading is not a positive int.
    Synchronous (the first call imports Metal); call it off the event loop."""
    device = _default_device()
    if device is None:
        return None
    try:
        value = device.recommendedMaxWorkingSetSize()
    except Exception as error:
        logger.debug("recommendedMaxWorkingSetSize unreadable: %s", error)
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _UINT_MAX:
        return None
    return int(value)
