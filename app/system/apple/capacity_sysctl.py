"""Apple unified-memory capacity probes (Metal design s7): the sysctl inputs
of the ``capacity.unified_weights_budget`` formula — ``hw.memsize_usable``
and ``iogpu.wired_limit_mb``.

Same contract as ``app/system/apple/memory.py``: parse text, degrade every
unreadable field to None, never raise. None means "not reported", never 0 —
except ``wired_limit_mb``, where 0 is a real value meaning "the OS default"
and the budget falls back to the device ceiling on it.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_UINT_MAX = 2**63 - 1

MEMSIZE_USABLE_CMD = ("sysctl", "-n", "hw.memsize_usable")
WIRED_LIMIT_CMD = ("sysctl", "-n", "iogpu.wired_limit_mb")
# Every argv read_capacity_sysctls runs — an async caller can pre-run these
# and hand the reader a lookup (app/system/gpu._apple_capacity).
CAPACITY_SYSCTL_CMDS = (MEMSIZE_USABLE_CMD, WIRED_LIMIT_CMD)


def parse_sysctl_uint(text: str) -> int | None:
    """``sysctl -n <key>`` -> a non-negative integer, or None.

    A non-empty ASCII all-digit string is the value; anything else —
    ``''``, ``-1``, ``1.5``, a run of spaces, ``abc``, non-ASCII digits, or
    a digit string with an embedded newline — is None. A value above 2**63 -
    1 is None: it cannot be a real memory size in bytes."""
    s = text.strip()
    if not s or not s.isascii() or not s.isdigit():
        return None
    value = int(s)
    if value > _UINT_MAX:
        return None
    return value


@dataclass
class CapacitySysctls:
    """One capacity reading; a field is None when that probe could not read it."""

    memsize_usable: int | None = None
    wired_limit_mb: int | None = None


def _probe(run: Callable[[list[str]], str | None], argv: list[str]) -> str | None:
    try:
        return run(argv)
    except Exception as error:  # a broken probe degrades to None, never raises
        logger.debug("apple capacity probe %r failed: %s", argv, error)
        return None


def read_capacity_sysctls(run: Callable[[list[str]], str | None]) -> CapacitySysctls:
    """Read ``hw.memsize_usable`` and ``iogpu.wired_limit_mb`` via ``run``
    (argv -> stdout, or None on failure) into a CapacitySysctls.

    Any failure — a raising runner, a None return, a garbage line — leaves
    just the affected field None; this function never raises. ``memsize_usable``
    of 0 is None (a machine with zero usable memory is an unreadable probe);
    ``wired_limit_mb`` of 0 stays 0 (it means "the OS default")."""
    memsize = parse_sysctl_uint(_probe(run, list(MEMSIZE_USABLE_CMD)) or "")
    if memsize == 0:
        memsize = None
    wired = parse_sysctl_uint(_probe(run, list(WIRED_LIMIT_CMD)) or "")
    return CapacitySysctls(memsize_usable=memsize, wired_limit_mb=wired)
