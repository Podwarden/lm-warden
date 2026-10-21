"""Unified-memory weights budget for the Metal backend (Metal design §7).

Pure functions, no I/O. On a GPU there is a per-device VRAM number and the
whole fit question is ``file_size`` against ``fit.weights_budget_bytes`` of
it; on unified memory there is one pool shared with the OS, every other
engine and the KV cache, and nothing to own. This module computes the
bytes the OS will actually let a model claim, taking every measurement as a
number: the probe code (``recommendedMaxWorkingSetSize``,
``iogpu.wired_limit_mb``, ``hw.memsize_usable``, per-engine footprints)
lives in ``app/system/apple/`` and feeds these parameters.

``app/models/fit.py`` is deliberately untouched: ``classify_fit`` still
takes a number, so this budget is a drop-in denominator for the existing
4-way ladder. The math derivation lives on the macOS P6 backlog item; the
constants below are labelled as judgement calls the way ``fit.py`` labels
its "locked by #82" thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Bytes the OS keeps for itself and every non-Warden consumer, taken ONLY on
# the hw.memsize fallback (no Metal ceiling readable):
# ``recommendedMaxWorkingSetSize`` is already Apple's answer to "how much may
# the GPU take and leave the OS its share", so reserving again on top of it
# double-counts (M1 16 GB: 8.84 GiB instead of 11.84 GiB). These are
# JUDGEMENT CALLS, not measured limits: macOS exposes no public "reserve for
# the OS" API, so we floor a fixed 3 GiB (a desktop with Finder, a browser
# and the compressor must stay usable) and take 15% of the usable pool for
# bigger machines, keeping whichever is larger. Both are parameters on
# ``unified_weights_budget`` so settings can override them without touching
# the formula — the way fit.py's thresholds are constants a spec change
# revisits, not hidden literals.
OS_RESERVE_FLOOR_BYTES = 3 * 1024**3
OS_RESERVE_FRACTION = 0.15


def unified_device_budget(device_ceiling: int, wired_limit_mb: int) -> int:
    """``min(device_ceiling, wired_ceiling)`` of the §7 formula — the bytes the
    GPU may wire before any reserve. A wired limit of 0 is the OS default
    (the device ceiling); a nonzero one can only LOWER the budget: a limit
    raised above the ceiling is honoured only if the OS raises
    ``recommendedMaxWorkingSetSize`` with it (the ceiling is the public
    API's answer; whether it follows ``iogpu.wired_limit_mb`` is one of the
    design's open §7 measurements)."""
    wired_ceiling = wired_limit_mb * 2**20 if wired_limit_mb else device_ceiling
    return min(device_ceiling, wired_ceiling)


def unified_weights_budget(
    device_ceiling: int,
    wired_limit_mb: int,
    memsize_usable: int,
    other_engines_bytes: int,
    kv_reserve: int,
    os_reserve_floor: int = OS_RESERVE_FLOOR_BYTES,
    os_reserve_fraction: float = OS_RESERVE_FRACTION,
    *,
    os_reserve: bool = True,
) -> int:
    """Bytes available for model weights on a unified-memory machine.

    Mirrors the Metal design §7 formula::

        device_ceiling = MTLDevice.recommendedMaxWorkingSetSize   # the OS's own answer
        wired_ceiling  = iogpu.wired_limit_mb * 2^20  if nonzero else device_ceiling
        budget         = min(device_ceiling, wired_ceiling)
        os_reserve     = 0  if the ceiling is the Metal reading (it already
                                leaves the OS its share)
                         max(3 GiB, 0.15 * hw.memsize_usable)  on the hw.memsize
                                fallback (no ceiling readable)
        weights_budget = budget - os_reserve - other_engines - kv_reserve

    ``os_reserve=False`` says ``device_ceiling`` is the Metal reading; the
    default (True) is the hw.memsize fallback arithmetic. The choice is made
    in ONE place — ``UnifiedCapacity.weights_budget`` — from whether a sane
    ceiling was read.

    ``wired_limit_mb`` of 0 means "the OS default" and falls back to the
    device ceiling — never hard-code a fraction of the pool, the ceiling is
    the public API's answer and it may move. ``memsize_usable`` (not
    ``hw.memsize``) is what the probe passes; they differ by hundreds of
    MB and the reserve must not double-count.

    May return a negative integer when the reserve, other engines and KV
    together exceed the budget — callers (``classify_fit``) handle that as
    the "doesn't fit at any size" signal, exactly like
    ``fit.weights_budget_bytes`` does for the CUDA path.

    The fractional reserve is rounded UP (``math.ceil``) so the budget
    rounds DOWN: a capacity figure may understate by a byte, never
    overstate.
    """
    budget = unified_device_budget(device_ceiling, wired_limit_mb)
    reserve = (
        max(os_reserve_floor, math.ceil(os_reserve_fraction * memsize_usable)) if os_reserve else 0
    )
    return budget - reserve - other_engines_bytes - kv_reserve


def _positive(value: int | None) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


@dataclass(frozen=True)
class UnifiedCapacity:
    """One reading of the Mac's capacity probes; any field may be None (unread).

    ``memsize`` is ``hw.memsize``, ``memsize_usable`` ``hw.memsize_usable``,
    ``wired_limit_mb`` ``iogpu.wired_limit_mb`` (0 = the OS default) and
    ``device_ceiling`` ``MTLDevice.recommendedMaxWorkingSetSize``. The
    properties resolve them FAIL CLOSED — an unknown never yields a bigger
    budget than ``hw.memsize``, and an unreadable ``hw.memsize`` is 0:

    * no ``memsize`` -> budget 0 (refuse every load; nothing bounds the rest);
    * no ceiling -> ``memsize`` (the pre-P6 stand-in) LESS the OS reserve; a
      ceiling above ``memsize`` is not a sane reading and is treated the
      same way (``device_budget`` clamps to ``memsize``, the reserve applies);
    * a sane ceiling (0 < ceiling <= memsize) is used as is, with NO OS
      reserve on top — the OS already kept its share out of it;
    * an unknown wired limit is the OS default (it can only lower the
      budget, see ``unified_device_budget``);
    * the reserve basis is ``memsize_usable`` when it is a sane reading
      (0 < usable <= memsize), else ``memsize`` — the larger basis, so a
      larger OS reserve.
    """

    memsize: int | None
    memsize_usable: int | None = None
    wired_limit_mb: int | None = None
    device_ceiling: int | None = None

    @property
    def device_budget(self) -> int:
        """Bytes the GPU may claim before the OS reserve (the GPU table's
        total and the shell feed's ``usable_bytes``); 0 when unknown."""
        memsize = _positive(self.memsize)
        if not memsize:
            return 0
        ceiling = min(_positive(self.device_ceiling) or memsize, memsize)
        return unified_device_budget(ceiling, _positive(self.wired_limit_mb))

    @property
    def has_device_ceiling(self) -> bool:
        """True when a sane Metal ceiling was read (0 < ceiling <= memsize):
        the budget is then the OS's own answer and takes no OS reserve."""
        memsize = _positive(self.memsize)
        ceiling = _positive(self.device_ceiling)
        return bool(memsize and ceiling and ceiling <= memsize)

    @property
    def reserve_basis(self) -> int:
        """The pool the OS reserve is a fraction of (``hw.memsize_usable``)."""
        memsize = _positive(self.memsize)
        usable = _positive(self.memsize_usable)
        return usable if usable and usable <= memsize else memsize

    def weights_budget(self, other_engines_bytes: int = 0, kv_reserve: int = 0) -> int:
        """``unified_weights_budget`` of this reading; 0 when the device budget
        is unknown (admission refuses on <= 0). The OS reserve is taken only
        on the hw.memsize fallback, never on top of a Metal ceiling."""
        budget = self.device_budget
        if budget <= 0:
            return 0
        return unified_weights_budget(
            budget,
            0,
            self.reserve_basis,
            other_engines_bytes,
            kv_reserve,
            os_reserve=not self.has_device_ceiling,
        )
