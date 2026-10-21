"""Apple memory telemetry (Metal design s8): ``vm_stat``, ``vm.swapusage``
and ``kern.memorystatus_level``.

Same contract as ``app/system/gpu.py``: parse text, degrade every unreadable
field to None, never raise. None means "not reported", never 0.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

_VM_STAT_HEADER = re.compile(r"page size of (\d+) bytes")
# One whole line each: a page count is an integer with vm_stat's trailing '.'
# (``Pages free:      4024.``); anything else on the line makes the field None.
_VM_STAT_KEYS = {
    "free": "Pages free",
    "active": "Pages active",
    "inactive": "Pages inactive",
    "wired": "Pages wired down",
    "compressed": "Pages occupied by compressor",
    "speculative": "Pages speculative",
}


def parse_vm_stat(text: str) -> dict[str, int | None]:
    """vm_stat output -> BYTES per page-count field, or None per field.

    The page size comes from the header line ``Mach Virtual Memory
    Statistics: (page size of N bytes)`` — 16384 on Apple Silicon — and is
    never assumed (no default 4096/16384): a missing or unparseable header
    makes every field None, and so does a page size of 0. vm_stat counts
    carry a trailing '.' (``Pages free: 1000.``). The byte math is exact
    integer math (no float), so a huge count can neither lose precision nor
    overflow."""
    page = _VM_STAT_HEADER.search(text)
    page_size = int(page.group(1)) if page is not None else 0
    if page_size <= 0:
        return {key: None for key in _VM_STAT_KEYS}
    out: dict[str, int | None] = {}
    for key, label in _VM_STAT_KEYS.items():
        m = re.search(rf"^{re.escape(label)}:[ \t]*(\d+)\.?[ \t]*$", text, re.MULTILINE)
        out[key] = int(m.group(1)) * page_size if m is not None else None
    return out


def _parse_swap_amount(raw: str) -> int | None:
    """One ``total = 2048.00M`` amount -> bytes. ``M`` = 2**20, ``G`` = 2**30,
    a bare number is bytes. None when the value or unit is not one of those."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([MG]?)", raw)
    if m is None:
        return None
    value = Decimal(m.group(1))
    unit = m.group(2)
    if unit == "M":
        factor = 2**20
    elif unit == "G":
        factor = 2**30
    else:
        factor = 1
    # Decimal, not float: exact for 2-decimal amounts, and a huge amount stays
    # a finite (rounded) number instead of inf, so int() cannot overflow.
    return int(value * factor)


def parse_swapusage(text: str) -> dict[str, int | None]:
    """``sysctl -n vm.swapusage`` -> {total_bytes, used_bytes, free_bytes}.

    Input like ``        total = 2048.00M  used = 1024.50M  free = 1023.50M
    (encrypted)``. Each field is parsed independently; an unparseable field
    (or a missing one) is None, never 0."""
    out = {}
    for key in ("total", "used", "free"):
        m = re.search(rf"{key} = (\S+)", text)
        out[f"{key}_bytes"] = _parse_swap_amount(m.group(1)) if m else None
    return out


def parse_memorystatus_level(text: str) -> int | None:
    """``sysctl -n kern.memorystatus_level`` -> the integer level, or None.

    The level is a non-negative integer (0 = nominal, 100 = terminal);
    anything else — ``abc``, ``-1``, ``1.5`` — is None."""
    s = text.strip()
    if not s or not s.isascii() or not s.isdigit():
        return None
    return int(s)


@dataclass
class MemoryPressure:
    """One memory reading; a field is None when that probe could not read it."""

    free: int | None = None
    active: int | None = None
    inactive: int | None = None
    wired: int | None = None
    compressed: int | None = None
    speculative: int | None = None
    swap_total_bytes: int | None = None
    swap_used_bytes: int | None = None
    swap_free_bytes: int | None = None
    pressure_level: int | None = None


def _probe(run: Callable[[list[str]], str | None], argv: list[str]) -> str | None:
    try:
        return run(argv)
    except Exception as error:  # a broken probe degrades to None, never raises
        logger.debug("apple memory probe %r failed: %s", argv, error)
        return None


def read_memory_pressure(run: Callable[[list[str]], str | None]) -> MemoryPressure:
    """Run ``vm_stat``, ``sysctl -n vm.swapusage`` and ``sysctl -n
    kern.memorystatus_level`` via ``run`` (argv -> stdout, or None on
    failure) and fold the three outputs into a MemoryPressure.

    Any failure — a raising runner, a None return, a garbage line — leaves
    just the affected fields None; this function never raises."""
    vm_stat = _probe(run, ["vm_stat"])
    swap = _probe(run, ["sysctl", "-n", "vm.swapusage"])
    level = _probe(run, ["sysctl", "-n", "kern.memorystatus_level"])

    result = MemoryPressure()
    if vm_stat is not None:
        pages = parse_vm_stat(vm_stat)
        result.free = pages["free"]
        result.active = pages["active"]
        result.inactive = pages["inactive"]
        result.wired = pages["wired"]
        result.compressed = pages["compressed"]
        result.speculative = pages["speculative"]
    if swap is not None:
        parsed = parse_swapusage(swap)
        result.swap_total_bytes = parsed["total_bytes"]
        result.swap_used_bytes = parsed["used_bytes"]
        result.swap_free_bytes = parsed["free_bytes"]
    if level is not None:
        result.pressure_level = parse_memorystatus_level(level)
    return result
