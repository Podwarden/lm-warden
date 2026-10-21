"""Apple GPU telemetry: parse `ioreg -r -c AGXAccelerator -d 2` into
``GpuLive`` / ``GpuInfo`` (Metal design §8).

This is the drop-in Apple counterpart to the nvidia-smi parsers in
``app/system/gpu.py``: a pure "parse text, degrade unsupported fields to
None, never raise" function. The ``ioreg`` subprocess itself is a later
task (p6f); these parsers take the captured stdout so they are unit-testable
on Linux with fixture text.

Two Apple facts shape the mapping:

* ``GpuLive.memory_total_mib`` is the §7 unified-memory *budget* (passed in
  as ``budget_mib``), **not** ``hw.memsize``. Every consumer asks "how much
  can a model use", and answering 16 GB on a box that refuses at ~11 is the
  wrong answer.
* ``compute_cap`` is ``None`` always — there is no CUDA compute capability on
  Apple silicon, and ``capability_warnings(None, ...)`` returns ``[]`` by
  construction, so the whole warning layer switches itself off with zero edits.
"""

import logging
import re
from typing import Any

from app.system.gpu import GpuInfo, GpuLive

logger = logging.getLogger(__name__)


def _entry_blocks(stdout: str) -> list[tuple[str, str]]:
    """Split ioreg output into ``(object_header, dict_body)`` pairs, one per
    AGXAccelerator object.

    ioreg prints each object as a header line (``+-o AGXAcceleratorG16G
    <class AGXAcceleratorG16G, ...>`` -- the class is the concrete
    ``AGXAccelerator<gen>`` subclass) immediately followed by its ``{`` ...
    ``}`` dictionary on the next line. We find each header, take the
    dictionary that opens on the following line up to its closing brace, and
    skip the header if it has no dictionary there (a malformed object
    degrades to nothing rather than raising). The header is
    not line-anchored: the first object sits at the top of the dump and later
    ones are nested under it (``| +-o`` / ``| -o``), and a line-anchored
    ``[+-]`` would demand a literal ``+`` where the tree prints a ``-``.
    """
    entries: list[tuple[str, str]] = []
    for m in re.finditer(r"[+-]o\s+(\S+)\s+<class\s+AGXAccelerator[^\n]*", stdout):
        obj = m.group(1)
        # The dictionary must open on the very next line. Its prefix is
        # whatever tree art ioreg draws there: ``  | {`` when the object has
        # children (``-d 2``), bare ``    {`` when it has none (``-d 1``),
        # ``    | |   |   | {`` deep inside ``ioreg -l``. Never search further
        # ahead: that would hand this header the NEXT object's dictionary.
        body_match = re.match(r"\r?\n[ \t|]*\{", stdout[m.end() :])
        if body_match is None:
            continue
        open_pos = m.end() + body_match.end() - 1  # position of '{'
        depth = 0
        close_pos = None
        for i in range(open_pos, len(stdout)):
            c = stdout[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break
        if close_pos is None:
            continue
        entries.append((obj, stdout[open_pos : close_pos + 1]))
    return entries


def _props(body: str) -> dict[str, Any]:
    """Top-level ``"key" = value`` pairs in a dictionary body.

    ioreg prints one property per line, so this reads line by line. A value
    that opens a ``{`` is a nested dictionary (e.g. ``PerformanceStatistics``)
    and is collected as its brace-balanced span — the closing ``}`` line
    (and any lines in between) are consumed by the span, not treated as
    siblings. Nested spans are returned raw (still a ``{...}`` string) and
    decoded on demand; scalars are unquoted to ``str`` or kept as their raw
    ``str``.
    """
    lines = body.splitlines()
    props: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r'"([^"]+)"\s*=\s*(\S.*)$', line)
        i += 1
        if m is None:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("{"):
            # gather the nested dict: keep consuming lines until the braces
            # balance, then store the span from the first '{' to its match.
            span_parts = [val]
            depth = val.count("{") - val.count("}")
            while depth > 0 and i < len(lines):
                span_parts.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
            props.setdefault(key, "\n".join(span_parts))
        else:
            # scalar on this line: strip any trailing '},' that closes an
            # enclosing dict on the same line.
            val = val.split(",")[0].strip().rstrip("}").strip()
            if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            props.setdefault(key, val)
    return props


def _ps_field(raw: object, name: str) -> object:
    """Pull a single field out of a PerformanceStatistics raw dict string.

    Returns the decoded value (int/float/str) or None when the field is
    absent or the value is not one of the accepted scalar shapes.
    """
    if not isinstance(raw, str):
        return None
    m = re.search(r'"' + re.escape(name) + r'"\s*=\s*("[^"]*"|-?\d+(?:\.\d+)?)', raw)
    if m is None:
        return None
    tok = m.group(1)
    if tok.startswith('"'):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return None


def parse_agx_facts(stdout: str) -> list[dict[str, Any]]:
    """Per-entry static facts: one dict per AGXAccelerator object, with keys
    ``model``, ``gpu_core_count`` and ``iosource_version``. A missing field is
    ``None`` — never 0 / "" — so a later task can surface "not reported"
    rather than a fabricated number. Garbage input yields ``[]``.
    """
    facts: list[dict[str, Any]] = []
    for obj, body in _entry_blocks(stdout):
        props = _props(body)
        model = props.get("model")
        facts.append(
            {
                "object": obj,
                "model": model if isinstance(model, str) else None,
                "gpu_core_count": _as_int(props.get("gpu-core-count")),
                "iosource_version": (
                    props.get("IOSourceVersion")
                    if isinstance(props.get("IOSourceVersion"), str)
                    else None
                ),
            }
        )
    return facts


def _as_int(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def parse_ioreg_agx(stdout: str, *, budget_mib: int) -> list[GpuLive]:
    """Parse ioreg AGXAccelerator text into ``GpuLive``.

    One ``GpuLive`` per AGXAccelerator object, indexed 0.. in output order.
    ``memory_total_mib`` is ``budget_mib`` (the §7 budget), and
    ``compute_cap`` is ``None`` always. Garbage input yields ``[]`` and no
    malformed object raises.
    """
    out: list[GpuLive] = []
    for index, (_obj, body) in enumerate(_entry_blocks(stdout)):
        props = _props(body)
        name = props.get("model")
        name = name if isinstance(name, str) and name else "Apple GPU"

        ps = props.get("PerformanceStatistics")
        raw_util = _ps_field(ps, "Device Utilization %")
        raw_used = _ps_field(ps, "In use system memory")

        # GpuLive.utilization_pct and memory_used_mib are non-optional ints, so
        # a missing/unreadable reading becomes 0 (never a None that would break
        # the type, never a clamp that hides a bad value). 0 means "not
        # reported", which is the honest floor for a reading we could not take.
        if isinstance(raw_util, int | float) and 0 <= raw_util <= 100:
            utilization = int(raw_util)
        else:
            utilization = 0
        if isinstance(raw_used, int | float) and raw_used >= 0:
            used = int(raw_used) // (2**20)
        else:
            used = 0

        out.append(
            GpuLive(
                index=index,
                uuid=f"apple-gpu-{index}",
                name=name,
                memory_total_mib=budget_mib,
                memory_used_mib=used,
                memory_free_mib=max(0, budget_mib - used),
                utilization_pct=utilization,
                compute_cap=None,
            )
        )
    return out


def parse_ioreg_agx_info(stdout: str, *, budget_mib: int) -> list[GpuInfo]:
    """The ``GpuInfo`` projection of ``parse_ioreg_agx`` (no uuid/free, no
    compute_cap) — the same parse, the lighter dataclass, for the DB-backed
    sampler path."""
    out: list[GpuInfo] = []
    for g in parse_ioreg_agx(stdout, budget_mib=budget_mib):
        out.append(
            GpuInfo(
                index=g.index,
                name=g.name,
                memory_total_mib=g.memory_total_mib,
                memory_used_mib=g.memory_used_mib,
                utilization_pct=g.utilization_pct,
            )
        )
    return out
