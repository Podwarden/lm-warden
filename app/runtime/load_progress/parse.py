"""Parse engine (vLLM, llama.cpp) load logs into ``LogEvent`` records.

This module is the bottom layer of the model-load-progress feature: later tasks
feed these events into a per-load tracker that drives a step checklist in the
UI and stall detection. The parser is deliberately total — it must never raise
for any ``str`` input — so callers can stream raw log chunks through it without
worrying about exceptions.

The two public entry points are :func:`parse_vllm_line` and
:func:`parse_llamacpp_line`, each fed one *line* (after :func:`split_log_chunk`
has broken a raw chunk into its constituent lines). The split is needed because
vLLM interleaves tqdm progress bars (separated by ``\\r``) and
``(Worker_TPn pid=…)`` / ``(EngineCore_DPn pid=…)`` log prefixes on a single
physical line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

__all__ = [
    "Step",
    "STEP_LABELS",
    "VLLM_STEPS",
    "LLAMACPP_STEPS",
    "LogEvent",
    "split_log_chunk",
    "parse_vllm_line",
    "parse_llamacpp_line",
]


class Step(str, Enum):
    """A named phase of model load, shared across engine backends.

    The string value is the key used in the step checklist / UI label map.
    """

    START = "START"
    WEIGHTS = "WEIGHTS"
    COMPILE = "COMPILE"
    KV = "KV"
    TUNE = "TUNE"
    GRAPHS = "GRAPHS"
    WARMUP = "WARMUP"


STEP_LABELS: dict[Step, str] = {
    Step.START: "Start engine",
    Step.WEIGHTS: "Load weights",
    Step.COMPILE: "Compile kernels",
    Step.KV: "Size KV cache",
    Step.TUNE: "Tune kernels",
    Step.GRAPHS: "Capture CUDA graphs",
    Step.WARMUP: "Warm-up request",
}

VLLM_STEPS: tuple[Step, ...] = (
    Step.START,
    Step.WEIGHTS,
    Step.COMPILE,
    Step.KV,
    Step.TUNE,
    Step.GRAPHS,
    Step.WARMUP,
)

LLAMACPP_STEPS: tuple[Step, ...] = (
    Step.START,
    Step.WEIGHTS,
    Step.WARMUP,
)


EventKind = Literal["step_start", "step_end", "counter", "engine_ready", "layout"]


@dataclass(frozen=True)
class LogEvent:
    """One event parsed from an engine load log line.

    ``replica`` is the raw source process id — the vLLM log prefix without the
    ``pid=…`` part, plus the pid for Worker lines so distinct processes stay
    distinct:

    - ``"EngineCore_DP0"`` for an ``(EngineCore_DP0 pid=…)`` line,
    - ``"EngineCore"`` for a bare ``(EngineCore pid=…)`` line (single engine),
    - ``"Worker_TP0:931964"`` for a ``(Worker_TP0 pid=931964)`` line,
    - ``"Worker_TP0_EP0:357962"`` for ``(Worker_TP0_EP0 pid=357962)``,
    - ``"APIServer"`` / ``"ApiServer_1"`` for API-server lines,
    - ``""`` for unprefixed lines (e.g. the top-level ``non-default args:`` line).

    ``dp_size`` is set only on ``layout`` events: the ``data_parallel_size``
    value from the args dict, or 1 if absent. ``None`` for all other events.
    """

    replica: str
    kind: EventKind
    step: Step | None
    count: int | None
    total: int | None
    text: str
    dp_size: int | None = None


# ---------------------------------------------------------------------------
# chunk splitting
# ---------------------------------------------------------------------------

# A ``(Worker_TPn pid=…)`` / ``(EngineCore_DPn pid=…)`` / bare ``(EngineCore pid=…)``
# / ``(ApiServer_n pid=…)`` / ``(APIServer pid=…)`` / ``(MainProcess pid=…)``
# prefix can be glued to the tail of a tqdm progress bar with no separator, so
# we split *before* each one as well as on ``\r`` and ``\n``.  ``EngineCore\b``
# matches the bare v0.32 form without matching ``EngineCore_DPn`` (already
# covered by ``EngineCore_``).  ``MainProcess`` is vLLM 0.32's data-parallel
# top-level coordinator prefix.
_PREFIX_RE = re.compile(
    r"(?=\((?:Worker_|EngineCore\b|EngineCore_|ApiServer_|APIServer\b|MainProcess\b))"
)
_NEWLINE_RE = re.compile(r"[\r\n]")


def split_log_chunk(chunk: str) -> list[str]:
    """Split a raw log chunk into individual lines.

    Splits on ``\\r`` and ``\\n`` (tqdm progress bars are ``\\r``-separated) and
    *before* each embedded ``(Worker_`` / ``(EngineCore`` / ``(EngineCore_`` /
    ``(ApiServer_`` / ``(APIServer`` / ``(MainProcess`` prefix, which vLLM
    glues to the tail of a progress bar with no separator. Empty segments are
    dropped.
    """
    parts: list[str] = []
    for piece in _NEWLINE_RE.split(chunk):
        parts.extend(_PREFIX_RE.split(piece))
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

# Source-prefix regexes.  ``replica`` is the raw source process id — the prefix
# name (without ``pid=…``), plus ``:pid`` for Worker lines so distinct processes
# stay distinct.  For every prefixed line, ``body`` is the line WITHOUT the
# ``(prefix pid=…)`` part.
# Worker: ``(Worker_TP0 pid=9)``, ``(Worker_TP0_EP0 pid=9)``, or bare
# ``(Worker pid=9)``.  Group 1 = full name (``Worker_TP0`` or ``Worker``);
# group 2 = pid; group 3 = body.
_WORKER_PREFIX_RE = re.compile(r"^\((Worker(?:_(?:TP|EP)\d+(?:_EP\d+)*)?)\s+pid=(\d+)\)\s*(.*)$")
# EngineCore: ``(EngineCore_DP2 pid=…)`` (DP) or ``(EngineCore pid=…)`` (bare).
# Group 1 = full name (``EngineCore_DP2`` or ``EngineCore``); group 2 = body.
_ENGINECORE_PREFIX_RE = re.compile(r"^\((EngineCore(?:_DP\d+)?)\s+pid=\d+\)\s*(.*)$")
# APIServer (v0.32, unnumbered) / ApiServer_n (v0.26, numbered).
# Group 1 = full name (``APIServer`` or ``ApiServer_0``); group 2 = body.
_APISERVER_PREFIX_RE = re.compile(r"^\((APIServer|ApiServer_\d+)\s+pid=\d+\)\s*(.*)$")
# MainProcess — vLLM 0.32 data-parallel top-level coordinator.  Group 1 = body.
_MAINPROCESS_PREFIX_RE = re.compile(r"^\((MainProcess)\s+pid=\d+\)\s*(.*)$")
# ``Loading safetensors checkpoint shards:  45% Completed | 30/66 […]``
_WEIGHTS_COUNTER_RE = re.compile(r"(\d+)/(\d+)\s")
# ``Capturing CUDA graphs (…):  N%|…| X/Y […]``
_GRAPHS_COUNTER_RE = re.compile(r"Capturing CUDA graphs.*?(\d+)/(\d+)\s*\[")
# ``'data_parallel_size': N`` in the non-default args dict.
_DP_SIZE_RE = re.compile(r"'data_parallel_size':\s*(\d+)")
# ``[AutoTuner]: Tuning <op>:  NN%|…| x/y […]`` — the op name is captured; x/y
# is the per-op counter.  Each op restarts at 0/y.  The line carries no
# ``(Worker… pid=)`` prefix, but a prefix can be glued after the tqdm tail
# and ``split_log_chunk`` already splits on prefixes.
_AUTOTUNER_RE = re.compile(r"\[AutoTuner\]:\s*Tuning\s+(.+?):\s*\d+%.*?(\d+)/(\d+)\s*\[")


def _vllm_replica(line: str) -> tuple[str, str]:
    """Return ``(source_id, body)`` for a vLLM log line.

    ``source_id`` is the raw source process id — the log prefix name without
    ``pid=…``, plus ``:pid`` for Worker lines:

    - ``(EngineCore_DP0 pid=…)`` → ``"EngineCore_DP0"``
    - ``(EngineCore pid=…)`` → ``"EngineCore"``
    - ``(Worker_TP0 pid=9)`` → ``"Worker_TP0:9"``
    - ``(Worker_TP0_EP0 pid=9)`` → ``"Worker_TP0_EP0:9"``
    - ``(Worker pid=9)`` → ``"Worker:9"``
    - ``(APIServer pid=…)`` → ``"APIServer"``
    - ``(ApiServer_1 pid=…)`` → ``"ApiServer_1"``
    - ``(MainProcess pid=…)`` → ``"MainProcess"`` (vLLM 0.32 data-parallel)
    - unprefixed → ``""``

    ``body`` is the line without the ``(prefix pid=…)`` part.
    """
    m = _WORKER_PREFIX_RE.match(line)
    if m:
        name, pid, body = m.group(1), m.group(2), m.group(3)
        return f"{name}:{pid}", body
    m = _ENGINECORE_PREFIX_RE.match(line)
    if m:
        return m.group(1), m.group(2)
    m = _APISERVER_PREFIX_RE.match(line)
    if m:
        return m.group(1), m.group(2)
    m = _MAINPROCESS_PREFIX_RE.match(line)
    if m:
        return "MainProcess", m.group(2)
    return "", line


def parse_vllm_line(line: str) -> list[LogEvent]:
    """Parse one vLLM log line into zero or more :class:`LogEvent` records.

    Total function: never raises for any ``str`` input.
    """
    if not line:
        return []
    try:
        return _parse_vllm_line(line)
    except Exception:  # noqa: BLE001 — total function contract
        return []


def _parse_vllm_line(line: str) -> list[LogEvent]:
    replica, body = _vllm_replica(line)
    events: list[LogEvent] = []

    # --- layout: non-default args (pre-fork) --------------------------------
    if "non-default args:" in body:
        m = _DP_SIZE_RE.search(body)
        dp_size = int(m.group(1)) if m else 1
        events.append(LogEvent(replica, "layout", None, None, None, body, dp_size=dp_size))
        return events

    # --- WEIGHTS: shard counter ---------------------------------------------
    if "Loading safetensors checkpoint shards" in body:
        m = _WEIGHTS_COUNTER_RE.search(body)
        if m:
            count, total = int(m.group(1)), int(m.group(2))
            events.append(LogEvent(replica, "counter", Step.WEIGHTS, count, total, body))
        return events

    # --- WEIGHTS: done ------------------------------------------------------
    if "Loading weights took" in body or "Model loading took" in body:
        events.append(LogEvent(replica, "step_end", Step.WEIGHTS, None, None, body))
        return events

    # --- COMPILE: start -----------------------------------------------------
    # v0.26 logs ``Dynamo bytecode transform time`` and ``torch.compile took``;
    # the brief also names ``torch.compile takes`` (present-tense variant).
    if (
        "Dynamo bytecode transform time" in body
        or "torch.compile takes" in body
        or "torch.compile took" in body
    ):
        events.append(LogEvent(replica, "step_start", Step.COMPILE, None, None, body))
        return events

    # --- KV: end (also implies COMPILE ended) --------------------------------
    if "Available KV cache memory:" in body or "GPU KV cache size:" in body:
        events.append(LogEvent(replica, "step_end", Step.KV, None, None, body))
        events.append(LogEvent(replica, "step_end", Step.COMPILE, None, None, body))
        return events

    # --- TUNE: AutoTuner counter --------------------------------------------
    # ``[AutoTuner]: Tuning <op>:  NN%|…| x/y […]`` — no ``(Worker… pid=)``
    # prefix, but a prefix can be glued after the tqdm tail and
    # ``split_log_chunk`` already splits on prefixes.  The ``text`` field
    # carries the op name in short form (leading ``flashinfer::`` stripped).
    if "[AutoTuner]:" in body and "Tuning" in body:
        m = _AUTOTUNER_RE.search(body)
        if m:
            op_raw = m.group(1).strip()
            op_short = op_raw.removeprefix("flashinfer::")
            count, total = int(m.group(2)), int(m.group(3))
            events.append(LogEvent(replica, "counter", Step.TUNE, count, total, op_short))
        return events

    # --- GRAPHS: counter -----------------------------------------------------
    if "Capturing CUDA graphs" in body:
        m = _GRAPHS_COUNTER_RE.search(body)
        if m:
            count, total = int(m.group(1)), int(m.group(2))
            events.append(LogEvent(replica, "counter", Step.GRAPHS, count, total, body))
        return events

    # --- GRAPHS: done -------------------------------------------------------
    if "Graph capturing finished" in body:
        events.append(LogEvent(replica, "step_end", Step.GRAPHS, None, None, body))
        return events

    # --- engine_ready -------------------------------------------------------
    if "init engine (profile, create kv cache, warmup model) took" in body:
        events.append(LogEvent(replica, "engine_ready", None, None, None, body))
        return events

    return events


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------

# ``loading tensors 25%`` — the progress percentage is the only count we have;
# treat it as count/100 for the counter event.
_LLAMACPP_TENSORS_RE = re.compile(r"loading tensors\s+(\d+)%")
_LLAMACPP_TENSORS_DONE_RE = re.compile(r"loading tensors done")


def parse_llamacpp_line(line: str) -> list[LogEvent]:
    """Parse one llama.cpp ``llama-server`` log line into zero or more events.

    Total function: never raises for any ``str`` input.
    """
    if not line:
        return []
    try:
        return _parse_llamacpp_line(line)
    except Exception:  # noqa: BLE001 — total function contract
        return []


def _parse_llamacpp_line(line: str) -> list[LogEvent]:
    events: list[LogEvent] = []
    body = line

    # --- engine_ready: ``model loaded`` / ``server is listening`` -----------
    # Check the most specific terminal markers first so ``model loaded`` does
    # not get caught by the broader ``loading model`` start pattern below.
    # Case-insensitive: llama.cpp may log "Model loaded" (capital M) on some
    # builds.
    body_lower = body.lower()
    if "model loaded" in body_lower or "server is listening" in body_lower:
        events.append(LogEvent("", "engine_ready", None, None, None, body))
        return events

    # --- WEIGHTS: end (``loading model took N ms``) --------------------------
    if "loading model took" in body:
        events.append(LogEvent("", "step_end", Step.WEIGHTS, None, None, body))
        return events

    # --- WEIGHTS: tensor-load counter (``loading tensors N%``) ---------------
    if "loading tensors" in body:
        m = _LLAMACPP_TENSORS_RE.search(body)
        if m:
            pct = int(m.group(1))
            events.append(LogEvent("", "counter", Step.WEIGHTS, pct, 100, body))
            return events
        if _LLAMACPP_TENSORS_DONE_RE.search(body):
            events.append(LogEvent("", "step_end", Step.WEIGHTS, None, None, body))
            return events
        return events

    # --- WEIGHTS: start (``loading model '<path>'``) ------------------------
    if "loading model" in body.lower():
        events.append(LogEvent("", "step_start", Step.WEIGHTS, None, None, body))
        return events

    return events
