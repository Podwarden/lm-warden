"""Per-load step tracker for the model-load-progress feature.

Consumes parsed log events (from Task 1's parser) and vital-sign samples to
track which load step is running, which are done, and when anything last
moved.  Later tasks put a stall policy and a UI on top of it.

The tracker is pure: it holds no I/O, no timers, no callbacks.  Callers feed
it events and vitals, and read its state.

Key concepts
------------

- **Source** (``LogEvent.replica``): the raw source process id from the parser
  — e.g. ``"EngineCore_DP0"``, ``"Worker_TP0:931964"``, ``""`` for unprefixed
  tqdm lines.  A *source* is one process; a *replica* in the DP sense is one
  data-parallel engine instance.  The tracker counts *distinct sources* for
  step-done quorum, not "replicas" — the naming follows the parser's.
- **dp_size**: the expected number of data-parallel engines, learned from the
  first ``layout`` event's ``dp_size`` field (default 1 until one arrives).
  A step is *done* when its ``step_end`` has been seen from >= dp_size
  distinct sources, OR a later step has started/ended/counted, OR
  ``engine_ready`` from >= dp_size sources closes all vLLM steps through
  GRAPHS.
- **Counters** (WEIGHTS shards, GRAPHS): the tracker keeps the latest
  ``(count, total)`` per source.  The step's progress is the MINIMUM count
  across sources that reported for the current total.  A reset to a lower
  count with the same or a new total starts a new pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from app.runtime.load_progress.parse import VLLM_STEPS, LogEvent, Step

__all__ = ["VitalsSample", "LoadTracker"]

# Vitals thresholds (brief, verbatim):
_GPU_MEM_CHANGE_MIB = 64  # GPU-mem change >= 64 MiB counts as progress
_READ_BYTES_INCR_BYTES = 16 * 1024 * 1024  # read-bytes increase >= 16 MiB
_CPU_THRESHOLD_FACTOR = 0.5  # CPU-seconds increase >= 0.5 * sample interval


@dataclass
class VitalsSample:
    """One vital-sign sample from the host.

    ``at`` is a monotonic clock (epoch seconds, UTC).  Any field may be
    ``None`` when the host could not measure it — ``None`` vitals never count
    as progress.
    """

    at: float
    cpu_seconds: float | None
    gpu_mem_mib: int | None
    read_bytes: int | None


# ---------------------------------------------------------------------------
# Per-step state
# ---------------------------------------------------------------------------


@dataclass
class _StepState:
    """Mutable state for one step."""

    started_at: float | None = None
    ended_at: float | None = None
    # True when the step was started by an explicit event (step_start,
    # counter, or step_end from a source).  False when the step was only
    # implicitly closed by a later step starting (e.g. TUNE on a model with
    # no AutoTuner).  The snapshot uses this to mark such steps ``skipped``.
    explicitly_started: bool = False
    # Sources that have reported a step_end for this step.
    end_sources: set[str] = field(default_factory=set)
    # Counter state per source: {source: (count, total)}.
    # For GRAPHS, a "pass" resets: when count drops below the current pass's
    # max count for that source, we treat it as a new pass.
    counters: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Track the current pass's max count per source (for reset detection).
    counter_pass_max: dict[str, int] = field(default_factory=dict)
    # Timestamp of the last counter advance (for last_progress_at).
    last_counter_at: float | None = None
    # Timestamp of the last activity (start, counter, or end) for this step.
    last_activity_at: float | None = None
    # TUNE only: the current op name (short form, stripped of
    # ``flashinfer::``) and the 1-based count of distinct ops seen.  Each op
    # restarts at 0/y; a new op name increments ``tune_op_count``.
    tune_op: str | None = None
    tune_op_count: int = 0


class LoadTracker:
    """Tracks the load steps for one model load.

    Pure: no I/O, no timers.  Feed events and vitals, read state.
    """

    def __init__(
        self,
        *,
        model_id: str,
        generation: int,
        backend: str,
        started_at: float,
        started_wall: datetime,
        usual: dict[Step, float] | None = None,
        usual_total_s: float | None = None,
        dp_size: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.generation = generation
        self.backend = backend
        self.started_at = started_at
        self.started_wall = started_wall
        self.usual = usual or {}
        #: Historical usual total load time (median of past loads), or None
        #: when no history exists. Set at construction from LoadHistoryRepo;
        #: the snapshot route reads it so ``usual_total_s`` reaches the API.
        self.usual_total_s = usual_total_s
        #: Summed GPU memory capacity (MiB) over the model's gpu_indices, or
        #: None when the probe failed or the value hasn't been computed.  Filled
        #: lazily by the load-progress route on the first snapshot (see
        #: ``_gpu_mem_total_once``), so every tracker gets it, including a
        #: watchdog restart; the snapshot reads it into ``gpu_mem_total_mib``.
        self.gpu_mem_total_mib: int | None = None
        #: True once the route has tried the GPU capacity probe for this load
        #: (successfully or not), so a failed probe is not retried every poll.
        self.gpu_mem_probed: bool = False
        #: False when the engine driver cannot attribute GPU memory at all
        #: (the docker driver). The UI then says "not available" instead of
        #: waiting for samples that will never come.
        self.gpu_mem_supported: bool = True

        # dp_size: constructor override (for tests) or learned from layout event.
        self._dp_size: int = dp_size if dp_size is not None else 1

        # Per-step state, keyed by Step.
        self._steps: dict[Step, _StepState] = {s: _StepState() for s in VLLM_STEPS}

        # START step starts immediately at started_at.
        self._steps[Step.START].started_at = started_at
        self._steps[Step.START].explicitly_started = True

        # Vitals: every sample for the whole load (the series is downsampled on output).
        self.vitals: list[VitalsSample] = []

        # Log tracking.
        self.last_log_at: float | None = None
        self.last_log_line: str | None = None

        # engine_ready sources (for closing all vLLM steps).
        self._ready_sources: set[str] = set()

        # The "current" step — the latest step that has started but not ended.
        # Used to detect "a later step started/ended/counted closes earlier steps."
        self._max_started_step_index: int = 0  # index into VLLM_STEPS

        # Profiling pass: on vLLM 0.32, a CUDA-graph capture pass runs BEFORE
        # the KV-size line as part of vLLM's memory profiling.  Graph-capture
        # counter/step_end events seen before KV is done belong to the KV
        # step, not GRAPHS.  Set to True when the KV step_end arrives.
        self._kv_done: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dp_size(self) -> int:
        return self._dp_size

    # ------------------------------------------------------------------
    # Event feeding
    # ------------------------------------------------------------------

    def feed(self, events: list[LogEvent], now: float, line: str) -> None:
        """Process a batch of parsed events and update step state.

        Also updates ``last_log_at`` / ``last_log_line`` for any non-empty
        line.
        """
        if line:
            self.last_log_at = now
            self.last_log_line = line

        for ev in events:
            self._handle_event(ev, now)

    def _handle_event(self, ev: LogEvent, now: float) -> None:
        if ev.kind == "layout":
            # Learn dp_size from the first layout event.
            if ev.dp_size is not None:
                self._dp_size = ev.dp_size
            return

        if ev.kind == "engine_ready":
            self._ready_sources.add(ev.replica)
            if len(self._ready_sources) >= self._dp_size:
                self._close_all_vllm_steps(now)
            return

        if ev.step is None:
            return

        step = ev.step

        # Profiling pass: on vLLM 0.32, a CUDA-graph capture pass runs BEFORE
        # the KV-size line.  Graph-capture counter/step_end events seen before
        # KV is done belong to the KV step, not GRAPHS.
        profiling = step == Step.GRAPHS and not self._kv_done
        if profiling:
            step = Step.KV

        step_idx = self._step_index(step)

        if ev.kind == "step_start":
            self._start_step(step, now, step_idx)
        elif ev.kind == "step_end" and profiling:
            # A profiling-pass "Graph capturing finished" is KV progress, never
            # KV's end: KV ends only on the KV-size line (or the other KV end
            # rule), so this adds no end source and does not close KV.
            kv = self._steps[Step.KV]
            if kv.started_at is None:
                kv.started_at = now
            kv.explicitly_started = True
            kv.last_activity_at = now
        elif ev.kind == "step_end":
            self._end_step(step, now, step_idx, ev.replica)
            # Set _kv_done only when the ORIGINAL event's step is KV (the
            # "GPU KV cache size" / "Available KV cache memory" line).
            if ev.step == Step.KV:
                self._kv_done = True
        elif ev.kind == "counter":
            if step == Step.TUNE:
                self._update_tune_counter(step, now, step_idx, ev)
            else:
                self._update_counter(step, now, step_idx, ev.replica, ev.count, ev.total)

    # ------------------------------------------------------------------
    # Step state helpers
    # ------------------------------------------------------------------

    def _step_index(self, step: Step) -> int:
        """Return the index of ``step`` in VLLM_STEPS, or -1 if not present."""
        for i, s in enumerate(VLLM_STEPS):
            if s == step:
                return i
        return -1

    def _start_step(self, step: Step, now: float, step_idx: int) -> None:
        state = self._steps[step]
        if state.started_at is None:
            state.started_at = now
        state.explicitly_started = True
        state.last_activity_at = now
        # A later step starting closes all earlier steps.
        if step_idx > self._max_started_step_index:
            self._close_steps_before_idx(step_idx, now)
            self._max_started_step_index = step_idx

    def _end_step(self, step: Step, now: float, step_idx: int, source: str) -> None:
        state = self._steps[step]
        state.explicitly_started = True
        if state.started_at is None:
            # If the previous step has ended, use its end time as this step's
            # start time — the step implicitly started when the previous one
            # ended.  Otherwise, use the previous step's last activity time,
            # or now as a last resort.
            if step_idx > 0:
                prev = self._steps[VLLM_STEPS[step_idx - 1]]
                if prev.ended_at is not None:
                    state.started_at = prev.ended_at
                elif prev.last_activity_at is not None:
                    state.started_at = prev.last_activity_at
                else:
                    state.started_at = now
            else:
                state.started_at = now
        state.end_sources.add(source)
        state.last_activity_at = now
        # A later step ending closes all earlier steps.
        if step_idx > self._max_started_step_index:
            self._close_steps_before_idx(step_idx, now)
            self._max_started_step_index = step_idx
        # Done if >= dp_size distinct sources sent step_end, OR (for WEIGHTS)
        # if any step_end was seen — weights loading is a point-in-time event:
        # once any worker reports "Loading weights took" / "Model loading
        # took", the weights are loaded for that replica and the step is done.
        # Other steps (KV, GRAPHS) need the full dp_size quorum.
        if len(state.end_sources) >= self._dp_size or step == Step.WEIGHTS:
            if state.ended_at is None:
                state.ended_at = now
        # Close earlier steps that haven't ended yet (steps with index < step_idx).
        self._close_steps_before_idx(step_idx, now)

    def _update_counter(
        self,
        step: Step,
        now: float,
        step_idx: int,
        source: str,
        count: int | None,
        total: int | None,
    ) -> None:
        if count is None or total is None:
            return
        state = self._steps[step]
        if state.started_at is None:
            state.started_at = now
        state.explicitly_started = True

        # START ends at the first WEIGHTS event of any kind.
        if step == Step.WEIGHTS:
            self._close_step(Step.START, now)

        # A later step counting closes all earlier steps.
        if step_idx > self._max_started_step_index:
            self._close_steps_before_idx(step_idx, now)
            self._max_started_step_index = step_idx

        # Counter reset detection for GRAPHS: if the count dropped below the
        # current pass's max count for this source, start a new pass.
        prev = state.counters.get(source)
        if prev is not None:
            prev_count, prev_total = prev
            pass_max = state.counter_pass_max.get(source, 0)
            # Reset: count went backwards (lower than what we've seen in this pass)
            # or total changed.
            if count < pass_max or total != prev_total:
                # New pass — reset all sources' counters for this step.
                state.counters.clear()
                state.counter_pass_max.clear()
        state.counters[source] = (count, total)
        state.counter_pass_max[source] = max(state.counter_pass_max.get(source, 0), count)
        state.last_counter_at = now
        state.last_activity_at = now

    def _update_tune_counter(
        self,
        step: Step,
        now: float,
        step_idx: int,
        ev: LogEvent,
    ) -> None:
        """Handle a TUNE (AutoTuner) counter event.

        AutoTuner lines carry no ``(Worker… pid=)`` prefix, so ``ev.replica``
        is ``""`` — one source.  The ``text`` field carries the op name (short
        form).  Each op restarts at 0/y; a new op name increments the 1-based
        ``tune_op_count``.
        """
        if ev.count is None or ev.total is None:
            return
        state = self._steps[step]
        if state.started_at is None:
            state.started_at = now
        state.explicitly_started = True

        # A later step counting closes all earlier steps.
        if step_idx > self._max_started_step_index:
            self._close_steps_before_idx(step_idx, now)
            self._max_started_step_index = step_idx

        # Track the current op name and count of distinct ops.
        op_name = ev.text
        if op_name and op_name != state.tune_op:
            state.tune_op = op_name
            state.tune_op_count += 1

        # Store the counter under the single source ("").
        state.counters[""] = (ev.count, ev.total)
        state.last_counter_at = now
        state.last_activity_at = now

    def _close_step(self, step: Step, now: float) -> None:
        state = self._steps[step]
        if state.started_at is None:
            state.started_at = now
        if state.ended_at is None:
            state.ended_at = now

    def _close_steps_before_idx(self, up_to_idx: int, now: float) -> None:
        """Close all steps with index < up_to_idx.

        Steps that haven't started yet get ``started_at`` set from the
        previous step's end time or last activity time (if available) so they
        have a non-negative duration.  This handles steps like KV that are
        only implicitly ended by a later step without an explicit start.
        """
        for i, s in enumerate(VLLM_STEPS):
            if i >= up_to_idx:
                break
            state = self._steps[s]
            if state.started_at is None:
                if i > 0:
                    prev = self._steps[VLLM_STEPS[i - 1]]
                    if prev.ended_at is not None:
                        state.started_at = prev.ended_at
                    elif prev.last_activity_at is not None:
                        state.started_at = prev.last_activity_at
                    else:
                        state.started_at = now
                else:
                    state.started_at = now
            if state.ended_at is None:
                state.ended_at = now

    def _close_all_vllm_steps(self, now: float) -> None:
        """engine_ready closes all vLLM steps through GRAPHS."""
        for s in VLLM_STEPS:
            if s == Step.WARMUP:
                break
            state = self._steps[s]
            if state.started_at is None:
                state.started_at = now
            if state.ended_at is None:
                state.ended_at = now

    # ------------------------------------------------------------------
    # Vitals
    # ------------------------------------------------------------------

    def add_vitals(self, s: VitalsSample) -> None:
        """Add a vitals sample. Every sample is kept for the whole load."""
        self.vitals.append(s)

    # ------------------------------------------------------------------
    # last_progress_at
    # ------------------------------------------------------------------

    def last_progress_at(self, *, include_cpu: bool) -> float:
        """Return the latest timestamp at which "progress" was observed.

        Progress is the latest of:
        - last log line (``last_log_at``),
        - step change (a step starting or ending),
        - counter advance (a WEIGHTS/GRAPHS counter ticking up),
        - GPU-mem change >= 64 MiB,
        - read-bytes increase >= 16 MiB,
        - (only if ``include_cpu``) CPU-seconds increase >= 0.5 * sample interval.

        ``None`` vitals never count.  Returns ``started_at`` when nothing has
        progressed yet (the START step's ``started_at`` is set in ``__init__``,
        so there is always at least one candidate).
        """
        candidates: list[float] = []

        # Log line
        if self.last_log_at is not None:
            candidates.append(self.last_log_at)

        # Step changes (start/end times)
        for state in self._steps.values():
            if state.started_at is not None:
                candidates.append(state.started_at)
            if state.ended_at is not None:
                candidates.append(state.ended_at)

        # Counter advances (last counter tick per step).
        for state in self._steps.values():
            if state.last_counter_at is not None:
                candidates.append(state.last_counter_at)

        # Vitals-based progress
        if len(self.vitals) >= 2:
            for i in range(1, len(self.vitals)):
                prev = self.vitals[i - 1]
                curr = self.vitals[i]
                interval = curr.at - prev.at
                if interval <= 0:
                    continue

                # GPU-mem change >= 64 MiB
                if prev.gpu_mem_mib is not None and curr.gpu_mem_mib is not None:
                    if abs(curr.gpu_mem_mib - prev.gpu_mem_mib) >= _GPU_MEM_CHANGE_MIB:
                        candidates.append(curr.at)
                        continue  # already counts; skip CPU check for this pair

                # Read-bytes increase >= 16 MiB
                if prev.read_bytes is not None and curr.read_bytes is not None:
                    if curr.read_bytes - prev.read_bytes >= _READ_BYTES_INCR_BYTES:
                        candidates.append(curr.at)
                        continue

                # CPU-seconds increase >= 0.5 * interval (only if include_cpu)
                if include_cpu and prev.cpu_seconds is not None and curr.cpu_seconds is not None:
                    threshold = _CPU_THRESHOLD_FACTOR * interval
                    if curr.cpu_seconds - prev.cpu_seconds >= threshold:
                        candidates.append(curr.at)

        return max(candidates) if candidates else 0.0

    # ------------------------------------------------------------------
    # Public state queries
    # ------------------------------------------------------------------

    def is_step_done(self, step: Step) -> bool:
        """Return True if the step has ended."""
        state = self._steps.get(step)
        if state is None:
            return False
        return state.ended_at is not None

    def is_step_started(self, step: Step) -> bool:
        """Return True if the step was explicitly started by an event
        (step_start, counter, or step_end), not just implicitly closed by a
        later step."""
        state = self._steps.get(step)
        if state is None:
            return False
        return state.explicitly_started

    def step_end_at(self, step: Step) -> float | None:
        """Return the end timestamp of a step, or None if not ended."""
        state = self._steps.get(step)
        if state is None:
            return None
        return state.ended_at

    def step_start_at(self, step: Step) -> float | None:
        """Return the start timestamp of a step, or None if not started."""
        state = self._steps.get(step)
        if state is None:
            return None
        return state.started_at

    def counter_progress(self, step: Step) -> tuple[int, int] | None:
        """Return (count, total) for a step's counter progress.

        The count is the MINIMUM count across sources that reported for the
        current total.  Returns None if no counters have been seen for the
        step.
        """
        state = self._steps.get(step)
        if state is None or not state.counters:
            return None
        # Find the current total (the total from the most recent counter update).
        # If totals differ between sources, use the total from the latest pass.
        # Group by total, take the group matching the latest pass.
        totals = {t for _, t in state.counters.values()}
        if len(totals) == 1:
            total = totals.pop()
            counts = [c for c, t in state.counters.values() if t == total]
            return (min(counts), total)
        # Multiple totals — pick the total that has the most sources reporting.
        total_counts = Counter(t for _, t in state.counters.values())
        total = total_counts.most_common(1)[0][0]
        counts = [c for c, t in state.counters.values() if t == total]
        return (min(counts), total)

    def tune_detail(self) -> tuple[str, int] | None:
        """Return ``(op_name, op_count)`` for the TUNE step, or ``None``.

        ``op_name`` is the current op's short-form name; ``op_count`` is the
        1-based count of distinct ops seen.  ``None`` when TUNE has not
        started (no AutoTuner lines yet).
        """
        state = self._steps.get(Step.TUNE)
        if state is None or state.tune_op is None:
            return None
        return (state.tune_op, state.tune_op_count)

    def step_durations(self) -> dict[Step, float]:
        """Return durations for steps that have both started and ended."""
        result: dict[Step, float] = {}
        for step, state in self._steps.items():
            if state.started_at is not None and state.ended_at is not None:
                result[step] = state.ended_at - state.started_at
        return result

    # ------------------------------------------------------------------
    # Warmup / done
    # ------------------------------------------------------------------

    def mark_warmup_started(self, now: float) -> None:
        """Start the WARMUP step, closing everything before it."""
        self._close_all_vllm_steps(now)
        state = self._steps[Step.WARMUP]
        if state.started_at is None:
            state.started_at = now

    def mark_done(self, now: float) -> None:
        """End the WARMUP step (the load is complete)."""
        state = self._steps[Step.WARMUP]
        if state.started_at is None:
            state.started_at = now
        state.ended_at = now

    def is_done(self) -> bool:
        """True after :meth:`mark_done` ended the WARMUP step."""
        return self._steps[Step.WARMUP].ended_at is not None
