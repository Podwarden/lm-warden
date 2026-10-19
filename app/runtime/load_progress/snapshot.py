"""Serialise a :class:`LoadTracker` + :class:`Verdict` into an API snapshot.

The route layer calls :func:`build_snapshot` and converts the plain dict it
returns into the pydantic ``LoadProgressOut`` response model.  Keeping the
serialiser a pure function (no I/O, no pydantic) lets it be unit-tested in
isolation from the FastAPI app and reused by the list endpoint, which maps it
over every active load.

``build_snapshot`` is **total**: it must never raise.  The UI polls it every
couple of seconds while a model loads, and a 500 mid-load would look like the
load failed.  Every tracker accessor that could conceivably raise (none do in
practice, but defensively) is guarded so a bad sample or a missing step is
serialised as ``None`` rather than propagated.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.runtime.load_progress.parse import (
    LLAMACPP_STEPS,
    STEP_LABELS,
    VLLM_STEPS,
    Step,
)

if TYPE_CHECKING:  # pragma: no cover - type-only, never imported at runtime
    from app.runtime.load_progress.policy import Verdict
    from app.runtime.load_progress.tracker import LoadTracker, VitalsSample

__all__ = ["build_snapshot"]

# Vitals windowing (brief, verbatim).
_MAX_SERIES_POINTS = 120  # downsample to ≤ 120 points for the API output
_LOG_QUIET_AGE_S = 60.0  # log age threshold for "log quiet"
_LOG_QUIET_CPU_CORES = 0.5  # CPU busy threshold for "log quiet but alive"
_LOG_QUIET_GPU_MIB = 64  # GPU-mem change threshold for "log quiet but alive"
_LOG_LINE_MAX_CHARS = 200  # last_log_line truncated to ≤ 200 chars


def build_snapshot(
    tracker: LoadTracker,
    verdict: Verdict | None,
    now: float,
    usual_total_s: float | None,
    usual_steps: dict[Step, float] | None,
    now_wall: datetime,
) -> dict[str, Any]:
    """Build the API snapshot dict for one active load.

    Total function: never raises.  Any per-step or per-vital error is swallowed
    and serialised as ``None`` / empty so a poll mid-load cannot 500.

    Arguments:
        tracker: the live :class:`LoadTracker` for this model.
        verdict: the latest :class:`Verdict` from the policy, or ``None`` if no
            tick has evaluated yet (treated as ``"progressing"``).
        now: monotonic seconds — the same clock the tracker uses.
        usual_total_s: the historical usual total load time, or ``None`` when
            no history exists yet (Task 7 fills this; the route passes ``None``
            for now unless the tracker carries it).
        usual_steps: per-step usual durations, or ``None`` when unknown.  The
            route passes the tracker's ``usual`` if it has one, else ``None``.
        now_wall: wall-clock now, for ``ready_eta`` / ``give_up_at``.
    """
    try:
        return _build_snapshot(tracker, verdict, now, usual_total_s, usual_steps, now_wall)
    except Exception:  # noqa: BLE001 — total function contract
        # Last-resort fallback: a minimal progressing snapshot so the UI still
        # gets a 200 rather than a 500.  This branch should be unreachable in
        # practice; the inner function already guards each accessor.
        started_at = tracker.started_at
        elapsed_s = max(0.0, now - started_at) if now is not None else 0.0
        return {
            "model_id": getattr(tracker, "model_id", ""),
            "backend": getattr(tracker, "backend", ""),
            "layout": None,
            "elapsed_s": elapsed_s,
            "state": "progressing",
            "stalled_for_s": 0.0,
            "give_up_at": None,
            "reason": None,
            "hint": None,
            "usual_total_s": None,
            "ready_eta": None,
            "current_step": STEP_LABELS.get(Step.START, "Start"),
            "steps": [],
            "vitals": _empty_vitals(),
        }


def _build_snapshot(
    tracker: LoadTracker,
    verdict: Verdict | None,
    now: float,
    usual_total_s: float | None,
    usual_steps: dict[Step, float] | None,
    now_wall: datetime,
) -> dict[str, Any]:
    started_at = tracker.started_at
    elapsed_s = max(0.0, now - started_at)

    # --- state from verdict (None ⇒ progressing) ----------------------------
    if verdict is None:
        state = "progressing"
        stalled_for_s = 0.0
        give_up_at_s: float | None = None
        reason: str | None = None
        hint: str | None = None
    else:
        state = verdict.state
        stalled_for_s = verdict.stalled_for_s
        give_up_at_s = verdict.give_up_at_s
        reason = verdict.reason
        hint = verdict.hint

    # give_up_at (datetime) = started_wall + give_up_at_s seconds.
    give_up_at: datetime | None = None
    if give_up_at_s is not None:
        try:
            give_up_at = tracker.started_wall + timedelta(seconds=give_up_at_s)
        except Exception:  # noqa: BLE001 — total function
            give_up_at = None

    # --- ready_eta only when usual_total_s is known -------------------------
    ready_eta: datetime | None = None
    if usual_total_s is not None:
        try:
            remaining = max(0.0, usual_total_s - elapsed_s)
            ready_eta = now_wall + timedelta(seconds=remaining)
        except Exception:  # noqa: BLE001 — total function
            ready_eta = None

    # --- steps --------------------------------------------------------------
    steps_list = _build_steps(tracker, state, now, usual_steps)
    current_step = _current_step_label(tracker)

    # --- vitals -------------------------------------------------------------
    vitals = _build_vitals(tracker, now)

    return {
        "model_id": tracker.model_id,
        "backend": tracker.backend,
        "layout": _layout(tracker),
        "elapsed_s": elapsed_s,
        "state": state,
        "stalled_for_s": stalled_for_s,
        "give_up_at": give_up_at,
        "reason": reason,
        "hint": hint,
        "usual_total_s": usual_total_s,
        "ready_eta": ready_eta,
        "current_step": current_step,
        "steps": steps_list,
        "vitals": vitals,
    }


def _steps_for_backend(tracker: LoadTracker) -> tuple[Step, ...]:
    """Return the step list for the tracker's backend."""
    if tracker.backend == "llamacpp":
        return LLAMACPP_STEPS
    return VLLM_STEPS


def _build_steps(
    tracker: LoadTracker,
    state: str,
    now: float,
    usual_steps: dict[Step, float] | None,
) -> list[dict[str, Any]]:
    """Build the per-step list with status, duration, elapsed, count, total."""
    steps_tuple = _steps_for_backend(tracker)
    usual = usual_steps or {}
    found_running = False
    out: list[dict[str, Any]] = []

    # Pre-compute which steps have explicitly started (from an event, not just
    # implicitly closed by a later step), so we can mark a never-started step
    # ``skipped`` when a later step in the order has already started.
    started_flags: dict[Step, bool] = {}
    for s in steps_tuple:
        try:
            started_flags[s] = tracker.is_step_started(s)
        except Exception:  # noqa: BLE001
            started_flags[s] = False

    for i, step in enumerate(steps_tuple):
        done = False
        try:
            done = tracker.is_step_done(step)
        except Exception:  # noqa: BLE001 — total function
            done = False

        started_at = None
        ended_at = None
        try:
            started_at = tracker.step_start_at(step)
        except Exception:  # noqa: BLE001
            started_at = None
        try:
            ended_at = tracker.step_end_at(step)
        except Exception:  # noqa: BLE001
            ended_at = None

        # A later step has started if any step after this one in the order was
        # explicitly started by an event.
        later_started = any(started_flags[s] for s in steps_tuple[i + 1 :])

        # Whether THIS step was explicitly started by an event (not just
        # implicitly closed by a later step).
        this_started = started_flags.get(step, False)

        # status
        if done and this_started:
            status = "done"
        elif done and not this_started and later_started:
            # The step was closed by a later step but never started itself —
            # the engine skipped it (e.g. TUNE on a model with no AutoTuner).
            status = "skipped"
        elif done and not this_started:
            # Done but not explicitly started and no later step — a degenerate
            # case (e.g. engine_ready closed everything at once).  Show as done
            # so the checklist still renders.
            status = "done"
        elif this_started and not found_running:
            # first not-done step that has started is running, or stalled when
            # the verdict is not progressing.
            if state != "progressing":
                status = "stalled"
            else:
                status = "running"
            found_running = True
        elif this_started and found_running:
            # a later started step while an earlier one is "running" — keep it
            # pending; the UI shows one running step at a time.
            status = "pending"
        elif not this_started and later_started:
            # This step never started but a later step did — the engine
            # skipped it (e.g. TUNE on a model with no AutoTuner).
            status = "skipped"
        elif not found_running:
            # first not-done step with no start yet — running (or stalled) so
            # the checklist has a "current" step even before the first event.
            if state != "progressing":
                status = "stalled"
            else:
                status = "running"
            found_running = True
        else:
            status = "pending"

        # duration_s / elapsed_s
        duration_s: float | None = None
        elapsed_s: float | None = None
        if started_at is not None:
            if ended_at is not None:
                duration_s = max(0.0, ended_at - started_at)
            else:
                elapsed_s = max(0.0, now - started_at)

        # count / total
        count_total: tuple[int, int] | None = None
        try:
            count_total = tracker.counter_progress(step)
        except Exception:  # noqa: BLE001
            count_total = None
        count = count_total[0] if count_total is not None else None
        total = count_total[1] if count_total is not None else None

        usual_s = usual.get(step) if usual else None

        # TUNE detail: op name and 1-based op count, for the step's
        # "op N · op_name · x/y" label in the UI.
        detail: str | None = None
        if step == Step.TUNE:
            try:
                tune_info = tracker.tune_detail()
            except Exception:  # noqa: BLE001
                tune_info = None
            if tune_info is not None:
                op_name, op_count = tune_info
                detail = f"{op_name} · op {op_count}"

        out.append(
            {
                "key": step.value,
                "label": STEP_LABELS.get(step, step.value),
                "status": status,
                "duration_s": duration_s,
                "elapsed_s": elapsed_s,
                "count": count,
                "total": total,
                "usual_s": usual_s,
                "detail": detail,
            }
        )
    return out


def _current_step_label(tracker: LoadTracker) -> str:
    """Return the label of the current (latest started, not-done) step.

    Falls back to the first step's label when nothing has started yet, so the
    UI always has a current-step string to show.
    """
    steps_tuple = _steps_for_backend(tracker)
    if not steps_tuple:
        return ""
    # Walk in reverse: the latest step that has started but not ended.
    for step in reversed(steps_tuple):
        try:
            started = tracker.step_start_at(step)
            ended = tracker.is_step_done(step)
        except Exception:  # noqa: BLE001
            continue
        if started is not None and not ended:
            return STEP_LABELS.get(step, step.value)
    # Nothing started yet — the first step is "current".
    first = steps_tuple[0]
    return STEP_LABELS.get(first, first.value)


def _layout(tracker: LoadTracker) -> str | None:
    """Layout string the tracker can compute on its own.

    The tracker knows ``dp_size`` but not the tensor-parallel size, so it
    cannot format ``"N replicas × TPm"`` / ``"TPm"``.  Always returns ``None``
    here; the route derives the TP size from the model row and overwrites
    ``layout`` on the returned dict.
    """
    return None


def _build_vitals(tracker: LoadTracker, now: float) -> dict[str, Any]:
    """Build the vitals snapshot: series, log age/line, quiet, gpu_mem_total."""
    all_samples = tracker.vitals

    # Build the per-point series: each point has t_s (seconds since load
    # start), cpu_cores (ΔCPU-s / Δt vs the previous sample), and gpu_mem_mib.
    # The first sample has cpu_cores None (no previous to diff against).
    series_raw: list[dict[str, Any]] = []
    for i, s in enumerate(all_samples):
        t_s = max(0.0, s.at - tracker.started_at)
        cpu_cores: float | None = None
        if i > 0:
            prev = all_samples[i - 1]
            if prev.cpu_seconds is not None and s.cpu_seconds is not None:
                dt = s.at - prev.at
                if dt > 0:
                    cpu_cores = max(0.0, (s.cpu_seconds - prev.cpu_seconds) / dt)
        series_raw.append({"t_s": t_s, "cpu_cores": cpu_cores, "gpu_mem_mib": s.gpu_mem_mib})

    # Downsample to ≤ 120 points: keep evenly spaced samples plus the newest.
    series = _downsample_series(series_raw, _MAX_SERIES_POINTS)

    # last_log_age_s / last_log_line
    last_log_age_s: float | None = None
    last_log_line: str | None = None
    if tracker.last_log_at is not None:
        last_log_age_s = max(0.0, now - tracker.last_log_at)
    if tracker.last_log_line is not None:
        line = tracker.last_log_line
        last_log_line = line[:_LOG_LINE_MAX_CHARS] if len(line) > _LOG_LINE_MAX_CHARS else line

    # log_quiet_normal: log age > 60 s AND (CPU busy > 0.5 cores in the last
    # 60 s OR GPU memory changed ≥ 64 MiB in the last 60 s).  Uses ALL
    # samples (not the downsampled set) for accurate 60 s windowing.
    log_quiet_normal = _log_quiet_normal(all_samples, tracker, now)

    # gpu_mem_total_mib: the summed capacity of the model's GPUs, if known.
    # Computed once by the route and stored on the tracker; None when the
    # probe failed or the tracker doesn't carry it.
    gpu_mem_total_mib = getattr(tracker, "gpu_mem_total_mib", None)

    return {
        "series": series,
        "last_log_age_s": last_log_age_s,
        "last_log_line": last_log_line,
        "log_quiet_normal": log_quiet_normal,
        "gpu_mem_total_mib": gpu_mem_total_mib,
        "gpu_mem_supported": getattr(tracker, "gpu_mem_supported", True),
    }


def _downsample_series(series: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Downsample to ≤ ``max_points``: keep evenly spaced samples plus the newest.

    If there are fewer than ``max_points`` points, return as-is.  Otherwise,
    select every ``step``-th point (evenly spaced) and always include the
    last point.  Duplicates of the last point are dropped.
    """
    if len(series) <= max_points:
        return series
    step = len(series) / max_points
    indices = [round(i * step) for i in range(max_points)]
    # Ensure the last index is the actual last point.
    if indices[-1] != len(series) - 1:
        indices[-1] = len(series) - 1
    # Deduplicate consecutive indices (small-step rounding can collide).
    out: list[dict[str, Any]] = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx:
            out.append(series[idx])
            prev_idx = idx
    return out


def _log_quiet_normal(samples: list[VitalsSample], tracker: LoadTracker, now: float) -> bool:
    """log age > 60 s AND (CPU busy > 0.5 cores in the last 60 s OR GPU memory
    changed >= 64 MiB in the last 60 s).
    """
    if tracker.last_log_at is None:
        return False
    log_age = now - tracker.last_log_at
    if log_age <= _LOG_QUIET_AGE_S:
        return False

    # CPU busy > 0.5 cores in the last 60 s: look at consecutive sample pairs
    # whose curr.at is within the last 60 s.
    cutoff = now - _LOG_QUIET_AGE_S
    cpu_busy = False
    gpu_moved = False
    for i in range(1, len(samples)):
        prev = samples[i - 1]
        curr = samples[i]
        if curr.at < cutoff:
            continue
        # CPU busy
        if not cpu_busy and prev.cpu_seconds is not None and curr.cpu_seconds is not None:
            dt = curr.at - prev.at
            if dt > 0:
                cores = (curr.cpu_seconds - prev.cpu_seconds) / dt
                if cores > _LOG_QUIET_CPU_CORES:
                    cpu_busy = True
        # GPU mem change >= 64 MiB
        if not gpu_moved and prev.gpu_mem_mib is not None and curr.gpu_mem_mib is not None:
            if abs(curr.gpu_mem_mib - prev.gpu_mem_mib) >= _LOG_QUIET_GPU_MIB:
                gpu_moved = True

    return cpu_busy or gpu_moved


def _empty_vitals() -> dict[str, Any]:
    return {
        "series": [],
        "last_log_age_s": None,
        "last_log_line": None,
        "log_quiet_normal": False,
        "gpu_mem_total_mib": None,
        "gpu_mem_supported": True,
    }
