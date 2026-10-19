"""Tests for the per-load step tracker (Task 2 of the model-load-progress plan)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from app.runtime.load_progress.parse import (
    LogEvent,
    Step,
    parse_vllm_line,
    split_log_chunk,
)
from app.runtime.load_progress.tracker import LoadTracker, VitalsSample

_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "engine_logs"

# Timestamps in vLLM logs are ``MM-DD HH:MM:SS`` (no year).  We pick a fixed
# year so the parsed monotonic clock is stable; the year doesn't matter for
# duration math as long as no log crosses Dec 31 → Jan 1.
_LOG_YEAR = 2026

_TS_RE = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def _parse_ts(line: str, prev: float | None) -> float:
    """Return the epoch float for the line's ``MM-DD HH:MM:SS`` timestamp.

    Lines without a timestamp reuse ``prev`` (or 0.0 if none seen yet).
    """
    m = _TS_RE.search(line)
    if not m:
        return prev if prev is not None else 0.0
    mo, d, h, mi, s = (int(g) for g in m.groups())
    dt = datetime(_LOG_YEAR, mo, d, h, mi, s, tzinfo=UTC)
    return dt.timestamp()


def FIXTURE(name: str) -> str:
    """Read a fixture log preserving raw bytes (``\\r`` must survive)."""
    p = _FIXTURE_DIR / name
    if not p.suffix:
        p = p.with_suffix(".log")
    return p.read_bytes().decode("utf-8", "replace")


def _replay(name: str) -> tuple[LoadTracker, list[float]]:
    """Replay a fixture through a LoadTracker, returning (tracker, timestamps).

    Each line is split with :func:`split_log_chunk`, parsed, and fed to the
    tracker with the timestamp parsed from that line.  Returns the tracker
    and the list of per-event timestamps (for gap analysis in tests).
    """
    text = FIXTURE(name)
    # First pass: find the first timestamp to use as started_at.
    first_ts: float | None = None
    prev_ts: float | None = None
    for seg in split_log_chunk(text):
        ts = _parse_ts(seg, prev_ts)
        if _TS_RE.search(seg):
            prev_ts = ts
            if first_ts is None:
                first_ts = ts
    tracker = LoadTracker(
        model_id="test",
        generation=0,
        backend="vllm",
        started_at=first_ts if first_ts is not None else 0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Second pass: feed events.
    prev_ts = None
    events_with_ts: list[tuple[float, list[LogEvent], str]] = []
    for seg in split_log_chunk(text):
        ts = _parse_ts(seg, prev_ts)
        if _TS_RE.search(seg):
            prev_ts = ts
        events = parse_vllm_line(seg)
        events_with_ts.append((ts, events, seg))
    for ts, events, seg in events_with_ts:
        tracker.feed(events, now=ts, line=seg)
    return tracker, [ts for ts, _, _ in events_with_ts]


# ---------------------------------------------------------------------------
# VitalsSample
# ---------------------------------------------------------------------------


def test_vitals_sample_dataclass_fields():
    s = VitalsSample(at=1.0, cpu_seconds=2.0, gpu_mem_mib=100, read_bytes=1024)
    assert s.at == 1.0
    assert s.cpu_seconds == 2.0
    assert s.gpu_mem_mib == 100
    assert s.read_bytes == 1024


def test_vitals_sample_allows_none_fields():
    s = VitalsSample(at=1.0, cpu_seconds=None, gpu_mem_mib=None, read_bytes=None)
    assert s.cpu_seconds is None
    assert s.gpu_mem_mib is None
    assert s.read_bytes is None


# ---------------------------------------------------------------------------
# dp_size from layout event
# ---------------------------------------------------------------------------


def test_dp_size_from_layout_event():
    """LoadTracker learns dp_size from the first layout event."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert t.dp_size == 1  # default before layout event
    t.feed(
        [LogEvent("", "layout", None, None, None, "non-default args: {...}", dp_size=3)],
        now=1.0,
        line="non-default args",
    )
    assert t.dp_size == 3


def test_dp_size_constructor_override():
    """The optional dp_size constructor arg overrides the default for tests."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
        dp_size=4,
    )
    assert t.dp_size == 4
    # A layout event still overrides the constructor default
    t.feed(
        [LogEvent("", "layout", None, None, None, "args", dp_size=2)],
        now=1.0,
        line="args",
    )
    assert t.dp_size == 2


# ---------------------------------------------------------------------------
# Step done needs dp_size distinct sources
# ---------------------------------------------------------------------------


def test_step_done_needs_dp_size_distinct_sources():
    """A step is done when its step_end is seen from >= dp_size distinct sources."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Layout: dp_size=3
    t.feed(
        [LogEvent("", "layout", None, None, None, "args", dp_size=3)],
        now=0.0,
        line="args",
    )
    # Two of three EngineCore sources send KV step_end → KV still running
    t.feed([LogEvent("EngineCore_DP0", "step_end", Step.KV, None, None, "kv")], now=1.0, line="kv0")
    t.feed([LogEvent("EngineCore_DP1", "step_end", Step.KV, None, None, "kv")], now=2.0, line="kv1")
    assert not t.is_step_done(Step.KV)
    # Third source → done
    t.feed([LogEvent("EngineCore_DP2", "step_end", Step.KV, None, None, "kv")], now=3.0, line="kv2")
    assert t.is_step_done(Step.KV)


def test_step_done_only_when_all_replicas_pass():
    """Brief's test: 3 replicas, two send KV step_end → KV still running; third → done."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
        dp_size=3,
    )
    t.feed([LogEvent("r0", "step_end", Step.KV, None, None, "kv")], now=1.0, line="kv0")
    t.feed([LogEvent("r1", "step_end", Step.KV, None, None, "kv")], now=2.0, line="kv1")
    assert not t.is_step_done(Step.KV)
    t.feed([LogEvent("r2", "step_end", Step.KV, None, None, "kv")], now=3.0, line="kv2")
    assert t.is_step_done(Step.KV)


# ---------------------------------------------------------------------------
# Later step end closes earlier steps
# ---------------------------------------------------------------------------


def test_later_step_end_closes_earlier_steps():
    """A step_end for a later step closes all earlier steps."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Feed a WEIGHTS counter (starts WEIGHTS, ends START)
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 1, 10, "w")], now=1.0, line="w")
    assert t.is_step_done(Step.START)
    # KV step_end (which also implies COMPILE end) should close WEIGHTS and COMPILE
    t.feed([LogEvent("r0", "step_end", Step.KV, None, None, "kv")], now=2.0, line="kv")
    assert t.is_step_done(Step.WEIGHTS)
    assert t.is_step_done(Step.COMPILE)
    assert t.is_step_done(Step.KV)


def test_later_step_start_closes_earlier_steps():
    """A step_start for a later step also closes all earlier steps."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # COMPILE step_start should close WEIGHTS (and START was already closed by WEIGHTS counter)
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 1, 10, "w")], now=1.0, line="w")
    assert not t.is_step_done(Step.WEIGHTS)
    t.feed([LogEvent("r0", "step_start", Step.COMPILE, None, None, "c")], now=2.0, line="c")
    assert t.is_step_done(Step.WEIGHTS)


def test_engine_ready_closes_all_vllm_steps():
    """engine_ready from >= dp_size distinct sources closes all vLLM steps through GRAPHS."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.feed([LogEvent("r0", "engine_ready", None, None, None, "ready")], now=5.0, line="ready")
    for step in (Step.START, Step.WEIGHTS, Step.COMPILE, Step.KV, Step.TUNE, Step.GRAPHS):
        assert t.is_step_done(step), f"{step} should be done after engine_ready"


# ---------------------------------------------------------------------------
# Counter progress — minimum across sources
# ---------------------------------------------------------------------------


def test_counter_progress_is_min_across_sources():
    """Brief's test: counter progress is the slowest replica."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
        dp_size=2,
    )
    # Two sources report WEIGHTS counters with different progress
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 30, 66, "w")], now=1.0, line="w0")
    t.feed([LogEvent("r1", "counter", Step.WEIGHTS, 20, 66, "w")], now=2.0, line="w1")
    prog = t.counter_progress(Step.WEIGHTS)
    assert prog is not None
    count, total = prog
    assert count == 20  # min across sources
    assert total == 66


def test_counter_progress_single_source():
    """With only one source, its count is the progress."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 5, 10, "w")], now=1.0, line="w")
    prog = t.counter_progress(Step.WEIGHTS)
    assert prog == (5, 10)


def test_counter_progress_none_when_no_counters():
    """No counter events for a step → None."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert t.counter_progress(Step.WEIGHTS) is None
    assert t.counter_progress(Step.GRAPHS) is None


# ---------------------------------------------------------------------------
# Graph capture second pass resets progress
# ---------------------------------------------------------------------------


def test_graph_capture_second_pass_resets_progress():
    """When a second graph-capture pass starts (count resets to a lower value
    with the same or new total), the tracker tracks the current pass."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Mark KV as done so GRAPHS counters are not routed to KV (profiling pass).
    t.feed([LogEvent("r0", "step_end", Step.KV, None, None, "kv")], now=0.0, line="kv")
    # First pass: 51/51 done
    t.feed([LogEvent("r0", "counter", Step.GRAPHS, 51, 51, "g1")], now=1.0, line="g1")
    prog = t.counter_progress(Step.GRAPHS)
    assert prog == (51, 51)
    # Second pass starts: 0/51 — reset to lower count with same total
    t.feed([LogEvent("r0", "counter", Step.GRAPHS, 0, 51, "g2")], now=2.0, line="g2")
    prog = t.counter_progress(Step.GRAPHS)
    assert prog == (0, 51)
    # Progress in second pass
    t.feed([LogEvent("r0", "counter", Step.GRAPHS, 5, 51, "g3")], now=3.0, line="g3")
    prog = t.counter_progress(Step.GRAPHS)
    assert prog == (5, 51)


def test_graph_capture_second_pass_new_total():
    """A reset to a lower count with a new total is also a new pass."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Mark KV as done so GRAPHS counters are not routed to KV (profiling pass).
    t.feed([LogEvent("r0", "step_end", Step.KV, None, None, "kv")], now=0.0, line="kv")
    t.feed([LogEvent("r0", "counter", Step.GRAPHS, 10, 10, "g1")], now=1.0, line="g1")
    assert t.counter_progress(Step.GRAPHS) == (10, 10)
    # New pass with different total
    t.feed([LogEvent("r0", "counter", Step.GRAPHS, 0, 51, "g2")], now=2.0, line="g2")
    assert t.counter_progress(Step.GRAPHS) == (0, 51)


# ---------------------------------------------------------------------------
# TUNE counter — AutoTuner op tracking
# ---------------------------------------------------------------------------


def test_tune_counter_tracks_op_and_count():
    """TUNE counter events track the current op name and the 1-based count of
    distinct ops seen.  Each op restarts at 0/y."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Mark KV done so TUNE is not routed to KV.
    t.feed([LogEvent("r0", "step_end", Step.KV, None, None, "kv")], now=0.0, line="kv")
    # First op: trtllm_batch_decode_mla, 1/22
    t.feed(
        [LogEvent("", "counter", Step.TUNE, 1, 22, "trtllm_batch_decode_mla")],
        now=1.0,
        line="t1",
    )
    assert t.counter_progress(Step.TUNE) == (1, 22)
    detail = t.tune_detail()
    assert detail == ("trtllm_batch_decode_mla", 1)
    # Second op: flashinfer::trtllm_fp4_block_scale_moe, 0/22 (new op)
    t.feed(
        [LogEvent("", "counter", Step.TUNE, 0, 22, "trtllm_fp4_block_scale_moe")],
        now=2.0,
        line="t2",
    )
    assert t.counter_progress(Step.TUNE) == (0, 22)
    detail = t.tune_detail()
    assert detail == ("trtllm_fp4_block_scale_moe", 2)


def test_tune_not_started_before_kv_done():
    """TUNE counter events before KV is done are NOT routed to TUNE — they
    don't occur in real logs (AutoTuner runs after KV), but the tracker
    handles it by treating AutoTuner lines like any other counter: TUNE is
    after KV in the step order, so a TUNE counter closes KV."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Feed a TUNE counter without first ending KV — TUNE starts, closing KV.
    t.feed(
        [LogEvent("", "counter", Step.TUNE, 1, 22, "trtllm_batch_decode_mla")],
        now=1.0,
        line="t1",
    )
    assert t.is_step_started(Step.TUNE)
    assert t.is_step_done(Step.KV), "KV should be closed when TUNE starts"


# ---------------------------------------------------------------------------
# START ends at first WEIGHTS event
# ---------------------------------------------------------------------------


def test_start_ends_at_first_weights_event():
    """START step ends at the first WEIGHTS event of any kind."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Before any WEIGHTS event, START is not done
    assert not t.is_step_done(Step.START)
    # First WEIGHTS counter → START is done
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 0, 66, "w")], now=1.0, line="w")
    assert t.is_step_done(Step.START)


# ---------------------------------------------------------------------------
# Vitals — last_progress_at
# ---------------------------------------------------------------------------


def test_last_log_line_updates_last_progress():
    """Any non-empty log line updates last_log_at / last_log_line."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.feed([], now=5.0, line="some log line")
    assert t.last_log_at == 5.0
    assert t.last_log_line == "some log line"
    # last_progress_at should be at least last_log_at
    assert t.last_progress_at(include_cpu=False) >= 5.0


def test_none_vitals_never_count_as_progress():
    """VitalsSample with all-None fields never counts as progress."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.feed([], now=10.0, line="log at 10")
    baseline = t.last_progress_at(include_cpu=False)
    # Add all-None vitals — should not advance progress
    t.add_vitals(VitalsSample(at=11.0, cpu_seconds=None, gpu_mem_mib=None, read_bytes=None))
    assert t.last_progress_at(include_cpu=False) == baseline


def test_cpu_only_counts_when_include_cpu():
    """Log flat, GPU flat, CPU rising: include_cpu=False unchanged, include_cpu=True advances."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # Establish a baseline with a log line and a vitals sample
    t.feed([], now=100.0, line="log")
    t.add_vitals(VitalsSample(at=100.0, cpu_seconds=50.0, gpu_mem_mib=1000, read_bytes=100_000_000))
    baseline_no_cpu = t.last_progress_at(include_cpu=False)
    baseline_cpu = t.last_progress_at(include_cpu=True)

    # GPU mem unchanged, read_bytes unchanged, CPU rises by 5 seconds
    t.add_vitals(VitalsSample(at=102.0, cpu_seconds=55.0, gpu_mem_mib=1000, read_bytes=100_000_000))
    # Without CPU: no progress (interval is 2s, threshold is 0.5 * 2 = 1.0s, CPU rose 5s
    # but we exclude CPU entirely)
    assert t.last_progress_at(include_cpu=False) == baseline_no_cpu
    # With CPU: progress advances (CPU rose 5s >= 0.5 * 2s = 1.0s threshold)
    assert t.last_progress_at(include_cpu=True) > baseline_cpu


def test_gpu_mem_change_counts_as_progress():
    """GPU memory change >= 64 MiB counts as progress."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.add_vitals(VitalsSample(at=1.0, cpu_seconds=0.0, gpu_mem_mib=1000, read_bytes=0))
    baseline = t.last_progress_at(include_cpu=False)
    # 63 MiB change — not enough
    t.add_vitals(VitalsSample(at=2.0, cpu_seconds=0.0, gpu_mem_mib=1063, read_bytes=0))
    assert t.last_progress_at(include_cpu=False) == baseline
    # 64 MiB change — enough
    t.add_vitals(VitalsSample(at=3.0, cpu_seconds=0.0, gpu_mem_mib=1127, read_bytes=0))
    assert t.last_progress_at(include_cpu=False) > baseline


def test_read_bytes_increase_counts_as_progress():
    """Read-bytes increase >= 16 MiB counts as progress."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.add_vitals(VitalsSample(at=1.0, cpu_seconds=0.0, gpu_mem_mib=1000, read_bytes=0))
    baseline = t.last_progress_at(include_cpu=False)
    # 15 MiB increase — not enough (< 16 MiB = 16_777_216 bytes)
    t.add_vitals(
        VitalsSample(at=2.0, cpu_seconds=0.0, gpu_mem_mib=1000, read_bytes=15 * 1024 * 1024)
    )
    assert t.last_progress_at(include_cpu=False) == baseline
    # >= 16 MiB increase from the previous sample — enough
    t.add_vitals(
        VitalsSample(at=3.0, cpu_seconds=0.0, gpu_mem_mib=1000, read_bytes=32 * 1024 * 1024)
    )
    assert t.last_progress_at(include_cpu=False) > baseline


def test_vitals_keeps_every_sample():
    """add_vitals keeps every sample for the whole load (no cap)."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    for i in range(70):
        t.add_vitals(VitalsSample(at=float(i), cpu_seconds=0.0, gpu_mem_mib=0, read_bytes=0))
    assert len(t.vitals) == 70
    assert t.vitals[-1].at == 69.0
    assert t.vitals[0].at == 0.0


# ---------------------------------------------------------------------------
# Warmup / done
# ---------------------------------------------------------------------------


def test_mark_warmup_started_closes_previous_steps():
    """mark_warmup_started starts WARMUP, closing everything before."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 1, 10, "w")], now=1.0, line="w")
    t.mark_warmup_started(now=2.0)
    # All vLLM steps before WARMUP should be done
    for step in (Step.START, Step.WEIGHTS, Step.COMPILE, Step.KV, Step.TUNE, Step.GRAPHS):
        assert t.is_step_done(step), f"{step} should be done when warmup starts"


def test_mark_done_ends_warmup():
    """mark_done ends the WARMUP step."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    t.mark_warmup_started(now=1.0)
    assert not t.is_step_done(Step.WARMUP)
    t.mark_done(now=5.0)
    assert t.is_step_done(Step.WARMUP)


# ---------------------------------------------------------------------------
# step_durations
# ---------------------------------------------------------------------------


def test_step_durations_positive():
    """step_durations returns positive durations for steps that have started and ended."""
    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=10.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # START starts at started_at, WEIGHTS counter ends START and starts WEIGHTS
    t.feed([LogEvent("r0", "counter", Step.WEIGHTS, 0, 10, "w")], now=15.0, line="w")
    # WEIGHTS end
    t.feed([LogEvent("r0", "step_end", Step.WEIGHTS, None, None, "we")], now=20.0, line="we")
    durations = t.step_durations()
    assert Step.START in durations
    assert durations[Step.START] > 0
    assert durations[Step.START] == 5.0  # 10→15


# ---------------------------------------------------------------------------
# Fixture replays
# ---------------------------------------------------------------------------


def test_replay_dp3_fixture_step_durations():
    """Replay the DP3 fixture: step order, all steps through GRAPHS done,
    TUNE skipped (no AutoTuner), WEIGHTS ends 13:50:55–13:50:59, durations
    positive and sum to roughly the span."""
    tracker, _ = _replay("vllm-0.26.0-dp3-tp2-fp8.log")
    # All vLLM steps through GRAPHS should be done
    for step in (Step.START, Step.WEIGHTS, Step.COMPILE, Step.KV, Step.GRAPHS):
        assert tracker.is_step_done(step), f"{step} not done in DP3 fixture"
    # TUNE has no AutoTuner in this fixture — never explicitly started.
    assert not tracker.is_step_started(Step.TUNE), "TUNE should not be started in DP3 fixture"
    # WEIGHTS end time should be between 13:50:55 and 13:50:59
    weights_end = tracker.step_end_at(Step.WEIGHTS)
    assert weights_end is not None
    weights_end_dt = datetime.fromtimestamp(weights_end, tz=UTC)
    # 13:50:55 ≤ end ≤ 13:50:59
    assert weights_end_dt.hour == 13 and weights_end_dt.minute == 50
    assert 55 <= weights_end_dt.second <= 59
    # Step durations should be non-negative
    durations = tracker.step_durations()
    for step, dur in durations.items():
        assert dur >= 0, f"{step} duration negative: {dur}"
    # Steps with explicit start events should have positive durations
    for step in (Step.START, Step.WEIGHTS, Step.COMPILE, Step.GRAPHS):
        if step in durations:
            assert durations[step] > 0, f"{step} duration not positive: {durations[step]}"
    # Sum of durations should be roughly the start→ready span.  There are
    # gaps between steps (e.g. engine init between WEIGHTS end and COMPILE
    # start) so the sum is less than the span — allow up to 60s slack.
    total_dur = sum(durations.values())
    ready_at = tracker.step_end_at(Step.GRAPHS)
    assert ready_at is not None
    span = ready_at - tracker.started_at
    assert abs(total_dur - span) <= 60.0, f"total {total_dur:.1f} vs span {span:.1f}"


def test_replay_v026_glm_fixture_step_durations():
    """Replay the 0.26 GLM fixture: step order with TUNE between KV and GRAPHS,
    all steps through GRAPHS done, durations non-negative, and a >300s gap
    between KV done and GRAPHS counting."""
    tracker, timestamps = _replay("vllm-0.26.0-tp8-glm52-nvfp4-b300.log")
    for step in (Step.START, Step.WEIGHTS, Step.COMPILE, Step.KV, Step.TUNE, Step.GRAPHS):
        assert tracker.is_step_done(step), f"{step} not done in 0.26 GLM fixture"
    durations = tracker.step_durations()
    for step, dur in durations.items():
        assert dur >= 0, f"{step} duration negative: {dur}"
    # TUNE is between KV and GRAPHS: TUNE start >= KV end, GRAPHS start >= TUNE end.
    kv_end = tracker.step_end_at(Step.KV)
    tune_start = tracker.step_start_at(Step.TUNE)
    tune_end = tracker.step_end_at(Step.TUNE)
    graphs_start = tracker.step_start_at(Step.GRAPHS)
    assert kv_end is not None
    assert tune_start is not None
    assert tune_end is not None
    assert graphs_start is not None
    assert tune_start >= kv_end, "TUNE should start after KV ends"
    assert graphs_start >= tune_end, "GRAPHS should start after TUNE ends"
    # Gap between KV done and GRAPHS counting: > 300s
    gap = graphs_start - kv_end
    assert gap > 300.0, f"gap between KV done and GRAPHS start is only {gap:.1f}s, expected > 300"


def test_replay_v032_glm_fixture_step_durations():
    """Replay the 0.32 GLM fixture: step order with TUNE done, KV covers the
    profiling pass (> 0), GRAPHS covers only the final capture (starts after
    the last AutoTuner line), TUNE counter shows 22 as total."""
    tracker, _ = _replay("vllm-0.32.0-tp8-glm52-nvfp4-b300.log")
    for step in (Step.START, Step.WEIGHTS, Step.COMPILE, Step.KV, Step.TUNE, Step.GRAPHS):
        assert tracker.is_step_done(step), f"{step} not done in 0.32 GLM fixture"
    durations = tracker.step_durations()
    for step, dur in durations.items():
        assert dur >= 0, f"{step} duration negative: {dur}"
    # Steps with explicit start events should have positive durations
    for step in (Step.START, Step.WEIGHTS, Step.GRAPHS):
        if step in durations:
            assert durations[step] > 0, f"{step} duration not positive: {durations[step]}"
    # KV duration is > 0 and covers the profiling pass (pre-KV graph capture).
    assert Step.KV in durations
    assert durations[Step.KV] > 0, f"KV duration should be > 0, got {durations[Step.KV]}"
    # TUNE counter shows 22 as total (the last AutoTuner op in the 0.32 fixture).
    tune_prog = tracker.counter_progress(Step.TUNE)
    assert tune_prog is not None
    assert tune_prog[1] == 22, f"TUNE total should be 22, got {tune_prog[1]}"
    # GRAPHS starts after the last AutoTuner line (TUNE end ≤ GRAPHS start).
    tune_end = tracker.step_end_at(Step.TUNE)
    graphs_start = tracker.step_start_at(Step.GRAPHS)
    assert tune_end is not None
    assert graphs_start is not None
    assert graphs_start >= tune_end, f"GRAPHS start {graphs_start} should be >= TUNE end {tune_end}"
    # Sum of durations should be roughly the start→ready span.
    total_dur = sum(durations.values())
    ready_at = tracker.step_end_at(Step.GRAPHS)
    assert ready_at is not None
    span = ready_at - tracker.started_at
    assert abs(total_dur - span) <= 60.0, f"total {total_dur:.1f} vs span {span:.1f}"


# ---------------------------------------------------------------------------
# TUNE skipped on models without AutoTuner
# ---------------------------------------------------------------------------


def test_replay_dp3_fixture_tune_skipped():
    """The vLLM 0.32 DP3 fixture (no AutoTuner) has TUNE skipped: the step
    is done (closed by a later step) but never explicitly started."""
    tracker, _ = _replay("vllm-0.32.0-dp3-tp2-fp8.log")
    # TUNE is done (closed by engine_ready or GRAPHS) but never started.
    assert tracker.is_step_done(Step.TUNE), "TUNE should be done (closed by later step)"
    assert not tracker.is_step_started(Step.TUNE), "TUNE should not be explicitly started"
    assert tracker.tune_detail() is None, "TUNE detail should be None (no AutoTuner)"


def test_replay_llamacpp_fixture_tune_skipped():
    """The llama.cpp fixture has no AutoTuner and no TUNE step at all (llama.cpp
    uses LLAMACPP_STEPS which doesn't include TUNE)."""
    from app.runtime.load_progress.parse import LLAMACPP_STEPS, parse_llamacpp_line

    p = _FIXTURE_DIR / "llamacpp-b10731.synthetic.log"
    text = p.read_bytes().decode("utf-8", "replace")
    tracker = LoadTracker(
        model_id="test",
        generation=0,
        backend="llamacpp",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )
    for seg in split_log_chunk(text):
        events = parse_llamacpp_line(seg)
        tracker.feed(events, now=0.0, line=seg)
    # TUNE is not in LLAMACPP_STEPS, so it's not in the tracker's step dict.
    assert Step.TUNE not in LLAMACPP_STEPS
    assert tracker._steps.get(Step.TUNE) is not None  # exists in dict but never started
    assert not tracker.is_step_started(Step.TUNE)


# ---------------------------------------------------------------------------
# Profiling-pass graph finish must never end KV
# ---------------------------------------------------------------------------


def test_profiling_graph_finish_does_not_end_kv():
    """On the 0.32 GLM fixture, the profiling-pass "Graph capturing finished"
    lines arrive before the KV-end lines.  Replaying up to the first KV-end
    line (dp_size 1), KV must still be open: the profiling finishes add no end
    source, so they cannot close KV early through the quorum."""
    from app.runtime.load_progress.parse import parse_vllm_line

    segs = split_log_chunk(FIXTURE("vllm-0.32.0-tp8-glm52-nvfp4-b300.log"))
    cut = next(
        i
        for i, seg in enumerate(segs)
        if "Available KV cache memory:" in seg or "GPU KV cache size:" in seg
    )
    prefix = segs[:cut]
    # Sanity: the profiling pass really is in the prefix.
    assert any("Graph capturing finished" in seg for seg in prefix)

    t = LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=0.0,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
        dp_size=1,
    )
    for i, seg in enumerate(prefix):
        t.feed(parse_vllm_line(seg), now=float(i), line=seg)

    assert t.is_step_started(Step.KV)
    assert not t.is_step_done(Step.KV), "profiling graph finish must not end KV"
    assert not t.is_step_done(Step.GRAPHS)
