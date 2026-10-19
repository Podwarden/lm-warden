"""Tests for the stall policy and ``evaluate`` (app/runtime/load_progress/policy.py).

The policy replaces a plain wall-clock deadline: from a per-load tracker it
decides whether a model load is progressing, stalled (amber warning), or should
be given up.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.runtime.load_progress.policy import StallPolicy, Verdict, evaluate
from app.runtime.load_progress.tracker import LoadTracker, VitalsSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(started_at: float = 0.0) -> LoadTracker:
    return LoadTracker(
        model_id="m",
        generation=0,
        backend="vllm",
        started_at=started_at,
        started_wall=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _policy(**kw) -> StallPolicy:
    defaults = dict(warn_s=300.0, give_up_s=1200.0, max_s=7200.0)
    defaults.update(kw)
    return StallPolicy(**defaults)


# ---------------------------------------------------------------------------
# from_settings
# ---------------------------------------------------------------------------


def test_from_settings_reads_three_fields():
    s = Settings(
        data_dir=Path("/tmp"),
        hf_cache_dir=Path("/tmp"),
        cookie_secret="x" * 32,
        container_gpu_count=0,
        load_stall_warn_s=111.0,
        load_stall_timeout_s=222.0,
        load_max_s=333.0,
    )
    p = StallPolicy.from_settings(s)
    assert p.warn_s == 111.0
    assert p.give_up_s == 222.0
    assert p.max_s == 333.0


# ---------------------------------------------------------------------------
# evaluate — progressing
# ---------------------------------------------------------------------------


def test_fresh_tracker_is_progressing():
    """No events, no vitals — evaluate must not raise; default to progressing."""
    t = _make_tracker(started_at=0.0)
    v = evaluate(t, now=10.0, policy=_policy())
    assert v.state == "progressing"
    assert v.reason is None
    assert v.hint is None


def test_unknown_log_strings_still_progress_via_vitals():
    """A log line the parser does not recognise still counts as progress
    (it feeds last_log_at), so the load stays progressing."""
    t = _make_tracker(started_at=0.0)
    t.feed([], now=5.0, line="some unrecognised noise")
    v = evaluate(t, now=10.0, policy=_policy())
    assert v.state == "progressing"


def test_recent_vitals_keep_progressing():
    """GPU-mem change is progress (include_cpu=False), so no stall."""
    t = _make_tracker(started_at=0.0)
    t.add_vitals(VitalsSample(at=0.0, cpu_seconds=0.0, gpu_mem_mib=0, read_bytes=0))
    t.add_vitals(VitalsSample(at=100.0, cpu_seconds=50.0, gpu_mem_mib=128, read_bytes=0))
    v = evaluate(t, now=101.0, policy=_policy())
    assert v.state == "progressing"


# ---------------------------------------------------------------------------
# evaluate — stalled (amber warning)
# ---------------------------------------------------------------------------


def test_stalled_when_no_progress_for_warn_s():
    """No log, no vitals, no step changes for warn_s -> stalled, not give_up."""
    t = _make_tracker(started_at=0.0)
    # A single log line at t=5 keeps last_progress_at(include_cpu=False) at 5.0.
    t.feed([], now=5.0, line="starting up")
    # now=5 + warn_s(300) = 305 -> exactly warn_s, so stalled_for_s >= warn_s.
    v = evaluate(t, now=305.0, policy=_policy())
    assert v.state == "stalled"
    assert v.stalled_for_s >= 300.0
    assert v.reason is not None
    assert "min" in v.reason


def test_spinning_cpu_is_stalled_not_given_up():
    """CPU is the only thing moving: the load is stalled (amber), but NOT
    given up, because CPU progress keeps last_progress_at(include_cpu=True)
    alive."""
    t = _make_tracker(started_at=0.0)
    # Two vitals samples with CPU advancing but GPU-mem and read-bytes flat.
    t.add_vitals(VitalsSample(at=0.0, cpu_seconds=0.0, gpu_mem_mib=100, read_bytes=0))
    t.add_vitals(VitalsSample(at=100.0, cpu_seconds=200.0, gpu_mem_mib=100, read_bytes=0))
    # At now=1000: no log for 1000s (stalled), but CPU moved at t=100.
    # warn-stall = now - last_progress_at(include_cpu=False).
    # last_progress_at(include_cpu=False) = 0.0 (only CPU moved, which is excluded).
    # So stalled_for_s = 1000 - 0 = 1000 >= warn_s(300) -> stalled.
    # give_up: now - last_progress_at(include_cpu=True) = 1000 - 100 = 900 < give_up_s(1200).
    #         now - started_at = 1000 < max_s(7200).
    # So NOT give_up.
    v = evaluate(t, now=1000.0, policy=_policy())
    assert v.state == "stalled"
    assert v.hint == "CPU busy but nothing else moving"


# ---------------------------------------------------------------------------
# evaluate — give_up
# ---------------------------------------------------------------------------


def test_flat_everything_gives_up_after_give_up_s():
    """Nothing moves at all: after give_up_s since the last CPU progress, give
    up."""
    t = _make_tracker(started_at=0.0)
    # No events, no vitals — last_progress_at returns 0.0 for both include_cpu
    # values (nothing happened, so candidates is empty -> 0.0).
    # Actually started_at is 0.0, and START step started_at = 0.0, so
    # last_progress_at returns 0.0 (the step start).
    # give_up: now - 0 >= give_up_s(1200) -> give up.
    v = evaluate(t, now=1200.0, policy=_policy())
    assert v.state == "give_up"
    assert v.reason is not None


def test_cap_gives_up_even_while_progressing():
    """Even if something is moving, exceeding max_s since start is a give-up."""
    t = _make_tracker(started_at=0.0)
    # A log line very recently — so warn-stall is small, but max_s is exceeded.
    t.feed([], now=7199.0, line="still going")
    # now - started_at = 7200 >= max_s(7200) -> give up.
    v = evaluate(t, now=7200.0, policy=_policy())
    assert v.state == "give_up"
    assert v.reason is not None
    assert "h" in v.reason  # "still not ready after {h} h (limit)"


# ---------------------------------------------------------------------------
# evaluate — hints
# ---------------------------------------------------------------------------


def test_nccl_hint_when_log_matches():
    """The last log line matching NCCL/init_process_group/distributed
    triggers the GPU-connect hint."""
    t = _make_tracker(started_at=0.0)
    t.feed([], now=5.0, line="NCCL init_process_group distributed")
    # Stall it: now far enough from last_progress_at(include_cpu=False).
    v = evaluate(t, now=5000.0, policy=_policy())
    assert v.state in ("stalled", "give_up")
    assert v.hint == "Looks stuck while the GPUs connect to each other"


def test_nccl_hint_case_insensitive():
    """The NCCL regex matches case-insensitively."""
    t = _make_tracker(started_at=0.0)
    t.feed([], now=5.0, line="nccl Init_Process_Group DISTRIBUTED")
    v = evaluate(t, now=5000.0, policy=_policy())
    assert v.hint == "Looks stuck while the GPUs connect to each other"


def test_no_hint_when_log_does_not_match_and_no_cpu_only():
    """Stalled but the log line is generic — no NCCL, and CPU is not the sole
    thing keeping it alive -> hint is None."""
    t = _make_tracker(started_at=0.0)
    t.feed([], now=5.0, line="loading weights")
    v = evaluate(t, now=5000.0, policy=_policy())
    # give_up because 5000 - 0 >= give_up_s(1200)
    assert v.state == "give_up"
    assert v.hint is None


# ---------------------------------------------------------------------------
# evaluate — totality (never raises)
# ---------------------------------------------------------------------------


def test_evaluate_is_total_on_fresh_tracker():
    """evaluate must never raise — it runs every few seconds inside the load
    path. A tracker with zero events and zero vitals must return a Verdict."""
    t = _make_tracker(started_at=0.0)
    # Should not raise, even though nothing has been fed.
    v = evaluate(t, now=0.0, policy=_policy())
    assert isinstance(v, Verdict)
    assert v.state in ("progressing", "stalled", "give_up")


# ---------------------------------------------------------------------------
# evaluate — give_up_at_s moves forward with progress
# ---------------------------------------------------------------------------


def test_give_up_at_moves_forward_with_progress():
    """When the load progresses, give_up_at_s (seconds since start at which the
    load WILL be given up if nothing moves) moves forward, because it is based
    on last_progress_at(include_cpu=True)."""
    t = _make_tracker(started_at=0.0)
    p = _policy()  # give_up_s=1200, max_s=7200

    # Early: nothing has happened. last_progress_at(include_cpu=True) = 0.0
    # (START step started at 0.0).
    v1 = evaluate(t, now=10.0, policy=p)
    # give_up_at_s = min(0.0 - 0.0 + 1200, 7200) = 1200.0
    assert v1.give_up_at_s == 1200.0

    # Progress: a log line arrives at t=500.
    t.feed([], now=500.0, line="weights shard 1 done")
    v2 = evaluate(t, now=510.0, policy=p)
    # last_progress_at(include_cpu=True) = 500.0
    # give_up_at_s = min(500.0 - 0.0 + 1200, 7200) = 1700.0
    assert v2.give_up_at_s == 1700.0
    assert v2.give_up_at_s > v1.give_up_at_s

    # More progress at t=2000.
    t.feed([], now=2000.0, line="weights shard 2 done")
    v3 = evaluate(t, now=2010.0, policy=p)
    # give_up_at_s = min(2000.0 - 0.0 + 1200, 7200) = 3200.0
    assert v3.give_up_at_s == 3200.0
    assert v3.give_up_at_s > v2.give_up_at_s


def test_give_up_at_capped_by_max_s():
    """give_up_at_s never exceeds max_s."""
    t = _make_tracker(started_at=0.0)
    p = _policy(give_up_s=99999.0, max_s=7200.0)
    v = evaluate(t, now=10.0, policy=p)
    assert v.give_up_at_s == 7200.0
