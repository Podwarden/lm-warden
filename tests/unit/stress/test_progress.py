"""What a run reports about itself while it is still running.

Reported 2026-09-04 with a screenshot: a thorough run in flight showed
"0 probe(s)", "No probes recorded yet", and a progress bar rendered FULL.
Observations are written once, at finish, so for a run that takes hours the
modal had nothing — and a full bar reads as "finished", which is the worst of
the three possible wrong answers.

The ETA is derived from what the phase actually knows: each phase declares how
many steps it plans, and the estimate is that phase's measured pace applied to
its own remainder. It is never extrapolated across phases, because the phases
have wildly different unit costs — a length rung is one HTTP probe, a sweep
candidate is an engine reload that can take ten minutes.
"""

from __future__ import annotations

from app.stress.progress import Progress


def test_a_fresh_phase_reports_no_eta():
    """Nothing has been measured yet, so any number would be invented."""
    p = Progress()
    p.enter("baseline", total=6, note="calibrating probe budgets")
    snap = p.snapshot(now=100.0)
    assert snap["phase"] == "baseline"
    assert snap["completed"] == 0
    assert snap["total"] == 6
    assert snap["eta_s"] is None


def test_the_eta_uses_the_phase_s_own_measured_pace():
    """Two steps in 10s, four left -> 20s."""
    p = Progress()
    p.enter("baseline", total=6, at=0.0)
    p.step(at=5.0)
    p.step(at=10.0)
    assert p.snapshot(now=10.0)["eta_s"] == 20.0


def test_a_phase_with_no_declared_total_reports_no_eta():
    """The sweep cannot know its candidate count until fit.py has pruned."""
    p = Progress()
    p.enter("sweep", total=None, at=0.0)
    p.step(at=5.0)
    assert p.snapshot(now=5.0)["eta_s"] is None
    assert p.snapshot(now=5.0)["total"] is None


def test_entering_a_phase_resets_the_pace():
    """A sweep candidate costs an engine reload; a length rung costs one HTTP
    probe. Carrying the ladder's pace into the sweep would understate the
    remaining time by orders of magnitude."""
    p = Progress()
    p.enter("length", total=4, at=0.0)
    p.step(at=1.0)
    p.enter("sweep", total=2, at=1.0)
    p.step(at=601.0)
    assert p.snapshot(now=601.0)["eta_s"] == 600.0


def test_overrunning_the_planned_total_never_reports_negative_time():
    """Bisection can take fewer or more steps than planned; an ETA that counts
    backwards is worse than none."""
    p = Progress()
    p.enter("length", total=2, at=0.0)
    for i in range(5):
        p.step(at=float(i + 1))
    assert p.snapshot(now=5.0)["eta_s"] is None


def test_the_note_says_what_is_happening_now():
    p = Progress()
    p.enter("length", total=4, note="climbing the ladder")
    p.step(note="probing 8192 tokens")
    assert p.snapshot(now=1.0)["note"] == "probing 8192 tokens"


def test_the_snapshot_uses_the_vocabulary_the_ui_already_reads():
    """`completed`/`total`, not `step`/`steps_total`.

    StressRunProgress reads `progress.completed` and `progress.total`. A second
    vocabulary here is how the results panel came to render nothing at all --
    the writer said one thing, the reader looked for another, and both were
    individually correct.
    """
    p = Progress()
    p.enter("baseline", total=6)
    assert set(p.snapshot(now=0.0)) >= {"phase", "note", "completed", "total", "eta_s"}


def test_a_snapshot_before_any_phase_is_still_valid():
    """The row is written from the heartbeat, which may tick before preflight
    finishes. A reader must never see a half-built object."""
    snap = Progress().snapshot(now=1.0)
    assert snap["phase"] is None
    assert snap["completed"] == 0
    assert snap["eta_s"] is None
