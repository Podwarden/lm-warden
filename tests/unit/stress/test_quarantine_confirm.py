"""A probe is only excluded when it fails the baseline TWICE.

Two runs aborted with `aborted_no_baseline` because both ladder probes were
quarantined. Only `needle` and `summary` judge the ladder, so two failures --
each decided on a SINGLE sample -- end the run, while four other probes may be
answering perfectly.

The design's own argument against single samples applies here more than
anywhere: `search.py` replicates every ladder verdict because one sample near a
threshold is a coin flip, and then the baseline, which shapes the entire run,
was deciding on one. A dropped stream trips ABRUPT via `finish_reason: None` --
a state gates.py calls "real in this codebase" -- and that alone was enough to
retire a probe permanently.

Confirming costs one extra probe per FAILING probe, at the baseline only, which
is the cheapest place in the run.
"""

from __future__ import annotations

from app.stress.gates import Gate
from app.stress.quarantine import confirm_exclusion


def test_a_probe_that_fails_twice_is_excluded():
    """Consistent failure is a property of the model, not of one request."""
    assert confirm_exclusion((Gate.REPEATING,), (Gate.REPEATING,)) is True


def test_a_probe_that_fails_then_passes_is_kept():
    """The first failure was noise. Retiring the probe would cost the run a
    whole oracle on the strength of one sample."""
    assert confirm_exclusion((Gate.ABRUPT,), ()) is False


def test_failing_differently_still_counts_as_failing():
    """The gate need not be the same one. What matters is that the probe cannot
    produce a usable reference, not which way it fails."""
    assert confirm_exclusion((Gate.ABRUPT,), (Gate.WRONG,)) is True


def test_a_probe_that_passed_first_is_never_excluded():
    """Confirmation only ever runs after a failure, but the predicate must not
    depend on its caller getting that right."""
    assert confirm_exclusion((), (Gate.WRONG,)) is False
    assert confirm_exclusion((), ()) is False
