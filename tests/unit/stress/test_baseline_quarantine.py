"""A probe that fails at the BASELINE must not judge the ladder (design §4.3).

Found on the second production run, 2026-09-04. With the budget fix in, the
baseline calibrated cleanly on `gpt-oss-20b-gguf` -- real answers from all six
probes, nothing starved -- and the run still confirmed nothing:

    needle   -> pass at 1024, 960, 896, 8192
    summary  -> repeating at EVERY length, the baseline included
    raw_confirmed: 0, first_observed_failure: 1024

The summary probe's `repeating` verdict was a false positive: asked the same
question directly, the model produced two clean sentences and stopped. But the
decisive evidence is structural, not aesthetic -- a gate that reads the same at
896 tokens as at 8192 carries NO information about where the model breaks,
while still vetoing every rung.

gates.py already states the principle for SHORT ("a degenerate baseline means
the reference itself was broken... refuse to judge rather than judge wrongly").
Nothing enforced it for the other gates.
"""

from __future__ import annotations

from app.stress.quarantine import ladder_probes, quarantined


def test_a_probe_that_tripped_at_baseline_is_quarantined():
    assert quarantined({"summary": ["repeating"]}) == ("summary",)


def test_a_clean_baseline_quarantines_nothing():
    assert quarantined({}) == ()
    assert quarantined({"summary": []}) == ()


def test_every_tripped_probe_is_quarantined_not_just_the_first():
    got = quarantined({"summary": ["repeating"], "json": ["wrong"]})
    assert set(got) == {"summary", "json"}


def test_quarantine_is_ordered_for_a_stable_record():
    assert quarantined({"summary": ["repeating"], "json": ["wrong"]}) == ("json", "summary")


# ---- what the ladder then runs ------------------------------------------


def test_the_ladder_drops_a_quarantined_probe():
    """The production case: needle discriminates, summary cannot."""
    assert ladder_probes(("needle", "summary"), ("summary",)) == ("needle",)


def test_the_ladder_is_unchanged_when_nothing_is_quarantined():
    assert ladder_probes(("needle", "summary"), ()) == ("needle", "summary")


def test_quarantining_a_probe_the_ladder_never_used_changes_nothing():
    """`json` fails at baseline but was never a ladder probe; the ladder is
    unaffected and the exclusion is still recorded for the operator."""
    assert ladder_probes(("needle", "summary"), ("json",)) == ("needle", "summary")


def test_an_empty_ladder_is_reported_rather_than_silently_passing():
    """If every ladder probe is quarantined there is no oracle left.

    Returning () lets the caller abort. Treating it as "nothing failed" would
    publish the axis maximum off the back of zero evidence -- the worst
    possible outcome for a feature whose whole value is that its numbers are
    measured.
    """
    assert ladder_probes(("needle", "summary"), ("needle", "summary")) == ()


# ---- the runner's bookkeeping is not a measured limit --------------------


def test_bookkeeping_keys_are_not_limits():
    """Seen in the shipped UI, 2026-09-04: `neighbours` and `baseline`
    rendered as MEASURED LIMITS cards, each with a blue "measured" badge, a
    value of "—" and "First observed failure: none observed".

    They are the runner's own notes. Presenting them as measurements next to a
    real one devalues the badge that carries the feature's entire credibility.
    """
    from app.stress.measured import is_axis

    assert is_axis("quality_context") is True
    assert is_axis("concurrency") is True
    assert is_axis("baseline") is False
    assert is_axis("neighbours") is False
    assert is_axis("aborted_reason") is False
