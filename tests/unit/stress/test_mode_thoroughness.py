"""A cached run only answers a question as demanding as the one it asked.

Reported by the operator, 2026-09-04: "it wont let restart with a longer test".
A `conservative` run had just completed, and asking for a longer mode returned
`200 reused` naming it. The cooldown asked "is there a recent measurement for
this fingerprint?" and never "does that measurement answer THIS request?".

The modes are not interchangeable. `thorough` confirms 7/7 rather than 5/5,
probes three needle offsets rather than one, and is the only mode that runs the
configuration sweep -- so it is the only mode that can produce a
`recommended_config` at all. Handing back a conservative run withholds exactly
what the operator asked for, while reporting success.
"""

from __future__ import annotations

from app.stress.modes import at_least_as_thorough


def test_the_same_mode_is_reusable():
    assert at_least_as_thorough("conservative", "conservative") is True
    assert at_least_as_thorough("thorough", "thorough") is True


def test_a_conservative_run_does_not_answer_a_thorough_request():
    """The reported bug. Thorough alone runs the sweep."""
    assert at_least_as_thorough(cached="conservative", requested="thorough") is False


def test_a_thorough_run_answers_a_lesser_request():
    """It did strictly more work: more confirmation, more offsets, the sweep."""
    assert at_least_as_thorough(cached="thorough", requested="conservative") is True
    assert at_least_as_thorough(cached="thorough", requested="quick") is True


def test_conservative_and_quick_satisfy_each_other():
    """They differ ONLY in crash budget, 0 against 2.

    Both confirm 5/5, both probe the single 0.5 needle offset, neither runs the
    sweep. A crash budget is a willingness to break the engine, not a stronger
    claim about the model. Ranking them apart would send an operator to the GPU
    for a number they already had -- which is what happened when this module
    first ranked `quick` above `conservative`, and CI caught it.
    """
    assert at_least_as_thorough(cached="quick", requested="conservative") is True
    assert at_least_as_thorough(cached="conservative", requested="quick") is True


def test_an_unknown_mode_is_never_treated_as_sufficient():
    """Gating a button: an unrecognised cached mode must not silently satisfy a
    request. Starting a run costs GPU time; withholding one costs the answer.
    """
    assert at_least_as_thorough(cached="banana", requested="thorough") is False
    assert at_least_as_thorough(cached=None, requested="quick") is False


def test_an_unknown_request_still_reuses_an_equal_or_better_run():
    """An unknown REQUEST cannot be ranked, so fall back to exact match only."""
    assert at_least_as_thorough(cached="thorough", requested="banana") is False
    assert at_least_as_thorough(cached="banana", requested="banana") is True
