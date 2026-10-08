"""A run that measured nothing must not hold the cooldown (design §7.2).

Found in production 2026-09-04, immediately after the reasoning-budget fix
shipped: the operator pressed "Stress test" twice and both POSTs returned
`200 reused` naming the OLD, valueless run. No new run started, no error
appeared, and the UI reveals its `force` control only on a 409 -- so the
button was inert for the full six-hour cooldown with no way out.

`current_for`'s docstring calls its result "the newest PUBLISHABLE
measurement" and lists four conditions. None of them asks whether the run
published anything. A run that completed having confirmed no value at all --
`recommended_config: None`, `invalid_reason: no_confirmed_value` -- satisfies
all four and blocks re-measurement, which is exactly backwards: a run that
learned nothing is the strongest possible reason to allow another.
"""

from __future__ import annotations

from app.stress.measured import has_measurement


def test_a_run_with_a_recommendation_is_a_measurement():
    assert (
        has_measurement({"quality_context": {"raw_confirmed": 32768}}, {"max_model_len": 32768})
        is True
    )


def test_a_confirmed_limit_alone_is_a_measurement():
    """A run can establish a bound without recommending a config change."""
    assert has_measurement({"quality_context": {"raw_confirmed": 65536}}, None) is True


def test_the_production_failure_is_not_a_measurement():
    """Verbatim from the run that blocked the retest."""
    limits = {
        "neighbours": [],
        "baseline": {"approx_tokens": 1024},
        "quality_context": {
            "axis_max": 8192,
            "raw_confirmed": 0,
            "first_observed_failure": 1024,
            "publishable": True,
            "invalid_reason": "no_confirmed_value",
        },
    }
    assert has_measurement(limits, None) is False


def test_bookkeeping_alone_is_not_a_measurement():
    """`baseline` and `neighbours` are the runner's own notes, not limits.

    Counting a key as evidence would make every run that got as far as the
    baseline look like it had measured something.
    """
    assert has_measurement({"baseline": {"approx_tokens": 1024}}, None) is False
    assert has_measurement({"neighbours": []}, None) is False


def test_nothing_at_all_is_not_a_measurement():
    assert has_measurement(None, None) is False
    assert has_measurement({}, None) is False


def test_a_zero_confirmed_axis_is_not_a_measurement():
    """raw_confirmed 0 means the search confirmed no value, not "zero tokens"."""
    assert has_measurement({"concurrency": {"raw_confirmed": 0}}, None) is False


def test_a_malformed_limits_blob_does_not_raise():
    """This gates a user-facing button; an exception here would 500 the POST."""
    for bad in ("text", [], {"quality_context": "nope"}, {"x": {"raw_confirmed": "n"}}):
        assert has_measurement(bad, None) is False


# ---- a limit may be a bare number ---------------------------------------


def test_a_scalar_axis_is_a_measurement():
    """The runner records scalars; `_normalise_limit` widens them on the way
    out. Requiring the object form read a real measurement as none, and broke
    four existing tests when this gate was first written."""
    assert has_measurement({"quality_context": 4096}, None) is True
    assert has_measurement({"a": 1}, None) is True


def test_a_zero_or_negative_scalar_is_not_a_measurement():
    assert has_measurement({"quality_context": 0}, None) is False
    assert has_measurement({"quality_context": -1}, None) is False


def test_a_boolean_is_not_a_measurement():
    """`True` is an int in Python; a flag is not a measured value."""
    assert has_measurement({"quality_context": True}, None) is False


def test_an_object_axis_carrying_only_a_value_is_a_measurement():
    assert has_measurement({"quality_context": {"value": 32768}}, None) is True


# ---- an axis the runner refused to stand behind --------------------------


def test_a_withheld_axis_is_not_a_measurement():
    """The runner records `invalid_reason` NEXT TO a positive raw_confirmed.

    The concurrency axis does exactly this when the prefix cache was not
    defeated: the number exists but measures deduplicated KV, and the runner
    says so. Counting it would let a run that published nothing usable hold the
    six-hour cooldown -- the same reader/writer disagreement this module was
    written to end.
    """
    limits = {"concurrency": {"raw_confirmed": 48, "invalid_reason": "prefix_cache_not_defeated"}}
    assert has_measurement(limits, None) is False


def test_a_withheld_axis_does_not_mask_a_good_one():
    """One bad axis must not discard the run's real finding."""
    limits = {
        "concurrency": {"raw_confirmed": 48, "invalid_reason": "prefix_cache_not_defeated"},
        "quality_context": {"raw_confirmed": 32768},
    }
    assert has_measurement(limits, None) is True
