"""Applying a measured setting to the model (design §6.4).

"after we did measurements we need to offer to apply the probed settings and
reload."

Applying is the one action in this feature that CHANGES the model rather than
observing it, so every refusal below exists because applying anyway would put
the model into a state the measurement does not describe.

Note what applying costs, which is why it is offered and never automatic: the
settings patch refuses to touch a loaded model, so this is unload -> persist ->
load. The model stops serving for the length of a load, and the measurement
that justified the change describes the configuration being replaced.
"""

from __future__ import annotations

from app.stress.apply import ApplyRefused, plan_apply


def _rec(**over):
    base = {"max_model_len": 32768, "provenance": "measured"}
    base.update(over)
    return base


def test_a_measured_recommendation_is_applied():
    plan = plan_apply(
        run_status="completed",
        run_fingerprint="fp-1",
        current_fingerprint="fp-1",
        recommended_config=_rec(),
        current_max_model_len=8192,
    )
    assert plan.max_model_len == 32768
    assert plan.previous == 8192


def test_a_run_that_is_still_going_cannot_be_applied():
    """Its bracket is unconfirmed; the number would be a lower bound wearing
    full confidence."""
    try:
        plan_apply(
            run_status="running",
            run_fingerprint="fp-1",
            current_fingerprint="fp-1",
            recommended_config=_rec(),
            current_max_model_len=8192,
        )
    except ApplyRefused as e:
        assert e.reason == "run_not_complete"
    else:
        raise AssertionError("a running run was applied")


def test_a_measurement_from_different_hardware_is_refused():
    """The fingerprint covers the GPUs, the engine build and the co-resident
    set. A number measured elsewhere describes a machine this is not."""
    try:
        plan_apply(
            run_status="completed",
            run_fingerprint="fp-OLD",
            current_fingerprint="fp-NEW",
            recommended_config=_rec(),
            current_max_model_len=8192,
        )
    except ApplyRefused as e:
        assert e.reason == "fingerprint_changed"
    else:
        raise AssertionError("a stale measurement was applied")


def test_a_run_with_no_recommendation_is_refused():
    """`choose_recommendation` returns None rather than guessing, and that
    silence must not be turned into a setting here."""
    for rec in (None, {}, {"limited_by": "quality"}):
        try:
            plan_apply(
                run_status="completed",
                run_fingerprint="fp-1",
                current_fingerprint="fp-1",
                recommended_config=rec,
                current_max_model_len=8192,
            )
        except ApplyRefused as e:
            assert e.reason == "no_recommendation"
        else:
            raise AssertionError(f"applied nothing: {rec!r}")


def test_a_nonsense_recommendation_is_refused():
    for bad in (0, -1, "big", True, 3.5):
        try:
            plan_apply(
                run_status="completed",
                run_fingerprint="fp-1",
                current_fingerprint="fp-1",
                recommended_config=_rec(max_model_len=bad),
                current_max_model_len=8192,
            )
        except ApplyRefused as e:
            assert e.reason == "no_recommendation"
        else:
            raise AssertionError(f"applied {bad!r}")


def test_applying_the_setting_already_in_force_is_refused():
    """Nothing to do, and doing it anyway would stop the model serving for the
    length of a load to arrive back where it started."""
    try:
        plan_apply(
            run_status="completed",
            run_fingerprint="fp-1",
            current_fingerprint="fp-1",
            recommended_config=_rec(),
            current_max_model_len=32768,
        )
    except ApplyRefused as e:
        assert e.reason == "already_applied"
    else:
        raise AssertionError("reloaded for no change")


def test_applying_from_an_unset_setting_is_allowed():
    """max_model_len is optional and commonly NULL -- that is the case where a
    measured value helps most."""
    plan = plan_apply(
        run_status="completed",
        run_fingerprint="fp-1",
        current_fingerprint="fp-1",
        recommended_config=_rec(),
        current_max_model_len=None,
    )
    assert plan.max_model_len == 32768
    assert plan.previous is None
