"""How far the probe budget may grow, derived from the model (not a constant).

"we should not hardcode anything for the model. our user could try any model
from huggingface and any gpu" (2026-09-04).

The escalation ladder was bounded by BUDGET_CAP = 1024, a constant chosen
because it suited the model in front of us. On `gpt-oss-20b-gguf` the needle
probe reached that ceiling and the run aborted with `aborted_no_baseline`;
raising the constant would have fixed that model and mis-sized every other one.
A model with a 4k context cannot spend 1024 tokens thinking, and a model with
262k can spend far more.

The bound that is actually true for every model: the answer has to fit in the
context alongside the prompt. That is measured per model, per configuration.
"""

from __future__ import annotations

from app.stress.probes import budget_cap_for, escalate_budget


def test_the_cap_is_what_the_context_has_left_after_the_prompt():
    """8192 context, 1000-token prompt -> 7192 for the answer."""
    assert budget_cap_for(context_window=8192, prompt_tokens=1000) == 7192


def test_a_small_context_model_gets_a_small_cap():
    """The case a constant 1024 got wrong in the other direction: a 2k model
    cannot spend 1024 tokens reasoning about a 1.5k prompt."""
    assert budget_cap_for(context_window=2048, prompt_tokens=1500) == 548


def test_a_large_context_model_is_not_held_to_someone_else_s_ceiling():
    cap = budget_cap_for(context_window=262144, prompt_tokens=1024)
    assert cap > 100_000


def test_an_unknown_context_falls_back_to_a_floor_not_to_infinity():
    """A model whose context we cannot read still gets swept.

    The floor is a bound on RUN COST -- how many tokens one probe may spend --
    which is a property of the harness, not of the model. It is not a claim
    about what any model needs.
    """
    assert budget_cap_for(context_window=None, prompt_tokens=1024) > 0
    assert budget_cap_for(context_window=0, prompt_tokens=1024) > 0


def test_a_prompt_that_fills_the_context_still_leaves_room_to_answer():
    """Otherwise the cap goes to zero or negative and escalation dies at the
    first step, reporting a model limit that is really arithmetic."""
    cap = budget_cap_for(context_window=1024, prompt_tokens=1024)
    assert cap > 0


def test_escalation_stops_at_the_derived_cap():
    cap = budget_cap_for(context_window=2048, prompt_tokens=1500)  # 548
    assert escalate_budget(128, cap=cap) == 512
    assert escalate_budget(512, cap=cap) == 548
    assert escalate_budget(548, cap=cap) is None


def test_escalation_still_climbs_by_four():
    assert escalate_budget(16, cap=100_000) == 64
    assert escalate_budget(64, cap=100_000) == 256


# ---- the baseline prompt is sized to the model too ------------------------


def test_the_baseline_fits_inside_a_small_context():
    """BASELINE_TOKENS is 1024. A model with a 512-token context cannot have a
    1024-token reference condition: every probe would be refused, and the run
    would report a model limit that is really our own default.
    """
    from app.stress.probes import baseline_tokens_for

    assert baseline_tokens_for(context_window=512) <= 512


def test_a_roomy_model_keeps_the_standard_baseline():
    """The reference must stay comparable across models wherever it can.

    SHORT is baseline-relative, so moving the baseline for no reason would make
    two runs of the same model incomparable.
    """
    from app.stress.probes import BASELINE_TOKENS, baseline_tokens_for

    assert baseline_tokens_for(context_window=131072) == BASELINE_TOKENS
    assert baseline_tokens_for(context_window=None) == BASELINE_TOKENS


def test_the_baseline_leaves_room_for_an_answer():
    """A baseline prompt that fills the whole context leaves nothing to
    generate into, so the reference answer would be empty by construction."""
    from app.stress.probes import baseline_tokens_for

    assert baseline_tokens_for(context_window=1024) < 1024
