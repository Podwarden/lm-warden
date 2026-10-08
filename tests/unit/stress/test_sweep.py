"""Planning the configuration sweep, and choosing the answer (design §6).

This is the part that answers the actual question -- "what should max_model_len
be for this model on this hardware?" -- rather than merely reporting where
things broke.

Two rules run through all of it:

* **Prediction prunes; measurement decides.** fit.py can say a candidate is
  certainly too big for the VRAM present, which is worth knowing because every
  reload costs up to load_timeout_s = 600. It cannot say a candidate will work:
  the measured 65,536 -> 131,072 context raise cost ~0 extra VRAM, so the
  prediction is unreliable in exactly the region that matters. No predicted
  number is ever published.
* **Sweeping is expensive.** Candidates must be few and informative.
"""

from __future__ import annotations

from app.stress.sweep import CandidateResult, choose_recommendation, plan_candidates


def _kv():
    """Per-token KV bytes for a small model: 2 * 24 * 8 * 64 * 2."""
    return 49152


# ---- the ladder ----------------------------------------------------------


def test_candidates_are_geometric():
    got = plan_candidates(
        current=8192, model_ceiling=131072, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert got == [8192, 16384, 32768, 65536, 131072]


def test_the_model_s_own_ceiling_is_never_exceeded():
    """Past max_position_embeddings the engine refuses, so a probe there
    measures the config file rather than the hardware."""
    got = plan_candidates(
        current=8192, model_ceiling=40000, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert max(got) <= 40000
    assert 40000 in got


def test_the_current_setting_is_always_a_candidate():
    """It is the baseline every other result is compared against."""
    got = plan_candidates(
        current=12345, model_ceiling=131072, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert 12345 in got


def test_candidates_are_sorted_and_unique():
    got = plan_candidates(
        current=16384, model_ceiling=131072, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert got == sorted(set(got))


# ---- prediction prunes ---------------------------------------------------


def test_candidates_beyond_what_vram_could_hold_are_dropped():
    """Each reload costs up to ten minutes; spending one on a certainty is waste.

    At 49,152 bytes/token a 12 GiB KV budget cannot reach 262,144 tokens, and
    no measurement is needed to know that.
    """
    budget = 12 * 1024**3
    got = plan_candidates(
        current=8192, model_ceiling=262144, vram_budget_bytes=budget, kv_bytes_per_token=_kv()
    )
    assert max(got) * _kv() <= budget


def test_pruning_never_removes_the_current_setting():
    """The model is running at it, so it demonstrably fits.

    If the estimate says otherwise the estimate is wrong, and dropping the one
    candidate we have direct evidence for would be perverse.
    """
    got = plan_candidates(
        current=131072, model_ceiling=131072, vram_budget_bytes=1024, kv_bytes_per_token=_kv()
    )
    assert 131072 in got


def test_no_kv_estimate_means_no_pruning():
    """A model whose config we cannot read still gets swept.

    Degrading to 'measure everything' is slower but correct; refusing to sweep
    would make the feature useless for exactly the unusual models that most
    need it.
    """
    got = plan_candidates(
        current=8192, model_ceiling=131072, vram_budget_bytes=8 * 1024**3, kv_bytes_per_token=None
    )
    assert 131072 in got


def test_a_lower_setting_can_be_proposed():
    """If the current context is too ambitious, the answer is to reduce it.

    The sweep is not only about finding headroom.
    """
    got = plan_candidates(
        current=131072, model_ceiling=131072, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert any(c < 131072 for c in got)


# ---- choosing the answer -------------------------------------------------


def test_the_recommendation_is_the_largest_context_that_held_up():
    results = [
        CandidateResult(max_model_len=8192, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=32768, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=131072, loaded=True, quality_ok=False),
    ]
    assert choose_recommendation(results).max_model_len == 32768


def test_a_candidate_that_would_not_load_is_not_recommended():
    """Load failure is usually KV-pool OOM -- cheap, clean, and decisive."""
    results = [
        CandidateResult(max_model_len=8192, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=65536, loaded=False, quality_ok=None),
    ]
    assert choose_recommendation(results).max_model_len == 8192


def test_a_crashed_candidate_is_not_recommended_even_if_it_loaded():
    results = [
        CandidateResult(max_model_len=8192, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=32768, loaded=True, quality_ok=True, crashed=True),
    ]
    assert choose_recommendation(results).max_model_len == 8192


def test_no_viable_candidate_returns_nothing_rather_than_guessing():
    """Silence is a usable answer. A fabricated setting is not."""
    results = [CandidateResult(max_model_len=8192, loaded=False, quality_ok=None)]
    assert choose_recommendation(results) is None


def test_the_recommendation_reports_why_it_stopped_there():
    """An operator needs to know whether the wall was memory or quality.

    Those have different remedies -- more VRAM versus a different quant -- so
    collapsing them into one number throws away the actionable half.
    """
    results = [
        CandidateResult(max_model_len=8192, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=32768, loaded=True, quality_ok=False, gate_tripped="abrupt"),
    ]
    rec = choose_recommendation(results)
    assert rec.limited_by == "quality"
    assert rec.next_failed_at == 32768


def test_a_load_failure_is_reported_as_a_memory_limit():
    results = [
        CandidateResult(max_model_len=8192, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=32768, loaded=False, quality_ok=None),
    ]
    assert choose_recommendation(results).limited_by == "load"


def test_reaching_the_top_of_the_ladder_is_reported_as_unbounded():
    """Nothing failed, so the model's own ceiling is the only limit found.

    Saying 'limited_by: quality' here would invent a wall that was never hit.
    """
    results = [
        CandidateResult(max_model_len=8192, loaded=True, quality_ok=True),
        CandidateResult(max_model_len=131072, loaded=True, quality_ok=True),
    ]
    rec = choose_recommendation(results)
    assert rec.limited_by == "model_ceiling"
    assert rec.next_failed_at is None
