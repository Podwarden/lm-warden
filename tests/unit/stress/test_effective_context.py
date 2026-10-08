"""What context the ENGINE will actually accept (design §6, budget sizing).

Third abort on the same model, 2026-09-04. `models.max_model_len` is NULL for
`gpt-oss-20b-gguf` — nobody set it — so `budget_cap_for` received
`context_window=None`, fell back to BUDGET_FLOOR, and the needle probe hit
exactly 1024 and starved. Meanwhile the engine's own `/props` reported
`n_ctx: 131072`. The harness had 131k available and used 1k.

Note the deliberate contradiction with `ceiling.py`, which REFUSES this same
number. That module answers "how far can this model go?", where the engine's
report is our own `--ctx-size` echoed back and using it would be circular. This
module answers "how many tokens will the engine accept from us right now?",
where that configured value is precisely the authority. One field, two
questions, two different correct answers.
"""

from __future__ import annotations

from app.stress.probes import context_from_engine


def test_llamacpp_props_reports_its_configured_context():
    """Recorded from llama-server b10731 on 2026-09-04."""
    props = {"default_generation_settings": {"n_ctx": 131072}, "total_slots": 4}
    assert context_from_engine(props) == 131072


def test_vllm_models_reports_max_model_len():
    payload = {"data": [{"id": "m", "max_model_len": 32768}]}
    assert context_from_engine(payload) == 32768


def test_the_ollama_shaped_llamacpp_models_list_carries_nothing():
    """llama.cpp's /v1/models is Ollama-shaped and has no context field. A
    uniform parser would not merely miss it, it would raise."""
    payload = {"models": [{"name": "m", "details": {"format": "gguf"}}]}
    assert context_from_engine(payload) is None


def test_a_malformed_payload_yields_nothing_rather_than_raising():
    for bad in (
        None,
        [],
        "text",
        {"default_generation_settings": "nope"},
        {"data": "nope"},
        {"default_generation_settings": {}},
    ):
        assert context_from_engine(bad) is None


def test_a_nonsense_context_is_rejected():
    """Zero or negative would make budget_cap_for fall to its floor anyway, but
    silently — better to say we learned nothing."""
    assert context_from_engine({"default_generation_settings": {"n_ctx": 0}}) is None
    assert context_from_engine({"data": [{"max_model_len": -1}]}) is None


def test_a_boolean_is_not_a_context():
    assert context_from_engine({"default_generation_settings": {"n_ctx": True}}) is None


def test_the_budget_then_follows_the_engine_not_the_floor():
    """The whole point: 131k of context is not a 1024-token budget."""
    from app.stress.probes import BUDGET_FLOOR, budget_cap_for

    ctx = context_from_engine({"default_generation_settings": {"n_ctx": 131072}})
    assert budget_cap_for(context_window=ctx, prompt_tokens=1024) > BUDGET_FLOOR * 100


def test_the_baseline_is_re_clamped_once_the_engine_has_spoken():
    """The constructor can only see the row.

    A model with a small engine context and no `max_model_len` set would
    otherwise keep the full default baseline and have every probe refused —
    the same failure as the budget, one layer up. Re-clamping only ever
    shrinks, so a roomy model is unaffected.
    """
    from app.stress.probes import BASELINE_TOKENS, baseline_tokens_for

    # Row says nothing; engine says 512.
    assert baseline_tokens_for(context_window=512) < BASELINE_TOKENS
    # Row says nothing; engine says 131072.
    assert baseline_tokens_for(context_window=131072) == BASELINE_TOKENS
