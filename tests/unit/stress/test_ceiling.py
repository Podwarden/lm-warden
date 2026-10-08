"""Resolving the model's own context ceiling (design §6).

Without this the sweep can only ask "is the current setting too ambitious?".
It can never ask "can we go higher?", which is half of the question the whole
feature exists to answer.

The subtlety that makes this non-trivial: what a running engine reports is
what it was CONFIGURED with, not what the model can do. Measured 2026-09-03 on
pw-bonus, llama-server b10731 serving qwen3.8-27b with `--ctx-size 131072`:

    GET /props -> default_generation_settings.n_ctx = 131072

which is our own flag echoed back. Reading that as a ceiling and then
"discovering" we can reach it would be circular -- the sweep would confirm the
setting it started from, every time, and report `limited_by: model_ceiling`.
"""

from __future__ import annotations

from app.stress.ceiling import ceiling_from_config, ceiling_from_engine

# ---- the engine is only authoritative when we imposed nothing -------------


def test_the_engine_value_is_the_ceiling_only_when_nothing_was_configured():
    """With max_model_len unset, vLLM falls back to the model's own maximum.

    That fallback IS the trained ceiling, so the engine has told us something
    we did not already know.
    """
    payload = {"data": [{"id": "m", "max_model_len": 262144}]}
    assert ceiling_from_engine(payload, configured=None) == 262144


def test_a_configured_value_makes_the_engine_report_circular():
    """We passed 32768, so the engine reporting 32768 tells us nothing.

    Accepting it would cap the sweep at the setting we are trying to improve
    on, and then report `model_ceiling` as though the model could go no
    further.
    """
    payload = {"data": [{"id": "m", "max_model_len": 32768}]}
    assert ceiling_from_engine(payload, configured=32768) is None


def test_an_engine_value_above_the_configured_one_is_still_informative():
    """Some engines report the model's maximum regardless of the flag.

    A value strictly greater than what we set cannot have come from our flag,
    so it is real evidence.
    """
    payload = {"data": [{"id": "m", "max_model_len": 262144}]}
    assert ceiling_from_engine(payload, configured=32768) == 262144


def test_the_ollama_shaped_response_yields_nothing():
    """llama.cpp's /v1/models is Ollama-shaped and carries no context at all.

    Recorded verbatim: {"models":[{"name":...,"details":{"format":"gguf"}}]}.
    A uniform parser would not merely miss the field, it would raise.
    """
    payload = {"models": [{"name": "qwen3.8-27b", "details": {"format": "gguf"}}]}
    assert ceiling_from_engine(payload, configured=None) is None


def test_a_malformed_payload_yields_nothing_rather_than_raising():
    for bad in (None, [], {"data": "nope"}, {"data": [{}]}, "text"):
        assert ceiling_from_engine(bad, configured=None) is None


# ---- the portable source -------------------------------------------------


def test_max_position_embeddings_is_the_ceiling():
    """The model's own config, which discovery already keeps."""
    assert ceiling_from_config({"max_position_embeddings": 131072}) == 131072


def test_a_missing_config_yields_nothing():
    """Common for GGUF repos, which often ship no config.json at all.

    unsloth/gpt-oss-20b-GGUF lists notebook.ipynb, params and template -- no
    config.json. Returning None keeps the sweep honest rather than inventing a
    ceiling from nothing.
    """
    assert ceiling_from_config(None) is None
    assert ceiling_from_config({}) is None


def test_a_nonsense_ceiling_is_rejected():
    """A zero or negative value would make the candidate ladder degenerate."""
    assert ceiling_from_config({"max_position_embeddings": 0}) is None
    assert ceiling_from_config({"max_position_embeddings": -1}) is None


def test_a_non_integer_ceiling_is_rejected():
    assert ceiling_from_config({"max_position_embeddings": "lots"}) is None
    assert ceiling_from_config({"max_position_embeddings": None}) is None


# ---- what it means downstream -------------------------------------------


def test_no_ceiling_still_lets_the_sweep_ask_the_other_question():
    """Degrading is not failing.

    With no ceiling the sweep cannot look for headroom, but it can still ask
    whether the current setting is too ambitious -- which is the more urgent
    question for a model that is crashing.
    """
    from app.stress.sweep import plan_candidates

    got = plan_candidates(
        current=131072, model_ceiling=None, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert got and max(got) <= 131072


def test_a_ceiling_unlocks_the_headroom_question():
    from app.stress.sweep import plan_candidates

    got = plan_candidates(
        current=8192, model_ceiling=131072, vram_budget_bytes=None, kv_bytes_per_token=None
    )
    assert max(got) == 131072
    assert len(got) > 1
