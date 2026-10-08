"""The classifier and accumulator against RECORDED llama.cpp behaviour.

Every payload here was captured verbatim from llama-server b10731 on pw-bonus
on 2026-09-03 (tests/fixtures/stress/llamacpp_recorded.json). That matters:
the synthetic shapes I would otherwise have invented were wrong in two ways
that would have shipped as bugs.

llama.cpp is a first-class backend -- it is serving both models on that host
right now -- so testing only against a vLLM-shaped fake would leave the path
that runs in production unexercised.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.stress.gates import Gate, Reference, evaluate
from app.stress.outcomes import Class, Observables, RefusalEvidence, classify
from app.stress.stream import StreamAccumulator

_REC = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "stress"
        / "llamacpp_recorded.json"
    ).read_text()
)


def _obs(**over) -> Observables:
    base = dict(
        http_status=200,
        body={},
        engine_healthy=True,
        process_alive=True,
        returncode=None,
        row_status="loaded",
        deadline_exceeded=False,
        host_oom_counters_moved=False,
        gates_tripped=(),
        warden_restarted=False,
        foreign_traffic_on_this_model=False,
        neighbour_unhealthy=False,
        status_changed_externally=False,
    )
    base.update(over)
    return Observables(**base)


# ---- structured refusal --------------------------------------------------


def test_the_recorded_over_context_refusal_is_classified_as_a_refusal():
    rec = _REC["refusal_over_context"]
    out = classify(
        _obs(http_status=rec["status"], body=rec["body"]),
        refusal=RefusalEvidence(beyond_declared_limit=True),
    )
    assert out.cls is Class.REFUSED


def test_error_type_alone_is_definitive_evidence():
    """llama.cpp names the condition in a machine-readable field.

    `exceed_context_size_error` is not prose that drifts across versions -- it
    is a typed discriminator. When it is present no confirmation probe is
    needed at all, which removes a whole round trip from the common case.
    """
    rec = _REC["refusal_over_context"]
    out = classify(
        _obs(http_status=rec["status"], body=rec["body"]),
        refusal=RefusalEvidence(),  # nothing else confirmed
    )
    assert out.cls is Class.REFUSED
    assert out.refusal_confidence == "high"


def test_a_different_error_type_is_not_a_context_refusal():
    """invalid_request_error is a harness bug, not a model limit.

    Recording it as a capacity limit would publish our own mistake as the
    model's ceiling.
    """
    rec = _REC["error_invalid_request"]
    out = classify(
        _obs(http_status=rec["status"], body=rec["body"]),
        refusal=RefusalEvidence(),
    )
    assert out.cls is not Class.REFUSED


def test_the_recorded_refusal_carries_exact_token_counts():
    """n_prompt_tokens/n_ctx beat any estimate we could make ourselves."""
    err = _REC["refusal_over_context"]["body"]["error"]
    assert err["n_prompt_tokens"] == 144052
    assert err["n_ctx"] == 131072


# ---- the unknown-model trap ---------------------------------------------


def test_an_unknown_model_name_produces_output_that_looks_like_degradation():
    """THE TRAP, recorded verbatim.

    A wrong model name does not error. llama.cpp returns 200 and serves the
    loaded model with empty content and finish_reason=length -- which trips
    REASONING_ONLY and ABRUPT together. A harness bug would be published as a
    measured quality limit.

    This test exists to pin the behaviour so the runner is obliged to send the
    real served_model_name rather than a placeholder.
    """
    body = _REC["unknown_model_is_NOT_an_error"]["body"]
    msg = body["choices"][0]["message"]
    a = StreamAccumulator()
    obs = a.observation(graded_ok=None)
    obs = type(obs)(
        content=msg["content"],
        reasoning_content=msg["reasoning_content"],
        finish_reason=body["choices"][0]["finish_reason"],
        output_tokens=4,
        repetition_tripped=False,
        graded_ok=None,
    )
    tripped = evaluate(obs, Reference(output_tokens=20, checks_length=True))
    assert Gate.REASONING_ONLY in tripped
    assert Gate.ABRUPT in tripped
    assert _REC["unknown_model_is_NOT_an_error"]["status"] == 200


# ---- streaming shapes ----------------------------------------------------


def test_the_recorded_first_frame_has_null_content():
    """delta.content is null on the opening frame; it must not append "None"."""
    a = StreamAccumulator()
    a.feed_line("data: " + json.dumps(_REC["stream_first_frame"]))
    assert a.content == ""


def test_the_recorded_usage_frame_has_empty_choices():
    """Usage arrives in a final frame with choices: [].

    An accumulator that only read usage from frames carrying a choice would
    miss it entirely and fall back to a delta count for the published prompt
    size.
    """
    a = StreamAccumulator()
    a.feed_line("data: " + json.dumps(_REC["stream_usage_frame"]))
    assert a.prompt_tokens == 54
    assert a.completion_tokens == 8


def test_reasoning_content_arrives_as_its_own_delta_field():
    a = StreamAccumulator()
    a.feed_line("data: " + json.dumps(_REC["stream_reasoning_frame"]))
    assert a.reasoning_content == "/"
    assert a.content == ""


def test_cached_tokens_are_visible_so_the_prefix_defeat_is_checkable():
    """prompt_tokens_details.cached_tokens exposes prefix-cache reuse.

    The design treats defeating the prefix cache as a precaution taken on
    faith. It is measurable: 50 of 54 tokens were served from cache on a
    repeated prompt, so the runner can ASSERT the unique preamble worked
    instead of assuming it.
    """
    a = StreamAccumulator()
    a.feed_line("data: " + json.dumps(_REC["stream_usage_frame"]))
    assert a.cached_tokens == 50


# ---- backend asymmetry ---------------------------------------------------


def test_llamacpp_publishes_no_vllm_metrics():
    """The concurrency axis cannot use preemption onset on this backend."""
    for absent in _REC["metrics_absent"]:
        assert absent not in _REC["metrics_present"]


def test_llamacpp_does_publish_a_deferral_counter():
    """Correcting an over-strong claim in the design.

    v2 said llama.cpp has no preemption-like signal at all. It publishes
    `llamacpp:requests_deferred`, which is a queueing signal and may serve a
    similar role. Whether it tracks preemption closely enough is unvalidated.
    """
    assert "llamacpp:requests_deferred" in _REC["metrics_present"]


def test_the_models_endpoint_is_ollama_shaped_not_openai_shaped():
    """A uniform /v1/models parser breaks on llama.cpp."""
    body = _REC["models"]["body"]
    assert "models" in body and "data" not in body
    assert "name" in body["models"][0] and "id" not in body["models"][0]
