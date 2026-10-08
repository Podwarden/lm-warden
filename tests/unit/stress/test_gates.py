"""Quality gates — the stress test's primary oracle.

Design §4.2. A run failed when the model stopped producing usable output, not
(only) when the engine died. Every gate here is a HARD BINARY: a graded
"coherence score" would need a threshold nobody could defend, and general
hallucination is not detectable at all. What is detectable is what actually
happens at context limits — answers get short, get cut off, start looping, or
stop following the instruction.

The gates are pure so the whole oracle is testable without a GPU.
"""

from __future__ import annotations

from app.stress.gates import Gate, ProbeObservation, Reference, evaluate


def _obs(**over) -> ProbeObservation:
    base = dict(
        content="Paris is the capital of France.",
        reasoning_content="",
        finish_reason="stop",
        output_tokens=8,
        repetition_tripped=False,
        graded_ok=True,
    )
    base.update(over)
    return ProbeObservation(**base)


def _ref(**over) -> Reference:
    base = dict(output_tokens=8, checks_length=True)
    base.update(over)
    return Reference(**base)


# ---- the healthy case ----------------------------------------------------


def test_a_good_answer_trips_nothing():
    assert evaluate(_obs(), _ref()) == ()


# ---- empty / reasoning-only ---------------------------------------------


def test_empty_content_trips_empty():
    assert Gate.EMPTY in evaluate(_obs(content="   \n  ", output_tokens=0), _ref())


def test_reasoning_only_is_its_own_gate_not_empty():
    """All tokens went to reasoning_content, none to content.

    A distinct and common failure: the model thought and never answered. It
    must not be reported as EMPTY, because the remedy differs — EMPTY suggests
    the engine produced nothing, reasoning-only suggests the reasoning budget
    swallowed the response.
    """
    tripped = evaluate(
        _obs(content="", reasoning_content="Let me think about this...", output_tokens=40),
        _ref(),
    )
    assert Gate.REASONING_ONLY in tripped
    assert Gate.EMPTY not in tripped


def test_whitespace_only_reasoning_does_not_rescue_an_empty_answer():
    tripped = evaluate(_obs(content="", reasoning_content="   ", output_tokens=0), _ref())
    assert Gate.EMPTY in tripped
    assert Gate.REASONING_ONLY not in tripped


# ---- short ---------------------------------------------------------------


def test_short_is_relative_to_this_probe_s_own_baseline():
    """Not an absolute threshold.

    Some models answer tersely when perfectly healthy. Gating absolutely would
    libel them; gating against the same probe's baseline answer does not.
    """
    assert Gate.SHORT in evaluate(_obs(output_tokens=2), _ref(output_tokens=40))
    assert Gate.SHORT not in evaluate(_obs(output_tokens=38), _ref(output_tokens=40))


def test_short_is_skipped_for_probes_whose_answer_is_meant_to_be_tiny():
    """`checks_length=False` for the one-word and echo probes.

    Their baseline is 1-2 tokens, so any ratio is noise, and WRONG already
    covers them. Running SHORT there would manufacture failures.
    """
    assert Gate.SHORT not in evaluate(
        _obs(output_tokens=1), _ref(output_tokens=2, checks_length=False)
    )


def test_short_never_trips_on_a_degenerate_baseline():
    """A baseline of 0 tokens means the reference itself was broken.

    Dividing by it, or trusting a ratio of it, would classify every later probe
    as degraded. Refuse to judge instead.
    """
    assert Gate.SHORT not in evaluate(_obs(output_tokens=0), _ref(output_tokens=0))


# ---- abrupt --------------------------------------------------------------


def test_finish_reason_length_is_abrupt():
    """max_tokens is set to ~4x the reference answer, so hitting it is runaway."""
    assert Gate.ABRUPT in evaluate(_obs(finish_reason="length"), _ref())


def test_a_stream_with_no_terminal_frame_is_abrupt():
    """finish_reason None means the stream just stopped.

    Real in this codebase: the wall-clock reaper and the disconnect path emit
    no terminal frame at all. It must not be read as a normal short answer.
    """
    assert Gate.ABRUPT in evaluate(_obs(finish_reason=None), _ref())


def test_stop_is_not_abrupt():
    assert Gate.ABRUPT not in evaluate(_obs(finish_reason="stop"), _ref())


# ---- repeating / wrong ---------------------------------------------------


def test_repetition_verdict_is_passed_in_not_recomputed():
    """RunawayDetector is fed streaming deltas; the gate takes its verdict.

    Keeping the detection in the probe client and the judgement here is what
    lets every gate stay pure.
    """
    assert Gate.REPEATING in evaluate(_obs(repetition_tripped=True), _ref())


def test_a_failed_grader_trips_wrong():
    assert Gate.WRONG in evaluate(_obs(graded_ok=False), _ref())


def test_an_ungraded_probe_cannot_trip_wrong():
    assert Gate.WRONG not in evaluate(_obs(graded_ok=None), _ref())


# ---- composition ---------------------------------------------------------


def test_gates_are_independent_and_all_reported():
    """The run needs to know WHICH gate ended the climb, not just that one did.

    `gate_tripped` is published (design §7.5), so a caller must be able to see
    every gate that fired, in a stable order.
    """
    tripped = evaluate(
        _obs(
            content="",
            reasoning_content="thinking",
            finish_reason="length",
            repetition_tripped=True,
            graded_ok=False,
            output_tokens=1,
        ),
        _ref(output_tokens=40),
    )
    assert Gate.REASONING_ONLY in tripped
    assert Gate.ABRUPT in tripped
    assert Gate.REPEATING in tripped
    assert Gate.WRONG in tripped
    assert list(tripped) == sorted(tripped, key=lambda g: g.value)


def test_evaluate_returns_a_tuple_so_a_verdict_cannot_be_mutated():
    assert isinstance(evaluate(_obs(), _ref()), tuple)
