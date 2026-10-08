"""Telling "the model failed" apart from "we starved it" (design §4.2).

Found by the first production run, 2026-09-04. `gpt-oss-20b-gguf` on
llama.cpp failed EVERY probe at the baseline condition with
`["abrupt", "reasoning_only", "wrong"]`, and the run correctly published
nothing. The model was healthy the whole time. Measured against the engine on
port 10007, same prompt, only the budget varied:

    max_tokens=16  -> finish_reason=length, content='',   reasoning truncated
    max_tokens=64  -> finish_reason=stop,   content='42', correct
    max_tokens=512 -> finish_reason=stop,   content='42', correct

The suite's graded probes budget 16-32 tokens, which is ~4x the ANSWER and
was sound for a model that answers directly. A reasoning model spends the
whole budget thinking -- 56 completion tokens for `1234 plus 56` -- so
`content` never starts. Every gate then fires truthfully about an artefact of
our own budget.

`models.supports_reasoning` cannot fix this: it is NULL for this very model
(tri-state; the chat-template detection did not fire) and 1 for qwen3.8-27b.
The response itself is the reliable signal, so this is measured, not consulted.
"""

from __future__ import annotations

from app.stress.gates import Gate, ProbeObservation, Reference, budget_starved, evaluate
from app.stress.probes import escalate_budget


def _obs(content="", reasoning="", finish="length", tokens=16, graded=None):
    return ProbeObservation(
        content=content,
        reasoning_content=reasoning,
        finish_reason=finish,
        output_tokens=tokens,
        repetition_tripped=False,
        graded_ok=graded,
    )


# ---- the signature -------------------------------------------------------


def test_the_starved_signature_is_recognised():
    """No answer, a chain of thought, and OUR ceiling stopped it."""
    assert budget_starved(_obs(reasoning="The user asks: what is 17 + 25?")) is True


def test_an_answer_means_the_budget_was_sufficient():
    """Reasoning plus an answer is a healthy reasoning model, whatever the
    finish_reason: the answer got out."""
    assert budget_starved(_obs(content="42", reasoning="thinking...")) is False


def test_silence_without_reasoning_is_a_real_defect_not_starvation():
    """EMPTY means the engine produced nothing at all.

    Escalating the budget there would spend minutes re-asking a model that has
    already answered the question by saying nothing.
    """
    assert budget_starved(_obs(reasoning="")) is False


def test_a_model_that_stopped_on_its_own_was_not_starved():
    """finish_reason 'stop' means the model chose to end.

    It reasoned, decided it was done, and emitted no answer. That is the
    REASONING_ONLY defect the gate exists for, and a bigger budget cannot fix
    it -- the model was not up against the ceiling.
    """
    assert budget_starved(_obs(reasoning="hmm", finish="stop")) is False


def test_a_dropped_stream_was_not_starved():
    """finish_reason None is the reaper or a disconnect, not our ceiling."""
    assert budget_starved(_obs(reasoning="hmm", finish=None)) is False


# ---- the gates keep telling the truth ------------------------------------


def test_the_gates_still_fire_on_a_starved_probe():
    """budget_starved does not suppress anything; it explains it.

    The observation really is abrupt and really is reasoning-only. The caller
    escalates and re-probes rather than the oracle lying about what it saw --
    keeping the gates honest is what lets the same gates judge the retry.
    """
    gates = evaluate(_obs(reasoning="thinking..."), Reference(0, False))
    assert Gate.ABRUPT in gates
    assert Gate.REASONING_ONLY in gates


# ---- the escalation ladder ----------------------------------------------


def test_the_ladder_climbs_by_four():
    """16 -> 64 is the step that fixed the measured case."""
    assert escalate_budget(16, cap=100_000) == 64
    assert escalate_budget(64, cap=100_000) == 256


def test_the_ladder_stops_at_the_cap():
    """Bounded on purpose: a run must not spend forever on a model that
    reasons without ever answering.

    The cap is supplied by the caller and derived from the model's own context
    (see test_budget_cap.py). It is required rather than defaulted so no call
    site can silently inherit a constant that was right for one model.
    """
    assert escalate_budget(256, cap=1024) == 1024
    assert escalate_budget(1024, cap=1024) is None


def test_the_ladder_never_overshoots_the_cap():
    assert escalate_budget(1023, cap=1024) == 1024


def test_a_generous_probe_still_gets_one_step():
    """The summary probe already budgets 256; it is not exempt."""
    assert escalate_budget(256, cap=4096) == 1024
