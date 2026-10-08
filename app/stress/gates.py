"""Quality gates — the stress test's primary oracle (design §4.2).

A stress run fails when the model stops producing USABLE OUTPUT. The engine
dying is one way that happens and not the most common one; the more useful
signal is that the model is still answering and the answers have gone bad.

Every gate here is a HARD BINARY, deliberately. A graded "coherence score"
would need a threshold with no defensible value, and general hallucination is
not detectable by any cheap, deterministic, model-agnostic means. What IS
detectable is what actually happens at a context limit: answers get short, get
cut off, start looping, or stop following the instruction. Those are binary,
objective, and exactly what a bisection oracle needs.

Everything here is pure. The two impure inputs — whether the repetition
detector tripped, and whether the grader passed — are computed by the probe
client and passed in as verdicts, which is what keeps the whole oracle
testable without a GPU.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

# A probe is SHORT when its answer falls below this fraction of the same
# probe's own baseline answer. Relative, never absolute: some models answer
# tersely when perfectly healthy, and an absolute floor would libel them.
SHORT_RATIO = 0.5


class Gate(str, enum.Enum):
    """Why a probe's output was judged unusable.

    The value strings are published (`gate_tripped`, design §7.5), so they are
    part of the wire contract and must not be renamed casually.
    """

    ABRUPT = "abrupt"
    EMPTY = "empty"
    REASONING_ONLY = "reasoning_only"
    REPEATING = "repeating"
    SHORT = "short"
    WRONG = "wrong"


@dataclass(frozen=True)
class ProbeObservation:
    """What one probe actually produced.

    ``finish_reason`` is ``None`` when the stream ended without a terminal
    frame. That is a real state in this codebase rather than a theoretical one:
    the wall-clock reaper and the client-disconnect path both drop the
    connection without emitting one, so "truncated with no explanation" must be
    distinguishable from "answered briefly".

    ``graded_ok`` is ``None`` for a probe with no checkable answer. None is not
    False: an ungraded probe cannot be wrong, only short, empty, cut off or
    looping.
    """

    content: str
    reasoning_content: str
    finish_reason: str | None
    output_tokens: int
    repetition_tripped: bool
    graded_ok: bool | None


@dataclass(frozen=True)
class Reference:
    """The same probe's answer at the baseline condition (design §4.3).

    ``checks_length`` is False for probes whose correct answer is one or two
    tokens — the echo and single-word probes. Their baseline is too small for a
    ratio to carry information, and WRONG already covers them; running SHORT
    there would manufacture failures out of healthy output.
    """

    output_tokens: int
    checks_length: bool


def budget_starved(obs: ProbeObservation) -> bool:
    """True when OUR ``max_tokens``, not the model, ended the generation.

    Exactly one situation produces all three of: no answer, a chain of thought,
    and a stop caused by the ceiling. A reasoning model spent the whole budget
    thinking and never reached its answer — so the ABRUPT and REASONING_ONLY
    gates are describing our own configuration rather than the model.

    Measured on `gpt-oss-20b-gguf` (llama.cpp, 2026-09-04): at ``max_tokens=16``
    the arithmetic probe returned ``content=''`` with a truncated chain; at 64
    the same prompt returned ``'42'``. That model needs 56 completion tokens to
    add two numbers.

    Each clause carries its own weight, so none may be dropped:

    * content present — the answer got out; a reasoning model that answers is
      simply healthy, whatever else the response says.
    * reasoning absent — nothing was generated at all, which is EMPTY: a real
      defect that a bigger budget cannot fix and must not be retried into.
    * ``finish_reason`` not ``length`` — the model was not against the ceiling.
      ``stop`` means it reasoned and chose to end without answering, which is
      the genuine REASONING_ONLY defect; ``None`` is the reaper or a dropped
      connection.

    This is measured rather than read from ``models.supports_reasoning``: that
    column is tri-state and was NULL for the very model that exposed the bug,
    while being 1 for another. The response is the only reliable witness.
    """
    return (
        not obs.content.strip()
        and bool(obs.reasoning_content.strip())
        and obs.finish_reason == "length"
    )


def evaluate(obs: ProbeObservation, ref: Reference) -> tuple[Gate, ...]:
    """Return every gate this observation trips, in stable order.

    Every gate is evaluated — the caller needs to know WHICH gate ended the
    climb, not merely that one did, because `gate_tripped` is published and an
    operator reading "abrupt" reaches for a different remedy than one reading
    "repeating".
    """
    tripped: set[Gate] = set()

    has_content = bool(obs.content.strip())
    has_reasoning = bool(obs.reasoning_content.strip())

    # EMPTY and REASONING_ONLY are mutually exclusive readings of the same
    # observation (no content), separated by whether the model produced
    # reasoning instead. They are distinct because the remedies differ: EMPTY
    # points at the engine, reasoning-only points at the reasoning budget
    # having swallowed the answer.
    if not has_content:
        tripped.add(Gate.REASONING_ONLY if has_reasoning else Gate.EMPTY)

    # A degenerate baseline means the reference itself was broken. Judging
    # against it would classify every later probe as degraded, so refuse to
    # judge rather than judge wrongly.
    if ref.checks_length and ref.output_tokens > 0:
        if obs.output_tokens < ref.output_tokens * SHORT_RATIO:
            tripped.add(Gate.SHORT)

    # `length` is only a defect because the caller sets max_tokens to ~4x the
    # reference answer; at that budget, exhausting it means runaway rather than
    # a legitimately long reply.
    if obs.finish_reason == "length" or obs.finish_reason is None:
        tripped.add(Gate.ABRUPT)

    if obs.repetition_tripped:
        tripped.add(Gate.REPEATING)

    if obs.graded_ok is False:
        tripped.add(Gate.WRONG)

    return tuple(sorted(tripped, key=lambda g: g.value))
