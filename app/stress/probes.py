"""Probe generation and grading (design §4.5, §4.6).

Everything is generated from one template plus a seed. There is no corpus
directory, no manifest hash and no integrity test: bench-v2 had all three —
seven size buckets, dozens of JSON files — and they were a meaningful fraction
of the 23k lines that got it deleted.

Two properties here are load-bearing rather than tidy:

**The unique preamble defeats prefix caching.** vLLM's prefix cache is on by
default and block-aligned from the start of the prompt. Concurrent probes that
share a prefix have their KV deduplicated, so the concurrency axis would report
a number far above anything real traffic achieves. The entropy therefore has to
sit at position 0 — merely being present somewhere is not enough — and it has
to differ per concurrent slot, so simultaneous probes cannot dedupe against
each other either.

**Grading is normalised.** A model that hyphenates or spaces an answer
differently has not failed to recall it. With M/M confirmation, one formatting
quirk would kill a rung and silently truncate the published context.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field

# Roughly one token per four characters of English prose. Only used to AIM a
# probe: the recorded length always comes from the engine's own
# usage.prompt_tokens, because our tokenizer count omits the chat template,
# role tokens and BOS and so systematically undercounts.
CHARS_PER_TOKEN = 4

_ALPHANUM = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")

_FILLER_SENTENCES = (
    "The survey team recorded ambient conditions at each station before dawn.",
    "Sediment cores were catalogued by depth and stored in numbered trays.",
    "Wind speed held steady through the afternoon with occasional gusts inland.",
    "Two observers cross-checked every reading against the printed log sheet.",
    "The northern transect required a detour around the flooded access road.",
)


def unique_preamble(run_id: str, slot: int = 0) -> str:
    """High-entropy text for position 0 of every prompt.

    Derived from the run id so a run is reproducible, and from the slot so two
    probes running concurrently in the same run cannot share a cacheable
    prefix.
    """
    digest = hashlib.sha256(f"{run_id}:{slot}".encode()).hexdigest()
    return f"[ref {digest[:32]}]"


@dataclass(frozen=True)
class NeedlePrompt:
    text: str
    needle: str
    needle_value: str


def _filler(rng: random.Random, approx_chars: int) -> str:
    """Deterministic prose with unique section numbering.

    The numbering is not decoration: it keeps long fillers from containing long
    verbatim repeats, which would otherwise trip the repetition gate on a
    summary probe for reasons that have nothing to do with the model.
    """
    out: list[str] = []
    size = 0
    n = 0
    while size < approx_chars:
        n += 1
        line = f"§{n}. {rng.choice(_FILLER_SENTENCES)}"
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


def make_needle_prompt(
    *, run_id: str, seed: int, approx_tokens: int, offset: float, slot: int = 0
) -> NeedlePrompt:
    """Filler with one planted fact at a controlled relative position.

    ``offset`` is where the needle sits as a fraction of the filler. 0.5 is the
    informative case; the outer positions are primacy and recency and are the
    easy ones, which is why ``quick`` mode drops them.
    """
    rng = random.Random(f"{run_id}:{seed}:{offset}")
    value = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    needle = f"the vault code is {value}"

    approx_chars = max(1, approx_tokens * CHARS_PER_TOKEN)
    body = _filler(rng, approx_chars)
    cut = int(len(body) * min(max(offset, 0.0), 1.0))
    # Snap to a line boundary so the needle is never spliced mid-sentence,
    # which would make it harder to retrieve for reasons we did not intend to
    # measure.
    nl = body.find("\n", cut)
    cut = nl if nl != -1 else len(body)

    text = (
        f"{unique_preamble(run_id, slot)}\n"
        f"{body[:cut]}\n"
        f"Note: {needle}.\n"
        f"{body[cut:]}\n\n"
        f"What is the vault code? Answer with the code only."
    )
    return NeedlePrompt(text=text, needle=value, needle_value=value)


# ---- graders -------------------------------------------------------------


def _normalise(s: str) -> str:
    return _ALPHANUM.sub("", s.lower())


def grade_substring(expected: str, actual: str) -> bool:
    """Normalised containment.

    Case, hyphens, em-dashes and spacing are all stripped from both sides, so
    ``QX7-41839`` matches ``qx7 41839`` and ``QX7—41839``. A partial token
    still fails, because normalisation removes separators, not characters.
    """
    return _normalise(expected) in _normalise(actual)


def grade_regex(pattern: str, actual: str) -> bool:
    return re.match(pattern, actual, re.IGNORECASE) is not None


def grade_json_equal(expected: object, actual: str) -> bool:
    """Parse-and-compare, tolerating a code fence.

    Models wrap JSON in fences constantly and that is not a structural failure,
    so stripping one is not leniency — it is avoiding a false positive on the
    gate.
    """
    text = _FENCE.sub("", actual.strip())
    try:
        return json.loads(text) == expected
    except (ValueError, TypeError):
        return False


def grade_numeric(expected: float, actual: str, *, tol: float) -> bool:
    """First number in the output, within a relative tolerance."""
    m = _NUMBER.search(actual)
    if not m:
        return False
    try:
        got = float(m.group(0))
    except ValueError:
        return False
    return abs(got - expected) <= max(abs(expected) * tol, tol)


# ---- the suite -----------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One graded probe.

    ``checks_length`` is False where the correct answer is one or two tokens:
    a length ratio against such a baseline is noise, and the grader already
    covers correctness there.

    ``allow_repetition_gate`` is True only for the free-text summary. The
    detector emits shingles at every character offset, so a repeated token
    trips it — and the echo probe repeats a token on purpose.
    """

    id: str
    messages: list[dict]
    max_tokens: int
    checks_length: bool
    allow_repetition_gate: bool
    expected: object = None
    grader: str = "none"
    tol: float = 0.0
    meta: dict = field(default_factory=dict)


#: The reference condition for a stress run. SHORT is baseline-relative
#: (design §4.3), so every later gate is judged against a probe taken here.
#: Kept identical across models wherever it fits, so two runs of the same model
#: stay comparable -- but clamped for models that cannot hold it.
BASELINE_TOKENS = 1024

#: Fraction of a small model's context the baseline PROMPT may occupy. The
#: remainder is what the reference answer generates into; a baseline that fills
#: the context produces an empty reference by construction, and every later
#: SHORT verdict would be measured against nothing.
BASELINE_CONTEXT_SHARE = 0.5


def baseline_tokens_for(*, context_window: int | None) -> int:
    """The reference prompt size for THIS model.

    ``BASELINE_TOKENS`` is a harness default, not a fact about any model. A
    model with a 512-token context cannot have a 1024-token reference: every
    probe would be refused and the run would report a limit that is really our
    own default. An unknown context keeps the default, because guessing smaller
    would make runs incomparable for no measured reason.
    """
    if not context_window or context_window <= 0:
        return BASELINE_TOKENS
    return min(BASELINE_TOKENS, max(1, int(context_window * BASELINE_CONTEXT_SHARE)))


#: Floor for the escalation ceiling, used only when the model's context is
#: unknown. This is a bound on RUN COST -- how much one probe may spend -- and
#: is a property of the harness, not a claim about what any model needs. Every
#: model whose context we CAN read is bounded by that instead.
BUDGET_FLOOR = 1024

#: The smallest budget worth probing at all. Not a claim about any model:
#: a probe granted zero tokens measures nothing.
MIN_ANSWER_BUDGET = 16


def _positive_int(value: object) -> int | None:
    return value if not isinstance(value, bool) and isinstance(value, int) and value > 0 else None


def context_from_engine(payload: object) -> int | None:
    """The context the ENGINE will accept, read from its own report.

    Used to size probe budgets when ``models.max_model_len`` is unset, which is
    the common case: nobody has to set it, and on 2026-09-04 a NULL row sent the
    needle probe into ``BUDGET_FLOOR`` and aborted three runs while the engine
    was serving 131,072 tokens.

    Deliberately the opposite policy to ``ceiling.ceiling_from_engine``, which
    REFUSES this same number. That module asks "how far can this model go?",
    where a running engine reports our own ``--ctx-size`` back at us and
    believing it would be circular. This asks "how many tokens will the engine
    accept from us right now?", and for that the configured value is exactly
    the authority. One field, two questions, two correct answers.

    Both engine shapes, because the harness must work for either:

    * llama.cpp ``GET /props`` -> ``default_generation_settings.n_ctx``
    * vLLM ``GET /v1/models`` -> ``data[0].max_model_len``

    llama.cpp's own ``/v1/models`` is Ollama-shaped and carries no context at
    all, so a caller may have to try both endpoints; this reads whichever it is
    handed and returns None for anything it does not recognise.
    """
    if not isinstance(payload, dict):
        return None

    settings = payload.get("default_generation_settings")
    if isinstance(settings, dict):
        found = _positive_int(settings.get("n_ctx"))
        if found:
            return found

    data = payload.get("data")
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                found = _positive_int(entry.get("max_model_len"))
                if found:
                    return found
    return None


def budget_cap_for(*, context_window: int | None, prompt_tokens: int) -> int:
    """The largest ``max_tokens`` worth trying for a probe on this model.

    Derived, never constant. The escalation ladder used to stop at a hardcoded
    1024, chosen because it suited the model in front of us: on
    `gpt-oss-20b-gguf` the needle probe hit that ceiling and the run aborted
    with `aborted_no_baseline`. Raising the constant would have fixed that one
    model and mis-sized every other — a 2k-context model cannot spend 1024
    tokens reasoning about a 1.5k prompt, and a 262k model can spend far more.

    The bound that holds for every model is arithmetic: the answer has to fit
    in the context alongside the prompt. Anything beyond it the engine refuses,
    so probing there measures the configuration rather than the model.

    A prompt that already fills the context still gets ``BUDGET_FLOOR``: the
    alternative is a cap of zero, which kills escalation at the first step and
    reports a model limit that is really a subtraction.
    """
    if not context_window or context_window <= 0:
        return BUDGET_FLOOR
    # A KNOWN context is the real bound, and the floor must not override it: a
    # 2k model with a 1.5k prompt has 548 tokens left, and capping at 1024
    # would just hand the engine a request it refuses. MIN_ANSWER_BUDGET keeps
    # a prompt that fills the context from producing a cap of zero, which would
    # kill escalation at the first step and report a subtraction as a model
    # limit.
    return max(MIN_ANSWER_BUDGET, context_window - max(0, prompt_tokens))


def escalate_budget(current: int, *, cap: int) -> int | None:
    """The next ``max_tokens`` to try, or None when the ladder is exhausted.

    Steps by 4x because that is the granularity the measurement supports and
    not more: 16 -> 64 is precisely the step that turned ``content=''`` into
    an answer on the first model this was measured against. A 2x ladder would
    spend twice the probes to learn the same thing; a 10x one would overshoot
    and make the ABRUPT gate meaningless at the budget it settles on.

    ``cap`` is required rather than defaulted, so that no call site can
    silently inherit a constant that was right for one model. See
    :func:`budget_cap_for`.
    """
    if current >= cap:
        return None
    return min(current * 4, cap)


def build_suite(*, run_id: str, seed: int, approx_tokens: int) -> list[Probe]:
    """Six probes, deterministic for a given (run_id, seed).

    ``max_tokens`` is ~4x each probe's reference answer throughout, because the
    ABRUPT gate reads ``finish_reason == "length"`` as a defect and that is
    only sound when the budget was never the real constraint.
    """
    rng = random.Random(f"{run_id}:{seed}:suite")
    pre = unique_preamble(run_id)

    a, b = rng.randint(1000, 9999), rng.randint(11, 99)
    echo_token = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
    needle = make_needle_prompt(run_id=run_id, seed=seed, approx_tokens=approx_tokens, offset=0.5)

    def msg(text: str) -> list[dict]:
        return [{"role": "user", "content": f"{pre}\n{text}"}]

    return [
        Probe(
            id="arithmetic",
            messages=msg(f"What is {a} plus {b}? Reply with the number only."),
            max_tokens=16,
            checks_length=False,
            allow_repetition_gate=False,
            expected=float(a + b),
            grader="numeric",
            tol=0.001,
        ),
        Probe(
            id="echo",
            messages=msg(f"Repeat exactly, with no other text: {echo_token}"),
            max_tokens=24,
            checks_length=False,
            allow_repetition_gate=False,
            expected=echo_token,
            grader="substring",
        ),
        Probe(
            id="one_word",
            messages=msg(
                "Is the sky blue on a clear day? Answer with exactly one word: yes or no."
            ),
            max_tokens=16,
            checks_length=False,
            allow_repetition_gate=False,
            expected=r"^\W*(yes|no)\W*$",
            grader="regex",
        ),
        Probe(
            id="json",
            messages=msg('Output exactly this JSON and nothing else: {"a": 1}'),
            max_tokens=32,
            checks_length=False,
            allow_repetition_gate=False,
            expected={"a": 1},
            grader="json",
        ),
        Probe(
            id="needle",
            messages=msg(needle.text),
            max_tokens=24,
            checks_length=False,
            allow_repetition_gate=False,
            expected=needle.needle_value,
            grader="substring",
            meta={"offset": 0.5, "approx_tokens": approx_tokens},
        ),
        Probe(
            id="summary",
            messages=msg(
                f"{needle.text}\n\nNow ignore the code. In two sentences, "
                "summarise what the survey team recorded."
            ),
            max_tokens=256,
            checks_length=True,
            allow_repetition_gate=True,
            expected="survey",
            grader="substring",
        ),
    ]
