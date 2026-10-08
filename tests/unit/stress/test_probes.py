"""Probe generation and grading (design §4.5, §4.6, §9.2).

Two properties here are not cosmetic and would silently invalidate whole axes
if they regressed:

* **Every prompt carries a unique high-entropy preamble at position 0.** vLLM's
  prefix cache is block-aligned and on by default, so concurrent probes sharing
  a prefix get their KV deduplicated and the measured concurrency comes out far
  above what real traffic ever gets. Without this the concurrency axis measures
  nothing at all.
* **Needles are graded on normalised text.** An exact substring match false-fails
  on hyphenation, spacing or a tokenizer split, and with M/M confirmation a
  single formatting quirk kills the rung and truncates the published limit.
"""

from __future__ import annotations

import json

from app.stress.probes import (
    build_suite,
    grade_json_equal,
    grade_numeric,
    grade_regex,
    grade_substring,
    make_needle_prompt,
    unique_preamble,
)

# ---- prefix-cache defeat -------------------------------------------------


def test_two_runs_get_different_preambles():
    assert unique_preamble("run-a") != unique_preamble("run-b")


def test_the_preamble_is_stable_within_one_run():
    """Reproducibility: the same run must be able to repeat a probe verbatim."""
    assert unique_preamble("run-a") == unique_preamble("run-a")


def test_the_preamble_sits_at_position_zero():
    """A shared prefix anywhere before the unique part still gets cached.

    vLLM's cache is block-aligned from the start of the prompt, so the entropy
    has to be first — not merely present.
    """
    p = make_needle_prompt(run_id="r1", seed=1, approx_tokens=200, offset=0.5)
    assert p.text.startswith(unique_preamble("r1"))


def test_concurrent_probes_in_one_run_do_not_share_a_prefix():
    """Even inside a run, simultaneous probes must not dedupe against each other."""
    a = make_needle_prompt(run_id="r1", seed=1, approx_tokens=200, offset=0.5, slot=0)
    b = make_needle_prompt(run_id="r1", seed=1, approx_tokens=200, offset=0.5, slot=1)
    assert a.text[:64] != b.text[:64]


# ---- the needle ----------------------------------------------------------


def test_the_needle_is_placed_at_the_requested_relative_offset():
    early = make_needle_prompt(run_id="r", seed=1, approx_tokens=2000, offset=0.1)
    late = make_needle_prompt(run_id="r", seed=1, approx_tokens=2000, offset=0.9)
    assert early.text.index(early.needle) < late.text.index(late.needle)


def test_the_needle_does_not_occur_in_the_filler():
    """A high-entropy token cannot collide with prose by accident.

    If it could, recall would pass for the wrong reason and the effective
    context would be over-reported.
    """
    p = make_needle_prompt(run_id="r", seed=1, approx_tokens=4000, offset=0.5)
    assert p.text.count(p.needle) == 1


def test_the_needle_is_regenerated_per_run():
    """It must not be memorisable across runs, or a model could learn it."""
    a = make_needle_prompt(run_id="r1", seed=1, approx_tokens=200, offset=0.5)
    b = make_needle_prompt(run_id="r2", seed=1, approx_tokens=200, offset=0.5)
    assert a.needle != b.needle


def test_prompt_length_tracks_the_request():
    small = make_needle_prompt(run_id="r", seed=1, approx_tokens=200, offset=0.5)
    large = make_needle_prompt(run_id="r", seed=1, approx_tokens=4000, offset=0.5)
    assert len(large.text) > len(small.text) * 5


# ---- graders -------------------------------------------------------------


def test_substring_grading_is_normalised():
    """The model may hyphenate, space or case the answer differently.

    None of those are recall failures, and treating them as such would truncate
    the published context on a formatting quirk.
    """
    assert grade_substring("QX7-41839", "The vault code is qx7 41839.") is True
    assert grade_substring("QX7-41839", "The code is QX7—41839") is True
    assert grade_substring("QX7-41839", "I don't know.") is False


def test_substring_grading_does_not_match_a_partial_token():
    assert grade_substring("QX741839", "QX7418") is False


def test_regex_grading():
    assert grade_regex(r"^\s*(yes|no)\s*$", "yes") is True
    assert grade_regex(r"^\s*(yes|no)\s*$", "Yes, because...") is False


def test_json_grading_ignores_formatting_but_not_content():
    assert grade_json_equal({"a": 1}, '{"a": 1}') is True
    assert grade_json_equal({"a": 1}, '  {"a":1}  ') is True
    assert grade_json_equal({"a": 1}, '{"a": 2}') is False


def test_json_grading_tolerates_a_fenced_answer():
    """Models wrap JSON in code fences constantly; that is not a failure."""
    assert grade_json_equal({"a": 1}, '```json\n{"a": 1}\n```') is True


def test_json_grading_fails_on_unparseable_output():
    assert grade_json_equal({"a": 1}, "sure, here you go!") is False


def test_numeric_grading_has_tolerance():
    assert grade_numeric(1234.0, "The answer is 1234", tol=0.001) is True
    assert grade_numeric(1234.0, "1234.0001", tol=0.001) is True
    assert grade_numeric(1234.0, "9999", tol=0.001) is False


def test_numeric_grading_fails_when_no_number_is_present():
    assert grade_numeric(1234.0, "I cannot compute that", tol=0.001) is False


# ---- the suite -----------------------------------------------------------


def test_the_suite_has_the_six_designed_probes():
    suite = build_suite(run_id="r", seed=1, approx_tokens=1000)
    assert len(suite) == 6
    assert {p.id for p in suite} == {"arithmetic", "echo", "one_word", "json", "needle", "summary"}


def test_only_the_summary_probe_runs_the_repetition_gate():
    """Shingles fire at every character offset, so a repeated token trips it.

    The echo probe deliberately repeats a token, and the graded probes emit
    short structured answers. Running the gate there would manufacture
    failures out of correct output.
    """
    suite = {p.id: p for p in build_suite(run_id="r", seed=1, approx_tokens=1000)}
    assert suite["summary"].allow_repetition_gate is True
    assert suite["echo"].allow_repetition_gate is False
    assert suite["json"].allow_repetition_gate is False


def test_tiny_answer_probes_do_not_check_length():
    """Their baseline is 1-2 tokens, so a ratio carries no information."""
    suite = {p.id: p for p in build_suite(run_id="r", seed=1, approx_tokens=1000)}
    assert suite["one_word"].checks_length is False
    assert suite["echo"].checks_length is False
    assert suite["summary"].checks_length is True


def test_max_tokens_is_generous_so_that_hitting_it_means_runaway():
    """ABRUPT reads finish_reason == 'length' as a defect.

    That is only sound when the budget was never the constraint, so every probe
    gets roughly 4x its reference answer.
    """
    suite = {p.id: p for p in build_suite(run_id="r", seed=1, approx_tokens=1000)}
    assert suite["one_word"].max_tokens >= 16
    assert suite["summary"].max_tokens >= 128


def test_the_suite_is_deterministic_for_a_given_run_and_seed():
    a = build_suite(run_id="r", seed=7, approx_tokens=500)
    b = build_suite(run_id="r", seed=7, approx_tokens=500)
    assert [p.messages for p in a] == [p.messages for p in b]


def test_every_probe_carries_the_run_s_unique_preamble():
    for p in build_suite(run_id="rX", seed=1, approx_tokens=500):
        assert unique_preamble("rX") in json.dumps(p.messages)
