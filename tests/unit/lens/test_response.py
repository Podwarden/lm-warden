"""Prompt Lens response signals R-01..R-10 and their likely causes (spec 8, 11.3)."""

import json
from dataclasses import replace

from app.lens.response import RESPONSE_CAUSES, bad_turns, response_findings
from app.lens.schema_lint import LINT_NESTED_NO_REQUIRED, LINT_NO_TYPE, LINT_SHORT_DESC
from app.lens.turns import turns_of
from tests.unit.lens.seed import args_hash, seeded_lint, seeded_rows, tool_hash


def _turns(rows, **kw):
    ts = turns_of(rows, own=True)
    return [replace(t, **kw) for t in ts] if kw else ts


def _run(rows, lint=None, **kw):
    return response_findings(_turns(rows), lint=lint or {}, **kw)


def _of(findings, rule):
    return [f for f in findings if f.rule == rule]


def _one(findings, rule):
    hits = _of(findings, rule)
    assert len(hits) == 1, [(f.rule, f.cause, f.turns) for f in findings]
    return hits[0]


def _at(rows, at, **kw):
    return [replace(r, **kw) if int(r.id.rsplit("-", 1)[1]) in at else r for r in rows]


def test_causes_are_the_spec_codes():
    assert RESPONSE_CAUSES[-1] == "none_found"
    assert {"limit_below_need", "parser_mismatch_admin", "server_load"} <= set(RESPONSE_CAUSES)
    assert len(RESPONSE_CAUSES) == 21


def test_findings_hold_codes_and_numbers():
    for fault in ("bad_schema", "loop", "empty_reply_template", "truncating_max_tokens"):
        rows = seeded_rows(fault)
        for f in _run(rows, seeded_lint(fault)):
            assert f.cause in RESPONSE_CAUSES
            assert all(isinstance(v, int) for v in f.counts.values())
            json.dumps(f.to_json())


# -- R-01 / R-02 ------------------------------------------------------------------


def test_r01_thinking_used_the_budget():
    rows = _at(
        seeded_rows("truncating_max_tokens", cut_at=(8, 15), client_max_tokens=4_096),
        (8, 15),
        reasoning_tokens=3_500,
    )
    f = _one(_run(rows), "R-01")
    assert f.cause == "thinking_used_budget"


def test_r01_cause_order_limit_first_then_thinking():
    # Cuts >= 10 % of turns (3 of 24) with most of each cut reply as
    # reasoning: the limit test comes first (spec 8.2) and wins.
    rows = _at(
        seeded_rows("truncating_max_tokens", cut_at=(8, 15, 20)),
        (8, 15, 20),
        reasoning_tokens=900,
    )
    f = _one(_run(rows), "R-01")
    assert f.cause == "limit_below_need" and f.counts["p90"] == 1_025 > f.counts["m"]
    # Under 10 % (2 of 24): the p90 is an uncut reply, thinking is reachable.
    rows = _at(
        seeded_rows("truncating_max_tokens", cut_at=(8, 15)),
        (8, 15),
        reasoning_tokens=900,
    )
    f = _one(_run(rows), "R-01")
    assert f.cause == "thinking_used_budget" and f.counts["p90"] == 200


def test_r01_limit_below_need_when_uncut_replies_are_larger():
    rows = [replace(r, completion_tokens=400) for r in seeded_rows("clean")]
    rows = _at(rows, (8, 15), finish_reason="length", max_tokens_req=256, completion_tokens=256)
    f = _one(_run(rows), "R-01")
    assert f.cause == "limit_below_need" and f.counts["p90"] == 400 and f.counts["m"] == 256


def test_r02_window_cut_is_a_finding_from_one_turn():  # spec 11.3: >= 1, guard none
    rows = _at(seeded_rows("clean", turns=10), (10,), finish_reason="length")
    ts = _turns(rows, max_model_len=9_410)
    f = _one(response_findings(ts, lint={}), "R-02")
    assert f.turns == (10,) and "info" not in f.counts and f.cause == "window_filled"
    assert f.saving.tokens == ts[9].completion_tokens == 200 and f.confidence == "high"
    assert not _of(response_findings(ts, lint={}), "R-01")
    assert bad_turns(ts)["R-02"] == (10,)
    rows = _at(seeded_rows("clean", turns=10), (9, 10), finish_reason="length")
    ts = _turns(rows, max_model_len=9_410)
    f = _one(response_findings(ts, lint={}), "R-02")
    assert f.cause == "window_filled" and "info" not in f.counts


# -- R-03 -------------------------------------------------------------------------


def test_r03_reported_once_as_info_recommendation_from_two():
    rows = _at(seeded_rows("clean"), (6,), finish_reason="runaway_think")
    f = _one(_run(rows), "R-03")
    assert f.counts.get("info") == 1 and f.confidence == "low" and f.saving.tokens == 0
    rows = _at(seeded_rows("clean"), (6, 12), finish_reason="runaway_repeat")
    f = _one(_run(rows), "R-03")
    assert "info" not in f.counts and f.turns == (6, 12) and f.counts["repeat"] == 2
    assert bad_turns(_turns(rows))["R-03"] == (6, 12)


# -- R-04 -------------------------------------------------------------------------


def test_r04_cause_order_follows_the_lint_bits():
    rows = seeded_rows("bad_schema")
    bad = tool_hash(3)
    for flags, cause in (
        (LINT_NO_TYPE | LINT_SHORT_DESC, "schema_untyped"),
        (LINT_NESTED_NO_REQUIRED | LINT_SHORT_DESC, "nested_no_required"),
        (LINT_SHORT_DESC, "short_description"),
        (0, "none_found"),
    ):
        assert _one(_run(rows, {bad: flags}), "R-04").cause == cause


def test_r04_similar_tools_when_the_family_is_large():
    rows = seeded_rows("bad_schema")
    f = _one(_run(rows, {}, family_sizes={"fn": 24}), "R-04")
    assert f.cause == "similar_tools"


def test_r04_share_threshold_and_a12():
    # Two bad of six calls: 33 % >= 20 %, two bad turns: a finding.
    rows = seeded_rows("bad_schema", bad_calls_at=(7, 9), ok_calls_at=(3, 5, 12, 14))
    assert _one(_run(rows), "R-04").counts["bad"] == 2
    # One bad call is never a recommendation (A-12), whatever its share.
    rows = seeded_rows("bad_schema", bad_calls_at=(7,), ok_calls_at=(3,))
    assert not _of(_run(rows), "R-04")
    # Two bad of twelve calls: under 20 % and under 3.
    ok = (3, 5, 10, 12, 14, 16, 18, 20, 22, 24)
    rows = seeded_rows("bad_schema", bad_calls_at=(7, 9), ok_calls_at=ok)
    assert not _of(_run(rows), "R-04")


def test_r04_reads_old_two_element_verdicts_too():
    """I4: rows stored before the call index ([hash8, code]) give the same R-04."""
    rows = seeded_rows("bad_schema")
    old = []
    for r in rows:
        bad = json.loads(r.calls_bad) if r.calls_bad else None
        if bad:
            r = replace(r, calls_bad=json.dumps([e[1:] for e in bad]))
        old.append(r)
    assert any(r.calls_bad not in (None, "[]") for r in old)
    new_f = _one(_run(rows), "R-04")
    old_f = _one(_run(old), "R-04")
    assert (old_f.turns, old_f.counts, old_f.tool) == (new_f.turns, new_f.counts, new_f.tool)


def test_r04_relayed_rows_without_verdicts_are_not_judged():
    rows = [replace(r, calls_bad=None) for r in seeded_rows("bad_schema")]
    assert not _of(_run(rows), "R-04")


# -- R-05 -------------------------------------------------------------------------


def test_r05_no_result_between_calls_is_a_harness_cause():
    rows = seeded_rows("loop")
    rows = [replace(r, lens_repeat_results=0) for r in rows]
    # The client never ran the tool: no tool message arrived after 6, 7, 8.
    out = []
    for r in rows:
        n = int(r.id.rsplit("-", 1)[1])
        if n in (7, 8, 9):
            msgs = {**r.shape["msgs"], "tool": 5}
            r = replace(r, shape={**r.shape, "msgs": msgs})
        out.append(r)
    f = _one(_run(out), "R-05")
    assert f.cause == "harness_no_result" and f.turns == (6, 7, 8, 9)


def test_r05_long_identical_results_have_no_named_cause():
    f = _one(_run(seeded_rows("loop", result_tokens=4_000)), "R-05")
    assert f.cause == "none_found"


# -- R-06 -------------------------------------------------------------------------


def _failing(rows, calls_at, *, tokens=8, same_args=True, tool=2):
    """``tool`` is called on ``calls_at`` and every result comes back failed."""
    h = tool_hash(tool)
    out = []
    for r in rows:
        n = int(r.id.rsplit("-", 1)[1])
        if n in calls_at:
            a = args_hash(7 if same_args else n)
            r = replace(
                r,
                finish_reason="tool_calls",
                reply_chars=0,
                calls_n=1,
                tools_out=(("fn", h, a),),
                calls_bad="[]",
            )
        if n - 1 in calls_at:
            r = replace(r, tools_in=json.dumps([["fn", tokens, True]]))
        out.append(r)
    return out


def test_r06_retries_after_errors():
    rows = _failing(seeded_rows("clean"), (10, 11, 12))
    f = _one(_run(rows), "R-06")
    assert f.turns == (10, 11, 12) and f.counts["failed"] == 3
    assert f.cause == "error_too_short" and f.confidence == "medium"
    assert bad_turns(_turns(rows))["R-06"] == (11, 12)


def test_r06_same_args_cause_and_threshold():
    rows = _failing(seeded_rows("clean"), (10, 11, 12), tokens=200)
    assert _one(_run(rows), "R-06").cause == "same_args_retry"
    rows = _failing(seeded_rows("clean"), (10, 11, 12), tokens=200, same_args=False)
    assert _one(_run(rows), "R-06").cause == "none_found"
    # One run of two: under the threshold (≥ 2 runs or ≥ 3 failed).
    rows = _failing(seeded_rows("clean"), (10, 11))
    assert not _of(_run(rows), "R-06")
    # Two runs of two: a finding.
    rows = _failing(seeded_rows("clean"), (4, 5, 14, 15))
    assert _one(_run(rows), "R-06").counts["runs"] == 2


def test_r06_parallel_failed_calls_in_one_turn_are_one_bad_turn():  # A-12
    h = tool_hash(2)
    rows = seeded_rows("clean")
    rows = _at(
        rows,
        (10,),
        finish_reason="tool_calls",
        reply_chars=0,
        calls_n=3,
        tools_out=tuple(("fn", h, args_hash(i)) for i in range(3)),
        calls_bad="[]",
    )
    rows = _at(rows, (11,), tools_in=json.dumps([["fn", 8, True]] * 3))
    assert not _of(_run(rows), "R-06")
    assert bad_turns(_turns(rows))["R-06"] == ()


# -- R-07 / R-08 ------------------------------------------------------------------


def test_r07_empty_reply_after_tool_result():
    rows = seeded_rows("empty_reply_template")
    assert _one(_run(rows), "R-07").cause == "template_swallowed"
    rows = seeded_rows("empty_reply_template", reasoning=0)
    rows = _at(rows, (5, 11, 17), tools_in=json.dumps([["fn", 50, False]]))
    f = _one(_run(rows), "R-07")
    assert f.cause == "no_after_tool_instruction" and f.turns == (5, 11, 17)
    assert bad_turns(_turns(rows))["R-07"] == (5, 11, 17)


def test_r07_one_empty_reply_is_not_a_finding():
    assert not _of(_run(seeded_rows("empty_reply_template", empty_at=(5,))), "R-07")


def test_r08_long_thinking_short_answer_is_info_never_bad():
    rows = _at(
        seeded_rows("clean"),
        (4, 8, 12),
        reasoning_tokens=3_000,
        completion_tokens=3_020,
        calls_n=0,
        tools_out=(),
    )
    f = _one(_run(rows), "R-08")
    assert f.cause == "thinking_for_short_step" and f.counts["info"] == 1
    assert f.confidence == "low" and f.saving.tokens == 0
    assert "R-08" not in bad_turns(_turns(rows))
    two = _at(seeded_rows("clean"), (4, 8), reasoning_tokens=3_000, completion_tokens=3_020)
    assert not _of(_run(two), "R-08")
