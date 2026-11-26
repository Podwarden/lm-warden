"""Prompt Lens session analysis: three layers over one session's rows (Task 1.7)."""

import json
from dataclasses import replace

from app.lens.analyse import AnalyseContext, analyse_session, findings_for_request
from app.lens.catalogue import RULES
from tests.unit.lens.seed import seeded_rows, seeded_toolsets

CANARY = "CANARY-7f3e9b21-lens"


def _ctx(fault="clean", **kw):
    base = {
        "toolsets": seeded_toolsets(fault),
        "recent_calls": {},
        "prefill_rate": None,
        "max_model_len": None,
        "route_tool_limit": None,
    }
    base.update(kw)
    return AnalyseContext(**base)


def analyse(rows, fault="clean", **kw):
    return analyse_session(rows, own=True, ctx=_ctx(fault, **kw))


def rules(a):
    return {f.rule for f in a.findings}


def of(findings, rule):
    return [f for f in findings if f.rule == rule]


def one(findings, rule):
    hits = of(findings, rule)
    assert len(hits) == 1, [(f.rule, f.cause, f.turns) for f in findings]
    return hits[0]


def rows_with(*, truncations, **kw):
    cut_at = (8, 15, 20, 22)[:truncations]
    return seeded_rows("truncating_max_tokens", cut_at=cut_at, **kw)


# -- the brief's tests -----------------------------------------------------------


def test_bad_schema_gives_r04_with_schema_cause():
    a = analyse(seeded_rows("bad_schema"), "bad_schema")
    f = one(a.findings, "R-04")
    assert f.cause == "schema_undocumented" and f.turns == (7, 9, 12, 14) and f.counts["bad"] == 4
    # Six calls of that tool in all (two ok, four bad), read from tools_out.
    assert f.counts["calls"] == 6


def test_single_truncation_is_not_a_finding():  # A-12
    assert "R-01" not in rules(analyse(rows_with(truncations=1)))
    f = one(analyse(rows_with(truncations=3)).findings, "R-01")
    assert f.turns == (8, 15, 20) and f.counts["m"] == 1_024
    # 3 of 24 replies cut at 1,024: the p90 need (cuts at +1) is above the cap.
    assert f.cause == "limit_below_need" and f.counts["p90"] == 1_025


def test_max_tokens_probe_under_16_ignored():
    a = analyse(rows_with(truncations=3, client_max_tokens=16))
    assert "R-01" not in rules(a)
    assert "R-01" not in a.bad or not a.bad["R-01"]


def test_loop_needs_identical_results_in_between():
    a = analyse(seeded_rows("loop"))
    f = one(a.findings, "R-05")
    assert f.turns == (6, 7, 8, 9) and f.counts["calls"] == 4
    assert f.cause == "result_not_final"
    assert a.bad["R-05"] == (7, 8, 9)
    # Polling: the same call, but every result differs (nothing repeated).
    polling = [replace(r, lens_repeat_results=0) for r in seeded_rows("loop")]
    assert "R-05" not in rules(analyse(polling))
    # Re-reads with other calls in between are not a loop either.
    assert "R-05" not in rules(analyse(seeded_rows("reread_file")))


def test_leaked_markup_is_admin_cause_not_prompt():
    a = analyse(seeded_rows("bad_schema", bad_code="leaked_markup"), "bad_schema")
    f = one(a.findings, "R-04")
    assert f.cause == "parser_mismatch_admin"
    assert not a.bad.get("R-04")
    # An engine cause never ranks as a prompt fix (A-11).
    assert f.saving.tokens == 0 and f.saving.prefill_s is None


def test_queue_dominated_ttft_gives_no_prompt_advice():  # A-11
    rows = [replace(r, ttft_s=5.0, queued_s=4.0) for r in seeded_rows("clean", growth=1_500)]
    a = analyse(rows, prefill_rate=10_000.0)
    f = one(a.findings, "R-10")
    assert f.cause == "server_load" and f.saving.tokens == 0
    assert not any(a.bad.values())


def test_recompute_dominated_ttft_points_at_efficiency():
    rows = [
        replace(r, ttft_s=3.0, queued_s=0.1, lens_lost_tokens=20_000, lens_cause="sys_changed")
        if r.lens_cause is None
        else r
        for r in seeded_rows("clean", growth=1_500)
    ]
    f = one(analyse(rows, prefill_rate=10_000.0).findings, "R-10")
    assert f.cause == "recompute_dominates" and f.saving.prefill_s


def _guarded(rows, at, rule="fake_secret_rule"):
    out = []
    for r in rows:
        n = int(r.id.rsplit("-", 1)[1])
        if n >= at:
            g = {"m": "sanitize", "b": 0, "f": [[rule, 1, 1 if n == at else 0]], "r": 0}
            r = replace(r, guard=json.dumps(g))
        out.append(r)
    return out


def test_r09_guard_secret_in_tool_results():
    rows = _guarded(seeded_rows("clean"), 5)
    a = analyse(rows, guard_where={"req-fake-0005": {"tool_result": 1}})
    f = one(a.findings, "R-09")
    assert f.cause == "secret_in_tool_results" and f.counts["values"] == 1
    assert f.confidence == "high" and 5 in f.turns
    # Without the guard log's role, the turn's new content (tool results only)
    # still points at the results, at medium confidence.
    f = one(analyse(rows).findings, "R-09")
    assert f.cause == "secret_in_tool_results" and f.confidence == "medium"
    # A match in the system prompt (logged role) names the system prompt.
    first = _guarded(seeded_rows("clean"), 1)
    f = one(analyse(first, guard_where={"req-fake-0001": {"system": 1}}).findings, "R-09")
    assert f.cause == "secret_in_system"
    # Never a bad turn (not in R, spec 10.2).
    assert "R-09" not in a.bad


def test_findings_for_request_returns_only_that_turns_findings():
    a = analyse(seeded_rows("bad_schema"), "bad_schema")
    got = findings_for_request(a, "req-fake-0009")
    assert "R-04" in {f.rule for f in got}
    assert all(9 in f.turns for f in got)
    assert all(f in a.findings for f in got)
    assert findings_for_request(a, "req-fake-0002") == [f for f in a.findings if 2 in f.turns]
    assert findings_for_request(a, "req-nope") == []


# -- markers, coverage, all layers ----------------------------------------------


def test_markers_and_coverage():
    a = analyse(seeded_rows("bad_schema"), "bad_schema")
    assert "R-04" in a.markers[7] and 2 not in a.markers
    assert a.coverage == {"E": 1.0, "C": 1.0, "R": 1.0}
    assert [t.n for t in a.turns] == list(range(1, 25))


def test_relayed_rows_have_no_response_coverage():
    a = analyse(seeded_rows("no_marks_anthropic"))
    assert a.coverage["R"] == 0.0
    assert "E-06" in rules(a)  # the efficiency layer still runs


def test_runs_all_three_layers():
    assert "E-01" in rules(analyse(seeded_rows("timestamp_line")))
    assert "C-01" in rules(analyse(seeded_rows("unused_tools"), "unused_tools"))
    assert "R-07" in rules(analyse(seeded_rows("empty_reply_template")))


def test_clean_session_has_no_response_findings():
    a = analyse(seeded_rows("clean"))
    assert not [f for f in a.findings if RULES[f.rule].layer == "R"]
    assert not any(a.bad.values())


def test_waste_terms_get_the_toolset_entries():
    a = analyse(seeded_rows("unused_tools"), "unused_tools")
    assert a.waste["C-01"] > 0


def test_empty_session():
    a = analyse([])
    assert a.findings == [] and a.markers == {} and a.turns == []
    assert a.coverage == {"E": 0.0, "C": 0.0, "R": 0.0}


def test_findings_are_text_free():
    rows = [replace(r, token_name=CANARY, model=CANARY) for r in seeded_rows("bad_schema")]
    a = analyse(rows, "bad_schema")
    assert CANARY not in json.dumps([f.to_json() for f in a.findings])
