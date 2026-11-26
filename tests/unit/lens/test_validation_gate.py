"""The seeded validation gate (analysis spec 20, former L17; plan Task 2.6).

One fake session per fault, built from codes and counts only, runs through the
whole analysis: ``analyse_session``, ``terms_of``, ``score_layer``, ``overall``,
``top_fixes``. Each fault must produce its rule and cause; the clean control
must fire nothing and score 100 on every scored layer; the top 3 must follow the
seeded savings; one bad turn of a kind is never a recommendation (A-12).
"""

from __future__ import annotations

import pytest

from app.lens.analyse import AnalyseContext, analyse_session
from app.lens.ranking import top_fixes
from app.lens.scoring import overall, score_layer, score_session, terms_of
from tests.unit.lens import seed
from tests.unit.lens.seed import seeded_rows, seeded_toolsets

#: fault -> (rule, cause). A cause of None is the finding's own: C-01 and C-02
#: name no cause. The plan's table lists E-06 with None as well, but E-06's finding
#: carries the layer 1 cause ``provider_no_marks`` (app/lens/efficiency.py), so the
#: gate asserts that cause for it rather than skipping the check.
EXPECTED = {
    "timestamp_line": ("E-01", "sys_changed"),
    "tools_reordered": ("E-02", "tools_reordered"),
    "unused_tools": ("C-01", None),
    "reread_file": ("C-02", None),
    "bad_schema": ("R-04", "schema_undocumented"),
    "loop": ("R-05", "result_not_final"),
    "empty_reply_template": ("R-07", "template_swallowed"),
    "truncating_max_tokens": ("R-01", "limit_below_need"),
    "no_marks_anthropic": ("E-06", "provider_no_marks"),
}


def run(rows, toolsets=None, **ctx):
    base = {
        "recent_calls": {},
        "prefill_rate": None,
        "max_model_len": None,
        "route_tool_limit": None,
    }
    a = analyse_session(
        rows,
        own=True,
        ctx=AnalyseContext(toolsets=toolsets or seeded_toolsets("clean"), **{**base, **ctx}),
    )
    return a


def analyse_fault(fault, **params):
    return run(seeded_rows(fault, **params), seeded_toolsets(fault, **params))


@pytest.mark.parametrize("fault,want", EXPECTED.items())
def test_each_seeded_fault_produces_its_rule_and_cause(fault, want):
    rule, cause = want
    a = analyse_fault(fault)
    hits = [f for f in a.findings if f.rule == rule]
    assert hits, [(f.rule, f.cause) for f in a.findings]
    assert hits[0].cause == cause
    # ...and it is the fault, not noise: the finding is in the session's top fixes.
    assert rule in {f.rule for f in top_fixes(a.findings, None)}


def test_each_fault_lowers_the_layer_it_belongs_to():
    layer_of = {"E": "E", "C": "C", "R": "R"}
    for fault, (rule, _cause) in EXPECTED.items():
        a = analyse_fault(fault)
        layer = layer_of[rule[0]]
        s = score_layer(layer, terms_of(a)[layer], a, seed=fault)
        if s.scored:
            assert s.value is not None and s.value < 100, (fault, s)


def test_clean_control_fires_nothing_and_scores_100_on_every_scored_layer():
    a = analyse_fault("clean")
    assert a.findings == []
    scores = score_session(a, seed="control")
    scored = [s for s in scores.values() if s.scored]
    assert {s.layer for s in scored} == {"E", "C", "R", "overall"}
    assert all(s.value == 100 for s in scored), {k: s.value for k, s in scores.items()}
    assert top_fixes(a.findings, None) == []
    # A relayed session without reply fields scores what it can: R is not scored.
    relay = score_session(analyse_fault("no_marks_anthropic"), seed="relay")
    assert not relay["R"].scored and relay["R"].reason


def test_top3_order_matches_seeded_savings():
    # Four faults in one session, with savings 67,200 (C-01) > 30,800 (E-02) >
    # 22,500 (C-02) > 20,800 (E-01: turns 9 and 17 lose the 8,800 and
    # 12,000 tokens before them): the top 3 are the first three, in that order.
    over = {"every": 8}
    rows = seeded_rows("unused_tools")
    rows = seed._sys_line(rows, seed._params("timestamp_line", over))
    rows = seed._tools_reordered(rows, seed._params("tools_reordered", {}))
    rows = seed._reread_file(rows, seed._params("reread_file", {}))
    a = run(rows, seeded_toolsets("unused_tools"))
    saved = {f.rule: f.saving.tokens for f in a.findings}
    assert {k: v for k, v in saved.items() if k != "C-06"} == {
        "C-01": 67_200,
        "E-02": 30_800,
        "C-02": 22_500,
        "E-01": 20_800,
    }
    top = top_fixes(a.findings, None)
    assert [f.rule for f in top] == ["C-01", "E-02", "C-02"]
    # Each is also what its fault yields alone.
    for rule, fault, params in (
        ("C-01", "unused_tools", {}),
        ("E-02", "tools_reordered", {}),
        ("C-02", "reread_file", {}),
        ("E-01", "timestamp_line", over),
    ):
        alone = {f.rule: f.saving.tokens for f in analyse_fault(fault, **params).findings}
        assert alone[rule] == saved[rule]


def test_one_bad_turn_each_is_not_a_recommendation():  # A-12
    singles = [
        ("bad_schema", {"bad_calls_at": (7,), "ok_calls_at": ()}),
        ("truncating_max_tokens", {"cut_at": (8,)}),
        ("empty_reply_template", {"empty_at": (5,)}),
        ("timestamp_line", {"every": 23}),  # one changed turn: turn 24
        ("reread_file", {"read_at": (4,)}),
        ("loop", {"loop_at": (6,)}),
    ]
    for fault, params in singles:
        a = analyse_fault(fault, **params)
        fired = [(f.rule, f.turns) for f in a.findings if f.rule[0] in "ECR" and f.rule != "C-06"]
        assert fired == [], (fault, fired)
        assert top_fixes(a.findings, None) == []
    # The one bad turn still counts in R (a measured turn), just never as advice.
    a = analyse_fault("empty_reply_template", empty_at=(5,))
    s = score_layer("R", terms_of(a)["R"], a, seed="s")
    assert s.scored and s.value is not None and s.value < 100


def test_overall_of_a_faulty_session_is_the_weighted_mean():
    a = analyse_fault("timestamp_line")
    layers = {k: score_layer(k, terms_of(a)[k], a, seed="s") for k in ("E", "C", "R")}
    o = overall(layers)
    want = round(sum(w * layers[k].value for k, w in (("E", 0.45), ("C", 0.15), ("R", 0.40))))
    assert o.value == want
