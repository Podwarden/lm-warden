"""Prompt Lens scores: formulas, minimums, confidence, ranges (analysis Task 2.1)."""

from dataclasses import replace

import pytest

from app.lens.analyse import AnalyseContext, analyse_session
from app.lens.scoring import (
    SCORE_VERSION,
    WEIGHTS,
    Score,
    Terms,
    bootstrap_range,
    overall,
    score_layer,
    score_session,
    terms_of,
)
from tests.unit.lens.seed import seeded_rows, seeded_toolsets


def analyse(rows, fault="clean", **kw):
    ctx = AnalyseContext(
        toolsets=seeded_toolsets(fault),
        recent_calls={},
        prefill_rate=None,
        max_model_len=None,
        route_tool_limit=None,
        **kw,
    )
    return analyse_session(rows, own=True, ctx=ctx)


def sc(value, layer="E", conf="high"):
    if value is None:
        return Score(layer, False, None, None, None, None, 0, "x")
    return Score(layer, True, value, value, value, conf, 24, None)


A = analyse(seeded_rows("clean"))


def test_constants():
    assert SCORE_VERSION == 1
    assert WEIGHTS == {"E": 0.45, "C": 0.15, "R": 0.40}


def test_c_formula():
    t = Terms("C", total=600_000, base=0, by_rule={"C-01": 60_000, "C-07": 15_000})
    # 24 clean requests offer tools: the unused-tools term is allowed.
    s = score_layer("C", t, A, seed="s")
    assert s.value == round(100 * (1 - 2 * 75_000 / 600_000))
    assert s.inputs["waste"] == 75_000 and s.inputs["prompt"] == 600_000


def test_c_unused_tools_term_needs_five_tool_turns():
    few = analyse(seeded_rows("clean", turns=4))
    t = Terms("C", total=40_000, base=0, by_rule={"C-01": 10_000, "C-02": 2_000})
    s = score_layer("C", t, few, seed="s")
    assert s.value == round(100 * (1 - 2 * 2_000 / 40_000))
    assert "C-01" in s.excluded_rules


def test_e_excludes_server_causes():  # A-11
    rows = [
        replace(r, lens_cause="server_evicted", lens_lost_tokens=r.lens_cf_tokens)
        if r.lens_cf_tokens
        else r
        for r in seeded_rows("clean")
    ]
    a = analyse(rows)
    s = score_layer("E", terms_of(a)["E"], a, seed="s")
    assert (s.scored, s.value) == (True, 100)


def test_e_counts_a_fixable_cause():
    a = analyse(seeded_rows("timestamp_line"), "timestamp_line")
    s = score_layer("E", terms_of(a)["E"], a, seed="s")
    assert s.scored and s.value is not None and s.value < 20
    assert s.inputs["lost"] > 0


def test_below_minimum_is_not_scored_with_reason():  # A-7
    a = analyse(seeded_rows("clean", turns=3))
    s = score_layer("E", terms_of(a)["E"], a, seed="s")
    assert (s.scored, s.value, s.reason) == (False, None, "2 of 3 continuing turns")
    assert "value" not in s.to_json() and "lo" not in s.to_json()


def test_e_needs_reusable_tokens():
    a = analyse(seeded_rows("clean", turns=5, base_prompt=300, growth=100))
    s = score_layer("E", terms_of(a)["E"], a, seed="s")
    assert not s.scored and s.reason is not None and "4,096" in s.reason


def test_r_below_minimum():
    a = analyse(seeded_rows("clean", turns=4))
    s = score_layer("R", terms_of(a)["R"], a, seed="s")
    assert (s.scored, s.reason) == (False, "4 of 5 assistant turns")


def test_r_formula_blends_turns_and_tokens():
    a = analyse(seeded_rows("empty_reply_template"), "empty_reply_template")
    t = terms_of(a)["R"]
    s = score_layer("R", t, a, seed="s")
    bad = len(a.bad["R-07"])
    assert bad == 3 and s.inputs["bad"] == 3
    # An empty reply wastes no tokens of its own: only the turn share counts.
    out = s.inputs["waste_out"]
    expect = 100 * (1 - 0.5 * bad / s.inputs["turns"] - 0.5 * out / s.inputs["completion"])
    assert s.value == round(expect)


def test_r_a_turn_bad_under_two_rules_counts_once():
    a = analyse(seeded_rows("clean"))
    a.bad.update({"R-01": (7, 8), "R-07": (8, 9)})
    s = score_layer("R", terms_of(a)["R"], a, seed="s")
    assert s.inputs["bad"] == 3


def test_overall_weights_and_redistribution():
    assert overall({"E": sc(48), "C": sc(71, "C"), "R": sc(74, "R")}).value == round(
        0.45 * 48 + 0.15 * 71 + 0.40 * 74
    )
    o = overall({"E": sc(None), "C": sc(71, "C"), "R": sc(74, "R")})
    assert o.value == round((0.15 * 71 + 0.40 * 74) / 0.55) and o.inputs["from"] == ["C", "R"]


def test_overall_needs_two_layers():
    o = overall({"E": sc(None), "C": sc(None, "C"), "R": sc(74, "R")})
    assert (o.scored, o.reason) == (False, "1 of 2 layers scored")


def test_overall_confidence_is_the_lowest():
    o = overall({"E": sc(50, conf="high"), "C": sc(60, "C", "low"), "R": sc(70, "R", "medium")})
    assert o.confidence == "low"


def test_confidence_labels_and_predicted_caps_e_at_medium():
    a = analyse(seeded_rows("clean", turns=24))
    assert score_layer("E", terms_of(a)["E"], a, seed="s").confidence == "high"
    short = analyse(seeded_rows("clean", turns=10))
    assert score_layer("E", terms_of(short)["E"], short, seed="s").confidence == "medium"
    tiny = analyse(seeded_rows("clean", turns=6))
    assert score_layer("E", terms_of(tiny)["E"], tiny, seed="s").confidence == "low"
    # A relayed turn without usage: the cache figure is predicted.
    rows = [replace(r, relayed=True, upstream="upstream.example") for r in seeded_rows("clean")]
    rows[5] = replace(rows[5], cached_tokens=None)
    pred = analyse(rows)
    assert score_layer("E", terms_of(pred)["E"], pred, seed="s").confidence == "medium"


def test_range_is_stable_for_the_same_seed():
    a = analyse(seeded_rows("timestamp_line", every=3), "timestamp_line")
    s1 = score_layer("E", terms_of(a)["E"], a, seed="ref-1")
    s2 = score_layer("E", terms_of(a)["E"], a, seed="ref-1")
    assert (s1.lo, s1.hi) == (s2.lo, s2.hi)
    assert s1.lo <= s1.value <= s1.hi
    assert bootstrap_range([(1.0, 10.0)] * 5, lambda s: 50.0, "x") == (50, 50)
    assert bootstrap_range([], lambda s: 50.0, "x") == (0, 0)


def test_exclude_removes_a_rules_term():  # used by won't fix in S3
    a = analyse(seeded_rows("timestamp_line"), "timestamp_line")
    t = terms_of(a)["E"]
    assert score_layer("E", t, a, seed="s").value < 100
    s = score_layer("E", t, a, seed="s", exclude=frozenset({"E-01"}))
    assert s.value == 100 and s.excluded_rules == ("E-01",)
    r = analyse(seeded_rows("empty_reply_template"), "empty_reply_template")
    s = score_layer("R", terms_of(r)["R"], r, seed="s", exclude=frozenset({"R-07"}))
    assert s.value == 100


def test_lineage_gap_lowers_coverage_not_value():  # Review Focus 1
    rows = seeded_rows("timestamp_line", every=2)
    full = analyse(rows, "timestamp_line")
    # Turns 14-24 lost their parent in memory: lens fields null, "not attributed".
    gap = [
        replace(
            r,
            lens_cause=None,
            lens_lost_tokens=None,
            lens_cf_tokens=None,
            cobs2_edge="new",
            cobs2_parent_req=None,
        )
        if int(r.id.rsplit("-", 1)[1]) >= 14
        else r
        for r in rows
    ]
    g = analyse(gap, "timestamp_line")
    sf = score_layer("E", terms_of(full)["E"], full, seed="s")
    sg = score_layer("E", terms_of(g)["E"], g, seed="s")
    assert g.coverage["E"] < full.coverage["E"]
    assert abs(sg.value - sf.value) <= 5  # the same shape of loss, fewer turns
    assert sg.confidence != "high" and sf.confidence in ("high", "medium")


def test_clean_session_scores_100_everywhere():
    s = score_session(A, seed="s")
    assert {k: v.value for k, v in s.items()} == {"E": 100, "C": 100, "R": 100, "overall": 100}


@pytest.mark.parametrize("k", ["E", "C", "R"])
def test_to_json_is_numbers_only(k):
    j = score_session(A, seed="s")[k].to_json()
    assert j["scored"] is True and isinstance(j["value"], int)
