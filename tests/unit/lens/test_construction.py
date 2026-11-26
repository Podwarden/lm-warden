"""Prompt Lens construction checks C-01..C-09 (analysis spec 7.2, 11.2, 10.2)."""

import json
from dataclasses import replace

from app.lens.catalogue import RULES
from app.lens.construction import construction_findings, waste_terms
from app.lens.turns import turns_of
from tests.unit.lens import seed
from tests.unit.lens.seed import seeded_rows, seeded_toolsets


def _run(rows, *, entries=None, recent=None, max_len=None, limit=None, fault="clean", **kw):
    ts = turns_of(rows, own=True)
    return construction_findings(
        ts,
        toolset_entries=seeded_toolsets(fault, **kw) if entries is None else entries,
        recent_calls=recent or {},
        max_model_len=max_len,
        route_tool_limit=limit,
    )


def _of(findings, rule):
    return [f for f in findings if f.rule == rule]


def _one(findings, rule):
    hits = _of(findings, rule)
    assert len(hits) == 1, [f.rule for f in findings]
    return hits[0]


def _dw(rows, **dw):
    """Rows whose shape carries these dead weight chars."""
    return [
        replace(r, shape={**r.shape, "dw": {"dup": 0, "boil": 0, "reason": 0, **dw}}) for r in rows
    ]


def test_clean_session_has_no_construction_finding():
    assert _run(seeded_rows("clean")) == []
    assert waste_terms(turns_of(seeded_rows("clean"), own=True)) == {
        "C-01": 0,
        "C-02": 0,
        "C-07": 0,
    }


# -- C-01 ----------------------------------------------------------------------


def test_c01_unused_tools():
    f = _one(_run(seeded_rows("unused_tools"), fault="unused_tools"), "C-01")
    assert f.counts["unused"] == 7 and f.counts["offered"] == 12
    assert f.counts["rarely"] == 0 and "info" not in f.counts
    # 7 tools x 1,600 chars x 0.25 = 2,800 tokens on each of 24 turns
    assert f.counts["tokens"] == 2_800 and f.counts["turns"] == 24
    assert f.saving.tokens == 2_800 * 24
    assert f.tool is None and f.confidence == "high"
    assert f.turns == tuple(range(1, 25))


def test_c01_needs_ten_turns():
    assert _of(_run(seeded_rows("unused_tools", turns=9), fault="unused_tools"), "C-01") == []
    assert _of(_run(seeded_rows("unused_tools", turns=10), fault="unused_tools"), "C-01")


def test_c01_needs_1000_tokens_per_turn_or_5_percent():
    kw = {"tool_chars": 400}  # 7 x 400 x 0.25 = 700 tokens per turn
    assert _of(
        _run(seeded_rows("unused_tools", **kw), fault="unused_tools", **kw), "C-01"
    )  # 700 of 7,000+ >= 5 %
    kw = {"tool_chars": 400, "base_prompt": 40_000}
    assert _of(_run(seeded_rows("unused_tools", **kw), fault="unused_tools", **kw), "C-01") == []


def test_c01_tool_called_in_a_recent_session_is_rarely_used_info():
    rows = seeded_rows("unused_tools")
    # all 7 never-called tools were called in earlier sessions: no C-01 fault, one info
    recent = {seed.tool_hash(i): 2 for i in range(5, 12)}
    f = _one(_run(rows, recent=recent, fault="unused_tools"), "C-01")
    assert f.counts["info"] == 1 and f.counts["unused"] == 0 and f.counts["rarely"] == 7
    assert f.saving.tokens == 0 and f.confidence == "low"
    # two of them stay unused for the key (800 tokens, 7 %): a C-01 fault, 5 rarely used
    recent = {seed.tool_hash(i): 2 for i in range(7, 12)}
    f = _one(_run(rows, recent=recent, fault="unused_tools"), "C-01")
    assert f.counts["unused"] == 2 and f.counts["rarely"] == 5 and "info" not in f.counts
    assert f.counts["tokens"] == 800 and f.saving.tokens == 800 * 24


def test_c01_rarely_used_is_not_waste():
    rows = seeded_rows("unused_tools")
    ts = turns_of(rows, own=True)
    ents = seeded_toolsets("unused_tools")
    recent = {seed.tool_hash(i): 1 for i in range(5, 12)}
    assert waste_terms(ts, toolset_entries=ents)["C-01"] == 2_800 * 24
    assert waste_terms(ts, toolset_entries=ents, recent_calls=recent)["C-01"] == 0


def test_c01_without_entries_for_the_toolset_is_silent():
    assert _of(_run(seeded_rows("unused_tools"), entries={}), "C-01") == []


# -- C-02 ----------------------------------------------------------------------


def test_c02_reread_file():
    f = _one(_run(seeded_rows("reread_file")), "C-02")
    assert f.counts["repeats"] == 2 and f.counts["tokens"] == 900
    assert f.turns == (10, 15)
    assert f.tool == seed.tool_hash(0)
    # 900 x turns 10..24 (15) + 900 x turns 15..24 (10)
    assert f.saving.tokens == 900 * 15 + 900 * 10


def test_c02_needs_two_repeats():
    assert _of(_run(seeded_rows("reread_file", read_at=(4, 9))), "C-02") == []


def test_c02_ignores_results_under_500_tokens():
    assert _of(_run(seeded_rows("reread_file", result_tokens=400)), "C-02") == []


def test_c02_duplicated_messages_count_too():
    rows = seeded_rows("clean")
    # a whole message duplicated: 3,000 chars = 750 tokens from turn 5 on, twice more
    rows = [
        replace(r, shape={**r.shape, "dw": {"dup": c, "boil": 0, "reason": 0}})
        for r, c in zip(rows, [0] * 4 + [3_000] * 8 + [6_000] * 12, strict=True)
    ]
    f = _one(_run(rows), "C-02")
    assert f.counts["repeats"] == 2 and f.counts["tokens"] == 750


def test_waste_terms_c02_is_dup_plus_repeats():
    rows = _dw(seeded_rows("reread_file"), dup=400)  # 100 tokens a turn
    got = waste_terms(turns_of(rows, own=True))
    assert got["C-02"] == 100 * 24 + 900 * 5 + 1_800 * 10


# -- C-03 ----------------------------------------------------------------------

HEAD = {"base_prompt": 30_000, "growth": 100, "sys_chars": 100_000}  # 25,000 + 900 tokens of head


def test_c03_head_share():
    f = _one(_run(seeded_rows("clean", **HEAD)), "C-03")
    assert f.counts["share"] >= 70 and f.counts["tokens"] >= 8_000
    assert f.counts["share"] <= 100


def test_c03_below_8000_tokens_is_silent():
    rows = seeded_rows("clean", base_prompt=7_000, growth=0, sys_chars=24_000)
    assert _of(_run(rows), "C-03") == []


def test_c03_below_70_percent_is_silent():
    assert (
        _of(_run(seeded_rows("clean", base_prompt=60_000, growth=0, sys_chars=100_000)), "C-03")
        == []
    )


def test_c03_info_when_head_reused_on_90_percent_of_turns():
    # the clean seed reuses the whole previous prompt on every continuing turn
    f = _one(_run(seeded_rows("clean", **HEAD)), "C-03")
    assert f.counts["info"] == 1 and f.counts["reused"] == 100
    cold = [
        replace(r, cached_tokens=0) if i % 2 else r
        for i, r in enumerate(seeded_rows("clean", **HEAD))
    ]
    f = _one(_run(cold), "C-03")
    assert "info" not in f.counts and f.counts["reused"] < 90


# -- C-04 ----------------------------------------------------------------------


def test_c04_tool_results_dominate():
    f = _one(_run(seeded_rows("clean", growth=9_000)), "C-04")
    assert f.counts["share"] >= 70 and f.counts["largest"] >= 8_000
    assert f.confidence == "medium"
    # excess over 8,000 tokens per result, for every turn it stays in history
    assert f.saving.tokens > 0


def test_c04_needs_a_big_single_result():
    assert _of(_run(seeded_rows("clean", growth=3_000)), "C-04") == []


def test_c04_small_results_share_is_not_enough():
    assert _of(_run(seeded_rows("clean", growth=400)), "C-04") == []


# -- C-05 ----------------------------------------------------------------------


def test_c05_projects_the_window():
    # 6,000 + 400 per turn; 80 % of 20,000 is 16,000: reached at turn 26, within another 24
    f = _one(_run(seeded_rows("clean"), max_len=20_000), "C-05")
    assert f.counts["growth"] == 400 and f.counts["p90"] == 400
    assert f.counts["fill_turn"] == 36  # 100 %: (20,000 - 15,200) / 400 = 12 more turns after 24
    assert f.saving.tokens == 0


def test_c05_no_projection_under_five_turns():
    assert _of(_run(seeded_rows("clean", turns=4, growth=3_000), max_len=20_000), "C-05") == []


def test_c05_needs_a_window_and_a_close_horizon():
    assert _of(_run(seeded_rows("clean")), "C-05") == []
    assert _of(_run(seeded_rows("clean"), max_len=1_000_000), "C-05") == []


def test_c05_compaction_before_half_the_window():
    rows = seeded_rows("clean", growth=100)
    rows[11] = replace(rows[11], cobs2_edge="compacted_from", prompt_tokens=2_000)
    f = _one(_run(rows, max_len=50_000), "C-05")
    assert f.counts["compact_at"] == 14  # the prompt before the compaction: 7,000 of 50,000
    assert f.turns == (12,)


# -- C-06 ----------------------------------------------------------------------


def test_c06_volatile_before_stable_carries_no_saving():
    f = _one(_run(seeded_rows("timestamp_line")), "C-06")
    assert f.saving.tokens == 0 and f.counts["line"] == 2 and f.counts["turns"] == 23
    assert _of(_run(seeded_rows("clean")), "C-06") == []


# -- C-07 ----------------------------------------------------------------------


def test_c07_boilerplate_over_ten_percent():
    rows = _dw(seeded_rows("clean", base_prompt=10_000, growth=0), boil=6_000)  # 1,500 of 10,000
    f = _one(_run(rows), "C-07")
    assert f.counts["share"] == 15 and f.counts["tokens"] == 1_500
    assert f.saving.tokens == 1_500 * 24
    assert waste_terms(turns_of(rows, own=True))["C-07"] == 750 * 24  # boil x 0.5


def test_c07_under_ten_percent_is_silent():
    rows = _dw(seeded_rows("clean", base_prompt=10_000, growth=0), boil=3_000)
    assert _of(_run(rows), "C-07") == []


def test_waste_uses_each_turns_own_tokens_per_char():
    rows = _dw(seeded_rows("clean", base_prompt=10_000, growth=0), boil=4_000)
    rows = [
        replace(r, shape={**r.shape, "chars": r.shape["chars"] * 2}) if i < 12 else r
        for i, r in enumerate(rows)
    ]
    # 12 turns at 0.125 tokens/char (500 boil tokens) and 12 at 0.25 (1,000)
    assert waste_terms(turns_of(rows, own=True))["C-07"] == (250 + 500) * 12


# -- C-08 ----------------------------------------------------------------------


def _ents(sizes):
    return {seed.TOOLSET: [(seed.tool_hash(i), "fn", c) for i, c in enumerate(sizes)]}


def test_c08_schema_outlier():
    ents = _ents([1_000] * 5 + [16_000])  # 4,000 tokens, 16 x the median
    f = _one(_run(seeded_rows("clean", tools=6), entries=ents), "C-08")
    assert f.tool == seed.tool_hash(5)
    assert f.counts["tokens"] == 4_000 and f.counts["x"] == 16 and f.counts["used"] == 0
    assert f.saving.tokens == (16_000 - 1_000) // 4 * 24


def test_c08_marks_how_often_the_tool_was_used():
    ents = _ents([1_000] * 3 + [16_000])
    rows = seeded_rows("clean", tools=4, called_tools=4)
    f = _one(_run(rows, entries=ents), "C-08")
    assert f.tool == seed.tool_hash(3) and f.counts["used"] == 3  # replies 7, 15, 23 call it


def test_c08_thresholds():
    assert (
        _of(_run(seeded_rows("clean", tools=6), entries=_ents([1_000] * 5 + [2_900])), "C-08") == []
    )  # < 3x
    assert (
        _of(_run(seeded_rows("clean", tools=6), entries=_ents([100] * 5 + [7_000])), "C-08") == []
    )  # < 2,000 tokens


# -- C-09 ----------------------------------------------------------------------


def test_c09_only_with_a_modelled_limit():
    rows = seeded_rows("clean", tools=100, called_tools=3, tool_chars=40)
    f = _one(_run(rows, limit=128, entries={}), "C-09")
    assert f.counts == {"tools": 100, "limit": 128}
    assert f.saving.tokens == 0
    assert _of(_run(rows, limit=None, entries={}), "C-09") == []
    assert (
        _of(_run(seeded_rows("clean", tools=99, called_tools=3), limit=128, entries={}), "C-09")
        == []
    )


# -- hygiene -------------------------------------------------------------------


def test_findings_are_catalogue_rules_in_order_and_numbers_only():
    rows = seeded_rows("unused_tools", **{**HEAD, "growth": 9_000})
    out = _run(rows, fault="unused_tools", max_len=200_000, limit=128)
    ids = [f.rule for f in out]
    order = list(RULES)
    assert ids == sorted(ids, key=order.index)
    for f in out:
        assert f.rule in RULES and f.rule.startswith("C-")
        json.dumps(f.to_json())
        assert all(isinstance(v, int) for v in f.counts.values())


def test_a_turn_without_stored_fields_does_not_raise():
    rows = [replace(r, shape=None, toolset=None) for r in seeded_rows("unused_tools")]
    assert _run(rows, fault="unused_tools") == []
    assert waste_terms(turns_of(rows, own=True)) == {"C-01": 0, "C-02": 0, "C-07": 0}


def test_two_element_and_three_element_tools_out_are_both_read():
    rows = seeded_rows("unused_tools")
    rows = [replace(r, tools_out=tuple(e[:2] for e in r.tools_out)) for r in rows]
    assert _one(_run(rows, fault="unused_tools"), "C-01").counts["unused"] == 7
