"""Prompt Lens turn model and efficiency rules E-01..E-09 (analysis spec 11.1, 10.2)."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.lens.efficiency import efficiency_findings
from app.lens.probes import SPAN_CLASSES
from app.lens.turns import LensTurn, turns_of
from tests.unit.lens import seed
from tests.unit.lens.seed import seeded_rows, seeded_toolsets


def _f(rows, *, cross_sessions=0, **kw):
    return efficiency_findings(turns_of(rows, own=True), cross_sessions=cross_sessions, **kw)


def _rules(rows, **kw):
    return [f.rule for f in _f(rows, **kw)]


def _one(findings, rule):
    hits = [f for f in findings if f.rule == rule]
    assert len(hits) == 1, [f.rule for f in findings]
    return hits[0]


# -- turn model ----------------------------------------------------------------


def test_turns_are_numbered_from_one_in_start_order():
    rows = seeded_rows("clean", turns=5)
    ts = turns_of(list(reversed(rows)), own=True)
    assert [t.n for t in ts] == [1, 2, 3, 4, 5]
    assert [t.req_id for t in ts] == [r.id for r in rows]
    assert all(isinstance(t, LensTurn) for t in ts)


def test_turn_carries_the_lens_fields():
    rows = seeded_rows("bad_schema")
    t = turns_of(rows, own=True)[6]  # turn 7: a bad call
    assert t.n == 7 and t.lane == seed.LANE and t.edge == "continue"
    assert t.calls_bad == ((0, seed.tool_hash(3), "missing_arg"),)
    assert t.tools_out == (("fn", seed.tool_hash(3), seed.args_hash(2_007)),)
    assert t.tools_lint == (0, 0, 0, 2) and t.tools_in == 4
    assert t.calls_n == 1 and t.finish_reason == "tool_calls"
    assert t.max_tokens_req == 4_096 and t.reasoning_tokens == 0
    assert t.toolset == seed.TOOLSET and t.shape is not None
    assert t.dead_weight == (0, 0, 0)


def test_own_lens_reuse_is_capped_by_own_reusable():
    r = replace(seeded_rows("clean", turns=2)[1], cached_tokens=9_000, reusable_tokens_own=5_000)
    r = replace(r, cache_outcome="hit", cache_outcome_own="partial", reusable_tokens=9_000)
    own = turns_of([r], own=True)[0]
    fleet = turns_of([r], own=False)[0]
    assert own.reused_own == 5_000 and own.outcome == "partial"
    assert fleet.reused_own == 9_000 and fleet.outcome == "hit"


def test_gap_is_per_lane():
    rows = seeded_rows("clean", turns=3)
    rows[1] = replace(rows[1], cobs2_conv="conv-fake-other")
    ts = turns_of(rows, own=True)
    assert ts[0].gap_s is None and ts[1].gap_s is None
    assert ts[2].gap_s == pytest.approx(60.0 - 2.0)


def test_malformed_json_fields_read_as_unknown():
    r = replace(seeded_rows("clean", turns=1)[0], calls_bad="{nope", tools_lint="[1, ")
    t = turns_of([r], own=True)[0]
    assert t.calls_bad is None and t.tools_lint is None


def test_turn_dead_weight_is_tokens_at_the_turns_own_ratio():
    r = seeded_rows("clean", turns=1)[0]
    r = replace(r, shape={**(r.shape or {}), "dw": {"dup": 400, "boil": 800, "reason": 40}})
    assert turns_of([r], own=True)[0].dead_weight == (100, 200, 10)


# -- E-01 ----------------------------------------------------------------------


def test_e01_fires_on_timestamp_line_twice_not_once():  # A-12
    assert [f.rule for f in _f(seeded_rows("timestamp_line"))] == ["E-01"]
    assert _f(seeded_rows("timestamp_line", turns=2)) == []


def test_e01_counts_turns_line_and_lost():
    rows = seeded_rows("timestamp_line", turns=6)
    f = _one(_f(rows), "E-01")
    assert f.turns == (2, 3, 4, 5, 6)
    assert f.cause == "sys_changed" and f.confidence == "high"
    lost = sum(r.lens_lost_tokens or 0 for r in rows[1:])
    assert f.counts["turns"] == 5 and f.counts["line"] == 2 and f.counts["lost"] == lost
    assert f.saving.tokens == lost and f.saving.money is None
    assert f.counts["class"] == SPAN_CLASSES.index("time")  # a code: the class's index
    assert f.to_json()["counts"]["line"] == 2


def test_e01_path_line():
    f = _one(_f(seeded_rows("path_line")), "E-01")
    assert f.counts["line"] == 4 and f.counts["turns"] == 23


def test_e01_across_sessions_with_one_turn_here():
    assert _f(seeded_rows("timestamp_line", turns=2), cross_sessions=1)[0].rule == "E-01"
    f = _f(seeded_rows("timestamp_line", turns=2), cross_sessions=1)[0]
    assert f.counts["sessions"] == 2 and f.confidence == "medium"


def test_e01_across_sessions_counts_per_line():
    """I5: the per-line count from the key's own rows; another line's count
    does not make this line's single turn a finding."""
    rows = seeded_rows("timestamp_line", turns=2)
    assert _f(rows, cross_sessions={2: 1})[0].counts["sessions"] == 2
    assert _f(rows, cross_sessions={7: 3}) == []
    assert _f(rows, cross_sessions={}) == []


def test_e01_needs_1024_lost_tokens():
    assert _f(seeded_rows("timestamp_line", turns=4, lost=300)) == []
    assert _rules(seeded_rows("timestamp_line", turns=5, lost=300)) == ["E-01"]  # 4 x 300


def test_e01_class_other_is_low_confidence():
    f = _one(_f(seeded_rows("timestamp_line", span_class="other")), "E-01")
    assert f.confidence == "low" and "class" not in f.counts  # message without class


def test_e01_ignores_a_non_volatile_line():
    # a number that changes is content, not a timestamp: no E-01
    assert _f(seeded_rows("timestamp_line", span_class="number")) == []


def test_e01_date_line_once_a_day_is_expected():
    midnight = datetime(2026, 11, 23, tzinfo=UTC).timestamp()
    # three turns, a day apart: the date changes once a day, as it should
    rows = seeded_rows(
        "timestamp_line", turns=3, span_class="date", start=midnight - 600, step_s=86_400.0
    )
    assert _f(rows) == []
    # every turn within one day: the "date" line changes more often than days do
    assert _rules(seeded_rows("timestamp_line", turns=4, span_class="date")) == ["E-01"]


def test_e01_one_finding_per_line():
    rows = seeded_rows("timestamp_line", turns=7)
    rows = [replace(r, lens_span_line=9) if i % 2 else r for i, r in enumerate(rows)]
    lines = sorted(f.counts["line"] for f in _f(rows) if f.rule == "E-01")
    assert lines == [2, 9]


def test_prefill_rate_gives_seconds():
    f = _one(_f(seeded_rows("timestamp_line", turns=3), prefill_rate=1_000.0), "E-01")
    assert f.saving.prefill_s == pytest.approx(f.saving.tokens / 1_000.0)


# -- E-02 .. E-09 -------------------------------------------------------------


def test_e02_tools_reordered():
    f = _one(_f(seeded_rows("tools_reordered")), "E-02")
    assert f.turns == (6, 13, 19) and f.cause == "tools_reordered"
    assert f.confidence == "high"


def test_e02_needs_1024_lost():
    assert _f(seeded_rows("tools_reordered", lost=200)) == []


def _with_cause(rows, at, cause, **kw):
    out = []
    for r in rows:
        n = int(r.id.rsplit("-", 1)[1])
        if n in at:
            prev = rows[n - 2].prompt_tokens
            r = replace(
                r,
                lens_cause=cause,
                lens_cf_tokens=prev,
                lens_lost_tokens=prev,
                cached_tokens=0,
                cache_outcome_own="diverged",
                **kw,
            )
        out.append(r)
    return out


def test_e03_tools_changed_needs_two_turns():
    rows = seeded_rows("clean")
    assert _f(_with_cause(rows, {5}, "tools_changed", lens_break="tools")) == []
    f = _one(_f(_with_cause(rows, {5, 9}, "tools_changed", lens_break="tools")), "E-03")
    assert f.turns == (5, 9)


def test_e04_history_rewritten_needs_two_turns_and_ignores_forks():
    rows = seeded_rows("clean")
    two = _with_cause(rows, {5, 9}, "history_rewritten", lens_break="node:3")
    f = _one(_f(two), "E-04")
    assert f.counts["message"] == 3 and f.cause == "history_rewritten"
    # an explicit user edit: the lineage reads a fork, never counted
    forks = _with_cause(rows, {5, 9}, "history_rewritten", lens_break="node:3", cobs2_edge="fork")
    assert _f(forks) == []
    assert _f(_with_cause(rows, {5, 9}, "fork", cobs2_edge="fork")) == []


def test_e05_reasoning_switches():
    rows = seeded_rows("clean")
    assert _f(_with_cause(rows, {6}, "reasoning_dropped", lens_break="node:4")) == []
    rows = _with_cause(rows, {6}, "reasoning_dropped", lens_break="node:4")
    rows = _with_cause(rows, {10}, "reasoning_kept", lens_break="node:8")
    f = _one(_f(rows), "E-05")
    assert f.counts["switches"] == 2 and f.turns == (6, 10)


def test_e06_no_marks_on_anthropic():
    rows = seeded_rows("no_marks_anthropic")
    f = _one(_f(rows), "E-06")
    assert f.cause == "provider_no_marks" and f.turns == tuple(range(2, 25))
    assert f.counts["marks"] == 0 and f.confidence == "medium"  # predicted: an upper bound
    assert "E-08" not in _rules(rows)


def test_e06_needs_three_turns():
    assert _f(seeded_rows("no_marks_anthropic", turns=3)) == []
    assert _rules(seeded_rows("no_marks_anthropic", turns=4)) == ["E-06"]


def test_e06_below_the_provider_minimum_is_e08():
    rows = seeded_rows("no_marks_anthropic", base_prompt=600, growth=10)
    assert _rules(rows) == ["E-08"]
    f = _one(_f(rows), "E-08")
    assert f.cause == "provider_below_min"


def test_e06_rechecks_the_minimum():
    # a stored provider_no_marks whose prefix is under the minimum is not E-06
    rows = seeded_rows("no_marks_anthropic", base_prompt=600, growth=10)
    rows = [replace(r, lens_cause="provider_no_marks") if r.lens_cause else r for r in rows[1:]]
    assert "E-06" not in _rules(rows)


def test_e07_compaction_with_a_changed_head():
    rows = seeded_rows("clean", turns=12)
    rows = _with_cause(rows, {8}, "compacted", lens_break="sys", cobs2_edge="compacted_from")
    f = _one(_f(rows), "E-07")
    assert f.turns == (8,)
    head = rows[7].lens_cf_tokens or 0
    assert f.saving.tokens == head * 5  # the compaction turn and the 4 after it
    kept = seeded_rows("clean", turns=12)
    kept = _with_cause(kept, {8}, "compacted", lens_break="node:0", cobs2_edge="compacted_from")
    assert _f(kept) == []


def test_e08_server_eviction_is_shown_not_scored():
    from app.lens.catalogue import RULES

    f = _one(_f(seeded_rows("evicted_pauses")), "E-08")
    assert f.cause == "server_evicted" and f.turns == (8, 15, 21)
    assert f.counts["pause_s"] == 928  # the 900 s pause on top of the 28 s between turns
    assert RULES["E-08"].scored is False
    assert _rules(seeded_rows("evicted_pauses")) == ["E-08"]


def test_e08_needs_two_and_excludes_relays_and_short_gaps():
    assert _f(seeded_rows("evicted_pauses", evicted_at=(8,))) == []
    relayed = [replace(r, relayed=True) for r in seeded_rows("evicted_pauses")]
    assert _f(relayed) == []
    no_pause = seeded_rows("evicted_pauses", pause_s=0.0)
    assert _f(no_pause) == []


def test_e09_media_changed():
    rows = seeded_rows("clean")
    assert _f(_with_cause(rows, {7}, "media_changed", lens_break="node:2")) == []
    f = _one(_f(_with_cause(rows, {7, 11}, "media_changed", lens_break="node:2")), "E-09")
    assert f.counts["message"] == 2


def test_misrouted_and_new_conversation_never_fire():
    rows = seeded_rows("clean")
    rows = _with_cause(rows, {3, 5, 7}, "misrouted")
    rows = _with_cause(rows, {4, 6}, "new_conversation")
    assert _f(rows) == []


def test_unknown_cause_codes_are_ignored():
    assert _f(_with_cause(seeded_rows("clean"), {3, 5}, "not_a_cause")) == []


# -- seeds ---------------------------------------------------------------------


def test_clean_session_fires_nothing():
    assert _f(seeded_rows("clean")) == []
    assert efficiency_findings([], cross_sessions=3) == []


@pytest.mark.parametrize(
    "fault",
    [
        "unused_tools",
        "reread_file",
        "bad_schema",
        "loop",
        "empty_reply_template",
        "truncating_max_tokens",
    ],
)
def test_construction_and_response_seeds_fire_no_efficiency_rule(fault):
    assert _f(seeded_rows(fault)) == []


def test_every_fault_is_seeded_with_fake_ids():
    for fault in seed.FAULTS:
        rows = seeded_rows(fault)
        assert len(rows) == 24 and all("fake" in r.id for r in rows)
        assert all(r.token_id == seed.TOKEN for r in rows)
    with pytest.raises(KeyError):
        seeded_rows("nope")
    with pytest.raises(TypeError):
        seeded_rows("clean", nope=1)


def test_seed_unused_tools_shape():
    rows = seeded_rows("unused_tools")
    entries = seeded_toolsets("unused_tools")[seed.TOOLSET]
    called = {e[1] for r in rows for e in r.tools_out}
    offered = {h for h, _f, _c in entries}
    assert len(offered) == 12 and len(offered - called) == 7
    unused_chars = sum(c for h, _f, c in entries if h not in called)
    assert unused_chars / seed.CHARS_PER_TOKEN >= 1_000 and len(rows) >= 10


def test_seed_reread_file_and_loop():
    reread = turns_of(seeded_rows("reread_file"), own=True)
    assert max(t.repeat_results or 0 for t in reread) >= 2 * 900
    calls = [
        t.tools_out[0] for t in reread if t.tools_out and t.tools_out[0][2] == seed.args_hash(1)
    ]
    assert len(calls) == 3  # read three times: two repeats
    loop = turns_of(seeded_rows("loop"), own=True)
    same = [
        t.n
        for t in loop
        if t.tools_out and t.tools_out[0][1:] == (seed.tool_hash(1), seed.args_hash(1))
    ]
    assert same == [6, 7, 8, 9]


def test_seed_bad_schema_reply_and_limit_faults():
    bad = turns_of(seeded_rows("bad_schema"), own=True)
    assert [t.n for t in bad if t.calls_bad] == [7, 9, 12, 14]
    lint = bad[0].tools_lint
    assert lint is not None and lint[3] & 2 and not lint[3] & 1  # NO_DESC, not NO_TYPE
    empty = turns_of(seeded_rows("empty_reply_template"), own=True)
    hit = [t for t in empty if t.reply_chars == 0 and not t.calls_n]
    assert len(hit) >= 2 and all(t.finish_reason == "stop" and t.reasoning_tokens for t in hit)
    cut = turns_of(seeded_rows("truncating_max_tokens"), own=True)
    lens = [t for t in cut if t.finish_reason == "length"]
    assert len(lens) >= 2 and all(t.completion_tokens == t.max_tokens_req for t in lens)


def test_seed_rows_hold_no_text():
    for fault in seed.FAULTS:
        for r in seeded_rows(fault):
            blob = json.dumps([r.shape, r.calls_bad, r.tools_lint, r.tools_out])
            assert "CANARY" not in blob


def test_calls_bad_reads_indexed_and_old_entries():
    """I4: ``[k, hash8, code]`` entries; the old ``[hash8, code]`` pairs (rows
    stored before the index) read as ``k = None``; junk is skipped."""
    r = seeded_rows("clean", turns=1)[0]
    stored = [
        [1, "aa", "type"],
        ["bb", "json"],
        [None, "cc", "leaked_markup"],
        "junk",
        [True, "dd", "x"],
        [2],
    ]
    t = turns_of([replace(r, calls_bad=json.dumps(stored))], own=True)[0]
    assert t.calls_bad == (
        (1, "aa", "type"),
        (None, "bb", "json"),
        (None, "cc", "leaked_markup"),
        (None, "dd", "x"),
    )
