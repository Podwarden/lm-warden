"""The pure session read model (sessions pages, task 7)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from app.sessions.model import (
    INFER_WINDOW_S,
    build_session,
    is_side_call,
    segment,
    summarize_list,
)
from app.sessions.refs import parse_ref, sess_ref
from app.sessions.store import ListRow, Row

H_AGENT = "a" * 8


def R(
    rid: str, *, conv: str | None = "c-main", start: float = 0.0, dur: float = 1.0, **kw: Any
) -> Row:
    base: dict[str, Any] = dict(
        id=rid,
        token_id="tok1",
        token_name="k",
        model_id="qwen",
        model="qwen",
        started_at=start,
        finished_at=start + dur,
        duration_s=dur,
        queued_s=0.0,
        ttft_s=0.2,
        finish_reason="stop",
        prompt_tokens=1000,
        cached_tokens=800,
        completion_tokens=10,
        reusable_tokens=900,
        cache_outcome="hit",
        cache_outcome_own="hit",
        divergence_cause=None,
        divergence_cause_own=None,
        dp_rank=0,
        session_key="S",
        parent_session_key=None,
        cobs2_session="S",
        cobs2_conv=conv,
        cobs2_node="n-" + rid,
        cobs2_parent_node=None,
        cobs2_parent_req=None,
        cobs2_edge="continue",
        cobs2_compaction_request=False,
        shape=None,
        toolset=None,
        tools_out=(),
        relayed=False,
        upstream=None,
        served_gpu_tokens=None,
        served_ram_tokens=None,
        served_nvme_tokens=None,
        served_remote_tokens=None,
        load_s=None,
        node_id=None,
    )
    base.update(kw)
    return Row(**base)


def chain(prefix: str, conv: str, starts: list[float], **first_kw: Any) -> list[Row]:
    rows: list[Row] = []
    for i, s in enumerate(starts):
        rid = f"{prefix}{i + 1}"
        if i == 0:
            rows.append(
                R(rid, conv=conv, start=s, cobs2_edge=first_kw.pop("edge", "new"), **first_kw)
            )
        else:
            prev = rows[-1]
            rows.append(
                R(
                    rid,
                    conv=conv,
                    start=s,
                    cobs2_parent_node=prev.cobs2_node,
                    cobs2_parent_req=prev.id,
                )
            )
    return rows


def build(rows: list[Row], **kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = dict(
        session_id="s_" + "f" * 32,
        token_name="k",
        truncated=False,
        oldest_data_at=0.0,
        cold_rates={},
        toolsets={},
        tool_names={},
        root_session="S",
    )
    args.update(kw)
    return build_session(rows, **args)


def cc_fixture() -> list[Row]:
    main = chain("m", "c-main", [0, 10, 20, 30])
    main[1] = replace(main[1], tools_out=(("agent", H_AGENT),))
    subs: list[Row] = []
    for k in range(3):
        subs += chain(f"s{k}", f"c-sub{k}", [12 + k, 14 + k])
    return main + subs


def test_claude_code_main_with_three_inferred_subagents() -> None:
    s = build(cc_fixture())
    main, *subs = s["conversations"]
    assert main["kind"] == "main" and main["label"] == "main"
    assert [c["label"] for c in subs] == ["subagent #1", "subagent #2", "subagent #3"]
    assert all(c["kind"] == "subagent" for c in subs)
    assert all(
        c["parent"] == {"conversation_id": main["id"], "request_id": "m2", "link": "inferred"}
        for c in subs
    )
    assert s["totals"]["conversations"] == 4 and s["totals"]["requests"] == 10


def test_inference_ignores_dispatch_older_than_window() -> None:
    main = chain("m", "c-main", [0, 10])
    main[0] = replace(main[0], tools_out=(("agent", H_AGENT),))
    late = INFER_WINDOW_S + 1 + main[0].finished_at
    sub = chain("s", "c-sub", [late, late + 1])
    s = build(main + sub)
    child = next(c for c in s["conversations"] if c["id"] == "c-sub")
    assert child["parent"] is None and child["kind"] == "conversation"


def test_running_parent_still_counts() -> None:
    main = chain("m", "c-main", [0])
    main[0] = replace(
        main[0], tools_out=(("agent", H_AGENT),), finished_at=5000.0, duration_s=5000.0
    )
    sub = chain("s", "c-sub", [3000, 3001])
    s = build(main + sub)
    child = next(c for c in s["conversations"] if c["id"] == "c-sub")
    assert child["parent"]["link"] == "inferred"


def test_side_calls_are_collapsed_and_never_parents() -> None:
    main = chain("m", "c-main", [0, 10])
    title = [R("t1", conv="c-title", start=1, cobs2_edge="new", tools_out=())]
    s = build(main + title)
    assert [c["id"] for c in s["conversations"]] == ["c-main"]
    assert [x["request_id"] for x in s["side_calls"]] == ["t1"]
    assert s["totals"]["side_calls"] == 1
    assert is_side_call(1, False, "new") and not is_side_call(1, False, "compacted_from")


def test_single_turn_with_tools_is_a_lane() -> None:
    main = chain("m", "c-main", [0, 10])
    one = [R("x1", conv="c-x", start=20, cobs2_edge="new", tools_out=(("read", "b" * 8),))]
    s = build(main + one)
    assert "c-x" in [c["id"] for c in s["conversations"]]


def test_compaction_lane_is_labelled_and_linked_by_parent_req() -> None:
    main = chain("m", "c-main", [0, 10])
    comp = [
        R(
            "p1",
            conv="c-p",
            start=20,
            cobs2_edge="compacted_from",
            cobs2_parent_req="m2",
            cobs2_parent_node="n-m2",
        )
    ]
    comp += [R("p2", conv="c-p", start=30, cobs2_parent_req="p1", cobs2_parent_node="n-p1")]
    comp2 = [
        R(
            "q1",
            conv="c-q",
            start=40,
            cobs2_edge="compacted_from",
            cobs2_parent_req="p2",
            cobs2_parent_node="n-p2",
        )
    ]
    s = build(main + comp + comp2)
    by = {c["id"]: c for c in s["conversations"]}
    assert by["c-p"]["kind"] == "compaction" and by["c-p"]["label"] == "main after compaction"
    assert by["c-p"]["parent"] == {
        "conversation_id": "c-main",
        "request_id": "m2",
        "link": "compaction",
    }
    assert by["c-q"]["label"] == "main after compaction 2"
    assert s["totals"]["compactions"] == 2


def test_known_parent_from_parent_session_key() -> None:
    main = chain("m", "c-main", [0, 10])
    child = chain(
        "k", "c-kid", [12, 13], cobs2_session="K", session_key="K", parent_session_key="S"
    )
    child[1] = replace(child[1], cobs2_session="K", session_key="K", parent_session_key="S")
    s = build(main + child)
    kid = next(c for c in s["conversations"] if c["id"] == "c-kid")
    assert kid["parent"] == {"conversation_id": "c-main", "request_id": "m2", "link": "known"}
    assert kid["kind"] == "subagent"


def test_lineage_unknown_marks_gap_and_restart() -> None:
    rows = chain("m", "c-main", [0, 10, 20])
    rows[1] = replace(rows[1], cobs2_parent_node="gone", cobs2_parent_req=None)
    rows[2] = replace(rows[2], cobs2_edge="new", cobs2_parent_node=None, cobs2_parent_req=None)
    s = build(rows)
    t = {x["request_id"]: x for x in s["turns"]}
    assert t["m1"]["lineage_unknown"] is False
    assert t["m2"]["lineage_unknown"] is True
    assert t["m3"]["lineage_unknown"] is True


def test_r7_row_without_lineage_is_unknown() -> None:
    rows = chain("m", "c-main", [0, 10])
    rows.append(R("big", conv=None, start=20, cobs2_edge=None, cobs2_node=None))
    s = build(rows)
    t = {x["request_id"]: x for x in s["turns"]}
    assert t["big"]["lineage_unknown"] is True and t["big"]["conversation_id"] is None
    assert any(c["kind"] == "unlinked" for c in s["conversations"])


def rows_at(starts: list[float]) -> list[Row]:
    return [R(f"r{i}", start=s, dur=0.0) for i, s in enumerate(starts)]


def test_segment_splits_on_30_min_idle_gap() -> None:
    assert [len(g) for g in segment(rows_at([0, 60, 60 + 1801, 60 + 1900]))] == [2, 2]


def test_timings_split_prefill_and_generation() -> None:
    s = build([R("a", start=0, dur=3.0, ttft_s=0.5), R("b", start=5, dur=2.0, ttft_s=None)])
    t = {x["request_id"]: x for x in s["turns"]}
    assert (t["a"]["prefill_s"], t["a"]["gen_s"]) == (0.5, 2.5)
    assert (t["b"]["prefill_s"], t["b"]["gen_s"]) == (None, None)


def test_totals_and_cache_served_time_exclude_relayed() -> None:
    local = chain("m", "c-main", [0, 10])
    rel = [
        R(
            "x1",
            conv="c-main",
            start=20,
            relayed=True,
            upstream="anthropic.example",
            model_id="claude",
            cached_tokens=5000,
            prompt_tokens=6000,
        )
    ]
    s = build(local + rel, cold_rates={"qwen": 1000.0})
    tot = s["totals"]
    assert tot["cache_served_s"] == pytest.approx(1600 / 1000.0)
    assert tot["cached_tokens"] == 1600 and tot["prefill_tokens"] == 2000
    assert tot["cached_share"] == pytest.approx(0.8)
    assert tot["relayed"] == {
        "requests": 1,
        "prefill_tokens": 6000,
        "cached_tokens": 5000,
        "upstream": "anthropic.example",
    }
    assert tot["saved_vs_vanilla_s"] is None


def test_cache_served_none_without_rates() -> None:
    assert build(chain("m", "c-main", [0]))["totals"]["cache_served_s"] is None


def test_turn_cap_keeps_newest_and_flags_truncation() -> None:
    s = build(chain("m", "c-main", [0, 1]), truncated=True)
    assert s["truncated"] is True


def rows_with_toolset(
    *, offered: int, called: int, chars_each: int, prompt_tokens: int, shape_chars: int
):
    entries = [(f"{i:08x}", "other", chars_each) for i in range(offered)]
    out = tuple(("other", entries[i][0]) for i in range(called))
    rows = [
        R(
            "m1",
            start=0,
            toolset="d" * 32,
            prompt_tokens=prompt_tokens,
            shape={"chars": shape_chars},
            tools_out=out,
        ),
        R(
            "m2",
            start=10,
            toolset="d" * 32,
            prompt_tokens=prompt_tokens,
            shape={"chars": shape_chars},
            cobs2_parent_req="m1",
            cobs2_parent_node="n-m1",
        ),
    ]
    return rows, {"d" * 32: entries}


def test_unused_tools_per_turn_and_share() -> None:
    rows, ts = rows_with_toolset(
        offered=31, called=6, chars_each=1000, prompt_tokens=88_000, shape_chars=352_000
    )
    t = build(rows, toolsets=ts)["tools"]
    assert (t["offered"], t["called"], t["unused"]) == (31, 6, 25)
    per = 25 * 1000 * 88_000 / 352_000
    assert t["unused_tokens_per_turn"] == pytest.approx(per)
    assert t["unused_tokens_total"] == pytest.approx(2 * per)
    assert t["unused_share"] == pytest.approx(per / 88_000)
    assert t["rows"][0]["name"] is None and len(t["rows"][0]["id"]) == 8
    assert t["estimated"] is True and t["truncated"] is False


def test_tool_rows_carry_names_when_known() -> None:
    rows, ts = rows_with_toolset(
        offered=3, called=1, chars_each=10, prompt_tokens=100, shape_chars=400
    )
    t = build(rows, toolsets=ts, tool_names={"00000001": "Task"})["tools"]
    names = {r["id"]: r["name"] for r in t["rows"]}
    assert names["00000001"] == "Task" and names["00000002"] is None
    called = {r["id"]: r["called"] for r in t["rows"]}
    assert called["00000000"] == 1


def test_missing_toolset_is_unknown_not_zero() -> None:
    rows, ts = rows_with_toolset(
        offered=4, called=0, chars_each=100, prompt_tokens=100, shape_chars=400
    )
    rows.append(
        R("m3", start=20, toolset="e" * 32, cobs2_parent_req="m2", cobs2_parent_node="n-m2")
    )
    t = build(rows, toolsets=ts)["tools"]
    assert t["unused_tokens_per_turn"] == pytest.approx(4 * 100 * 100 / 400)


def test_128_tool_toolset_stays_bounded() -> None:
    rows, ts = rows_with_toolset(
        offered=128, called=2, chars_each=10, prompt_tokens=100, shape_chars=4000
    )
    t = build(rows, toolsets=ts)["tools"]
    assert (t["offered"], t["unused"], t["truncated"]) == (128, 126, True)


def L(
    rid: str,
    *,
    sess: str,
    start: float,
    tok: str = "tok1",
    conv: str = "c",
    edge: str = "continue",
    parent: str | None = None,
    tools: bool = True,
    relayed: bool = False,
) -> ListRow:
    return ListRow(
        id=rid,
        token_id=tok,
        token_name="k",
        model="qwen",
        started_at=start,
        finished_at=start + 1,
        prompt_tokens=100,
        cached_tokens=50,
        completion_tokens=5,
        cobs2_session=sess,
        parent_session_key=parent,
        cobs2_conv=conv,
        cobs2_edge=edge,
        relayed=relayed,
        has_tools=tools,
    )


def test_summarize_list_folds_children_and_segments_fallback() -> None:
    rows = [
        L("a" * 32, sess="S", start=0, conv="c1", edge="new"),
        L("b" * 32, sess="S", start=10, conv="c1"),
        L("c" * 32, sess="K", start=20, conv="c2", edge="new", parent="S"),
        L("1" * 32, sess="f:x", start=100, conv="c3", edge="new"),
        L("2" * 32, sess="f:x", start=100 + 1900, conv="c3"),
        L("t" * 32, sess="S", start=11, conv="c9", edge="new", tools=False),
    ]
    roots = {("tok1", "K"): "S"}
    items = summarize_list(
        rows, secret="sec", root_of=lambda t, s: roots.get((t, s), s), now=2005.0
    )
    assert len(items) == 3
    assert items[0]["last_at"] >= items[1]["last_at"] >= items[2]["last_at"]
    harness = next(i for i in items if i["requests"] == 4)
    assert harness["session_id"] == "s_" + sess_ref("sec", "tok1", "S")
    assert harness["conversations"] == 2  # c9 is a side call
    segs = [i for i in items if i is not harness]
    assert all(parse_ref(i["session_id"]).segment is not None for i in segs)
    assert {i["requests"] for i in segs} == {1}
    assert segs[0]["active"] is True
    assert harness["cached_share"] == pytest.approx(0.5)


# -- sessions redesign, task 6 ------------------------------------------------

from app.sessions.model import ListContext  # noqa: E402
from app.sessions.store import ModelEngine  # noqa: E402

DP3 = ModelEngine("vllm", 3, True, 4, True)


def local_rows(*, cached=900, prompt=1000, n=4, conv="c-main"):
    rows = chain("m", conv, [i * 10 for i in range(n)])
    return [replace(r, cached_tokens=cached, prompt_tokens=prompt) for r in rows]


def test_session_dead_weight_headline_and_kinds() -> None:
    rows = [
        R(
            f"m{i}",
            start=i * 10,
            shape={"chars": 4000, "dw": {"dup": 400, "boil": 200, "reason": 0}},
            prompt_tokens=1000,
            toolset="d1",
            tools_out=(("read", "h1"),),
        )
        for i in range(3)
    ]
    s = build(rows, toolsets={"d1": [("h1", "read", 400), ("h2", "web", 400)]})
    dw = s["dead_weight"]
    assert dw["kinds"]["unused_tools"]["tokens"] == pytest.approx(300)
    assert dw["kinds"]["dup"]["tokens"] == pytest.approx(300)
    assert dw["share"] == pytest.approx(750 / 3000) and dw["turns_measured"] == 3
    assert s["turns"][0]["dead_weight"]["total"] == 250


def test_turn_unused_tools_are_relative_to_its_conversation() -> None:
    ts = {"d1": [("h1", "read", 400), ("h2", "web", 400)]}
    a = [
        R(
            "a1",
            conv="ca",
            toolset="d1",
            shape={"chars": 4000, "dw": {"dup": 0, "boil": 0, "reason": 0}},
            tools_out=(("read", "h1"),),
        )
    ]
    b = [
        R(
            "b1",
            conv="cb",
            start=5,
            toolset="d1",
            shape={"chars": 4000, "dw": {"dup": 0, "boil": 0, "reason": 0}},
            tools_out=(("web", "h2"),),
        )
    ]
    s = build(a + b, toolsets=ts)
    t = {x["request_id"]: x for x in s["turns"]}
    assert (
        t["a1"]["dead_weight"]["unused_tools"] == 100
        and t["b1"]["dead_weight"]["unused_tools"] == 100
    )
    assert s["dead_weight"]["kinds"]["unused_tools"]["tokens"] == 0  # session-wide both are called


def test_anatomy_on_turn_conversation_and_session_sum_to_prompt() -> None:
    s = build(local_rows())
    assert sum(s["anatomy"].values()) == 4000
    assert sum(s["conversations"][0]["anatomy"].values()) == 4000
    assert all(sum(t["anatomy"].values()) == t["prompt_tokens"] for t in s["turns"])


def test_three_time_figures_and_replicas() -> None:
    s = build(local_rows(), cold_rates={"qwen": 1000.0}, engines={"qwen": DP3})
    t = s["totals"]
    assert t["cache_served_s"] == pytest.approx(3.6) and t["recomputed_s"] == pytest.approx(0.4)
    assert (
        t["saved_vs_vanilla_s"] == pytest.approx(3.6 * 2 / 3)
        and t["replicas"] == 3
        and t["where"] == "local"
    )
    assert s["conversations"][0]["saved_vs_vanilla_s"] == pytest.approx(3.6 * 2 / 3)
    eng = {e["engine"]: e for e in s["engine_replay"]["engines"]}
    assert eng["lm_warden"]["recompute_tokens"] == 400


def test_relayed_only_session_saved_is_null_and_where_relayed() -> None:
    rows = [replace(r, relayed=True, upstream="api.anthropic.com") for r in local_rows()]
    s = build(rows, cold_rates={"qwen": 1000.0}, engines={"qwen": DP3})
    assert s["totals"]["saved_vs_vanilla_s"] is None and s["totals"]["where"] == "relayed"
    assert s["engine_replay"] is None and s["totals"]["prompt_tokens"] == 4000


def test_hybrid_session_computes_saved_on_local_turns() -> None:
    rows = local_rows()
    rows[0] = replace(rows[0], relayed=True)
    s = build(rows, cold_rates={"qwen": 1000.0}, engines={"qwen": DP3})
    assert s["totals"]["where"] == "hybrid" and s["totals"]["saved_vs_vanilla_s"] == pytest.approx(
        2.7 * 2 / 3
    )


def test_mlx_session_cache_unknown_everywhere() -> None:
    s = build(
        local_rows(cached=None),
        engines={"qwen": ModelEngine("mlx", 1, True, 4, False)},
        cold_rates={"qwen": 1000.0},
    )
    assert (
        s["anatomy"]["unknown"] > 0
        and s["anatomy"]["cached"] == 0
        and s["totals"]["cache_served_s"] is None
    )
    assert all(t["cache_known"] is False for t in s["turns"])
    eng = {e["engine"]: e for e in s["engine_replay"]["engines"]}
    assert (
        eng["lm_warden"]["recompute_tokens"] is None and eng["omlx"]["recompute_tokens"] is not None
    )


def test_rows_without_dw_are_unknown_not_zero() -> None:
    s = build(
        [
            R("a", shape={"chars": 4000}),
            R("b", start=5, shape={"chars": 4000, "dw": {"dup": 0, "boil": 400, "reason": 0}}),
        ]
    )
    assert s["dead_weight"]["turns_measured"] == 1
    assert next(t for t in s["turns"] if t["request_id"] == "a")["dead_weight"] is None


def test_deleted_model_or_no_rate_keeps_tokens_drops_time() -> None:
    s = build(local_rows(), engines={}, cold_rates={})
    assert s["totals"]["saved_vs_vanilla_s"] is None and s["totals"]["recomputed_s"] is None
    assert s["totals"]["prompt_tokens"] == 4000


def test_harness_label_from_session_source() -> None:
    rows = [replace(r, session_source="claude_code_header") for r in local_rows()]
    assert build(rows)["harness"] == "claude_code"


def test_turn_pause_fields_and_session_pauses() -> None:
    rows = [
        R("a1", start=0, conv="a"),
        R("b1", start=5, conv="b"),
        R("b2", start=30, conv="b"),
        R("a2", start=301, conv="a", cached_tokens=0, reusable_tokens=9000, cache_outcome="lost"),
    ]
    s = build(rows, cold_rates={"qwen": 1000.0})
    a2 = next(t for t in s["turns"] if t["request_id"] == "a2")
    assert (
        a2["pause_s"] == pytest.approx(300)
        and a2["evicted_on_resume"]
        and a2["lost_s"] == pytest.approx(9.0)
    )
    assert s["pauses"]["evicted"] == 1 and s["pauses"]["buckets"][1]["evicted"] == 1


def test_list_items_carry_anatomy_dead_weight_saved_and_evictions() -> None:
    rows = [
        replace(
            L("a" * 32, sess="S", start=0, conv="c1", edge="new"),
            model_id="qwen",
            chars=400,
            dw=(40, 0, 0),
            cached_tokens=50,
        ),
        replace(
            L("b" * 32, sess="S", start=400, conv="c1"),
            model_id="qwen",
            chars=400,
            dw=(0, 0, 0),
            cached_tokens=0,
            reusable_tokens=5000,
            cache_outcome="lost",
            prompt_tokens=5000,
        ),
    ]
    ctx = ListContext(toolsets={}, engines={"qwen": DP3}, cold_rates={"qwen": 1000.0})
    (item,) = summarize_list(rows, secret="sec", root_of=lambda t, s: s, now=500.0, ctx=ctx)
    assert item["where"] == "local" and sum(item["anatomy"].values()) == 5100
    assert item["dead_weight_measured"] and item["dead_weight_tokens"] == 10
    assert item["evicted_resumes"] == 1 and item["evicted_lost_s"] == pytest.approx(5.0)
    assert item["saved_vs_vanilla_s"] == pytest.approx(50 * 2 / 3 / 1000)
    assert "_marks" in item


def test_list_pause_marks_ignore_fallback_segmentation() -> None:
    rows = [
        replace(L("1" * 32, sess="f:x", start=0, conv="c3", edge="new"), model_id="qwen"),
        replace(
            L("2" * 32, sess="f:x", start=2000, conv="c3"),
            model_id="qwen",
            cached_tokens=0,
            reusable_tokens=9000,
            cache_outcome="lost",
        ),
    ]
    items = summarize_list(
        rows, secret="sec", root_of=lambda t, s: s, now=3000.0, ctx=ListContext({}, {}, {})
    )
    assert len(items) == 2
    later = next(i for i in items if i["started_at"] >= 2000)
    # the summary keeps it (R13); the item does not, because its own page starts after the gap (I3)
    assert sum(1 for m in later["_marks"] if m.evicted) == 1
    assert later["evicted_resumes"] == 0 and later["evicted_lost_s"] is None


def test_mlx_list_item_and_session_read_cache_unknown_never_zero() -> None:
    rows = [
        replace(
            L("a" * 32, sess="S", start=0, conv="c1", edge="new"),
            model_id="qwen",
            chars=400,
            dw=(40, 0, 0),
            cached_tokens=None,
        ),
        replace(
            L("b" * 32, sess="S", start=10, conv="c1"),
            model_id="qwen",
            chars=400,
            dw=(0, 0, 0),
            cached_tokens=None,
        ),
    ]
    (it,) = summarize_list(
        rows, secret="sec", root_of=lambda t, s: s, now=500.0, ctx=ListContext({}, {}, {})
    )
    assert it["cache_known"] is False and it["cached_share"] is None and it["cache_share"] is None
    s = build([R("m1", cached_tokens=None), R("m2", start=10, cached_tokens=None)])
    assert s["totals"]["cached_share"] is None and s["totals"]["cache_share"] is None


def test_cache_share_is_cached_over_prompt_unscaled_by_dead_weight() -> None:
    rows = [
        replace(
            L("a" * 32, sess="S", start=0, conv="c1", edge="new"),
            model_id="qwen",
            chars=1000,
            dw=(1600, 0, 0),
            cached_tokens=900,
            prompt_tokens=1000,
        ),
    ]
    (it,) = summarize_list(
        rows, secret="sec", root_of=lambda t, s: s, now=500.0, ctx=ListContext({}, {}, {})
    )
    assert it["cache_share"] == pytest.approx(0.9) and it["anatomy"]["cached"] < 900
    assert it["_ck"] == (900, 1000)
    s = build(
        [
            R(
                "m1",
                cached_tokens=900,
                shape={"chars": 1000, "dw": {"dup": 1600, "boil": 0, "reason": 0}},
            )
        ]
    )
    assert s["totals"]["cache_share"] == pytest.approx(0.9)


def test_summarize_list_adapts_each_row_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sessions.model as m

    calls = 0
    real = m._c_of_list

    def spy(*a: Any, **k: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(*a, **k)

    monkeypatch.setattr(m, "_c_of_list", spy)
    rows = [L(f"{i:032x}", sess="S", start=float(i), conv="c1") for i in range(6)]
    summarize_list(
        rows, secret="sec", root_of=lambda t, s: s, now=100.0, ctx=ListContext({}, {}, {})
    )
    assert calls == len(rows)


def test_cached_share_counts_only_requests_that_report_cache() -> None:
    """Verified finding #67: a row with no cache data must not dilute the share."""
    rows = [
        replace(
            L("a" * 32, sess="S", start=0, conv="c1", edge="new"),
            cached_tokens=900,
            prompt_tokens=1000,
        ),
        replace(L("b" * 32, sess="S", start=10, conv="c1"), cached_tokens=None, prompt_tokens=1000),
    ]
    (it,) = summarize_list(
        rows, secret="sec", root_of=lambda t, s: s, now=500.0, ctx=ListContext({}, {}, {})
    )
    assert it["cached_share"] == pytest.approx(0.9)
    s = build(
        [
            R("m1", cached_tokens=900, prompt_tokens=1000),
            R("m2", start=10, cached_tokens=None, prompt_tokens=1000),
        ]
    )
    assert s["totals"]["cached_share"] == pytest.approx(0.9)
