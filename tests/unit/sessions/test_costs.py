"""Dead weight, prompt anatomy and the three time figures (sessions redesign, task 3)."""

from dataclasses import astuple

import pytest

from app.sessions.costs import (
    Anatomy,
    anatomy_json,
    anatomy_of,
    dead_weight_json,
    dead_weight_turn_json,
    recomputed_s,
    saved_vs_plain_s,
    served_s,
    tokens_per_char,
    turn_dead_weight,
    where_of,
)
from app.sessions.store import ModelEngine


def test_turn_dead_weight_converts_chars_with_the_rows_ratio():
    d = turn_dead_weight(prompt_tokens=1000, chars=4000, dw=(400, 200, 0), unused_tool_chars=800)
    assert (d.unused_tools, d.dup, d.boil, d.reason, d.total) == (200, 100, 50, 0, 350)
    assert dead_weight_turn_json(d)["share"] == pytest.approx(0.35)


def test_dw_mapping_form_is_accepted():
    d = turn_dead_weight(
        prompt_tokens=100, chars=400, dw={"dup": 40, "boil": 0, "reason": 0}, unused_tool_chars=None
    )
    assert d.dup == 10 and d.unused_tools is None and d.total == 10


def test_missing_dw_is_unknown_not_zero():
    d = turn_dead_weight(prompt_tokens=1000, chars=4000, dw=None, unused_tool_chars=None)
    assert d.total is None and dead_weight_turn_json(d) is None


def test_total_is_capped_at_prompt():
    d = turn_dead_weight(prompt_tokens=100, chars=100, dw=(90, 90, 0), unused_tool_chars=90)
    assert d.total == 100


def test_ratio_falls_back_to_a_quarter():
    assert (
        tokens_per_char(100, None) == 0.25
        and tokens_per_char(100, 0) == 0.25
        and tokens_per_char(100, 50) == 2
    )


def test_dead_weight_json_counts_measured_turns_only():
    j = dead_weight_json(
        [
            turn_dead_weight(prompt_tokens=1000, chars=4000, dw=(400, 0, 0), unused_tool_chars=0),
            turn_dead_weight(prompt_tokens=1000, chars=4000, dw=None, unused_tool_chars=None),
        ]
    )
    assert (j["turns"], j["turns_measured"]) == (2, 1) and j["share"] == pytest.approx(0.1)
    assert j["kinds"]["dup"]["share"] == pytest.approx(0.1)
    assert j["kinds"]["dup"]["per_turn"] == pytest.approx(100) and j["estimated"] is True


def test_dead_weight_json_unknown_everywhere():
    j = dead_weight_json(
        [turn_dead_weight(prompt_tokens=10, chars=None, dw=None, unused_tool_chars=None)]
    )
    assert (
        j["total_tokens"] is None
        and j["share"] is None
        and j["kinds"]["boil"] == {"tokens": None, "share": None, "per_turn": None}
    )


def test_anatomy_partitions_the_prompt():
    a = anatomy_of(prompt=1000, cached=700, prev_prompt=950, dead=200)
    assert a.dead == 200 and sum(astuple(a)) == pytest.approx(1000)
    assert (a.cached, a.recomputed, a.new) == pytest.approx((560, 200, 40))


def test_anatomy_first_turn_is_all_new_beyond_cache():
    a = anatomy_of(prompt=1000, cached=100, prev_prompt=None, dead=0)
    assert a.recomputed == 0 and a.new == 900 and a.cached == 100


def test_anatomy_unknown_cache():
    assert anatomy_of(prompt=100, cached=None, prev_prompt=None, dead=10) == Anatomy(
        0, 10, 0, 0, 90
    )


def test_anatomy_json_sums_exactly():
    a = Anatomy(100 / 3, 100 / 3, 100 / 3, 0, 0)
    j = anatomy_json(a)
    assert sum(j.values()) == 100 and set(j) == {"cached", "dead", "recomputed", "new", "unknown"}


@pytest.mark.parametrize(
    "eng,expect",
    [
        (ModelEngine("vllm", 3, True, 4, True), 2.0),
        (ModelEngine("vllm", 1, True, 4, True), 0.0),
        (ModelEngine("vllm", 3, False, 4, True), 0.0),
        (None, None),
    ],
)
def test_saved_vs_plain(eng, expect):
    assert saved_vs_plain_s(3000, eng, 1000.0) == expect


def test_time_figures_unknown_without_rate_or_cache():
    assert (
        served_s(100, None) is None and served_s(None, 10.0) is None and served_s(100, 10.0) == 10
    )
    assert recomputed_s(100, 40, 10.0) == 6 and recomputed_s(100, None, 10.0) is None
    assert saved_vs_plain_s(None, ModelEngine("vllm", 3, True, 4, True), 1.0) is None


def test_where_of():
    assert [where_of(1, 0), where_of(0, 2), where_of(1, 1)] == ["local", "relayed", "hybrid"]


def test_a_turn_without_dw_is_not_measured_even_with_unused_tools_known():
    # rows recorded before dead weight was tracked still carry a toolset (I2)
    d = turn_dead_weight(prompt_tokens=1000, chars=4000, dw=None, unused_tool_chars=800)
    assert d.measured is False and d.unused_tools == pytest.approx(200)
    j = dead_weight_json([d, d])
    assert (j["turns"], j["turns_measured"]) == (2, 0)
    assert j["total_tokens"] is None and j["share"] is None and j["prompt_tokens"] == 0
    assert j["kinds"]["unused_tools"]["tokens"] == 400
    assert dead_weight_turn_json(d) is None


def test_dead_weight_json_reports_the_measured_prompt_total():
    j = dead_weight_json(
        [
            turn_dead_weight(prompt_tokens=1000, chars=4000, dw=(400, 0, 0), unused_tool_chars=0),
            turn_dead_weight(prompt_tokens=9000, chars=None, dw=None, unused_tool_chars=None),
        ]
    )
    assert j["prompt_tokens"] == 1000 and j["total_tokens"] == 100
