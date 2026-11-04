"""Replay a session through each engine's cache rules (sessions redesign, task 4)."""

from app.sessions.replay import ENGINES, ReplayTurn, replay
from app.sessions.store import ModelEngine

DP3 = ModelEngine("vllm", 3, True, 4, True)


def T(lane, t, p, c=None, rate=1000.0, eng=DP3):
    return ReplayTurn(lane, t, p, c, rate, eng)


def by(r):
    return {e["engine"]: e for e in r["engines"]}


def test_single_lane_all_engines_reuse_the_previous_prompt():
    r = replay([T("a", 0, 1000, 0), T("a", 1, 1200, 1000)], slots=4)
    e = by(r)
    assert (
        e["omlx"]["recompute_tokens"]
        == e["llamacpp"]["recompute_tokens"]
        == e["mlx_lm"]["recompute_tokens"]
        == 1200
    )
    assert e["lm_warden"]["recompute_tokens"] == 1200 and e["lm_warden"]["measured"] is True
    assert [x["engine"] for x in r["engines"]] == list(ENGINES)
    assert r["replicas"] == 3 and r["disk_load_excluded"] is True and r["turns"] == 2


def test_plain_vllm_loses_1_minus_1_over_n_of_cached():
    e = by(replay([T("a", 0, 1000, 0), T("a", 1, 1300, 900)], slots=4))
    assert e["vllm"]["recompute_tokens"] == round(1000 + 400 + 900 * (2 / 3))
    assert e["vllm"]["recompute_s"] == e["vllm"]["recompute_tokens"] / 1000.0


def test_dp1_or_affinity_off_vllm_equals_measured():
    for eng in (ModelEngine("vllm", 1, True, 4, True), ModelEngine("vllm", 3, False, 4, True)):
        e = by(replay([T("a", 0, 1000, 0, eng=eng), T("a", 1, 1300, 900, eng=eng)], slots=4))
        assert e["vllm"]["recompute_tokens"] == e["lm_warden"]["recompute_tokens"] == 1400


def test_mlx_lm_loses_the_prefix_when_lanes_interleave():
    e = by(
        replay(
            [T("a", 0, 100, 0), T("b", 1, 100, 0), T("a", 2, 150, 100), T("b", 3, 150, 100)],
            slots=4,
        )
    )
    assert e["mlx_lm"]["recompute_tokens"] == 500 and e["omlx"]["recompute_tokens"] == 300


def test_llamacpp_lru_beyond_s_lanes_recomputes():
    turns = [T(lane, i, 100, 0) for i, lane in enumerate("abc")] + [T("a", 3, 120, 100)]
    assert by(replay(turns, slots=2))["llamacpp"]["recompute_tokens"] == 420
    assert by(replay(turns, slots=3))["llamacpp"]["recompute_tokens"] == 320


def test_unknown_cache_leaves_measured_rows_none_but_replays_others():
    e = by(replay([T("a", 0, 100), T("a", 1, 150)], slots=4))
    assert e["lm_warden"]["recompute_tokens"] is None and e["lm_warden"]["recompute_s"] is None
    assert e["vllm"]["recompute_tokens"] is None
    assert e["omlx"]["recompute_tokens"] == 150


def test_turns_without_rate_are_skipped_from_time_only():
    r = replay([T("a", 0, 100, 0, rate=None), T("a", 1, 150, 100)], slots=4)
    e = by(r)
    assert r["skipped_time"] == 1
    assert e["omlx"]["recompute_tokens"] == 150 and e["omlx"]["recompute_s"] == 50 / 1000.0


def test_empty_is_none():
    assert replay([], slots=4) is None


def test_null_lane_is_its_own_lane():
    e = by(replay([T(None, 0, 100, 0), T(None, 1, 100, 0)], slots=4))
    assert e["omlx"]["recompute_tokens"] == 200
