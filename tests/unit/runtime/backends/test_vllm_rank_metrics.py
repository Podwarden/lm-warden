"""Per-rank reading of vLLM's engine="i" label series (#286 task 4)."""

from pathlib import Path

from app.runtime.backends.vllm.metrics import RankReading, read_ranks

DP3 = Path(__file__).parents[3] / "fixtures" / "vllm_metrics_dp3.txt"
LEGACY = Path(__file__).parents[2] / "stats" / "fixtures" / "vllm_metrics_0_25_1.txt"


def test_three_ranks_with_their_own_numbers():
    ranks = read_ranks(DP3.read_text())
    assert sorted(ranks) == [0, 1, 2]
    assert ranks[0] == RankReading(2.0, 0.0, 0.25, 1234.0, 1500.0)
    assert ranks[1].requests_running == 5.0
    assert ranks[1].requests_waiting == 3.0
    assert ranks[1].kv_cache_usage_perc == 0.5
    assert ranks[1].prefix_cache_hits == 400.0


def test_rank_with_zero_queries_still_appears():
    ranks = read_ranks(DP3.read_text())
    assert ranks[2].prefix_cache_queries == 0.0


def test_unlabeled_series_is_ignored():
    ranks = read_ranks(DP3.read_text())
    assert all(r.requests_running != 99.0 for r in ranks.values())


def test_exposition_without_engine_labels_is_empty():
    assert read_ranks(LEGACY.read_text()) == {}


def test_empty_and_garbage_bodies_are_empty():
    assert read_ranks("") == {}
    assert read_ranks("not prometheus at all") == {}


def test_legacy_spellings_are_accepted():
    body = 'vllm:gpu_cache_usage_perc{engine="0"} 0.1\nvllm:prefix_cache_hits{engine="0"} 3\n'
    r = read_ranks(body)[0]
    assert r.kv_cache_usage_perc == 0.1
    assert r.prefix_cache_hits == 3.0
