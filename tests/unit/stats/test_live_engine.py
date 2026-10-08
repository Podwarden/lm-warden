"""Unit tests for the Plane-A engine-metrics scraper (``app/stats/live_engine``).

The SSE generator itself loops indefinitely (a live feed); as with the header
stream tests we don't coax the TestClient into closing a stream mid-iter.
Instead we pin the observable contract by exercising the pure building blocks:

  * the inline Prometheus parser against a captured 0.25.1 metrics blob,
  * the KV absolute-token derivation from ``cache_config_info``,
  * histogram percentile / mean math,
  * per-second rate + interval prefix-hit-rate deltas vs a previous frame,
  * a missing/renamed metric mapping to ``null`` rather than raising,
  * the shared TTL cache collapsing concurrent scrapes to one fetch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.runtime.backends.metrics import EngineReading
from app.runtime.backends.vllm import VllmBackend
from app.stats.live_engine import (
    Metrics,
    RateState,
    ScrapeResult,
    _interval_seconds,
    _MetricsCache,
    build_frame,
    hist_mean,
    hist_quantile,
    parse_prometheus,
)

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_0_25_1.txt"


@pytest.fixture
def sample_text() -> str:
    return FIXTURE.read_text()


@pytest.fixture
def metrics(sample_text: str) -> Metrics:
    return Metrics(parse_prometheus(sample_text))


@pytest.fixture
def reading(sample_text: str):
    """Sub-project C: build_frame consumes an EngineReading, not a Metrics.

    The vLLM metric NAMES moved into app/runtime/backends/vllm/metrics.py, so
    the frame tests below go through the backend to get there. The parser tests
    above still use ``metrics`` directly -- the parser changed module, not
    behaviour.
    """
    return VllmBackend().parse_metrics(sample_text)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_skips_comments_and_blanks(metrics: Metrics):
    # Comment/HELP/TYPE lines never become samples.
    assert metrics.value("vllm:num_requests_running") == 3.0
    assert metrics.value("vllm:num_requests_waiting") == 2.0


def test_parser_filters_by_label(metrics: Metrics):
    assert metrics.value("vllm:num_requests_waiting_by_reason", reason="capacity") == 2.0
    assert metrics.value("vllm:num_requests_waiting_by_reason", reason="deferred") == 0.0


def test_parser_aggregates_split_series(metrics: Metrics):
    # request_success_total is split by finished_reason — filtering isolates one.
    assert metrics.value("vllm:request_success_total", finished_reason="stop") == 40210.0
    # No filter sums across the three finished_reason series.
    assert metrics.value("vllm:request_success_total") == 40210.0 + 118.0 + 33.0


def test_parser_scientific_notation(metrics: Metrics):
    assert metrics.value("vllm:estimated_flops_per_gpu_total") == pytest.approx(1.2e15)


def test_info_labels(metrics: Metrics):
    info = metrics.info("vllm:cache_config_info")
    assert info is not None
    assert info["block_size"] == "16"
    assert info["num_gpu_blocks"] == "14050"


def test_missing_metric_is_none(metrics: Metrics):
    assert metrics.value("vllm:this_metric_does_not_exist") is None
    assert metrics.info("vllm:no_such_info") is None
    assert metrics.histogram("vllm:no_such_histogram") is None


# --------------------------------------------------------------------------- #
# Histogram math
# --------------------------------------------------------------------------- #


def test_hist_quantile_interpolates():
    # 100 observations across e2e buckets; p50 rank=50 lands in (5,10].
    buckets = [
        (1.0, 0.0),
        (5.0, 20.0),
        (10.0, 60.0),
        (30.0, 90.0),
        (60.0, 99.0),
        (float("inf"), 100.0),
    ]
    assert hist_quantile(buckets, 0.5) == pytest.approx(8.75)
    assert hist_quantile(buckets, 0.9) == pytest.approx(30.0)
    assert hist_quantile(buckets, 0.99) == pytest.approx(60.0)


def test_hist_quantile_empty_or_zero():
    assert hist_quantile([], 0.5) is None
    assert hist_quantile([(1.0, 0.0), (float("inf"), 0.0)], 0.5) is None


def test_hist_mean():
    assert hist_mean(1200.0, 100.0) == pytest.approx(12.0)
    assert hist_mean(None, 100.0) is None
    assert hist_mean(1200.0, 0.0) is None


# --------------------------------------------------------------------------- #
# Frame build — first frame (rates null) + derivations
# --------------------------------------------------------------------------- #


def test_first_frame_shape_and_derivations(reading):
    frame, state = build_frame(
        reading,
        model="qwopus3.6-27b-coder-fp8-model",
        model_id="90b6c566e02afa8f",
        max_model_len=224800,
        prev=None,
        scrape_monotonic=100.0,
    )

    assert frame["model"] == "qwopus3.6-27b-coder-fp8-model"
    assert frame["model_id"] == "90b6c566e02afa8f"
    assert frame["max_model_len"] == 224800
    assert frame["scrape_error"] is None

    eng = frame["engine"]
    assert eng["num_requests_running"] == 3
    assert eng["num_requests_waiting"] == 2
    assert eng["waiting_by_reason"] == {"capacity": 2, "deferred": 0}
    assert eng["kv_cache_usage_perc"] == pytest.approx(0.5)
    # 16 * 14050 = 224800 total; 0.5 * 224800 = 112400 used.
    assert eng["kv_tokens_total"] == 224800
    assert eng["kv_tokens_used"] == 112400
    assert eng["engine_sleep_state"] == 0
    assert eng["preemptions_total"] == 12
    # First frame → all rates null.
    assert eng["preemptions_per_s"] is None
    assert frame["throughput"]["generation_tokens_per_s"] is None
    assert frame["throughput"]["prompt_tokens_per_s"] is None
    assert frame["throughput"]["generation_tokens_total"] == 500000
    assert frame["cache"]["prefix_hit_rate"] is None
    assert frame["cache"]["prefix_hit_rate_cumulative"] == pytest.approx(0.55)

    lat = frame["latency"]
    assert lat["e2e_p50"] == pytest.approx(8.75)
    assert lat["e2e_p90"] == pytest.approx(30.0)
    assert lat["e2e_p99"] == pytest.approx(60.0)
    assert lat["ttft_p50"] == pytest.approx(0.2)
    assert lat["ttft_mean"] == pytest.approx(0.34)
    assert lat["itl_p50"] == pytest.approx(0.025)

    # MFU is deleted, field and panel: mfu_estimate was a literal None stub and
    # the FLOPs counter beside it was cumulative rendered as per-second.
    assert "mfu" not in frame
    assert frame["finished"] == {"stop": 40210, "length": 118, "abort": 33}

    # State captured for next-frame rate math.
    assert state.monotonic == 100.0
    assert state.gen == 500000.0
    assert state.prefix_hits == 1100.0


def test_second_frame_rates(reading):
    # Previous frame two seconds earlier with lower cumulative counters.
    prev = RateState(
        monotonic=98.0,
        gen=499900.0,
        prompt=999000.0,
        preempt=10.0,
        prefix_hits=1000.0,
        prefix_queries=1800.0,
    )
    frame, _ = build_frame(
        reading,
        model="m",
        model_id="id",
        max_model_len=224800,
        prev=prev,
        scrape_monotonic=100.0,  # dt = 2.0s
    )
    assert frame["throughput"]["generation_tokens_per_s"] == pytest.approx(50.0)
    assert frame["throughput"]["prompt_tokens_per_s"] == pytest.approx(500.0)
    assert frame["engine"]["preemptions_per_s"] == pytest.approx(1.0)
    # Interval hit rate = delta_hits / delta_queries = 100 / 200 = 0.5.
    assert frame["cache"]["prefix_hit_rate"] == pytest.approx(0.5)


def test_counter_reset_yields_null_rate(reading):
    # Engine restarted: previous counters were HIGHER than current → refuse the
    # spurious negative-rate spike, emit null.
    prev = RateState(
        monotonic=98.0,
        gen=999999999.0,
        prompt=999999999.0,
        preempt=999.0,
        prefix_hits=None,
        prefix_queries=None,
    )
    frame, _ = build_frame(
        reading, model="m", model_id="id", max_model_len=1, prev=prev, scrape_monotonic=100.0
    )
    assert frame["throughput"]["generation_tokens_per_s"] is None
    assert frame["engine"]["preemptions_per_s"] is None


# --------------------------------------------------------------------------- #
# Missing metrics never crash → null
# --------------------------------------------------------------------------- #


def test_missing_metrics_map_to_null():
    # A near-empty engine (only one gauge exposed) must build a frame, not raise.
    r = VllmBackend().parse_metrics('vllm:num_requests_running{model_name="x"} 1.0\n')
    frame, _ = build_frame(
        r, model="x", model_id="id", max_model_len=None, prev=None, scrape_monotonic=0.0
    )
    assert frame["engine"]["num_requests_running"] == 1
    assert frame["engine"]["kv_cache_usage_perc"] is None
    assert frame["engine"]["kv_tokens_total"] is None
    assert frame["engine"]["kv_tokens_used"] is None
    assert frame["throughput"]["generation_tokens_total"] is None
    assert frame["latency"]["e2e_p50"] is None
    assert frame["latency"]["ttft_mean"] is None
    assert frame["cache"]["prefix_hit_rate_cumulative"] is None
    assert frame["finished"] == {"stop": None, "length": None, "abort": None}


def test_kv_used_null_when_info_missing():
    # Usage% present but cache_config_info absent → can't derive absolute tokens.
    r = VllmBackend().parse_metrics('vllm:kv_cache_usage_perc{model_name="x"} 0.87\n')
    frame, _ = build_frame(
        r, model="x", model_id="id", max_model_len=None, prev=None, scrape_monotonic=0.0
    )
    assert frame["engine"]["kv_cache_usage_perc"] == pytest.approx(0.87)
    assert frame["engine"]["kv_tokens_total"] is None
    assert frame["engine"]["kv_tokens_used"] is None


# --------------------------------------------------------------------------- #
# Issue #280: a data-parallel engine's per-rank KV series
# --------------------------------------------------------------------------- #

# vLLM >= 0.26 labels every per-rank series with ``engine="0".."N-1"``
# (PrometheusStatLogger in vllm/v1/metrics/loggers.py), so a data-parallel
# engine's single /metrics endpoint carries ONE kv_cache_usage_perc series per
# rank -- each a 0..1 fraction of THAT rank's pool -- plus one cache_config_info
# series per rank, each carrying that rank's block_size * num_gpu_blocks.
# Production (issue #280): 7 x RTX 5090, DP 7 x TP1, the panel showed 100% at
# "8.1M / 1.2M tokens" -- used ~6.75x the pool, physically impossible.
_DP_FRACTIONS = (0.97, 0.96, 0.97, 0.96, 0.97, 0.96, 0.96)  # sums to 6.75
DP7 = "".join(
    f'vllm:kv_cache_usage_perc{{model_name="qwen3.8-27b",engine="{i}"}} {f}\n'
    for i, f in enumerate(_DP_FRACTIONS)
) + "".join(
    'vllm:cache_config_info{block_size="16",num_gpu_blocks="75000",'
    f'num_cpu_blocks="0",cache_dtype="auto",engine="{i}"}} 1.0\n'
    for i in range(7)
)


def test_dp_frame_never_shows_used_exceeding_total():
    """Issue #280: raw per-rank gauges must reach the frame as an engine-wide
    fraction over the engine-wide pool, never as the raw sum over one rank's
    pool (which is how used outgrew total by the rank count)."""
    r = VllmBackend().parse_metrics(DP7)
    assert r is not None
    frame, _ = build_frame(
        r, model="qwen3.8-27b", model_id="id", max_model_len=None, prev=None, scrape_monotonic=0.0
    )
    eng = frame["engine"]
    # Engine-wide fraction = sum of per-rank fractions / rank count.
    assert eng["kv_cache_usage_perc"] == pytest.approx(sum(_DP_FRACTIONS) / 7)
    # Engine-wide pool = per-rank block_size * num_gpu_blocks * rank count.
    assert eng["kv_tokens_total"] == 7 * 16 * 75000
    # Engine-wide used = sum of per-rank fractions * per-rank pool.
    assert eng["kv_tokens_used"] == round(sum(_DP_FRACTIONS) * 16 * 75000)
    # The invariant the panel broke: used never exceeds the pool.
    assert eng["kv_tokens_used"] <= eng["kv_tokens_total"]


# --------------------------------------------------------------------------- #
# Shared TTL cache collapses concurrent scrapes to one fetch
# --------------------------------------------------------------------------- #


def test_metrics_cache_dedups_within_ttl():
    calls = {"n": 0}
    clock = {"t": 0.0}

    async def fake_fetch(host, port, at, backend):
        calls["n"] += 1
        # Sub-project C: the cache hands the backend to the fetch so the
        # metrics PATH and the metric DIALECT both come from it.
        assert backend.capabilities.name == "vllm"
        return ScrapeResult(metrics=EngineReading(), error=None, monotonic=at)

    backend = VllmBackend()
    cache = _MetricsCache(ttl=1.5, clock=lambda: clock["t"], fetch=fake_fetch)

    async def scenario():
        # Three hits inside the TTL window → one fetch.
        await cache.get("model-a", "127.0.0.1", 8001, backend)
        await cache.get("model-a", "127.0.0.1", 8001, backend)
        await cache.get("model-a", "127.0.0.1", 8001, backend)
        assert calls["n"] == 1
        # Advance past the TTL → a fresh fetch.
        clock["t"] = 2.0
        await cache.get("model-a", "127.0.0.1", 8001, backend)
        assert calls["n"] == 2
        # A different model_id keys a separate entry → its own fetch.
        await cache.get("model-b", "127.0.0.1", 8002, backend)
        assert calls["n"] == 3

    asyncio.run(scenario())
    assert cache.invocations == 3


def test_interval_seconds_clamp(monkeypatch):
    monkeypatch.delenv("VW_STATS_LIVE_INTERVAL_S", raising=False)
    assert _interval_seconds() == 2.0
    monkeypatch.setenv("VW_STATS_LIVE_INTERVAL_S", "0.1")
    assert _interval_seconds() == 0.5  # floored
    monkeypatch.setenv("VW_STATS_LIVE_INTERVAL_S", "5")
    assert _interval_seconds() == 5.0
    monkeypatch.setenv("VW_STATS_LIVE_INTERVAL_S", "garbage")
    assert _interval_seconds() == 2.0
