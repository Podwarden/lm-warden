"""The llamacpp: dialect, against Task 2's real capture.

Three things are worth reading carefully, because none of them is a renaming and
all three are why the EngineReading seam exists at all:

1. llamacpp:prompt_tokens_total EXCLUDES cached tokens, where vLLM's INCLUDES
   them. The prefix-cache hit rate therefore has to be composed here --
   queries = processed + cached -- rather than read off a series. Getting this
   wrong makes the cache panel read >100%.
2. There is no KV-usage gauge, no cache_config_info, no preemption counter, no
   latency histogram and no per-finished_reason split. Those fields stay None,
   and None must reach the frontend as an em dash, never as 0.
3. We do NOT read llamacpp:prompt_tokens_seconds or predicted_tokens_seconds,
   even though they are labelled as throughput. Upstream resets their bucket on
   every scrape (server-context.cpp:4657-4659) -- verified by hand at capture
   time, see tests/fixtures/llamacpp/README.md property 1 -- so a second scraper
   silently halves them. The counters are cumulative and the stats layer already
   differentiates counters, so the rates come from there instead.
"""

from __future__ import annotations

from pathlib import Path

from app.runtime.backends.llamacpp import LlamaCppBackend

FIX = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "llamacpp"
BODY = (FIX / "metrics.txt").read_text()
BACKEND = LlamaCppBackend()


def test_maps_the_gauges():
    r = BACKEND.parse_metrics(BODY)
    assert r.requests_running is not None  # llamacpp:requests_processing
    assert r.requests_waiting is not None  # llamacpp:requests_deferred


def test_maps_the_token_counters():
    r = BACKEND.parse_metrics(BODY)
    assert r.prompt_tokens_total is not None  # llamacpp:prompt_tokens_total
    assert r.generation_tokens_total is not None  # llamacpp:tokens_predicted_total


def test_the_captured_numbers_are_the_ones_that_come_out():
    """The capture drove exactly one 16-token completion off a 2-token prompt,
    so these are hand-checkable rather than merely non-None."""
    r = BACKEND.parse_metrics(BODY)
    assert r.prompt_tokens_total == 2.0
    assert r.generation_tokens_total == 16.0
    assert r.requests_running == 0.0
    assert r.requests_waiting == 0.0


def test_prefix_cache_queries_are_composed_not_read():
    """queries = processed + cached, because llama.cpp's prompt_tokens_total
    excludes cached tokens. Read as-is, hits/queries would exceed 1."""
    from app.stats.prometheus import Metrics, parse_prometheus

    m = Metrics(parse_prometheus(BODY))
    processed = m.value("llamacpp:prompt_tokens_total") or 0.0
    cached = m.value("llamacpp:prompt_tokens_cached_total") or 0.0
    r = BACKEND.parse_metrics(BODY)
    assert r.prefix_cache_hits == cached
    assert r.prefix_cache_queries == processed + cached
    if r.prefix_cache_queries:
        assert 0.0 <= r.prefix_cache_hits / r.prefix_cache_queries <= 1.0


def test_a_cache_heavy_scrape_never_exceeds_one_hundred_percent():
    """The failure the composition prevents, stated as a case rather than as a
    comment: with more cached than processed tokens, reading
    hits/prompt_tokens_total directly would give 3.0 -- a 300% hit rate."""
    body = "llamacpp:prompt_tokens_total 100\n" "llamacpp:prompt_tokens_cached_total 300\n"
    r = BACKEND.parse_metrics(body)
    assert r.prefix_cache_hits == 300.0
    assert r.prefix_cache_queries == 400.0
    assert r.prefix_cache_hits / r.prefix_cache_queries == 0.75


def test_unreported_concepts_are_none_not_zero():
    r = BACKEND.parse_metrics(BODY)
    for absent in (
        "kv_cache_usage_perc",
        "kv_tokens_total",
        "engine_sleep_state",
        "preemptions_total",
        "waiting_capacity",
        "waiting_deferred",
        "mm_cache_hits",
        "external_prefix_cache_hits",
        "flops_per_gpu_total",
        "finished_stop",
        "finished_length",
        "finished_abort",
        "ttft_hist",
        "itl_hist",
        "tpot_hist",
        "e2e_hist",
    ):
        assert getattr(r, absent) is None, f"{absent} should be absent, not 0"


def test_reset_on_scrape_gauges_are_not_used_for_throughput():
    """Guard against a future 'optimisation' that reads the convenient-looking
    per-second gauges. They are averaged over the inter-scrape window and RESET
    each scrape, so two scrapers halve them."""
    src = (
        Path(__file__).resolve().parents[4]
        / "app"
        / "runtime"
        / "backends"
        / "llamacpp"
        / "metrics.py"
    ).read_text()
    assert "llamacpp:prompt_tokens_seconds" not in src
    assert "llamacpp:predicted_tokens_seconds" not in src


def test_empty_body_is_none():
    assert BACKEND.parse_metrics("") is None


def test_a_501_body_does_not_raise():
    """Without --metrics, llama-server answers 501 with a JSON error envelope.
    args.py always passes --metrics, but a hand-edited extra_args could still
    produce this, and it must degrade to 'no metrics', not to a stack trace."""
    body = '{"error":{"message":"This server does not support metrics endpoint.",' '"code":501}}'
    r = BACKEND.parse_metrics(body)
    assert r is None or r.requests_running is None


def test_the_second_scrape_reads_the_same_counters():
    """The property the whole rate derivation rests on, checked through the
    parser rather than only against the raw text."""
    a = BACKEND.parse_metrics(BODY)
    b = BACKEND.parse_metrics((FIX / "metrics_second_scrape.txt").read_text())
    assert a.prompt_tokens_total == b.prompt_tokens_total
    assert a.generation_tokens_total == b.generation_tokens_total
    assert a.prefix_cache_queries == b.prefix_cache_queries
