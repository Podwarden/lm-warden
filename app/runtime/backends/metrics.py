"""EngineReading -- one scrape, expressed in terms the frame understands.

The seam between a backend's metric NAMES and the dashboard's frame. A backend
turns its own exposition text into one of these; app/stats/live_engine.py turns
one of these into a frame, deriving per-second rates against the connection's
previous reading. Neither half needs to know the other's vocabulary.

THE ONE RULE
------------
Every field is a float or None, and None means THIS ENGINE DOES NOT REPORT IT.
Never 0. Never a placeholder. vLLM publishes latency histograms, a KV-usage
gauge and a preemption counter; llama.cpp publishes none of them, and publishes
a prompt-token counter that EXCLUDES cached tokens where vLLM's includes them.
Rendering an absent number as zero is how a dashboard tells an operator the
engine is idle when it is merely silent about that one thing. It is the same
failure mode design spec §9.3 forbids for a compatibility verdict: fluent output
that happens to be wrong, which sub-project E now calls ``unreliable`` (as
distinct from ``slower``, which is correct output at reduced performance). A
zero here is the metrics-layer version of that -- confidently precise, and
false.

Adding a field is additive: it defaults to None, every existing backend keeps
working, and the frame renders it absent until some backend fills it in.
"""

from __future__ import annotations

from dataclasses import dataclass

# (ascending [(le, cumulative_count)], sum, count) -- Prometheus histogram shape,
# exactly what app.stats.prometheus.Metrics.histogram returns.
Histogram = tuple[list[tuple[float, float]], float | None, float | None]


@dataclass(frozen=True)
class EngineReading:
    # --- instantaneous gauges -------------------------------------------
    requests_running: float | None = None
    requests_waiting: float | None = None
    waiting_capacity: float | None = None
    waiting_deferred: float | None = None
    kv_cache_usage_perc: float | None = None
    # Absolute KV capacity in tokens. vLLM does not publish this directly -- it
    # is block_size * num_gpu_blocks off the cache_config_info LABELS, which is
    # exactly the kind of dialect-specific derivation that belongs in a backend
    # rather than in build_frame().
    kv_tokens_total: float | None = None
    engine_sleep_state: float | None = None
    # --- cumulative counters (the stats layer differentiates these) ------
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    preemptions_total: float | None = None
    prefix_cache_hits: float | None = None
    prefix_cache_queries: float | None = None
    mm_cache_hits: float | None = None
    mm_cache_queries: float | None = None
    external_prefix_cache_hits: float | None = None
    external_prefix_cache_queries: float | None = None
    flops_per_gpu_total: float | None = None
    finished_stop: float | None = None
    finished_length: float | None = None
    finished_abort: float | None = None
    # --- histograms ------------------------------------------------------
    ttft_hist: Histogram | None = None
    itl_hist: Histogram | None = None
    tpot_hist: Histogram | None = None
    e2e_hist: Histogram | None = None
