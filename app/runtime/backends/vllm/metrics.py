"""vLLM's Prometheus dialect: the names, and the derivations they need.

Every ``vllm:`` string in this repository lives in this file. They came out of
app/stats/live_engine.py's build_frame(), which named 31 of them inline while
also doing rate math and SSE framing; sub-project C split the three jobs once a
second dialect existed to justify the seam.

Two things here are NOT renamings, and they are why a name-lookup table would
not have been enough:

* ``kv_tokens_total`` is composed from the LABELS of vllm:cache_config_info
  (block_size * num_gpu_blocks), not read from a series.
* several counters have two spellings across vLLM versions -- 0.25.1 renamed
  gpu_cache_usage_perc to kv_cache_usage_perc and appended _total to others --
  so the lookups go through Metrics.value_any and take whichever exists. That
  version-drift tolerance is why an absent metric reads None rather than
  crashing the stream, and it must survive any tidying.
* Data-parallel engines (vLLM >= 0.26) label every per-rank series
  engine="0".."N-1" and serve them all from ONE /metrics endpoint. The KV
  usage gauge is then a per-rank FRACTION, so Metrics.value's additive sum
  runs to N instead of 1, and cache_config_info carries each rank's OWN pool.
  Both are normalised here by the engine count the exposition itself carries:
  the frame gets the engine-wide fraction over the engine-wide pool, and
  kv_tokens_used can never exceed kv_tokens_total (issue #280).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.runtime.backends.metrics import EngineReading
from app.stats.prometheus import Metrics, parse_prometheus

#: The KV usage gauge under both spellings, in the order read() prefers them,
#: so _engine_count and the value lookup can never look at different series.
_KV_USAGE_GAUGES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def _engine_count(m: Metrics) -> int:
    """How many engine (data-parallel rank) series the KV gauge exposes.

    vLLM >= 0.26 labels each rank's series ``engine="0".."N-1"``; older
    single-engine expositions carry no engine label at all and read 1.
    """
    for name in _KV_USAGE_GAUGES:
        engines = m.label_values(name, "engine")
        if engines:
            return len(engines)
    return 1


def _kv_tokens_total(m: Metrics, engines: int) -> float | None:
    """The engine-wide KV pool: block_size * num_gpu_blocks PER RANK, times
    the rank count. The info series carries the first rank's labels, which on
    a data-parallel engine is every rank's (they share one cache config)."""
    info = m.info("vllm:cache_config_info")
    if info is None:
        return None
    try:
        return float(int(info["block_size"]) * int(info["num_gpu_blocks"]) * engines)
    except (KeyError, ValueError):
        return None


def read(body: str) -> EngineReading | None:
    """Parse vLLM exposition text into an EngineReading, or None if unusable."""
    if not body or not body.strip():
        return None
    m = Metrics(parse_prometheus(body))
    engines = _engine_count(m)
    # A data-parallel engine's /metrics carries the gauge once per rank, each
    # a 0..1 fraction of that rank's pool: the additive sum is the engine-wide
    # fraction only after dividing by the rank count, and the pool is per-rank
    # (issue #280). A single engine (no engine labels) divides by 1 and reads
    # exactly as it always did.
    kv_usage = m.value_any(*_KV_USAGE_GAUGES)
    return EngineReading(
        requests_running=m.value("vllm:num_requests_running"),
        requests_waiting=m.value("vllm:num_requests_waiting"),
        waiting_capacity=m.value("vllm:num_requests_waiting_by_reason", reason="capacity"),
        waiting_deferred=m.value("vllm:num_requests_waiting_by_reason", reason="deferred"),
        kv_cache_usage_perc=kv_usage / engines if kv_usage is not None else None,
        kv_tokens_total=_kv_tokens_total(m, engines),
        engine_sleep_state=m.value("vllm:engine_sleep_state"),
        prompt_tokens_total=m.value("vllm:prompt_tokens_total"),
        generation_tokens_total=m.value("vllm:generation_tokens_total"),
        preemptions_total=m.value("vllm:num_preemptions_total"),
        prefix_cache_hits=m.value_any("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
        prefix_cache_queries=m.value_any(
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"
        ),
        mm_cache_hits=m.value_any("vllm:mm_cache_hits_total", "vllm:mm_cache_hits"),
        mm_cache_queries=m.value_any("vllm:mm_cache_queries_total", "vllm:mm_cache_queries"),
        external_prefix_cache_hits=m.value_any(
            "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_hits"
        ),
        external_prefix_cache_queries=m.value_any(
            "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_queries",
        ),
        flops_per_gpu_total=m.value("vllm:estimated_flops_per_gpu_total"),
        finished_stop=m.value("vllm:request_success_total", finished_reason="stop"),
        finished_length=m.value("vllm:request_success_total", finished_reason="length"),
        finished_abort=m.value("vllm:request_success_total", finished_reason="abort"),
        ttft_hist=m.histogram("vllm:time_to_first_token_seconds"),
        itl_hist=m.histogram("vllm:inter_token_latency_seconds"),
        tpot_hist=m.histogram("vllm:time_per_output_token_seconds"),
        e2e_hist=m.histogram("vllm:e2e_request_latency_seconds"),
    )


@dataclass(frozen=True)
class RankReading:
    """One data-parallel rank's slice of the exposition (``engine="<rank>"``)."""

    requests_running: float | None = None
    requests_waiting: float | None = None
    kv_cache_usage_perc: float | None = None
    prefix_cache_hits: float | None = None
    prefix_cache_queries: float | None = None


def read_ranks(body: str) -> dict[int, RankReading]:
    """Per-rank readings from the ``engine="i"`` label series (vLLM >= 0.26).

    ``{}`` when the exposition carries no engine-labelled series (a single
    engine, or vLLM < 0.26) or is unparsable. Unlabelled samples are ignored.
    Unlike read(), nothing is summed across ranks: each rank is filtered by
    its own label value.
    """
    if not body or not body.strip():
        return {}
    m = Metrics(parse_prometheus(body))
    ids: set[int] = set()
    for name in (
        *_KV_USAGE_GAUGES,
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    ):
        for label in m.label_values(name, "engine"):
            if label.isdigit():
                ids.add(int(label))
    out: dict[int, RankReading] = {}
    for i in sorted(ids):
        e = str(i)
        out[i] = RankReading(
            requests_running=m.value("vllm:num_requests_running", engine=e),
            requests_waiting=m.value("vllm:num_requests_waiting", engine=e),
            kv_cache_usage_perc=m.value_any(*_KV_USAGE_GAUGES, engine=e),
            prefix_cache_hits=m.value_any(
                "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits", engine=e
            ),
            prefix_cache_queries=m.value_any(
                "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries", engine=e
            ),
        )
    return out
