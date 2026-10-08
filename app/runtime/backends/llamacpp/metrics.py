"""llama.cpp's Prometheus dialect.

Every ``llamacpp:`` string in this repository lives here. Note the prefix uses a
COLON, matching vLLM's convention and upstream's own emission
(tools/server/server-task.cpp) -- it is unusual (Prometheus reserves the colon
for recording rules) but it is what the server prints, so it is what we match.

WHAT IS NOT A RENAMING, AND WHY THIS FILE EXISTS
------------------------------------------------
* ``prompt_tokens_total`` EXCLUDES cached tokens; vLLM's includes them. The
  prefix-cache query count is therefore composed here as processed + cached.
  Read as-is, the dashboard's hit rate would exceed 100%.
* llama.cpp publishes no KV-usage gauge, no cache_config_info, no preemption
  counter, no latency histograms and no per-finished_reason split. Those stay
  None, which the frame renders as absent. That is the honest answer, and design
  spec §9.3 requires it be visibly different from zero.
* The two ``*_tokens_seconds`` GAUGES look like exactly the throughput numbers
  the dashboard wants, and they are a trap: upstream averages them over the
  window between scrapes and RESETS the bucket on every scrape
  (server-context.cpp:4657-4659), so an external Prometheus pointed at the same
  engine silently halves ours. Verified by hand at capture time -- the two
  scrapes in tests/fixtures/llamacpp/ differ in exactly those two lines and
  nowhere else. We differentiate the cumulative counters instead, which is what
  the stats layer already does for vLLM. A unit test asserts those two names
  never appear in this file.

Reported by llama.cpp and deliberately NOT mapped, because the frame has no home
for them and widening the schema is a separate product decision with a UI cost:
prompt_seconds_total, tokens_predicted_seconds_total, n_decode_total,
n_tokens_max, n_busy_slots_per_decode, and the speculative-decoding counters.
Doing it in the same change that introduces a second dialect would make a
regression in one indistinguishable from a bug in the other.
"""

from __future__ import annotations

from app.runtime.backends.metrics import EngineReading
from app.stats.prometheus import Metrics, parse_prometheus


def read(body: str) -> EngineReading | None:
    """Parse llama-server exposition text into an EngineReading, or None."""
    if not body or not body.strip():
        return None
    m = Metrics(parse_prometheus(body))

    processed = m.value("llamacpp:prompt_tokens_total")
    cached = m.value("llamacpp:prompt_tokens_cached_total")
    queries = None
    if processed is not None or cached is not None:
        queries = (processed or 0.0) + (cached or 0.0)

    return EngineReading(
        requests_running=m.value("llamacpp:requests_processing"),
        requests_waiting=m.value("llamacpp:requests_deferred"),
        prompt_tokens_total=processed,
        generation_tokens_total=m.value("llamacpp:tokens_predicted_total"),
        prefix_cache_hits=cached,
        prefix_cache_queries=queries,
        # Everything else defaults to None: llama.cpp does not report it, and
        # the frame must render that as absent rather than as idle.
    )
