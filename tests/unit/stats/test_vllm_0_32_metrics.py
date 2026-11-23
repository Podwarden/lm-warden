"""vLLM 0.32's exposition, read the way the Stats page needs it.

The fixture is the exposition vLLM 0.32.0's own PrometheusStatLogger writes
(names, label sets and label values are real; see its header). Two readings
were wrong against it:

* ``vllm:engine_sleep_state`` is one gauge PER STATE (sleep_state="awake" /
  "weights_offloaded" / "discard_all"), 1 on the current one. Summing the
  series read 1 for an AWAKE engine, which the page showed as "sleep 1".
* the lifetime prefix-cache rate counted local hits only. With a KV connector
  the reuse arrives as external hits while local hits stay 0, so the page said
  0 % beside a 28 % hit. 0.32 also exports prompt_tokens_cached (local +
  external) over prompt_tokens; the rate is null, never 0, when the engine
  exports neither family or has prefix caching off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.runtime.backends.vllm import VllmBackend
from app.stats.live_engine import build_frame

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_0_32_0.txt"

_AWAKE = (
    'sleep_state="awake"} 1.0',
    'sleep_state="weights_offloaded"} 0.0',
    'sleep_state="discard_all"} 0.0',
)


def _text() -> str:
    return FIXTURE.read_text()


def _sleep(text: str, awake: str, offloaded: str, discard: str) -> str:
    for old, new in zip(
        _AWAKE,
        (
            f'sleep_state="awake"}} {awake}',
            f'sleep_state="weights_offloaded"}} {offloaded}',
            f'sleep_state="discard_all"}} {discard}',
        ),
        strict=True,
    ):
        assert old in text
        text = text.replace(old, new)
    return text


def _frame(text: str) -> dict:
    reading = VllmBackend().parse_metrics(text)
    frame, _ = build_frame(
        reading,
        model="m",
        model_id="mid",
        max_model_len=262144,
        prev=None,
        scrape_monotonic=1.0,
    )
    return frame


def _set(text: str, series: str, value: str) -> str:
    """Replace the value of the one sample line starting with ``series``."""
    out = []
    hit = 0
    for line in text.splitlines():
        if line.startswith(series):
            line = line.rsplit(" ", 1)[0] + " " + value
            hit += 1
        out.append(line)
    assert hit == 1, series
    return "\n".join(out) + "\n"


def _drop(text: str, *prefixes: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith(prefixes)) + "\n"


# ---- sleep state ------------------------------------------------------------


def test_an_awake_0_32_engine_is_not_sleeping():
    eng = _frame(_text())["engine"]
    assert eng["engine_sleeping"] is False
    assert eng["engine_sleep_state"] == 0  # the level: 0 = awake, never the series sum


def test_sleep_level_1_is_weights_offloaded():
    eng = _frame(_sleep(_text(), "0.0", "1.0", "0.0"))["engine"]
    assert eng["engine_sleeping"] is True
    assert eng["engine_sleep_state"] == 1


def test_sleep_level_2_is_discard_all():
    eng = _frame(_sleep(_text(), "0.0", "0.0", "1.0"))["engine"]
    assert eng["engine_sleeping"] is True
    assert eng["engine_sleep_state"] == 2


def test_data_parallel_ranks_all_awake_is_not_sleeping():
    text = _text()
    rank1 = [
        line.replace('engine="0"', 'engine="1"')
        for line in text.splitlines()
        if line.startswith("vllm:engine_sleep_state{")
    ]
    eng = _frame(text + "\n".join(rank1) + "\n")["engine"]
    assert eng["engine_sleeping"] is False
    assert eng["engine_sleep_state"] == 0


def test_an_unlabelled_older_gauge_keeps_its_level_meaning():
    text = _drop(_text(), "vllm:engine_sleep_state{")
    eng = _frame(text + 'vllm:engine_sleep_state{model_name="m"} 0.0\n')["engine"]
    assert eng["engine_sleeping"] is False and eng["engine_sleep_state"] == 0
    eng = _frame(text + 'vllm:engine_sleep_state{model_name="m"} 1.0\n')["engine"]
    assert eng["engine_sleeping"] is True and eng["engine_sleep_state"] == 1


def test_no_sleep_gauge_is_unknown_not_awake():
    eng = _frame(_drop(_text(), "vllm:engine_sleep_state{"))["engine"]
    assert eng["engine_sleeping"] is None
    assert eng["engine_sleep_state"] is None


# ---- prefix cache -----------------------------------------------------------


def test_lifetime_prefix_hit_rate_from_the_0_32_fixture():
    cache = _frame(_text())["cache"]
    # prompt_tokens_cached / prompt_tokens = 83952 / 207792
    assert cache["prefix_cache_hit_rate"] == pytest.approx(83952 / 207792)
    assert cache["prefix_hit_rate_cumulative"] == pytest.approx(83952 / 207792)


def test_external_connector_hits_count_when_local_hits_are_zero():
    text = _set(_text(), "vllm:prefix_cache_hits_total{", "0.0")
    text = _set(text, "vllm:external_prefix_cache_queries_total{", "207792.0")
    text = _set(text, "vllm:external_prefix_cache_hits_total{", "58182.0")
    text = _set(text, "vllm:prompt_tokens_cached_total{", "58182.0")
    cache = _frame(text)["cache"]
    assert cache["prefix_cache_hit_rate"] == pytest.approx(58182 / 207792)
    assert cache["prefix_cache_hit_rate"] > 0.27


def test_falls_back_to_local_plus_external_hits_without_prompt_tokens_cached():
    text = _drop(_text(), "vllm:prompt_tokens_cached_total{")
    text = _set(text, "vllm:prefix_cache_hits_total{", "0.0")
    text = _set(text, "vllm:external_prefix_cache_hits_total{", "58182.0")
    cache = _frame(text)["cache"]
    assert cache["prefix_cache_hit_rate"] == pytest.approx(58182 / 207792)


def test_hit_rate_is_null_when_no_prefix_family_is_exported():
    text = _drop(
        _text(),
        "vllm:prompt_tokens_cached",
        "vllm:prefix_cache_",
        "vllm:external_prefix_cache_",
    )
    assert _frame(text)["cache"]["prefix_cache_hit_rate"] is None


def test_hit_rate_is_null_with_prefix_caching_off():
    text = _text().replace('enable_prefix_caching="True"', 'enable_prefix_caching="False"')
    assert _frame(text)["cache"]["prefix_cache_hit_rate"] is None


def test_hit_rate_is_null_before_any_prompt():
    text = _set(_text(), "vllm:prompt_tokens_total{", "0.0")
    text = _set(text, "vllm:prefix_cache_queries_total{", "0.0")
    assert _frame(text)["cache"]["prefix_cache_hit_rate"] is None
