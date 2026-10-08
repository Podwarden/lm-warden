"""The Prometheus text parser, as a module of its own.

It used to sit inside app/stats/live_engine.py, wedged between the SSE route and
the vLLM metric-name table. It is dialect-independent -- it does not know what
``vllm:`` or ``llamacpp:`` mean -- so sub-project C lifts it out and both
backends import it. The tests below are the ones that prove it is generic: every
assertion uses a metric name from neither dialect.
"""

from __future__ import annotations

import math

from app.stats.prometheus import (
    Metrics,
    PromSample,
    hist_mean,
    hist_quantile,
    parse_prometheus,
)


def test_parses_a_bare_sample():
    assert parse_prometheus("foo_total 1.5\n") == [
        PromSample(name="foo_total", labels={}, value=1.5)
    ]


def test_parses_labels_and_skips_comments():
    text = '# HELP foo x\n# TYPE foo gauge\nfoo{a="1",b="two"} 3\n'
    assert parse_prometheus(text)[0].labels == {"a": "1", "b": "two"}


def test_malformed_line_is_dropped_not_raised():
    """A partial scrape must never kill the SSE stream."""
    assert parse_prometheus("this is not prometheus\nfoo 1\n") == [
        PromSample(name="foo", labels={}, value=1.0)
    ]


def test_value_sums_across_label_series():
    m = Metrics(parse_prometheus('foo{i="0"} 1\nfoo{i="1"} 2\n'))
    assert m.value("foo") == 3.0


def test_value_filters_by_label():
    m = Metrics(parse_prometheus('foo{r="a"} 1\nfoo{r="b"} 2\n'))
    assert m.value("foo", r="b") == 2.0


def test_missing_name_is_none_not_zero():
    assert Metrics([]).value("nope") is None


def test_value_any_takes_the_first_present_spelling():
    m = Metrics(parse_prometheus("new_name 7\n"))
    assert m.value_any("old_name", "new_name") == 7.0


def test_info_returns_the_first_series_labels():
    m = Metrics(parse_prometheus('foo_info{block_size="16"} 1\n'))
    assert m.info("foo_info") == {"block_size": "16"}


def test_histogram_and_quantile():
    text = 'h_bucket{le="1"} 1\nh_bucket{le="2"} 3\nh_bucket{le="+Inf"} 4\n' "h_sum 6\nh_count 4\n"
    hist = Metrics(parse_prometheus(text)).histogram("h")
    assert hist is not None
    buckets, s, c = hist
    assert (s, c) == (6.0, 4.0)
    assert hist_mean(s, c) == 1.5
    assert hist_quantile(buckets, 0.5) == 1.5


def test_absent_histogram_is_none():
    assert Metrics([]).histogram("h") is None


def test_quantile_of_empty_histogram_is_none():
    assert hist_quantile([], 0.5) is None
    assert hist_quantile([(1.0, 0.0), (math.inf, 0.0)], 0.5) is None
