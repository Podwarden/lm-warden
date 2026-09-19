"""app/tokens/series -- the pure half of GET /api/tokens/{id}/series.

The HTTP contract (queries, 404/422, chain) is in test_series_api.py.
"""

import calendar

import pytest

from app.tokens import series

DAY = 86400


@pytest.mark.parametrize(
    ("span_s", "width"),
    [
        (3600, 1),          # 1h  -- the stats page's table
        (6 * 3600, 1),      # 6h
        (DAY, 5),           # 24h
        (7 * DAY, 30),      # 7d
        (30 * DAY, 120),
        (90 * DAY, 360),
        (366 * DAY, 10080),
    ],
)
def test_ladder_pick(span_s, width):
    assert series.pick_bin_minutes(span_s) == width


def test_ladder_honours_a_smaller_max_bins():
    # 7d = 10080 min: 60-min bins would be 168 > 100, 120-min bins are 84.
    assert series.pick_bin_minutes(7 * DAY, max_bins=100) == 120


def test_ladder_falls_back_to_the_widest_rung():
    assert series.pick_bin_minutes(366 * DAY, max_bins=1) == 10080


def test_every_pick_fits_max_bins_up_to_the_longest_span():
    for days in range(1, 367):
        width = series.pick_bin_minutes(days * DAY)
        assert -(-days * 1440 // width) <= series.MAX_BINS, days


def test_validate_window_accepts_a_normal_window():
    assert series.validate_window(1000.0, 4600.0, now_s=5000.0) == 4600.0


def test_validate_window_clamps_a_future_to_instead_of_refusing():
    # Ruled in 2026-09-18 (clock-skew review): a client clock that runs fast
    # must not turn a "last hour" poll into an empty chart -- `to` beyond the
    # server's own clock is clamped to it rather than rejected.
    assert series.validate_window(1000.0, 5000.0 + 3600, now_s=5000.0) == 5000.0


@pytest.mark.parametrize(
    ("from_s", "to_s", "message"),
    [
        (4600.0, 1000.0, "before"),                          # reversed
        (1000.0, 1000.0, "before"),                          # empty
        (5000.0 + 10, 5000.0 + 3600, "after the server's current time"),  # 'from' is ALSO future
        (5000.0, 5000.0 + 3600, "after the server's current time"),       # 'from' == now: empty
        (5000.0 + 3600, 5000.0 + 10, "before"),               # reversed AND future: the order wins
        (5000.0 - 366 * DAY - 1, 5000.0, "366 days"),        # too long
        (5000.0 - 366 * DAY - 1, 5000.0 + 3600, "366 days"),  # too long even after clamping 'to'
    ],
)
def test_validate_window_refuses(from_s, to_s, message):
    with pytest.raises(series.SeriesWindowError, match=message):
        series.validate_window(from_s, to_s, now_s=5000.0)


def test_validate_window_edges_are_inclusive():
    assert series.validate_window(1000.0, 5000.0, now_s=5000.0) == 5000.0              # to == now, unclamped
    assert series.validate_window(5000.0 - 366 * DAY, 5000.0, now_s=5000.0) == 5000.0  # exactly 366d


def test_minute_window_includes_partial_minutes():
    assert series.minute_window(120.0, 3601.0) == (2, 61)
    assert series.minute_window(150.0, 3600.0) == (2, 60)  # aligned end is exclusive


def test_day_bins_start_at_utc_midnight():
    t = calendar.timegm((2026, 9, 17, 13, 45, 0))
    key = series.bin_of_minute(t // 60, 1440)
    assert key * 60 == calendar.timegm((2026, 9, 17, 0, 0, 0))


def test_chain_token_ids():
    chain = ["a", "b", "c"]
    assert series.chain_token_ids(chain, "b", include_earlier=True) == ["a", "b"]
    assert series.chain_token_ids(chain, "b", include_earlier=False) == ["b"]
    assert series.chain_token_ids(chain, "a", include_earlier=True) == ["a"]
    assert series.chain_token_ids(chain, "zz", include_earlier=True) == ["zz"]


def test_timing_bins_percentiles_and_missing_values():
    rows = [
        (6000.0, 0.1, 1.0, 10.0),
        (6010.0, 0.2, 2.0, 20.0),
        (6020.0, 0.3, 3.0, 30.0),
        (6030.0, None, None, 40.0),  # not measured: counts in n only
    ]
    got = series.timing_bins(rows, 1)
    assert list(got) == [100]
    b = got[100]
    assert b["n"] == 4
    assert b["queue_p50"] == pytest.approx(0.2)
    assert b["queue_p95"] == pytest.approx(0.29)
    assert b["ttft_p50"] == pytest.approx(2.0)
    assert b["ttft_p95"] == pytest.approx(2.9)
    assert b["duration_p50"] == pytest.approx(25.0)
    assert b["duration_p95"] == pytest.approx(38.5)


def test_timing_bins_null_when_nothing_measured():
    b = series.timing_bins([(6000.0, None, None, None)], 1)[100]
    assert b == {
        "n": 1,
        "queue_p50": None, "queue_p95": None,
        "ttft_p50": None, "ttft_p95": None,
        "duration_p50": None, "duration_p95": None,
    }


def test_sparse_minutes_average_to_sum_over_width():
    # Two minutes with traffic inside a 5-minute bin. Mean of the rows present
    # would be 7.5 requests/min; the bin really averaged 15 / 5 = 3.
    out = series.build_series(
        token_ids=["t"], from_minute=100, to_minute=110, bin_minutes=5,
        count_rows=[(100, 15, 1500, 150, 1000, 100)],
        timing_rows=[], timing_total=0, timing_stride=1, latency_since=None,
    )
    (b,) = out["bins"]
    assert b["requests_per_min"] == 3.0
    assert b["prompt_per_min"] == 300.0
    assert b["completion_per_min"] == 30.0
    assert (b["peak_prompt"], b["peak_completion"], b["requests"]) == (1000, 100, 15)
    assert b["n"] == 0 and b["ttft_p50"] is None


def test_build_series_shape_and_merge():
    out = series.build_series(
        token_ids=["a", "b"], from_minute=100, to_minute=120, bin_minutes=5,
        count_rows=[(100, 2, 20, 4, 10, 2)],
        timing_rows=[(110 * 60 + 5.0, 0.5, 1.0, 2.0)],  # a bin with timings only
        timing_total=1, timing_stride=1,
        latency_since=1234.5,
    )
    assert list(out) == [
        "token_ids", "from_minute", "to_minute", "bin_minutes",
        "latency_since", "timing_sample", "totals", "bins",
        "by_model_since", "by_model", "model_bins",
    ]
    assert out["timing_sample"] == {"total": 1, "used": 1, "stride": 1}
    assert out["token_ids"] == ["a", "b"]
    assert (out["from_minute"], out["to_minute"], out["bin_minutes"]) == (100, 120, 5)
    assert out["latency_since"] == 1234.5
    assert out["totals"] == {"requests": 2, "prompt_tokens": 20, "completion_tokens": 4}
    assert [b["minute"] for b in out["bins"]] == [100, 110]
    assert list(out["bins"][0]) == [
        "minute", "requests_per_min", "prompt_per_min", "completion_per_min",
        "peak_prompt", "peak_completion", "requests", "n",
        "queue_p50", "queue_p95", "ttft_p50", "ttft_p95", "duration_p50", "duration_p95",
    ]
    timings_only = out["bins"][1]
    assert (timings_only["requests"], timings_only["requests_per_min"]) == (0, 0.0)
    assert timings_only["n"] == 1 and timings_only["ttft_p50"] == 1.0


def test_timing_sample_reports_a_stride_and_n_counts_sampled_rows():
    # 3 sampled rows standing in for 6 (every 2nd): n is what was read.
    rows = [(6000.0, 0.1, 1.0, 2.0), (6010.0, 0.1, 1.0, 2.0), (6020.0, 0.1, 1.0, 2.0)]
    out = series.build_series(
        token_ids=["t"], from_minute=100, to_minute=101, bin_minutes=1,
        count_rows=[(100, 6, 60, 6, 10, 1)],
        timing_rows=rows, timing_total=6, timing_stride=2, latency_since=None,
    )
    assert out["timing_sample"] == {"total": 6, "used": 3, "stride": 2}
    (b,) = out["bins"]
    assert b["n"] == 3
    assert b["requests"] == 6  # counts stay exact


# ---- per model (0036) --------------------------------------------------------


_DESC_A1 = (
    '{"backend":"vllm","dtype":"auto","engine_channel":"stable",'
    '"engine_image":"vllm/vllm-openai:v0.11.0","engine_vllm_version":"0.11.0",'
    '"extra_args":["--foo"],"hf_repo":"org/a","hf_revision":"a1b2c3d4e5",'
    '"max_model_len":32768,"quantization":"awq"}'
)


def test_by_model_rows_group_variants_under_their_model():
    out = series.by_model_rows([
        ("id-a", "model-a", "va1", _DESC_A1, 1000.0, 2, 400, 100),
        ("id-b", "gone-id", "vb1", None, None, 1, 200, 50),
        ("id-a", "model-a", "va2", '{"backend":"llamacpp"}', 2000.0, 1, 200, 50),
    ])
    a, b = out
    assert {k: v for k, v in a.items() if k != "variants"} == {
        "model_id": "id-a", "model": "model-a", "requests": 3, "prompt_tokens": 600,
        "completion_tokens": 150, "total_tokens": 750, "share": 0.75,
    }
    assert list(a)[-1] == "variants"
    assert [v["variant_id"] for v in a["variants"]] == ["va1", "va2"]  # busiest first
    assert a["variants"][0] == {
        "variant_id": "va1", "backend": "vllm", "engine_channel": "stable",
        "engine_vllm_version": "0.11.0", "engine_version": None, "quantization": "awq",
        "dtype": "auto", "hf_revision": "a1b2c3d4e5", "hf_commit": None, "max_model_len": 32768,
        "engine_image_tag": "v0.11.0", "first_seen": 1000.0,
        "requests": 2, "prompt_tokens": 400, "completion_tokens": 100,
        "total_tokens": 500, "share": 0.5,
    }
    assert a["variants"][1]["backend"] == "llamacpp"
    assert a["variants"][1]["share"] == 0.25
    # A variant with no model_variants row still counts; its summary is empty.
    (vb,) = b["variants"]
    assert (vb["backend"], vb["first_seen"], vb["total_tokens"]) == (None, None, 250)
    assert b["share"] == 0.25
    # Extra args and the repo split variants apart but are not summarised.
    assert "extra_args" not in a["variants"][0] and "hf_repo" not in a["variants"][0]


def test_the_summary_prefers_what_the_launch_actually_ran():
    desc = (
        '{"backend":"vllm","engine_image":null,"engine_image_used":"vllm/vllm-openai:v0.10.1",'
        '"engine_version":null,"hf_revision":"main",'
        '"hf_commit":"a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"}'
    )
    (m,) = series.by_model_rows([("id-a", "a", "v", desc, 1.0, 1, 1, 1)])
    (v,) = m["variants"]
    assert (v["engine_image_tag"], v["hf_revision"], v["hf_commit"]) == (
        "v0.10.1", "main", "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    )
    baked = '{"backend":"llamacpp","engine_version":"b10731"}'
    (m,) = series.by_model_rows([("id-a", "a", "v", baked, 1.0, 1, 1, 1)])
    assert (m["variants"][0]["engine_version"], m["variants"][0]["engine_image_tag"]) == (
        "b10731", None,
    )


def test_by_model_rows_share_is_zero_when_nothing_was_counted():
    # Requests that carried no tokens must not divide by zero.
    (only,) = series.by_model_rows([("id-a", "a", "v", None, None, 2, 0, 0)])
    assert only["share"] == 0.0 and only["variants"][0]["share"] == 0.0
    assert series.by_model_rows([]) == []


@pytest.mark.parametrize(("image", "tag"), [
    (None, None),
    ("vllm/vllm-openai:v0.11.0", "v0.11.0"),
    ("registry.local:5000/vllm/vllm-openai:nightly", "nightly"),
    ("registry.local:5000/vllm/vllm-openai", "latest"),
    ("vllm/vllm-openai@sha256:0123456789abcdef0123", "sha256:0123456789ab"),
])
def test_image_tag(image, tag):
    assert series.image_tag(image) == tag


def test_model_bins_are_sparse_and_sum_over_width():
    out = series.model_bins(
        [(100, "id-a", 50, 10), (100, "id-b", 5, 0), (110, "id-b", 25, 5)], 5,
    )
    assert out == [
        {"minute": 100, "models": {
            "id-a": {"prompt_per_min": 10.0, "completion_per_min": 2.0},
            "id-b": {"prompt_per_min": 1.0, "completion_per_min": 0.0},
        }},
        # 105 had no rows: absent. 110 carries only the model that had traffic.
        {"minute": 110, "models": {
            "id-b": {"prompt_per_min": 5.0, "completion_per_min": 1.0},
        }},
    ]


def test_build_series_defaults_to_no_per_model_split():
    out = series.build_series(
        token_ids=["t"], from_minute=100, to_minute=110, bin_minutes=5,
        count_rows=[], timing_rows=[], timing_total=0, timing_stride=1,
        latency_since=None,
    )
    assert (out["by_model"], out["model_bins"], out["by_model_since"]) == ([], [], None)
