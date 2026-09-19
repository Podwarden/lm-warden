"""GET /api/tokens/{id}/series -- the HTTP contract (spec 2026-09-18 §3.4).

Usage rows and request_history rows are written straight into the database;
the pure binning is pinned in test_series.py. Every window sits in the past,
anchored at a UTC midnight ten days ago, so no assertion depends on the clock.
"""

import json
import math
import sqlite3
import time

import pytest

from tests.conftest import csrf_header, jwt_login, seed_admin_user

DAY = 86400
BASE = (int(time.time()) // DAY - 10) * DAY  # a UTC midnight, 10 days ago
BASE_MIN = BASE // 60
UNKNOWN = "deadbeefdeadbeefdeadbeefdeadbeef"


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path, {**jwt_login(client), **csrf_header(client)}


def _create(client, hdrs, name="harness") -> str:
    r = client.post("/api/tokens", json={"name": name}, headers=hdrs)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _rotate(client, hdrs, tid) -> str:
    r = client.post(f"/api/tokens/{tid}/rotate", json={"grace_hours": 24}, headers=hdrs)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _usage(db_path, rows):
    """rows: (token_id, minute, requests, prompt_tokens, completion_tokens)"""
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO token_usage_minute(token_id, minute, requests, prompt_tokens, "
            "completion_tokens) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()


def _history(db_path, rows):
    """rows: (id, finished_at, token_id, queued_s, ttft_s, duration_s)"""
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_name, "
            "token_id, queued_s, ttft_s, duration_s, started_iso) "
            "VALUES (?, ?, 'id-model-a', 'model-a', 'key-a', ?, ?, ?, ?, 'x')",
            rows,
        )
        db.commit()


def _series(client, hdrs, tid, frm, to, **params):
    q = {"from": frm, "to": to, **params}
    return client.get(f"/api/tokens/{tid}/series", params=q, headers=hdrs)


def _chain(client, hdrs, db_path):
    """A rotated pair: `a` (earlier) and `b` (current), both with traffic."""
    a = _create(client, hdrs)
    b = _rotate(client, hdrs, a)
    _usage(db_path, [
        (a, BASE_MIN + 1, 2, 100, 10),
        (b, BASE_MIN + 1, 3, 200, 20),  # same minute as a's row
        (b, BASE_MIN + 7, 1, 250, 5),
    ])
    return a, b


# ---- errors ------------------------------------------------------------------


def test_requires_a_session(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    r = client.get(f"/api/tokens/{tid}/series", params={"from": BASE, "to": BASE + 3600})
    assert r.status_code == 401


def test_unknown_token_is_404(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    r = _series(client, hdrs, UNKNOWN, BASE, BASE + 3600)
    assert r.status_code == 404
    # The handler's own detail -- a missing route would answer "Not Found".
    assert r.json()["detail"] == "token not found"


@pytest.mark.parametrize(
    "case",
    ["reversed", "empty", "from_beyond_now", "too_long", "max_bins_0", "max_bins_361", "no_to"],
)
def test_bad_windows_are_422(tmp_data_dir, client, case):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    now = time.time()
    params = {
        "reversed": {"from": BASE + 10, "to": BASE},
        "empty": {"from": BASE, "to": BASE},
        # 'to' clamps to now, but 'from' is ALSO past now, so it stays >= 'to'.
        "from_beyond_now": {"from": now + 10, "to": now + 3600},
        "too_long": {"from": now - 367 * DAY, "to": now},
        "max_bins_0": {"from": BASE, "to": BASE + 3600, "max_bins": 0},
        "max_bins_361": {"from": BASE, "to": BASE + 3600, "max_bins": 361},
        "no_to": {"from": BASE},
    }[case]
    r = client.get(f"/api/tokens/{tid}/series", params=params, headers=hdrs)
    assert r.status_code == 422, r.text


def test_a_window_entirely_in_the_future_says_so(tmp_data_dir, client):
    # #251: after 'to' is clamped, from >= to -- but the client DID send
    # from < to, so "'from' must be before 'to'" would be wrong.
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    now = time.time()
    r = _series(client, hdrs, tid, now + 600, now + 3600)
    assert r.status_code == 422
    assert r.json()["detail"] == "'from' is after the server's current time"
    r = _series(client, hdrs, tid, BASE + 10, BASE)
    assert r.status_code == 422
    assert r.json()["detail"] == "'from' must be before 'to'"


def test_future_to_is_clamped_to_the_servers_now(tmp_data_dir, client):
    # Ruled in 2026-09-18 (clock-skew review): a client clock that runs fast
    # must not turn a "last hour" poll into an empty chart -- 'to' past the
    # server's own clock is clamped to it rather than 422ed.
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    now = time.time()
    r = _series(client, hdrs, tid, now - 3600, now + 3600)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["to_minute"] == math.ceil(now / 60)
    assert body["to_minute"] < math.ceil((now + 3600) / 60)


# ---- binning -----------------------------------------------------------------


@pytest.mark.parametrize(("span", "width"), [(3600, 1), (6 * 3600, 1), (DAY, 5), (7 * DAY, 30)])
def test_presets_pick_the_stats_page_widths(tmp_data_dir, client, span, width):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    body = _series(client, hdrs, tid, BASE, BASE + span).json()
    assert body["bin_minutes"] == width
    assert (body["from_minute"], body["to_minute"]) == (BASE_MIN, BASE_MIN + span // 60)


def test_sparse_minutes_average_to_sum_over_width(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _usage(db_path, [(tid, BASE_MIN, 10, 1000, 100), (tid, BASE_MIN + 2, 5, 500, 50)])

    body = _series(client, hdrs, tid, BASE, BASE + DAY).json()  # 5-minute bins
    assert body["bin_minutes"] == 5
    (b,) = body["bins"]  # sparse: only the bin with data
    assert b["minute"] == BASE_MIN
    assert b["requests"] == 15
    assert b["requests_per_min"] == 3.0  # 15 / 5, not the 7.5 mean of two rows
    assert b["prompt_per_min"] == 300.0
    assert b["completion_per_min"] == 30.0
    assert (b["peak_prompt"], b["peak_completion"]) == (1000, 100)
    assert body["totals"] == {"requests": 15, "prompt_tokens": 1500, "completion_tokens": 150}


# ---- chain -------------------------------------------------------------------


def test_chain_1_includes_earlier_keys_and_chain_0_does_not(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    a, b = _chain(client, hdrs, db_path)

    both = _series(client, hdrs, b, BASE, BASE + 3600).json()  # chain defaults to 1
    assert both["token_ids"] == [a, b]
    assert both["totals"] == {"requests": 6, "prompt_tokens": 550, "completion_tokens": 35}

    alone = _series(client, hdrs, b, BASE, BASE + 3600, chain=0).json()
    assert alone["token_ids"] == [b]
    assert alone["totals"] == {"requests": 4, "prompt_tokens": 450, "completion_tokens": 25}

    # The earlier key's own page never reaches forward to its successor.
    earlier = _series(client, hdrs, a, BASE, BASE + 3600).json()
    assert earlier["token_ids"] == [a]


def test_peak_is_the_chains_busiest_minute(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    _a, b = _chain(client, hdrs, db_path)
    bins = {x["minute"]: x for x in _series(client, hdrs, b, BASE, BASE + DAY).json()["bins"]}
    # Minute 1 carried 100 + 200 across the two keys: the chain peaked at 300,
    # although neither key alone ever passed 250.
    assert bins[BASE_MIN]["peak_prompt"] == 300
    assert bins[BASE_MIN + 5]["peak_prompt"] == 250


# ---- timings -----------------------------------------------------------------


def test_timings_per_bin_and_null_for_bins_without_requests(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    other = _create(client, hdrs, "other")
    _history(db_path, [
        ("r1", BASE + 30, tid, 0.1, 1.0, 10.0),
        ("r2", BASE + 31, tid, 0.2, 2.0, 20.0),
        ("r3", BASE + 32, tid, 0.3, 3.0, 30.0),
        ("r4", BASE + 33, other, 9.0, 9.0, 90.0),  # another key: excluded
    ])
    _usage(db_path, [(tid, BASE_MIN + 5, 1, 10, 1)])  # a bin with usage, no timings

    bins = {x["minute"]: x for x in _series(client, hdrs, tid, BASE, BASE + 3600).json()["bins"]}
    first = bins[BASE_MIN]
    assert first["n"] == 3
    assert first["queue_p50"] == pytest.approx(0.2)
    assert first["queue_p95"] == pytest.approx(0.29)
    assert first["ttft_p50"] == pytest.approx(2.0)
    assert first["ttft_p95"] == pytest.approx(2.9)
    assert first["duration_p50"] == pytest.approx(20.0)
    assert first["duration_p95"] == pytest.approx(29.0)
    assert (first["requests"], first["requests_per_min"]) == (0, 0.0)

    quiet = bins[BASE_MIN + 5]
    assert quiet["n"] == 0
    for k in ("queue", "ttft", "duration"):
        assert quiet[f"{k}_p50"] is None and quiet[f"{k}_p95"] is None


def test_latency_since_is_the_first_row_with_a_token_id(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert _series(client, hdrs, tid, BASE, BASE + 3600).json()["latency_since"] is None

    _history(db_path, [
        ("pre", BASE - 5000, None, None, 1.0, 2.0),  # written before 0033: no token_id
        ("post", BASE + 30, tid, 0.1, 1.0, 2.0),
    ])
    assert _series(client, hdrs, tid, BASE, BASE + 3600).json()["latency_since"] == BASE + 30


def test_timing_sample_reports_the_stride_and_n_counts_sampled_rows(
    tmp_data_dir, client, monkeypatch
):
    from app.stats import request_history

    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _history(db_path, [(f"r{i}", BASE + 30 + i, tid, 0.1, 1.0, 2.0) for i in range(5)])
    _usage(db_path, [(tid, BASE_MIN, 5, 50, 5)])

    body = _series(client, hdrs, tid, BASE, BASE + 3600).json()
    assert body["timing_sample"] == {"total": 5, "used": 5, "stride": 1}

    monkeypatch.setattr(request_history, "TIMING_ROW_CAP", 2)
    body = _series(client, hdrs, tid, BASE, BASE + 3600).json()
    assert body["timing_sample"] == {"total": 5, "used": 2, "stride": 3}  # ceil(5/2)
    (b,) = body["bins"]
    assert b["n"] == 2          # sampled rows
    assert b["requests"] == 5   # token_usage_minute: exact regardless


def test_timings_0_skips_request_history_entirely(tmp_data_dir, client, monkeypatch):
    # #251: the history strip draws usage only, so it asks for timings=0 and
    # must not pay for the COUNT and up-to-50k-row read it never shows.
    from app.stats import request_history

    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _history(db_path, [("r1", BASE + 30, tid, 0.1, 1.0, 2.0)])
    _usage(db_path, [(tid, BASE_MIN, 1, 10, 1)])

    with_timings = _series(client, hdrs, tid, BASE, BASE + 3600).json()  # default 1
    assert with_timings["bins"][0]["n"] == 1
    assert with_timings["latency_since"] == BASE + 30

    async def _refuse(*_a, **_kw):
        raise AssertionError("timings=0 must not touch request_history")

    monkeypatch.setattr(request_history, "query_token_timings", _refuse)
    monkeypatch.setattr(request_history, "earliest_token_finished_at", _refuse)
    r = _series(client, hdrs, tid, BASE, BASE + 3600, timings=0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["timing_sample"] == {"total": 0, "used": 0, "stride": 1}
    assert body["latency_since"] is None
    (b,) = body["bins"]
    assert b["n"] == 0
    for k in ("queue", "ttft", "duration"):
        assert b[f"{k}_p50"] is None and b[f"{k}_p95"] is None
    # The usage half is untouched.
    assert (b["requests"], b["prompt_per_min"]) == (1, 10.0)
    assert body["totals"] == with_timings["totals"]


@pytest.mark.parametrize("value", ["2", "-1", "yes"])
def test_timings_must_be_0_or_1(tmp_data_dir, client, value):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert _series(client, hdrs, tid, BASE, BASE + 3600, timings=value).status_code == 422


# ---- per model (0036) --------------------------------------------------------


def _model_usage(db_path, rows):
    """rows: (token_id, model_id, minute, requests, prompt_tokens, completion_tokens)
    or the same with a variant id after the model id; default variant is
    ``<model_id>-v1``."""
    full = [r if len(r) == 7 else (r[0], r[1], f"{r[1]}-v1", *r[2:]) for r in rows]
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO token_model_usage_minute(token_id, model_id, variant_id, minute, "
            "requests, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
            full,
        )
        db.commit()


def _variant(db_path, variant_id, model_id, descriptor, first_seen):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO model_variants(id, model_id, served_model_name, descriptor, "
            "first_seen) VALUES (?, ?, 'first-name', ?, ?)",
            (variant_id, model_id, json.dumps(descriptor), first_seen),
        )
        db.commit()


def _model(db_path, model_id, name):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices) "
            "VALUES (?, ?, 'org/repo', '[0]')",
            (model_id, name),
        )
        db.commit()


def _chain_by_model(client, hdrs, db_path):
    """A rotated pair: `a` used model-a; `b` used model-a and model-b."""
    a = _create(client, hdrs)
    b = _rotate(client, hdrs, a)
    _model(db_path, "id-a", "model-a")
    _model(db_path, "id-b", "model-b")
    _model_usage(db_path, [
        (a, "id-a", BASE_MIN + 1, 2, 100, 10),
        (b, "id-a", BASE_MIN + 1, 1, 50, 40),
        (b, "id-b", BASE_MIN + 2, 3, 600, 200),
        (b, "id-b", BASE_MIN + 7, 1, 150, 50),
    ])
    return a, b


def test_by_model_totals_names_order_and_share(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    _a, b = _chain_by_model(client, hdrs, db_path)

    body = _series(client, hdrs, b, BASE, BASE + 3600).json()
    models = [{k: v for k, v in m.items() if k != "variants"} for m in body["by_model"]]
    assert models == [
        {"model_id": "id-b", "model": "model-b", "requests": 4, "prompt_tokens": 750,
         "completion_tokens": 250, "total_tokens": 1000, "share": pytest.approx(1000 / 1200)},
        {"model_id": "id-a", "model": "model-a", "requests": 3, "prompt_tokens": 150,
         "completion_tokens": 50, "total_tokens": 200, "share": pytest.approx(200 / 1200)},
    ]
    assert [[v["variant_id"] for v in m["variants"]] for m in body["by_model"]] == [
        ["id-b-v1"], ["id-a-v1"],
    ]


def test_by_model_honours_the_chain_and_the_window(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    a, b = _chain_by_model(client, hdrs, db_path)

    alone = _series(client, hdrs, b, BASE, BASE + 3600, chain=0).json()["by_model"]
    assert [(m["model_id"], m["requests"], m["total_tokens"]) for m in alone] == [
        ("id-b", 4, 1000), ("id-a", 1, 90),
    ]
    # The earlier key never reaches forward to its successor's model-b traffic.
    earlier = _series(client, hdrs, a, BASE, BASE + 3600).json()["by_model"]
    assert [(m["model_id"], m["total_tokens"], m["share"]) for m in earlier] == [
        ("id-a", 110, 1.0),
    ]
    # [BASE + 5 min, ...) leaves out minutes 1 and 2.
    late = _series(client, hdrs, b, BASE + 300, BASE + 3600).json()["by_model"]
    assert [(m["model_id"], m["total_tokens"]) for m in late] == [("id-b", 200)]


def test_a_deleted_model_falls_back_to_its_stored_id(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _model_usage(db_path, [(tid, "id-gone", BASE_MIN, 1, 10, 1)])
    (only,) = _series(client, hdrs, tid, BASE, BASE + 3600).json()["by_model"]
    assert (only["model_id"], only["model"]) == ("id-gone", "id-gone")

    # With a recorded variant, the name it was first seen under wins over the id.
    other = _create(client, hdrs, "other")
    _variant(db_path, "gone-v2", "id-gone2", {"backend": "vllm"}, 5.0)
    _model_usage(db_path, [(other, "id-gone2", "gone-v2", BASE_MIN, 1, 10, 1)])
    (only,) = _series(client, hdrs, other, BASE, BASE + 3600).json()["by_model"]
    assert only["model"] == "first-name"


def test_two_variants_of_one_model_split_under_it(tmp_data_dir, client):
    # The try-stack or the settings form rewrote the models row in place: one
    # model id, two variants. The card lists one model with both.
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _model(db_path, "id-a", "model-a")
    _variant(db_path, "va-old", "id-a", {
        "backend": "vllm", "engine_vllm_version": "0.9.0", "quantization": "fp8",
        "engine_image": "vllm/vllm-openai:v0.9.0", "extra_env": "never-here",
    }, BASE - 100.0)
    _variant(db_path, "va-new", "id-a", {
        "backend": "vllm", "engine_vllm_version": "0.11.0", "quantization": "awq",
        "dtype": "auto", "hf_revision": "a1b2c3d", "max_model_len": 32768,
    }, BASE + 600.0)
    _model_usage(db_path, [
        (tid, "id-a", "va-old", BASE_MIN + 1, 1, 100, 20),
        (tid, "id-a", "va-new", BASE_MIN + 11, 2, 300, 80),
        (tid, "id-a", "va-new", BASE_MIN + 12, 1, 50, 10),
    ])
    (m,) = _series(client, hdrs, tid, BASE, BASE + 3600).json()["by_model"]
    assert (m["model"], m["requests"], m["total_tokens"], m["share"]) == ("model-a", 4, 560, 1.0)
    new, old = m["variants"]
    assert new == {
        "variant_id": "va-new", "backend": "vllm", "engine_channel": None,
        "engine_vllm_version": "0.11.0", "engine_version": None, "quantization": "awq",
        "dtype": "auto", "hf_revision": "a1b2c3d", "hf_commit": None, "max_model_len": 32768,
        "engine_image_tag": None,
        "first_seen": BASE + 600.0, "requests": 3, "prompt_tokens": 350,
        "completion_tokens": 90, "total_tokens": 440, "share": pytest.approx(440 / 560),
    }
    assert (old["variant_id"], old["engine_image_tag"], old["quantization"]) == (
        "va-old", "v0.9.0", "fp8",
    )
    assert "extra_env" not in old
    # The chart stays per MODEL: the two variants share one line.
    bins = _series(client, hdrs, tid, BASE, BASE + 3600).json()["model_bins"]
    assert [list(b["models"]) for b in bins] == [["id-a"], ["id-a"], ["id-a"]]


def test_model_bins_are_sparse_per_bin_and_per_model(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    _a, b = _chain_by_model(client, hdrs, db_path)

    body = _series(client, hdrs, b, BASE, BASE + DAY).json()  # 5-minute bins
    assert body["bin_minutes"] == 5
    assert body["model_bins"] == [
        {"minute": BASE_MIN, "models": {
            # a's 100/10 + b's 50/40 in minute 1, both on model-a: summed ÷ 5.
            "id-a": {"prompt_per_min": 30.0, "completion_per_min": 10.0},
            "id-b": {"prompt_per_min": 120.0, "completion_per_min": 40.0},
        }},
        {"minute": BASE_MIN + 5, "models": {  # model-a had no traffic here
            "id-b": {"prompt_per_min": 30.0, "completion_per_min": 10.0},
        }},
    ]


def test_by_model_since_is_the_first_row_store_wide(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    other = _create(client, hdrs, "other")
    body = _series(client, hdrs, tid, BASE, BASE + 3600).json()
    assert (body["by_model_since"], body["by_model"], body["model_bins"]) == (None, [], [])

    # Another key's older row still marks when recording began.
    _model_usage(db_path, [
        (other, "id-a", BASE_MIN - 30, 1, 1, 1),
        (tid, "id-a", BASE_MIN + 3, 1, 1, 1),
    ])
    assert _series(client, hdrs, tid, BASE, BASE + 3600).json()["by_model_since"] == (
        (BASE_MIN - 30) * 60
    )


def test_by_model_0_skips_the_per_model_split(tmp_data_dir, client, monkeypatch):
    from app.db.repos.tokens import TokenModelUsageRepo

    db_path, hdrs = _ready(client, tmp_data_dir)
    _a, b = _chain_by_model(client, hdrs, db_path)
    _usage(db_path, [(b, BASE_MIN, 1, 10, 1)])

    async def _refuse(*_a, **_kw):
        raise AssertionError("by_model=0 must not touch token_model_usage_minute")

    for name in ("by_variant", "model_bins", "earliest_minute"):
        monkeypatch.setattr(TokenModelUsageRepo, name, _refuse)
    r = _series(client, hdrs, b, BASE, BASE + 3600, by_model=0, timings=0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["by_model"], body["model_bins"], body["by_model_since"]) == ([], [], None)
    assert body["totals"]["requests"] == 1  # the usage half is untouched


@pytest.mark.parametrize("value", ["2", "-1", "yes"])
def test_by_model_must_be_0_or_1(tmp_data_dir, client, value):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert _series(client, hdrs, tid, BASE, BASE + 3600, by_model=value).status_code == 422
