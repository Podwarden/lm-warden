"""S7 (#124) — /api/stats/v2/* endpoint contract tests.

These pin the JSON shape that dev-2's frontend slice consumes. Any change
to the response shape will break the FE chart wiring, so the asserts are
intentionally explicit about keys + types.

v1 endpoints (``/api/stats/models``, ``/api/stats/gpus``) coexist with v2
per CTO decision #7 and are covered separately in ``test_stats_api.py`` —
this module does NOT exercise v1, leaving v1's behaviour contract
unchanged.
"""

import sqlite3
import time

from tests.conftest import jwt_login, seed_admin_user


def _seed_v2_fixture(db_path) -> int:
    """Seed a small but representative dataset for the v2 endpoints:
    * 1 loaded model (status='loaded' + model_runtime row)
    * 2 GPUs reporting samples + power across two minutes
    * 2 api_tokens with usage rows; one orphan token_id with NO matching row

    Returns the earliest minute seeded into model_samples.
    """
    seed_admin_user(db_path)
    now_min = int(time.time() // 60)
    with sqlite3.connect(db_path) as db:
        # Active model + runtime
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-active', 'served-name-1', 'o/r', 'main', '[0,1]', 2, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port, started_at, "
            "health_ok, last_health_at) VALUES "
            "('m-active', 1234, 10000, datetime('now'), 1, datetime('now'))"
        )
        # A second model that's NOT loaded — must be excluded from active_models.
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-idle', 'served-name-2', 'o/r2', 'main', '[]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'registered')"
        )
        # gpu_samples (2 GPUs x 2 minutes)
        db.executemany(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, "
            "memory_used_mib, memory_total_mib, name) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (0, now_min - 1, 40, 1000, 16000, "GPU0"),
                (1, now_min - 1, 50, 2000, 16000, "GPU1"),
                (0, now_min, 80, 8000, 16000, "GPU0"),
                (1, now_min, 60, 4000, 16000, "GPU1"),
            ],
        )
        # power_samples: GPU0 averages 100W (200/2), GPU1 averages 150W (300/2)
        # at now_min; both have one sample at now_min-1.
        db.executemany(
            "INSERT INTO power_samples(gpu_idx, minute, watts_sum, samples) " "VALUES (?, ?, ?, ?)",
            [
                (0, now_min - 1, 90.0, 1),
                (1, now_min - 1, 140.0, 1),
                (0, now_min, 200.0, 2),  # avg 100
                (1, now_min, 300.0, 2),  # avg 150
            ],
        )
        # api_tokens — id 't-heavy' is the bigger consumer, 't-light' the
        # smaller. We also insert a token_usage_minute row keyed by 't-orphan'
        # with NO matching api_tokens entry, to verify the LEFT JOIN
        # gracefully surfaces "(unknown)" rather than dropping the row.
        db.executemany(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES (?, ?, ?, ?, 'inference')",
            [
                ("t-heavy", "Heavy Key", "pwm_aaa", "h1"),
                ("t-light", "Light Key", "pwm_bbb", "h2"),
            ],
        )
        db.executemany(
            "INSERT INTO token_usage_minute(token_id, minute, requests, "
            "prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?)",
            [
                ("t-heavy", now_min - 1, 10, 1000, 500),
                ("t-heavy", now_min, 5, 500, 250),
                ("t-light", now_min, 2, 50, 25),
                ("t-orphan", now_min, 1, 10, 5),
            ],
        )
        # model_samples — the SAME traffic, seen through the model dimension.
        # The overview's token series and tps read this table rather than
        # token_usage_minute, because only this one can be filtered by model
        # (?models=), and reading two different tables for the filtered and the
        # unfiltered case would make "every model selected" disagree with "no
        # filter". The per-key endpoint still reads token_usage_minute; it asks
        # a per-key question. Totals here are kept identical so the assertions
        # below stay checkable by hand: 1500 at now_min-1, 840 at now_min.
        db.executemany(
            "INSERT INTO model_samples(model_id, minute, requests, "
            "prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?)",
            [
                ("m-active", now_min - 1, 10, 1000, 500),
                ("m-active", now_min, 8, 560, 280),
            ],
        )
        db.commit()
    return now_min - 1


# ---------------------------------------------------------------------------
# Auth contract — both endpoints must 401 without a JWT (CSRF doesn't apply
# to GET). v1 covers this for /api/stats/models but the v2 paths are NEW
# router entries; the contract test belongs here.
# ---------------------------------------------------------------------------


def test_v2_overview_requires_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/stats/v2/overview?range=1h")
    assert r.status_code == 401


def test_v2_tokens_per_key_requires_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/stats/v2/tokens-per-key?range=1h")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Range validation — invalid value must 400 (matches v1 behaviour).
# ---------------------------------------------------------------------------


def test_v2_overview_invalid_range_400(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/overview?range=banana", headers=auth)
    assert r.status_code == 400


def test_v2_tokens_per_key_invalid_range_400(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/tokens-per-key?range=banana", headers=auth)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Overview shape contract — keys present, types correct, math right.
# ---------------------------------------------------------------------------


def test_v2_overview_shape_and_current_math(tmp_data_dir, client):
    client.get("/healthz")
    earliest_sample_minute = _seed_v2_fixture(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/overview?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()

    # Top-level contract.
    assert set(body.keys()) == {
        "range",
        "now_minute",
        "since_minute",
        "tokens_measured_from_minute",
        "current",
        "active_models",
        "series",
        # Null on an unfiltered request; a model selection echoes here so the
        # page can state what its numbers cover. See test_v2_model_selection.py.
        "selected_model_ids",
        "selected_gpu_indices",
    }
    assert body["selected_model_ids"] is None
    assert body["selected_gpu_indices"] is None
    assert body["range"] == "1h"
    assert isinstance(body["now_minute"], int)
    assert isinstance(body["since_minute"], int)
    assert body["since_minute"] < body["now_minute"]
    # Left bound of the token data: the earliest minute with a row in
    # model_samples over the WHOLE table (the fixture's first row), not the
    # selection's — the chart draws a measured zero inside it, "no data"
    # before it.
    assert body["tokens_measured_from_minute"] == earliest_sample_minute
    assert isinstance(body["tokens_measured_from_minute"], int)

    # current snapshot is the most-recent-minute aggregate:
    # vram used = 8000+4000 = 12000, total = 16000+16000 = 32000 → 38%
    # util = max(80, 60) = 80
    # power = 100 + 150 = 250 W
    current = body["current"]
    assert current["vram_used_mib"] == 12000
    assert current["vram_total_mib"] == 32000
    assert current["vram_pct"] == 38
    assert current["gpu_util_pct"] == 80
    assert current["power_w"] == 250.0
    # TPS = model_samples at the latest minute, (560 + 280) / 60 = 14.0.
    assert current["tps"] == 14.0

    # active_models — only loaded models present.
    ids = [m["id"] for m in body["active_models"]]
    assert ids == ["m-active"]
    assert body["active_models"][0]["served_model_name"] == "served-name-1"

    # series shape — every key, every row has the documented sub-keys.
    series = body["series"]
    assert set(series.keys()) == {"vram", "util", "power", "tokens"}
    for k, expected_keys in (
        ("vram", {"minute", "used_mib", "total_mib"}),
        ("util", {"minute", "max_pct"}),
        ("power", {"minute", "watts"}),
        (
            "tokens",
            {"minute", "prompt", "completion", "cached", "cached_measured_requests", "requests"},
        ),
    ):
        assert series[k], f"series.{k} unexpectedly empty"
        for row in series[k]:
            assert set(row.keys()) == expected_keys, (k, row)


def test_v2_overview_empty_db_returns_zeroes(tmp_data_dir, client):
    """Fresh warden with no samples yet: current values floor to 0 / None
    rather than 500'ing. The FE relies on this to render an empty chart."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/overview?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    cur = body["current"]
    assert cur["vram_used_mib"] == 0
    assert cur["vram_total_mib"] == 0
    assert cur["vram_pct"] == 0
    assert cur["gpu_util_pct"] == 0
    assert cur["power_w"] is None
    assert cur["tps"] == 0.0
    assert body["active_models"] == []
    assert body["series"]["vram"] == []
    assert body["series"]["power"] == []
    assert body["series"]["tokens"] == []
    # No model_samples rows at all: no left bound, so the FE can say
    # "no data" instead of "measured zero since the beginning of time".
    assert body["tokens_measured_from_minute"] is None


def test_v2_overview_power_series_skips_null_buckets(tmp_data_dir, client):
    """A minute where the only power_samples row has samples=0 (degenerate;
    shouldn't actually happen but the SQL uses NULLIF) must NOT emit a
    None-watts entry — the chart can't render that."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    now_min = int(time.time() // 60)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "INSERT INTO power_samples(gpu_idx, minute, watts_sum, samples) "
            "VALUES (0, ?, 0.0, 0)",
            (now_min,),
        )
        db.commit()
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/overview?range=1h", headers=auth)
    assert r.status_code == 200
    assert r.json()["series"]["power"] == []


# ---------------------------------------------------------------------------
# tokens-per-key — JOIN correctness + ordering + orphan handling.
# ---------------------------------------------------------------------------


def test_v2_tokens_per_key_join_and_ordering(tmp_data_dir, client):
    client.get("/healthz")
    _seed_v2_fixture(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/tokens-per-key?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["range"] == "1h"
    rows = body["rows"]
    # Heavy first (1500+750 = 2250 total), then light (75), then orphan (15).
    assert [r["token_id"] for r in rows] == ["t-heavy", "t-light", "t-orphan"]
    heavy = rows[0]
    assert heavy["name"] == "Heavy Key"
    assert heavy["prefix"] == "pwm_aaa"
    assert heavy["requests"] == 15
    assert heavy["prompt_tokens"] == 1500
    assert heavy["completion_tokens"] == 750
    assert heavy["total_tokens"] == 2250
    # Orphan must surface with the placeholder name + null prefix rather than
    # being silently dropped — historical usage data deserves to be visible.
    orphan = rows[2]
    assert orphan["name"] == "(unknown)"
    assert orphan["prefix"] is None
    assert orphan["total_tokens"] == 15


def test_v2_tokens_per_key_omits_unused_tokens(tmp_data_dir, client):
    """Tokens that exist in api_tokens but have NO usage in the window
    must not appear — keeps the response bounded by activity."""
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('t-unused', 'Unused', 'pwm_zzz', 'h9', 'inference')"
        )
        db.commit()
    auth = jwt_login(client)
    r = client.get("/api/stats/v2/tokens-per-key?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == []
