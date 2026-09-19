"""GET /api/stats/v2/requests and /api/stats/v2/latency -- the per-request
history endpoints behind the requests chart and the latency distributions.

Rows are written straight into request_history (the store's write path has
its own tests); these pin the HTTP contract: window and selection rules,
striding, coverage, the two latency bases and their labels.
"""

import sqlite3
import time

from tests.conftest import jwt_login, seed_admin_user


def _seed_models(db_path):
    seed_admin_user(db_path)
    with sqlite3.connect(db_path) as db:
        for mid, name in (("id-model-a", "model-a"), ("id-model-b", "model-b")):
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status) "
                f"VALUES ('{mid}', '{name}', 'o/r', 'main', '[0]', 1, NULL, NULL, "
                "0.9, 0, '[]', 'loaded')"
            )
        db.commit()


def _insert(db_path, rows):
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_name, "
            "client_ip, prompt_tokens, completion_tokens, duration_s, ttft_s, "
            "finish_reason, orphan, started_iso) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        db.commit()


def _row(i, *, at, model="a", token="key-a", dur=2.0, ttft=0.5, gen=50, reason="stop"):
    return (
        f"r{i}", at, f"id-model-{model}", f"model-{model}", token, "10.0.0.1",
        100, gen, dur, ttft, reason, 0, "2026-09-05T18:59:00Z",
    )


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_models(db_path)
    return db_path, jwt_login(client)


# ---- /api/stats/v2/requests -------------------------------------------------


def test_requests_requires_a_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/stats/v2/requests").status_code == 401
    assert client.get("/api/stats/v2/latency").status_code == 401


def test_requests_honours_the_window_and_is_newest_first(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [
        _row(1, at=now - 30),
        _row(2, at=now - 3000),      # inside 1h
        _row(3, at=now - 2 * 3600),  # outside 1h, inside 6h
    ])
    body = client.get("/api/stats/v2/requests?range=1h", headers=auth).json()
    assert body["range"] == "1h"
    assert body["total"] == 2
    assert body["stride"] == 1
    assert [r["id"] for r in body["requests"]] == ["r1", "r2"]
    assert body["selected_model_ids"] is None
    body = client.get("/api/stats/v2/requests?range=6h", headers=auth).json()
    assert body["total"] == 3


def test_requests_row_carries_the_finished_record_fields(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    _insert(db_path, [_row(1, at=time.time() - 1, ttft=None, reason="length")])
    r = client.get("/api/stats/v2/requests?range=1h", headers=auth).json()["requests"][0]
    assert r["model"] == "model-a" and r["model_id"] == "id-model-a"
    assert r["token_name"] == "key-a" and r["client_ip"] == "10.0.0.1"
    assert r["prompt_tokens"] == 100 and r["completion_tokens"] == 50
    assert r["duration_s"] == 2.0 and r["ttft_s"] is None
    assert r["finish_reason"] == "length" and r["orphan"] is False
    assert isinstance(r["finished_at"], float)


def test_requests_scopes_on_the_model_row_id(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_row(1, at=now - 1, model="a"), _row(2, at=now - 2, model="b")])
    body = client.get(
        "/api/stats/v2/requests?range=1h&models=id-model-b", headers=auth
    ).json()
    assert [r["id"] for r in body["requests"]] == ["r2"]
    assert body["selected_model_ids"] == ["id-model-b"]
    # The served name is not an id; same 400 as the overview.
    assert client.get(
        "/api/stats/v2/requests?range=1h&models=model-b", headers=auth
    ).status_code == 400
    assert client.get("/api/stats/v2/requests?models=", headers=auth).status_code == 400
    assert client.get("/api/stats/v2/requests?range=2h", headers=auth).status_code == 400


def test_requests_strides_a_window_larger_than_the_limit(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_row(i, at=now - 3000 + i) for i in range(50)])
    body = client.get("/api/stats/v2/requests?range=1h&limit=10", headers=auth).json()
    assert body["total"] == 50
    assert body["stride"] == 5
    assert len(body["requests"]) == 10
    ids = [r["id"] for r in body["requests"]]
    assert ids[0] == "r49" and ids[-1] == "r4"  # span kept, not the newest ten


def test_requests_coverage_says_when_history_begins(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    # Store empty: nothing covers anything.
    cov = client.get("/api/stats/v2/requests?range=1h", headers=auth).json()["coverage"]
    assert cov["earliest_epoch"] is None
    assert cov["covers_window"] is False
    assert cov["retention_days"] == 30 and cov["max_rows"] == 200_000
    # History that began 10 minutes ago covers nothing that asks for an hour.
    _insert(db_path, [_row(1, at=now - 600)])
    cov = client.get("/api/stats/v2/requests?range=1h", headers=auth).json()["coverage"]
    assert cov["earliest_epoch"] == _approx(now - 600)
    assert cov["covers_window"] is False
    # ...and does cover an hour once a row is old enough.
    _insert(db_path, [_row(2, at=now - 4000)])
    cov = client.get("/api/stats/v2/requests?range=1h", headers=auth).json()["coverage"]
    assert cov["covers_window"] is True


def test_requests_coverage_is_false_when_retention_is_shorter_than_the_window(
    tmp_data_dir, client, monkeypatch
):
    """A 7d window against a 1-day retention can never be served in full, and
    the page must say so rather than draw one day under a '7d' heading."""
    db_path, auth = _ready(client, tmp_data_dir)
    _insert(db_path, [_row(1, at=time.time() - 8 * 86400)])  # would cover 7d
    settings = client.app.state.settings
    import dataclasses

    client.app.state.settings = dataclasses.replace(
        settings, request_history_retention_days=1
    )
    cov = client.get("/api/stats/v2/requests?range=7d", headers=auth).json()["coverage"]
    assert cov["retention_days"] == 1
    assert cov["covers_window"] is False
    cov = client.get("/api/stats/v2/requests?range=1h", headers=auth).json()["coverage"]
    assert cov["covers_window"] is True


class _approx(float):
    def __eq__(self, other):
        return abs(float(self) - float(other)) < 1.0


# ---- /api/stats/v2/latency -------------------------------------------------


def test_latency_window_basis_uses_every_row_in_the_window(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [
        _row(1, at=now - 10, dur=1.0, ttft=0.1, gen=11),
        _row(2, at=now - 20, dur=3.0, ttft=0.3, gen=11),
        _row(3, at=now - 2 * 3600, dur=100.0, ttft=50.0),  # outside 1h
    ])
    body = client.get("/api/stats/v2/latency?range=1h", headers=auth).json()
    assert body["basis"] == "window" and body["n"] is None
    assert body["count"] == 2
    assert body["span_s"] == _approx(10.0)
    assert body["ttft"]["count"] == 2 and body["ttft"]["p50"] == 0.2
    assert body["duration"]["p50"] == 2.0
    # ITL: (1-0.1)/10 = 0.09 and (3-0.3)/10 = 0.27 -> mean 0.18
    assert body["itl"]["count"] == 2
    assert abs(body["itl"]["mean"] - 0.18) < 1e-9
    # Buckets in the engine-histogram wire shape: cumulative, +Inf as null.
    b = body["ttft"]["buckets"]
    assert b["le"][-1] is None
    assert b["counts"][-1] == 2
    assert all(x <= y for x, y in zip(b["counts"], b["counts"][1:], strict=False))


def test_latency_last_basis_ignores_the_window_and_says_its_span(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_row(i, at=now - i * 3600, dur=float(i + 1)) for i in range(5)])
    body = client.get("/api/stats/v2/latency?range=1h&basis=last&n=3", headers=auth).json()
    assert body["basis"] == "last" and body["n"] == 3
    assert body["since_epoch"] is None
    assert body["count"] == 3
    assert body["span_s"] == _approx(2 * 3600)
    assert body["duration"]["p50"] == 2.0  # the newest three: 1, 2, 3
    assert client.get("/api/stats/v2/latency?basis=median", headers=auth).status_code == 400


def test_latency_scopes_by_model_and_is_empty_not_zero_when_nothing_matches(
    tmp_data_dir, client
):
    db_path, auth = _ready(client, tmp_data_dir)
    _insert(db_path, [_row(1, at=time.time() - 1, model="a")])
    body = client.get(
        "/api/stats/v2/latency?range=1h&models=id-model-b", headers=auth
    ).json()
    assert body["count"] == 0
    assert body["ttft"]["p50"] is None
    assert body["duration"]["count"] == 0
    assert body["coverage"]["earliest_epoch"] is not None  # store-wide


# ---- the superseded endpoint, kept for a UI older than this API -------------


def test_live_finished_reads_the_store_now(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_row(1, at=now - 1), _row(2, at=now - 2, model="b")])
    body = client.get("/api/stats/live/finished", headers=auth).json()
    assert body["available"] is True
    assert body["retained_seconds"] == 30 * 86400
    assert [r["id"] for r in body["requests"]] == ["r1", "r2"]
    body = client.get("/api/stats/live/finished?models=id-model-b", headers=auth).json()
    assert [r["id"] for r in body["requests"]] == ["r2"]
    assert client.get("/api/stats/live/finished?models=", headers=auth).status_code == 400


def test_the_requests_row_carries_the_admission_queue_wait(tmp_data_dir, client):
    """0032: queue wait is on the row the chart and table read.

    It is reported alongside ttft_s rather than folded into it: the clock
    ttft_s is measured from is read after admission, so the wait was never
    inside it (app/proxy/routes.py).
    """
    db_path, auth = _ready(client, tmp_data_dir)
    now = time.time()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_name, "
            "client_ip, prompt_tokens, completion_tokens, duration_s, ttft_s, "
            "finish_reason, orphan, started_iso, queued_s) "
            "VALUES ('r-q', ?, 'id-model-a', 'model-a', 'key-a', '10.0.0.1', "
            "100, 50, 2.0, 0.5, 'stop', 0, '2026-09-10T18:59:00Z', 3.25)",
            (now - 10,),
        )
        # A row from before the migration: absent, not zero.
        db.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, duration_s, "
            "ttft_s, started_iso) VALUES ('r-old', ?, 'id-model-a', 'model-a', 2.0, 0.5, 'x')",
            (now - 20,),
        )
        db.commit()

    r = client.get("/api/stats/v2/requests?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    rows = {row["id"]: row for row in r.json()["requests"]}
    assert rows["r-q"]["queued_s"] == 3.25
    assert rows["r-q"]["ttft_s"] == 0.5  # untouched by the queue figure
    assert rows["r-old"]["queued_s"] is None
