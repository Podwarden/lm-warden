"""B3: per-request speed figures and the low-volume summary rules.

Definitions (design section 3.3):
  TTFT      = queued_s (warden admission queue) + ttft_s (slot -> first token)
  decode    = (completion - 1) / (duration - ttft), completion >= 16
  prefill   = (prompt - cached) / ttft_s, engine-measured cached only,
              fresh >= 512, prefilled alone (others are points, not in p50)
  duration  = duration_s
Low volume: n=0 none, 1..4 the values, 5..19 p50 + max, >=20 p50 + p95, p99 from 100.
"""

from __future__ import annotations

import sqlite3
import time

from app.stats.speed import row_metrics, summarize
from tests.conftest import jwt_login, seed_admin_user


def _row(**kw):
    base = dict(
        queued_s=0.5,
        ttft_s=1.5,
        duration_s=11.5,
        prompt_tokens=4000,
        completion_tokens=501,
        cached_tokens=1000,
        inflight_same_rank_at_start=0,
    )
    base.update(kw)
    return base


def test_row_metrics_definitions():
    m = row_metrics(_row())
    assert m["ttft"] == 2.0  # warden queue included
    assert m["decode"] == 50.0  # 500 / 10 s
    assert m["prefill"] == 2000.0  # 3000 computed / 1.5 s
    assert m["prefill_alone"] is True
    assert m["duration"] == 11.5
    assert m["prompt_speed"] == round(4000 / 1.5, 1)


def test_row_metrics_exclusions():
    assert row_metrics(_row(completion_tokens=10))["decode"] is None  # too short
    assert row_metrics(_row(cached_tokens=None))["prefill"] is None  # not measured
    assert row_metrics(_row(cached_tokens=3600))["prefill"] is None  # fresh < 512
    m = row_metrics(_row(inflight_same_rank_at_start=2))
    assert m["prefill"] is not None and m["prefill_alone"] is False
    m = row_metrics(_row(ttft_s=None, queued_s=None))  # non-streaming
    assert m["ttft"] is None and m["decode"] is None and m["prefill"] is None
    assert row_metrics(_row(queued_s=None))["ttft"] == 1.5


def test_summary_low_volume_rules():
    assert summarize([]) == {"n": 0, "mode": "none"}
    assert summarize([3.0, 1.0]) == {"n": 2, "mode": "list", "values": [3.0, 1.0]}
    s = summarize([float(i) for i in range(1, 11)])
    assert s["mode"] == "max" and s["max"] == 10.0 and s["p50"] == 5.5 and "p95" not in s
    s = summarize([float(i) for i in range(1, 41)])
    assert s["mode"] == "p95" and s["p99"] is None and s["p95"] > s["p50"]
    s = summarize([float(i) for i in range(1, 201)])
    assert s["p99"] is not None


def _seed(db_path):
    seed_admin_user(db_path)
    now = time.time()
    with sqlite3.connect(db_path) as db:
        for mid in ("ma", "mb"):
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status) "
                f"VALUES ('{mid}', '{mid}-name', 'o/r', 'main', '[0]', 1, NULL, NULL, "
                "0.9, 0, '[]', 'loaded')"
            )
        rows = []
        for i in range(25):
            tok, mid = ("tA", "ma") if i % 2 else ("tB", "ma" if i % 3 else "mb")
            rows.append(
                (
                    f"r{i}",
                    now - 60 - i,
                    mid,
                    f"{mid}-name",
                    tok,
                    f"name-{tok}",
                    "10.0.0.9",
                    4000,
                    501,
                    11.5,
                    1.5,
                    0.5,
                    1000,
                    0,
                    "stop",
                    0,
                    "",
                )
            )
        db.executemany(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_id, token_name, "
            "client_ip, prompt_tokens, completion_tokens, duration_s, ttft_s, queued_s, "
            "cached_tokens, inflight_same_rank_at_start, finish_reason, orphan, started_iso) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        db.commit()


def test_speed_route_one_model_and_groups(tmp_data_dir, client):
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    h = jwt_login(client)
    r = client.get("/api/stats/v2/speed?range=1h&model=ma", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {p["model_id"] for p in body["points"]} == {"ma"}
    assert body["summary"]["ttft"]["p50"] == 2.0
    assert body["summary"]["decode"]["p50"] == 50.0
    assert body["summary"]["warden_queue"]["p50"] == 0.5
    assert body["last_finished_at"] is not None
    # group by key across every model: one row per (key, model), never mixed
    g = client.get("/api/stats/v2/speed?range=1h&group=key", headers=h).json()["groups"]
    keys = {(x["key"], x["model_id"]) for x in g}
    assert ("tB", "ma") in keys and ("tB", "mb") in keys and ("tA", "ma") in keys
    assert all("p50_decode" in x for x in g)
    assert client.get("/api/stats/v2/speed?range=1h&group=bad", headers=h).status_code == 400
    assert client.get("/api/stats/v2/speed?range=1h&model=nope", headers=h).status_code == 400


def test_ip_groups_carry_no_speed_across_models():
    """Review finding 10: an IP group spans models, so it carries counts only."""
    from app.stats.speed import speed_groups

    rows = [
        dict(
            _row(),
            token_id="t",
            token_name="n",
            client_ip="1.1.1.1",
            model_id=m,
            model=m,
            finish_reason="stop",
        )
        for m in ("a", "b")
        for _ in range(5)
    ]
    (g,) = speed_groups(rows, "ip")
    assert g["requests"] == 10
    assert g["p50_ttft"] is None and g["p50_decode"] is None and g["p50_prefill"] is None


async def test_window_rows_are_cached_briefly(tmp_path):
    """Review finding 19: two refreshes in the same 15 s bucket read the window once."""
    from app.db.database import open_db
    from app.db.migrations import apply_migrations
    from app.stats import speed

    p = tmp_path / "t.db"
    async with open_db(p) as db:
        await apply_migrations(db)
        speed.clear_cache()
        a = await speed.cached_speed_rows(db, str(p), since=1000.0, model_ids=None)
        await db.execute(
            "INSERT INTO request_history(id, finished_at, model_id, model, prompt_tokens, "
            "completion_tokens, duration_s, orphan, started_iso) VALUES ('x', 2000, 'm', 'm', 1, 1, 1, 0, '')"
        )
        await db.commit()
        b = await speed.cached_speed_rows(db, str(p), since=1001.0, model_ids=None)
        c = await speed.cached_speed_rows(db, str(p), since=1001.0, model_ids=None, token_id="k")
    assert a == b == []  # same bucket: cached
    assert c == []  # a different key's lens is its own entry (and finds no rows for k)
