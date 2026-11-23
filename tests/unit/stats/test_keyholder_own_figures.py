"""A key holder's stats carry no figure derived from another key's requests:
forest speeds, in-flight learned rates and estimates, queue counts. Admins keep
the fleet views. Two keys with very different speeds make any mixing visible."""

import sqlite3
import time

from app.stats import prefill_model, routes_forest
from tests.unit.proxy.test_request_registry import _mk as _live
from tests.unit.stats.test_forest_routes import _forest_login, _insert, _r, _ready_key


def _learn_rows(db_path, tok, ttft, n=20, now=None):
    now = time.time() if now is None else now
    with sqlite3.connect(db_path) as db:
        for i in range(n):
            db.execute(
                "INSERT INTO request_history(id, finished_at, model_id, model, token_id, token_name,"
                " prompt_tokens, completion_tokens, cached_tokens, duration_s, ttft_s, started_iso,"
                " orphan, inflight_same_rank_at_start) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"{tok}-{i}", now - 100 - i, "qwen", "qwen", tok, tok, 2000, 10, 0, ttft + 1,
                 ttft, "2026-11-21T00:00:00Z", 0, 0),
            )  # fmt: skip
        db.commit()


def test_forest_speeds_are_the_key_holders_own(tmp_data_dir, client, monkeypatch):
    monkeypatch.setattr(routes_forest, "FOREST_TTL_S", 0.0)
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    now = time.time()
    own = [_r(i, now - 100 - i, p=1000) for i in range(5)]  # (1000-900)/0.5 = 200 tok/s
    other = [
        _r(100 + i, now - 100 - i, tok="tok2", name="o", sk="B", p=5000) for i in range(5)
    ]  # 8200 tok/s
    _insert(db_path, own + other)
    kh = client.get("/api/stats/forest?range=6h", headers=_forest_login(client, plaintext)).json()
    assert kh["models"]["v1"]["prefill_tps"] == 200.0
    assert kh["models"]["v1"]["n"] == 5
    ad = client.get("/api/stats/forest?range=6h", headers=admin).json()
    assert ad["models"]["v1"]["n"] == 10 and ad["models"]["v1"]["prefill_tps"] > 200.0


def test_forest_omits_speeds_below_the_own_floor(tmp_data_dir, client, monkeypatch):
    monkeypatch.setattr(routes_forest, "FOREST_TTL_S", 0.0)
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    now = time.time()
    own = [_r(i, now - 100 - i) for i in range(routes_forest.TPUT_MIN_OWN - 1)]
    other = [_r(100 + i, now - 100 - i, tok="tok2", name="o", sk="B") for i in range(20)]
    _insert(db_path, own + other)
    kh = client.get("/api/stats/forest?range=6h", headers=_forest_login(client, plaintext)).json()
    assert kh["trees"] and "v1" not in kh["models"]
    ad = client.get("/api/stats/forest?range=6h", headers=admin).json()
    assert ad["models"]["v1"]["n"] == routes_forest.TPUT_MIN_OWN - 1 + 20


def test_own_state_learns_from_one_key_only(tmp_data_dir, client):
    db_path, _plaintext, _admin = _ready_key(client, tmp_data_dir)
    _learn_rows(db_path, "tok1", 1.0)  # 2000 tok/s
    _learn_rows(db_path, "tok2", 0.1)  # 20000 tok/s
    assert prefill_model.own_state(db_path, "tok1").rate_for("qwen", 1.0) == (2000.0, "learned")
    assert prefill_model.own_state(db_path, "tok2").rate_for("qwen", 1.0) == (20000.0, "learned")
    assert prefill_model.own_state(db_path, "tok3").rate_for("qwen", 1.0) == (1.0, "hint")


async def test_inflight_learned_figures_are_own_and_no_queue_counts(tmp_data_dir, client):
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    _learn_rows(db_path, "tok1", 1.0)
    _learn_rows(db_path, "tok2", 0.1)
    fleet = prefill_model.PrefillModelState()
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "SELECT model_id, prompt_tokens, cached_tokens, cache_est_tokens, ttft_s,"
            " inflight_same_rank_at_start, gap_s, reusable_tokens FROM request_history"
        )
        names = [d[0] for d in cur.description]
        fleet.replace(prefill_model.build([dict(zip(names, r, strict=True)) for r in cur]))
    client.app.state.prefill_model = fleet
    reg = client.app.state.request_registry
    t0 = time.monotonic()
    est = dict(cache_est_tokens=100, cache_est_raw_tokens=500, est_decayed=True, gap_s=10.0)
    await reg.register(_live(id="a0", token_id="tok1", started_monotonic=t0 - 2, **est))
    await reg.register(_live(id="a1", token_id="tok1", started_monotonic=t0 - 1, **est))
    await reg.register(_live(id="b", token_id="tok2", started_monotonic=t0 - 3))
    kh = client.get("/api/stats/requests", headers=_forest_login(client, plaintext)).json()
    ad = client.get("/api/stats/requests", headers=admin).json()
    k = {r["id"]: r for r in kh["requests"]}
    a = {r["id"]: r for r in ad["requests"]}
    assert set(k) == {"a0", "a1"}
    assert k["a1"]["prefill_tok_s"] == 2000.0 and k["a1"]["prefill_rate_source"] == "learned"
    assert a["a1"]["prefill_tok_s"] == 11000.0
    assert k["a1"]["ahead_count"] is None and k["a1"]["ahead_fresh_tokens"] is None
    assert isinstance(a["a1"]["ahead_count"], int)
    # The proxy's fleet-learned decay is undone: no own gap samples, no decay.
    assert k["a1"]["cache_est_tokens"] == 500 and k["a1"]["est_decayed"] is False
    assert a["a1"]["cache_est_tokens"] == 100


async def test_inflight_without_own_samples_has_no_learned_rate(tmp_data_dir, client):
    db_path, plaintext, _admin = _ready_key(client, tmp_data_dir)
    _learn_rows(db_path, "tok2", 0.1)
    fleet = prefill_model.PrefillModelState()
    fleet.replace({"qwen": prefill_model.ModelStats(rate_tok_s=20000.0, rate_samples=20)})
    client.app.state.prefill_model = fleet
    await client.app.state.request_registry.register(_live(id="a", token_id="tok1"))
    kh = client.get("/api/stats/requests", headers=_forest_login(client, plaintext)).json()
    (row,) = kh["requests"]
    assert row["prefill_tok_s"] is None and row["prefill_rate_source"] is None
    assert row["est_accuracy"] is None


async def test_key_holder_never_gets_a_queued_cause(tmp_data_dir, client):
    """D6: no queue count for a key holder, so no "queued behind N" cause either
    (it rested on the queue count); the expected start stays. Admins keep it."""
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    _learn_rows(db_path, "tok1", 1.0)
    reg = client.app.state.request_registry
    t0 = time.monotonic()
    await reg.register(_live(id="a0", token_id="tok1", started_monotonic=t0 - 2))
    await reg.register(_live(id="a1", token_id="tok1", started_monotonic=t0 - 1))
    kh = client.get("/api/stats/requests", headers=_forest_login(client, plaintext)).json()
    ad = client.get("/api/stats/requests", headers=admin).json()
    k = {r["id"]: r for r in kh["requests"]}
    a = {r["id"]: r for r in ad["requests"]}
    assert a["a1"]["slow_cause"] == "queued"
    assert k["a1"]["slow_cause"] is None and k["a1"]["ahead_count"] is None
    assert k["a1"]["expected_start_s"] is not None
