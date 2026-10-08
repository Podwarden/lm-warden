"""GET /api/models/{id}/dp-routing: DpRoutingState joined with per-rank metrics (#286 task 4)."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.proxy.routes_dp import RankScrapeCache
from tests.conftest import bearer, jwt_login, seed_admin_token, seed_admin_user

DP3 = (Path(__file__).parents[2] / "fixtures" / "vllm_metrics_dp3.txt").read_text()


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="x",
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=[0, 1, 2],
        tensor_parallel_size=1,
        data_parallel_size=3,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=["--max-num-seqs", "32"],
        status="loaded",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


@pytest.fixture
def db_path(client: TestClient, tmp_data_dir: Path) -> Path:
    p = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(p, allowed_gpu_indices=[0, 1, 2])
    return p


@pytest.fixture
def auth(client: TestClient) -> dict[str, str]:
    return jwt_login(client)


def _seed(db_path: Path, **over) -> None:
    async def _go():
        async with open_db(str(db_path)) as db:
            await ModelRepo(db).insert(_row(**over))

    asyncio.run(_go())


def _fetch_returning(body=None, error=None):
    calls: list[tuple[str, int]] = []

    async def _fetch(host: str, port: int):
        calls.append((host, port))
        return body, error

    return _fetch, calls


def _install(client, fetch):
    client.app.state.dp_rank_scrape_cache = RankScrapeCache(fetch=fetch)
    client.app.state.supervisor._ports["m1"] = 19099


def test_unknown_model_is_404(client, db_path, auth):
    assert client.get("/api/models/nope/dp-routing", headers=auth).status_code == 404


def test_no_credential_is_401(client, db_path):
    assert client.get("/api/models/m1/dp-routing").status_code == 401


def test_admin_token_is_accepted(client, db_path):
    _seed(db_path, status="registered")
    _, secret = seed_admin_token(db_path)
    r = client.get("/api/models/m1/dp-routing", headers=bearer(secret))
    assert r.status_code == 200


def test_unloaded_single_replica_model(client, db_path, auth):
    _seed(db_path, status="registered", gpu_indices=[0], data_parallel_size=1, extra_args=[])
    r = client.get("/api/models/m1/dp-routing", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["data_parallel_size"] == 1
    assert len(body["ranks"]) == 1
    assert body["since"] is None
    assert body["spill_threshold"] == 64
    assert body["spill_threshold_source"] == "auto"
    assert body["engine_metrics"]["available"] is False
    assert body["engine_metrics"]["error"] == "model not loaded"
    rank = body["ranks"][0]
    assert rank["requests_running"] is None and rank["prefix_cache_hit_rate"] is None


def test_loaded_dp3_joins_counters_and_engine_series(client, db_path, auth, monkeypatch):
    monkeypatch.setattr("app.proxy.routes_dp.engine_epoch", lambda state, model: "e")
    _seed(db_path)
    fetch, calls = _fetch_returning(DP3)
    _install(client, fetch)
    state = client.app.state.dp_routing
    for _ in range(3):
        state.route("m1", dp=3, key="a", threshold=8, pinned=None, affinity_enabled=True, epoch="e")
    r = client.get("/api/models/m1/dp-routing", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == "m1"
    assert body["affinity_enabled"] is True
    assert body["spill_threshold"] == 8
    assert body["spill_threshold_source"] == "auto"
    assert body["since"] is not None
    assert body["totals"]["placed"] == 1 and body["totals"]["sticky"] == 2
    assert body["totals"]["in_flight"] == 3
    assert [x["rank"] for x in body["ranks"]] == [0, 1, 2]
    assert sum(x["sticky"] for x in body["ranks"]) == 2
    assert sum(x["placed"] for x in body["ranks"]) == 1
    assert sum(x["assigned_sessions"] for x in body["ranks"]) == 1
    r0, r1, r2 = body["ranks"]
    assert r0["requests_running"] == 2.0 and r0["kv_cache_usage_perc"] == 0.25
    assert r0["prefix_cache_hits"] == 1234.0 and r0["prefix_cache_queries"] == 1500.0
    assert r0["prefix_cache_hit_rate"] == round(1234 / 1500, 4)
    assert r1["requests_waiting"] == 3.0
    assert r1["prefix_cache_hit_rate"] == 0.5
    assert r2["prefix_cache_queries"] == 0.0 and r2["prefix_cache_hit_rate"] is None
    em = body["engine_metrics"]
    assert em["available"] is True and em["error"] is None and em["scraped_at"]
    assert calls == [("127.0.0.1", 19099)]


def test_setting_threshold_reports_source_setting(client, db_path, auth):
    _seed(db_path, dp_spill_threshold=5, dp_affinity_enabled=0)
    fetch, _ = _fetch_returning(DP3)
    _install(client, fetch)
    body = client.get("/api/models/m1/dp-routing", headers=auth).json()
    assert body["spill_threshold"] == 5
    assert body["spill_threshold_source"] == "setting"
    assert body["affinity_enabled"] is False


def test_scrape_failure_keeps_counters_and_nulls_metrics(client, db_path, auth):
    _seed(db_path)
    fetch, _ = _fetch_returning(None, "ConnectError")
    _install(client, fetch)
    client.app.state.dp_routing.route(
        "m1", dp=3, key="a", threshold=8, pinned=None, affinity_enabled=True
    )
    r = client.get("/api/models/m1/dp-routing", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["engine_metrics"]["available"] is False
    assert body["engine_metrics"]["error"] == "ConnectError"
    assert body["totals"]["placed"] == 1
    assert len(body["ranks"]) == 3
    assert all(x["requests_running"] is None for x in body["ranks"])


def test_scrape_ok_without_engine_series(client, db_path, auth):
    _seed(db_path)
    fetch, _ = _fetch_returning("vllm:num_requests_running 1.0\n")
    _install(client, fetch)
    body = client.get("/api/models/m1/dp-routing", headers=auth).json()
    assert body["engine_metrics"]["available"] is False
    assert body["engine_metrics"]["error"] == "no per-replica series in /metrics"
    assert all(x["kv_cache_usage_perc"] is None for x in body["ranks"])


def test_scrape_is_cached_within_ttl(client, db_path, auth):
    _seed(db_path)
    fetch, calls = _fetch_returning(DP3)
    _install(client, fetch)
    client.get("/api/models/m1/dp-routing", headers=auth)
    client.get("/api/models/m1/dp-routing", headers=auth)
    assert len(calls) == 1


def test_fetch_exception_does_not_break_endpoint(client, db_path, auth):
    _seed(db_path)

    async def boom(host, port):
        raise RuntimeError("kaput")

    _install(client, boom)
    r = client.get("/api/models/m1/dp-routing", headers=auth)
    assert r.status_code == 200
    assert r.json()["engine_metrics"]["available"] is False
    assert "kaput" in r.json()["engine_metrics"]["error"]


def test_default_fetch_is_bounded_by_a_short_timeout():
    from app.proxy import routes_dp

    assert 0 < routes_dp.SCRAPE_TIMEOUT_S <= 2.0


async def test_scrape_of_one_model_does_not_block_another():
    import asyncio

    gate = asyncio.Event()
    calls: list[int] = []

    async def fetch(host, port):
        calls.append(port)
        if port == 1:
            await gate.wait()  # model A's engine is slow
        return "", None

    cache = RankScrapeCache(fetch=fetch)
    slow = asyncio.create_task(cache.get("A", "h", 1))
    await asyncio.sleep(0)
    await asyncio.wait_for(cache.get("B", "h", 2), timeout=1)  # must not wait on A
    assert 2 in calls and not slow.done()
    gate.set()
    await slow
