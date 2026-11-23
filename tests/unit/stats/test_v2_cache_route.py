"""GET /api/stats/v2/cache -- the operator prompt-cache observation."""

import sqlite3
import time

import pytest

from tests.conftest import jwt_login, seed_admin_user
from tests.unit.stats.test_v2_requests import _seed_models


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_models(db_path)
    return db_path, jwt_login(client)


def _seed_history(db_path, **over):
    row = dict(
        id="r1",
        finished_at=time.time() - 5,
        model_id="id-model-a",
        model="model-a",
        token_name="key-a",
        prompt_tokens=1000,
        completion_tokens=1,
        duration_s=1.0,
        orphan=0,
        started_iso="2026-09-05T18:59:00Z",
    )
    row.update(over)
    with sqlite3.connect(db_path) as db:
        db.execute(
            f"INSERT INTO request_history({','.join(row)}) VALUES ({','.join('?' * len(row))})",
            tuple(row.values()),
        )
        db.commit()


def test_cache_route_requires_a_session(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert client.get("/api/stats/v2/cache").status_code == 401


def test_cache_route_shape_and_prefill_saved(tmp_data_dir, client):
    db_path, auth = _ready(client, tmp_data_dir)
    _seed_history(
        db_path,
        cached_tokens=900,
        reusable_tokens=1000,
        cache_outcome="hit",
        cached_source="engine",
    )
    r = client.get("/api/stats/v2/cache?range=1h", headers=auth)
    assert r.status_code == 200
    body = r.json()
    [m] = body["models"]
    assert m["model"] == "model-a"
    assert m["backend"] in ("vllm", "llamacpp", None)
    assert m["rate_source"] in ("learned", "hint")
    assert m["prefill_saved_s"] == pytest.approx(900 / 2000.0)  # hint rate, no learning yet
    assert m["outcomes"]["hit"] == 1 and m["top_diverging"] == []


def test_cache_route_rejects_bad_range(tmp_data_dir, client):
    _, auth = _ready(client, tmp_data_dir)
    assert client.get("/api/stats/v2/cache?range=5y", headers=auth).status_code in (400, 422)


def test_cache_route_reports_the_engine_cache_block(tmp_data_dir, client):
    # The card says "engine caches in ~1.6k-token blocks" from these fields:
    # unknown (or classic 16-token pages) reads None, never a guess.
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo
    from app.proxy.routes_dp import engine_epoch

    db_path, auth = _ready(client, tmp_data_dir)
    _seed_history(db_path, cached_tokens=0, reusable_tokens=1000, cached_source="engine")
    [m] = client.get("/api/stats/v2/cache?range=1h", headers=auth).json()["models"]
    assert m["engine_block_tokens"] is None and m["engine_drops_last_block"] is False
    assert m["engine_match_unit_tokens"] is None

    async def go():
        async with open_db(str(db_path)) as db:
            return await ModelRepo(db).get("id-model-a")

    model = client.portal.call(go)
    epoch = engine_epoch(client.app.state, model)
    client.app.state.engine_block_sizes.set("id-model-a", epoch, 1584, drops_last_block=True)
    [m] = client.get("/api/stats/v2/cache?range=1h", headers=auth).json()["models"]
    assert m["engine_block_tokens"] == 1584 and m["engine_drops_last_block"] is True
    assert m["engine_match_unit_tokens"] == 1584
    client.app.state.engine_block_sizes.set(
        "id-model-a", epoch, 1584, drops_last_block=True, match_unit=16
    )
    [m] = client.get("/api/stats/v2/cache?range=1h", headers=auth).json()["models"]
    assert m["engine_block_tokens"] == 1584 and m["engine_match_unit_tokens"] == 16
    client.app.state.engine_block_sizes.set("id-model-a", epoch, 16)
    [m] = client.get("/api/stats/v2/cache?range=1h", headers=auth).json()["models"]
    assert m["engine_block_tokens"] is None and m["engine_drops_last_block"] is False
