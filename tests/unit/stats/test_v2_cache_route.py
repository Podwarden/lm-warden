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
