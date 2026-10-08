"""GET /api/tokens/{id}/cache -- the own-lens prompt-cache observation."""

import sqlite3
import time

from tests.conftest import csrf_header, seed_admin_user
from tests.unit.tokens.test_tokens_v2_api import _jwt_login


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


def test_token_cache_404_for_unknown(tmp_data_dir, client):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    assert client.get("/api/tokens/nope/cache", headers=auth).status_code == 404


def test_token_cache_rejects_unknown_range(tmp_data_dir, client):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    tid = client.post(
        "/api/tokens", json={"name": "c"}, headers={**auth, **csrf_header(client)}
    ).json()["id"]
    assert client.get(f"/api/tokens/{tid}/cache?range=1y", headers=auth).status_code == 422


def test_token_cache_has_no_cross_token_fields(tmp_data_dir, client):
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    auth = _jwt_login(client)
    tid = client.post(
        "/api/tokens", json={"name": "c"}, headers={**auth, **csrf_header(client)}
    ).json()["id"]
    _seed_history(
        db_path,
        id="mine",
        token_id=tid,
        cached_tokens=0,
        reusable_tokens=1000,
        reusable_tokens_own=0,
        cache_outcome="misrouted",
        cache_outcome_own="cold",
        cached_source="engine",
    )
    # Another key's row must not appear in this key's lens.
    _seed_history(
        db_path,
        id="theirs",
        token_id="other",
        cached_tokens=0,
        cache_outcome="diverged",
        cached_source="engine",
    )
    r = client.get(f"/api/tokens/{tid}/cache?range=24h", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_id"] == tid and body["range"] == "24h"
    [m] = body["models"]
    assert m["requests"] == 1
    assert "top_diverging" not in m
    assert m["outcomes"]["misrouted"] == 0 and m["outcomes"]["cold"] == 1
    assert m["outcomes"]["diverged"] == 0
    assert m["rate_source"] in ("learned", "hint")
