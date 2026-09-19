"""S5 (#104) API surface coverage:

  * POST   /api/tokens with priority
  * PATCH  /api/tokens/{id}
  * GET    /api/tokens/{id}/usage
  * POST   /api/tokens/{id}/test
  * GET    /api/tokens  surfaces priority + usage_24h
  * Pydantic Field(ge=, le=) bounds → 422 (NOT 500)
  * the removed per-token rate limit: 422 on create/PATCH, absent everywhere
"""

import sqlite3

from tests.conftest import csrf_header, seed_admin_user


def _seed_done(db_path):
    seed_admin_user(db_path)


def _jwt_login(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_create_with_priority(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/tokens",
        json={"name": "fast", "priority": 9},
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["priority"] == 9
    assert "rate_limit_tps" not in body


def test_create_defaults_priority_5(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post("/api/tokens", json={"name": "default"}, headers={**auth, **h})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["priority"] == 5


def test_create_rejects_priority_out_of_range(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/tokens",
        json={"name": "bad", "priority": 99},
        headers={**auth, **h},
    )
    assert r.status_code == 422


def test_create_rejects_the_removed_rate_limit(tmp_data_dir, client):
    """Per-token rate limits were removed. A client still sending one gets a
    422 naming the field -- any value, null included -- rather than a 2xx that
    silently sets no limit. Nothing is created."""
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    for value in (500, None, 0):
        r = client.post(
            "/api/tokens",
            json={"name": "bad", "rate_limit_tps": value},
            headers={**auth, **h},
        )
        assert r.status_code == 422, (value, r.text)
        assert "rate_limit_tps" in r.text
    assert client.get("/api/tokens", headers=auth).json()["items"] == []


def test_patch_rejects_the_removed_rate_limit(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    tid = client.post(
        "/api/tokens", json={"name": "p", "priority": 5}, headers={**auth, **h},
    ).json()["id"]
    for value in (200, None):
        r = client.patch(
            f"/api/tokens/{tid}",
            json={"rate_limit_tps": value, "priority": 8},
            headers={**auth, **h},
        )
        assert r.status_code == 422, (value, r.text)
        assert "rate_limit_tps" in r.text
    # The whole body was refused: priority is untouched.
    assert client.get(f"/api/tokens/{tid}", headers=auth).json()["priority"] == 5


def test_patch_updates_priority_only(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    created = client.post(
        "/api/tokens",
        json={"name": "p", "priority": 5},
        headers={**auth, **h},
    )
    tid = created.json()["id"]

    r = client.patch(f"/api/tokens/{tid}", json={"priority": 8}, headers={**auth, **h})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["priority"] == 8
    assert body["name"] == "p"  # untouched
    assert "rate_limit_tps" not in body
    assert "rate_limit_tps" not in client.get(f"/api/tokens/{tid}", headers=auth).json()


def test_patch_404_for_unknown_token(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.patch(
        "/api/tokens/deadbeefdeadbeefdeadbeefdeadbeef",
        json={"priority": 4},
        headers={**auth, **h},
    )
    assert r.status_code == 404


def test_list_surfaces_priority_and_usage_24h(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/tokens",
        json={"name": "list", "priority": 6},
        headers={**auth, **h},
    )
    tid = create.json()["id"]

    # Seed a usage row directly so the totals are non-zero.
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO token_usage_minute(token_id, minute, requests, "
            "                               prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?)",
            (tid, 99999999, 3, 42, 17),  # minute far in the past — still in 24h window? no
        )
        db.commit()

    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = {it["id"]: it for it in r.json()["items"]}
    assert tid in items
    item = items[tid]
    assert "rate_limit_tps" not in item
    assert item["priority"] == 6
    assert "usage_24h" in item
    assert set(item["usage_24h"].keys()) == {
        "requests", "prompt_tokens", "completion_tokens", "total_tokens"
    }


def test_get_usage_returns_buckets_and_totals(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "u"}, headers={**auth, **h})
    tid = create.json()["id"]

    # Insert 3 usage rows close enough to now() that they fall inside the
    # default 24h window.
    import time
    minute_now = int(time.time() // 60)
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        for offset in (-10, -5, -1):
            db.execute(
                "INSERT INTO token_usage_minute(token_id, minute, requests, "
                "                               prompt_tokens, completion_tokens) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, minute_now + offset, 1, 10, 5),
            )
        db.commit()

    r = client.get(f"/api/tokens/{tid}/usage?range=24h", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["range"] == "24h"
    assert len(body["buckets"]) == 3
    assert body["totals"]["requests"] == 3
    assert body["totals"]["prompt_tokens"] == 30
    assert body["totals"]["completion_tokens"] == 15
    assert body["totals"]["total_tokens"] == 45


def test_get_usage_rejects_unknown_range(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "u"}, headers={**auth, **h})
    tid = create.json()["id"]

    r = client.get(f"/api/tokens/{tid}/usage?range=1y", headers=auth)
    assert r.status_code == 422


def test_get_usage_404_for_unknown_token(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.get("/api/tokens/deadbeefdeadbeefdeadbeefdeadbeef/usage", headers=auth)
    assert r.status_code == 404


def test_post_test_returns_token_health_summary(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/tokens",
        json={"name": "probe", "priority": 7},
        headers={**auth, **h},
    )
    tid = create.json()["id"]

    r = client.post(f"/api/tokens/{tid}/test", headers={**auth, **h})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_id"] == tid
    assert body["ok"] is True
    assert "rate_limit_tps" not in body
    assert body["priority"] == 7
    assert body["revoked"] is False
    # proxy_reachable depends on the test harness — must at least be a bool.
    assert isinstance(body["proxy_reachable"], bool)
    # allowed_models is a list (empty if no models loaded in this test app).
    assert isinstance(body["allowed_models"], list)


def test_post_test_reports_paused(tmp_data_dir, client):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    tid = client.post("/api/tokens", json={"name": "probe"}, headers={**auth, **h}).json()["id"]

    body = client.post(f"/api/tokens/{tid}/test", headers={**auth, **h}).json()
    assert body["paused"] is False

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE api_tokens SET paused_at = datetime('now') WHERE id = ?", (tid,))
        db.commit()
    body = client.post(f"/api/tokens/{tid}/test", headers={**auth, **h}).json()
    assert body["paused"] is True
