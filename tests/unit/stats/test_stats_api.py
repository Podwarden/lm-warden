import sqlite3
import time

from tests.conftest import jwt_login, seed_admin_user


def _seed_done(db_path):
    # #55 fix — admin user + setup_state via shared barrier-aware helper.
    # The model/samples seed below is domain-specific to this test module
    # and stays inline.
    seed_admin_user(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status) "
            "VALUES ('m1', 'm1', 'o/r', 'main', '[0]', 1, NULL, NULL, 0.9, 0, '[]', 'registered')"
        )
        now_min = int(time.time() // 60)
        db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES ('m1', ?, 5, 100, 50)",
            (now_min,),
        )
        # 25h-old row
        db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES ('m1', ?, 9, 999, 999)",
            (now_min - 25 * 60,),
        )
        db.commit()


def _jwt_auth(client, username="admin", password="hunter2"):
    # #55 fix — delegate to the barrier+retry helper.
    return jwt_login(client, username=username, password=password)


def test_stats_models_filters_by_range(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)
    r = client.get("/api/stats/models?range=24h", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["prompt_tokens"] == 100


def test_stats_range_invalid_returns_400(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_auth(client)
    r = client.get("/api/stats/models?range=banana", headers=auth)
    assert r.status_code == 400


def test_stats_requires_session(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/stats/models")
    assert r.status_code == 401


def test_stats_gpus_returns_name_field(tmp_data_dir, client):
    """gpu_samples.name (added in migration 0013) flows through to the
    /api/stats/gpus response so the historical chart can label gauges."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        now_min = int(time.time() // 60)
        db.execute(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, "
            "memory_used_mib, memory_total_mib, name) "
            "VALUES (0, ?, 87, 12450, 16376, 'NVIDIA RTX A4000')",
            (now_min,),
        )
        # Pre-0013 row: name is NULL. API must still return it (as null).
        db.execute(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, "
            "memory_used_mib, memory_total_mib) "
            "VALUES (1, ?, 0, 100, 16376)",
            (now_min,),
        )
        db.commit()
    auth = _jwt_auth(client)
    r = client.get("/api/stats/gpus?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    rows = r.json()
    by_idx = {row["gpu_index"]: row for row in rows}
    assert by_idx[0]["name"] == "NVIDIA RTX A4000"
    assert by_idx[0]["memory_used_mib"] == 12450
    assert by_idx[1]["name"] is None
