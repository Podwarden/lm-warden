"""Tests for GET /api/models/templates endpoint."""
from tests.conftest import jwt_login, seed_admin_user


# #55 fix — route through the shared barrier+retry helpers.
def _seed_done(db_path):
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def test_list_templates_returns_list(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.get("/api/models/templates", headers=auth)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_list_templates_contains_gpt_oss_20b(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.get("/api/models/templates", headers=auth)
    assert r.status_code == 200
    templates = r.json()
    ids = [t["id"] for t in templates]
    assert "gpt-oss-20b" in ids


def test_gpt_oss_20b_template_shape(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.get("/api/models/templates", headers=auth)
    assert r.status_code == 200
    templates = r.json()
    t = next(x for x in templates if x["id"] == "gpt-oss-20b")
    assert t["hf_repo"] == "openai/gpt-oss-20b"
    assert t["dtype"] == "bfloat16"
    assert t["max_model_len"] == 32000
    assert t["tensor_parallel_size"] == 2
    assert t["gpu_memory_utilization"] == 0.7
    assert t["trust_remote_code"] is True
    assert isinstance(t["extra_args"], list)
    assert isinstance(t["extra_env"], dict)
    assert t["extra_env"].get("VLLM_USE_V1") == "1"


def test_list_templates_requires_auth(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models/templates")
    assert r.status_code == 401


def test_qwen38_27b_fp8_template_shape(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.get("/api/models/templates", headers=auth)
    assert r.status_code == 200
    t = next(x for x in r.json() if x["id"] == "qwen3.8-27b-fp8")
    assert t["hf_repo"] == "Qwen/Qwen3.8-27B-FP8"
    assert t["max_model_len"] == 262144
    assert t["tensor_parallel_size"] == 4
    assert t["gpu_memory_utilization"] == 0.95
    assert t["trust_remote_code"] is False
    assert t["extra_args"] == [
        "--enable-prefix-caching",
        "--kv-cache-dtype", "fp8",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
    ]
    assert t["engine"] is None
    assert t["source"] == "builtin"
