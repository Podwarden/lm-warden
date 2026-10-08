"""Templates carry data_parallel_size (#286 re-review N4a)."""

import sqlite3

from app.templates.registry import ModelTemplate, template_from_dict, template_to_dict
from tests.conftest import csrf_header, jwt_login, seed_admin_user

TPL = {"id": "dp-tpl", "label": "dp", "hf_repo": "org/model"}


def _login(client, tmp_data_dir, gpus):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=gpus)
    auth = jwt_login(client, username="admin", password="hunter2")
    return {**auth, **csrf_header(client)}


def test_roundtrip_and_legacy_blob_defaults_to_one():
    t = ModelTemplate(
        id="x",
        label="x",
        hf_repo="a/b",
        hf_revision="main",
        dtype="auto",
        max_model_len=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        data_parallel_size=7,
    )
    d = template_to_dict(t)
    assert d["data_parallel_size"] == 7
    assert template_from_dict(d).data_parallel_size == 7
    d.pop("data_parallel_size")
    assert template_from_dict(d).data_parallel_size == 1


def test_explicit_dp_is_saved_and_register_from_template_restores_it(tmp_data_dir, client):
    h = _login(client, tmp_data_dir, list(range(4)))
    r = client.post("/api/models/templates", json={**TPL, "data_parallel_size": 4}, headers=h)
    assert r.status_code == 201, r.text
    listed = {t["id"]: t for t in client.get("/api/models/templates", headers=h).json()}
    assert listed["dp-tpl"]["data_parallel_size"] == 4
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "from-tpl",
            "gpu_indices": [0, 1, 2, 3],
            "template_id": "dp-tpl",
        },
        headers=h,
    )
    assert r.status_code == 201, r.text
    body = client.get(f"/api/models/{r.json()['id']}", headers=h).json()
    assert body["data_parallel_size"] == 4 and body["tensor_parallel_size"] == 1


def test_explicit_body_dp_wins_over_template(tmp_data_dir, client):
    h = _login(client, tmp_data_dir, list(range(4)))
    client.post("/api/models/templates", json={**TPL, "data_parallel_size": 4}, headers=h)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "ovr",
            "gpu_indices": [0],
            "template_id": "dp-tpl",
            "data_parallel_size": 1,
        },
        headers=h,
    )
    assert r.status_code == 201, r.text


def test_save_working_combo_keeps_layout_of_the_live_row(tmp_data_dir, client):
    h = _login(client, tmp_data_dir, list(range(4)))
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "live",
            "hf_repo": "org/model",
            "gpu_indices": [0, 1, 2, 3],
            "data_parallel_size": 2,
        },
        headers=h,
    )
    assert r.status_code == 201, r.text
    live = r.json()["id"]
    r = client.post("/api/models/templates", json={**TPL, "model_id": live}, headers=h)
    assert r.status_code == 201, r.text
    listed = {t["id"]: t for t in client.get("/api/models/templates", headers=h).json()}
    assert listed["dp-tpl"]["data_parallel_size"] == 2
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        assert db.execute("SELECT COUNT(*) FROM engine_templates").fetchone()[0] == 1
