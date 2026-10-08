"""``ModelSpec`` rules for the data-parallel layout (#286)."""

import json
import sqlite3
from pathlib import Path

import bcrypt
import pytest
from pydantic import ValidationError

from app.models.schemas import ModelSpec
from tests.conftest import csrf_header


def _kw(**over):
    base = dict(
        served_model_name="m1",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=list(range(7)),
        gpu_memory_utilization=0.9,
    )
    base.update(over)
    return base


def test_dp_derives_tp_one_on_seven_gpus():
    s = ModelSpec(**_kw(data_parallel_size=7))
    assert s.tensor_parallel_size == 1
    assert s.data_parallel_size == 7


def test_dp_two_on_eight_gpus_derives_tp_four():
    s = ModelSpec(**_kw(gpu_indices=list(range(8)), data_parallel_size=2))
    assert s.tensor_parallel_size == 4


def test_explicit_tp_and_dp_accepted():
    s = ModelSpec(**_kw(gpu_indices=[0, 1, 2, 3], tensor_parallel_size=2, data_parallel_size=2))
    assert (s.tensor_parallel_size, s.data_parallel_size) == (2, 2)


def test_product_mismatch_refused():
    with pytest.raises(ValidationError) as ei:
        ModelSpec(**_kw(gpu_indices=[0, 1, 2, 3], tensor_parallel_size=4, data_parallel_size=2))
    assert "len(gpu_indices)" in str(ei.value)


def test_non_dividing_dp_refused():
    with pytest.raises(ValidationError) as ei:
        ModelSpec(**_kw(data_parallel_size=3))
    assert "must divide" in str(ei.value)


def test_dp_on_llamacpp_refused():
    with pytest.raises(ValidationError) as ei:
        ModelSpec(**_kw(gpu_indices=[0, 1], data_parallel_size=2, backend="llamacpp"))
    assert "vllm" in str(ei.value)


@pytest.mark.parametrize(
    "extra",
    [
        ["--data-parallel-size", "7"],
        ["--tensor-parallel-size=1"],
        ["--pipeline-parallel-size", "2"],
    ],
)
def test_parallel_flags_in_extra_args_refused(extra):
    with pytest.raises(ValidationError) as ei:
        ModelSpec(**_kw(gpu_indices=[0], extra_args=extra))
    msg = str(ei.value)
    assert "tensor_parallel_size" in msg and "data_parallel_size" in msg


def test_spill_threshold_zero_refused():
    with pytest.raises(ValidationError):
        ModelSpec(**_kw(gpu_indices=[0], dp_spill_threshold=0))


def test_dp_affinity_enabled_accepts_string_false():
    s = ModelSpec(**_kw(gpu_indices=[0], dp_affinity_enabled="false"))
    assert s.dp_affinity_enabled is False


def test_defaults():
    s = ModelSpec(**_kw(gpu_indices=[0]))
    assert (s.data_parallel_size, s.dp_affinity_enabled, s.dp_spill_threshold) == (1, True, None)


def _seed_done(db_path: Path) -> None:
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": list(range(7))}),),
        )
        db.commit()


def test_api_create_and_get_publish_the_three_keys(client, tmp_data_dir):
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "demo",
            "hf_repo": "org/model",
            "gpu_indices": list(range(7)),
            "data_parallel_size": 7,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    got = client.get(f"/api/models/{mid}", headers=auth).json()
    assert got["data_parallel_size"] == 7
    assert got["tensor_parallel_size"] == 1
    assert got["dp_affinity_enabled"] is True
    assert got["dp_spill_threshold"] is None
    # The list endpoint publishes the same keys.
    listed = client.get("/api/models", headers=auth).json()
    row = next(
        m for m in (listed["models"] if isinstance(listed, dict) else listed) if m["id"] == mid
    )
    assert row["dp_affinity_enabled"] is True
