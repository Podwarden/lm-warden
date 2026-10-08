"""argv and env for the data-parallel layout (#286)."""

import json
import sqlite3
from pathlib import Path

import bcrypt
import pytest

from app.db.repos.models import ModelRow
from app.runtime.backends.vllm.args import build_vllm_args
from app.runtime.backends.vllm.env import build_subprocess_env
from tests.conftest import csrf_header


def _row(**over) -> ModelRow:
    base = dict(
        id="m1",
        served_model_name="qwen",
        hf_repo="a/b",
        hf_revision="main",
        gpu_indices=list(range(7)),
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="registered",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
    )
    base.update(over)
    return ModelRow(**base)


def test_dp_row_emits_one_tp_flag_and_the_dp_flag():
    args = build_vllm_args(_row(data_parallel_size=7), port=10000)
    assert args.count("--tensor-parallel-size") == 1
    assert args[args.index("--tensor-parallel-size") + 1] == "1"
    assert args[args.index("--data-parallel-size") + 1] == "7"


def test_single_replica_row_has_no_dp_flag_and_unchanged_argv():
    args = build_vllm_args(_row(gpu_indices=[0, 1], tensor_parallel_size=2), port=10000)
    assert "--data-parallel-size" not in args
    assert args == [
        "--model", "a/b", "--host", "127.0.0.1", "--port", "10000",
        "--served-model-name", "qwen", "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.9", "--dtype", "auto", "--max-model-len", "2048",
        "--revision", "main", "--scheduling-policy", "priority",
        "--enable-prompt-tokens-details",
    ]  # fmt: skip


def test_pp_strategy_with_dp():
    args = build_vllm_args(
        _row(
            gpu_indices=[0, 1, 2, 3],
            tensor_parallel_size=2,
            data_parallel_size=2,
            parallelism_strategy="pp",
        ),
        port=10000,
    )
    i = args.index("--pipeline-parallel-size")
    assert args[i : i + 2] == ["--pipeline-parallel-size", "2"]
    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert "--tensor-parallel-size" not in args


def test_env_accepts_tp_times_dp():
    env = build_subprocess_env(_row(data_parallel_size=7), hf_token="x", hf_cache_dir="/hfcache")
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6"


def test_env_refuses_tp_seven_dp_seven():
    with pytest.raises(ValueError):
        build_subprocess_env(
            _row(tensor_parallel_size=7, data_parallel_size=7),
            hf_token="x",
            hf_cache_dir="/hfcache",
        )


def test_effective_argv_endpoint_lists_tp_once(client, tmp_data_dir: Path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": list(range(7))}),),
        )
        db.commit()
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
    argv = client.get(f"/api/models/{r.json()['id']}/effective-argv", headers=auth).json()["argv"]
    assert argv.count("--tensor-parallel-size") == 1
    assert argv[argv.index("--data-parallel-size") + 1] == "7"
