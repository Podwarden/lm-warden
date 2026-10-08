"""Endpoint tests for ``GET /api/models/{id}/suggest-config`` (#113).

Hermetic — no HF, no nvidia-smi. Mirrors the shape of
``test_fit_preview_route.py``: monkeypatches the discovery seam and
installs a fake GPU probe cache on ``app.state``. The pure heuristic
correctness lives in ``test_suggest.py``; this file locks the wire
shape, auth, 404, and the AWQ→fp8 path through the live route.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import bcrypt

from app.models import discovery as discovery_mod
from app.models import routes_api as routes_api_mod
from tests.conftest import csrf_header
from tests.fakes.fake_hf import (
    FakeHfApi,
    FakeModelInfo,
    FakeSibling,
    make_config_fetcher,
    make_hf_api_factory,
)

GIB = 1024**3
MIB = 1024 * 1024


def _seed_done(db_path: Path) -> None:
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.commit()


def _jwt_login(client) -> dict[str, str]:
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _create_model(
    client, auth, *, hf_repo: str, gpu_indices: list[int], served_model_name: str = "test-model"
) -> str:
    r = client.post(
        "/api/models",
        json={
            "served_model_name": served_model_name,
            "hf_repo": hf_repo,
            "gpu_indices": gpu_indices,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _install_fake_discover(monkeypatch, *, info, config):
    api = FakeHfApi(info)
    hf_factory = make_hf_api_factory(api)
    cfg_fetcher = make_config_fetcher(config)

    async def _fake(repo_id, revision, token):
        return await discovery_mod.discover_repo_files(
            repo_id,
            revision,
            token,
            hf_api_factory=hf_factory,
            config_fetcher=cfg_fetcher,
        )

    monkeypatch.setattr(routes_api_mod, "discover_repo_files", _fake)


@dataclass
class _FakeGpuLive:
    index: int
    name: str
    memory_total_mib: int
    uuid: str = "GPU-fake"
    memory_used_mib: int = 0
    memory_free_mib: int = 0
    utilization_pct: int = 0


@dataclass
class _FakeSnap:
    gpus: list[_FakeGpuLive] = field(default_factory=list)
    apps: list = field(default_factory=list)
    probe_error: str | None = None


class _FakeProbeCache:
    def __init__(self, snap: _FakeSnap) -> None:
        self._snap = snap

    async def get(self) -> _FakeSnap:
        return self._snap


def _install_fake_gpus(client, gpus: list[_FakeGpuLive]) -> None:
    client.app.state.gpu_probe_cache = _FakeProbeCache(_FakeSnap(gpus=gpus))


# ---- Realistic configs ---------------------------------------------------


def _llama_info(repo_id="meta-llama/Llama-3.1-8B-Instruct") -> FakeModelInfo:
    return FakeModelInfo(
        id=repo_id,
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("tokenizer.json", size=2048),
            FakeSibling("model-00001-of-00002.safetensors", size=8 * GIB),
            FakeSibling("model-00002-of-00002.safetensors", size=1 * GIB),
        ],
    )


_LLAMA_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}


def _gemma_awq_info() -> FakeModelInfo:
    return FakeModelInfo(
        id="hugging-quants/gemma-2-27b-it-AWQ-INT4",
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("model.safetensors", size=15 * GIB),
        ],
    )


_GEMMA_AWQ_CONFIG = {
    "hidden_size": 4608,
    "num_hidden_layers": 46,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "max_position_embeddings": 8192,
    "torch_dtype": "float16",
    "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
}


# ---- Happy path: non-AWQ model ------------------------------------------


def test_suggest_config_returns_starting_points_for_dense_model(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_llama_info(), config=_LLAMA_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="A4000", memory_total_mib=24 * 1024),
        ],
    )
    model_id = _create_model(
        client,
        auth,
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        gpu_indices=[0],
    )

    r = client.get(f"/api/models/{model_id}/suggest-config", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gpu_memory_utilization"] == 0.92
    assert body["max_model_len"] == 131072
    assert body["kv_cache_dtype"] is None
    assert "starting points" in body["disclaimer"].lower()
    assert "never auto-applied" in body["disclaimer"].lower()


# ---- #113 — AWQ-INT4 model triggers fp8 KV cache recommendation ---------


def test_suggest_config_awq_model_recommends_fp8_kv_cache(tmp_data_dir, client, monkeypatch):
    """This is the exact failure mode #113 documents: a 27B AWQ-INT4
    model that crashes OOM at long context because vLLM keeps the KV
    cache at bf16 by default. The suggest endpoint should surface
    ``kv_cache_dtype='fp8'`` so the wizard prompts the operator to flip
    it."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_gemma_awq_info(), config=_GEMMA_AWQ_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="A4000", memory_total_mib=24 * 1024),
            _FakeGpuLive(index=1, name="A4000", memory_total_mib=24 * 1024),
        ],
    )
    model_id = _create_model(
        client,
        auth,
        hf_repo="hugging-quants/gemma-2-27b-it-AWQ-INT4",
        gpu_indices=[0, 1],
        served_model_name="gemma-awq",
    )

    r = client.get(f"/api/models/{model_id}/suggest-config", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kv_cache_dtype"] == "fp8"
    assert body["max_model_len"] == 8192


# ---- 404 on unknown model_id --------------------------------------------


def test_suggest_config_404_when_model_missing(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    r = client.get("/api/models/nonexistent-id/suggest-config", headers=auth)
    assert r.status_code == 404


# ---- Auth required ------------------------------------------------------


def test_suggest_config_requires_jwt(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models/anything/suggest-config")
    assert r.status_code in (401, 403)


# ---- Disclaimer field is ALWAYS present (contract guard) ----------------


def test_suggest_config_payload_always_has_disclaimer(tmp_data_dir, client, monkeypatch):
    """Hard contract: the FE relies on ``disclaimer`` being present in
    every response (it's how the wizard renders the warning banner).
    Regression guard against a future refactor that omits it on the
    'no recommendations' branch."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_llama_info(), config={})
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="A4000", memory_total_mib=24 * 1024),
        ],
    )
    model_id = _create_model(
        client,
        auth,
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        gpu_indices=[0],
    )

    r = client.get(f"/api/models/{model_id}/suggest-config", headers=auth)
    assert r.status_code == 200, r.text
    assert "disclaimer" in r.json()
