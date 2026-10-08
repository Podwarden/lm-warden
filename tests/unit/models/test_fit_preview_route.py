"""Endpoint tests for ``POST /api/models/fit-preview`` (#85).

Hermetic — no HF, no nvidia-smi. We monkeypatch the route's discovery
seam (same hook ``test_discover_route.py`` uses) and install a fake
GPU probe cache on ``app.state``. The route's job is to glue config.json
+ file list + GPU VRAM together and call the pure helpers in
``app/models/fit.py``; we lock the wire shape + verdict here, the
numerical correctness of the math lives in ``test_fit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.models import discovery as discovery_mod
from app.models import routes_api as routes_api_mod
from tests.conftest import csrf_header, jwt_login, seed_admin_user
from tests.fakes.fake_hf import (
    FakeHfApi,
    FakeModelInfo,
    FakeSibling,
    make_config_fetcher,
    make_hf_api_factory,
)

GIB = 1024**3
MIB = 1024 * 1024


# #55 fix — route through the shared barrier+retry helpers.
def _seed_done(db_path: Path) -> None:
    seed_admin_user(db_path)


def _jwt_login(client) -> dict[str, str]:
    return jwt_login(client)


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
    return api


# ---- Fake GPU probe ------------------------------------------------------


@dataclass
class _FakeGpuLive:
    index: int
    name: str
    memory_total_mib: int
    uuid: str = "GPU-fake"
    memory_used_mib: int = 0
    memory_free_mib: int = 0
    utilization_pct: int = 0
    # #176 — default sm_86 (RTX A4000); tests override for other arches.
    compute_cap: float | None = 8.6


@dataclass
class _FakeSnap:
    gpus: list[_FakeGpuLive] = field(default_factory=list)
    apps: list = field(default_factory=list)
    probe_error: str | None = None


class _FakeProbeCache:
    """Drop-in for ``app.system.routes_gpus._ProbeCache``.

    The fit-preview route reads ``app.state.gpu_probe_cache`` and calls
    ``get()`` — we install one of these before the request so the route
    never shells out to nvidia-smi.
    """

    def __init__(self, snap: _FakeSnap) -> None:
        self._snap = snap

    async def get(self) -> _FakeSnap:
        return self._snap


def _install_fake_gpus(client, gpus: list[_FakeGpuLive], probe_error: str | None = None):
    snap = _FakeSnap(gpus=gpus, probe_error=probe_error)
    client.app.state.gpu_probe_cache = _FakeProbeCache(snap)


# ---- Realistic Qwen2.5-7B-Instruct config + shards -----------------------


def _qwen_info() -> FakeModelInfo:
    return FakeModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("tokenizer.json", size=2048),
            FakeSibling("model-00001-of-00002.safetensors", size=5 * GIB),
            FakeSibling("model-00002-of-00002.safetensors", size=2 * GIB + GIB // 2),
            FakeSibling("README.md", size=4096),
        ],
    )


_QWEN_CONFIG = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
}


# ---- Happy path: green verdict ------------------------------------------


def test_fit_preview_green_on_2x_a4000(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "filename": "model-00001-of-00002.safetensors",
            "gpu_indices": [0, 1],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 7.5 GiB aggregate weights / ~28 GiB budget -> green.
    assert body["verdict"] == "green"
    # Shard aggregation: file_size must reflect both shards summed.
    assert body["breakdown"]["file_size"] == 5 * GIB + (2 * GIB + GIB // 2)
    assert body["breakdown"]["total_vram"] == 2 * 16376 * MIB
    assert body["breakdown"]["dtype_bytes"] == 2
    # No recommendation needed for green.
    assert body["recommended_max_model_len"] is None
    # No GGUF warning for a safetensors file.
    assert not any("gguf" in w for w in body["warnings"])


# ---- Red verdict + recommendation ---------------------------------------


def test_fit_preview_red_on_single_a4000_with_recommendation(tmp_data_dir, client, monkeypatch):
    """Pick the 5 GiB shard alone on a single 16 GiB card with max context +
    batch=4. Shard aggregation makes file_size 7.5 GiB, KV at 32 K * 4 batches
    eats budget hard so the verdict lands at orange/red and the route returns
    a recommended ``max_model_len`` strictly below 32 768."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "filename": "model-00001-of-00002.safetensors",
            "gpu_indices": [0],
            "max_batch_size": 4,
            "gpu_memory_utilization": 0.9,
            # Force using max_position_embeddings = 32768 from the config.
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 7.5 GiB weights + KV reserve for 32 K * 4 batches on a 16 GiB card -> red.
    assert body["verdict"] in ("orange", "red")
    # Recommendation should be a positive int below the configured 32 K.
    rec = body["recommended_max_model_len"]
    assert rec is not None
    assert 0 < rec < 32768


# ---- Shard aggregation (#88) — covers safetensors + multi-part GGUF -----


def test_fit_preview_aggregates_7_shard_safetensors_set(tmp_data_dir, client, monkeypatch):
    """Pick shard 1 of a 7-shard safetensors set: file_size must be the SUM
    of all 7 shards (not the slice). This is the canonical shard-aggregation
    contract — without it a 70 GB Llama-3 70B set shipped as 7x10 GB shards
    classifies green on a 24 GB card because we measure 10 GB / 22 GB budget
    instead of 70 GB / 22 GB. Fit-preview must call ``classify_fit`` exactly
    once against the aggregate."""
    info = FakeModelInfo(
        id="meta-llama/Meta-Llama-3-70B",
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("tokenizer.json", size=2048),
            # 7 shards x 10 GiB each = 70 GiB aggregate.
            *[
                FakeSibling(f"model-{i:05d}-of-00007.safetensors", size=10 * GIB)
                for i in range(1, 8)
            ],
            FakeSibling("README.md", size=4096),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=info, config=_QWEN_CONFIG)
    # Single A4000 — 16 GiB, well under 70 GiB aggregate; we expect red.
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "meta-llama/Meta-Llama-3-70B",
            # Pick shard 1 — aggregation must hit the whole set, not just 10 GiB.
            "filename": "model-00001-of-00007.safetensors",
            "gpu_indices": [0],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 7 * 10 GiB = 70 GiB aggregate, NOT the 10 GiB slice.
    assert body["breakdown"]["file_size"] == 7 * 10 * GIB
    assert body["verdict"] == "red"


def test_fit_preview_aggregates_multi_part_gguf_split(tmp_data_dir, client, monkeypatch):
    """Multi-part GGUF splits (e.g. llama.cpp's ``model-Q5_K_M-00001-of-00003.gguf``
    convention) must aggregate the same way safetensors shards do. Without
    this fix the wizard would let an operator green-light a 40 GB GGUF on a
    16 GB card because the per-shard size (e.g. 14 GB) fits the budget
    even though the whole set won't load at once.

    This is #88's extension of the #85 safetensors-only aggregator."""
    info = FakeModelInfo(
        id="unsloth/Llama-3.3-70B-GGUF",
        siblings=[
            # Three-part Q5_K_M split: 14 + 14 + 12 = 40 GiB aggregate.
            FakeSibling("Llama-3.3-70B-Q5_K_M-00001-of-00003.gguf", size=14 * GIB),
            FakeSibling("Llama-3.3-70B-Q5_K_M-00002-of-00003.gguf", size=14 * GIB),
            FakeSibling("Llama-3.3-70B-Q5_K_M-00003-of-00003.gguf", size=12 * GIB),
            # A separate single-file quant (must NOT be summed in).
            FakeSibling("Llama-3.3-70B-Q3_K_S.gguf", size=22 * GIB),
            FakeSibling("README.md", size=4096),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    # GGUF repos often lack config.json — pass None and expect the existing
    # config_incomplete warning path; that's orthogonal to shard aggregation.
    _install_fake_discover(monkeypatch, info=info, config=None)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "unsloth/Llama-3.3-70B-GGUF",
            # Pick shard 2 — aggregation must hit all three parts.
            "filename": "Llama-3.3-70B-Q5_K_M-00002-of-00003.gguf",
            "gpu_indices": [0, 1],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 14 + 14 + 12 = 40 GiB aggregate, NOT just the 14 GiB slice and NOT
    # bleeding in the unrelated Q3_K_S single file.
    assert body["breakdown"]["file_size"] == (14 + 14 + 12) * GIB
    # 40 GiB > ~29 GiB budget on 2x A4000 → red.
    assert body["verdict"] == "red"
    # Sanity: the GGUF dequant warning still fires for any .gguf filename.
    assert any("gguf" in w.lower() for w in body["warnings"])


def test_fit_preview_does_not_cross_shard_boundary_between_gguf_quant_families(
    tmp_data_dir,
    client,
    monkeypatch,
):
    """Two GGUF quant families (Q4_K_M and Q5_K_M) shipped as 3-shard sets in
    the same repo MUST stay separated by ``of-N`` + prefix. Picking the
    Q5_K_M shard 1 must aggregate only the three Q5 parts, not also bleed
    in Q4 — otherwise we'd double-count and over-classify red."""
    info = FakeModelInfo(
        id="unsloth/Llama-3.3-70B-GGUF",
        siblings=[
            # Q5 family — 14 + 14 + 12 = 40 GiB.
            FakeSibling("Llama-3.3-70B-Q5_K_M-00001-of-00003.gguf", size=14 * GIB),
            FakeSibling("Llama-3.3-70B-Q5_K_M-00002-of-00003.gguf", size=14 * GIB),
            FakeSibling("Llama-3.3-70B-Q5_K_M-00003-of-00003.gguf", size=12 * GIB),
            # Q4 family — 11 + 11 + 10 = 32 GiB.
            FakeSibling("Llama-3.3-70B-Q4_K_M-00001-of-00003.gguf", size=11 * GIB),
            FakeSibling("Llama-3.3-70B-Q4_K_M-00002-of-00003.gguf", size=11 * GIB),
            FakeSibling("Llama-3.3-70B-Q4_K_M-00003-of-00003.gguf", size=10 * GIB),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=info, config=None)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "unsloth/Llama-3.3-70B-GGUF",
            "filename": "Llama-3.3-70B-Q5_K_M-00001-of-00003.gguf",
            "gpu_indices": [0, 1],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Aggregation locked to the Q5_K_M family only.
    assert body["breakdown"]["file_size"] == (14 + 14 + 12) * GIB


# ---- Per-file (GGUF) gets the dequant warning ---------------------------


def test_fit_preview_gguf_emits_dequant_warning(tmp_data_dir, client, monkeypatch):
    info = FakeModelInfo(
        id="unsloth/Qwen3.6-27B-MTP-GGUF",
        siblings=[
            FakeSibling("Qwen3.6-27B-Q5_K_M.gguf", size=19 * GIB + GIB // 2),
            FakeSibling("Qwen3.6-27B-Q4_K_M.gguf", size=16 * GIB),
            FakeSibling("README.md", size=4096),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    # GGUF repos often have no config.json; missing config => warning, not 500.
    _install_fake_discover(monkeypatch, info=info, config=None)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
            "filename": "Qwen3.6-27B-Q5_K_M.gguf",
            "gpu_indices": [0, 1],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 19.5 GiB / 28.8 GiB cap ~= 0.68 -> yellow.
    assert body["verdict"] in ("yellow", "orange")
    assert any("gguf" in w.lower() for w in body["warnings"])
    assert any("config_incomplete" in w for w in body["warnings"])


# ---- Capability warnings (#176) -----------------------------------------


_FP8_CONFIG = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
    # FP8 W8A8 via llmcompressor / compressed-tensors.
    "quantization_config": {"quant_method": "fp8"},
}


def test_fit_preview_warns_fp8_on_sm86(tmp_data_dir, client, monkeypatch):
    """An FP8 candidate on a sm_86 A4000 (cc 8.6 < 8.9) must append a
    capability warning to the same warnings[] list — emulated FP8, prefer
    AWQ/GPTQ INT4 or bf16."""
    info = FakeModelInfo(
        id="neuralmagic/Qwen2.5-7B-Instruct-FP8",
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("model.safetensors", size=8 * GIB),
            FakeSibling("README.md", size=4096),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=info, config=_FP8_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376, compute_cap=8.6),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376, compute_cap=8.6),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "neuralmagic/Qwen2.5-7B-Instruct-FP8",
            "filename": "model.safetensors",
            "gpu_indices": [0, 1],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    warnings = r.json()["warnings"]
    assert any("fp8" in w.lower() and "emulated" in w.lower() for w in warnings), warnings
    assert any("awq" in w.lower() or "bf16" in w.lower() for w in warnings), warnings


def test_fit_preview_no_capability_warning_for_fp8_on_ada(tmp_data_dir, client, monkeypatch):
    """Same FP8 candidate on an Ada card (cc 8.9) → no emulated-fp8 warning."""
    info = FakeModelInfo(
        id="neuralmagic/Qwen2.5-7B-Instruct-FP8",
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("model.safetensors", size=8 * GIB),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=info, config=_FP8_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA L40S", memory_total_mib=46068, compute_cap=8.9),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "neuralmagic/Qwen2.5-7B-Instruct-FP8",
            "filename": "model.safetensors",
            "gpu_indices": [0],
            "max_batch_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    warnings = r.json()["warnings"]
    assert not any("fp8" in w.lower() and "emulated" in w.lower() for w in warnings), warnings


# ---- Validation errors ---------------------------------------------------


def test_fit_preview_404_when_filename_not_in_repo(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "filename": "model.fake-quant.bin",
            "gpu_indices": [0],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "filename_not_found"


def test_fit_preview_422_when_gpu_index_missing_from_probe(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "filename": "config.json",  # exists, but request gpu_indices=[5] doesn't.
            "gpu_indices": [5],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "gpu_index_missing"
    assert body["detail"]["available"] == [0]


def test_fit_preview_requires_auth(tmp_data_dir, client):
    """No JWT cookie -> ``require_jwt`` must reject. We pass the CSRF token so
    the request gets past the CSRF middleware (otherwise it 403s before the
    auth dep ever runs); this isolates the assertion to the auth check."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "o/r",
            "filename": "file.safetensors",
            "gpu_indices": [0],
        },
        headers=csrf_header(client),
    )
    assert r.status_code == 401


# ---- HF error envelopes ---------------------------------------------------


def _make_fake_response(status_code: int):
    class _Req:
        method = "GET"
        url = "https://huggingface.co/api/models/fake"

    class _R:
        def __init__(self, code: int) -> None:
            self.status_code = code
            self.headers: dict[str, str] = {}
            self.text = ""
            self.request = _Req()

        def json(self):
            return {}

    return _R(status_code)


def test_fit_preview_propagates_hf_auth_error(tmp_data_dir, client, monkeypatch):
    """Gated repo without a valid token: route surfaces 401 auth_required."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    from huggingface_hub.utils import GatedRepoError

    _install_fake_discover(
        monkeypatch,
        info=GatedRepoError("gated", response=_make_fake_response(403)),
        config={},
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "meta-llama/Llama-3-70B",
            "filename": "model.safetensors",
            "gpu_indices": [0],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 401
    body = r.json()
    assert body["detail"]["error_code"] == "auth_required"


def test_fit_preview_propagates_hf_not_found(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    from huggingface_hub.utils import RepositoryNotFoundError

    _install_fake_discover(
        monkeypatch,
        info=RepositoryNotFoundError("nope", response=_make_fake_response(404)),
        config={},
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "does/not-exist",
            "filename": "anything.safetensors",
            "gpu_indices": [0],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["detail"]["error_code"] == "repo_not_found"


# ---- Backend-aware budget + real KV shape --------------------------------
#
# Reported from a live install: gpt-oss-20b-Q8_0.gguf (11.28 GiB) showed
# "won't fit" on a 16 GiB card with llama.cpp selected. Three separate errors
# compounded -- vLLM's gpu_memory_utilization applied to a backend with no
# such flag, `head_dim` derived (45) instead of read from the config (64),
# and 12 sliding-attention layers charged the full 131072-token context.

_GPT_OSS_CONFIG = {
    "hidden_size": 2880,
    "num_hidden_layers": 24,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
    "head_dim": 64,
    "sliding_window": 128,
    "layer_types": ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(24)],
}


def _gpt_oss_info():
    return FakeModelInfo(
        id="openai/gpt-oss-20b-GGUF",
        siblings=[
            FakeSibling("gpt-oss-20b-Q8_0.gguf", int(11.28 * GIB)),
            FakeSibling("config.json", 1024),
        ],
    )


def _post_fit(client, auth, **over):
    body = {
        "repo_id": "openai/gpt-oss-20b-GGUF",
        "filename": "gpt-oss-20b-Q8_0.gguf",
        "gpu_indices": [0],
        "gpu_memory_utilization": 0.9,
    }
    body.update(over)
    return client.post(
        "/api/models/fit-preview", json=body, headers={**auth, **csrf_header(client)}
    )


def _setup(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_gpt_oss_info(), config=_GPT_OSS_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="Quadro RTX 5000", memory_total_mib=16384, compute_cap=7.5),
        ],
    )
    return auth


def test_llamacpp_is_not_charged_vllms_gpu_memory_utilization(tmp_data_dir, client, monkeypatch):
    """llama-server has no gpu_memory_utilization flag, so 0.9 must not apply.

    The operator's value is still in the payload -- the dialog always sends
    it -- and must be ignored rather than silently shrinking the card by 10%.
    """
    auth = _setup(tmp_data_dir, client, monkeypatch)

    llama = _post_fit(client, auth, backend="llamacpp").json()
    vllm = _post_fit(client, auth, backend="vllm").json()

    # Same card, same weights, same KV -- the ONLY difference is the cap.
    assert llama["breakdown"]["total_vram"] == vllm["breakdown"]["total_vram"]
    assert llama["breakdown"]["kv_reserve"] == vllm["breakdown"]["kv_reserve"]
    assert llama["breakdown"]["weights_budget"] > vllm["breakdown"]["weights_budget"]

    # And the reported defect: it fits on llama.cpp.
    assert vllm["verdict"] != "green"
    assert llama["verdict"] != "red", (
        "11.28 GiB of weights on a 16 GiB card is not 'won't fit' for an "
        "engine that does not reserve a fractional cap"
    )


def test_fit_preview_defaults_to_vllm_when_backend_is_omitted(tmp_data_dir, client, monkeypatch):
    """Back-compat: an older client that sends no `backend` is unchanged."""
    auth = _setup(tmp_data_dir, client, monkeypatch)
    omitted = _post_fit(client, auth).json()
    explicit = _post_fit(client, auth, backend="vllm").json()
    assert omitted["breakdown"] == explicit["breakdown"]
    assert omitted["verdict"] == explicit["verdict"]


def test_fit_preview_rejects_an_unknown_backend(tmp_data_dir, client, monkeypatch):
    """Refuse rather than silently pricing the model for the wrong engine."""
    auth = _setup(tmp_data_dir, client, monkeypatch)
    r = _post_fit(client, auth, backend="not-a-backend")
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "unknown_backend"


def test_kv_reserve_reads_head_dim_and_discounts_sliding_layers(tmp_data_dir, client, monkeypatch):
    """The KV term must reflect the config, not a derivation of it.

    12 of the 24 layers slide with a 128-token window; the other 12 hold the
    full 131072. Per layer per token = 2 * 8 * 64 * 2 = 2048 bytes.
    """
    auth = _setup(tmp_data_dir, client, monkeypatch)
    body = _post_fit(client, auth, backend="llamacpp").json()
    expected = 2048 * (12 * 131072 + 12 * 128)
    assert body["breakdown"]["kv_reserve"] == expected


# ---- Field parity with POST /api/models ---------------------------------
#
# These two endpoints describe the same model moments apart in the Add Model
# flow, and they used to disagree on both of their headline fields:
# ``repo_id`` here vs ``hf_repo`` on create, and ``filename`` required here
# vs optional there. A caller who wrote the create body and then tried the
# pre-check got ``{"type":"missing","loc":["body","repo_id"]}`` -- a field
# name that appears nowhere in the endpoint they had just used.


def test_fit_preview_accepts_hf_repo_the_create_endpoints_spelling(
    tmp_data_dir,
    client,
    monkeypatch,
):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
            "filename": "model-00001-of-00002.safetensors",
            "gpu_indices": [0, 1],
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "green"


def test_fit_preview_still_accepts_repo_id(tmp_data_dir, client, monkeypatch):
    """The frontend and any existing client send ``repo_id``; it must keep
    working, or aligning the names would be a breaking change dressed up as
    a consistency fix."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "filename": "model-00001-of-00002.safetensors",
            "gpu_indices": [0],
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text


def test_fit_preview_requires_one_of_the_two_repo_fields(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.post(
        "/api/models/fit-preview",
        json={
            "filename": "model-00001-of-00002.safetensors",
            "gpu_indices": [0],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422
    assert "hf_repo is required" in r.text


def test_fit_preview_resolves_the_weights_file_when_filename_is_omitted(
    tmp_data_dir,
    client,
    monkeypatch,
):
    """``filename`` is optional, as on POST /api/models. An ordinary
    multi-file HF repo carries exactly one weights format, so making the
    caller name a shard was asking them to repeat what discovery knew."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_qwen_info(), config=_QWEN_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
            _FakeGpuLive(index=1, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
            "gpu_indices": [0, 1],
            "max_model_len": 4096,
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Resolving a shard must still aggregate the WHOLE set -- the same answer
    # naming shard 1 explicitly gives.
    assert body["breakdown"]["file_size"] == 5 * GIB + (2 * GIB + GIB // 2)
    assert body["verdict"] == "green"


def test_fit_preview_will_not_guess_between_gguf_quantisations(
    tmp_data_dir,
    client,
    monkeypatch,
):
    """A GGUF repo publishes several INDEPENDENT quantisations that fit very
    differently. Picking one would answer a question the caller did not ask,
    so this is the one case where ``filename`` is still required -- and the
    error lists the candidates instead of saying "Field required"."""
    info = FakeModelInfo(
        id="unsloth/Llama-3.3-70B-GGUF",
        siblings=[
            FakeSibling("Llama-3.3-70B-Q5_K_M.gguf", size=40 * GIB),
            FakeSibling("Llama-3.3-70B-Q3_K_S.gguf", size=22 * GIB),
            FakeSibling("README.md", size=4096),
        ],
    )
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=info, config=None)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="NVIDIA RTX A4000", memory_total_mib=16376),
        ],
    )

    r = client.post(
        "/api/models/fit-preview",
        json={
            "hf_repo": "unsloth/Llama-3.3-70B-GGUF",
            "gpu_indices": [0],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["error_code"] == "filename_required"
    assert set(detail["candidates"]) == {
        "Llama-3.3-70B-Q5_K_M.gguf",
        "Llama-3.3-70B-Q3_K_S.gguf",
    }


# ---- #286: data-parallel replicas -----------------------------------------


def _setup_seven_gpus(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_gpt_oss_info(), config=_GPT_OSS_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=i, name="Quadro RTX 5000", memory_total_mib=32768, compute_cap=7.5)
            for i in range(7)
        ],
    )
    return auth


def test_fit_preview_data_parallel_sizes_one_replica(tmp_data_dir, client, monkeypatch):
    auth = _setup_seven_gpus(tmp_data_dir, client, monkeypatch)
    r = _post_fit(client, auth, gpu_indices=list(range(7)), data_parallel_size=7)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["breakdown"]["total_vram"] == 32 * GIB
    assert body["breakdown"]["replicas"] == 7
    assert body["breakdown"]["gpus_per_replica"] == 1
    assert any(w.startswith("data_parallel:") for w in body["warnings"])


def test_fit_preview_data_parallel_must_divide_gpu_count(tmp_data_dir, client, monkeypatch):
    auth = _setup_seven_gpus(tmp_data_dir, client, monkeypatch)
    r = _post_fit(client, auth, gpu_indices=list(range(7)), data_parallel_size=3)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error_code"] == "data_parallel_invalid"
    assert detail["gpu_count"] == 7
    assert detail["data_parallel_size"] == 3


def test_fit_preview_omitted_data_parallel_is_unchanged(tmp_data_dir, client, monkeypatch):
    auth = _setup_seven_gpus(tmp_data_dir, client, monkeypatch)
    body = _post_fit(client, auth, gpu_indices=list(range(7))).json()
    assert body["breakdown"]["total_vram"] == 7 * 32 * GIB
    assert body["breakdown"]["replicas"] == 1
    assert body["breakdown"]["gpus_per_replica"] == 7
    assert not any(w.startswith("data_parallel:") for w in body["warnings"])


def test_fit_preview_data_parallel_sizes_the_smallest_replica(tmp_data_dir, client, monkeypatch):
    """#286 review #6: with mixed GPU sizes the verdict is for the SMALLEST replica,
    not the selection's average share. Replicas are consecutive groups of the
    sorted gpu_indices (the order the row stores and vLLM assigns ranks in)."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    _install_fake_discover(monkeypatch, info=_gpt_oss_info(), config=_GPT_OSS_CONFIG)
    _install_fake_gpus(
        client,
        [
            _FakeGpuLive(index=0, name="A", memory_total_mib=16384, compute_cap=7.5),
            _FakeGpuLive(index=1, name="A", memory_total_mib=16384, compute_cap=7.5),
            _FakeGpuLive(index=2, name="B", memory_total_mib=32768, compute_cap=7.5),
            _FakeGpuLive(index=3, name="B", memory_total_mib=32768, compute_cap=7.5),
        ],
    )
    # unsorted on purpose: replicas are {0,1} (32 GiB) and {2,3} (64 GiB)
    r = _post_fit(client, auth, gpu_indices=[3, 2, 1, 0], data_parallel_size=2)
    assert r.status_code == 200, r.text
    assert r.json()["breakdown"]["total_vram"] == 32 * GIB
