"""Contract + error-mapping tests for ``GET /api/models/discover`` (#84).

We don't hit huggingface.co. Instead we monkeypatch
``app.models.routes_api.discover_repo_files`` to a thin shim that wraps
the real helper, calling it with the in-tree ``FakeHfApi`` /
``make_config_fetcher`` seams. That way the cache + envelope + JWT path
through routes_api.py is exercised end-to-end while the HF I/O stays
hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import discovery as discovery_mod
from app.models import routes_api as routes_api_mod
from tests.conftest import jwt_login, seed_admin_user
from tests.fakes.fake_hf import (
    FakeHfApi,
    FakeModelInfo,
    FakeSibling,
    make_config_fetcher,
    make_hf_api_factory,
)

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "tests/fixtures/discovery/qwen-gguf.json"


# #55 fix — route through the shared barrier+retry helpers.
def _seed_done(db_path: Path) -> None:
    """Same as test_routes_crud._seed_done — admin user + setup complete."""
    seed_admin_user(db_path)


def _jwt_login(client) -> dict[str, str]:
    return jwt_login(client)


def _install_fake_discover(monkeypatch, *, info, config):
    """Replace ``discover_repo_files`` at the route's import site.

    The shim forwards to the real implementation but plumbs through a
    fake HfApi + config fetcher. This exercises the real classification /
    envelope code while keeping the test hermetic.
    """
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


# ---- Happy path ----------------------------------------------------------


def test_discover_returns_files_and_config(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    info = FakeModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        private=False,
        gated=False,
        siblings=[
            FakeSibling("config.json", size=1024),
            FakeSibling("tokenizer.json", size=2048),
            FakeSibling("model-00001-of-00002.safetensors", size=5_000_000_000),
            FakeSibling("model-00002-of-00002.safetensors", size=2_500_000_000),
            FakeSibling("README.md", size=4096),
        ],
    )
    config = {
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
        # #176 — quantization_config now survives the reduction.
        "quantization_config": {"quant_method": "awq"},
        # Extra keys must be filtered out — only the canonical keys come through.
        "vocab_size": 152064,
        "model_type": "qwen2",
    }
    _install_fake_discover(monkeypatch, info=info, config=config)

    r = client.get(
        "/api/models/discover?repo_id=Qwen/Qwen2.5-7B-Instruct",
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["repo"] == {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "main",
        "private": False,
        "gated": False,
    }
    kinds = {f["filename"]: f["kind"] for f in body["files"]}
    assert kinds["config.json"] == "config"
    assert kinds["tokenizer.json"] == "tokenizer"
    assert kinds["model-00001-of-00002.safetensors"] == "safetensors_sharded"
    assert kinds["README.md"] == "other"
    # Config keys: the six locked by #82, quantization_config (#176), and the
    # three KV-shape keys the fit math reads. No extras. The route reads
    # discovery["config"], so a key missing here is one the fit preview cannot
    # see -- which is how head_dim and interleaved sliding-window attention
    # went uncounted.
    assert set(body["config"].keys()) == {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "torch_dtype",
        "quantization_config",
        "head_dim",
        "sliding_window",
        "layer_types",
    }
    assert body["config"]["quantization_config"] == {"quant_method": "awq"}
    assert body["errors"] == []


# ---- Wire-shape contract -------------------------------------------------


def test_discover_wire_shape_matches_fixture(tmp_data_dir, client, monkeypatch):
    """Snapshot test against ``tests/fixtures/discovery/qwen-gguf.json``.

    This is the contract dev-2 will mirror in #86 for FE mocking; if a
    backend change drifts the wire shape, this test fails loudly with a
    diff and we update the fixture intentionally.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    fixture = json.loads(FIXTURE_PATH.read_text())
    info = FakeModelInfo(
        id=fixture["repo"]["id"],
        private=fixture["repo"]["private"],
        gated=fixture["repo"]["gated"],
        siblings=[FakeSibling(f["filename"], size=f["size"]) for f in fixture["files"]],
    )
    config = dict(fixture["config"])
    _install_fake_discover(monkeypatch, info=info, config=config)

    r = client.get(
        f"/api/models/discover?repo_id={fixture['repo']['id']}",
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Strip fixture metadata before comparing.
    expected = {k: v for k, v in fixture.items() if not k.startswith("_")}
    assert body == expected


# ---- Error envelopes -----------------------------------------------------


def _make_fake_response(status_code: int):
    """Minimal stand-in for ``requests.Response`` — enough for HfHubHTTPError.

    huggingface_hub 0.30+ requires ``response=`` on its error constructors so
    the SDK can inspect ``response.status_code``. We don't need a real
    requests Response — only the attribute access pattern.
    """

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


def test_discover_401_envelope_when_hf_gated(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    from huggingface_hub.utils import GatedRepoError

    _install_fake_discover(
        monkeypatch,
        info=GatedRepoError("gated repo, supply token", response=_make_fake_response(403)),
        config={},
    )
    r = client.get(
        "/api/models/discover?repo_id=meta-llama/Llama-3-70B",
        headers=auth,
    )
    assert r.status_code == 401, r.text
    body = r.json()
    # FastAPI wraps HTTPException(detail=...) under "detail".
    assert body["detail"]["error_code"] == "auth_required"
    assert body["detail"]["repo_id"] == "meta-llama/Llama-3-70B"
    assert body["detail"]["revision"] == "main"


def test_discover_404_envelope_when_repo_missing(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    from huggingface_hub.utils import RepositoryNotFoundError

    _install_fake_discover(
        monkeypatch,
        info=RepositoryNotFoundError("not found", response=_make_fake_response(404)),
        config={},
    )
    r = client.get(
        "/api/models/discover?repo_id=this/does-not-exist&revision=main",
        headers=auth,
    )
    assert r.status_code == 404
    body = r.json()
    assert body["detail"]["error_code"] == "repo_not_found"
    assert body["detail"]["repo_id"] == "this/does-not-exist"


def test_discover_missing_config_returns_null_and_error(tmp_data_dir, client, monkeypatch):
    """Pure GGUF repo with no config.json -> ``config: null`` + ``errors`` entry."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    info = FakeModelInfo(
        id="some/gguf-only-repo",
        siblings=[FakeSibling("model.Q4_K_M.gguf", size=1234)],
    )
    # config=None mirrors what the real fetcher returns on EntryNotFoundError.
    _install_fake_discover(monkeypatch, info=info, config=None)

    r = client.get(
        "/api/models/discover?repo_id=some/gguf-only-repo",
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["config"] is None
    assert "config_not_found" in body["errors"]


# ---- Auth + caching ------------------------------------------------------


def test_discover_requires_auth(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models/discover?repo_id=some/repo")
    assert r.status_code == 401


def test_discover_cache_collapses_concurrent_calls(tmp_data_dir, client, monkeypatch):
    """A repeat call within the TTL must not re-invoke the HF fetcher."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    info = FakeModelInfo(
        id="some/repo",
        siblings=[FakeSibling("config.json", size=42)],
    )
    api = _install_fake_discover(monkeypatch, info=info, config={"hidden_size": 4096})

    r1 = client.get("/api/models/discover?repo_id=some/repo", headers=auth)
    r2 = client.get("/api/models/discover?repo_id=some/repo", headers=auth)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # First request invokes HfApi; second hits the cache.
    assert len(api.calls) == 1

    # Different revision = different cache key -> re-invokes.
    r3 = client.get(
        "/api/models/discover?repo_id=some/repo&revision=v1.0",
        headers=auth,
    )
    assert r3.status_code == 200
    assert len(api.calls) == 2


@pytest.mark.parametrize(
    "filename,size",
    [
        # Real HfApi may set size=None for files where metadata wasn't
        # populated; we coerce to 0 rather than crash.
        ("blob-without-metadata", None),
    ],
)
def test_discover_handles_missing_size_metadata(tmp_data_dir, client, monkeypatch, filename, size):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)

    info = FakeModelInfo(
        id="some/repo",
        siblings=[FakeSibling(filename, size=size)],
    )
    _install_fake_discover(monkeypatch, info=info, config=None)
    r = client.get("/api/models/discover?repo_id=some/repo", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["files"][0]["size"] == 0
