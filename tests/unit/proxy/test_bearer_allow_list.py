"""Regression tests: bearer token allowed_models enforcement on /v1/* routes."""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

from app.db.repos.tokens import hash_token  # noqa: E402 (project-local import)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(db_path, *, allowed_models=None, extra_models=None):
    """Seed DB with setup-done state, one or more models, and one token.

    Returns the plaintext token string.

    Parameters
    ----------
    allowed_models:
        CSV string or None — stored verbatim in api_tokens.allowed_models.
    extra_models:
        list of (id, served_model_name, status) tuples to add alongside the
        default "qwen3" (loaded) model.
    """
    plaintext = "vw_allowlisttest1234567890abcdef"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        # Default model: qwen3, loaded
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen3','qwen3','Qwen/Qwen3','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        if extra_models:
            for mid, served, status in extra_models:
                db.execute(
                    "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
                    "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
                    "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
                    "VALUES (?,?,'r','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)",
                    (mid, served, json.dumps([0]), status),
                )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, allowed_models) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "tok-al",
                "allowlist-test",
                plaintext[:8],
                hash_token(plaintext),
                "inference",
                allowed_models,
            ),
        )
        db.commit()
    return plaintext


def _fake_upstream_resp(
    body: bytes = b'{"id":"x","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}',
):
    """Return a mock httpx response suitable for patching AsyncClient.send."""
    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "application/json"}
    fake.aread = AsyncMock(return_value=body)
    fake.aclose = AsyncMock()
    return fake


def _stub_tokenizer(client):
    """Replace the app's tokenizer cache with a no-op stub (always returns 0)."""
    stub = MagicMock()
    stub.count = AsyncMock(return_value=0)
    client.app.state.tokenizers = stub


# ---------------------------------------------------------------------------
# Test 1: unrestricted token (allowed_models=None) allows any model
# ---------------------------------------------------------------------------


def test_unrestricted_token_allows_any_model(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db", allowed_models=None)
    # Register port so _resolve_target succeeds
    client.app.state.supervisor._ports["qwen3"] = 19200
    _stub_tokenizer(client)

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_upstream_resp())),
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test 2: restricted token allows a model that is listed
# ---------------------------------------------------------------------------


def test_restricted_token_allows_listed_model(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db", allowed_models="qwen3,llama3")
    client.app.state.supervisor._ports["qwen3"] = 19201
    _stub_tokenizer(client)

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_upstream_resp())),
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test 3: restricted token denies a model not in the list — 403 before _resolve_target
# ---------------------------------------------------------------------------


def test_restricted_token_denies_unlisted_model(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db", allowed_models="qwen3")

    # We deliberately do NOT register a port for llama3 — if 403 fires before
    # _resolve_target, the response must be 403, not 404.
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "token not allowed for model 'llama3'"


def test_restricted_token_denies_unlisted_model_completions(tmp_data_dir, client):
    """Same check on /v1/completions endpoint."""
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db", allowed_models="qwen3")

    r = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "llama3", "prompt": "hello"},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "token not allowed for model 'llama3'"


# ---------------------------------------------------------------------------
# Test 4: GET /v1/models filters listing by allow-list
# ---------------------------------------------------------------------------


def test_list_models_filters_by_allow_list(tmp_data_dir, client):
    client.get("/healthz")
    # Seed with two loaded models: qwen3 (default) + llama3
    plaintext = _seed(
        tmp_data_dir / "vllm-warden.db",
        allowed_models="qwen3",
        extra_models=[("llama3", "llama3", "loaded")],
    )
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "qwen3" in ids
    assert "llama3" not in ids


def test_list_models_unrestricted_token_shows_all_loaded(tmp_data_dir, client):
    """Sanity: unrestricted token sees all loaded models."""
    client.get("/healthz")
    plaintext = _seed(
        tmp_data_dir / "vllm-warden.db",
        allowed_models=None,
        extra_models=[("llama3", "llama3", "loaded")],
    )
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "qwen3" in ids
    assert "llama3" in ids
