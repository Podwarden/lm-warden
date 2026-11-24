"""Supervisor.load refuses a bundled pin that names no install in this image.

A bare ``engine_vllm_version`` is a pin on a bundled install. When this
image does not carry that install, load() must refuse BEFORE any GPU is
claimed, with a message naming the missing id and the present ones.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.runtime.supervisor import EnginePinUnsupported, Supervisor

PRESENT_ID = "0.34.0rc1.dev1202-pr69165"


class _Settings:
    data_dir = "/data"
    hf_token_path = "/data/hf-token"
    hf_cache_dir = "/root/.cache/huggingface"
    load_max_s = 7200.0


class _M:
    id = "qwen"
    hf_repo = "Qwen/Qwen3.5-9B"
    hf_revision = "main"
    served_model_name = "qwen3.5-9b"
    gpu_indices = [1, 2]
    tensor_parallel_size = 2
    max_model_len = 8192
    dtype = "auto"
    gpu_memory_utilization = 0.90
    extra_env = {}
    engine_channel = None
    engine_vllm_version = None
    engine_image = None


@pytest.mark.asyncio
async def test_unknown_bundled_pin_refuses_before_gpu_claim(tmp_path, monkeypatch):
    root = tmp_path / "engines"
    eng = root / PRESENT_ID
    (eng / "bin").mkdir(parents=True)
    vllm = eng / "bin" / "vllm"
    vllm.write_text("#!/bin/sh\n")
    vllm.chmod(0o755)
    monkeypatch.setenv("VW_ENGINES_ROOT", str(root))

    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    sup = Supervisor(settings)
    claim = MagicMock()
    sup.gpus.claim = claim

    m = _M()
    m.engine_vllm_version = "nope-1"

    with pytest.raises(EnginePinUnsupported) as exc:
        await sup.load(m, port=10001)

    assert "nope-1" in str(exc.value)
    assert PRESENT_ID in str(exc.value)
    assert claim.not_called()
