"""Tests for VLLM_ENGINE_READY_TIMEOUT_S in the vLLM subprocess env.

Task 4: ``build_subprocess_env`` sets ``VLLM_ENGINE_READY_TIMEOUT_S`` to
``str(int(load_max_s))`` unless the model's filtered ``extra_env`` already sets
it (the operator wins).
"""

from __future__ import annotations

from app.runtime.backends.vllm.env import build_subprocess_env


def _model(extra_env: dict | None = None) -> object:
    class M:
        id = "test"
        gpu_indices = [0, 1]
        tensor_parallel_size = 2

    M.extra_env = extra_env or {}
    return M()


def test_engine_ready_timeout_env_defaults_to_cap():
    """With no VLLM_ENGINE_READY_TIMEOUT_S in extra_env, the builder sets it to
    int(load_max_s)."""
    env = build_subprocess_env(
        _model(),
        hf_token="tok",
        hf_cache_dir="/d",
        load_max_s=7200.0,
    )
    assert env["VLLM_ENGINE_READY_TIMEOUT_S"] == "7200"


def test_engine_ready_timeout_env_truncates_to_int():
    """A fractional load_max_s is truncated to int."""
    env = build_subprocess_env(
        _model(),
        hf_token="tok",
        hf_cache_dir="/d",
        load_max_s=7200.9,
    )
    assert env["VLLM_ENGINE_READY_TIMEOUT_S"] == "7200"


def test_engine_ready_timeout_respects_extra_env():
    """The operator's VLLM_ENGINE_READY_TIMEOUT_S in extra_env wins."""
    env = build_subprocess_env(
        _model({"VLLM_ENGINE_READY_TIMEOUT_S": "999"}),
        hf_token="tok",
        hf_cache_dir="/d",
        load_max_s=7200.0,
    )
    assert env["VLLM_ENGINE_READY_TIMEOUT_S"] == "999"


def test_engine_ready_timeout_absent_when_load_max_s_not_passed():
    """Backwards-compatible: if load_max_s is not passed (old callers), the key
    is simply absent — no crash, no spurious default."""
    env = build_subprocess_env(
        _model(),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert "VLLM_ENGINE_READY_TIMEOUT_S" not in env
