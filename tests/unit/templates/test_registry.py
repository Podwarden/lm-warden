"""Tests for the built-in model template registry."""

import dataclasses

import pytest

from app.templates.registry import get_template, list_templates


def test_list_templates_returns_at_least_one():
    templates = list_templates()
    assert len(templates) >= 1


def test_gpt_oss_20b_present():
    templates = list_templates()
    ids = [t.id for t in templates]
    assert "gpt-oss-20b" in ids


def test_gpt_oss_20b_has_expected_values():
    t = get_template("gpt-oss-20b")
    assert t is not None
    assert t.hf_repo == "openai/gpt-oss-20b"
    assert t.dtype == "bfloat16"
    assert t.max_model_len == 32000
    assert t.tensor_parallel_size == 2
    assert t.gpu_memory_utilization == 0.7
    assert t.trust_remote_code is True
    assert t.hf_revision == "main"


def test_gpt_oss_20b_extra_env_sample_keys():
    t = get_template("gpt-oss-20b")
    assert t is not None
    assert t.extra_env.get("VLLM_USE_V1") == "1"
    assert t.extra_env.get("NCCL_IB_DISABLE") == "1"
    assert t.extra_env.get("CUDA_MODULE_LOADING") == "LAZY"
    assert t.extra_env.get("OMP_NUM_THREADS") == "1"
    assert t.extra_env.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"


def test_gpt_oss_20b_async_output_proc_disabled_for_91():
    """#91 workaround lock-in. Do NOT flip this back to "1" without first
    verifying upstream vLLM has landed per-request isolation in
    `vllm/entrypoints/openai/harmony_utils.py` — the singleton
    `_harmony_encoding` still exists in v0.21.0 and re-enabling the async
    output processor reintroduces the `RuntimeError: Already borrowed`
    panic on concurrent /v1/chat/completions requests.
    """
    t = get_template("gpt-oss-20b")
    assert t is not None
    assert t.extra_env.get("VLLM_USE_ASYNC_OUTPUT_PROC") == "0"


def test_get_template_returns_none_for_unknown():
    assert get_template("does-not-exist") is None


def test_template_is_frozen():
    """ModelTemplate must be frozen so callers can't mutate shared state."""
    t = get_template("gpt-oss-20b")
    assert t is not None

    assert dataclasses.fields(t)  # it's a dataclass
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.dtype = "float16"  # type: ignore[misc]
