import pytest
from pydantic import ValidationError

from app.models.schemas import ModelCreate


def test_model_create_minimal():
    m = ModelCreate(
        served_model_name="qwen3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
        gpu_indices=[1, 2],
    )
    assert m.tensor_parallel_size == 2
    assert m.hf_revision == "main"


def test_model_create_tp_must_match_gpu_count_when_unset_default_to_len():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0, 1, 2],
    )
    assert m.tensor_parallel_size == 3


def test_model_create_explicit_tp_must_match():
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="x",
            hf_repo="o/r",
            gpu_indices=[0, 1, 2],
            tensor_parallel_size=2,
        )


def test_model_create_rejects_empty_gpus():
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="x", hf_repo="o/r", gpu_indices=[])


def test_model_create_served_name_slug():
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="bad name!", hf_repo="o/r", gpu_indices=[0])


# ---------------------------------------------------------------------------
# extra_env allowlist enforcement at the API boundary (defense-in-depth)
# ---------------------------------------------------------------------------


def test_extra_env_rejects_hard_locked_cuda_visible_devices():
    with pytest.raises(ValidationError, match="hard-locked"):
        ModelCreate(
            served_model_name="x",
            hf_repo="o/r",
            gpu_indices=[0],
            extra_env={"CUDA_VISIBLE_DEVICES": "9"},
        )


def test_extra_env_rejects_hard_locked_path():
    with pytest.raises(ValidationError, match="hard-locked"):
        ModelCreate(
            served_model_name="x",
            hf_repo="o/r",
            gpu_indices=[0],
            extra_env={"PATH": "/evil"},
        )


def test_extra_env_rejects_unknown_key():
    with pytest.raises(ValidationError, match="allowlist"):
        ModelCreate(
            served_model_name="x",
            hf_repo="o/r",
            gpu_indices=[0],
            extra_env={"FOO_BAR": "1"},
        )


def test_extra_env_accepts_allowed_prefix():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0],
        extra_env={"VLLM_USE_V1": "1"},
    )
    assert m.extra_env == {"VLLM_USE_V1": "1"}


def test_extra_env_accepts_exact_match_cuda_module_loading():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0],
        extra_env={"CUDA_MODULE_LOADING": "LAZY"},
    )
    assert m.extra_env == {"CUDA_MODULE_LOADING": "LAZY"}


def test_extra_env_empty_dict_accepted():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0],
    )
    assert m.extra_env == {}


# ---------------------------------------------------------------------------
# #106: hf_config_repo / tokenizer_repo plumbing for GGUF repos that omit
# config.json. Both share the hf_repo owner/name slug regex; empty strings
# normalise to None so a cleared FE field round-trips correctly.
# ---------------------------------------------------------------------------


def test_hf_config_and_tokenizer_repo_accepted():
    """Happy path — both fields accept owner/name slugs and round-trip."""
    m = ModelCreate(
        served_model_name="x",
        hf_repo="unsloth/Qwen3-GGUF",
        gpu_indices=[0],
        hf_config_repo="Qwen/Qwen3-30B-A3B",
        tokenizer_repo="Qwen/Qwen3-30B-A3B",
    )
    assert m.hf_config_repo == "Qwen/Qwen3-30B-A3B"
    assert m.tokenizer_repo == "Qwen/Qwen3-30B-A3B"


def test_hf_config_repo_empty_string_normalises_to_none():
    """The FE Input ships an empty string when the operator clears the field.
    The before-validator must coerce both empty and whitespace-only to None
    so the pattern regex doesn't reject a cleared field."""
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0],
        hf_config_repo="",
        tokenizer_repo="   ",
    )
    assert m.hf_config_repo is None
    assert m.tokenizer_repo is None


def test_hf_config_repo_rejects_bad_slug():
    """Non-owner/name strings must still be rejected — slug validation is the
    whole point of the field over a free-text path."""
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="x",
            hf_repo="o/r",
            gpu_indices=[0],
            hf_config_repo="not-a-slug",
        )


# ---------------------------------------------------------------------------
# Sub-project B: ModelCreate.backend (decision D6)
# ---------------------------------------------------------------------------


def test_backend_defaults_to_vllm():
    m = ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0])
    assert m.backend == "vllm"


def test_a_client_that_never_sends_backend_behaves_exactly_as_today():
    """D6: 'A client that never sends backend behaves exactly as today.'"""
    body = {"served_model_name": "x", "hf_repo": "a/b", "gpu_indices": [0]}
    assert ModelCreate(**body).backend == "vllm"


def test_unknown_backend_is_rejected_at_the_api_boundary():
    # "llamacpp" was the unknown name in sub-project B; C registers it, so the
    # test needs a name this build genuinely lacks.
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0], backend="sglang")


def test_the_literal_matches_the_registry():
    """The Literal and the registry must never drift: a backend the registry
    can serve but the API rejects is invisible; the reverse creates rows that
    cannot load."""
    from typing import get_args

    from app.runtime.backends import registry

    field = ModelCreate.model_fields["backend"]
    assert set(get_args(field.annotation)) == set(registry.available())


def test_extra_env_is_validated_against_the_selected_backends_allowlist():
    ModelCreate(
        served_model_name="x",
        hf_repo="a/b",
        gpu_indices=[0],
        extra_env={"VLLM_LOGGING_LEVEL": "DEBUG"},
    )
    with pytest.raises(ValidationError, match="not in the allowlist"):
        ModelCreate(
            served_model_name="x",
            hf_repo="a/b",
            gpu_indices=[0],
            extra_env={"LLAMA_ARG_N_GPU_LAYERS": "99"},
        )


def test_hard_locked_keys_are_still_rejected_at_write_time():
    with pytest.raises(ValidationError, match="hard-locked"):
        ModelCreate(
            served_model_name="x",
            hf_repo="a/b",
            gpu_indices=[0],
            extra_env={"CUDA_VISIBLE_DEVICES": "0"},
        )


def test_tp_consistency_is_unchanged():
    """D4: tensor_parallel_size stays a pure GPU-count invariant. This test
    exists to fail if a future hand tries to make it backend-dependent."""
    m = ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0, 1])
    assert m.tensor_parallel_size == 2
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="x", hf_repo="a/b", gpu_indices=[0, 1], tensor_parallel_size=1
        )


# ---------------------------------------------------------------------------
# Sub-project C: the second backend at the API boundary
# ---------------------------------------------------------------------------


def test_model_create_accepts_llamacpp():
    m = ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0], backend="llamacpp")
    assert m.backend == "llamacpp"


def test_extra_env_allowlist_is_per_backend():
    """WRITE-time semantics: an off-allowlist key RAISES here.

    Deliberately the opposite of test_ggml_env_passes_through in
    tests/unit/runtime/backends/test_llamacpp_plan.py, which asserts the same
    key is DROPPED. The two pin different layers and both already ship for
    vLLM: ModelCreate raises so an operator typing a key gets an immediate,
    descriptive 422, while filter_extra_env drops at spawn time because by then
    the row may predate the check (written while it was still a vLLM row, or
    PATCHed straight into the DB) and refusing to launch over an inert env key
    would be the worse failure.
    """
    ModelCreate(
        served_model_name="x",
        hf_repo="a/b",
        gpu_indices=[0],
        backend="llamacpp",
        extra_env={"GGML_CUDA_P2P": "1"},
    )
    with pytest.raises(ValidationError, match="allowlist"):
        ModelCreate(
            served_model_name="x",
            hf_repo="a/b",
            gpu_indices=[0],
            backend="llamacpp",
            extra_env={"VLLM_LOGGING_LEVEL": "DEBUG"},
        )


def test_the_same_key_is_accepted_on_a_vllm_row():
    """Which is what makes the previous test a NARROWING rather than a new
    global rule."""
    ModelCreate(
        served_model_name="x",
        hf_repo="a/b",
        gpu_indices=[0],
        backend="vllm",
        extra_env={"VLLM_LOGGING_LEVEL": "DEBUG"},
    )


def test_tp_consistent_is_unchanged_for_llamacpp():
    """D4. The invariant is about GPU COUNT and is backend-independent."""
    with pytest.raises(ValidationError, match="tensor_parallel_size"):
        ModelCreate(
            served_model_name="x",
            hf_repo="a/b",
            gpu_indices=[0, 1],
            backend="llamacpp",
            tensor_parallel_size=1,
        )


def test_llamacpp_columns_are_accepted_on_create():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="a/b",
        gpu_indices=[0],
        backend="llamacpp",
        filename="w-IQ3_XXS.gguf",
        mmproj_filename="mmproj-BF16.gguf",
        n_gpu_layers=40,
    )
    assert m.mmproj_filename == "mmproj-BF16.gguf"
    assert m.n_gpu_layers == 40


def test_llamacpp_columns_default_to_none():
    m = ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0])
    assert m.mmproj_filename is None
    assert m.n_gpu_layers is None


def test_n_gpu_layers_is_bounded():
    """0 is legal and MEANS something (everything on CPU); a negative is not,
    and neither is a number no model has."""
    ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0], n_gpu_layers=0)
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0], n_gpu_layers=-1)
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="x", hf_repo="a/b", gpu_indices=[0], n_gpu_layers=100000)
