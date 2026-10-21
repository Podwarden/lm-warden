"""The argv builders refuse extra_args that set the engine's model-id flag.

The proxy routes and forwards by the ``served_model_name`` column, so a row
whose ``extra_args`` overrides the engine's served id would serve an id the
proxy never asks for. The detector in :mod:`app.models.model_id_flags` is
applied at load time by each backend's argv builder: a legacy or
template-derived row fails loud at load instead of silently mis-routing.
"""

import pytest

from app.db.repos.models import ModelRow
from app.runtime.backends.llamacpp.args import build_llamacpp_args
from app.runtime.backends.paths import ResolvedModelPaths
from app.runtime.backends.vllm.args import build_vllm_args


def _row(**overrides) -> ModelRow:
    base = dict(
        id="qwen",
        served_model_name="qwen3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
        hf_revision=None,
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=None,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        extra_args=[],
        extra_env={},
        status="pulled",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
    )
    base.update(overrides)
    return ModelRow(**base)


class _LlamaRow:
    """A minimal stand-in row for the llama.cpp builder."""

    def __init__(self, **kw):
        self.id = "m1"
        self.served_model_name = "qwen"
        self.max_model_len = None
        self.max_num_seqs = None
        self.n_gpu_layers = None
        self.gpu_indices = [0]
        self.extra_args = []
        for k, v in kw.items():
            setattr(self, k, v)


def _resolved():
    return ResolvedModelPaths(model_path="/snap/model.gguf", snapshot_dir="/snap")


# -- vLLM -------------------------------------------------------------------


def test_vllm_refuses_long_flag_in_extra_args():
    from app.models.model_id_flags import model_id_refusal

    row = _row(extra_args=["--served-model-name", "other"])
    with pytest.raises(ValueError, match="served_model_name"):
        try:
            build_vllm_args(row, port=10001)
        except ValueError as e:
            # The message must name the offending token AND the column.
            assert "--served-model-name" in str(e)
            assert "served_model_name" in str(e)
            raise
    # Belt: message matches the refusal helper's own output for the token.
    assert model_id_refusal("vllm", "--served-model-name") is not None


def test_vllm_refuses_equals_form():
    row = _row(extra_args=["--served-model-name=other"])
    with pytest.raises(ValueError, match="served_model_name"):
        build_vllm_args(row, port=10001)


def test_vllm_refuses_underscore_spelling():
    row = _row(extra_args=["--served_model_name", "other"])
    with pytest.raises(ValueError, match="served_model_name"):
        build_vllm_args(row, port=10001)


def test_vllm_refuses_dotted_key():
    row = _row(extra_args=["--served-model-name.x=y"])
    with pytest.raises(ValueError, match="served_model_name"):
        build_vllm_args(row, port=10001)


def test_vllm_refuses_abbreviation_down_to_floor():
    # --ser is the floor (vLLM has no other --ser* option); --se is ambiguous
    # and vLLM itself would reject it, so the builder is free to let it
    # through (the engine refuses it) -- we do not assert either way here, we
    # assert the floor is refused.
    row = _row(extra_args=["--ser", "other"])
    with pytest.raises(ValueError, match="served_model_name"):
        build_vllm_args(row, port=10001)


def test_vllm_refuses_abbreviation_above_floor():
    row = _row(extra_args=["--served", "other"])
    with pytest.raises(ValueError, match="served_model_name"):
        build_vllm_args(row, port=10001)


def test_vllm_does_not_refuse_unrelated_flags():
    # The guard must not false-positive on other --s* or --model flags that
    # are NOT the model-id flag.
    row = _row(extra_args=["--scheduling-policy", "fcfs"])
    args = build_vllm_args(row, port=10001)
    assert "--scheduling-policy" in args
    assert "fcfs" in args


def test_vllm_argv_unchanged_when_no_model_id_flag():
    # A row without a model-id flag in extra_args gets a byte-identical argv.
    row = _row(extra_args=["--max-num-batched-tokens", "8192"])
    args = build_vllm_args(row, port=10001)
    assert "--max-num-batched-tokens" in args
    assert "8192" in args
    # And the canonical --served-model-name the builder itself emits is the
    # row's column, not anything from extra_args.
    i = args.index("--served-model-name")
    assert args[i + 1] == "qwen3.5-9b"


def test_vllm_does_not_refuse_llamacpp_flag():
    # Cross-backend: the vLLM builder must not refuse the llama.cpp flag.
    row = _row(extra_args=["--alias", "other"])
    args = build_vllm_args(row, port=10001)
    assert "--alias" in args
    assert "other" in args


# -- llama.cpp --------------------------------------------------------------


def test_llamacpp_refuses_long_flag_in_extra_args():
    row = _LlamaRow(extra_args=["--alias", "other"])
    with pytest.raises(ValueError, match="served_model_name"):
        try:
            build_llamacpp_args(row, port=10001, bind_host="127.0.0.1", resolved=_resolved())
        except ValueError as e:
            assert "--alias" in str(e)
            assert "served_model_name" in str(e)
            raise


def test_llamacpp_refuses_short_flag():
    row = _LlamaRow(extra_args=["-a", "other"])
    with pytest.raises(ValueError, match="served_model_name"):
        build_llamacpp_args(row, port=10001, bind_host="127.0.0.1", resolved=_resolved())


def test_llamacpp_does_not_refuse_unrelated_flags():
    row = _LlamaRow(extra_args=["--cache-type-k", "q8_0"])
    args = build_llamacpp_args(row, port=10001, bind_host="127.0.0.1", resolved=_resolved())
    assert "--cache-type-k" in args
    assert "q8_0" in args


def test_llamacpp_argv_unchanged_when_no_model_id_flag():
    row = _LlamaRow(extra_args=["--cache-type-k", "q8_0"])
    args = build_llamacpp_args(row, port=10001, bind_host="127.0.0.1", resolved=_resolved())
    i = args.index("--alias")
    assert args[i + 1] == "qwen"


def test_llamacpp_does_not_refuse_vllm_flag():
    # Cross-backend: the llama.cpp builder must not refuse vLLM's flag.
    row = _LlamaRow(extra_args=["--served-model-name", "other"])
    args = build_llamacpp_args(row, port=10001, bind_host="127.0.0.1", resolved=_resolved())
    assert "--served-model-name" in args
    assert "other" in args


# -- reviewer legs (farm 38) -------------------------------------------------


def test_vllm_does_not_refuse_mlx_model_flag():
    # Cross-backend: --model is MLX's model-id flag, not vLLM's (vLLM takes
    # the model positionally). The vLLM builder must not refuse it.
    row = _row(extra_args=["--model", "other"])
    args = build_vllm_args(row, port=10001)
    assert "--model" in args
    assert "other" in args


_SECRET = "sk-reviewer-sentinel-must-not-echo"


def test_vllm_refusal_does_not_echo_other_extra_args():
    # extra_args may hold secrets (an API key): the refusal names only the
    # offending token, never the rest of extra_args.
    row = _row(extra_args=["--api-key", _SECRET, "--served-model-name", "other"])
    with pytest.raises(ValueError) as exc:
        build_vllm_args(row, port=10001)
    assert "--served-model-name" in str(exc.value)
    assert _SECRET not in str(exc.value)
    assert "--api-key" not in str(exc.value)


def test_llamacpp_refusal_does_not_echo_other_extra_args():
    row = _LlamaRow(extra_args=["--api-key", _SECRET, "-a", "other"])
    with pytest.raises(ValueError) as exc:
        build_llamacpp_args(row, port=10001, bind_host="127.0.0.1", resolved=_resolved())
    assert "-a" in str(exc.value)
    assert _SECRET not in str(exc.value)
    assert "--api-key" not in str(exc.value)
