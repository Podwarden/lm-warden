"""Tests for the extra_env allowlist merge in build_subprocess_env."""

import logging

import pytest

from app.runtime.backends.vllm.env import build_subprocess_env


def _model(extra_env: dict) -> object:
    class M:
        id = "test"
        gpu_indices = [0, 1]
        tensor_parallel_size = 2

    M.extra_env = extra_env
    return M()


# ---------------------------------------------------------------------------
# Allowlist acceptance
# ---------------------------------------------------------------------------


def test_vllm_prefix_accepted():
    env = build_subprocess_env(_model({"VLLM_USE_V1": "1"}), hf_token="tok", hf_cache_dir="/d")
    assert env["VLLM_USE_V1"] == "1"


def test_nccl_prefix_accepted():
    env = build_subprocess_env(_model({"NCCL_DEBUG": "WARN"}), hf_token="tok", hf_cache_dir="/d")
    assert env["NCCL_DEBUG"] == "WARN"


def test_triton_prefix_accepted():
    env = build_subprocess_env(_model({"TRITON_FOO": "bar"}), hf_token="tok", hf_cache_dir="/d")
    assert env["TRITON_FOO"] == "bar"


def test_omp_prefix_accepted():
    env = build_subprocess_env(_model({"OMP_NUM_THREADS": "1"}), hf_token="tok", hf_cache_dir="/d")
    assert env["OMP_NUM_THREADS"] == "1"


def test_pytorch_prefix_accepted():
    env = build_subprocess_env(
        _model({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_torch_prefix_accepted():
    env = build_subprocess_env(
        _model({"TORCH_CUDNN_V8_API_ENABLED": "1"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert env["TORCH_CUDNN_V8_API_ENABLED"] == "1"


def test_cuda_module_loading_exact_accepted():
    env = build_subprocess_env(
        _model({"CUDA_MODULE_LOADING": "LAZY"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert env["CUDA_MODULE_LOADING"] == "LAZY"


def test_faulthandler_on_by_default():
    """A fatal signal must never kill the engine silently.

    2026-08-18: EngineCore died 20 times in a night with no traceback, no exit
    code and no core file. faulthandler is the cheapest thing that turns that
    into a printed stack, so it is a default rather than an opt-in.
    """
    env = build_subprocess_env(_model({}), hf_token="tok", hf_cache_dir="/d")
    assert env["PYTHONFAULTHANDLER"] == "1"


def test_faulthandler_exact_accepted():
    env = build_subprocess_env(
        _model({"PYTHONFAULTHANDLER": "0"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert env["PYTHONFAULTHANDLER"] == "0"


def test_pythonunbuffered_still_unreachable_via_the_faulthandler_hatch():
    """Adding PYTHONFAULTHANDLER by NAME (not by a "PYTHON" prefix) is what
    keeps PYTHONUNBUFFERED locked — the exact failure the hard-lock comment
    warns a future hand about."""
    with pytest.raises(ValueError, match="PYTHONUNBUFFERED"):
        build_subprocess_env(
            _model({"PYTHONUNBUFFERED": "0"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )


# ---------------------------------------------------------------------------
# Allowlist rejection (silent drop)
# ---------------------------------------------------------------------------


def test_unknown_key_silently_dropped():
    env = build_subprocess_env(
        _model({"MY_CUSTOM_VAR": "foo"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert "MY_CUSTOM_VAR" not in env


def test_arbitrary_env_var_silently_dropped():
    env = build_subprocess_env(
        _model({"SECRET_TOKEN": "s3cr3t", "HOME": "/root"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert "SECRET_TOKEN" not in env
    assert "HOME" not in env


# ---------------------------------------------------------------------------
# Hard-locked keys raise ValueError
# ---------------------------------------------------------------------------


def test_cuda_visible_devices_in_extra_env_raises():
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        build_subprocess_env(
            _model({"CUDA_VISIBLE_DEVICES": "99"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )


def test_hf_home_in_extra_env_raises():
    with pytest.raises(ValueError, match="HF_HOME"):
        build_subprocess_env(
            _model({"HF_HOME": "/evil"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )


def test_hf_hub_cache_in_extra_env_raises():
    """The model-cache target is hard-locked: an operator cannot repoint the
    engine's HF cache via extra_env (which would re-open the 2026-06-15
    wrong-volume ENOSPC failure mode)."""
    with pytest.raises(ValueError, match="HF_HUB_CACHE"):
        build_subprocess_env(
            _model({"HF_HUB_CACHE": "/evil"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )


def test_hf_token_in_extra_env_raises():
    with pytest.raises(ValueError, match="HUGGING_FACE_HUB_TOKEN"):
        build_subprocess_env(
            _model({"HUGGING_FACE_HUB_TOKEN": "stolen"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )


def test_path_in_extra_env_raises():
    with pytest.raises(ValueError, match="PATH"):
        build_subprocess_env(
            _model({"PATH": "/tmp/evil/bin"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )


# ---------------------------------------------------------------------------
# Override of defaults
# ---------------------------------------------------------------------------


def test_vllm_logging_level_overrides_default():
    """VLLM_LOGGING_LEVEL=DEBUG from extra_env should override the default INFO."""
    env = build_subprocess_env(
        _model({"VLLM_LOGGING_LEVEL": "DEBUG"}),
        hf_token="tok",
        hf_cache_dir="/d",
    )
    assert env["VLLM_LOGGING_LEVEL"] == "DEBUG"


# ---------------------------------------------------------------------------
# Hard-locked keys are always correct regardless of extra_env
# ---------------------------------------------------------------------------


def test_hard_locked_keys_set_correctly_even_with_empty_extra_env():
    """Baseline: hard-locked keys come out with expected values."""
    env = build_subprocess_env(_model({}), hf_token="my-token", hf_cache_dir="/mydata")
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["HF_HUB_CACHE"] == "/mydata"
    assert "HF_HOME" not in env
    assert env["HUGGING_FACE_HUB_TOKEN"] == "my-token"
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"


# ---------------------------------------------------------------------------
# Operator visibility: dropped keys are logged at INFO
# ---------------------------------------------------------------------------


def test_dropped_keys_logged_at_info(caplog):
    """Bypass ModelCreate's validator and assert filter_extra_env logs drops.

    DB-level injection / manual SQL / migrations can still produce non-allowlist
    keys in the row even after the API-boundary validator (Fix #1) is in place.
    Operators need a signal when env they configured is silently dropped.
    """
    with caplog.at_level(logging.INFO, logger="app.runtime.backends.vllm.env"):
        env = build_subprocess_env(
            _model({"FOO_BAR": "x", "BAZ_QUX": "y", "VLLM_USE_V1": "1"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )
    assert "VLLM_USE_V1" in env
    assert "FOO_BAR" not in env
    msgs = [r.getMessage() for r in caplog.records if r.name == "app.runtime.backends.vllm.env"]
    assert any("dropped 2" in m and "FOO_BAR" in m and "BAZ_QUX" in m for m in msgs), msgs


def test_no_drop_log_when_all_keys_allowed(caplog):
    with caplog.at_level(logging.INFO, logger="app.runtime.backends.vllm.env"):
        build_subprocess_env(
            _model({"VLLM_USE_V1": "1", "NCCL_DEBUG": "WARN"}),
            hf_token="tok",
            hf_cache_dir="/d",
        )
    drop_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.name == "app.runtime.backends.vllm.env" and "dropped" in r.getMessage()
    ]
    assert drop_msgs == []


# ---------------------------------------------------------------------------
# Sub-project B: the allowlist becomes per-backend; the hard lock stays global
# ---------------------------------------------------------------------------


def test_pythonpath_is_hard_locked():
    """Spec §2 defence-in-depth. PYTHONPATH is on no allowlist today, so this
    changes nothing an operator can currently do -- it documents WHY the key
    can never be added: PYTHONPATH is precisely how extra_env would
    reintroduce runtime patching through the back door."""
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    assert "PYTHONPATH" in HARD_LOCKED_ENV_KEYS


def test_ld_preload_is_hard_locked():
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    assert "LD_PRELOAD" in HARD_LOCKED_ENV_KEYS


@pytest.mark.parametrize("key", ["PYTHONPATH", "LD_PRELOAD"])
def test_setting_a_newly_locked_key_raises(key):
    from app.runtime.backends.vllm.env import build_subprocess_env

    model = _model({key: "/tmp/evil"})
    with pytest.raises(ValueError, match="hard-locked"):
        build_subprocess_env(model, hf_token="t", hf_cache_dir="/c")


def test_filter_takes_the_allowlist_as_a_parameter():
    """Per-backend prefixes are a NARROWING. A backend that does not accept
    VLLM_* must drop them, and the drop must stay silent (never raise) --
    that half of the contract is what keeps a template portable."""
    from app.runtime.backends.vllm.env import filter_extra_env

    out = filter_extra_env(
        {"VLLM_LOGGING_LEVEL": "DEBUG", "LLAMA_ARG_N_GPU_LAYERS": "99"},
        prefixes=("LLAMA_",),
        exact=frozenset(),
    )
    assert out == {"LLAMA_ARG_N_GPU_LAYERS": "99"}


def test_hard_lock_is_enforced_regardless_of_the_backend_allowlist():
    """HARD_LOCKED_ENV_KEYS is GLOBAL: every key in it is locked for reasons
    unrelated to which server runs. A per-backend prefix must never unlock
    one."""
    from app.runtime.backends.vllm.env import filter_extra_env

    with pytest.raises(ValueError, match="hard-locked"):
        filter_extra_env(
            {"HF_HUB_CACHE": "/somewhere/else"},
            prefixes=("HF_",),
            exact=frozenset(),
        )
