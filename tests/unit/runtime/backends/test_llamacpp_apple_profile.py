"""The apple profile's llama.cpp seam: no CUDA env, and an OOM message that
does not tell the operator to lower n_gpu_layers.

Metal design s6: on Apple silicon the GPU and the CPU share the Mac's memory,
so n_gpu_layers is not a memory dial and the CUDA device-selection env is
meaningless. The profile reaches this backend through
``registry.backends_for`` -- the same seam the rest of the suite uses -- and
never through a module-level PROFILE read inside env.py/diagnostics.py.

Every NEW name in this file is imported INSIDE the test functions, so the
file still collects on a tree that does not have them yet (a module-level
import of a missing name makes the whole file an ImportError, which is red
for the wrong reason).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.runtime.backends import registry
from app.runtime.backends.paths import ResolvedModelPaths

FIX = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "llamacpp"
SNAP = "/cache/models--org--repo/snapshots/abc123"


@dataclass
class StubModel:
    id: str = "m1"
    served_model_name: str = "qwen"
    hf_repo: str = "org/repo"
    hf_revision: str = "main"
    filename: str | None = "model-IQ3_XXS.gguf"
    mmproj_filename: str | None = None
    gpu_indices: list[int] = field(default_factory=lambda: [0])
    tensor_parallel_size: int = 1
    parallelism_strategy: str = "auto"
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    n_gpu_layers: int | None = None
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


def _resolved() -> ResolvedModelPaths:
    return ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP)


def _plan(backend, model):
    """The plan call shaped like test_llamacpp_plan.py's _plan helper: the
    env is the only member under test here, but plan() must be the caller
    that produces it, not a direct build_llamacpp_env call."""
    return backend.plan(
        model,
        port=10001,
        bind_host=backend.bind_host("local"),
        resolved=_resolved(),
        engine_driver="local",
        hf_cache_dir="/cache",
    )


# ---------------------------------------------------------------------------
# env.py: build_llamacpp_env(profile=...)
# ---------------------------------------------------------------------------


def test_apple_env_has_no_cuda_keys_single_gpu():
    """The red-first assertion: on apple, gpu_indices=[0] carries no CUDA_*
    key at all -- no CUDA_DEVICE_ORDER, no CUDA_VISIBLE_DEVICES -- because
    there is no CUDA device to select."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    env = build_llamacpp_env(StubModel(gpu_indices=[0]), profile="apple")
    cuda = [k for k in env if k.startswith("CUDA_")]
    assert cuda == []


def test_apple_env_has_no_cuda_keys_two_gpus():
    """One per field the rule covers: the drop is per-key, not an artifact of
    the single-gpu join. Two indices would have produced '0,1' on linux; on
    apple there is still no CUDA_VISIBLE_DEVICES and no order pin."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    env = build_llamacpp_env(StubModel(gpu_indices=[0, 1]), profile="apple")
    cuda = [k for k in env if k.startswith("CUDA_")]
    assert cuda == []


def test_apple_env_keeps_ggml_extra_env_and_drops_cuda_module_loading():
    """The allowlist is profile-independent for the operator's knobs: GGML_*
    (which on Mac is GGML_METAL_*) still passes, and a CUDA_MODULE_LOADING
    carried in extra_env -- legal on linux via the exact-allow -- is dropped
    on apple."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    model = StubModel(
        extra_env={"GGML_METAL_X": "1", "CUDA_MODULE_LOADING": "LAZY"},
    )
    env = build_llamacpp_env(model, profile="apple")
    assert env["GGML_METAL_X"] == "1"
    assert "CUDA_MODULE_LOADING" not in env


def test_apple_path_is_the_bundle_engine_path():
    """P5 llama.cpp-Metal placement: the bundle ships llama-server in the
    embedded interpreter's own bin dir (scripts/macos/bundle-llamacpp.sh lays
    it at Contents/Resources/python/bin/llama-server, next to mlx_lm.server),
    so on apple the bare argv[0] resolves through the SAME hard-set PATH the
    MLX backend emits: dirname(sys.executable), then /usr/bin, /bin. No
    /usr/local/bin and no /opt/homebrew/bin -- a user-installed llama-server
    there would be an unsigned, user-supplied engine (Metal design s3)."""
    import os
    import sys

    from app.runtime.backends.llamacpp.env import build_llamacpp_env
    from app.runtime.backends.mlx import ENGINE_PATH

    env = build_llamacpp_env(StubModel(), profile="apple")
    assert env["PATH"] == ENGINE_PATH
    parts = env["PATH"].split(":")
    assert parts[0] == os.path.dirname(sys.executable)
    assert parts[-2:] == ["/usr/bin", "/bin"]
    assert "/usr/local/bin" not in parts[1:]
    assert "/opt/homebrew/bin" not in parts[1:]
    assert "" not in parts


def test_apple_plan_argv0_is_the_bare_binary_on_the_engine_path():
    """plan() stays pure: argv[0] is the bare name on both profiles, and the
    apple plan's PATH is the one the bundled binary sits on."""
    from app.runtime.backends.mlx import ENGINE_PATH

    apple = _plan(registry.backends_for("apple")["llamacpp"], StubModel())
    linux = _plan(registry.backends_for("linux")["llamacpp"], StubModel())
    assert apple.argv[0] == linux.argv[0] == "llama-server"
    assert apple.env["PATH"] == ENGINE_PATH
    assert linux.env["PATH"] == "/usr/local/bin:/usr/bin:/bin"


def test_linux_env_is_byte_identical_to_today():
    """The golden: on linux the profile changes nothing. CUDA_DEVICE_ORDER is
    still pinned PCI_BUS_ID (the 2026-05-08 incident) and CUDA_VISIBLE_DEVICES
    still joins the indices."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    env = build_llamacpp_env(StubModel(gpu_indices=[0, 1]), profile="linux")
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"


def test_linux_default_profile_is_the_linux_env():
    """No keyword at all is linux: a caller that predates the profile axis
    gets today's env, byte-for-byte."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    env = build_llamacpp_env(StubModel(gpu_indices=[0]))
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_linux_env_keeps_cuda_module_loading_from_extra_env():
    """The other arm of the exact-allow: on linux (explicit and by default) a
    CUDA_MODULE_LOADING in extra_env still passes through. Only apple drops
    it; a linux env that lost it would be a silent regression the golden
    plan tests do not pin."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    model = StubModel(extra_env={"GGML_CUDA_P2P": "1", "CUDA_MODULE_LOADING": "LAZY"})
    for env in (build_llamacpp_env(model, profile="linux"), build_llamacpp_env(model)):
        assert env["CUDA_MODULE_LOADING"] == "LAZY"
        assert env["GGML_CUDA_P2P"] == "1"


def test_apple_env_still_refuses_hard_locked_llama_keys():
    """The security argument is profile-independent: a hard-locked LLAMA_*
    key is refused loudly at build time on apple exactly as on linux -- the
    apple profile narrows the CUDA env, it does not widen the allowlist."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    model = StubModel(extra_env={"LLAMA_ARG_HOST": "0.0.0.0"})
    with pytest.raises(ValueError, match="hard-locked"):
        build_llamacpp_env(model, profile="apple")


def test_apple_empty_gpu_indices_still_raises():
    """The existing early return stays above the profile branch: empty
    gpu_indices is a row error, not a platform question."""
    from app.runtime.backends.llamacpp.env import build_llamacpp_env

    with pytest.raises(ValueError, match="gpu_indices"):
        build_llamacpp_env(StubModel(gpu_indices=[]), profile="apple")


# ---------------------------------------------------------------------------
# diagnostics.py: diagnose_llamacpp_log(profile=...)
# ---------------------------------------------------------------------------


def test_apple_oom_does_not_tell_the_operator_to_lower_n_gpu_layers():
    """The red-first assertion for the diagnoser: on apple the OOM message is
    the unified-memory answer -- it names Apple silicon, and it neither
    suggests reducing n_gpu_layers (which saves nothing there) nor mentions
    the context (the OOM rule's load-bearing discipline, still)."""
    from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

    d = diagnose_llamacpp_log((FIX / "log_oom.txt").read_text(), profile="apple")
    assert d is not None
    assert "Apple silicon" in d.message
    assert "reduce n_gpu_layers" not in d.message
    assert "context" not in d.message
    assert d.recommended_max_model_len is None


def test_apple_oom_message_is_the_unified_memory_answer():
    """The exact message, pinned: the operator must be told that CPU layers
    run in the same memory, and pointed at a smaller quant, unloading other
    models, or a bigger Mac instead."""
    from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

    d = diagnose_llamacpp_log((FIX / "log_oom.txt").read_text(), profile="apple")
    assert d is not None
    assert d.message == (
        "llama.cpp ran out of memory placing the model. On Apple silicon the "
        "GPU and the CPU share the Mac's memory, so lowering n_gpu_layers "
        "saves nothing: it only runs layers on the CPU, slower, in the same "
        "memory. Pick a smaller quantisation, unload other models, or use a "
        "Mac with more memory."
    )


def test_linux_oom_still_says_reduce_n_gpu_layers():
    """The linux OOM message is byte-for-byte today's: the n_gpu_layers lever
    is real on CUDA and must not be silenced for the wrong audience."""
    from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

    d = diagnose_llamacpp_log((FIX / "log_oom.txt").read_text(), profile="linux")
    assert d is not None
    assert "reduce n_gpu_layers" in d.message
    assert d.recommended_max_model_len is None


def test_apple_wrong_shard_equals_linux_result():
    """The profile only changes the OOM branch. Every other rule -- here the
    wrong-shard one, which interpolates a shard number -- answers the same
    diagnosis on both profiles."""
    from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

    text = (FIX / "log_wrong_shard.txt").read_text()
    apple = diagnose_llamacpp_log(text, profile="apple")
    linux = diagnose_llamacpp_log(text, profile="linux")
    assert apple is not None
    assert apple == linux


def test_apple_non_oom_branches_are_profile_independent():
    """One more non-OOM arm: the arch rule also interpolates, and the
    context rule's recommended_max_model_len must survive unchanged on
    apple -- the profile is the OOM message, not a fork of the whole
    grammar."""
    from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

    for name in ("log_unknown_arch.txt", "log_ctx_capped.txt"):
        text = (FIX / name).read_text()
        apple = diagnose_llamacpp_log(text, profile="apple")
        linux = diagnose_llamacpp_log(text)
        assert apple == linux


def test_apple_diagnose_default_profile_is_the_linux_diagnosis():
    """No keyword at all is linux: existing callers keep today's OOM message."""
    from app.runtime.backends.llamacpp.diagnostics import diagnose_llamacpp_log

    d = diagnose_llamacpp_log((FIX / "log_oom.txt").read_text())
    assert d is not None
    assert "reduce n_gpu_layers" in d.message


# ---------------------------------------------------------------------------
# LlamaCppBackend + registry: the profile is set at construction
# ---------------------------------------------------------------------------


def test_registry_apple_llamacpp_plan_env_has_no_cuda_keys():
    """The end-to-end red: registry.backends_for('apple') builds the backend
    WITH the profile, so a full plan() -- argv, env, the way the supervisor
    calls it -- comes back with no CUDA_* key in the env."""
    plan = _plan(registry.backends_for("apple")["llamacpp"], StubModel(gpu_indices=[0, 1]))
    cuda = [k for k in plan.env if k.startswith("CUDA_")]
    assert cuda == []


def test_registry_linux_llamacpp_plan_env_has_cuda_visible_devices():
    """The linux branch keeps LlamaCppBackend() unprofiled: the plan's env
    still carries CUDA_VISIBLE_DEVICES joined from the indices."""
    plan = _plan(registry.backends_for("linux")["llamacpp"], StubModel(gpu_indices=[0, 1]))
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_backend_stores_profile_and_passes_it_to_both_members():
    """plan() hands the profile to build_llamacpp_env and diagnose() hands it
    to diagnose_llamacpp_log; the default constructor is linux, and the
    capabilities stay the shared class attribute (one CAPABILITIES object,
    not a per-instance copy)."""
    from app.runtime.backends.llamacpp import LlamaCppBackend

    apple = LlamaCppBackend(profile="apple")
    linux = LlamaCppBackend()
    assert apple.capabilities is linux.capabilities

    plan = _plan(apple, StubModel())
    assert not [k for k in plan.env if k.startswith("CUDA_")]

    d = apple.diagnose((FIX / "log_oom.txt").read_text())
    assert "Apple silicon" in d.message
    d2 = linux.diagnose((FIX / "log_oom.txt").read_text())
    assert "reduce n_gpu_layers" in d2.message
