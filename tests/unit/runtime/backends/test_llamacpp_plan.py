"""LlamaCppBackend.plan() -- argv and env, pinned before they are written.

The golden file is the llama.cpp analogue of B's tests/fixtures/launch_golden.json.
It is hand-written FIRST from the flag table in the plan, then the implementation
is written until it matches -- so the argv is a decision that was reviewed, not
an artefact of whatever the code happened to emit.

Every flag spelling in the golden is checked against
tests/fixtures/llamacpp/help.txt (Task 2's real --help capture) by
test_every_flag_exists_in_the_captured_help, which is what stops this file from
freezing a flag that llama.cpp does not have.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.runtime.backends.llamacpp import LlamaCppBackend
from app.runtime.backends.paths import ResolvedModelPaths

_REPO = Path(__file__).resolve().parents[4]
GOLDEN = _REPO / "tests" / "fixtures" / "llamacpp_launch_golden.json"
HELP = _REPO / "tests" / "fixtures" / "llamacpp" / "help.txt"

BACKEND = LlamaCppBackend()
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


@dataclass
class Case:
    name: str
    model: StubModel
    resolved: ResolvedModelPaths
    port: int = 10001
    driver: str = "local"


CASES = [
    Case(
        "minimal_single_gpu",
        StubModel(),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
    ),
    Case(
        "vision_with_mmproj",
        StubModel(mmproj_filename="mmproj-BF16.gguf"),
        ResolvedModelPaths(
            model_path=f"{SNAP}/model-IQ3_XXS.gguf",
            mmproj_path=f"{SNAP}/mmproj-BF16.gguf",
            snapshot_dir=SNAP,
        ),
    ),
    Case(
        "context_and_slots",
        StubModel(max_model_len=8192, max_num_seqs=2),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
    ),
    Case(
        "partial_offload",
        StubModel(n_gpu_layers=40),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
    ),
    Case(
        "two_gpus",
        StubModel(gpu_indices=[0, 1], tensor_parallel_size=2),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
    ),
    Case(
        "tp_strategy_still_layer_split",
        StubModel(gpu_indices=[0, 1], tensor_parallel_size=2, parallelism_strategy="tp"),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
    ),
    Case(
        "extra_args_last",
        StubModel(extra_args=["--cache-type-k", "q8_0"]),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
    ),
    Case(
        "docker_driver_binds_wide",
        StubModel(),
        ResolvedModelPaths(model_path=f"{SNAP}/model-IQ3_XXS.gguf", snapshot_dir=SNAP),
        driver="docker",
    ),
    Case(
        "sharded_first_shard",
        StubModel(filename="model-IQ3_XXS-00001-of-00003.gguf"),
        ResolvedModelPaths(
            model_path=f"{SNAP}/model-IQ3_XXS-00001-of-00003.gguf",
            snapshot_dir=SNAP,
        ),
    ),
]


def _plan(case: Case):
    return BACKEND.plan(
        case.model,
        port=case.port,
        bind_host=BACKEND.bind_host(case.driver),
        resolved=case.resolved,
        engine_driver=case.driver,
        hf_cache_dir="/cache",
    )


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_argv_matches_golden(case):
    assert _plan(case).argv == json.loads(GOLDEN.read_text())[case.name]["argv"]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_env_matches_golden(case):
    assert _plan(case).env == json.loads(GOLDEN.read_text())[case.name]["env"]


def test_argv0_is_the_binary():
    """LaunchPlan.argv includes argv[0]; LocalSubprocessDriver execs it directly
    (B Task 7). Nothing downstream prepends a program name any more."""
    assert _plan(CASES[0]).argv[0] == "llama-server"


def test_every_flag_exists_in_the_captured_help():
    """The golden cannot freeze a flag llama.cpp does not have. Checked against
    Task 2's REAL --help capture, so a flag that upstream renames fails here at
    the next fixture refresh rather than at 3am on the host."""
    help_text = HELP.read_text()
    golden = json.loads(GOLDEN.read_text())
    flags = {tok for case in golden.values() for tok in case["argv"] if tok.startswith("--")}
    missing = [
        f
        for f in sorted(flags)
        if not re.search(rf"(^|[\s,]){re.escape(f)}([\s,=]|$)", help_text, re.M)
    ]
    assert missing == [], f"flags absent from llama-server --help: {missing}"


def test_bind_host_is_loopback_under_the_local_driver():
    """#211, restated for this backend's own server. llama-server's OpenAI API
    is unauthenticated -- and its /health is explicitly exempt from even the
    --api-key check -- so under the in-container subprocess driver, where the
    engine shares the warden's netns, binding anything but loopback publishes a
    free LLM API on the pod IP."""
    assert BACKEND.bind_host("local") == "127.0.0.1"
    assert BACKEND.bind_host("k8s") == "127.0.0.1"
    assert BACKEND.bind_host("anything-unknown") == "127.0.0.1"


def test_bind_host_is_wide_under_the_docker_driver():
    assert BACKEND.bind_host("docker") == "0.0.0.0"


def test_metrics_flag_is_unconditional():
    """Without --metrics, GET /metrics answers 501 not_supported_error and the
    whole live-stats panel is blank for every llama.cpp model."""
    assert "--metrics" in _plan(CASES[0]).argv


def test_webui_is_disabled():
    assert "--no-webui" in _plan(CASES[0]).argv


def test_single_gpu_uses_split_mode_none():
    argv = _plan(CASES[0]).argv
    i = argv.index("--split-mode")
    assert argv[i + 1] == "none"


def test_multi_gpu_uses_layer_split_regardless_of_strategy():
    """D4 and §9.3. llama.cpp's row mode is deprecated upstream and its tensor
    mode is experimental with hard constraints; mapping parallelism_strategy='tp'
    onto either would present llama.cpp as a tensor-parallel peer of vLLM, which
    it is not. The UI hides the control for this backend instead."""
    for name in ("two_gpus", "tp_strategy_still_layer_split"):
        case = next(c for c in CASES if c.name == name)
        argv = _plan(case).argv
        assert argv[argv.index("--split-mode") + 1] == "layer"
    assert "--tensor-split" not in _plan(CASES[4]).argv


def test_no_model_path_raises_before_any_process_starts():
    with pytest.raises(ValueError, match="model_path"):
        BACKEND.plan(StubModel(), port=1, bind_host="127.0.0.1", resolved=ResolvedModelPaths())


def test_no_model_path_says_where_it_looked_when_it_knows():
    """The resolver hands back the snapshot directory even when the file inside
    it is missing. Surfacing it turns "needs a path" into something an operator
    can act on without reading the code."""
    with pytest.raises(ValueError, match="/cache/models--org--repo"):
        BACKEND.plan(
            StubModel(),
            port=1,
            bind_host="127.0.0.1",
            resolved=ResolvedModelPaths(snapshot_dir=SNAP),
        )


def test_env_does_not_carry_the_hf_token():
    """llama-server only touches HuggingFace for the -hf download path, which we
    never pass. Handing a credential to a process that has no use for it widens
    the blast radius of a compromise for nothing."""
    env = _plan(CASES[0]).env
    assert "HUGGING_FACE_HUB_TOKEN" not in env
    assert "HF_TOKEN" not in env


def test_env_pins_device_order():
    """The 2026-05-08 CUDA_VISIBLE_DEVICES incident, restated. ggml enumerates
    CUDA devices in CUDA's order; without PCI_BUS_ID, NVML may reorder by SM
    count and gpu_indices stops meaning what the operator picked. Bonus mixes an
    A4000 and a Quadro RTX 5000, so this is live, not theoretical."""
    env = _plan(CASES[0]).env
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_llama_arg_env_is_not_on_the_allowlist():
    """Every llama.cpp CLI flag has an LLAMA_ARG_* env equivalent -- including
    LLAMA_ARG_HOST, LLAMA_ARG_MODEL, LLAMA_ARG_PORT and
    LLAMA_ARG_ENDPOINT_METRICS. Allowing the LLAMA_ prefix through extra_env
    would let a model row re-bind the unauthenticated API to 0.0.0.0 (#211
    again), swap the served weights, move the port out from under the health
    probe, or silently disable metrics. GGML_ has no such power."""
    caps = BACKEND.capabilities
    assert "LLAMA_" not in caps.env_prefixes
    assert caps.env_prefixes == ("GGML_",)


def test_ggml_env_passes_through():
    """LAUNCH-time semantics: filter_extra_env keeps what is on the allowlist
    and DROPS the rest silently.

    That is not in tension with test_extra_env_allowlist_is_per_backend (Task 8),
    which asserts the same key RAISES -- the two pin different layers, and both
    behaviours already exist for vLLM:

      * ``ModelCreate`` (the API, write time) RAISES on an off-allowlist key, so
        an operator typing one gets an immediate, descriptive 422 rather than
        env that mysteriously never appears. B made that validator per-backend.
      * ``filter_extra_env`` (the launch, spawn time) DROPS silently, because by
        then the row may predate the check -- it could have been written when
        the row was still vLLM, or by a PATCH that bypasses ModelCreate -- and
        refusing to launch a model over an inert env key would be a worse
        failure than ignoring it. The plan's Global Constraint pins exactly this
        function's semantics "byte-for-byte", and this is them.
    """
    model = StubModel(
        extra_env={"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1", "VLLM_LOGGING_LEVEL": "DEBUG"}
    )
    case = Case("x", model, ResolvedModelPaths(model_path=f"{SNAP}/m.gguf", snapshot_dir=SNAP))
    env = _plan(case).env
    assert env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] == "1"
    assert "VLLM_LOGGING_LEVEL" not in env  # narrowing, and it is silent


@pytest.mark.parametrize(
    "key",
    [
        "LLAMA_ARG_HOST",
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_PORT",
        "LLAMA_ARG_ENDPOINT_METRICS",
        "LLAMA_CACHE",
    ],
)
def test_dangerous_llama_env_is_hard_locked(key):
    """Belt and braces on top of the allowlist: these five are refused LOUDLY at
    write time rather than dropped silently, because each one breaches a
    boundary rather than merely doing nothing."""
    from app.runtime.backends.vllm.env import HARD_LOCKED_ENV_KEYS

    assert key in HARD_LOCKED_ENV_KEYS


def test_hard_locked_llama_env_raises_at_launch_too():
    """The hard-locked set is checked FIRST in filter_extra_env and is global, so
    it is a raise at BOTH layers -- unlike the allowlist, which raises at the API
    and drops at the launch."""
    model = StubModel(extra_env={"LLAMA_ARG_HOST": "0.0.0.0"})
    case = Case("x", model, ResolvedModelPaths(model_path=f"{SNAP}/m.gguf", snapshot_dir=SNAP))
    with pytest.raises(ValueError, match="hard-locked"):
        _plan(case)


def test_capabilities_are_honest():
    caps = BACKEND.capabilities
    assert caps.name == "llamacpp"
    assert caps.display_name == "llama.cpp"
    assert caps.health_path == "/health"
    assert caps.metrics_path == "/metrics"
    assert caps.supports_vision is True
    assert caps.supports_request_priority is False
    assert caps.supports_tensor_parallel is False  # layer split is not TP
    assert caps.supports_pipeline_parallel is True
    assert caps.supports_lora is False
    # The ENGINE fact: upstream publishes per-build tagged server images, so
    # llama.cpp's version is choosable in principle -- same as vLLM's.
    assert caps.supports_version_pin is True


@pytest.mark.parametrize(
    "driver",
    [
        None,
        type("Subprocess", (), {"supports_engine_image": False})(),
        type("Docker", (), {"supports_engine_image": True})(),
        type("K8s", (), {"supports_engine_image": True})(),
        type("StandIn", (), {})(),
    ],
)
def test_version_pin_is_unavailable_under_every_driver(driver):
    """False everywhere, and NOT because the driver cannot swap an image --
    the Docker and K8s stand-ins above can. It is because nothing in this build
    can resolve a llama.cpp version to an image: backends/vllm/images.py is
    vLLM-only and C adds no counterpart. Returning the driver's answer here
    would enable a selector whose selection resolves to a *vLLM* image and
    launches vLLM under the operator's llama.cpp name."""
    assert BACKEND.version_pin_available(driver) is False


def test_version_pin_available_rejects_a_driver_NAME():
    """The defect this class is guarded for, and the reason the guard is a
    raise rather than a convention.

    bind_host takes the driver NAME; version_pin_available takes the driver
    OBJECT. The usual implementation is
    bool(getattr(driver, "supports_engine_image", False)), so passing a name
    silently yields False -- failing closed, logging nothing, and returning the
    RIGHT answer for the WRONG reason in the subprocess case, which is why every
    subprocess fixture still passes. Sub-project E shipped exactly this into
    review with the parameter annotated `driver: str`.

    This backend needs the raise MORE than vLLM does: it returns False
    unconditionally, so no return value could ever reveal a mistyped argument.
    Without the guard, a caller passing a name would be wrong forever and never
    find out."""
    with pytest.raises(TypeError, match="not the name"):
        BACKEND.version_pin_available("docker")
    with pytest.raises(TypeError, match="not the name"):
        BACKEND.version_pin_available("local")


def test_bind_host_rejects_a_driver_OBJECT():
    """The symmetric guard. A driver object reaching bind_host would compare
    unequal to "docker" and silently return loopback -- safe under #211's
    fail-narrow rule but wrong: a docker engine would be unreachable and the
    failure would surface 40 seconds later as a health-probe timeout instead of
    here."""
    with pytest.raises(TypeError, match="takes the driver NAME"):
        BACKEND.bind_host(type("Docker", (), {"supports_engine_image": True})())


def test_the_two_error_messages_name_each_other():
    """What makes the guards useful rather than merely strict: whoever hits one
    is holding the wrong mental model, so the message has to point at the member
    they actually wanted."""
    with pytest.raises(TypeError) as e:
        BACKEND.version_pin_available("docker")
    assert "bind_host" in str(e.value)
    with pytest.raises(TypeError) as e:
        BACKEND.bind_host(object())
    assert "version_pin_available" in str(e.value)


def test_signatures_match_B_s_disambiguated_names():
    """B renamed these at dcdd57e precisely so the Protocol stops inviting the
    error. A backend that drifts back to a bare `driver` on both re-opens it."""
    import inspect

    assert "driver_name" in inspect.signature(LlamaCppBackend.bind_host).parameters
    assert "driver" in inspect.signature(LlamaCppBackend.version_pin_available).parameters


def test_health_url():
    assert BACKEND.health_url("127.0.0.1", 10001) == "http://127.0.0.1:10001/health"
