import pytest

from app.runtime.backends import Backend, LaunchPlan
from app.runtime.backends.vllm import VllmBackend
from app.runtime.backends.vllm.diagnostics import EngineDiagnosis
from tests.unit.runtime.backends.corpus import CASES, load_golden

GOLDEN = load_golden()
BACKEND = VllmBackend()


def test_vllm_backend_satisfies_the_protocol():
    assert isinstance(BACKEND, Backend)


def test_capabilities_are_advertised_not_inferred():
    caps = BACKEND.capabilities
    assert caps.name == "vllm"
    assert caps.display_name == "vLLM"
    assert caps.supports_tensor_parallel is True
    assert caps.supports_pipeline_parallel is True
    assert caps.max_gpus is None
    assert caps.health_path == "/health"
    assert caps.metrics_path == "/metrics"
    assert caps.supports_request_priority is True
    assert caps.openai_paths == frozenset({"/v1/chat/completions", "/v1/completions", "/v1/models"})


def test_env_allowlist_on_capabilities_matches_the_historical_value():
    """The allowlist is a security boundary. Moving it onto the backend must
    not change WHICH keys it accepts -- only where the tuple lives."""
    caps = BACKEND.capabilities
    assert caps.env_prefixes == ("VLLM_", "TRITON_", "NCCL_", "PYTORCH_", "TORCH_", "OMP_")
    assert caps.env_exact == frozenset(
        {"CUDA_MODULE_LOADING", "PYTHONFAULTHANDLER", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"}
    )


class _SubprocessDriver:
    supports_engine_image = False


class _DockerDriver:
    supports_engine_image = True


class _StandInDriver:
    pass


def test_supports_version_pin_is_true_and_driver_invariant():
    """The ENGINE fact. vLLM publishes many versioned upstream images, so the
    answer to "can this engine be pinned at all?" is True, full stop -- it does
    not depend on how we happen to launch it."""
    assert BACKEND.capabilities.supports_version_pin is True


def test_version_pin_is_NOT_available_under_the_subprocess_driver():
    """The DEPLOYMENT fact, and the answer for `bonus` as deployed. It must
    not change.

    The engine runs as a child of the warden process, so its vLLM version is
    whatever the warden image baked in -- nothing to pin, however pinnable vLLM
    is in principle. This is exactly what /api/system/engine has always
    reported as supports_version_select=false."""
    assert BACKEND.version_pin_available(_SubprocessDriver()) is False


def test_version_pin_is_available_under_the_docker_driver():
    assert BACKEND.version_pin_available(_DockerDriver()) is True


def test_the_two_fields_disagree_on_bonus_and_that_is_the_point():
    """Guard against the two being collapsed back into one boolean.

    On the deployment we actually run, the engine CAN be pinned and this
    deployment CANNOT honour it. A single boolean has to pick one truth and
    mislead about the other -- which is why E split them."""
    subprocess_driver = _SubprocessDriver()
    assert BACKEND.capabilities.supports_version_pin is True
    assert BACKEND.version_pin_available(subprocess_driver) is False


def test_passing_a_driver_NAME_to_version_pin_available_raises():
    """Regression guard for a defect that has already shipped once.

    E typed `version_pin_available(driver: str)` and called it with "docker".
    getattr on a str returns the default, so it answered False and disabled the
    version selector on every docker deployment. It is camouflaged: the
    "subprocess" case returns the correct answer for the wrong reason, so only
    a docker assertion catches it.

    An unknown driver OBJECT is a legitimate unknown and answers False. A
    STRING is a programming error and must not be given a plausible answer.
    """
    for name in ("local", "docker", "k8s", ""):
        with pytest.raises(TypeError, match="not the name"):
            BACKEND.version_pin_available(name)


def test_passing_a_driver_OBJECT_to_bind_host_raises():
    """The mirror-image mistake. A driver object compares unequal to "docker",
    so bind_host would return loopback -- safe under #211's fail-narrow rule,
    but wrong: a docker engine would be unreachable and the failure would
    surface as a health timeout 40 seconds later rather than here."""
    with pytest.raises(TypeError, match="takes the driver NAME"):
        BACKEND.bind_host(_DockerDriver())


def test_version_pin_availability_fails_CLOSED_on_an_unknown_driver():
    """Advertising and enforcing get OPPOSITE defaults, deliberately.

    Here we are answering "may the operator pick a version?" -- and offering a
    pin we cannot honour is the #177 bug, where a silently-discarded pin
    launched the warden-baked version instead. So an unknown driver gets False.

    Supervisor.load's REFUSAL path keeps the opposite default
    (getattr(driver, "supports_engine_image", True)) so a test stand-in driver
    is never wrongly blocked from loading. Both directions fail safe; they just
    fail safe towards different things."""
    assert BACKEND.version_pin_available(_StandInDriver()) is False
    assert BACKEND.version_pin_available(None) is False


@pytest.mark.parametrize(
    "driver_name,expected",
    [("local", "127.0.0.1"), ("docker", "0.0.0.0"), ("k8s", "127.0.0.1"), ("", "127.0.0.1")],
)
def test_bind_host_defaults_narrow(driver_name, expected):
    """#211: any unknown driver NAME resolves to loopback, so forgetting to
    update this is an outage, never a breach."""
    assert BACKEND.bind_host(driver_name) == expected


def test_health_url():
    assert BACKEND.health_url("10.0.0.5", 10001) == "http://10.0.0.5:10001/health"


def test_diagnose_delegates_to_the_moved_grammar():
    d = BACKEND.diagnose("torch.OutOfMemoryError: CUDA out of memory.")
    assert isinstance(d, EngineDiagnosis)
    assert "ran out of memory" in d.message
    assert BACKEND.diagnose("") is None


def test_parse_metrics_returns_none_on_garbage():
    assert BACKEND.parse_metrics("") is None


def test_parse_metrics_normalises_a_data_parallel_kv_gauge():
    """#280: vLLM >= 0.26 exposes one kv_cache_usage_perc series per engine
    (``engine="0".."N-1"``), each a 0..1 fraction of THAT rank's pool, and one
    cache_config_info series per rank. The reading must come out as the
    engine-wide fraction (sum / ranks) over the engine-wide pool (per-rank
    blocks * ranks), or build_frame's used = perc * total exceeds the pool."""
    r = BACKEND.parse_metrics(
        'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.5\n'
        'vllm:kv_cache_usage_perc{model_name="m",engine="1"} 0.25\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="100",'
        'num_cpu_blocks="0",cache_dtype="auto",engine="0"} 1.0\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="100",'
        'num_cpu_blocks="0",cache_dtype="auto",engine="1"} 1.0\n'
    )
    assert r is not None
    assert r.kv_cache_usage_perc == pytest.approx(0.375)
    assert r.kv_tokens_total == 2 * 16 * 100


def test_parse_metrics_single_engine_label_is_unchanged():
    """One engine label is still one engine: a DP-1 exposition carries the same
    engine="0" labels and must read exactly as the pre-#280 code did."""
    r = BACKEND.parse_metrics(
        'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.5\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="100",'
        'num_cpu_blocks="0",cache_dtype="auto",engine="0"} 1.0\n'
    )
    assert r is not None
    assert r.kv_cache_usage_perc == pytest.approx(0.5)
    assert r.kv_tokens_total == 16 * 100


def test_parse_metrics_reads_a_vllm_gauge():
    """Sub-project C: parse_metrics returns an EngineReading, not a Metrics.

    B's version handed the raw Metrics accessor back and left build_frame() to
    do the name lookups. C moved the 31 ``vllm:`` names into
    app/runtime/backends/vllm/metrics.py, so what comes out is already
    dialect-free -- which is the whole point of the seam.
    """
    r = BACKEND.parse_metrics(
        "# TYPE vllm:num_requests_running gauge\n" 'vllm:num_requests_running{model_name="x"} 3.0\n'
    )
    assert r is not None
    assert r.requests_running == 3.0
    # Everything this body does not publish reads absent, never zero.
    assert r.kv_cache_usage_perc is None


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_plan_reproduces_the_golden_launch(case):
    plan = BACKEND.plan(
        case.model,
        port=case.port,
        bind_host=BACKEND.bind_host(case.driver),
        overrides=case.overrides,
        hf_token=case.hf_token,
        hf_cache_dir=case.hf_cache_dir,
        engine_driver=case.driver,
    )
    assert isinstance(plan, LaunchPlan)
    # argv[0:2] is the launch head that used to be hard-coded in the driver.
    assert plan.argv[:2] == ["vllm", "serve"]
    assert plan.argv[2:] == GOLDEN[case.name]["argv"]
    assert plan.env == GOLDEN[case.name]["env"]
    assert plan.port == case.port
    assert plan.gpu_indices == list(case.model.gpu_indices)
    assert plan.image is None


def test_plan_filters_extra_env_through_the_backends_own_allowlist():
    from tests.unit.runtime.backends.corpus import StubModel

    model = StubModel(extra_env={"VLLM_LOGGING_LEVEL": "DEBUG", "LLAMA_ARG_N_GPU_LAYERS": "99"})
    plan = BACKEND.plan(model, port=1, bind_host="127.0.0.1", hf_token="t", hf_cache_dir="/c")
    assert plan.env["VLLM_LOGGING_LEVEL"] == "DEBUG"
    assert "LLAMA_ARG_N_GPU_LAYERS" not in plan.env
