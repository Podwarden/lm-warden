import pytest

from app.db.repos.models import ModelRow
from app.runtime.backends.vllm.args import build_vllm_args
from app.runtime.supervisor import Supervisor
from tests.fakes.fake_engine import FakeDriver


def _row(**overrides) -> ModelRow:
    base = dict(
        id="qwen",
        served_model_name="qwen3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
        hf_revision="main",
        gpu_indices=[0, 1],
        tensor_parallel_size=2,
        dtype="auto",
        max_model_len=8192,
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


def test_build_vllm_args_minimal(monkeypatch):
    # Pin the env explicitly so this stays a pure argv-shape regression guard
    # and doesn't tangle with the bind-host contract (covered by the #211
    # tests below).
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    args = build_vllm_args(_row(), port=10001, driver="local")
    assert args[0] == "--model"
    assert args[1] == "Qwen/Qwen3.5-9B"
    assert args[args.index("--revision") + 1] == "main"
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--port") + 1] == "10001"
    assert args[args.index("--served-model-name") + 1] == "qwen3.5-9b"
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_vllm_args_bind_host_local_driver_is_loopback(monkeypatch):
    # #211: the local-subprocess driver is what production runs, and its
    # engine is a child process in THIS netns — the control plane reaches it
    # on loopback. Binding 0.0.0.0 there (the pre-#211 behaviour) published
    # the engine's unauthenticated /v1/* on the pod IP, so anything that
    # could route to the pod got a free LLM API with no bearer check, no
    # priority scheduling and no token accounting. Loopback is both sufficient and
    # the security boundary; this test is the regression guard.
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    args = build_vllm_args(_row(), port=10001, driver="local")
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_vllm_args_bind_host_defaults_to_loopback_when_driver_omitted(monkeypatch):
    # The `driver` kwarg defaults to the NARROW bind on purpose (#211): a
    # caller that never learned about the parameter (an out-of-tree script, a
    # future call site) must fail towards "unreachable off-host", never
    # towards "unauthenticated API on every interface". Asserting the default
    # directly, because the whole point is the caller that forgets to pass it.
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    args = build_vllm_args(_row(), port=10001)
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_vllm_args_bind_host_unknown_driver_is_loopback(monkeypatch):
    # Same fail-narrow rule for a driver name this function doesn't know
    # (#211). A future out-of-netns driver that needs a wider bind must
    # announce itself with a loud failure — health probe times out at first
    # load — rather than silently inheriting 0.0.0.0.
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    args = build_vllm_args(_row(), port=10001, driver="podman")
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_vllm_args_bind_host_docker_driver_binds_all_interfaces(monkeypatch):
    # Regression for #180, preserved through #211: the docker-socket driver
    # runs each engine as a SEPARATE container. Loopback inside that netns is
    # unreachable from both the api container and docker-proxy (which
    # forwards to the container's eth0 and gets RST), so the supervisor
    # health probe times out and the row flips to ``failed`` even though the
    # engine logged ``Application startup complete.``. This driver must keep
    # binding 0.0.0.0 with the env var UNSET.
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    args = build_vllm_args(_row(), port=10001, driver="docker")
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_build_vllm_args_bind_host_env_overrides_local_default(monkeypatch):
    # VW_ENGINE_BIND_HOST stays the operator escape hatch (#211) — e.g.
    # binding one specific pod IP that a sidecar needs. It just no longer
    # decides the safe default. It MUST still win over the local driver's
    # loopback, read per-call so no module reimport is needed.
    monkeypatch.setenv("VW_ENGINE_BIND_HOST", "10.42.0.7")
    args = build_vllm_args(_row(), port=10001, driver="local")
    assert args[args.index("--host") + 1] == "10.42.0.7"


def test_build_vllm_args_bind_host_env_overrides_docker_default(monkeypatch):
    # The escape hatch cuts the other way too: an operator on the docker
    # driver who fronts the engine differently can narrow it (#211).
    monkeypatch.setenv("VW_ENGINE_BIND_HOST", "127.0.0.1")
    args = build_vllm_args(_row(), port=10001, driver="docker")
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_vllm_args_bind_host_blank_env_is_treated_as_unset(monkeypatch):
    # An exported-but-empty VW_ENGINE_BIND_HOST (trivially easy in a compose
    # file or a k8s env block: ``VW_ENGINE_BIND_HOST=""``) would otherwise
    # emit ``--host ''`` and kill the engine at argparse. Treat blank as
    # unset and fall back to the driver default (#211).
    monkeypatch.setenv("VW_ENGINE_BIND_HOST", "  ")
    args = build_vllm_args(_row(), port=10001, driver="docker")
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_build_vllm_args_omits_revision_when_none():
    args = build_vllm_args(_row(hf_revision=None), port=10001)
    assert "--revision" not in args


def test_build_vllm_args_includes_dtype_and_max_len():
    args = build_vllm_args(_row(dtype="bfloat16", max_model_len=4096), port=10001)
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--max-model-len") + 1] == "4096"


# --- Generic overrides kwarg (preserved post-bench-removal) -----------------


def test_overrides_none_is_byte_identical_to_legacy_path():
    """Passing overrides=None must produce exactly the same argv as the
    original single-arg call. Guards against regressing the existing load path."""
    row = _row()
    assert build_vllm_args(row, port=10001) == build_vllm_args(
        row, port=10001, overrides=None
    )
    # Empty dict must also be a no-op.
    assert build_vllm_args(row, port=10001) == build_vllm_args(
        row, port=10001, overrides={}
    )


def test_overrides_apply_quantization_and_max_num_seqs():
    args = build_vllm_args(
        _row(),
        port=10001,
        overrides={"quantization": "mxfp4", "max_num_seqs": 64},
    )
    assert args[args.index("--quantization") + 1] == "mxfp4"
    assert args[args.index("--max-num-seqs") + 1] == "64"


def test_overrides_replace_tp_gpu_mem_max_len():
    args = build_vllm_args(
        _row(tensor_parallel_size=2, gpu_memory_utilization=0.9, max_model_len=8192),
        port=10001,
        overrides={
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 16384,
        },
    )
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert args[args.index("--max-model-len") + 1] == "16384"


def test_overrides_do_not_mutate_model_row():
    """Overrides apply in-memory only — the ModelRow is read, never written."""
    row = _row(tensor_parallel_size=2, gpu_memory_utilization=0.9, max_model_len=8192)
    build_vllm_args(
        row,
        port=10001,
        overrides={
            "quantization": "fp8",
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 32768,
            "max_num_seqs": 128,
        },
    )
    assert row.tensor_parallel_size == 2
    assert row.gpu_memory_utilization == 0.9
    assert row.max_model_len == 8192


def test_override_max_model_len_to_none_omits_flag():
    """An explicit None override means 'let vLLM pick' even if the row has a value."""
    args = build_vllm_args(
        _row(max_model_len=8192),
        port=10001,
        overrides={"max_model_len": None},
    )
    assert "--max-model-len" not in args


def test_unknown_override_keys_are_ignored():
    """Forward compatibility: an unknown key from a future load-config dimension
    must not crash the builder."""
    args = build_vllm_args(
        _row(),
        port=10001,
        overrides={"some_future_knob": 42},
    )
    # No flag named --some-future-knob, builder returned successfully.
    assert "--some-future-knob" not in args


# --- Parallelism strategy (#88) ----------------------------------------------


def test_parallelism_strategy_auto_emits_tensor_parallel_size():
    """``auto`` is the pre-#85 default and must keep emitting --tensor-parallel-size
    so existing models (legacy rows + the wizard's default pick) don't regress."""
    args = build_vllm_args(_row(parallelism_strategy="auto"), port=10001)
    assert "--tensor-parallel-size" in args
    assert "--pipeline-parallel-size" not in args
    assert args[args.index("--tensor-parallel-size") + 1] == "2"


def test_parallelism_strategy_tp_emits_tensor_parallel_size():
    """``tp`` is the explicit form of the auto default — same flag, same N."""
    args = build_vllm_args(_row(parallelism_strategy="tp"), port=10001)
    assert "--tensor-parallel-size" in args
    assert "--pipeline-parallel-size" not in args
    assert args[args.index("--tensor-parallel-size") + 1] == "2"


def test_parallelism_strategy_pp_emits_pipeline_parallel_size():
    """``pp`` swaps the TP flag for --pipeline-parallel-size; N is identical
    (schemas enforces tensor_parallel_size == len(gpu_indices))."""
    args = build_vllm_args(_row(parallelism_strategy="pp"), port=10001)
    assert "--pipeline-parallel-size" in args
    assert "--tensor-parallel-size" not in args
    assert args[args.index("--pipeline-parallel-size") + 1] == "2"


def test_parallelism_strategy_pp_single_host_not_blocked():
    """vLLM handles single-host PP fine (CTO-decided in #82 plan). The builder
    must NOT raise / refuse for a single-host row — gpu_indices on one host
    is the common case."""
    # Two GPUs, single host — exactly the scenario the dispatch calls out.
    args = build_vllm_args(
        _row(gpu_indices=[0, 1], tensor_parallel_size=2, parallelism_strategy="pp"),
        port=10001,
    )
    assert "--pipeline-parallel-size" in args
    assert args[args.index("--pipeline-parallel-size") + 1] == "2"


def test_parallelism_strategy_pp_with_tp_override_uses_override_value_on_pp_flag():
    """An external ``tensor_parallel_size`` override remains the N source
    even when strategy=pp — the override is the parallelism *degree*, not
    specifically the TP flag."""
    args = build_vllm_args(
        _row(parallelism_strategy="pp", tensor_parallel_size=2),
        port=10001,
        overrides={"tensor_parallel_size": 4},
    )
    assert args[args.index("--pipeline-parallel-size") + 1] == "4"
    assert "--tensor-parallel-size" not in args


def test_parallelism_strategy_missing_attr_defaults_to_auto():
    """Defensive: a row constructed without the parallelism_strategy attr
    (e.g. an ad-hoc fake in another test, or a future caller that builds
    a stand-in object) must fall through to the TP flag — same as the
    pre-#85 builder."""

    class _BareRow:
        hf_repo = "o/r"
        hf_revision = "main"
        served_model_name = "x"
        gpu_indices = [0]
        tensor_parallel_size = 1
        dtype = None
        max_model_len = None
        gpu_memory_utilization = 0.9

    args = build_vllm_args(_BareRow(), port=10001)
    assert "--tensor-parallel-size" in args
    assert "--pipeline-parallel-size" not in args


# --- GGUF quant suffix (#100) ------------------------------------------------


def test_gguf_filename_appends_quant_suffix_to_model_arg():
    """vLLM 0.20.0 requires GGUF models to be addressed as ``repo_id:quant_type``.
    The launcher must derive the quant tag from ``filename`` and join it to
    ``hf_repo`` with a colon. Without this, vLLM subprocess startup fails rc=1
    on every GGUF deployment since #85."""
    args = build_vllm_args(
        _row(
            hf_repo="unsloth/Qwen3.6-27B-GGUF",
            filename="Qwen3.6-27B-Q5_K_M.gguf",
        ),
        port=10001,
    )
    assert args[args.index("--model") + 1] == "unsloth/Qwen3.6-27B-GGUF:Q5_K_M"


def test_gguf_filename_none_falls_back_to_bare_repo():
    """A row with no filename (legacy / whole-repo GGUF, or safetensors) must
    not crash and must emit the bare hf_repo — same as the pre-#100 builder."""
    args = build_vllm_args(_row(filename=None), port=10001)
    assert args[args.index("--model") + 1] == "Qwen/Qwen3.5-9B"


def test_non_gguf_filename_does_not_append_suffix():
    """If ``filename`` is set on a safetensors row (e.g. partial download
    bookkeeping), the .gguf guard must skip it. No colon, no surprise."""
    args = build_vllm_args(
        _row(
            hf_repo="meta-llama/Llama-3-8B",
            filename="model-00001-of-00002.safetensors",
        ),
        port=10001,
    )
    assert args[args.index("--model") + 1] == "meta-llama/Llama-3-8B"


def test_gguf_filename_without_quant_tag_falls_back_to_bare_repo():
    """An unquantized / non-standard GGUF filename (no ``-Q*`` segment) leaves
    the model arg as the bare repo. Conservative: better to let vLLM error
    visibly than to guess a wrong tag."""
    args = build_vllm_args(
        _row(
            hf_repo="some-org/SomeModel-GGUF",
            filename="model.gguf",
        ),
        port=10001,
    )
    assert args[args.index("--model") + 1] == "some-org/SomeModel-GGUF"


def test_gguf_filename_case_insensitive_extension():
    """Some HF repos ship ``.GGUF`` (uppercase). The endswith guard is case-
    insensitive so the suffix still gets derived."""
    args = build_vllm_args(
        _row(
            hf_repo="some-org/Model-GGUF",
            filename="Model-Q4_K_M.GGUF",
        ),
        port=10001,
    )
    assert args[args.index("--model") + 1] == "some-org/Model-GGUF:Q4_K_M"


def test_gguf_filename_extended_quant_variant():
    """Unsloth-style extended quant tags (``UD-Q4_K_XL``, ``IQ4_NL``) — the
    pattern anchors on ``-Q\\d`` so ``-Q4_K_XL`` matches and yields ``Q4_K_XL``.
    Variants without a leading ``Q<digit>`` (e.g. pure ``IQ4_NL``) intentionally
    fall back to the bare repo — fixable in a follow-up if/when we see one in
    the wild."""
    args = build_vllm_args(
        _row(
            hf_repo="unsloth/Model-GGUF",
            filename="Model-UD-Q4_K_XL.gguf",
        ),
        port=10001,
    )
    assert args[args.index("--model") + 1] == "unsloth/Model-GGUF:Q4_K_XL"


# --- --hf-config-path / --tokenizer plumbing (#106) ---------------------------


def test_hf_config_repo_emits_flag():
    """``hf_config_repo`` set on the row must produce ``--hf-config-path <repo>``.
    Required for GGUF repos that omit config.json (unsloth republishes); vLLM
    0.20.0 errors out at startup without this flag."""
    args = build_vllm_args(
        _row(
            hf_repo="unsloth/Qwen3-30B-A3B-GGUF",
            filename="Qwen3-30B-A3B-Q5_K_M.gguf",
            hf_config_repo="Qwen/Qwen3-30B-A3B",
        ),
        port=10001,
    )
    assert "--hf-config-path" in args
    assert args[args.index("--hf-config-path") + 1] == "Qwen/Qwen3-30B-A3B"


def test_tokenizer_repo_emits_flag():
    """``tokenizer_repo`` set on the row must produce ``--tokenizer <repo>``.
    Used when the quantized repo's tokenizer is missing/stale and the operator
    wants the upstream tokenizer."""
    args = build_vllm_args(
        _row(tokenizer_repo="Qwen/Qwen3-30B-A3B"),
        port=10001,
    )
    assert "--tokenizer" in args
    assert args[args.index("--tokenizer") + 1] == "Qwen/Qwen3-30B-A3B"


def test_both_repos_emit_both_flags():
    """The common unsloth case: same upstream for config + tokenizer. Both
    flags are emitted; ordering is stable (config before tokenizer) so
    operators reading vllm logs see them adjacent."""
    args = build_vllm_args(
        _row(
            hf_config_repo="Qwen/Qwen3-30B-A3B",
            tokenizer_repo="Qwen/Qwen3-30B-A3B",
        ),
        port=10001,
    )
    assert args[args.index("--hf-config-path") + 1] == "Qwen/Qwen3-30B-A3B"
    assert args[args.index("--tokenizer") + 1] == "Qwen/Qwen3-30B-A3B"


def test_neither_repo_omits_flags():
    """Default state (both None) must NOT emit either flag — pre-#106
    behaviour preserved for non-GGUF and self-contained GGUF rows."""
    args = build_vllm_args(_row(), port=10001)
    assert "--hf-config-path" not in args
    assert "--tokenizer" not in args


# --- Priority scheduling policy (#173 part B) --------------------------------


def test_scheduling_policy_priority_emitted_by_default():
    """Every engine launches with vLLM's priority scheduler so the per-request
    ``priority`` field the proxy injects (#173 part B) is honoured by the
    engine's own waiting queue. With all-equal priorities this is identical to
    FCFS, so it is a safe unconditional default."""
    args = build_vllm_args(_row(), port=10001)
    assert args[args.index("--scheduling-policy") + 1] == "priority"


def test_scheduling_policy_overridable_via_extra_args():
    """extra_args is appended last, so an operator who wants plain FCFS can
    override the default by setting ``--scheduling-policy fcfs`` there
    (argparse uses the last occurrence)."""
    args = build_vllm_args(
        _row(extra_args=["--scheduling-policy", "fcfs"]),
        port=10001,
    )
    # Default still present, but the override is the LAST occurrence → wins.
    idxs = [i for i, a in enumerate(args) if a == "--scheduling-policy"]
    assert len(idxs) == 2
    assert args[idxs[-1] + 1] == "fcfs"


# --- Supervisor threads the driver into the builder (#211) -------------------
#
# The tests above pin the builder's contract in isolation; these two pin the
# ONE production call site that has to honour it (Supervisor.load →
# build_vllm_args). They live in this file rather than in
# test_supervisor_driver.py because what they assert is the cmd_builder
# contract — that the argv the engine actually launches with carries the bind
# host the active driver requires — not the supervisor's driver dispatch.
#
# Without them, deleting the ``driver=`` kwarg from the supervisor's call
# would leave every builder test above green while the docker driver silently
# regressed to loopback (#180 all over again). It cannot regress the other way
# — the builder's default is the narrow bind — but a green suite that doesn't
# notice a driver-selection bug at all is worth closing.


class _SupSettings:
    """Minimal stand-in for the settings object Supervisor.load reads."""

    def __init__(self, tmp_path, engine_driver: str):
        self.data_dir = str(tmp_path)
        self.hf_token_path = str(tmp_path / "tok")
        self.hf_cache_dir = str(tmp_path / "hf-cache")
        self.engine_driver = engine_driver


class _SupModel:
    id = "qwen"
    hf_repo = "Qwen/Q"
    hf_revision = "main"
    served_model_name = "q"
    gpu_indices = [0]
    tensor_parallel_size = 1
    max_model_len = 4096
    dtype = "auto"
    gpu_memory_utilization = 0.9
    extra_env: dict = {}
    extra_args: list = []


@pytest.mark.asyncio
async def test_supervisor_load_binds_loopback_under_local_driver(tmp_path, monkeypatch):
    # The production default (app/config.py: VW_ENGINE_DRIVER defaults to
    # "local", i.e. the in-pod subprocess engine). The spawned argv must carry
    # 127.0.0.1 so the unauthenticated /v1/* never appears on the pod IP.
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    drv = FakeDriver()
    sup = Supervisor(_SupSettings(tmp_path, "local"), driver=drv)
    await sup.load(_SupModel(), port=8001)
    args = drv.spawned[0].argv
    assert args[args.index("--host") + 1] == "127.0.0.1"
    # Reap the exit-watcher task; a bare load() leaves it pending and pytest
    # prints "Task was destroyed but it is pending!" at interpreter shutdown.
    await sup.unload("qwen", force=True)


@pytest.mark.asyncio
async def test_supervisor_load_binds_all_interfaces_under_docker_driver(
    tmp_path, monkeypatch
):
    # The docker driver's engine lives in its own netns, so 0.0.0.0 has to
    # survive all the way into the spawned spec — a loopback bind there is the
    # #180 "Application startup complete. …then health probe timeout" failure.
    monkeypatch.delenv("VW_ENGINE_BIND_HOST", raising=False)
    drv = FakeDriver()
    sup = Supervisor(_SupSettings(tmp_path, "docker"), driver=drv)
    await sup.load(_SupModel(), port=8001)
    args = drv.spawned[0].argv
    assert args[args.index("--host") + 1] == "0.0.0.0"
    await sup.unload("qwen", force=True)
