"""MlxBackend -- capabilities and plan() for ``mlx_lm.server``.

Pinned from the Metal design s5 (the MLX backend) before the backend exists:
this file is red today only because the names it imports do not.

Every import of a NEW name happens INSIDE a test function, never at module
level: a module-level import of ``MlxBackend`` would make this whole file an
ImportError on the base tree, which runs nothing and is red for the wrong
reason.

The consistency legs compare ``engine_model_id`` against the value ``plan()``
actually puts on the ``--model`` flag, so the two cannot drift: the proxy
re-points a request ``body["model"]`` at whatever the engine loaded, and on MLX
that is the LOCAL PATH, not a registered name (``mlx_lm.server`` has no
``--served-model-name``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.runtime.backends import Backend
from app.runtime.backends.paths import ResolvedModelPaths


@dataclass
class StubModel:
    """A minimal model row for MLX.

    MLX needs a resolved directory, not a file, so the stub carries the fields
    ``plan()`` and ``engine_model_id`` read; ``gpu_indices`` is non-empty
    because launch needs a card index even though the engine sees one pool.
    """

    id: str = "m1"
    served_model_name: str = "qwen"
    gpu_indices: list[int] = field(default_factory=lambda: [0])
    max_model_len: int | None = 8192
    max_num_seqs: int | None = None
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


def _resolved():
    return ResolvedModelPaths(
        model_path="/cache/models--org--repo/snapshots/abc123",
        snapshot_dir="/cache/models--org--repo/snapshots/abc123",
    )


def test_mlx_backend_is_a_backend():
    from app.runtime.backends.mlx import MlxBackend

    assert isinstance(MlxBackend(), Backend)


def test_mlx_argv0_is_the_bare_console_script_name():
    from app.runtime.backends.mlx import MlxBackend

    plan = MlxBackend().plan(StubModel(), port=1234, bind_host="127.0.0.1", resolved=_resolved())
    assert plan.argv[0] == "mlx_lm.server"


def test_mlx_capabilities_fields():
    """One leg per capability field the Metal design s5 names."""
    from app.runtime.backends.mlx import MlxBackend

    c = MlxBackend().capabilities
    assert c.name == "mlx"
    assert c.max_gpus == 1
    assert c.supports_tensor_parallel is False
    assert c.supports_pipeline_parallel is False
    assert c.supports_data_parallel is False
    assert c.supports_request_priority is False
    assert c.needs_local_model_path is True
    assert c.metrics_path is None
    assert c.vram_cap_fraction == 1.0
    assert c.supports_version_pin is False
    # p5g fields (Metal design s5 B/C + the cached_tokens answer).
    assert c.health_means_ready is False
    assert c.telemetry == "proxy_observed"
    assert c.cached_tokens_reporting == "none"


def test_mlx_model_host_port_and_concurrency():
    from app.runtime.backends.mlx import MlxBackend

    plan = MlxBackend().plan(
        StubModel(max_num_seqs=32),
        port=4321,
        bind_host="10.0.0.5",
        resolved=_resolved(),
    )
    argv = plan.argv
    # --model is the resolved local path.
    assert argv[argv.index("--model") + 1] == ("/cache/models--org--repo/snapshots/abc123")
    assert argv[argv.index("--host") + 1] == "10.0.0.5"
    assert argv[argv.index("--port") + 1] == "4321"
    # max_num_seqs maps to --decode-concurrency.
    assert argv[argv.index("--decode-concurrency") + 1] == "32"


def test_mlx_decode_concurrency_boundaries():
    """max_num_seqs=1 -> concurrency 1, max_num_seqs=32 -> 32, None -> no flag.

    The flag is only emitted when the row sets a value, so omitting it and
    setting the minimum are distinct, and each must be pinned.
    """
    from app.runtime.backends.mlx import MlxBackend

    one = (
        MlxBackend()
        .plan(StubModel(max_num_seqs=1), port=1, bind_host="127.0.0.1", resolved=_resolved())
        .argv
    )
    assert one[one.index("--decode-concurrency") + 1] == "1"

    thirty_two = (
        MlxBackend()
        .plan(StubModel(max_num_seqs=32), port=1, bind_host="127.0.0.1", resolved=_resolved())
        .argv
    )
    assert thirty_two[thirty_two.index("--decode-concurrency") + 1] == "32"

    none = (
        MlxBackend()
        .plan(StubModel(max_num_seqs=None), port=1, bind_host="127.0.0.1", resolved=_resolved())
        .argv
    )
    assert "--decode-concurrency" not in none


def test_mlx_max_model_len_adds_no_flag():
    """max_model_len is advisory on MLX -- it feeds fit math, not argv. A set
    value must not produce a flag (there is none)."""
    from app.runtime.backends.mlx import MlxBackend

    plan = MlxBackend().plan(
        StubModel(max_model_len=16384),
        port=1,
        bind_host="127.0.0.1",
        resolved=_resolved(),
    )
    argv = plan.argv
    # The ONLY flags emitted: the model, host, port. max_model_len (and the
    # absent max_num_seqs) contribute nothing.
    flags = [a for a in argv if a.startswith("--")]
    assert flags == ["--model", "--host", "--port"]
    # The resolved path is still the --model value regardless.
    assert argv[argv.index("--model") + 1] == ("/cache/models--org--repo/snapshots/abc123")


def test_mlx_env_has_pythonunbuffered_and_path():
    from app.runtime.backends.mlx import MlxBackend

    plan = MlxBackend().plan(StubModel(), port=1, bind_host="127.0.0.1", resolved=_resolved())
    assert plan.env["PYTHONUNBUFFERED"] == "1"
    assert "PATH" in plan.env and plan.env["PATH"]


def test_mlx_version_pin_available_false():
    """False for every driver: None and a docker-shaped driver object."""
    from app.runtime.backends.mlx import MlxBackend

    class _Docker:
        supports_engine_image = True

    backend = MlxBackend()
    assert backend.version_pin_available(None) is False
    assert backend.version_pin_available(_Docker()) is False


def test_mlx_registry_unchanged():
    """NOT registered yet (Metal design s10: DEFAULT_BACKEND migration first)."""
    from app.runtime.backends import registry

    assert registry.available() == ("llamacpp", "vllm")


def test_mlx_health_url_uses_capabilities_path():
    from app.runtime.backends.mlx import MlxBackend

    backend = MlxBackend()
    assert backend.health_url("h", 8000) == f"http://h:8000{backend.capabilities.health_path}"


def test_mlx_engine_model_id_is_the_model_path():
    """engine_model_id == the exact string plan() passes to --model (the
    resolved local path), so the proxy can pin body['model'] to what the
    engine loaded."""
    from app.runtime.backends.mlx import MlxBackend

    backend = MlxBackend()
    row = StubModel()
    resolved = _resolved()
    argv = backend.plan(row, port=1, bind_host="127.0.0.1", resolved=resolved).argv
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == backend.engine_model_id(row, resolved)
    assert backend.engine_model_id(row, resolved) == ("/cache/models--org--repo/snapshots/abc123")


def test_mlx_engine_model_id_without_a_resolved_snapshot_refuses():
    """No resolved snapshot -> no pure answer: ValueError, never None and never
    the served name (an id mlx_lm.server would treat as a model to download)."""
    import pytest

    from app.runtime.backends.mlx import MlxBackend

    backend = MlxBackend()
    for resolved in (None, ResolvedModelPaths()):
        with pytest.raises(ValueError, match="model directory"):
            backend.engine_model_id(StubModel(), resolved)


def test_mlx_plan_refuses_without_a_resolved_snapshot():
    """--model takes a DIRECTORY: no snapshot (repo not pulled) is refused
    before any spawn -- never ``--model None``. A file pin alone
    (``model_path`` without ``snapshot_dir``) is not a directory either."""
    import pytest

    from app.runtime.backends.mlx import MlxBackend

    for resolved in (
        None,
        ResolvedModelPaths(),
        ResolvedModelPaths(model_path="/cache/x/model.safetensors"),
    ):
        with pytest.raises(ValueError, match="model directory"):
            MlxBackend().plan(StubModel(), port=1, bind_host="127.0.0.1", resolved=resolved)


def test_mlx_extra_env_hard_locked_keys_are_refused():
    """The child env goes through the shared hard-locked filter: the macOS
    interposition keys and LD_PRELOAD are refused loudly, not dropped."""
    import pytest

    from app.runtime.backends.mlx import MlxBackend

    for key in (
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "LD_PRELOAD",
        "PYTHONPATH",
        "PATH",
        "PYTHONUNBUFFERED",
    ):
        with pytest.raises(ValueError, match="hard-locked"):
            MlxBackend().plan(
                StubModel(extra_env={key: "x"}),
                port=1,
                bind_host="127.0.0.1",
                resolved=_resolved(),
            )


def test_mlx_extra_env_off_allowlist_is_dropped_and_hard_set_keys_win():
    from app.runtime.backends.mlx import ENGINE_PATH, MlxBackend

    env = (
        MlxBackend()
        .plan(
            StubModel(extra_env={"MLX_SOMETHING": "1", "VLLM_X": "1"}),
            port=1,
            bind_host="127.0.0.1",
            resolved=_resolved(),
        )
        .env
    )
    assert env == {"PYTHONUNBUFFERED": "1", "PATH": ENGINE_PATH}


def test_mlx_path_puts_the_running_venv_bin_first():
    """The interpreter's own directory is entry 0 (mlx_lm.server is installed
    into the warden's own venv, next to the interpreter), then the fixed
    fallbacks. Host-independent: it asserts the exact shape, not that a
    particular user dir is absent, so it holds wherever the host interpreter
    lives (including /usr/local/bin/python on the gate image)."""
    import os
    import sys

    from app.runtime.backends.mlx import MlxBackend

    path = (
        MlxBackend()
        .plan(StubModel(), port=1, bind_host="127.0.0.1", resolved=_resolved())
        .env["PATH"]
    )
    entries = path.split(":")
    assert entries[0] == os.path.dirname(sys.executable)
    assert entries[1:] == ["/usr/bin", "/bin"]
    assert "" not in entries


def test_mlx_engine_path_venv_interpreter():
    """The relocatable-bundle venv interpreter: its bin dir first, then the
    fixed fallbacks."""
    from app.runtime.backends.mlx import engine_path

    assert (
        engine_path("/Applications/LM Warden.app/Contents/Resources/venv/bin/python3")
        == "/Applications/LM Warden.app/Contents/Resources/venv/bin:/usr/bin:/bin"
    )


def test_mlx_engine_path_interpreter_under_usr_local_bin():
    """The gate's case, by injection: the interpreter's own dir is kept as
    entry 0 and NO other user dir is added -- only the fixed fallbacks follow."""
    from app.runtime.backends.mlx import engine_path

    assert engine_path("/usr/local/bin/python") == "/usr/local/bin:/usr/bin:/bin"
    assert engine_path("/opt/homebrew/bin/python3") == "/opt/homebrew/bin:/usr/bin:/bin"


def test_mlx_engine_path_empty_executable_has_no_leading_colon():
    """An empty executable (embedded interpreter) contributes nothing, so the
    result has no leading ':' that would put the working directory on PATH."""
    from app.runtime.backends.mlx import engine_path

    assert engine_path("") == "/usr/bin:/bin"


def test_mlx_engine_path_is_what_plan_uses():
    """ENGINE_PATH is exactly engine_path(sys.executable), so the module
    constant and the helper cannot drift."""
    import sys

    from app.runtime.backends.mlx import ENGINE_PATH, engine_path

    assert ENGINE_PATH == engine_path(sys.executable)


def test_mlx_parse_metrics_is_none():
    """No /metrics endpoint (Metal design s5 C): parse_metrics answers None."""
    from app.runtime.backends.mlx import MlxBackend

    backend = MlxBackend()
    assert backend.parse_metrics("foo 1\nbar 2\n") is None


def test_mlx_diagnose_delegates_to_diagnostics():
    from app.runtime.backends.mlx import MlxBackend
    from app.runtime.backends.mlx.diagnostics import diagnose_mlx_log

    backend = MlxBackend()
    tail = "warden: engine exited rc=-9"
    assert backend.diagnose(tail) == diagnose_mlx_log(tail)
    # Unrecognised text -> None, so the caller keeps its generic message.
    assert backend.diagnose("nothing here at all") is None
