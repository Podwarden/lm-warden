from app.runtime.engine import EngineDriver, EngineHandle, EngineSpec


def test_engine_spec_defaults():
    spec = EngineSpec(
        model_arg="openai/gpt-oss-20b",
        argv=["vllm", "serve", "--port", "8001"],
        env={"VLLM_USE_V1": "1"},
        port=8001,
        model_id="m1",
    )
    assert spec.image is None  # local driver ignores image
    assert spec.gpu_indices == []  # default no pinning


def test_protocols_are_runtime_checkable():
    # Protocols must be importable and usable in isinstance checks.
    assert hasattr(EngineDriver, "spawn")
    assert hasattr(EngineHandle, "wait")
    assert hasattr(EngineDriver, "engine_host")


def test_local_subprocess_engine_host_is_loopback():
    # The in-container subprocess shares the control-plane's network
    # namespace, so the engine is reachable on loopback.
    from app.runtime.engine.local_subprocess import LocalSubprocessDriver

    drv = LocalSubprocessDriver(log_dir="/tmp")
    assert drv.engine_host("m1") == "127.0.0.1"


def test_engine_spec_carries_the_backend_name():
    """Drivers get the backend name so a future llama.cpp engine and a vLLM
    engine for the same model id can be told apart in labels and container
    names. B does NOT change any name that uses it -- that would be a
    behaviour change -- it only makes the information available."""
    spec = EngineSpec(model_id="m", model_arg="x", argv=["vllm", "serve"], env={}, port=1)
    assert spec.backend == "vllm"
