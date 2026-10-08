from app.models.stack_classifier import classify


def test_cuda_arch():
    r = classify(
        "RuntimeError: CUDA error: no kernel image is available for execution on the device (sm_86)"
    )
    assert r.category == "cuda_arch_unsupported"
    assert r.suggestion  # non-empty human hint


def test_oom():
    r = classify("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert r.category == "oom"


def test_quant():
    r = classify("ValueError: Quantization method awq is not supported for the current GPU")
    assert r.category == "quant_unsupported"


def test_version_mismatch():
    r = classify("ImportError: vllm 0.20.0 requires torch==2.5.1 but found 2.4.0")
    assert r.category == "version_mismatch"


def test_unknown():
    r = classify("some totally unexpected traceback")
    assert r.category == "unknown"
    assert r.suggestion
