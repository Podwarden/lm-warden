import pytest

from app.runtime.backends import registry
from app.runtime.backends.vllm import VllmBackend


def test_default_is_vllm():
    assert registry.DEFAULT_BACKEND == "vllm"


def test_get_returns_the_vllm_backend():
    assert isinstance(registry.get("vllm"), VllmBackend)


def test_registry_does_not_answer_capability_questions():
    """Operator ruling, 2026-09-01: capabilities -- supports_version_pin
    included -- are the backend's to answer, via `.capabilities` and
    `.version_pin_available(driver)`. The registry resolves names, nothing
    else."""
    assert not hasattr(registry, "supports_version_pin")


def test_none_and_empty_decode_to_the_default():
    """D6: models.backend is nullable and decodes to 'vllm'. Every caller that
    reads a legacy row goes through here, so the default lives in ONE place."""
    assert registry.get(None).capabilities.name == "vllm"
    assert registry.get("").capabilities.name == "vllm"


def test_unknown_backend_raises_rather_than_falling_back():
    """An unknown name is a bug or a downgrade, not a reason to silently
    launch vLLM against a row that asked for something else."""
    with pytest.raises(registry.UnknownBackendError) as e:
        # "llamacpp" was the unknown name in sub-project B; C registers it.
        registry.get("sglang")
    assert "sglang" in str(e.value)
    assert "vllm" in str(e.value)  # the message lists what IS available


def test_available_lists_both_backends():
    """B asserted ("vllm",) alone; sub-project C registers llamacpp SECOND and
    relies on available() sorting to present it FIRST."""
    assert registry.available() == ("llamacpp", "vllm")


def test_available_is_sorted():
    """Sub-project C asserts `available() == ("llamacpp", "vllm")` and that
    /api/system/backends lists them in that order, while registering
    llamacpp SECOND. Sorting here is what makes both true; insertion order
    would break them. Pinned so the ordering is a contract, not an accident
    of a single-entry dict."""
    assert list(registry.available()) == sorted(registry.available())


def test_is_known():
    assert registry.is_known("vllm")
    assert registry.is_known("llamacpp")
    assert registry.is_known(None)
    assert not registry.is_known("sglang")


def test_is_known_does_not_construct_capabilities():
    """is_known answers a name question and must not need a driver."""
    assert registry.is_known("vllm")


def test_llamacpp_is_registered():
    assert registry.get("llamacpp").capabilities.name == "llamacpp"


def test_default_is_still_vllm():
    """D6. NULL means vLLM, and adding a backend must not change that."""
    assert registry.DEFAULT_BACKEND == "vllm"
    assert registry.get(None).capabilities.name == "vllm"


def test_the_registry_has_no_version_pin_helper():
    """B retired registry.supports_version_pin at 837d202 when it adopted E's
    two-field split. The registry resolves NAMES; capability questions belong to
    the backend. Adding llamacpp must not resurrect the helper."""
    assert not hasattr(registry, "supports_version_pin")


def test_both_backends_declare_the_engine_fact_as_pinnable():
    """supports_version_pin is the driver-invariant ENGINE fact: both engines
    publish per-build tagged upstream images, so both are pinnable in principle.
    It is NOT the field a UI gates a control on -- see the next test."""
    assert registry.get("vllm").capabilities.supports_version_pin is True
    assert registry.get("llamacpp").capabilities.supports_version_pin is True


def test_the_deployment_fact_differs_between_the_two_backends():
    """And this is why the split exists. Under a driver that CAN swap images,
    vLLM's pin becomes available and llama.cpp's does not -- because there is no
    llama.cpp image resolver in this build, not because of anything about the
    driver. A single boolean could not have said that."""

    class _Docker:
        supports_engine_image = True

    d = _Docker()
    assert registry.get("vllm").version_pin_available(d) is True
    assert registry.get("llamacpp").version_pin_available(d) is False
