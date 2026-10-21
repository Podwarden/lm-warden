"""The two defaulted health/telemetry fields on BackendCapabilities.

Metal design s5 B/C (decision, planner iter 22): ``health_means_ready`` and
``telemetry`` are ENGINE facts, driver-invariant, so they are defaulted
capabilities fields -- not a name check in the supervisor or the stats reader.
The defaults (True / "engine_prometheus") leave vLLM and llama.cpp byte-
identical to today; MLX will answer False / "proxy_observed" when it
registers.
"""

from dataclasses import FrozenInstanceError

import pytest

_BASE = dict(
    name="x",
    display_name="X",
    supports_tensor_parallel=True,
    supports_pipeline_parallel=True,
    max_gpus=None,
    openai_paths=frozenset({"/v1/completions"}),
    health_path="/health",
    metrics_path="/metrics",
    supports_request_priority=True,
    supports_lora=False,
    supports_vision=False,
    supports_version_pin=False,
    needs_local_model_path=False,
    env_prefixes=("X_",),
    env_exact=frozenset(),
    vram_cap_fraction=None,
)


def _caps(**over):
    from app.runtime.backends import BackendCapabilities

    base = dict(_BASE)
    base.update(over)
    return BackendCapabilities(**base)


def test_bare_capabilities_defaults_health_means_ready_true():
    """A backend that predates the field gets the safe answer: its /health
    means ready, as it always has for vLLM and llama.cpp."""
    assert _caps().health_means_ready is True


def test_bare_capabilities_defaults_telemetry_engine_prometheus():
    """A backend that predates the field is assumed to scrape its own
    engine's /metrics, which is how this whole stats path works today."""
    assert _caps().telemetry == "engine_prometheus"


def test_explicit_values_round_trip():
    """The MLX answers -- liveness-only health, proxy-observed telemetry --
    must be expressible, not just the defaults."""
    caps = _caps(health_means_ready=False, telemetry="proxy_observed")
    assert caps.health_means_ready is False
    assert caps.telemetry == "proxy_observed"


def test_vllm_and_llamacpp_inherit_the_defaults_from_the_registry():
    """Both registered backends report the Linux answers through the
    dataclass defaults -- their constructor sites are NOT edited for this."""
    from app.runtime.backends import registry

    for name in ("vllm", "llamacpp"):
        caps = registry.get(name).capabilities
        assert caps.health_means_ready is True, name
        assert caps.telemetry == "engine_prometheus", name


def test_capabilities_stay_frozen_with_the_new_fields():
    """The defaults did not unfreeze the dataclass: advertising a
    capability is still a one-time, immutable answer."""
    caps = _caps(health_means_ready=False, telemetry="proxy_observed")
    with pytest.raises(FrozenInstanceError):
        caps.health_means_ready = True
    with pytest.raises(FrozenInstanceError):
        caps.telemetry = "engine_prometheus"
