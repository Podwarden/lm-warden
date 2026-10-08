import pytest

from app.runtime.backends.vllm.images import (
    KNOWN_CHANNELS,
    UnsupportedChannelError,
    resolve_image,
)


def test_cuda_stable_resolves_to_upstream_openai_tag():
    assert resolve_image("cuda-stable", "0.20.0") == "vllm/vllm-openai:v0.20.0"


def test_cuda_edge_and_legacy_use_same_upstream_family():
    assert resolve_image("cuda-edge", "0.21.0") == "vllm/vllm-openai:v0.21.0"
    assert resolve_image("cuda-legacy", "0.18.0") == "vllm/vllm-openai:v0.18.0"


def test_explicit_image_override_wins_over_channel():
    # An explicit image short-circuits resolution entirely — the channel
    # and version are ignored (used by user templates that pin a digest).
    assert resolve_image("cuda-stable", "0.20.0", image="my.reg/custom:abc") == "my.reg/custom:abc"


def test_unknown_channel_raises():
    with pytest.raises(UnsupportedChannelError):
        resolve_image("totally-made-up", "0.20.0")


def test_non_cuda_channels_are_explicitly_unsupported_for_now():
    # D1/D4: the only exercised path is CUDA via upstream vllm/vllm-openai.
    # We refuse to fabricate an unverified tag scheme for rocm/cpu/xpu —
    # raising is more honest than emitting a wrong tag.
    for ch in ("rocm", "cpu", "xpu"):
        assert ch in KNOWN_CHANNELS
        with pytest.raises(UnsupportedChannelError):
            resolve_image(ch, "0.20.0")


def test_version_may_already_carry_v_prefix():
    # Accept both "0.20.0" and "v0.20.0" so a template author can't break
    # resolution by including the prefix the upstream tag uses.
    assert resolve_image("cuda-stable", "v0.20.0") == "vllm/vllm-openai:v0.20.0"
