"""Channel + vLLM version → engine container image (the #161-minimal map).

A template's engine axis is ``(channel, vllm_version)``; a driver needs a
concrete image. This module is the single place that mapping lives.

Scope (locked decisions D1/D4): the only path exercised end-to-end is CUDA
via the **upstream** ``vllm/vllm-openai`` tags. We deliberately do NOT
fabricate tag schemes for rocm/cpu/xpu — those channels are *known* (so the
UI can list them and reject early) but resolving one raises rather than
emitting an unverified tag. Patched per-channel images are #161-full, later.
"""

from __future__ import annotations

# Channels the system recognises. Listed so the UI/validation can reject an
# unknown channel up front; membership here does NOT imply resolvable.
KNOWN_CHANNELS: frozenset[str] = frozenset(
    {"cuda-stable", "cuda-edge", "cuda-legacy", "rocm", "xpu", "cpu"}
)

# Channels we can actually turn into an image today, and the upstream image
# family each maps to. All three CUDA channels currently share the upstream
# build; they diverge once #161-full ships patched per-channel images.
_RESOLVABLE: dict[str, str] = {
    "cuda-stable": "vllm/vllm-openai",
    "cuda-edge": "vllm/vllm-openai",
    "cuda-legacy": "vllm/vllm-openai",
}


class UnsupportedChannelError(ValueError):
    """Raised when a channel is unknown, or known but not yet resolvable."""


def resolve_image(channel: str, vllm_version: str, *, image: str | None = None) -> str:
    """Return the engine container image for an engine axis.

    An explicit ``image`` short-circuits resolution (user templates may pin
    a digest). Otherwise ``(channel, vllm_version)`` maps to an upstream tag.
    """
    if image:
        return image
    family = _RESOLVABLE.get(channel)
    if family is None:
        if channel in KNOWN_CHANNELS:
            raise UnsupportedChannelError(
                f"channel {channel!r} is known but not yet resolvable to an "
                "image (only CUDA channels are wired today — see #161)"
            )
        raise UnsupportedChannelError(f"unknown engine channel {channel!r}")
    version = vllm_version[1:] if vllm_version.startswith("v") else vllm_version
    return f"{family}:v{version}"
