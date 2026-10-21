"""name -> Backend.

The ONE place a backend is registered. Adding a backend is adding a line to
``_BACKENDS``; adding a driver is adding a class under ``app/runtime/engine/``.
Neither multiplies the other, which is the whole point of decision D1.

Name resolution ONLY. Capability questions go to the backend --
``registry.get(name).capabilities`` and ``.version_pin_available(driver)``.
"""

from __future__ import annotations

from app.platform_profile import PROFILE, PROFILES
from app.runtime.backends import Backend
from app.runtime.backends.llamacpp import LlamaCppBackend
from app.runtime.backends.mlx import MlxBackend
from app.runtime.backends.vllm import VllmBackend


def _check_profile(profile: str) -> None:
    """Both per-profile answers reject a profile the deployment cannot be,
    the way get() rejects a backend name: loud, at the seam, naming the
    accepted values. A typo in LMW_PLATFORM_PROFILE must fail here and at
    import -- never silently serve the wrong backend set."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; accepted values: {', '.join(PROFILES)}")


def backends_for(profile: str) -> dict[str, Backend]:
    """The backend set a profile runs. PURE in its argument, and it answers
    FRESH: every call builds its own dict, so mutating the result can never
    leak into the live registry. The profile, not the process's OS identity,
    is the seam -- resolving by the running platform would make the unit
    suite see one registry on CI and another on a developer's Mac (OVERSEER,
    planner iter 22)."""
    _check_profile(profile)
    # Registered in the order they arrived; PRESENTED sorted -- available()
    # sorts, deliberately, so registration order and presentation order stay
    # independent.
    if profile == "apple":
        return {
            # profile="apple" reaches the backend's env builder (no CUDA keys)
            # and its OOM diagnosis (unified-memory message); the linux branch
            # below keeps the unprofiled backend, byte-identical to today.
            "llamacpp": LlamaCppBackend(profile="apple"),
            "mlx": MlxBackend(),
        }
    return {
        "vllm": VllmBackend(),
        "llamacpp": LlamaCppBackend(),
    }


def default_for(profile: str) -> str:
    """The backend a NULL/'' models.backend row decodes to, per profile.
    ``linux`` is vLLM exactly as D6 has always said; ``apple`` is MLX, which
    registers in this profile and is its default. A default that is not in
    backends_for(profile) would make get(None) raise on every legacy row of
    that profile, which the 0050 migration and this seam exist to prevent."""
    _check_profile(profile)
    return "mlx" if profile == "apple" else "vllm"


_BACKENDS: dict[str, Backend] = backends_for(PROFILE)
DEFAULT_BACKEND = default_for(PROFILE)


class UnknownBackendError(ValueError):
    """Raised when a model row names a backend this build does not have.

    Deliberately NOT a silent fallback to the default. A row asking for a
    backend we cannot serve is either a bug or a downgrade past the release
    that added it; launching vLLM instead would serve the wrong program under
    the operator's chosen name.
    """


def available() -> tuple[str, ...]:
    # SORTED, not insertion-ordered, and that is a contract rather than a
    # detail: sub-project C registers "llamacpp" after "vllm" but asserts
    # available() == ("llamacpp", "vllm") and that /api/system/backends lists
    # them in that order. Sorting here is what makes a registration order and
    # a presentation order independent of each other.
    return tuple(sorted(_BACKENDS))


def is_known(name: str | None) -> bool:
    return not name or name in _BACKENDS


def get(name: str | None) -> Backend:
    """Resolve a backend name to its implementation.

    ``None`` and ``""`` resolve to :data:`DEFAULT_BACKEND` -- decision D6:
    ``models.backend`` is nullable, existing rows are not backfilled, and NULL
    means vLLM. Putting that default here rather than at each read site is what
    keeps it to one line when a second backend lands.
    """
    key = name or DEFAULT_BACKEND
    try:
        return _BACKENDS[key]
    except KeyError:
        raise UnknownBackendError(
            f"unknown backend {key!r}; this build has {', '.join(available())}"
        ) from None


# NOTE: there is deliberately no supports_version_pin() here. Capability
# questions -- that one included -- are answered by the backend itself:
# Backend.capabilities.supports_version_pin (the engine fact) and
# Backend.version_pin_available(driver) (the deployment fact). Operator ruling,
# 2026-09-01: "can you pick a version" is a fact about the engine. Adding a
# helper here would be a second way to ask one question.
