"""name -> Backend.

The ONE place a backend is registered. Adding a backend is adding a line to
``_BACKENDS``; adding a driver is adding a class under ``app/runtime/engine/``.
Neither multiplies the other, which is the whole point of decision D1.

Name resolution ONLY. Capability questions go to the backend --
``registry.get(name).capabilities`` and ``.version_pin_available(driver)``.
"""

from __future__ import annotations

from app.runtime.backends import Backend
from app.runtime.backends.llamacpp import LlamaCppBackend
from app.runtime.backends.vllm import VllmBackend

DEFAULT_BACKEND = "vllm"

# Registered in the order they arrived; PRESENTED sorted -- available() sorts,
# deliberately, so registration order and presentation order stay independent.
_BACKENDS: dict[str, Backend] = {
    "vllm": VllmBackend(),
    "llamacpp": LlamaCppBackend(),
}


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
