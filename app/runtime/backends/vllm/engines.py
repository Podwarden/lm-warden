"""Registry of vLLM installs baked into the image under the engines root.

A model row's ``engine_vllm_version`` names one of these installs when it is
a bare version id with no ``engine_channel``/``engine_image`` (see
``bundled_pin``). The vLLM backend then launches this install's ``vllm``
binary instead of the one on PATH.

The root is read from the ``VW_ENGINES_ROOT`` environment variable at call
time (default ``/opt/lmw-engines``), so tests can point it at a tmp dir.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

VERSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")

_DEFAULT_ROOT = "/opt/lmw-engines"


def _engines_root() -> Path:
    return Path(os.environ.get("VW_ENGINES_ROOT", _DEFAULT_ROOT))


def __getattr__(name: str) -> Path:
    # ``ENGINES_ROOT`` is a read-at-call-time alias for the resolved root,
    # so ``from ... import ENGINES_ROOT`` and ``engines.ENGINES_ROOT`` both
    # reflect the current environment.
    if name == "ENGINES_ROOT":
        return _engines_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_version_id(value: str) -> str:
    """Return ``value`` unchanged when it is a safe version id, else raise."""
    if not isinstance(value, str) or not VERSION_ID_RE.fullmatch(value):
        raise ValueError(f"engine_vllm_version: invalid version id {value!r}")
    return value


def bundled_engine_bin(version_id: str) -> Path | None:
    """Return the executable ``bin/vllm`` for a present engine, else None."""
    validate_version_id(version_id)
    bin_path = _engines_root() / version_id / "bin" / "vllm"
    if bin_path.is_file() and os.access(bin_path, os.X_OK):
        return bin_path
    return None


def list_bundled_engines() -> list[str]:
    """Return the sorted ids of subdirectories holding an executable vllm."""
    root = _engines_root()
    if not root.is_dir():
        return []
    ids = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        bin_path = child / "bin" / "vllm"
        if bin_path.is_file() and os.access(bin_path, os.X_OK):
            ids.append(child.name)
    return sorted(ids)


def bundled_pin(model: Any) -> str | None:
    """The row's bare ``engine_vllm_version`` naming a bundled install.

    A version id is a bundled pin only when the row carries no
    ``engine_channel`` / ``engine_image`` -- those select the docker-driver
    image path (``_resolve_engine_image``), never a bundled install.
    """
    version = getattr(model, "engine_vllm_version", None)
    if not isinstance(version, str) or not version:
        return None
    if getattr(model, "engine_channel", None) or getattr(model, "engine_image", None):
        return None
    return version
