"""The baked llama.cpp build id, read from the environment the Dockerfile sets.

Decision D2: ``llama-server`` is compiled into the warden image from a pinned
upstream tag, so its version is a property of the IMAGE, not of a model row.
The Dockerfile stamps that tag into ``VW_LLAMACPP_BUILD``; this module reads it.

No subprocess, no import-time I/O, no parsing of ``--version``. Shelling out to
the binary to ask its version would mean the control plane could not answer
``GET /api/system/backends`` without an exec -- on a dev box where the binary is
absent that would be a 500 for a purely informational field. An env var read is
cheap enough to do per call, so a rebuilt image is picked up without a restart
of anything that caches this.
"""

from __future__ import annotations

import os

# What the Dockerfile stamps. A bNNNNN upstream build tag, e.g. "b10731".
BUILD_ENV_VAR = "VW_LLAMACPP_BUILD"


def baked_build() -> str | None:
    """The upstream build tag compiled into this image, or None if unstamped.

    ``None`` is the honest answer on a dev checkout and in the test suite, where
    no binary was built. Callers render it as "unknown", never as a guess.
    """
    v = os.environ.get(BUILD_ENV_VAR, "").strip()
    return v or None


def version_string() -> str | None:
    """A display string for the engine version, or None.

    Shaped to read the same way vLLM's does on the same route: a bare version
    with no vendor prefix, because the vendor is already the backend's name.
    """
    build = baked_build()
    return build if build else None
