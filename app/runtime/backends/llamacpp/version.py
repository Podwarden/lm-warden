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

THE MAC BUNDLE has no Dockerfile and its launcher (the Swift shell) knows
nothing about llama.cpp, so on the apple profile -- and only there -- an unset
``VW_LLAMACPP_BUILD`` falls back to a one-line stamp file that
scripts/macos/bundle-llamacpp.sh writes NEXT TO the llama-server it places:
``Contents/Resources/python/bin/llama-server.build``, the embedded
interpreter's own dir (the same dirname(sys.executable) the apple engine PATH
resolves the binary through). The script's LLAMACPP_TAG stays the one pin; a
unit test holds it to the Dockerfile's ARG LLAMACPP_BUILD. Still no subprocess:
a missing, unreadable or malformed stamp (not one short ASCII token in
its first 64 bytes) reads as None, never an error. The linux profile
never reads the file, so the image's behaviour is unchanged.
"""

from __future__ import annotations

import os
import re
import sys

# What the Dockerfile stamps. A bNNNNN upstream build tag, e.g. "b11719".
BUILD_ENV_VAR = "VW_LLAMACPP_BUILD"

# What scripts/macos/bundle-llamacpp.sh writes beside the bundled binary.
STAMP_NAME = "llama-server.build"


# A stamp is one short ASCII token (the script writes "bNNNNN\n"). Read at most
# this many bytes, so a stale or garbage file can never be slurped whole or
# surface as an unbounded string on the API; anything else reads as None.
_STAMP_MAX_BYTES = 64
_STAMP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,31}")


def _stamp_build(bin_dir: str | None) -> str | None:
    d = (
        bin_dir
        if bin_dir is not None
        else (os.path.dirname(sys.executable) if sys.executable else "")
    )
    if not d:
        return None
    path = os.path.join(d, STAMP_NAME)
    # A regular file only: a FIFO or device would block or stream forever.
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read(_STAMP_MAX_BYTES)
    except OSError:
        return None
    lines = raw.splitlines()
    v = lines[0].strip() if lines else b""
    try:
        text = v.decode("ascii")
    except UnicodeDecodeError:
        return None
    return text if _STAMP_RE.fullmatch(text) else None


def baked_build(*, profile: str = "linux", bin_dir: str | None = None) -> str | None:
    """The upstream build tag compiled into this image, or None if unstamped.

    ``None`` is the honest answer on a dev checkout and in the test suite, where
    no binary was built. Callers render it as "unknown", never as a guess.

    The env var wins on every profile. On ``profile="apple"`` an unset one falls
    back to the bundle's stamp file in ``bin_dir`` (default: the interpreter's
    own dir); see the module docstring.
    """
    v = os.environ.get(BUILD_ENV_VAR, "").strip()
    if v:
        return v
    if profile == "apple":
        return _stamp_build(bin_dir)
    return None


def version_string(*, profile: str = "linux", bin_dir: str | None = None) -> str | None:
    """A display string for the engine version, or None.

    Shaped to read the same way vLLM's does on the same route: a bare version
    with no vendor prefix, because the vendor is already the backend's name.
    """
    build = baked_build(profile=profile, bin_dir=bin_dir)
    return build if build else None
