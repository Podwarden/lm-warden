"""Regression net for R2 — `make format` was not idempotent on the committed tree.

The pinned ruff (requirements-dev.txt: ruff 0.8.4, line-length 100 in
pyproject.toml) reformatted 378 committed files — a blank line after module
docstrings, and lines re-joined/split. CI only runs `ruff check`, which stayed
green, so the drift was latent: the format lane produced a huge unrelated diff
that could not serve as a pre-push gate.

The fix commits the formatted tree, so the checked-in style is the canonical
one. This test keeps it that way: it fails whenever the committed tree is no
longer the fixed point of the pinned formatter, for whatever reason (a hand
edit, a different ruff version, a config change).

The formatter is the one installed in the *running interpreter*
(``sys.executable -m ruff`` — the requirements-dev.txt pin pip-installed into
it, whose wheel ships ``python -m ruff``), not whatever ``ruff`` is first on
PATH. Why this matters: on ollama.h, ``~/.local/bin/ruff`` 0.14.14 shadowed
the venv's 0.8.4 under ``scripts/farm/remote.sh run``, and the PATH-resolved
ruff 0.14.14 reported 89 files would be reformatted while 0.8.4 reported
819 files already formatted — a green pinned tree red under a newer PATH
ruff, and vice versa.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pinned_ruff_version() -> str:
    """The `ruff==X` version pinned in requirements-dev.txt."""
    for line in (REPO_ROOT / "requirements-dev.txt").read_text().splitlines():
        if line.startswith("ruff=="):
            # Strip a trailing comment (ruff==0.8.4  # why) and whitespace.
            return line.split("#", 1)[0].strip().split("==", 1)[1]
    raise AssertionError("no ruff== pin found in requirements-dev.txt")


def test_committed_tree_is_ruff_format_fixed_point():
    """`ruff format --check app/ tests/` under the running interpreter's pinned
    ruff must report nothing to reformat: `make format` on a clean tree must
    leave the tree unchanged."""
    import importlib.util

    assert importlib.util.find_spec("ruff") is not None, (
        "the pinned ruff from requirements-dev.txt is missing from this "
        f"interpreter ({sys.executable}); install requirements-dev.txt into "
        "the interpreter that runs the unit lane"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "app/", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "the committed tree is not the fixed point of the pinned ruff format; "
        f"`make format` would rewrite files:\n{proc.stdout}\n{proc.stderr}"
    )


def test_the_formatter_is_the_pinned_one():
    """The running interpreter's ruff is the requirements-dev.txt pin: a newer
    ruff first on PATH (ollama.h: ~/.local/bin/ruff 0.14.14 over the venv's
    0.8.4) must not change which formatter runs, because the test invokes
    `sys.executable -m ruff`, never a bare PATH `ruff`."""
    import importlib.util

    assert importlib.util.find_spec("ruff") is not None, (
        "the pinned ruff from requirements-dev.txt is missing from this "
        f"interpreter ({sys.executable}); install requirements-dev.txt into "
        "the interpreter that runs the unit lane"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"`-m ruff --version` failed: {proc.stderr}"
    # `ruff 0.8.4` — the version is everything after "ruff ".
    version = proc.stdout.strip().split(" ", 1)[1]
    assert version == _pinned_ruff_version(), (
        f"the running interpreter's ruff is {version}, but requirements-dev.txt "
        f"pins ruff=={_pinned_ruff_version()}; pip install requirements-dev.txt "
        f"into {sys.executable}"
    )
