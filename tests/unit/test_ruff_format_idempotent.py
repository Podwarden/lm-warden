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
"""

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_committed_tree_is_ruff_format_fixed_point():
    """`ruff format --check app/ tests/` (the pinned ruff from the deps image)
    must report nothing to reformat: `make format` on a clean tree must leave
    the tree unchanged."""
    ruff = shutil.which("ruff")
    assert ruff is not None, (
        "ruff is not on PATH; the deps image that runs the unit lane must "
        "provide the pinned ruff from requirements-dev.txt"
    )
    proc = subprocess.run(
        [ruff, "format", "--check", "app/", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "the committed tree is not the fixed point of the pinned ruff format; "
        f"`make format` would rewrite files:\n{proc.stdout}\n{proc.stderr}"
    )
