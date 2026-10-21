"""The `macos` pytest marker: registered, and deselected everywhere except the
macos CI job and macos-lane workers.

The marker is the Python-side half of Task 6 (the Swift package skeleton is
the other). It lets a test declare "I genuinely need Apple Silicon" and be
skipped on every other runner by default, without each call site repeating
`-m "not macos"`. The default `addopts` in `pyproject.toml` does that
globally, and the Makefile targets that pass their own `-m` (which replaces
`addopts`' `-m`) carry `not macos` too.

This test module is itself not marked `macos` — it runs everywhere, and
verifies the marker plumbing rather than needing a Mac.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# The `pytester` fixture (pytest's own plugin for testing pytest plugins and
# CLI behaviour in an isolated tmp dir) is shipped disabled by default; it
# must be explicitly enabled. We enable it here so this test module is
# self-contained and does not require a conftest change.
pytest_plugins = ("pytester",)

#: The marker line as it must appear in `pyproject.toml`'s markers list.
MACOS_MARKER_DESCRIPTION = (
    "macos: needs Apple Silicon; selected only by the macos CI job and " "macos-lane workers"
)

#: Repo-relative paths whose `-m` expressions must carry `not macos` because
#: they pass their own `-m` on the command line, which replaces `addopts`'s
#: `-m "not live and not macos"`. Without the `not macos` in each, those
#: invocations would run macos tests on Linux.
_MAKEFILE_M_EXPRESSIONS = {
    "Makefile:test-unit": "not integration and not macos",
    "Makefile:test-integration": "integration and not macos",
}


def test_macos_marker_registered() -> None:
    """`pytest --markers` lists the `macos` marker with its description."""
    markers = subprocess_run_pytest_markers()
    assert MACOS_MARKER_DESCRIPTION in markers, (
        "the `macos` marker is not registered (or its description differs). "
        "Expected to find:\n  " + MACOS_MARKER_DESCRIPTION + "\n"
        "in `pytest --markers` output."
    )


def test_macos_marker_deselects_by_default(pytester: pytest.Pytester) -> None:
    """A `@pytest.mark.macos` test in a tmp dir is deselected by the default
    `addopts` (`-m "not live and not macos"`).

    We run pytest in an isolated copy with the repo's `pyproject.toml` so the
    `addopts` and the registered marker both apply, then assert the macos
    test is reported as deselected and never executed.
    """
    pytester.makepyfile(
        test_macos_only="""
            import pytest

            @pytest.mark.macos
            def test_needs_apple_silicon():
                # Must never run on a non-macos runner. If it does, the marker
                # is not deselecting; fail loudly.
                assert False, "macos-marked test ran on a non-macos runner"
            """
    )
    # Carry the repo's pyproject.toml so addopts (-m "not live and not macos")
    # and the registered marker apply in the spawned pytest.
    repo_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pytester.makefile(".toml", pyproject=repo_pyproject.read_text())

    result = pytester.runpytest("-q", "--no-header")
    # The macos test must be deselected, not run. pytest reports a deselected
    # item as "1 deselected" and 0 passed/failed.
    result.assert_outcomes(deselected=1, passed=0, failed=0)


def test_makefile_m_expressions_exclude_macos() -> None:
    """The Makefile targets that pass their own `-m` carry `not macos`.

    A command-line `-m` replaces `addopts`' `-m "not live and not macos"`, so
    every Makefile invocation that passes `-m` must repeat `not macos` or it
    would run macos tests on Linux. This guards against that regression.
    """
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text()
    for label, expected in _MAKEFILE_M_EXPRESSIONS.items():
        target = label.split(":", 1)[1]
        # Find the target's recipe line. We look for the `-m '...'` (or
        # `-m "..."`) expression within the target's recipe block.
        assert _target_m_expression_contains(text, target, expected), (
            f"{label}: the `-m` expression does not contain `{expected}`. "
            f"Without it, `make {target}` would run macos tests on Linux "
            "(a command-line -m replaces addopts' -m)."
        )


# --- helpers -----------------------------------------------------------------


def _target_m_expression_contains(makefile_text: str, target: str, expected_expr: str) -> bool:
    """True when the Makefile `target`'s recipe carries `-m '<expected_expr>'`
    (single or double quoted). The match is on the literal expression, with
    whitespace collapsed, so reformatting the `-m` string breaks the test
    loudly rather than silently.
    """
    # Locate the target line and its following recipe (indented lines until a
    # blank line or a new non-indented line).
    lines = makefile_text.splitlines()
    pattern = re.compile(rf"^{re.escape(target)}:\s*$")
    start = None
    for i, line in enumerate(lines):
        if pattern.match(line):
            start = i + 1
            break
    if start is None:
        return False
    recipe_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith(("\t", " ")):
            recipe_lines.append(line)
        elif line.strip() == "":
            continue
        else:
            break
    recipe = "\n".join(recipe_lines)
    # Collapse whitespace in both the recipe and the expected expression so
    # minor reformatting (single vs double quotes, extra spaces) is tolerated
    # only when the tokens still match in order.
    collapsed = re.sub(r"\s+", " ", recipe)
    expected_collapsed = re.sub(r"\s+", " ", expected_expr)
    # Accept either single- or double-quoted -m expression.
    return f"-m '{expected_collapsed}'" in collapsed or f'-m "{expected_collapsed}"' in collapsed


def subprocess_run_pytest_markers() -> str:
    """Run `pytest --markers` in the repo root and return its stdout.

    Uses ``sys.executable -m pytest`` so the same interpreter that runs this
    test runs the spawned ``pytest`` — portable across the main checkout and
    the worktree (which does not have its own ``.venv``).
    """
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
