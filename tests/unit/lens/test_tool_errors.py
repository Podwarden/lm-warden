"""The tool-error heuristic on OpenAI-format results (analysis spec 8.1)."""

import pytest

from app.lens.tool_errors import looks_failed


@pytest.mark.parametrize(
    "text,failed",
    [
        ("Error: no such file", True),
        ("Traceback (most recent call last):", True),
        ("exit code 2", True),
        ("all good", False),
        ("ERROR something", True),
        ("RuntimeException in thread main", True),
        ("Command failed with non-zero exit status", True),
        ("Process exited: exit code 0", False),
        ("exit code 10", True),
        ("fine\nError: only on the second line", False),
        ("   Error: indented first line", True),
        ("", False),
        # I1: a word inside an ordinary result is not a failure.
        ("No errors found", False),
        ("0 errors", False),
        ("from .errors import ValidationError", False),
        ("src/error_handler.py", False),
        ("errors: none", False),
        ("Command failed with non-zero exit status 1", True),
        ("exit code 1", True),
        ("ValueError: bad input", True),
        ("Exception: boom", True),
        ("error[E0308]: mismatched types", True),
    ],
)
def test_looks_failed(text, failed):
    assert looks_failed(text) is failed


def test_only_the_first_line_is_read():
    assert looks_failed("ok" + "x" * 10_000 + "\nerror") is False


def test_garbage():
    assert looks_failed(None) is False  # type: ignore[arg-type]
    assert looks_failed(5) is False  # type: ignore[arg-type]
