"""The tool-error heuristic (analysis spec 8.1, "Retries after tool errors").

An OpenAI-format tool result carries no error flag, so the lens guesses from
its first line. Labelled a heuristic wherever it shows. Returns a boolean only:
no result text is kept (A-3, A-10).
"""

from __future__ import annotations

import re

#: Anchored (I1): the line starts with the word (``Error:``, ``ERROR ...``,
#: ``Traceback (most recent call last):``), or names a non-zero exit. A word
#: inside an ordinary result ("No errors found", ``src/error_handler.py``,
#: ``from .errors import``) is not a failure.
_FAILED = re.compile(
    r"^(?:error|exception|traceback)\b|non-zero exit|\bexit code [1-9]\d*\b", re.IGNORECASE
)
#: An exception class name first (``ValueError: ...``, ``RuntimeException in
#: thread main``): case-sensitive, so ``errors: none`` stays ok.
_TYPED = re.compile(r"^[A-Z]\w*(?:Error|Exception)\b")
#: Only the head of the first line is read (a result can be megabytes on one line).
_FIRST_LINE_CHARS = 512


def looks_failed(text: str) -> bool:
    """Whether the first line of ``text`` reads like a failure."""
    if not isinstance(text, str) or not text:
        return False
    head = text[:_FIRST_LINE_CHARS].lstrip()
    first = head.split("\n", 1)[0]
    return _FAILED.search(first) is not None or _TYPED.match(first) is not None
