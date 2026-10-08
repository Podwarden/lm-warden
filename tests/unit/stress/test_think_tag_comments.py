"""R24: R22's comment rewording lost the literal think tags.

The implementer output channel strips angle-bracket think tags, so R22's
edits landed mangled: the double-backtick pairs in app/proxy/routes.py kept
their padding space but lost the tag's brackets, and the comment in
app/stress/stream.py lost the whole tag pair (leaving a double space). The
expected tag strings below are built from parts so this test file itself
cannot be mangled the same way.
"""

from __future__ import annotations

from pathlib import Path

OPEN_TAG = "<" + "think" + ">"
CLOSE_TAG = "<" + "/" + "think" + ">"

# The mangled forms R22 left behind.
MANGLED_RST = "``" + " think" + "``"  # double-backtick pair, tag stripped
MANGLED_PLAIN = "literal" + "  tags"  # whole tag pair stripped, double space

REPO = Path(__file__).resolve().parents[3]
ROUTES = REPO / "app" / "proxy" / "routes.py"
STREAM = REPO / "app" / "stress" / "stream.py"


def _docstring(source: str, func: str) -> str:
    start = source.index("def " + func)
    a = source.index('"""', start)
    b = source.index('"""', a + 3)
    return source[a : b + 3]


def _line(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"no line containing {needle!r}")


def test_no_mangled_tag_forms_in_either_file() -> None:
    for path in (ROUTES, STREAM):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            assert MANGLED_RST not in line, f"{path.name}:{n} still mangled: {line!r}"
            assert MANGLED_PLAIN not in line, f"{path.name}:{n} still mangled: {line!r}"


def test_godmode_docstring_names_the_inline_tag() -> None:
    doc = _docstring(ROUTES.read_text(), "_parse_sse_godmode")
    line = _line(doc, "NOT inline")
    assert OPEN_TAG in line, f"inline tag missing: {line!r}"


def test_feed_detector_docstring_names_both_tags() -> None:
    doc = _docstring(ROUTES.read_text(), "_feed_detector")
    line = _line(doc, "tags and stream the reasoning chain")
    assert OPEN_TAG in line, f"open tag missing: {line!r}"
    assert CLOSE_TAG in line, f"close tag missing: {line!r}"


def test_stress_comment_names_both_tags() -> None:
    line = _line(STREAM.read_text(), "strips the literal")
    assert OPEN_TAG in line, f"open tag missing: {line!r}"
    assert CLOSE_TAG in line, f"close tag missing: {line!r}"
