"""Guard: no published Markdown links into the internal ``docs/`` tree.

``publish/exclude.txt`` strips ``docs/`` from the GitHub snapshot while keeping
``documents/`` (the deny-by-default invariant: anything in ``docs/`` is
internal, ``documents/`` is public). A link that points a reader into
``docs/`` is therefore a dead link on the published copy and a leak of an
internal path at the same stroke. This test pins the rule on every published
Markdown file so a new reference is caught the day it lands.

What the rule matches and what it deliberately does NOT:

* MATCHES — a bare ``docs/`` path: ``docs/operating.md``,
  ``docs/superpowers/specs/...``. The guard is a negative lookbehind, not a
  plain substring, so the character immediately before ``docs`` decides.
* DOES NOT MATCH — ``documents/`` (the public tree; it never contains the
  literal ``docs/``), any URL such as ``https://example.com/docs/`` (a ``/``
  before ``docs``) and a word that merely ends in ``docs/`` (``mydocs/``).

``README.md`` and ``macos/README.md`` must exist -- they are the two
definite surfaces of this guard; every other published file is skipped with
its name if it is absent (a document that later lands is guarded the moment
it does).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Published Markdown a human reads. The definite surfaces must exist; the
# rest are skipped by name if absent. ``documents/`` and ``.github/`` are
# globbed recursively, not listed, so a new public document or template is
# guarded the moment it lands. ``changelog.md`` is published too (as
# CHANGELOG.md) but is deliberately NOT guarded: it is release history and
# quotes the internal paths each release touched, by design.
_GLOBBED = sorted(
    p.relative_to(REPO_ROOT).as_posix()
    for root in ("documents", ".github")
    for p in (REPO_ROOT / root).rglob("*.md")
)
PUBLISHED_FILES: tuple[str, ...] = tuple(
    dict.fromkeys(
        ["README.md", "macos/README.md", ".github/CONTRIBUTING.md"]
        + _GLOBBED
        + ["SECURITY.md", "CODE_OF_CONDUCT.md", "app/landing/agent-guide.md"]
    )
)

# A bare internal ``docs/`` path. The lookbehind rejects a word character
# (``mydocs/``) and a path character (``/`` -> a URL or ``../docs/``); a
# space, backtick, parenthesis or line start still admits it.
INTERNAL_DOCS = re.compile(r"(?<![\w/.-])docs/")

# The strings the rule must separate: the match leg, then the no-match legs.
MATCHES = (
    "docs/operating.md",
    "see docs/x.md",
    "`docs/superpowers/specs/x.md`",
    "[design](docs/x.md)",
)
NO_MATCH = (
    "documents/INSTALL.md",
    "https://example.com/docs/",
    "../docs-old/x",
    "mydocs/x.md",
    "my-docs/x.md",
    "sub.docs/x.md",
)


def test_internal_docs_rule_separates_match_from_no_match() -> None:
    """The regex pins its own boundary: one leg on each side, per arm.

    If the lookbehind ever drifts, one of these literals flips sides and the
    test fails by name rather than a whole published file going quiet."""
    for line in MATCHES:
        assert INTERNAL_DOCS.search(line), f"should match the internal docs/ rule: {line!r}"
    for line in NO_MATCH:
        assert not INTERNAL_DOCS.search(line), f"should NOT match the internal docs/ rule: {line!r}"


def _offenders(path: Path, rel: str) -> list[str]:
    """Every line of ``path`` that names the internal ``docs/`` tree, as
    ``rel:lineno: line``."""
    text = path.read_text(encoding="utf-8")
    return [
        f"{rel}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(text.splitlines(), start=1)
        if INTERNAL_DOCS.search(line)
    ]


def test_scan_flags_a_mid_line_reference_by_file_and_line(tmp_path: Path) -> None:
    """The file scan finds a reference anywhere on a line (not only at its
    start) and names the line; a clean line and a URL line stay quiet."""
    f = tmp_path / "x.md"
    f.write_text(
        "clean\nsee https://example.com/docs/ ok\nthe `docs/x.md` design\n", encoding="utf-8"
    )
    assert _offenders(f, "x.md") == ["x.md:3: the `docs/x.md` design"]


@pytest.mark.parametrize("rel", PUBLISHED_FILES)
def test_no_reference_to_internal_docs(rel: str) -> None:
    """A published file must not point a reader into the internal ``docs/``
    tree. The definite surfaces must exist; the rest skip by name."""
    path = REPO_ROOT / rel
    if rel in ("README.md", "macos/README.md"):
        assert path.is_file(), f"definite published surface is missing: {rel}"
    elif not path.is_file():
        pytest.skip(f"not present: {rel}")

    offenders = _offenders(path, rel)
    assert not offenders, (
        "published file references the internal docs/ tree (stripped from the "
        "public snapshot by publish/exclude.txt):\n  " + "\n  ".join(offenders)
    )
