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
        + ["SECURITY.md", "CODE_OF_CONDUCT.md"]
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


# ---------------------------------------------------------------------------
# Pages only the vendor build serves.
#
# The product website (home, feature pages, FAQ, /coding-agents/..., the Mac
# download redirect, llms.txt, the sitemap) is served by the vendor build of
# lmwarden.com, not by an LM Warden instance: the Community build answers 404
# on every one of these paths (spec 2026-11-04). A published document may link
# to them on https://lmwarden.com/..., never as a path on the reader's own
# instance (a bare /download/mac, or http://localhost:8080/llms.txt).
# ---------------------------------------------------------------------------

#: First path segment of every page the vendor website serves.
WEBSITE_ONLY_SEGMENTS = (
    "coding-agents",
    "agent-tuning",
    "gpu-monitoring",
    "cache-aware-routing",
    "api-keys",
    "features",
    "faq",
    "privacy",
    "download",
    "llms.txt",
    "llms-full.txt",
    "sitemap.xml",
    "_landing",
)
_WEBSITE_ONLY = "|".join(re.escape(s) for s in WEBSITE_ONLY_SEGMENTS)

#: A link target (Markdown ``](...)``, HTML ``href=``/``src=``) or a bare
#: ``http(s)://`` URL whose path starts with a website-only segment, on any
#: host but lmwarden.com, or with no host at all (instance-relative).
WEBSITE_LINK = re.compile(
    r"(?:\]\(|(?:href|src)=[\"']|(?<![\w/])(?=https?://))"
    r"(?:https?://(?!(?:www\.)?lmwarden\.com[/:\"')\s])[^/\s\"')]+)?"
    rf"/(?:(?:{_WEBSITE_ONLY})(?=[/\"')#?\s]|$))",
    re.IGNORECASE,
)

WEBSITE_MATCHES = (
    "[Mac](/download/mac)",
    "[agents](/coding-agents/aider)",
    '<a href="/llms.txt">',
    "see http://localhost:8080/llms.txt for the index",
    "[faq](https://my-instance.example/faq)",
    "[x](/sitemap.xml)",
)
WEBSITE_NO_MATCH = (
    "[Mac](https://lmwarden.com/download/mac)",
    "[agents](https://lmwarden.com/coding-agents)",
    "https://www.lmwarden.com/llms.txt",
    "[install](documents/INSTALL.md#download)",
    "[repo](https://github.com/Podwarden/lm-warden/releases/download/v1/x.dmg)",
    "[api](/documents/API.md)",
    "[anchor](#claude-code-and-coding-agents-on-your-gpus)",
    "GET /api/keys returns",
    "[ui](/ui/api-keys)",
)

#: README.md and the public documents, the files a reader follows links from.
WEBSITE_GUARDED: tuple[str, ...] = ("README.md",) + tuple(
    sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "documents").glob("*.md"))
)


def test_website_rule_separates_match_from_no_match() -> None:
    for line in WEBSITE_MATCHES:
        assert WEBSITE_LINK.search(line), f"should match the website-only rule: {line!r}"
    for line in WEBSITE_NO_MATCH:
        assert not WEBSITE_LINK.search(line), f"should NOT match the website-only rule: {line!r}"


@pytest.mark.parametrize("rel", WEBSITE_GUARDED)
def test_no_link_to_a_page_only_the_vendor_build_serves(rel: str) -> None:
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    offenders = [
        f"{rel}:{n}: {line.strip()}"
        for n, line in enumerate(text.splitlines(), start=1)
        if WEBSITE_LINK.search(line)
    ]
    assert not offenders, (
        "published file links to a website page as a path on the reader's own "
        "instance; the Community build does not serve it. Link to "
        "https://lmwarden.com/... instead:\n  " + "\n  ".join(offenders)
    )


# Relative image and link targets in the README must resolve inside the public
# tree. The README's screenshots used to live under the website's assets
# (app/landing/assets/), which left core with the site split and is not in the
# GitHub snapshot: a broken image on the repo's front page.
_RELATIVE_TARGET = re.compile(r"(?:\]\(|(?:href|src)=\")(?!https?://|mailto:|#|/)([^)\"\s#?]+)")
_NOT_PUBLIC = ("site/", "ee/", "docs/", "app/landing/", "frontend/src/ee/")


# publish/stage.sh renames these on the way to GitHub. Compare against the
# repo spelling: a case-insensitive macOS checkout would hide the mismatch,
# Linux CI does not.
_PUBLISH_RENAMES = {"CHANGELOG.md": "changelog.md"}


def test_readme_relative_targets_exist_in_the_public_tree() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    bad = []
    for target in _RELATIVE_TARGET.findall(text):
        target = target.removeprefix("./")
        # Either spelling: the repo has the source name, a published stage the target.
        exists = (REPO_ROOT / target).exists() or (
            REPO_ROOT / _PUBLISH_RENAMES.get(target, target)
        ).exists()
        if target.startswith(_NOT_PUBLIC) or not exists:
            bad.append(target)
    assert not bad, "README.md points at files missing from the public tree:\n  " + "\n  ".join(
        sorted(set(bad))
    )
