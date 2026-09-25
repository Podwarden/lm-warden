"""Guard: the product's user-visible name is "LM Warden" -- never "LLM Warden"
or "vLLM Warden".

The product runs two inference backends (vLLM and llama.cpp), so it was
renamed from vLLM Warden to LLM Warden and then, on 2026-09-23, to
**LM Warden**. The display name follows; identity strings (volumes,
``COMPOSE_PROJECT_NAME``, container names, ``VW_*`` ...) are deliberately
untouched.

What this guard forbids and what it deliberately does NOT forbid:

* FORBIDDEN — either former two-word product name with a whitespace
  separator, in any casing: ``vLLM Warden``, ``vllm warden``, ``VLLM Warden``,
  ``LLM Warden``, ``llm warden``.
* ALLOWED — a line that says ``formerly``: the one-line history note
  ("LM Warden was formerly called LLM Warden ...") that README, OPERATING and
  the Hub page carry so a reader who knew the old name finds the product. It
  is the only sanctioned way to print an old name on a public surface.
* ALLOWED — the noun ``LLM`` on its own ("self-hosted LLM gateway", "LLM
  inference"). Only ``LLM`` immediately followed by ``Warden`` is the name.
* ALLOWED — the bare engine name ``vLLM``. The product serves vLLM, so
  "vLLM version", "the vLLM engine", "live vLLM logs" and ``vllm/vllm-openai``
  are all *correct* prose that this guard must never break. A test that
  forbade the bare word would block true statements about the engine.
* ALLOWED — the identity slug ``vllm-warden`` / ``vllm_warden`` (image names,
  the Hub catalogue slug and ``app_family``, k8s namespaces, ``/data``
  paths, container names, module paths, repo URLs). Those are a separate,
  breaking piece of work; the hyphen/underscore forms are not matched here.

The separator is the entire discrimination rule: whitespace means somebody
wrote the product's *name*; a hyphen or underscore means an *identifier*.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Whitespace separator only -- see the module docstring. `vllm-warden` and
# `vllm_warden` are identity and must NOT match. `v?` covers both former
# names; the leading \b keeps a word that merely ends in "llm" out.
OLD_PRODUCT_NAME = re.compile(r"\bv?llm\s+warden", re.IGNORECASE)

# A line carrying this word is a history note and may name a former name.
HISTORY_NOTE = re.compile(r"\bformerly\b", re.IGNORECASE)

PRODUCT_NAME = "LM Warden"


# The surfaces a human actually reads. DISCOVERY, NOT AN ENUMERATION -- that
# distinction is the whole point of this block, so do not "simplify" it back
# into a list of filenames.
#
# This used to carry `SCANNED_FILES = ("README.md",)`, and that tuple went
# stale the moment the README was split by section: five public documents and
# CONTRIBUTING.md appeared, none of them named here, so the majority of the
# public prose had no guard at all and nobody was told. `MANAGED_DOCUMENTS` in
# scripts/sync-shared-docs.py has the same failure mode for the same reason,
# and the catalogue row is the proof it is not theoretical -- its name, its
# About text and the template's `stack_label` all still said "vLLM Warden"
# because no check enumerated them.
#
# So the surfaces are found, not listed: every `.md` at the repository root,
# every `.md` under `documents/` and `.github/` at any depth, plus the code
# and catalogue trees below. A document added next month is guarded the day it
# lands.
#
# A repo-wide walk is still wrong, hence named trees rather than
# `REPO_ROOT.rglob`: the CI runner's build directory keeps untracked
# node_modules and root-owned caches from sibling jobs, and `docs/` is the
# internal tree, whose plans and postmortems are historical records of work
# done under the old name and are left as written.
SCANNED_TREES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("app", (".py", ".html")),
    ("frontend/src", (".ts", ".tsx", ".html", ".svg", ".css")),
    ("deploy/hub", (".md", ".json")),
    ("publish", (".md", ".txt", ".sh")),
    # The public documentation folder (INSTALL, API, HAZARDS, OPERATING,
    # ARCHITECTURE today). Published verbatim to GitHub, so it is the largest
    # block of prose a stranger reads.
    ("documents", (".md",)),
    # CONTRIBUTING.md, and any community-health file added beside it
    # (SECURITY.md, CODE_OF_CONDUCT.md, issue templates) -- GitHub renders
    # every one of them.
    (".github", (".md",)),
)

# Every Markdown file at the repository root is public prose and is scanned,
# with ONE deliberate exemption. Deny-by-default, the same posture as
# publish/exclude.txt: a new root document is guarded the moment it lands, and
# opting one out means editing this set in the open rather than forgetting to
# opt one in.
#
# changelog.md is the exemption. It is a historical record -- every release
# written before the display rename is quoted in it under the old name -- and
# rewriting history to satisfy a guard would destroy the record it exists to
# keep.
ROOT_MARKDOWN_EXEMPT: frozenset[str] = frozenset({"changelog.md"})


def _iter_surfaces() -> list[Path]:
    out: list[Path] = []
    for rel, suffixes in SCANNED_TREES:
        root = REPO_ROOT / rel
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                out.append(path)
    for path in sorted(REPO_ROOT.glob("*.md")):
        if path.is_file() and path.name not in ROOT_MARKDOWN_EXEMPT:
            out.append(path)
    return out


def test_the_scan_actually_covers_something() -> None:
    """Self-check: a discovery that silently matched nothing would make every
    assertion below vacuously true. Each name below stands for one branch of
    _iter_surfaces, so a tree that stops resolving fails here by name rather
    than by turning the guard into a no-op."""
    surfaces = _iter_surfaces()
    assert len(surfaces) > 50, f"surface scan found only {len(surfaces)} files"
    assert any(p.name == "landing.html" for p in surfaces)
    assert any(p.name == "nav-bar.tsx" for p in surfaces)
    assert any(p.name == "README.md" for p in surfaces)

    rels = {str(p.relative_to(REPO_ROOT)) for p in surfaces}

    # The root Markdown glob.
    assert "README.md" in rels

    # The public documentation folder. Asserted as a set relation, not a
    # hard-coded list: whatever `documents/` holds must be scanned in full,
    # so a sixth document is covered without touching this test.
    on_disk = {
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "documents").rglob("*.md")
        if p.is_file()
    }
    assert on_disk, "documents/ holds no Markdown -- has the public docs folder moved?"
    assert on_disk <= rels, f"documents/ files not scanned: {sorted(on_disk - rels)}"

    # The community-health folder GitHub renders.
    assert ".github/CONTRIBUTING.md" in rels


def test_the_root_markdown_exemption_is_deliberate_and_minimal() -> None:
    """The root glob is deny-by-default; exactly one file opts out.

    A second name appearing here means somebody silenced the guard instead of
    fixing prose, so it is pinned rather than merely documented."""
    assert ROOT_MARKDOWN_EXEMPT == frozenset({"changelog.md"})
    assert (REPO_ROOT / "changelog.md").is_file(), "the exemption names a file that is gone"
    assert not any(p.name == "changelog.md" for p in _iter_surfaces())


def test_no_user_visible_surface_says_a_former_name() -> None:
    """No page title, header, wordmark, nav label, log line, error message,
    catalogue field or README sentence may present the product as
    "vLLM Warden" or "LLM Warden" -- except a "formerly" history note."""
    offenders: list[str] = []
    for path in _iter_surfaces():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - CI hygiene
            continue
        rel = str(path.relative_to(REPO_ROOT))
        pattern = OLD_PRODUCT_NAME
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) and not HISTORY_NOTE.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "the product is called 'LM Warden'; these surfaces still use a "
        "former name:\n  " + "\n  ".join(offenders)
    )


def test_the_guard_still_permits_bare_engine_references() -> None:
    """The regex must not fire on the inference engine. If it ever does, the
    guard would force prose that is factually wrong now that llama.cpp ships
    alongside vLLM."""
    for legitimate in (
        "vLLM 0.26.0",
        "the vLLM engine",
        "vllm/vllm-openai:v0.26.0",
        "live vLLM logs",
        "with vLLM process attribution",
        "vLLM Python package version",
        "Selected MarlinFP8LinearKernel",
        "a self-hosted LLM gateway",
        "LLM inference on your GPUs",
        "LM Warden",
    ):
        assert not OLD_PRODUCT_NAME.search(legitimate), legitimate


def test_the_guard_still_permits_the_identity_slug() -> None:
    """The hyphen/underscore forms are identity and out of scope for the
    display rename."""
    for identity in (
        "vllm-warden",
        "pw-vllm-warden",
        "/data/vllm-warden.db",
        "vllm-warden-engine-abc123",
        "app.vllm_warden",
        "registry.podwarden.com/podwarden/apps/vllm-warden",
        "registry.podwarden.com/podwarden/apps/llm-warden",
        "llm_warden",
        "lmwarden.com",
    ):
        assert not OLD_PRODUCT_NAME.search(identity), identity


@pytest.mark.parametrize(
    "spelling",
    [
        "vLLM Warden",
        "vllm warden",
        "VLLM Warden",
        "vLLM  Warden",
        "LLM Warden",
        "llm warden",
        "Welcome to LLM Warden",
    ],
)
def test_the_guard_catches_every_casing(spelling: str) -> None:
    assert OLD_PRODUCT_NAME.search(spelling), spelling


def test_the_wordmark_and_page_titles_carry_the_new_name() -> None:
    """Positive half: renaming by deletion would pass the guard above."""
    # The landing page's <title> is filled in at import (app/landing/routes.py
    # TITLE), so pin the rendered page rather than the template.
    from app.landing.routes import render_landing

    landing = render_landing("")
    assert f"<title>{PRODUCT_NAME}: " in landing
    assert f'<a class="brand" href="/" aria-label="{PRODUCT_NAME} home">' in landing
    assert f"<span>{PRODUCT_NAME}</span>" in landing

    nav = (REPO_ROOT / "frontend/src/components/nav-bar.tsx").read_text(encoding="utf-8")
    assert f"<span>{PRODUCT_NAME}</span>" in nav

    # The browser tab title and the FastAPI/OpenAPI title used to carry the
    # identity slug (`vllm-warden`) as if it were the product's name. Those
    # two are display surfaces, so they get the real name -- but the slug
    # form is invisible to the regex above, hence these explicit pins.
    layout = (REPO_ROOT / "frontend/src/app/layout.tsx").read_text(encoding="utf-8")
    assert f"title: '{PRODUCT_NAME}'" in layout

    main = (REPO_ROOT / "app/main.py").read_text(encoding="utf-8")
    assert re.search(rf'FastAPI\(\s*title="{re.escape(PRODUCT_NAME)}"', main)

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {PRODUCT_NAME}\n")

    hub = (REPO_ROOT / "deploy/hub/README-hub.md").read_text(encoding="utf-8")
    assert hub.startswith(f"# {PRODUCT_NAME}\n")

    # The first-run pages and the login heading are the name a new operator
    # meets before the nav bar exists.
    login = (REPO_ROOT / "frontend/src/app/login/page.tsx").read_text(encoding="utf-8")
    assert f">{PRODUCT_NAME}</h1>" in login
    welcome = (REPO_ROOT / "frontend/src/app/setup/welcome/page.tsx").read_text(encoding="utf-8")
    assert f"Welcome to {PRODUCT_NAME}" in welcome

    # The installer's banner.
    install = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    assert f"%s%s{PRODUCT_NAME}%s %sinstaller%s" in install


# Outward-facing ADDRESSES (repository and image paths) are the other half of
# the rename: a stale one sends a reader to a redirect today and to a 404 the
# day somebody reuses the old name. The old paths may appear only in the
# explicit transition text that explains the move.
OLD_ADDRESS = re.compile(r"Podwarden/v?llm-warden\b|podwarden/apps/v?llm-warden\b")
ADDRESS_TRANSITION_ALLOWED: frozenset[str] = frozenset(
    {
        # The "Upgrading across the rename" section.
        "documents/OPERATING.md",
    }
)


def test_public_surfaces_use_the_new_repository_and_image_addresses() -> None:
    offenders: list[str] = []
    for path in [*_iter_surfaces(), REPO_ROOT / "install.sh"]:
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ADDRESS_TRANSITION_ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - CI hygiene
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if OLD_ADDRESS.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these public surfaces still point at a pre-rename repository or "
        "image path (use Podwarden/lm-warden, podwarden/apps/lm-warden):\n  "
        + "\n  ".join(offenders)
    )
