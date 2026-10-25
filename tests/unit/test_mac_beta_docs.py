"""Guard: the Mac beta docs tell the truth about the download and the
Keychain prompt.

A beta DMG is ad-hoc signed, so every ad-hoc build is a new signing identity
to the login Keychain: macOS asks for the login password to let LM Warden read
its sign-in secret (the item ``com.podwarden.lmwarden``) on every launch. The
reader must be told that, and told only what was observed -- no promise that
"Always Allow" makes it stop, no claim beyond a Developer ID-signed release.

The download link must be described accurately too:
lmwarden.com/download/mac answers 307 straight to the DMG file, not to the
releases page. A stale sentence that misdescribes the redirect sends a reader
to look for a file the link already gives them.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    """The text of a published file under the repository root."""
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """One Markdown section, from its heading line to the next heading of the
    same or higher level (or end of file), whitespace-collapsed so a wrapped
    line cannot hide a phrase.

    ``heading`` is the exact line, hashes and all ("## Beta builds")."""
    lines = text.splitlines()
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        match = re.match(r"^(#{1,6})\s", lines[i])
        if match and len(match.group(1)) <= level:
            end = i
            break
    return " ".join(" ".join(lines[start:end]).split())


def _plain(text: str) -> str:
    """``text`` with every Markdown link ``[label](url)`` reduced to its label,
    so a forbidden sentence cannot hide behind link brackets ("redirects to
    the [GitHub releases page](...)" reads as "redirects to the GitHub
    releases page")."""
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)


def test_beta_builds_explains_the_keychain_password_prompt() -> None:
    """Leg 1: '## Beta builds' carries the facts a first launch depends on.

    Red today: the section has no Keychain mention at all."""
    section = _section(_read("documents/INSTALL.md"), "## Beta builds")
    for phrase in (
        "Keychain",
        "login password",
        "every launch",
        "Developer ID",
        "com.podwarden.lmwarden",
    ):
        assert phrase in section, (
            f"documents/INSTALL.md '## Beta builds' is missing {phrase!r}: a beta "
            "reader is not told about the Keychain password prompt"
        )


def test_beta_builds_does_not_claim_a_redirect_to_the_releases_page() -> None:
    """Leg 2: the first paragraph of '## Beta builds' describes the link
    accurately: it downloads the DMG itself, and the checksum URL is stated.

    Red today: it still says the link 'redirects to the GitHub releases
    page', and the sha256 link is never named."""
    section = _section(_read("documents/INSTALL.md"), "## Beta builds")
    assert "redirects to the GitHub releases page" not in _plain(section), (
        "documents/INSTALL.md '## Beta builds' still claims /download/mac "
        "redirects to the GitHub releases page; it answers 307 to the DMG"
    )
    assert "download/mac.sha256" in section, (
        "documents/INSTALL.md '## Beta builds' must name the checksum URL "
        "lmwarden.com/download/mac.sha256"
    )


def test_chapter_intro_does_not_claim_source_is_all_that_exists() -> None:
    """Leg 3: the '# On a Mac (preview)' intro must not say a source build
    is all that exists -- the beta DMG is published.

    Red today: the stale sentence is still in the intro."""
    chapter = _section(_read("documents/INSTALL.md"), "# On a Mac (preview)")
    assert "What exists today is a build from source" not in _plain(chapter), (
        "documents/INSTALL.md '# On a Mac (preview)' still says a source build "
        "is all that exists, though the beta DMG is published"
    )


def test_chapter_heading_is_unrenamed_and_unique() -> None:
    """Leg 4 (control): the heading line exists exactly once -- the anchor
    #on-a-mac-preview is used by .github/CONTRIBUTING.md and macos/README.md,
    so a rename or a duplicate breaks them silently. Green today."""
    count = _read("documents/INSTALL.md").splitlines().count("# On a Mac (preview)")
    assert count == 1, (
        f"documents/INSTALL.md must contain the line '# On a Mac (preview)' "
        f"exactly once; found {count}"
    )


def test_readme_mac_beta_explains_the_keychain_password_prompt() -> None:
    """Leg 5: '## Mac app (beta)' in README.md names the launch-time Keychain
    prompt, with 'login password', and links to the INSTALL.md section.

    Red today: the section says 'Keychain' only in the feature bullet and has
    no 'login password' (the #beta-builds link is already there)."""
    section = _section(_read("README.md"), "## Mac app (beta)")
    for phrase in ("Keychain", "login password", "documents/INSTALL.md#beta-builds"):
        assert phrase in section, (
            f"README.md '## Mac app (beta)' is missing {phrase!r}: a beta reader "
            "is not told about the Keychain password prompt at launch"
        )


def test_readme_mac_beta_pins_the_current_beta_from_downloads() -> None:
    """Leg 6 (guard): the README hard-codes the current beta's tag, version and
    DMG name by hand (it renders on GitHub, not from this app), so the names in
    app/landing/downloads.py must still appear in it. Catches the next beta
    bump forgetting the README. Green today."""
    from app.landing import downloads

    section = _section(_read("README.md"), "## Mac app (beta)")
    for phrase in (
        downloads.MAC_BETA_TAG,
        downloads.MAC_BETA_VERSION,
        downloads.MAC_DMG_NAME + ".sha256",
    ):
        assert phrase in section, (
            f"README.md '## Mac app (beta)' no longer carries {phrase!r} from "
            "app/landing/downloads.py -- a beta bump forgot the README"
        )
