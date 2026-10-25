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


def test_mac_beta_size_text_says_110_mb() -> None:
    """The download is about 110 MB (LZMA-compressed since v2026.10.24.2),
    not about 360 MB: every doc that sizes the download must say the new
    size. Red today: the docs still say about 360 MB."""
    files_and_sections = (
        ("README.md", None),
        ("documents/INSTALL.md", "## Beta builds"),
        ("app/landing/llms.txt", None),
        ("app/landing/llms-full.txt", None),
    )
    for rel, heading in files_and_sections:
        text = _read(rel)
        section = _section(text, heading) if heading else text
        assert "about 110 MB" in section, (
            f"{rel}{' ' + heading + ' ' if heading else ''}must say 'about 110 MB': "
            "the Mac beta DMG is about 110 MB, not 360"
        )
        assert "360 MB" not in section, (
            f"{rel}{' ' + heading + ' ' if heading else ''}still says '360 MB': "
            "the Mac beta DMG is about 110 MB"
        )


def test_landing_mac_download_size_says_110_mb() -> None:
    """The landing page and FAQ size the Mac beta download at about 110 MB
    (LZMA-compressed since v2026.10.24.2), in the English source and in all
    ten locales. Red today: the catalogs still say 360.

    The locale unit differs (ru writes 'МБ', the rest 'MB' or 'Mo'), so the
    leg is '110&' (the number next to the non-breaking space) plus 'no 360
    anywhere in the two segments'."""
    import json

    locales = ("fr", "de", "es", "pt-BR", "ru", "uk", "pl", "zh-CN", "ja", "hi")
    targets = [
        ("app/landing/i18n/en.json", "home.p-mac.li-1"),
        ("app/landing/i18n/en/faq.json", "faq.a-2"),
    ]
    for loc in locales:
        targets.append((f"app/landing/i18n/{loc}.json", "home.p-mac.li-1"))
        targets.append((f"app/landing/i18n/{loc}/faq.json", "faq.a-2"))
    for rel, key in targets:
        seg = json.loads(_read(rel))[key]
        text = seg["text"]
        assert "110&" in text, f"{rel} {key!r} does not size the download at 110 MB: {text!r}"
        assert "360" not in text, f"{rel} {key!r} still sizes the download at 360 MB: {text!r}"
    home_html = _read("app/landing/pages/home.html")
    assert "360&nbsp;MB" not in home_html, "app/landing/pages/home.html still says '360&nbsp;MB'"
    faq = _read("app/landing/faq.json")
    assert "360&#8239;MB" not in faq, "app/landing/faq.json still says '360&#8239;MB'"


def test_readme_mac_beta_pins_the_current_beta_from_downloads() -> None:
    """Leg 6 (guard): the README hard-codes the current beta's tag, version,
    DMG name and size text by hand (it renders on GitHub, not from this
    app), so the names in app/landing/downloads.py must still appear in it.
    Catches the next beta bump forgetting the README. Green today."""
    from app.landing import downloads

    section = _section(_read("README.md"), "## Mac app (beta)")
    for phrase in (
        downloads.MAC_BETA_TAG,
        downloads.MAC_BETA_VERSION,
        downloads.MAC_DMG_NAME,
        downloads.MAC_DMG_NAME + ".sha256",
        downloads.MAC_DMG_SIZE_TEXT,
    ):
        assert phrase in section, (
            f"README.md '## Mac app (beta)' no longer carries {phrase!r} from "
            "app/landing/downloads.py -- a beta bump forgot the README"
        )
