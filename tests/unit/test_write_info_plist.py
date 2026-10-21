"""Unit tests for ``scripts/macos/write-info-plist.sh``.

Not marked ``macos``: the script only needs ``bash`` plus ``mkdir``/``printf``
and the plists are parsed with the stdlib ``plistlib``, so it also runs in
linux CI. Every test runs the script against a ``tmp_path`` and parses the
result.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "write-info-plist.sh"

USAGE = "usage: write-info-plist.sh <plist_path> <version> <executable> [bundle_id]"


def _write(plist: Path, *args: str) -> subprocess.CompletedProcess:
    # The Sparkle vars are popped here too: a signing job that exports one of
    # them must not turn the older legs red (half-set is exit 2) or add keys.
    return subprocess.run(
        ["bash", str(SCRIPT), str(plist), *args],
        capture_output=True,
        text=True,
        env=_sparkle_env(),
    )


def _load(plist: Path) -> dict:
    with plist.open("rb") as fh:
        return plistlib.load(fh)


def _sparkle_env(**extra: str) -> dict[str, str]:
    # A copy of the environment with both Sparkle vars popped first, then only
    # the leg's own vars set, so a var inherited from the CI env cannot leak
    # into a leg that needs it unset.
    env = {**os.environ}
    env.pop("LMW_SPARKLE_FEED_URL", None)
    env.pop("LMW_SPARKLE_PUBLIC_ED_KEY", None)
    env.update(extra)
    return env


def _write_sparkle(plist: Path, *args: str, **sparkle: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(plist), *args],
        capture_output=True,
        text=True,
        env=_sparkle_env(**sparkle),
    )


def test_lsuielement_true(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden")
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["LSUIElement"] is True


def test_local_network_usage_description_exact(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden")
    assert proc.returncode == 0, proc.stderr
    expected = (
        "LM Warden uses your local network only when you turn on Share on "
        "Local Network, so other devices on your network can reach its "
        "model server."
    )
    assert _load(plist)["NSLocalNetworkUsageDescription"] == expected


def test_default_bundle_id_is_the_app_id(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden")
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["CFBundleIdentifier"] == "com.podwarden.lmwarden"


def test_custom_bundle_id(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden", "com.podwarden.lmwarden.smoke")
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["CFBundleIdentifier"] == "com.podwarden.lmwarden.smoke"


def test_existing_keys_unchanged(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden")
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert info["CFBundleExecutable"] == "LMWarden"
    assert info["CFBundlePackageType"] == "APPL"
    # "0.0.1" is not a release tag: CFBundleVersion is the dev value 0.
    assert info["CFBundleVersion"] == "0"
    assert info["CFBundleShortVersionString"] == "0.0.1"
    assert str(info["LSMinimumSystemVersion"]) == "14.0"
    assert info["NSHighResolutionCapable"] is True


@pytest.mark.parametrize(
    "bundle_id",
    ["", "a b", "a<b", "a&b", "a\nb", ".lead"],
)
def test_invalid_bundle_id_exits_2_and_writes_nothing(tmp_path: Path, bundle_id: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden", bundle_id)
    assert proc.returncode == 2
    assert "write-info-plist.sh: invalid bundle id" in proc.stderr
    assert USAGE in proc.stderr
    assert not plist.exists()


def test_minimal_valid_bundle_id(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden", "a")
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["CFBundleIdentifier"] == "a"


@pytest.mark.parametrize("bundle_id", ["\u00e9", "a\u00e9", "a.\u00e9", "\uff21"])
def test_non_ascii_bundle_id_rejected_under_utf8_locale(tmp_path: Path, bundle_id: str) -> None:
    # Under a UTF-8 locale bash 3.2's [A-Za-z] ranges also match non-ASCII
    # letters; the id must still be ASCII-only.
    plist = tmp_path / "Info.plist"
    env = {**os.environ, "LC_ALL": "en_US.UTF-8"}
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(plist), "0.0.1", "LMWarden", bundle_id],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2
    assert "write-info-plist.sh: invalid bundle id" in proc.stderr
    assert not plist.exists()


@pytest.mark.parametrize("bundle_id", ["a-b", "Z.9", "com.podwarden.lm-warden.smoke"])
def test_valid_bundle_id_charset(tmp_path: Path, bundle_id: str) -> None:
    # Every arm of the allowed set after the first char: dash, dot, upper, digit.
    plist = tmp_path / "Info.plist"
    proc = _write(plist, "0.0.1", "LMWarden", bundle_id)
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["CFBundleIdentifier"] == bundle_id


# --- Sparkle auto-update keys (P8): LMW_SPARKLE_FEED_URL / LMW_SPARKLE_PUBLIC_ED_KEY ---

SPARKLE_BOTH_SET_MSG = (
    "write-info-plist.sh: LMW_SPARKLE_FEED_URL and LMW_SPARKLE_PUBLIC_ED_KEY must be set together"
)
VALID_FEED_URL = "https://example.com/lm-warden/appcast.xml"
VALID_ED_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
# The feed keys are written only for a release tag (a dev build never gets them).
TAG = "v2026.10.21.1"


def test_sparkle_keys_written_when_both_set(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL=VALID_FEED_URL,
        LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY,
    )
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert info["SUFeedURL"] == VALID_FEED_URL
    assert info["SUPublicEDKey"] == VALID_ED_KEY


def test_sparkle_keys_absent_when_neither_set(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(plist, TAG, "LMWarden")
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert "SUFeedURL" not in info
    assert "SUPublicEDKey" not in info


def test_sparkle_keys_absent_when_both_empty(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL="",
        LMW_SPARKLE_PUBLIC_ED_KEY="",
    )
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert "SUFeedURL" not in info
    assert "SUPublicEDKey" not in info


def test_neither_set_bytes_identical_to_both_empty(tmp_path: Path) -> None:
    # Unset and empty are the same: no new keys, and the file is byte-identical
    # to today's output for the same args.
    neither = tmp_path / "neither.plist"
    empty = tmp_path / "empty.plist"
    proc = _write_sparkle(neither, TAG, "LMWarden")
    assert proc.returncode == 0, proc.stderr
    proc = _write_sparkle(
        empty,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL="",
        LMW_SPARKLE_PUBLIC_ED_KEY="",
    )
    assert proc.returncode == 0, proc.stderr
    assert neither.read_bytes() == empty.read_bytes()
    assert "SUFeedURL" not in neither.read_text()


@pytest.mark.parametrize(
    ("feed_url", "ed_key"),
    [(VALID_FEED_URL, ""), ("", VALID_ED_KEY)],
)
def test_only_one_sparkle_var_set_exits_2(tmp_path: Path, feed_url: str, ed_key: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL=feed_url,
        LMW_SPARKLE_PUBLIC_ED_KEY=ed_key,
    )
    assert proc.returncode == 2
    assert SPARKLE_BOTH_SET_MSG in proc.stderr
    assert USAGE in proc.stderr
    assert not plist.exists()


def test_sparkle_key_order_in_raw_text(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL=VALID_FEED_URL,
        LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY,
    )
    assert proc.returncode == 0, proc.stderr
    raw = plist.read_text()
    feed = raw.index("SUFeedURL")
    ed = raw.index("SUPublicEDKey")
    local = raw.index("NSLocalNetworkUsageDescription")
    assert local < feed < ed
    # Two-space indent like the other lines; no other Sparkle key at all.
    assert f"  <key>SUFeedURL</key><string>{VALID_FEED_URL}</string>\n" in raw
    assert f"  <key>SUPublicEDKey</key><string>{VALID_ED_KEY}</string>\n" in raw
    assert "SUEnableAutomaticChecks" not in raw


# --- LMW_SPARKLE_FEED_URL validation: https:// + at least one char, conservative charset ---


@pytest.mark.parametrize(
    "feed_url",
    [
        "http://x/a.xml",
        "https://",
        "https://x/a?b=1",
        "https://x/a&b",
        "https://x/a b",
        "https://x/<a>",
        "https://x/é",
    ],
)
def test_invalid_sparkle_feed_url_exits_2(tmp_path: Path, feed_url: str) -> None:
    plist = tmp_path / "Info.plist"
    env = _sparkle_env(LMW_SPARKLE_FEED_URL=feed_url, LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY)
    env["LC_ALL"] = "en_US.UTF-8"
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(plist), TAG, "LMWarden"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2
    assert "write-info-plist.sh: invalid LMW_SPARKLE_FEED_URL" in proc.stderr
    assert USAGE in proc.stderr
    assert not plist.exists()


@pytest.mark.parametrize(
    "feed_url",
    [
        "https://example.com/lm-warden/appcast.xml",
        "https://example.com:8443/a_b~c/%41.xml",
    ],
)
def test_valid_sparkle_feed_url(tmp_path: Path, feed_url: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL=feed_url,
        LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY,
    )
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["SUFeedURL"] == feed_url


# --- LMW_SPARKLE_PUBLIC_ED_KEY validation: 44 chars, first 43 base64, 44th '=' ---


def _mixed_case_ed_key() -> str:
    # 43 base64 chars mixing upper, lower, digits and both / and +, then '='.
    key = "A" * 13 + "a" * 13 + "B" * 12 + "0" * 3 + "/" + "+"
    # 13 + 13 + 12 + 3 + 1 + 1 = 43
    assert len(key) == 43
    return key + "="


@pytest.mark.parametrize(
    "ed_key",
    [
        "A" * 43,
        "A" * 45,
        "A" * 44,
        # The length boundary on its own: 43 and 45 chars that still end in '='.
        "A" * 42 + "=",
        "A" * 44 + "=",
        "A" * 21 + "-" + "A" * 22 + "=",
        "A" * 21 + "_" + "A" * 22 + "=",
        "A" * 21 + "é" + "A" * 21 + "=",
    ],
)
def test_invalid_sparkle_ed_key_exits_2(tmp_path: Path, ed_key: str) -> None:
    plist = tmp_path / "Info.plist"
    env = _sparkle_env(LMW_SPARKLE_FEED_URL=VALID_FEED_URL, LMW_SPARKLE_PUBLIC_ED_KEY=ed_key)
    env["LC_ALL"] = "en_US.UTF-8"
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(plist), TAG, "LMWarden"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2
    assert "write-info-plist.sh: invalid LMW_SPARKLE_PUBLIC_ED_KEY" in proc.stderr
    assert USAGE in proc.stderr
    assert not plist.exists()


@pytest.mark.parametrize("ed_key", [VALID_ED_KEY, _mixed_case_ed_key()])
def test_valid_sparkle_ed_key(tmp_path: Path, ed_key: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL=VALID_FEED_URL,
        LMW_SPARKLE_PUBLIC_ED_KEY=ed_key,
    )
    assert proc.returncode == 0, proc.stderr
    assert _load(plist)["SUPublicEDKey"] == ed_key


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
def test_sparkle_plist_passes_plutil_lint(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_sparkle(
        plist,
        TAG,
        "LMWarden",
        LMW_SPARKLE_FEED_URL="https://example.com:8443/a_b~c/%41.xml",
        LMW_SPARKLE_PUBLIC_ED_KEY=_mixed_case_ed_key(),
    )
    assert proc.returncode == 0, proc.stderr
    lint = subprocess.run(["plutil", "-lint", str(plist)], capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr
