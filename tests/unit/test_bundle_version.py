"""Unit tests for ``scripts/macos/bundle-version.sh`` (P8) and its callers.

The script maps a build VERSION to the app's CFBundleVersion, the number
Sparkle compares: a CalVer release tag ``vYYYY.MM.DD.N`` becomes the integer
``YYYYMMDDNN``, anything else (short sha, ``dev``) becomes ``0``, and a
malformed tag is exit 2. Not marked ``macos``: it only needs ``bash`` (the
plists are parsed with ``plistlib``); the ``plutil`` and ``make`` legs skip
where those are absent.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "bundle-version.sh"
PLIST_SCRIPT = REPO / "scripts" / "macos" / "write-info-plist.sh"

VALID_FEED_URL = "https://example.com/lm-warden/appcast.xml"
VALID_ED_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

needs_make = pytest.mark.skipif(
    shutil.which("make") is None,
    reason="needs make (absent from the CI deps image; runs on the bare host and the M1)",
)


def _map(version: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), version], capture_output=True, text=True, env=env)


def _bv(version: str) -> str:
    proc = _map(version)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert out.endswith("\n") and out.count("\n") == 1, repr(out)
    return out.strip()


# --- the mapping ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v2026.10.21.1", "2026102101"),
        ("v2026.10.21.3", "2026102103"),
        ("v2026.10.21.10", "2026102110"),
        ("v2026.10.21.99", "2026102199"),
        ("v2026.05.15.4", "2026051504"),
        ("v2026.08.09.1", "2026080901"),  # 08/09: not parsed as octal
        ("v2024.02.29.1", "2024022901"),  # leap day
        ("v2000.02.29.1", "2000022901"),  # divisible by 400: leap
        ("v2026.01.31.2", "2026013102"),
        ("v2026.04.30.2", "2026043002"),
        ("v2099.12.31.99", "2099123199"),
    ],
)
def test_release_tag_maps_to_yyyymmddnn(tag: str, expected: str) -> None:
    assert _bv(tag) == expected


@pytest.mark.parametrize(
    "version", ["dev", "abc1234", "18591a88", "0.0.1", "1.2.3-test", "V2026.10.21.1"]
)
def test_non_tag_version_maps_to_zero(version: str) -> None:
    assert _bv(version) == "0"


# Strictly increasing across every boundary that matters: N 9 -> 10 on one
# day, N 99 -> next day's 1, day -> day, month end -> next month, year end ->
# next year, Feb 28/29 -> Mar 1.
ORDERED_TAGS = [
    "v2025.12.31.1",
    "v2025.12.31.99",
    "v2026.01.01.1",
    "v2026.01.09.1",
    "v2026.01.10.1",
    "v2026.01.31.5",
    "v2026.02.01.1",
    "v2026.02.28.1",
    "v2026.03.01.1",
    "v2026.09.30.1",
    "v2026.10.01.1",
    "v2026.10.21.1",
    "v2026.10.21.2",
    "v2026.10.21.9",
    "v2026.10.21.10",
    "v2026.10.21.11",
    "v2026.10.21.99",
    "v2026.10.22.1",
    "v2026.11.01.1",
    "v2026.12.31.1",
    "v2027.01.01.1",
    "v2028.02.29.1",
    "v2028.03.01.1",
]


def test_ordering_is_strictly_increasing_across_boundaries() -> None:
    numbers = [int(_bv(tag)) for tag in ORDERED_TAGS]
    assert all(a < b for a, b in zip(numbers, numbers[1:], strict=False)), list(
        zip(ORDERED_TAGS, numbers, strict=True)
    )
    # Every release sorts above the dev value, and the largest fits in int32.
    assert min(numbers) > 0
    assert int(_bv("v2099.12.31.99")) < 2**31


def test_ordering_is_the_same_as_string_ordering() -> None:
    # Fixed width (10 digits) means lexical == numeric, so a comparator that
    # falls back to string compare still orders them right.
    values = [_bv(tag) for tag in ORDERED_TAGS]
    assert all(len(v) == 10 for v in values)
    assert values == sorted(values)


@pytest.mark.parametrize(
    "tag",
    [
        "v2026.13.01.1",  # month 13
        "v2026.00.01.1",  # month 0
        "v2026.10.00.1",  # day 0
        "v2026.10.32.1",  # day 32
        "v2026.04.31.1",  # April has 30
        "v2026.06.31.1",  # June has 30
        "v2026.09.31.1",  # September has 30
        "v2026.11.31.1",  # November has 30
        "v2025.02.29.1",  # not a leap year
        "v2100.02.29.1",  # century, not /400 (also out of the year range)
        "v2026.02.30.1",
        "v2026.10.21.0",  # N starts at 1
        "v2026.10.21.01",  # leading zero
        "v2026.10.21.100",  # N > 99 does not fit two digits
        "v2026.1.21.1",  # month not zero-padded
        "v2026.10.2.1",  # day not zero-padded
        "v26.10.21.1",  # two-digit year
        "v1999.10.21.1",  # year below range
        "v2100.10.21.1",  # year above range
        "v2026.10.21",  # no N
        "v2026.10.21.",
        "v2026.10.21.1.1",
        "v2026.10.21.1-rc1",
        "v2026.10.21.1 ",
        "v2026.10.21.1\n",
        "v2026-10-21-1",
        "v2026.10.21.a",
        "v1",
        "v0",
    ],
)
def test_malformed_tag_exits_2_with_nothing_on_stdout(tag: str) -> None:
    proc = _map(tag)
    assert proc.returncode == 2, (tag, proc.stdout)
    assert proc.stdout == ""
    assert "bundle-version.sh: malformed release tag" in proc.stderr


def test_non_ascii_digits_rejected_under_utf8_locale() -> None:
    # A fullwidth digit must not pass as [0-9] under a UTF-8 locale.
    env = {**os.environ, "LC_ALL": "en_US.UTF-8"}
    proc = _map("v2026.10.21.１", env=env)
    assert proc.returncode == 2
    assert proc.stdout == ""


@pytest.mark.parametrize("argv", [[], [""], ["v2026.10.21.1", "extra"]])
def test_usage_exits_2(argv: list[str]) -> None:
    proc = subprocess.run(["bash", str(SCRIPT), *argv], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "usage: bundle-version.sh <version>" in proc.stderr
    assert proc.stdout == ""


# --- write-info-plist.sh uses it ------------------------------------------------


def _write_plist(plist: Path, version: str, **sparkle: str) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env.pop("LMW_SPARKLE_FEED_URL", None)
    env.pop("LMW_SPARKLE_PUBLIC_ED_KEY", None)
    env.update(sparkle)
    return subprocess.run(
        ["bash", str(PLIST_SCRIPT), str(plist), version, "LMWarden"],
        capture_output=True,
        text=True,
        env=env,
    )


def _load(plist: Path) -> dict:
    with plist.open("rb") as fh:
        return plistlib.load(fh)


def test_plist_release_tag_gets_numeric_bundle_version(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_plist(plist, "v2026.10.21.12")
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert info["CFBundleVersion"] == "2026102112"
    # The human-readable version stays the tag (make-dmg.sh checks it).
    assert info["CFBundleShortVersionString"] == "v2026.10.21.12"


@pytest.mark.parametrize("version", ["dev", "abc1234"])
def test_plist_dev_build_gets_zero(tmp_path: Path, version: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_plist(plist, version)
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert info["CFBundleVersion"] == "0"
    assert info["CFBundleShortVersionString"] == version


def test_plist_release_tag_with_feed_keeps_feed_keys(tmp_path: Path) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_plist(
        plist,
        "v2026.10.21.1",
        LMW_SPARKLE_FEED_URL=VALID_FEED_URL,
        LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY,
    )
    assert proc.returncode == 0, proc.stderr
    info = _load(plist)
    assert info["SUFeedURL"] == VALID_FEED_URL
    assert info["SUPublicEDKey"] == VALID_ED_KEY
    assert proc.stderr == ""


@pytest.mark.parametrize("version", ["dev", "abc1234", "0.0.1"])
def test_plist_dev_build_never_gets_feed_keys(tmp_path: Path, version: str) -> None:
    # Both feed vars set and valid, but a dev build (CFBundleVersion 0) would
    # see every appcast item as newer: the keys are left out and the plist is
    # byte-identical to the one written with neither var set.
    with_feed = tmp_path / "with.plist"
    without = tmp_path / "without.plist"
    proc = _write_plist(
        with_feed,
        version,
        LMW_SPARKLE_FEED_URL=VALID_FEED_URL,
        LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY,
    )
    assert proc.returncode == 0, proc.stderr
    assert "is not a release tag; leaving SUFeedURL/SUPublicEDKey out" in proc.stderr
    proc = _write_plist(without, version)
    assert proc.returncode == 0, proc.stderr
    assert with_feed.read_bytes() == without.read_bytes()
    assert "SUFeedURL" not in with_feed.read_text()
    assert "SUPublicEDKey" not in with_feed.read_text()


def test_plist_byte_identical_for_unchanged_inputs(tmp_path: Path) -> None:
    a = tmp_path / "a.plist"
    b = tmp_path / "b.plist"
    for p in (a, b):
        proc = _write_plist(p, "v2026.10.21.3")
        assert proc.returncode == 0, proc.stderr
    assert a.read_bytes() == b.read_bytes()
    raw = a.read_text()
    assert "  <key>CFBundleVersion</key><string>2026102103</string>\n" in raw
    assert "  <key>CFBundleShortVersionString</key><string>v2026.10.21.3</string>\n" in raw


@pytest.mark.parametrize("tag", ["v2026.13.01.1", "v2026.10.21.0", "v2026.10.21.100"])
def test_plist_malformed_tag_exits_2_and_writes_nothing(tmp_path: Path, tag: str) -> None:
    plist = tmp_path / "sub" / "Info.plist"
    proc = _write_plist(plist, tag)
    assert proc.returncode == 2
    assert "bundle-version.sh: malformed release tag" in proc.stderr
    assert "usage: write-info-plist.sh" in proc.stderr
    assert not plist.exists()
    assert not plist.parent.exists()


@pytest.mark.parametrize("version", ["a b", "a<b", "a&b", "1.0+x", "a\nb"])
def test_plist_invalid_version_charset_exits_2(tmp_path: Path, version: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_plist(plist, version)
    assert proc.returncode == 2
    assert "write-info-plist.sh: invalid version" in proc.stderr
    assert not plist.exists()


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
@pytest.mark.parametrize("version", ["v2026.10.21.10", "abc1234"])
def test_plist_bundle_version_via_plutil(tmp_path: Path, version: str) -> None:
    plist = tmp_path / "Info.plist"
    proc = _write_plist(
        plist,
        version,
        LMW_SPARKLE_FEED_URL=VALID_FEED_URL,
        LMW_SPARKLE_PUBLIC_ED_KEY=VALID_ED_KEY,
    )
    assert proc.returncode == 0, proc.stderr
    lint = subprocess.run(["plutil", "-lint", str(plist)], capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    got = subprocess.run(
        ["plutil", "-extract", "CFBundleVersion", "raw", "-o", "-", str(plist)],
        capture_output=True,
        text=True,
    )
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == ("2026102110" if version.startswith("v") else "0")


# --- make macos-dmg validates the tag before it builds -------------------------


def _make(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("VERSION", "OUT", "MAKEFLAGS")}
    return subprocess.run(
        ["make", "--no-print-directory", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


@needs_make
def test_make_dmg_dry_run_checks_the_tag_before_the_plist() -> None:
    proc = _make("-n", "macos-dmg", "VERSION=v2026.10.21.1", "OUT=/x/y.dmg")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    check = out.find('scripts/macos/bundle-version.sh "v2026.10.21.1"')
    plist = out.find("scripts/macos/write-info-plist.sh")
    dmg = out.find('scripts/macos/make-dmg.sh "dist/LM Warden.app" "v2026.10.21.1" "$OUT"')
    assert 0 <= check < plist < dmg, out
    assert "swift build" in out[check:plist]


@needs_make
@pytest.mark.parametrize("tag", ["v2026.13.01.1", "v2026.10.21.0", "v2026.10.21.01"])
def test_make_dmg_malformed_tag_builds_nothing(tag: str) -> None:
    # MAKE=false: if the tag check ever stops firing, the next recipe line
    # ($(MAKE) macos-bundle-python ...) fails at once instead of starting a
    # real swift/Python/web-UI build from a unit test (a 60 s timeout kills
    # make but leaves its build children running).
    proc = _make("macos-dmg", f"VERSION={tag}", "MAKE=false")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert f"make macos-dmg: VERSION={tag} is a malformed release tag" in proc.stderr
    assert "swift" not in proc.stdout and "bundle-python" not in proc.stdout


@needs_make
def test_make_dmg_dev_sha_passes_the_check() -> None:
    proc = _make("-n", "macos-dmg", "VERSION=abc1234", "OUT=/x/y.dmg")
    assert proc.returncode == 0, proc.stderr
    assert 'scripts/macos/make-dmg.sh "dist/LM Warden.app" "abc1234" "$OUT"' in proc.stdout
