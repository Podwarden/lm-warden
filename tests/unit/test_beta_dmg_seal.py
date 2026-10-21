"""``scripts/macos/beta-dmg-seal.sh`` against real (tiny) DMGs.

The seal is the last gate a beta DMG passes before it gets its checksum: it
mounts the DMG read-only, refuses one whose app carries any Sparkle (SU*) key
or a different version, and only then writes ``<dmg>.sha256``. A refused DMG
is left with no checksum, so publish:macos-beta (which needs both files) can
never publish it.

``macos``-marked: it needs hdiutil and plutil. The app inside the DMG is a
plain folder with a real Info.plist from write-info-plist.sh; the seal reads
the plist only (make-dmg.sh already verified the signature before packing).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]
SEAL = REPO / "scripts" / "macos" / "beta-dmg-seal.sh"
WRITE_PLIST = REPO / "scripts" / "macos" / "write-info-plist.sh"

TAG = "v2026.10.21.1"
FEED = "https://lmwarden.com/macos/appcast.xml"
ED_KEY = "A" * 43 + "="


def _dmg(
    tmp_path: Path, *, version: str = TAG, sparkle: bool = False, name: str | None = None
) -> Path:
    src = tmp_path / "src"
    app = src / "LM Warden.app" / "Contents"
    app.mkdir(parents=True)
    env = {**os.environ}
    env.pop("LMW_BETA_BUILD", None)
    env.pop("LMW_SPARKLE_FEED_URL", None)
    env.pop("LMW_SPARKLE_PUBLIC_ED_KEY", None)
    if sparkle:
        env["LMW_SPARKLE_FEED_URL"] = FEED
        env["LMW_SPARKLE_PUBLIC_ED_KEY"] = ED_KEY
    subprocess.run(
        ["bash", str(WRITE_PLIST), str(app / "Info.plist"), version, "LMWarden"],
        check=True,
        capture_output=True,
        env=env,
    )
    out = tmp_path / (name or f"LM-Warden-{TAG}-beta.dmg")
    subprocess.run(
        [
            "hdiutil",
            "create",
            "-volname",
            "LM Warden",
            "-srcfolder",
            str(src),
            "-ov",
            "-format",
            "UDZO",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _seal(dmg: Path, version: str = TAG) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SEAL), str(dmg), version], capture_output=True, text=True)


def test_a_feedless_beta_is_sealed_with_its_checksum(tmp_path: Path) -> None:
    dmg = _dmg(tmp_path)
    proc = _seal(dmg)
    assert proc.returncode == 0, proc.stderr
    sums = Path(f"{dmg}.sha256")
    assert sums.is_file()
    want = hashlib.sha256(dmg.read_bytes()).hexdigest()
    # shasum -a 256 format: "<hash>  <basename>", so `shasum -a 256 -c` works
    # next to the download.
    assert sums.read_text() == f"{want}  {dmg.name}\n"
    check = subprocess.run(
        ["shasum", "-a", "256", "-c", sums.name], cwd=tmp_path, capture_output=True, text=True
    )
    assert check.returncode == 0, check.stdout + check.stderr


def test_a_dmg_with_the_feed_keys_is_refused_with_no_checksum(tmp_path: Path) -> None:
    dmg = _dmg(tmp_path, sparkle=True)
    stale = Path(f"{dmg}.sha256")
    stale.write_text("stale\n")
    proc = _seal(dmg)
    assert proc.returncode == 1
    assert "SUFeedURL" in proc.stderr
    assert not stale.exists(), "a refused beta must not keep a checksum"


def test_a_version_mismatch_is_refused(tmp_path: Path) -> None:
    dmg = _dmg(tmp_path, version="v2026.10.21.2")
    proc = _seal(dmg)
    assert proc.returncode == 1
    assert not Path(f"{dmg}.sha256").exists()


def test_a_dmg_not_named_as_a_beta_is_refused(tmp_path: Path) -> None:
    dmg = _dmg(tmp_path, name=f"LM-Warden-{TAG}.dmg")
    proc = _seal(dmg)
    assert proc.returncode == 2
    assert not Path(f"{dmg}.sha256").exists()


def test_a_non_tag_version_is_refused(tmp_path: Path) -> None:
    dmg = _dmg(tmp_path, version="dev", name="LM-Warden-dev-beta.dmg")
    proc = _seal(dmg, "dev")
    assert proc.returncode == 2
    assert not Path(f"{dmg}.sha256").exists()


def test_nothing_stays_mounted(tmp_path: Path) -> None:
    dmg = _dmg(tmp_path, sparkle=True)
    _seal(dmg)
    info = subprocess.run(["hdiutil", "info"], capture_output=True, text=True, check=True).stdout
    for path in {str(dmg), str(dmg.resolve())}:
        assert path not in info, "the seal left the DMG attached"
