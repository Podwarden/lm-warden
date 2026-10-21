"""Sparkle.framework in "LM Warden.app" (P8 sparkle-embed).

``scripts/macos/embed-sparkle.sh`` fetches the pinned
Sparkle-for-Swift-Package-Manager.zip, checks its sha256, extracts the macOS
slice and places Sparkle.framework in ``Contents/Frameworks``;
``sign-bundle.sh`` signs its nested code inside-out and ``verify-bundle.sh``
requires each piece validly signed.

Not marked ``macos``: usage, ``--print-plan``, the pin-drift guard (script vs
macos/Package.swift), the refusal of a tampered cached zip (no network) and
the ``make -n`` wiring run anywhere bash/make/shasum are. The legs on the real
pinned zip are ``macos``: they need ditto/codesign/cc and the zip itself,
taken from ``~/Library/Caches/lmw-macos-sparkle`` when a copy there has the
pinned sha256, otherwise fetched once (network) by the script's own
``--fetch-only`` into a tmp cache.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "embed-sparkle.sh"
SIGN = REPO / "scripts" / "macos" / "sign-bundle.sh"
VERIFY = REPO / "scripts" / "macos" / "verify-bundle.sh"
PACKAGE = REPO / "macos" / "Package.swift"

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
needs_shasum = pytest.mark.skipif(shutil.which("shasum") is None, reason="needs shasum")
needs_mac = pytest.mark.skipif(
    not (
        sys.platform == "darwin"
        and all(shutil.which(t) for t in ("ditto", "codesign", "cc", "otool", "shasum"))
    ),
    reason="needs darwin with ditto, codesign, cc, otool and shasum",
)


def _run(*args: str, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, timeout=300)


def _script_pin(name: str) -> str:
    m = re.search(rf'^{name}="([^"]+)"$', SCRIPT.read_text(), re.M)
    assert m, f"{name} not pinned in {SCRIPT}"
    return m.group(1).replace("${SPARKLE_VERSION}", _script_version())


def _script_version() -> str:
    m = re.search(r'^SPARKLE_VERSION="([^"]+)"$', SCRIPT.read_text(), re.M)
    assert m
    return m.group(1)


def _app(tmp_path: Path, name: str = "LM Warden.app") -> Path:
    app = tmp_path / "staging" / name
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        "  <key>CFBundleIdentifier</key><string>com.podwarden.lmwarden</string>\n"
        "  <key>CFBundleExecutable</key><string>LMWarden</string>\n"
        "  <key>CFBundlePackageType</key><string>APPL</string>\n"
        "</dict>\n</plist>\n"
    )
    return app


# --- pins, usage, plan --------------------------------------------------------


def test_pin_is_a_stable_2x_and_matches_package_swift() -> None:
    version = _script_version()
    assert re.fullmatch(r"2\.\d+\.\d+", version), f"not a stable 2.x: {version}"
    url = _script_pin("SPARKLE_URL")
    sha = _script_pin("SPARKLE_SHA256")
    assert url == (
        f"https://github.com/sparkle-project/Sparkle/releases/download/{version}/"
        "Sparkle-for-Swift-Package-Manager.zip"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    pkg = PACKAGE.read_text()
    assert f'let sparkleVersion = "{version}"' in pkg
    assert f'let sparkleChecksum = "{sha}"' in pkg
    assert (
        'let sparkleURL = "https://github.com/sparkle-project/Sparkle/releases/download/'
        '\\(sparkleVersion)/Sparkle-for-Swift-Package-Manager.zip"' in pkg
    )
    assert re.search(
        r'\.binaryTarget\(\s*name: "Sparkle",\s*url: sparkleURL,\s*checksum: sparkleChecksum', pkg
    )
    assert '"-rpath", "-Xlinker", "@executable_path/../Frameworks"' in pkg


def test_usage_without_an_app_is_exit_2() -> None:
    proc = _run()
    assert proc.returncode == 2
    assert "usage: embed-sparkle.sh" in proc.stderr


def test_not_an_app_is_exit_2(tmp_path: Path) -> None:
    proc = _run(str(tmp_path), "--cache-dir", str(tmp_path / "c"), "--offline")
    assert proc.returncode == 2, proc.stderr
    assert "not an app bundle" in proc.stderr
    assert not (tmp_path / "c").exists()


def test_print_plan_touches_nothing(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    proc = _run("X.app", "--print-plan", "--cache-dir", str(cache))
    assert proc.returncode == 0, proc.stderr
    plan = dict(line.split("=", 1) for line in proc.stdout.splitlines())
    assert plan["sparkle_version"] == _script_version()
    assert plan["sparkle_sha256"] == _script_pin("SPARKLE_SHA256")
    assert plan["archive"] == f"{cache}/Sparkle-for-Swift-Package-Manager-{_script_version()}.zip"
    assert plan["dest"] == "X.app/Contents/Frameworks/Sparkle.framework"
    assert not cache.exists()


def test_verify_rejects_a_misspelled_or_extra_argument(tmp_path: Path) -> None:
    # A typo of --require-sparkle must not silently verify without it.
    app = _app(tmp_path)
    for extra in (["--require-sparkl"], ["--require-sparkle", "x"], ["x"]):
        res = _run(str(app), *extra, script=VERIFY)
        assert res.returncode == 2, (extra, res.stdout + res.stderr)
        assert "usage: verify-bundle.sh <app> [--require-sparkle]" in res.stderr


# --- verify: a tampered or absent archive is refused (no network) -------------


@needs_shasum
def test_tampered_cached_archive_is_refused_and_kept(tmp_path: Path) -> None:
    app = _app(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    bad = cache / f"Sparkle-for-Swift-Package-Manager-{_script_version()}.zip"
    bad.write_bytes(b"PK\x03\x04 not the pinned zip")
    proc = _run(str(app), "--cache-dir", str(cache))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "checksum mismatch for cached" in proc.stderr
    assert "refusing" in proc.stderr
    assert (
        bad.read_bytes() == b"PK\x03\x04 not the pinned zip"
    ), "a mismatching cache must not be replaced"
    assert not (app / "Contents" / "Frameworks").exists()


def test_offline_without_cache_is_refused(tmp_path: Path) -> None:
    app = _app(tmp_path)
    proc = _run(str(app), "--cache-dir", str(tmp_path / "cache"), "--offline")
    assert proc.returncode == 1
    assert "--offline and no cached archive" in proc.stderr
    assert not (app / "Contents" / "Frameworks").exists()


# --- make wiring ----------------------------------------------------------------


def _make(*args: str) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VERSION", "OUT", "MAKEFLAGS", "SPARKLE_CACHE")
    }
    return subprocess.run(
        ["make", "--no-print-directory", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


@needs_make
def test_macos_app_embeds_sparkle_before_the_adhoc_seal() -> None:
    proc = _make("-n", "macos-app", "SPARKLE_CACHE=/c/spk")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    plist = out.find("scripts/macos/write-info-plist.sh")
    embed = out.find(
        'scripts/macos/embed-sparkle.sh "dist/LM Warden.app" \\\n'
        '\t\t--cache-dir "/c/spk" --build-dir "macos/.build/release"'
    )
    seal = out.find('codesign --force --deep -s - "dist/LM Warden.app"')
    assert -1 < plist < embed < seal, out


@needs_make
def test_bundle_python_verifies_with_require_sparkle_and_dmg_passes_cache() -> None:
    out = _make("-n", "macos-bundle-python").stdout
    assert 'scripts/macos/verify-bundle.sh "dist/LM Warden.app" --require-sparkle' in out, out
    dmg = _make("-n", "macos-dmg", "VERSION=9.9.9", "SPARKLE_CACHE=/c/spk")
    assert dmg.returncode == 0, dmg.stderr
    assert 'SPARKLE_CACHE="/c/spk"' in dmg.stdout


# --- the real pinned zip (macos) -------------------------------------------------


@pytest.fixture(scope="module")
def pinned_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A cache dir holding the pinned zip: copied from the user cache when it
    matches the pin, else fetched by the script itself (network)."""
    if not (sys.platform == "darwin" and shutil.which("shasum")):
        pytest.skip("needs darwin")
    cache = tmp_path_factory.mktemp("sparkle-cache")
    name = f"Sparkle-for-Swift-Package-Manager-{_script_version()}.zip"
    user = Path.home() / "Library" / "Caches" / "lmw-macos-sparkle" / name
    if user.is_file() and hashlib.sha256(user.read_bytes()).hexdigest() == _script_pin(
        "SPARKLE_SHA256"
    ):
        shutil.copy(user, cache / name)
    else:
        proc = _run("--fetch-only", "--cache-dir", str(cache))
        if proc.returncode != 0:
            pytest.skip(f"could not fetch the pinned zip: {proc.stderr.strip()}")
        assert "(sha256 ok)" in proc.stdout
    return cache


def _embed(app: Path, cache: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run(str(app), "--cache-dir", str(cache), "--offline", *extra)


def _linking_main(app: Path, fw_parent: Path, rpath: bool = True) -> None:
    """A tiny C main that links @rpath/Sparkle.framework, like the Swift app."""
    args = [
        "cc",
        "-x",
        "c",
        "-",
        "-o",
        str(app / "Contents" / "MacOS" / "LMWarden"),
        "-F",
        str(fw_parent),
        "-framework",
        "Sparkle",
        "-mmacosx-version-min=14.0",
    ]
    if rpath:
        args.append("-Wl,-rpath,@executable_path/../Frameworks")
    res = subprocess.run(
        args, input="int main(void){return 0;}\n", capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stderr


def _placed_app(tmp_path: Path, cache: Path, rpath: bool = True) -> Path:
    app = _app(tmp_path)
    proc = _embed(app, cache)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _linking_main(app, app / "Contents" / "Frameworks", rpath=rpath)
    return app


@pytest.mark.macos
@needs_mac
def test_places_the_pinned_framework_with_symlinks(tmp_path: Path, pinned_cache: Path) -> None:
    app = _app(tmp_path)
    proc = _embed(app, pinned_cache)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fetch: cached" in proc.stdout and "(sha256 ok)" in proc.stdout
    fw = app / "Contents" / "Frameworks" / "Sparkle.framework"
    assert (fw / "Versions" / "Current").is_symlink()
    assert os.readlink(fw / "Versions" / "Current") == "B"
    assert (fw / "Sparkle").is_symlink() and (fw / "Updater.app").is_symlink()
    for rel in (
        "Versions/B/Autoupdate",
        "Versions/B/Updater.app/Contents/MacOS/Updater",
        "Versions/B/XPCServices/Downloader.xpc/Contents/MacOS/Downloader",
        "Versions/B/XPCServices/Installer.xpc/Contents/MacOS/Installer",
    ):
        assert (fw / rel).is_file(), rel
    # Re-running replaces the framework cleanly (no nested Sparkle.framework).
    again = _embed(app, pinned_cache)
    assert again.returncode == 0, again.stderr
    assert not (fw / "Sparkle.framework").exists()


@pytest.mark.macos
@needs_mac
def test_tampered_archive_content_is_refused(tmp_path: Path, pinned_cache: Path) -> None:
    # One flipped byte in an otherwise real zip: sha256 differs, nothing placed.
    cache = tmp_path / "cache"
    cache.mkdir()
    src = next(pinned_cache.glob("*.zip"))
    data = bytearray(src.read_bytes())
    data[len(data) // 2] ^= 0xFF
    (cache / src.name).write_bytes(bytes(data))
    app = _app(tmp_path)
    proc = _embed(app, cache)
    assert proc.returncode == 1
    assert "checksum mismatch" in proc.stderr
    assert not (app / "Contents" / "Frameworks").exists()


@pytest.mark.macos
@needs_mac
def test_build_dir_framework_must_match(tmp_path: Path, pinned_cache: Path) -> None:
    good = _app(tmp_path)
    assert _embed(good, pinned_cache).returncode == 0
    build = tmp_path / "build"
    build.mkdir()
    subprocess.run(
        [
            "ditto",
            str(good / "Contents" / "Frameworks" / "Sparkle.framework"),
            str(build / "Sparkle.framework"),
        ],
        check=True,
    )
    app = _app(tmp_path, "Other.app")
    ok = _embed(app, pinned_cache, "--build-dir", str(build))
    assert ok.returncode == 0, ok.stderr
    assert "matches the pinned framework" in ok.stdout
    with open(build / "Sparkle.framework" / "Versions" / "B" / "Sparkle", "ab") as fh:
        fh.write(b"\0")
    shutil.rmtree(app / "Contents" / "Frameworks")
    bad = _embed(app, pinned_cache, "--build-dir", str(build))
    assert bad.returncode == 1
    assert "differs from pinned" in bad.stderr
    assert not (app / "Contents" / "Frameworks").exists()


@pytest.mark.macos
@needs_mac
def test_signs_inside_out_and_verify_requires_sparkle(tmp_path: Path, pinned_cache: Path) -> None:
    app = _placed_app(tmp_path, pinned_cache)
    sig = _run(str(app), script=SIGN)
    assert sig.returncode == 0, sig.stdout + sig.stderr
    order = [ln[len("signed: ") :] for ln in sig.stdout.splitlines() if ln.startswith("signed: ")]
    fw = "LM Warden.app/Contents/Frameworks/Sparkle.framework"
    want = [
        f"{fw}/Versions/B/XPCServices/Downloader.xpc/Contents/MacOS/Downloader",
        f"{fw}/Versions/B/XPCServices/Downloader.xpc",
        f"{fw}/Versions/B/XPCServices/Installer.xpc",
        f"{fw}/Versions/B/Autoupdate",
        f"{fw}/Versions/B/Updater.app",
        fw,
        "LM Warden.app",
    ]
    idx = [order.index(w) for w in want]
    assert idx == sorted(idx), order
    assert order.index(f"{fw}/Versions/B/Updater.app/Contents/MacOS/Updater") < order.index(
        f"{fw}/Versions/B/Updater.app"
    )
    res = _run(str(app), "--require-sparkle", script=VERIFY)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "verify-bundle.sh: sparkle 5 nested items signed" in res.stdout
    assert res.stdout.splitlines()[-1].startswith("verify-bundle.sh: verified ")
    otool = subprocess.run(
        ["otool", "-L", str(app / "Contents" / "MacOS" / "LMWarden")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "@rpath/Sparkle.framework/Versions/B/Sparkle" in otool


@pytest.mark.macos
@needs_mac
def test_verify_fails_on_unsigned_xpc(tmp_path: Path, pinned_cache: Path) -> None:
    app = _placed_app(tmp_path, pinned_cache)
    assert _run(str(app), script=SIGN).returncode == 0
    xpc = app / "Contents/Frameworks/Sparkle.framework/Versions/B/XPCServices/Installer.xpc"
    subprocess.run(["codesign", "--remove-signature", str(xpc)], check=True)
    res = _run(str(app), script=VERIFY)
    assert res.returncode == 1
    assert "not validly signed: LM Warden.app/Contents/Frameworks/Sparkle.framework" in res.stderr


@pytest.mark.macos
@needs_mac
def test_verify_fails_when_a_linking_app_lacks_the_framework(
    tmp_path: Path, pinned_cache: Path
) -> None:
    app = _placed_app(tmp_path, pinned_cache)
    shutil.rmtree(app / "Contents" / "Frameworks")
    assert _run(str(app), script=SIGN).returncode == 0
    res = _run(str(app), script=VERIFY)
    assert res.returncode == 1
    assert (
        "verify-bundle.sh: missing: LM Warden.app/Contents/Frameworks/Sparkle.framework/"
        "Versions/B/XPCServices/Downloader.xpc" in res.stderr
    ), res.stderr


@pytest.mark.macos
@needs_mac
def test_verify_fails_without_the_bundle_rpath(tmp_path: Path, pinned_cache: Path) -> None:
    app = _placed_app(tmp_path, pinned_cache, rpath=False)
    assert _run(str(app), script=SIGN).returncode == 0
    res = _run(str(app), script=VERIFY)
    assert res.returncode == 1
    assert "no LC_RPATH @executable_path/../Frameworks" in res.stderr


@pytest.mark.macos
@needs_mac
def test_require_sparkle_fails_an_app_that_does_not_link_it(tmp_path: Path) -> None:
    app = _app(tmp_path)
    shutil.copy("/usr/bin/true", app / "Contents" / "MacOS" / "LMWarden")
    assert _run(str(app), script=SIGN).returncode == 0
    assert _run(str(app), script=VERIFY).returncode == 0
    res = _run(str(app), "--require-sparkle", script=VERIFY)
    assert res.returncode == 1
    assert "does not link @rpath/Sparkle.framework" in res.stderr
