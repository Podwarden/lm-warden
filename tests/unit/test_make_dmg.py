"""DMG packaging test for the macOS app (Task 7).

Selected only under ``-m macos`` (see ``pyproject.toml``); the default
``addopts`` deselect it. ``make macos-dmg`` builds the
bundled-Python ``LM Warden.app`` (``make macos-bundle-python``) and packs it
with ``scripts/macos/make-dmg.sh``; the test mounts the DMG read-only, reads
the Info.plist back through ``plistlib``, runs verify-bundle.sh on the
mounted app and imports ``app.main`` with the bundled interpreter. The mount
is always detached in a ``finally``. The smaller legs pack a tiny signed
stand-in app to pin make-dmg.sh's gates (bundled python present, version
match, verify before ``hdiutil``).

It shells out only to ``make``, ``hdiutil`` and ``codesign`` (present on
every macOS box) and imports no app code itself, so a CI venv that only has
``pytest`` + stdlib is enough. The full leg needs network for the first
python-build-standalone fetch (cached in ``PYTHON_CACHE`` afterwards), takes
~5 min and needs a clean app/ (bundle-python.sh refuses uncommitted edits), so
it is opt-in: ``LMW_DMG_BUILD=1`` (set by the CI test:macos job). Routine M1
gates run only the small legs.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]

MAKE_DMG = REPO / "scripts" / "macos" / "make-dmg.sh"


def _run(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, capture_output=True, text=True, **kw)


def _tiny_app(tmp_path: Path, version: str = "0.0.0-test", *, python: bool = True) -> Path:
    """A signed stand-in for the bundled-Python app: real Info.plist, a Mach-O
    executable and (by default) a Mach-O at the renamed interpreter's path,
    Contents/Resources/python/bin/LM Warden Server,
    ad-hoc signed inside-out by sign-bundle.sh."""
    app = tmp_path / "src" / "LM Warden.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    # copyfile, not copy2: copystat on a SIP-flagged system binary is EPERM.
    shutil.copyfile("/usr/bin/true", app / "Contents" / "MacOS" / "LMWarden")
    (app / "Contents" / "MacOS" / "LMWarden").chmod(0o755)
    if python:
        py = app / "Contents" / "Resources" / "python" / "bin"
        py.mkdir(parents=True)
        shutil.copyfile("/usr/bin/true", py / "LM Warden Server")
        (py / "LM Warden Server").chmod(0o755)
    _run(
        str(REPO / "scripts" / "macos" / "write-info-plist.sh"),
        str(app / "Contents" / "Info.plist"),
        version,
        "LMWarden",
    )
    _run(str(REPO / "scripts" / "macos" / "sign-bundle.sh"), str(app), "-")
    return app


def test_make_dmg_verifies_the_staged_copy_before_hdiutil() -> None:
    make_src = MAKE_DMG.read_text(encoding="utf-8")
    # verify-bundle.sh must run on the staged copy (what the DMG is made
    # from), after it is copied and before `hdiutil create`, so a bundle that
    # no longer verifies never reaches the DMG (P8: verify on every build).
    lines = make_src.splitlines()
    copy_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("ditto "))
    verify_idx = next(i for i, ln in enumerate(lines) if '"$verify_bundle"' in ln)
    hdiutil_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "hdiutil create \\")
    assert copy_idx < verify_idx < hdiutil_idx, make_src
    assert '"$app_root"' in lines[verify_idx], lines[verify_idx]
    assert '"$app_root"' in lines[copy_idx], lines[copy_idx]


def _make_dmg(scripts: Path, app: Path, version: str, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(scripts / "make-dmg.sh"), str(app), version, str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_make_dmg_stops_when_verify_fails(tmp_path: Path) -> None:
    # Behavioural leg for the text-order pin above: run a copy of make-dmg.sh
    # whose sibling verify-bundle.sh is a stub that records its argument,
    # checks the bundle it got is signed, and then fails. make-dmg.sh must
    # exit with the stub's status and write no DMG. Kills a commented-out
    # call, `|| true`, and a $verify_bundle that resolves to another script.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(MAKE_DMG, scripts / "make-dmg.sh")
    seen = tmp_path / "verify-args"
    stub = scripts / "verify-bundle.sh"
    stub.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" > {shlex.quote(str(seen))}\n'
        'codesign --verify --strict "$1" || exit 3\n'
        "exit 1\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    app = _tiny_app(tmp_path)
    out_dmg = tmp_path / "out.dmg"

    proc = _make_dmg(scripts, app, "0.0.0-test", out_dmg)

    assert proc.returncode == 1, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert not out_dmg.exists(), "a DMG was written although verify-bundle.sh failed"
    args = seen.read_text(encoding="utf-8").splitlines()
    assert len(args) == 1 and args[0].endswith("/LM Warden.app"), args
    # It verified the staged copy, the bytes that would ship, not the source.
    assert Path(args[0]) != app, args


def test_make_dmg_refuses_an_app_without_bundled_python(tmp_path: Path) -> None:
    app = _tiny_app(tmp_path, python=False)
    out_dmg = tmp_path / "out.dmg"
    proc = _make_dmg(MAKE_DMG.parent, app, "0.0.0-test", out_dmg)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "no bundled python" in proc.stderr
    assert "make macos-bundle-python" in proc.stderr
    assert not out_dmg.exists()


def test_make_dmg_refuses_a_version_mismatch(tmp_path: Path) -> None:
    app = _tiny_app(tmp_path, version="1.2.3")
    out_dmg = tmp_path / "out.dmg"
    proc = _make_dmg(MAKE_DMG.parent, app, "0.0.0-test", out_dmg)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "CFBundleShortVersionString is 1.2.3, not 0.0.0-test" in proc.stderr
    assert not out_dmg.exists()


def test_make_dmg_usage(tmp_path: Path) -> None:
    proc = _make_dmg(MAKE_DMG.parent, tmp_path / "missing.app", "1", tmp_path / "o.dmg")
    assert proc.returncode == 2
    assert "usage: make-dmg.sh <app> <version> <out.dmg>" in proc.stderr


@pytest.mark.skipif(
    os.environ.get("LMW_DMG_BUILD") != "1",
    reason="full bundle + DMG build (~5 min, network): opt in with LMW_DMG_BUILD=1",
)
def test_dmg_mounts_and_carries_version(tmp_path: Path) -> None:
    version = "0.0.0-test"
    out_dmg = tmp_path / f"LM-Warden-{version}.dmg"

    # Build the DMG via the make target: macos-bundle-python (swift build,
    # embed the pinned python + deps + app/, sign, verify) then make-dmg.sh.
    # The make target is the public surface; exercising it here keeps the
    # test honest about what CI actually does. PYTHON_CACHE, if set in the
    # environment, is passed through (make's `?=`), so CI can keep the
    # interpreter download outside the checkout.
    _run("make", "macos-dmg", f"VERSION={version}", f"OUT={out_dmg}", cwd=REPO)
    assert out_dmg.is_file(), f"make-dmg.sh produced no DMG at {out_dmg}"

    mount = tmp_path / "mount"
    mount.mkdir()

    try:
        _run(
            "hdiutil",
            "attach",
            "-nobrowse",
            "-readonly",
            "-mountpoint",
            str(mount),
            str(out_dmg),
        )
        # The volume root contains "LM Warden.app".
        apps = list(mount.glob("*.app"))
        assert len(apps) == 1, f"expected one .app on the DMG, got {apps}"
        app = apps[0]

        with (app / "Contents" / "Info.plist").open("rb") as fh:
            info = plistlib.load(fh)

        assert info.get("CFBundleExecutable") == "LMWarden"
        assert info.get("CFBundlePackageType") == "APPL"
        assert info.get("CFBundleShortVersionString") == version
        assert info.get("CFBundleIdentifier") == "com.podwarden.lmwarden"
        assert str(info.get("LSMinimumSystemVersion")) == "14.0"

        # The executable is present inside the bundle.
        assert (app / "Contents" / "MacOS" / "LMWarden").is_file()

        # Every nested Mach-O and the whole tree verify, as shipped.
        _run(str(REPO / "scripts" / "macos" / "verify-bundle.sh"), str(app))

        # The bundled interpreter runs from the read-only volume and imports
        # the server (-B: nothing is written into the signed bundle).
        # It is the renamed interpreter: the process is named for the product
        # (`ps -c`: the kernel's process name, what Activity Monitor shows),
        # and no python/python3 is left beside it.
        bin_dir = app / "Contents" / "Resources" / "python" / "bin"
        py = bin_dir / "LM Warden Server"
        assert not py.is_symlink()
        assert not [p.name for p in bin_dir.iterdir() if p.name.startswith("python")]
        out = _run(
            str(py),
            "-I",
            "-B",
            "-c",
            "import app.main, os, subprocess, sys; print(sys.prefix); "
            "print(subprocess.run(['ps', '-c', '-o', 'comm=', '-p', str(os.getpid())], "
            "capture_output=True, text=True, check=True).stdout.strip())",
        )
        prefix, comm = out.stdout.strip().splitlines()[-2:]
        assert prefix == str(app / "Contents" / "Resources" / "python")
        assert comm == "LM Warden Server"
    finally:
        # Always detach; a leaked mount blocks the tmp_path cleanup.
        # -force only when a plain detach fails (a busy volume).
        if os.path.ismount(mount):
            if (
                subprocess.run(
                    ["hdiutil", "detach", str(mount)], check=False, capture_output=True, text=True
                ).returncode
                != 0
            ):
                subprocess.run(
                    ["hdiutil", "detach", "-force", str(mount)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
