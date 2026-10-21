"""Verify a signed .app before shipping (P8, follows sign-bundle.sh).

Selected only under ``-m macos`` (see ``pyproject.toml``); the default
``addopts`` deselects it. ``scripts/macos/verify-bundle.sh`` finds every Mach-O
under ``<app>/Contents`` the same way ``sign-bundle.sh`` does (by ``file -b``
output containing "Mach-O", not by extension) and runs ``codesign --verify
--strict`` on each, in path order (LC_ALL=C), then
``codesign --verify --strict --deep`` on the app itself. A per-file verify of
a nested Mach-O checks the *outer* bundle's seal, so a signed bundle passes
only when every file it signed is validly signed; the ``--deep`` verify then
walks the whole tree. Verification uses ``--deep``; only *signing* with
``--deep`` is forbidden.

The tests shell out only to ``bash``, ``cp``, ``codesign`` and the real
``sign-bundle.sh`` to build a known-good signed fixture. The legs that run the
real ``codesign`` are skipped when it is not on PATH; no app code is imported.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]

SIGN_BUNDLE = REPO / "scripts" / "macos" / "sign-bundle.sh"


def _verify_bundle() -> Path:
    assert (
        REPO / "scripts" / "macos" / "verify-bundle.sh"
    ).is_file(), f"missing {REPO / 'scripts' / 'macos' / 'verify-bundle.sh'}"
    return REPO / "scripts" / "macos" / "verify-bundle.sh"


needs_codesign = pytest.mark.skipif(shutil.which("codesign") is None, reason="codesign not on PATH")


def _build_fixture_app(tmp_path: Path) -> Path:
    """A fake .app with two real Mach-Os (copied /usr/bin/true), a nested
    extensionless one, and a text file masquerading as a .so. Mirrors
    test_sign_bundle's fixture so sign-bundle.sh can sign it known-good."""
    staging = tmp_path / "staging"
    app_root = staging / "LM Warden.app"
    macos_dir = app_root / "Contents" / "MacOS"
    lib_dir = app_root / "Contents" / "Resources" / "lib"
    macos_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)
    shutil.copy("/usr/bin/true", macos_dir / "LMWarden")
    shutil.copy("/usr/bin/true", lib_dir / "helper")
    (lib_dir / "fake.so").write_text("definitely not mach-o\n")
    # codesign only treats a directory as an app bundle when it carries a
    # valid Info.plist, so write one (the same shape write-info-plist.sh
    # produces) or the final `codesign <app>` step is refused.
    (app_root / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        "  <key>CFBundleIdentifier</key><string>com.podwarden.lmwarden</string>\n"
        "  <key>CFBundleExecutable</key><string>LMWarden</string>\n"
        "  <key>CFBundlePackageType</key><string>APPL</string>\n"
        "</dict>\n</plist>\n"
    )
    return app_root


def _sign(app: Path) -> subprocess.CompletedProcess:
    assert SIGN_BUNDLE.is_file(), f"missing {SIGN_BUNDLE}"
    return subprocess.run(
        ["bash", str(SIGN_BUNDLE), str(app)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_verify(app: Path) -> subprocess.CompletedProcess:
    cmd = ["bash", str(_verify_bundle()), str(app)]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)


def _sign_then_verify(tmp_path: Path) -> tuple[Path, subprocess.CompletedProcess]:
    """Build the two-Mach-O fixture, sign it known-good, then verify it."""
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    return app, _run_verify(app)


@needs_codesign
def test_signed_bundle_passes(tmp_path: Path) -> None:
    app, res = _sign_then_verify(tmp_path)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "verified 2" in res.stdout, f"expected 'verified 2' (the two Mach-Os):\n{res.stdout}"
    parent = app.parent
    # The summary is the last line and is relative to the app's parent dir.
    assert res.stdout.splitlines()[-1] == "verify-bundle.sh: verified 2 Mach-O + bundle", res.stdout
    assert (
        str(parent) not in res.stdout
    ), f"summary must use a path relative to {parent}:\n{res.stdout}"


def _codesign(*args: str) -> None:
    res = subprocess.run(
        ["codesign", *args], check=False, capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, f"codesign {args} rc={res.returncode}\n{res.stdout}\n{res.stderr}"


@needs_codesign
def test_unsigned_nested_macho_fails(tmp_path: Path) -> None:
    # Top risk 1: a nested Mach-O left unsigned inside a bundle whose own seal
    # is valid. Sign the app, strip the nested helper's signature AFTER
    # signing, then re-seal only the outer app (as a signer that skipped the
    # nested object would). `codesign --verify --strict --deep <app>` PASSES
    # on this bundle (a loose Mach-O under Resources is sealed as data), so
    # only the per-file pass can catch it, and it must name the helper.
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    lib = app / "Contents" / "Resources" / "lib" / "helper"
    _codesign("--remove-signature", str(lib))
    _codesign("--force", "--sign", "-", str(app))
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: not validly signed: LM Warden.app/Contents/Resources/lib/helper"
        in res.stderr
    ), res.stderr
    assert "verified" not in res.stdout, res.stdout


@needs_codesign
def test_nested_macho_stripped_after_seal_fails(tmp_path: Path) -> None:
    # The same strip without the re-seal breaks the app's resource seal. The
    # main executable's own verify checks that seal and sorts first
    # (Contents/MacOS < Contents/Resources), so it is the path named.
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    _codesign("--remove-signature", str(app / "Contents" / "Resources" / "lib" / "helper"))
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: not validly signed: LM Warden.app/Contents/MacOS/LMWarden" in res.stderr
    ), res.stderr


@needs_codesign
def test_modified_main_binary_fails(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    # Append one byte to the main binary *after* the app is signed.
    main = app / "Contents" / "MacOS" / "LMWarden"
    with open(main, "ab") as fh:
        fh.write(b"\x00")
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: not validly signed: LM Warden.app/Contents/MacOS/LMWarden" in res.stderr
    ), res.stderr


@needs_codesign
def test_broken_bundle_seal_names_the_app(tmp_path: Path) -> None:
    # Delete the main executable after signing: the one Mach-O left (the
    # nested helper) still verifies on its own, so the per-file pass is clean
    # and only the final bundle verify fails. The path named is the app
    # itself, relative to its parent dir.
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    (app / "Contents" / "MacOS" / "LMWarden").unlink()
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        res.stderr.splitlines()[-1] == "verify-bundle.sh: not validly signed: LM Warden.app"
    ), res.stderr
    assert "Contents/" not in res.stderr.splitlines()[-1], res.stderr


@needs_codesign
def test_text_file_named_so_is_ignored(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    # A third, text file named like a dylib: not a Mach-O, so it is not
    # counted (the count stays 2) and the signed bundle still passes.
    (app / "Contents" / "Resources" / "lib" / "extra.so").write_text("not mach-o either\n")
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    res = _run_verify(app)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "verified 2" in res.stdout, f"a text .so was counted:\n{res.stdout}"
    assert res.stdout.splitlines()[-1] == "verify-bundle.sh: verified 2 Mach-O + bundle", res.stdout


def test_missing_app_exits_2(tmp_path: Path) -> None:
    _verify_bundle()
    absent = tmp_path / "nope.app"
    res = _run_verify(absent)
    assert res.returncode == 2, f"rc={res.returncode}, stderr={res.stderr}"
    assert "usage: verify-bundle.sh <app>" in res.stderr, f"no usage line on stderr:\n{res.stderr}"
    res0 = subprocess.run(
        ["bash", str(_verify_bundle())],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res0.returncode == 2, f"no-arg rc={res0.returncode}, stderr={res0.stderr}"
    assert (
        "usage: verify-bundle.sh <app>" in res0.stderr
    ), f"no usage line on stderr:\n{res0.stderr}"


@needs_codesign
def test_space_in_app_name(tmp_path: Path) -> None:
    app, res = _sign_then_verify(tmp_path)
    assert app.name == "LM Warden.app", f"fixture is not space-named: {app.name}"
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "verified 2" in res.stdout, res.stdout


# --- hardened runtime + entitlement set (P8 hardened-runtime) --------------
#
# After the per-file validity pass, every Mach-O must carry the hardened
# runtime flag, and its entitlements must be exactly the committed set for its
# role: the main executable macos/entitlements/LMWarden.entitlements, the
# server interpreter (Contents/Resources/python/bin/LM Warden Server)
# macos/entitlements/LMWardenServer.entitlements, everything else none.

ENT_DIR = REPO / "macos" / "entitlements"
SHELL_ENT = ENT_DIR / "LMWarden.entitlements"
SERVER_REL = "Contents/Resources/python/bin/LM Warden Server"


def _add_server(app: Path) -> Path:
    server = app / SERVER_REL
    server.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy("/usr/bin/true", server)
    return server


def _reseal(app: Path) -> None:
    # Re-sign only the outer app the way sign-bundle.sh does, so its seal and
    # its main executable are valid again over whatever was changed inside.
    _codesign(
        "--force", "--sign", "-", "--options", "runtime", "--entitlements", str(SHELL_ENT), str(app)
    )


def _plist(tmp_path: Path, name: str, keys: list[str]) -> Path:
    body = "".join(f"<key>{k}</key><true/>" for k in keys)
    p = tmp_path / name
    p.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        f'<plist version="1.0"><dict>{body}</dict></plist>\n'
    )
    return p


DLV = "com.apple.security.cs.disable-library-validation"


@needs_codesign
def test_runtime_and_entitlements_summary(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    _add_server(app)
    sig = _sign(app)
    assert sig.returncode == 0, f"sign failed rc={sig.returncode}\n{sig.stdout}\n{sig.stderr}"
    res = _run_verify(app)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    lines = res.stdout.splitlines()
    assert (
        "verify-bundle.sh: hardened runtime on 3 Mach-O, entitlements as committed (2 pinned)"
        in lines
    ), res.stdout
    assert lines[-1] == "verify-bundle.sh: verified 3 Mach-O + bundle", res.stdout


@needs_codesign
def test_nested_macho_without_runtime_fails(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, sig.stderr
    _codesign("--force", "--sign", "-", str(app / "Contents" / "Resources" / "lib" / "helper"))
    _reseal(app)
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: no hardened runtime: LM Warden.app/Contents/Resources/lib/helper"
        in res.stderr
    ), res.stderr
    assert "verified" not in res.stdout, res.stdout


@needs_codesign
def test_main_executable_without_entitlements_fails(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, sig.stderr
    _codesign("--force", "--sign", "-", "--options", "runtime", str(app))
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: entitlements differ from LMWarden.entitlements: "
        "LM Warden.app/Contents/MacOS/LMWarden" in res.stderr
    ), res.stderr


@needs_codesign
def test_server_with_extra_entitlement_fails(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    server = _add_server(app)
    sig = _sign(app)
    assert sig.returncode == 0, sig.stderr
    jit = _plist(tmp_path, "jit.plist", [DLV, "com.apple.security.cs.allow-jit"])
    _codesign(
        "--force", "--sign", "-", "--options", "runtime", "--entitlements", str(jit), str(server)
    )
    _reseal(app)
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: entitlements differ from LMWardenServer.entitlements: "
        f"LM Warden.app/{SERVER_REL}" in res.stderr
    ), res.stderr


@needs_codesign
def test_other_macho_with_entitlements_fails(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    sig = _sign(app)
    assert sig.returncode == 0, sig.stderr
    helper = app / "Contents" / "Resources" / "lib" / "helper"
    _codesign(
        "--force",
        "--sign",
        "-",
        "--options",
        "runtime",
        "--entitlements",
        str(SHELL_ENT),
        str(helper),
    )
    _reseal(app)
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: entitlements differ from none: "
        "LM Warden.app/Contents/Resources/lib/helper" in res.stderr
    ), res.stderr


@needs_codesign
def test_server_with_the_entitlement_set_false_fails(tmp_path: Path) -> None:
    # The value is compared, not just the key: disable-library-validation
    # <false/> is not the committed <true/>.
    app = _build_fixture_app(tmp_path)
    server = _add_server(app)
    sig = _sign(app)
    assert sig.returncode == 0, sig.stderr
    off = tmp_path / "off.plist"
    off.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        f'<plist version="1.0"><dict><key>{DLV}</key><false/></dict></plist>\n'
    )
    _codesign(
        "--force", "--sign", "-", "--options", "runtime", "--entitlements", str(off), str(server)
    )
    _reseal(app)
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        "verify-bundle.sh: entitlements differ from LMWardenServer.entitlements: "
        f"LM Warden.app/{SERVER_REL}" in res.stderr
    ), res.stderr


@needs_codesign
def test_python_bundle_without_the_server_interpreter_fails(tmp_path: Path) -> None:
    # A renamed interpreter (here the pre-rename python3.12) must not fall into
    # the "none" role and pass: a bundle that embeds Python needs the server
    # interpreter at SERVER_REL.
    app = _build_fixture_app(tmp_path)
    stray = app / "Contents" / "Resources" / "python" / "bin" / "python3.12"
    stray.parent.mkdir(parents=True)
    shutil.copy("/usr/bin/true", stray)
    sig = _sign(app)
    assert sig.returncode == 0, sig.stderr
    res = _run_verify(app)
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (
        f"verify-bundle.sh: server interpreter missing or not Mach-O: LM Warden.app/{SERVER_REL}"
        in res.stderr
    ), res.stderr


def test_role_paths_match_the_bundled_interpreter_name() -> None:
    # sign-bundle.sh and verify-bundle.sh key the server role on the path that
    # bundle-python.sh renames the interpreter to (SERVER_EXE, itself pinned to
    # ShellAppDelegate.bundledInterpreterName in test_bundle_python.py).
    bundle_python = (REPO / "scripts" / "macos" / "bundle-python.sh").read_text()
    m = re.search(r'^SERVER_EXE="([^"]+)"$', bundle_python, re.M)
    assert m, "bundle-python.sh has no SERVER_EXE line"
    assert SERVER_REL == f"Contents/Resources/python/bin/{m.group(1)}"
    for script in ("sign-bundle.sh", "verify-bundle.sh"):
        text = (REPO / "scripts" / "macos" / script).read_text()
        assert f'server_path="$contents/Resources/python/bin/{m.group(1)}"' in text, script


def test_bad_option_and_missing_entitlements_dir_exit_2(tmp_path: Path) -> None:
    app = tmp_path / "X.app"
    (app / "Contents").mkdir(parents=True)
    missing = ["--entitlements-dir", str(tmp_path / "none")]
    for extra in (["--bogus"], ["--entitlements-dir"], missing):
        res = subprocess.run(
            ["bash", str(_verify_bundle()), str(app), *extra],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert res.returncode == 2, (extra, res.returncode, res.stderr)
        assert "verify-bundle.sh:" in res.stderr or "usage" in res.stderr, res.stderr
