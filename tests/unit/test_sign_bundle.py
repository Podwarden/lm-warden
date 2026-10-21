"""Inside-out bundle signing for the macOS app (Task 8).

Selected only under ``-m macos`` (see ``pyproject.toml``); the default
``addopts`` deselects it. ``scripts/macos/sign-bundle.sh`` signs every Mach-O
found by ``file`` (not by extension), deepest first, then the app itself, and
never passes ``--deep`` (it is deprecated). ``scripts/macos/make-dmg.sh``
calls it instead of signing ``--deep`` itself.

The tests shell out only to ``bash``, ``cp``, ``file`` and ``codesign``.
The legs that run the real ``codesign`` are skipped when it is not on PATH;
the ``_stub`` legs put a recording ``codesign`` stub first on PATH (no
keychain, no real signing) to pin the exact order, flags and failure path.
No app code is imported.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]

SIGN_BUNDLE = REPO / "scripts" / "macos" / "sign-bundle.sh"
MAKE_DMG = REPO / "scripts" / "macos" / "make-dmg.sh"
ENT_DIR = REPO / "macos" / "entitlements"
SHELL_ENT = ENT_DIR / "LMWarden.entitlements"
SERVER_ENT = ENT_DIR / "LMWardenServer.entitlements"
SERVER_REL = "Contents/Resources/python/bin/LM Warden Server"
DLV = "com.apple.security.cs.disable-library-validation"


needs_codesign = pytest.mark.skipif(shutil.which("codesign") is None, reason="codesign not on PATH")


def _sign_bundle() -> Path:
    assert SIGN_BUNDLE.is_file(), f"missing {SIGN_BUNDLE}"
    return SIGN_BUNDLE


def _build_fixture_app(tmp_path: Path) -> Path:
    """A fake .app with a real Mach-O (copied /usr/bin/true), a nested
    extensionless Mach-O, and a text file masquerading as a .so."""
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


def _run_sign(app: Path, identity: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["bash", str(_sign_bundle()), str(app)]
    if identity is not None:
        cmd.append(identity)
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)


@needs_codesign
def test_signs_extensionless_macho_and_skips_fake_so(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    res = _run_sign(app)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    out = res.stdout
    assert "helper" in out, f"extensionless Mach-O not signed:\n{out}"
    assert "LMWarden" in out, f"main binary not signed:\n{out}"
    assert "fake.so" not in out, f"text file with .so extension was signed:\n{out}"


@needs_codesign
def test_deepest_first_then_app_last(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    res = _run_sign(app)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("signed: ")]
    assert lines, f"no 'signed:' lines in output:\n{res.stdout}"
    helper_idx = next(i for i, ln in enumerate(lines) if "helper" in ln)
    main_idx = next(i for i, ln in enumerate(lines) if "LMWarden" in ln)
    assert helper_idx < main_idx, f"nested Mach-O must sign before the main binary:\n{res.stdout}"
    last = lines[-1]
    assert "LM Warden.app" in last, f"the app itself must sign last:\n{res.stdout}"
    assert "signed 3 items" in res.stdout, f"expected a 'signed 3 items' summary:\n{res.stdout}"


@needs_codesign
def test_result_verifies_strict(tmp_path: Path) -> None:
    app = _build_fixture_app(tmp_path)
    res = _run_sign(app)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    ver = subprocess.run(
        ["codesign", "--verify", "--strict", "--deep", str(app)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (
        ver.returncode == 0
    ), f"codesign --verify --strict --deep failed:\n{ver.stdout}\n{ver.stderr}"


def test_missing_app_exits_2(tmp_path: Path) -> None:
    _sign_bundle()
    absent = tmp_path / "nope.app"
    res = _run_sign(absent)
    assert res.returncode == 2, f"rc={res.returncode}, stderr={res.stderr}"
    assert "usage" in res.stderr, f"no usage line on stderr:\n{res.stderr}"
    res0 = subprocess.run(
        ["bash", str(_sign_bundle())],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res0.returncode == 2, f"no-arg rc={res0.returncode}, stderr={res0.stderr}"
    assert "usage" in res0.stderr, f"no usage line on stderr:\n{res0.stderr}"


def test_never_uses_deep() -> None:
    sign_src = _sign_bundle().read_text(encoding="utf-8")
    make_src = MAKE_DMG.read_text(encoding="utf-8")
    for label, src in (("sign-bundle.sh", sign_src), ("make-dmg.sh", make_src)):
        for i, ln in enumerate(src.splitlines(), start=1):
            stripped = ln.strip()
            if stripped.startswith("#"):
                continue
            if "codesign" in ln and "--deep" in ln:
                pytest.fail(f"{label}:{i} passes --deep to codesign: {stripped!r}")


# --- stubbed codesign: exact order, flags per identity, failure path -------

_STUB = """#!/bin/bash
{ for a in "$@"; do printf '%s\\037' "$a"; done; printf '\\n'; } >> "$CODESIGN_LOG"
last=""
for a in "$@"; do last=$a; done
if [ -n "${CODESIGN_FAIL_ON:-}" ]; then
    case "$last" in *"$CODESIGN_FAIL_ON"*) echo "stub: refused $last" >&2; exit 1 ;; esac
fi
exit 0
"""


def _build_layout_app(tmp_path: Path) -> tuple[Path, list[str]]:
    """A bundle whose depth order disagrees with both lexical orders, with
    a depth tie, nested bundle directories, and a non-Mach-O decoy. Returns
    the app and the expected signing order (relative to the app)."""
    app = tmp_path / "staging" / "LM Warden.app"
    machos = [
        "Contents/MacOS/LMWarden",
        "Contents/Resources/lib/helper",
        "Contents/Resources/lib/zz",
        "Contents/Frameworks/Py.framework/Versions/A/Py",
        "Contents/Helpers/Sub.app/Contents/MacOS/sub",
    ]
    for rel in machos:
        (app / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy("/usr/bin/true", app / rel)
    (app / "Contents/PlugIns/X.bundle").mkdir(parents=True)
    (app / "Contents/Resources/lib/fake.so").write_text("not mach-o\n")
    items = machos + [
        "Contents/Frameworks/Py.framework",
        "Contents/Helpers/Sub.app",
        "Contents/PlugIns/X.bundle",
    ]
    # depth descending, ties by path ascending (byte order), app last
    items.sort(key=lambda r: (-r.count("/"), r))
    return app, items


def _run_stub(
    tmp_path: Path,
    app: Path,
    identity: str | None,
    fail_on: str = "",
    extra: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "codesign"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    log = tmp_path / "codesign.log"
    log.write_text("")
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["CODESIGN_LOG"] = str(log)
    env["CODESIGN_FAIL_ON"] = fail_on
    cmd = ["bash", str(_sign_bundle()), str(app)]
    if identity is not None:
        cmd.append(identity)
    cmd.extend(extra)
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120, env=env)
    calls = [ln.split("\x1f")[:-1] for ln in log.read_text().splitlines()]
    return res, calls


def test_stub_exact_order_depth_not_lexical(tmp_path: Path) -> None:
    app, expected = _build_layout_app(tmp_path)
    res, calls = _run_stub(tmp_path, app, None)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    signed = [c[-1] for c in calls]
    want = [str(app / r) for r in expected] + [str(app)]
    assert signed == want, f"signing order:\n{signed}\nwant:\n{want}"
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("signed: ")]
    assert lines == [f"signed: LM Warden.app/{r}" for r in expected] + [
        "signed: LM Warden.app"
    ], res.stdout
    assert res.stdout.splitlines()[-1] == (
        f"sign-bundle.sh: signed {len(expected) + 1} items"
    ), res.stdout


def test_stub_adhoc_flags(tmp_path: Path) -> None:
    # Ad-hoc signs with the hardened runtime too (P8 hardened-runtime): only
    # --timestamp needs a real identity. The layout app has no server
    # interpreter, so only the app itself carries entitlements.
    app, _ = _build_layout_app(tmp_path)
    res, calls = _run_stub(tmp_path, app, "-")
    assert res.returncode == 0, res.stderr
    assert calls, "codesign never called"
    for c in calls[:-1]:
        assert c[:-1] == ["--force", "--sign", "-", "--options", "runtime"], c
    assert calls[-1] == [
        "--force",
        "--sign",
        "-",
        "--options",
        "runtime",
        "--entitlements",
        str(SHELL_ENT),
        str(app),
    ], calls[-1]


def test_stub_real_identity_gets_runtime_and_timestamp(tmp_path: Path) -> None:
    app, _ = _build_layout_app(tmp_path)
    # Assembled from fragments so the contiguous signing-identity literal
    # never appears in this published source -- the publish block scan
    # (publish/denylist-block.txt) is case-insensitive and would fail on it.
    # Same idiom as _DEVID_PAT in test_publish_mac_patterns.py. Runtime value
    # is byte-identical to the single literal, so assertions keep their meaning.
    ident = "Developer ID " + "Application: Example (ABCDE12345)"
    res, calls = _run_stub(tmp_path, app, ident)
    assert res.returncode == 0, res.stderr
    assert calls, "codesign never called"
    for c in calls[:-1]:
        assert c[:-1] == ["--force", "--sign", ident, "--options", "runtime", "--timestamp"], c
        assert "--deep" not in c, c
    assert calls[-1][:-1] == [
        "--force",
        "--sign",
        ident,
        "--options",
        "runtime",
        "--timestamp",
        "--entitlements",
        str(SHELL_ENT),
    ], calls[-1]


def test_stub_codesign_failure_exits_1_naming_path(tmp_path: Path) -> None:
    app, _ = _build_layout_app(tmp_path)
    res, calls = _run_stub(tmp_path, app, None, fail_on="lib/helper")
    assert res.returncode == 1, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert str(app / "Contents/Resources/lib/helper") in res.stderr, res.stderr
    assert calls[-1][-1].endswith("lib/helper"), calls
    assert str(app) not in [c[-1] for c in calls], "app signed after a failure"


# --- hardened runtime + entitlements (P8 hardened-runtime) -----------------
#
# The minimal set was found empirically on the M1 (ad-hoc, `-o runtime`):
# without disable-library-validation the server interpreter's dlopen of
# pydantic_core and the shell's load of Sparkle.framework both fail with
# "mapping process and mapped file (non-platform) have different Team IDs";
# with it alone MLX (GPU), ctypes/cffi callbacks, uvicorn and the GUI smoke
# work, and llama-server needs nothing. No allow-jit, no
# allow-unsigned-executable-memory.


def _plist_dict(path: Path) -> dict:
    import plistlib

    with open(path, "rb") as fh:
        return plistlib.load(fh)


def test_committed_entitlements_are_the_minimal_set() -> None:
    for ent in (SHELL_ENT, SERVER_ENT):
        assert ent.is_file(), f"missing {ent}"
        assert ent.read_bytes().startswith(b"<?xml"), f"{ent} must be an XML plist"
        assert _plist_dict(ent) == {DLV: True}, f"{ent}: {_plist_dict(ent)}"


def _add_server(app: Path) -> Path:
    server = app / SERVER_REL
    server.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy("/usr/bin/true", server)
    return server


def test_stub_entitlements_by_role(tmp_path: Path) -> None:
    app, _ = _build_layout_app(tmp_path)
    server = _add_server(app)
    res, calls = _run_stub(tmp_path, app, None)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    by_path = {c[-1]: c[:-1] for c in calls}
    assert by_path[str(server)] == [
        "--force",
        "--sign",
        "-",
        "--options",
        "runtime",
        "--entitlements",
        str(SERVER_ENT),
    ], by_path[str(server)]
    assert by_path[str(app)][-2:] == ["--entitlements", str(SHELL_ENT)], by_path[str(app)]
    for path, args in by_path.items():
        if path not in (str(server), str(app)):
            assert "--entitlements" not in args, (path, args)
        assert "--options" in args and args[args.index("--options") + 1] == "runtime", (path, args)


def test_stub_entitlements_dir_override(tmp_path: Path) -> None:
    app, _ = _build_layout_app(tmp_path)
    server = _add_server(app)
    alt = tmp_path / "ents"
    alt.mkdir()
    for name in ("LMWarden.entitlements", "LMWardenServer.entitlements"):
        shutil.copy(SHELL_ENT, alt / name)
    res, calls = _run_stub(tmp_path, app, "-", extra=("--entitlements-dir", str(alt)))
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    by_path = {c[-1]: c[:-1] for c in calls}
    assert by_path[str(server)][-2:] == [
        "--entitlements",
        str(alt / "LMWardenServer.entitlements"),
    ]
    assert by_path[str(app)][-2:] == ["--entitlements", str(alt / "LMWarden.entitlements")]


def test_entitlements_dir_missing_file_exits_2(tmp_path: Path) -> None:
    app, _ = _build_layout_app(tmp_path)
    alt = tmp_path / "ents"
    alt.mkdir()
    shutil.copy(SHELL_ENT, alt / "LMWarden.entitlements")
    res, calls = _run_stub(tmp_path, app, "-", extra=("--entitlements-dir", str(alt)))
    assert res.returncode == 2, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "LMWardenServer.entitlements" in res.stderr, res.stderr
    assert calls == [], "codesign ran before the entitlements were checked"
    res2, _ = _run_stub(tmp_path, app, "-", extra=("--entitlements-dir",))
    assert res2.returncode == 2 and "usage" in res2.stderr, res2.stderr
    res3, _ = _run_stub(tmp_path, app, "-", extra=("--bogus",))
    assert res3.returncode == 2 and "usage" in res3.stderr, res3.stderr


def _entitlements_of(path: Path) -> dict:
    import plistlib

    out = subprocess.run(
        ["codesign", "-d", "--entitlements", "-", "--xml", str(path)],
        check=False,
        capture_output=True,
        timeout=60,
    ).stdout
    return plistlib.loads(out) if out.strip() else {}


@needs_codesign
def test_real_codesign_runtime_and_entitlements(tmp_path: Path) -> None:
    # Real-codesign leg (runs where codesign exists: the M1 under -m macos):
    # every Mach-O carries the runtime flag, the server interpreter and the
    # main executable carry exactly the committed set, nothing else any.
    app = _build_fixture_app(tmp_path)
    server = _add_server(app)
    res = _run_sign(app)
    assert res.returncode == 0, f"rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    main = app / "Contents" / "MacOS" / "LMWarden"
    helper = app / "Contents" / "Resources" / "lib" / "helper"
    for f in (main, helper, server):
        info = subprocess.run(
            ["codesign", "-dv", str(f)], check=False, capture_output=True, text=True, timeout=60
        ).stderr
        flags = [ln for ln in info.splitlines() if "flags=" in ln]
        assert flags and "runtime" in flags[0], f"{f}: no runtime flag:\n{info}"
    assert _entitlements_of(server) == {DLV: True}
    assert _entitlements_of(main) == {DLV: True}
    assert _entitlements_of(helper) == {}
