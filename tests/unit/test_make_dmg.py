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

import fnmatch
import os
import plistlib
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]

MAKE_DMG = REPO / "scripts" / "macos" / "make-dmg.sh"
PRUNE_LIST = REPO / "scripts" / "macos" / "python-prune.txt"


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


def _parse_prune_rules(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """The rules of python-prune.txt: (kind, operands), in list order.

    Same two kinds the bundle applies (bundle-python.sh step 5a):
    ``path <glob>`` removes every <python-root>/<glob> match, a bash glob
    whose ``*`` spans one path segment (``?`` and ``[set]`` like bash's);
    ``tree <dir> <name>`` removes every file or dir named <name> anywhere
    under <python-root>/<dir>. ``#`` comments and blank lines are ignored.
    """
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, rest = line.partition(" ")
        rules.append((kind, rest.split()))
    return rules


def _prune_rule_matches(rule: tuple[str, tuple[str, ...]], pyroot: Path) -> list[str]:
    """Rel-paths under <pyroot> the rule would remove.

    Mirrors prune_python: for ``path`` the match is <root>/<glob> anchored at
    the root, with ``*`` spanning one segment (``Path.glob``: the matches at
    the pattern's own depth, which is what the prune's awk segment count
    keeps); for ``tree`` a <root>/<dir> walk on a <name> basename match,
    skipping transformers' models/auto/ as the prune's ``! -path
    '*/models/auto/*'`` does (tree rules only, as in the prune).
    """

    # The prune's `! -path '*/models/auto/*'`: an adjacent models/auto pair of
    # whole components (auto_backup/ is NOT kept).
    def _in_auto(rel: str) -> bool:
        return "/models/auto/" in f"/{rel}/"

    kind, ops = rule
    if kind == "path":
        (glob,) = ops
        return sorted(str(p.relative_to(pyroot)) for p in pyroot.glob(glob))
    if kind == "tree":
        dir_, name = ops
        base = pyroot / dir_
        if not base.is_dir():
            return []
        out = []
        for p in [base, *base.rglob("*")]:
            if fnmatch.fnmatchcase(p.name, name):
                if not _in_auto(str(p.relative_to(pyroot))):
                    out.append(str(p.relative_to(pyroot)))
        return sorted(set(out))
    raise ValueError(f"unknown prune rule kind: {kind!r}")


def _plant(rel: str) -> str:
    """A concrete name for a prune glob: each wildcard becomes one letter."""
    return re.sub(r"\[[^]]*\]|[*?]", "x", rel)


def test_prune_matcher_agrees_with_the_prune(tmp_path: Path) -> None:
    # The full leg's no-leftover check is only as good as its matcher, so pin
    # the matcher here, fast, against the prune itself: plant one match per
    # python-prune.txt rule (plus a nested one per tree rule) and controls the
    # list must not touch, then (1) the matcher sees every planted path of
    # every rule -- a rule kind it cannot see would let the full leg pass an
    # unpruned bundle; (2) none of the controls; (3) after the real
    # bundle-python.sh --prune over the same tree it sees nothing and the
    # controls survive.
    root = tmp_path / "python"
    rules = _parse_prune_rules(PRUNE_LIST.read_text(encoding="utf-8"))
    planted: dict[tuple[str, str], set[str]] = {}
    for kind, ops in rules:
        if kind == "path":
            rels = {_plant(ops[0])}
        else:
            dir_, name = ops
            rels = {f"{dir_}/{_plant(name)}", f"{dir_}/pkg/sub/{_plant(name)}"}
        planted[(kind, " ".join(ops))] = rels
        for rel in rels:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("x", encoding="utf-8")
    controls = [
        "lib/python3.12/os.py",
        "lib/python3.12/site-packages/x/include",
        "lib/python3.12/site-packages/x/lib/python3.12/tkinter/__init__.py",
        "lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py",
        "lib/python3.12/site-packages/transformers/models/bert/configuration_bert.py",
    ]
    for rel in controls:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")

    for kind, ops in rules:
        got = set(_prune_rule_matches((kind, ops), root))
        want = planted[(kind, " ".join(ops))]
        assert want <= got, f"matcher misses {sorted(want - got)} for rule {kind} {' '.join(ops)}"
        assert (
            not got - want
        ), f"matcher over-matches {sorted(got - want)} for {kind} {' '.join(ops)}"

    _run(str(REPO / "scripts" / "macos" / "bundle-python.sh"), "--prune", str(root))
    left = [(k, o, m) for k, o in rules for m in _prune_rule_matches((k, o), root)]
    assert not left, f"matcher still sees after the real prune: {left[:20]}"
    for rel in controls:
        assert (root / rel).is_file(), f"control {rel} is gone"


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


def test_dmg_is_packed_as_ulmo(tmp_path: Path) -> None:
    # The shipped DMG is LZMA-compressed (ULMO): read-only, opens on macOS 14
    # (the app's minimum), and far smaller than the old zlib UDZO (the footprint
    # plan measured 128.3 MB vs 341.4 MB for the same app). Pin the format the
    # image actually carries, not just that a DMG exists: a UDZO image is red.
    app = _tiny_app(tmp_path)
    out_dmg = tmp_path / "out.dmg"
    proc = _make_dmg(MAKE_DMG.parent, app, "0.0.0-test", out_dmg)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert out_dmg.is_file(), "no DMG written"

    info = _run("hdiutil", "imageinfo", "-plist", str(out_dmg)).stdout
    format_ = plistlib.loads(info.encode("utf-8"))["Format"]
    assert format_ == "ULMO", f"hdiutil imageinfo Format is {format_!r}, expected ULMO"


def test_dmg_size_line_matches_file_size(tmp_path: Path) -> None:
    # After writing the DMG, make-dmg.sh prints exactly one line
    # `make-dmg.sh: wrote <abs> (<N> bytes)` whose N is the file's size. Pin the
    # exact byte count (not a range): the number must equal the bytes on disk.
    app = _tiny_app(tmp_path)
    out_dmg = tmp_path / "out.dmg"
    proc = _make_dmg(MAKE_DMG.parent, app, "0.0.0-test", out_dmg)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert out_dmg.is_file(), "no DMG written"

    size_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("make-dmg.sh: wrote ")]
    assert len(size_lines) == 1, f"expected exactly one 'wrote' line, got {size_lines!r}"
    m = re.match(r"make-dmg\.sh: wrote (.+) \((\d+) bytes\)$", size_lines[0])
    assert m, f"size line has no '<N> bytes' tail: {size_lines[0]!r}"
    path_in_line = m.group(1)
    n = int(m.group(2))
    # The path named is the DMG written, and the count is its real size.
    assert Path(path_in_line).resolve() == out_dmg.resolve(), (path_in_line, out_dmg)
    assert n == out_dmg.stat().st_size, f"line says {n} bytes, file is {out_dmg.stat().st_size}"


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

    # The shipped image is LZMA (ULMO) and far smaller than the old zlib UDZO:
    # the untrimmed ULMO was measured at 128.3 MB, and this one adds only the
    # web UI and llama-server on top. A 341 MB UDZO image is red; so is any
    # image that stops fitting comfortably under the 200 MB bound.
    dmg_bytes = out_dmg.stat().st_size
    assert dmg_bytes < 200_000_000, (
        f"DMG is {dmg_bytes} bytes; ULMO must stay under 200 MB "
        f"(measured 128.3 MB untrimmed; UDZO was 341 MB)"
    )
    image_format = plistlib.loads(
        _run("hdiutil", "imageinfo", "-plist", str(out_dmg)).stdout.encode("utf-8")
    )["Format"]
    assert (
        image_format == "ULMO"
    ), f"hdiutil imageinfo Format is {image_format!r}, expected ULMO ({dmg_bytes} bytes)"

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

        # (c) The llama-server pair and the renamed interpreter are in the
        # mounted app's python/bin. The .build stamp is load-bearing: on the
        # apple profile app/runtime/backends/llamacpp/version.py reads it (it
        # is "unknown" without the file, never an error).
        pybin = app / "Contents" / "Resources" / "python" / "bin"
        for name in ("llama-server", "llama-server.build", "LM Warden Server"):
            assert (pybin / name).is_file(), (
                f"mounted app is missing python/bin/{name} "
                f"(llama-server.build is read by app/runtime/backends/llamacpp/version.py)"
            )

        # (b) The M1 proof of the P11 trim: python-prune.txt (the list
        # bundle-python.sh step 5a applies) matches NOTHING left in the mounted
        # python tree. A rule that matches is the exact stale-list state the
        # build refuses -- here it means the prune never ran (or re-added what
        # it removed). A green full leg on an unpruned bundle would be a lie.
        prune_root = app / "Contents" / "Resources" / "python"
        leftover = []
        for rule in _parse_prune_rules(PRUNE_LIST.read_text(encoding="utf-8")):
            for rel in _prune_rule_matches(rule, prune_root):
                leftover.append((rule[0], " ".join(rule[1]), rel))
        assert not leftover, (
            f"{len(leftover)} python-prune.txt rule(s) still match inside the mounted "
            f"python tree -- the P11 prune did not run (first 20 of "
            f"{len(leftover)}): " + "; ".join(f"{k} {o} -> {m}" for k, o, m in leftover[:20])
        )

        # (a) The import net, run by the MOUNTED read-only interpreter: every
        # module the server needs imports -- the build's step 7 ran this at
        # bundle time, and the live leg re-runs it on the image that ships, so
        # a prune that dropped a shipped module turns the leg red here.
        mounted_interp = pybin / "LM Warden Server"
        # check=False: a failing guard exits 1, and its FAIL lines name the
        # module the prune dropped -- they belong in the failure message.
        guard_out = subprocess.run(
            [
                str(mounted_interp),
                "-I",
                "-B",
                str(REPO / "scripts" / "macos" / "bundle-import-guard.py"),
            ],
            capture_output=True,
            text=True,
        )
        guard_lines = guard_out.stdout.rstrip().splitlines()
        assert (
            guard_out.returncode == 0
            and guard_lines
            and guard_lines[-1].startswith("IMPORT-GUARD: ok ")
        ), (
            f"the import guard did not pass on the mounted bundle (exit {guard_out.returncode}): "
            f"{guard_out.stdout[-800:]!r} {guard_out.stderr[-400:]!r}"
        )

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
