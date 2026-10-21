"""Unit tests for ``scripts/macos/bundle-python.sh`` (P8, embedded Python).

Not marked ``macos``: every leg here stays off the network and off the real
interpreter -- usage, ``--print-plan``, the checksum gate on a cached tarball
under ``--offline``, the console-script relocation hook and the interpreter
rename hook -- so it needs only
``bash``, coreutils and ``awk`` and runs in linux CI too. The full build (fetch,
pip install, sign, verify, import) is the ``make macos-bundle-python`` run on
the M1. The app-from-HEAD legs also need ``git`` and the ``make macos-dmg`` legs
need ``make``; each skips with a reason where that tool is absent (the CI deps
image has neither), like the farm tests do.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "bundle-python.sh"
# The embedded interpreter's product name (Metal design s3).
SERVER_EXE = "LM Warden Server"

needs_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="needs git (absent from the CI deps image; runs on the bare host and the M1)",
)
needs_make = pytest.mark.skipif(
    shutil.which("make") is None,
    reason="needs make (absent from the CI deps image; runs on the bare host and the M1)",
)


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
    )


def _app(tmp_path: Path) -> Path:
    app = tmp_path / "LM Warden.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    return app


def _plan(*args: str) -> dict[str, str]:
    proc = _run(*args, "--print-plan")
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


# --- usage --------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--cache-dir"],
        ["--bogus", "x"],
        ["a.app", "b.app"],
    ],
)
def test_usage_errors_exit_2(args: list[str]) -> None:
    proc = _run(*args)
    assert proc.returncode == 2
    assert "usage: bundle-python.sh" in proc.stderr


def test_not_a_bundle_exits_2(tmp_path: Path) -> None:
    proc = _run(str(tmp_path))
    assert proc.returncode == 2
    assert "not an app bundle" in proc.stderr


def test_missing_requirements_exits_2(tmp_path: Path) -> None:
    proc = _run(str(_app(tmp_path)), "--requirements", str(tmp_path / "nope.txt"))
    assert proc.returncode == 2
    assert "requirements file not found" in proc.stderr


# --- the plan: pin, layout, pip flags ------------------------------------------


def test_plan_pins_one_exact_https_asset(tmp_path: Path) -> None:
    plan = _plan(str(_app(tmp_path)))
    assert plan["url"].startswith(
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
    )
    # one exact release asset: a dated tag, a 3.12 patch version, arm64 macOS, install_only
    assert re.search(
        r"/(\d{8})/cpython-3\.12\.\d+%2B\1-aarch64-apple-darwin-install_only\.tar\.gz$",
        plan["url"],
    ), plan["url"]
    assert re.fullmatch(r"[0-9a-f]{64}", plan["sha256"])


def test_plan_layout_matches_the_swift_launch_path(tmp_path: Path) -> None:
    app = _app(tmp_path)
    plan = _plan(str(app))
    # ShellAppDelegate.bundledPythonPath():
    #   <bundle>/Contents/Resources/python/bin/ + bundledInterpreterName
    assert plan["dest"] == f"{app}/Contents/Resources/python"
    assert plan["interpreter_name"] == SERVER_EXE
    assert plan["python"] == f"{app}/Contents/Resources/python/bin/{SERVER_EXE}"
    assert plan["site_packages"] == f"{app}/Contents/Resources/python/lib/python3.12/site-packages"
    swift = (REPO / "macos" / "Sources" / "LMWarden" / "ShellAppDelegate.swift").read_text()
    assert f'bundledInterpreterName = "{SERVER_EXE}"' in swift
    assert '"/Contents/Resources/python/bin/" + bundledInterpreterName' in swift
    # The kernel keeps MAXCOMLEN (16) bytes of a process name: all of it.
    assert len(SERVER_EXE.encode()) == 16


def test_launch_and_scripts_never_write_bytecode() -> None:
    # -B on the server argv: PYTHONDONTWRITEBYTECODE is ignored under -I.
    spec = (REPO / "macos" / "Sources" / "LMWarden" / "ServerLaunchSpec.swift").read_text()
    assert '"-I", "-B", "-m", "uvicorn", "app.main:app",' in spec


def test_plan_pip_is_hash_locked_wheels_only_no_deps(tmp_path: Path) -> None:
    plan = _plan(str(_app(tmp_path)))
    pip = plan["pip"].split()
    assert pip[:4] == ["python3", "-I", "-m", "pip"]
    for flag in ("--require-hashes", "--no-deps", "--isolated", "--no-compile"):
        assert flag in pip, flag
    assert pip[pip.index("--only-binary") + 1] == ":all:"
    assert pip[pip.index("-r") + 1] == str(REPO / "requirements-macos.txt")
    assert plan["requirements"] == str(REPO / "requirements-macos.txt")


def test_plan_defaults_and_overrides_cache_dir(tmp_path: Path) -> None:
    app = _app(tmp_path)
    default = _plan(str(app))
    assert default["tarball"].startswith(str(REPO / ".cache" / "macos-python") + "/")
    cache = tmp_path / "cache"
    plan = _plan(str(app), "--cache-dir", str(cache))
    assert Path(plan["tarball"]).parent == cache
    assert plan["tarball"].endswith("-aarch64-apple-darwin-install_only.tar.gz")
    assert plan["pip"].split()[plan["pip"].split().index("--cache-dir") + 1] == f"{cache}/pip"
    assert not cache.exists(), "--print-plan must not touch the filesystem"


def test_plan_relative_cache_dir_is_made_absolute(tmp_path: Path) -> None:
    app = _app(tmp_path)
    proc = _run(str(app), "--cache-dir", "rel-cache", "--print-plan", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    plan = dict(line.split("=", 1) for line in proc.stdout.splitlines())
    assert Path(plan["tarball"]).parent == tmp_path / "rel-cache"


def test_requirements_lock_is_hashed() -> None:
    """--require-hashes only holds if every pin in the lock carries a hash."""
    text = (REPO / "requirements-macos.txt").read_text()
    pins = re.findall(r"^([A-Za-z0-9_.\-\[\]]+==\S+)", text, flags=re.M)
    assert pins
    blocks = re.split(r"^(?=[A-Za-z0-9_.\-\[\]]+==)", text, flags=re.M)[1:]
    assert len(blocks) == len(pins)
    for block in blocks:
        assert "--hash=sha256:" in block, block.splitlines()[0]


# --- the checksum gate -----------------------------------------------------------


def test_offline_without_cached_tarball_fails(tmp_path: Path) -> None:
    app = _app(tmp_path)
    proc = _run(str(app), "--offline", "--cache-dir", str(tmp_path / "cache"))
    assert proc.returncode == 1
    assert "--offline" in proc.stderr and "is not in" in proc.stderr
    assert not (app / "Contents" / "Resources" / "python").exists()


def test_offline_corrupt_cached_tarball_is_refused(tmp_path: Path) -> None:
    app = _app(tmp_path)
    cache = tmp_path / "cache"
    tarball = Path(_plan(str(app), "--cache-dir", str(cache))["tarball"])
    cache.mkdir()
    tarball.write_bytes(b"not the pinned interpreter\n")
    proc = _run(str(app), "--offline", "--cache-dir", str(cache))
    assert proc.returncode == 1
    assert "sha256 mismatch" in proc.stderr
    assert not (app / "Contents" / "Resources" / "python").exists()
    assert not list(cache.glob("stage.*")), "nothing is staged from an unverified tarball"


# --- relocation of pip's console scripts -------------------------------------------


def _bin_dir(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "python" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / SERVER_EXE
    fake.write_text('#!/bin/sh\necho "PY $*"\n')
    fake.chmod(0o755)
    return bin_dir


def test_relocate_rewrites_both_distlib_header_forms(tmp_path: Path) -> None:
    bin_dir = _bin_dir(tmp_path)
    old = "/build/stage.abc"
    plain = bin_dir / "mlx_lm.server"
    plain.write_text(
        f"#!{old}/python/bin/python3.12\n# -*- coding: utf-8 -*-\nimport sys\nprint(sys.argv)\n"
    )
    spaced = bin_dir / "uvicorn"
    spaced.write_text(
        "#!/bin/sh\n"
        f'\'\'\'exec\' "{old} x/python/bin/python3.12" "$0" "$@"\n'
        "' '''\n"
        "import sys\n"
    )
    untouched = bin_dir / "idle3"
    untouched.write_text("#!/usr/bin/env python3\nprint(1)\n")

    proc = _run("--relocate-scripts", str(bin_dir), old)
    assert proc.returncode == 0, proc.stderr
    assert "relocated 2 scripts" in proc.stdout

    for script in (plain, spaced):
        text = script.read_text()
        assert old not in text
        assert text.startswith("#!/bin/sh\n'''exec' ")
        assert f"/'{SERVER_EXE}' -I -B \"$0\"" in text.splitlines()[1]
        assert text.splitlines()[2] == "' '''"
        assert text.rstrip().endswith("import sys") or "print(sys.argv)" in text
        assert os.access(script, os.X_OK)
        compile(text, str(script), "exec")  # still valid Python after the rewrite
    assert untouched.read_text() == "#!/usr/bin/env python3\nprint(1)\n"


def test_relocated_script_execs_the_sibling_interpreter(tmp_path: Path) -> None:
    bin_dir = _bin_dir(tmp_path)
    script = bin_dir / "mlx_lm.server"
    script.write_text("#!/somewhere/else/python3.12\nimport sys\n")
    assert _run("--relocate-scripts", str(bin_dir), "/somewhere/else").returncode == 0
    # Move the whole tree -- under a spaced name, as in "LM Warden.app" -- the
    # script must follow it, through a symlink too.
    moved = tmp_path / "LM Warden.app" / "python"
    moved.parent.mkdir()
    (tmp_path / "python").rename(moved)
    link = tmp_path / "link"
    link.symlink_to(moved / "bin" / "mlx_lm.server")
    for argv0 in (moved / "bin" / "mlx_lm.server", link):
        out = subprocess.run(
            [str(argv0), "--port", "1"], capture_output=True, text=True, timeout=30
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == f"PY -I -B {argv0} --port 1"


def test_relocate_usage(tmp_path: Path) -> None:
    proc = _run("--relocate-scripts", str(tmp_path / "missing"), "/x")
    assert proc.returncode == 2
    assert "usage: bundle-python.sh --relocate-scripts" in proc.stderr


# --- step 5b: the interpreter takes the product's name ---------------------------


def _pbs_bin(tmp_path: Path) -> Path:
    """A python-build-standalone bin/ as pip and relocation leave it: the real
    interpreter, its symlinks, PBS's stub scripts that exec python3.12, and a
    relocated console script."""
    bin_dir = tmp_path / "python" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python3.12").write_bytes(b"\xcf\xfa\xed\xfe fake interpreter")
    (bin_dir / "python3.12").chmod(0o755)
    stub = '#!/bin/sh\n\'\'\'exec\' "$(dirname -- "$(realpath -- "$0")")/python3.12" "$0" "$@"\n\' \'\'\'\n'
    for name in ("2to3-3.12", "idle3.12", "pydoc3.12", "python3.12-config"):
        (bin_dir / name).write_text(stub)
    for link, target in (
        ("python", "python3.12"),
        ("python3", "python3.12"),
        ("python3-config", "python3.12-config"),
        ("2to3", "2to3-3.12"),
        ("idle3", "idle3.12"),
        ("pydoc3", "pydoc3.12"),
    ):
        (bin_dir / link).symlink_to(target)
    (bin_dir / "uvicorn").write_text("#!/bin/sh\n'''exec' x/'LM Warden Server' -I -B\n' '''\n")
    return bin_dir


def test_rename_moves_the_interpreter_and_drops_every_python_name(tmp_path: Path) -> None:
    bin_dir = _pbs_bin(tmp_path)
    before = (bin_dir / "python3.12").read_bytes()
    proc = _run("--rename-interpreter", str(bin_dir))
    assert proc.returncode == 0, proc.stderr
    exe = bin_dir / SERVER_EXE
    # The file itself (own signature once signed), not a symlink, not a copy.
    assert exe.is_file() and not exe.is_symlink()
    assert exe.read_bytes() == before
    assert os.access(exe, os.X_OK)
    assert sorted(p.name for p in bin_dir.iterdir()) == [SERVER_EXE, "uvicorn"]


def test_rename_refuses_a_script_that_still_execs_python312(tmp_path: Path) -> None:
    bin_dir = _pbs_bin(tmp_path)
    (bin_dir / "stray").write_text("#!/stage/python/bin/python3.12\nprint(1)\n")
    proc = _run("--rename-interpreter", str(bin_dir))
    assert proc.returncode == 1
    assert "bin/stray still execs python3.12" in proc.stderr


def test_rename_refuses_a_dangling_link(tmp_path: Path) -> None:
    bin_dir = _pbs_bin(tmp_path)
    (bin_dir / "py").symlink_to("python3.12")
    proc = _run("--rename-interpreter", str(bin_dir))
    assert proc.returncode == 1
    assert "dangling link left in bin/: py" in proc.stderr


def test_rename_refuses_without_an_interpreter(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    proc = _run("--rename-interpreter", str(bin_dir))
    assert proc.returncode == 1
    assert "no interpreter to rename" in proc.stderr
    assert _run("--rename-interpreter", str(tmp_path / "missing")).returncode == 2


def test_full_run_renames_before_the_leak_check_and_imports_with_it() -> None:
    src = SCRIPT.read_text()
    body = src[src.index("# --- 2. stage") :]
    assert body.index('rename_interpreter "$spy/bin"') < body.index("# --- 6. no build-host paths")
    assert body.index('rename_interpreter "$spy/bin"') > body.index("compileall")
    assert "\"$dest/bin/$SERVER_EXE\" -I -B -c 'import ssl, sqlite3, app'" in body


# --- app/ ships from HEAD, never from a dirty tree ---------------------------------
#
# The script resolves ROOT from its own location (scripts/macos/../..), so a copy
# of it inside a throwaway git repo treats that repo as the checkout. No network:
# --copy-app is the step-4 hook alone, and the full-run leg stops at --offline.


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        },
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "macos").mkdir(parents=True)
    (repo / "scripts" / "macos" / "bundle-python.sh").write_bytes(SCRIPT.read_bytes())
    (repo / "app" / "sub").mkdir(parents=True)
    (repo / "app" / "__init__.py").write_text("")
    (repo / "app" / "main.py").write_text("COMMITTED = True\n")
    (repo / "app" / "sub" / "mod.py").write_text("X = 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "requirements-macos.txt").write_text("")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _copy_app(repo: Path, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "macos" / "bundle-python.sh"), "--copy-app", str(dest)],
        capture_output=True,
        text=True,
        timeout=60,
    )


@needs_git
def test_copy_app_ships_exactly_head(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # Ignored files are not "uncommitted changes" and are not shipped either.
    (repo / "app" / "__pycache__").mkdir()
    (repo / "app" / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\0")
    dest = tmp_path / "site"
    dest.mkdir()
    proc = _copy_app(repo, dest)
    assert proc.returncode == 0, proc.stderr
    shipped = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert shipped == ["app/__init__.py", "app/main.py", "app/sub/mod.py"], shipped
    assert (dest / "app" / "main.py").read_text() == "COMMITTED = True\n"
    assert "git archive" in proc.stdout


@needs_git
@pytest.mark.parametrize(
    "dirty",
    ["modified", "staged", "untracked", "deleted"],
)
def test_copy_app_refuses_a_dirty_app_tree(tmp_path: Path, dirty: str) -> None:
    repo = _repo(tmp_path)
    if dirty == "modified":
        (repo / "app" / "main.py").write_text("UNREVIEWED = True\n")
    elif dirty == "staged":
        (repo / "app" / "main.py").write_text("UNREVIEWED = True\n")
        _git(repo, "add", "app/main.py")
    elif dirty == "untracked":
        (repo / "app" / "new.py").write_text("UNREVIEWED = True\n")
    else:
        (repo / "app" / "sub" / "mod.py").unlink()
    dest = tmp_path / "site"
    dest.mkdir()
    proc = _copy_app(repo, dest)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "uncommitted changes under app/" in proc.stderr
    assert not any(dest.iterdir()), "nothing is copied from a dirty tree"


@needs_git
@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_copy_app_ships_head_even_when_status_is_blind(tmp_path: Path, flag: str) -> None:
    # git status cannot see an edit hidden by assume-unchanged / skip-worktree,
    # so the gate passes; the bytes must still be HEAD's (git archive), never
    # the working tree's.
    repo = _repo(tmp_path)
    _git(repo, "update-index", flag, "app/main.py")
    (repo / "app" / "main.py").write_text("UNREVIEWED = True\n")
    dest = tmp_path / "site"
    dest.mkdir()
    proc = _copy_app(repo, dest)
    assert proc.returncode == 0, proc.stderr
    assert (dest / "app" / "main.py").read_text() == "COMMITTED = True\n"


@needs_git
def test_dirty_outside_app_is_not_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "README.md").write_text("scratch\n")
    dest = tmp_path / "site"
    dest.mkdir()
    assert _copy_app(repo, dest).returncode == 0


@needs_git
def test_full_run_refuses_dirty_app_before_any_download(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    script = repo / "scripts" / "macos" / "bundle-python.sh"
    app = _app(tmp_path)
    cache = tmp_path / "cache"

    def run() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(script), str(app), "--offline", "--cache-dir", str(cache)],
            capture_output=True,
            text=True,
            timeout=60,
        )

    (repo / "app" / "main.py").write_text("UNREVIEWED = True\n")
    proc = run()
    assert proc.returncode == 1
    assert "uncommitted changes under app/" in proc.stderr
    assert not cache.exists(), "the dirty-tree gate runs before the cache is touched"
    # Clean again: the gate passes and the run gets as far as the offline fetch.
    _git(repo, "checkout", "--", "app/main.py")
    proc = run()
    assert proc.returncode == 1
    assert "uncommitted changes" not in proc.stderr
    assert "--offline" in proc.stderr and "is not in" in proc.stderr


@needs_git
def test_plan_names_the_app_source(tmp_path: Path) -> None:
    plan = _plan(str(_app(tmp_path)))
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert plan["app_source"] == f"git archive {head} app"


# --- make macos-dmg is built from the bundled-Python app ----------------------------


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
def test_macos_dmg_without_version_builds_nothing() -> None:
    proc = _make("macos-dmg")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "usage: make macos-dmg VERSION=" in proc.stderr
    assert "swift" not in proc.stdout and "bundle-python" not in proc.stdout


@needs_make
def test_macos_dmg_packs_the_bundled_python_app() -> None:
    proc = _make("-n", "macos-dmg", "VERSION=9.9.9", "OUT=/x/y.dmg", "PYTHON_CACHE=/c/pbs")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    def at(needle: str) -> int:
        i = out.find(needle)
        assert i >= 0, f"{needle!r} not in `make -n macos-dmg`:\n{out}"
        return i

    # macos-app (with the requested version) -> embed python -> sign -> verify -> DMG
    plist = at('VERSION="9.9.9"; ')
    assert plist < at("scripts/macos/write-info-plist.sh"), out
    embed = at('scripts/macos/bundle-python.sh "dist/LM Warden.app" --cache-dir "/c/pbs"')
    sign = at("scripts/macos/sign-bundle.sh")
    verify = at('scripts/macos/verify-bundle.sh "dist/LM Warden.app"')
    dmg = at('scripts/macos/make-dmg.sh "dist/LM Warden.app" "9.9.9" "$OUT"')
    assert plist < embed < sign < verify < dmg, out
    assert 'OUT="/x/y.dmg"' in out
    # The DMG no longer packs a bare swift binary dir.
    assert '.build/release"' not in out[dmg:], out
