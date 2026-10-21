"""Unit tests for ``scripts/macos/bundle-webui.sh`` (P8 slice 4, the UI export in the app).

Not marked ``macos``: no leg runs node, npm or codesign. The build path runs a
fake ``node_modules/.bin/next`` inside a throwaway git repo (the script resolves
ROOT from its own location, so a copy of it there treats that repo as the
checkout). The real ``LMW_UI_EXPORT=1 next build`` + sign + verify is the
default ``make macos-bundle-python`` run on the M1. Legs that need git,
make or node skip with a reason where the tool is absent (the CI deps image).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "bundle-webui.sh"
LAYOUT = REPO / "scripts" / "macos" / "verify-export-layout.sh"
# ShellAppDelegate.bundledWebUIPath(): the dir the launcher hands as VW_WEBUI_DIR.
WEBUI_REL = Path("Contents") / "Resources" / "webui"

needs_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="needs git (absent from the CI deps image; runs on the bare host and the M1)",
)
needs_make = pytest.mark.skipif(
    shutil.which("make") is None,
    reason="needs make (absent from the CI deps image; runs on the bare host and the M1)",
)
needs_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="the build path checks for node on PATH (the fake next never runs it)",
)


def _run(
    *args: str, script: Path = SCRIPT, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _app(tmp_path: Path) -> Path:
    app = tmp_path / "LM Warden.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    return app


def _export(out: Path, marker: str = "exported") -> Path:
    """A minimal export verify-export-layout.sh accepts."""
    (out / "_next" / "static").mkdir(parents=True)
    (out / "_next" / "static" / "main.js").write_text("// main\n")
    (out / "index.html").write_text(
        f'<html><head><script src="/ui/_next/static/main.js"></script></head><body>{marker}</body></html>\n'
    )
    (out / "models").mkdir()
    (out / "models" / "index.html").write_text("<html><body>models</body></html>\n")
    return out


def _plan(*args: str) -> dict[str, str]:
    proc = _run(*args, "--print-plan")
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


# --- usage --------------------------------------------------------------------


@pytest.mark.parametrize(
    "args", [[], ["--work-dir"], ["--export-dir"], ["--bogus", "x"], ["a.app", "b.app"]]
)
def test_usage_errors_exit_2(args: list[str]) -> None:
    proc = _run(*args)
    assert proc.returncode == 2
    assert "usage: bundle-webui.sh" in proc.stderr


def test_not_a_bundle_exits_2(tmp_path: Path) -> None:
    proc = _run(str(tmp_path))
    assert proc.returncode == 2
    assert "not an app bundle" in proc.stderr


def test_missing_export_dir_exits_2(tmp_path: Path) -> None:
    proc = _run(str(_app(tmp_path)), "--export-dir", str(tmp_path / "nope"))
    assert proc.returncode == 2
    assert "--export-dir is not a directory" in proc.stderr


# --- the plan -------------------------------------------------------------------


def test_plan_dest_is_the_launchers_webui_dir(tmp_path: Path) -> None:
    app = _app(tmp_path)
    plan = _plan(str(app))
    dest = str(app.resolve() / WEBUI_REL)
    assert plan["dest"] == dest
    assert plan["env"] == f"VW_WEBUI_DIR={dest}"
    assert plan["build"] == "LMW_UI_EXPORT=1 NEXT_TELEMETRY_DISABLED=1 next build"
    assert plan["node_modules"] == str(REPO / "frontend" / "node_modules")
    assert plan["work_dir"] == str(REPO / ".cache" / "macos-webui")
    assert plan["check"] == "scripts/macos/verify-export-layout.sh"
    assert not (app / WEBUI_REL).exists(), "--print-plan touches nothing"


def test_plan_swift_launcher_names_the_same_dir() -> None:
    # The Swift side of the contract: ShellAppDelegate.bundledWebUIPath and the
    # env key the server reads (app/main.py).
    swift = (REPO / "macos" / "Sources" / "LMWarden" / "ShellAppDelegate.swift").read_text()
    assert 'bundlePath + "/Contents/Resources/webui"' in swift
    spec = (REPO / "macos" / "Sources" / "LMWarden" / "ServerLaunchSpec.swift").read_text()
    assert 'webuiEnvironmentKey = "VW_WEBUI_DIR"' in spec
    # Both launch paths hand the server the gated bundle dir (no unit test can
    # boot NSApplication or the headless entry point, so pin the call sites).
    assert "webuiDir: Self.bundledWebUIDir()" in swift
    main = (REPO / "macos" / "Sources" / "LMWarden" / "main.swift").read_text()
    assert "webuiDir: ShellAppDelegate.bundledWebUIDir()" in main
    assert 'os.environ.get("VW_WEBUI_DIR")' in (REPO / "app" / "main.py").read_text()


@needs_git
def test_plan_names_the_frontend_source(tmp_path: Path) -> None:
    plan = _plan(str(_app(tmp_path)))
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert plan["source"] == f"git archive {head} frontend"


def test_plan_with_export_dir(tmp_path: Path) -> None:
    out = _export(tmp_path / "out")
    plan = _plan(str(_app(tmp_path)), "--export-dir", str(out))
    assert plan["source"] == f"export-dir {out.resolve()}"
    assert "build" not in plan and "node_modules" not in plan


def test_plan_relative_work_dir_is_made_absolute(tmp_path: Path) -> None:
    plan = _plan(str(_app(tmp_path)), "--work-dir", "rel/w")
    assert plan["work_dir"] == str(Path.cwd() / "rel" / "w")


# --- --export-dir: checks and copy ----------------------------------------------------


def test_export_dir_is_copied_into_resources_webui(tmp_path: Path) -> None:
    app = _app(tmp_path)
    stale = app / WEBUI_REL
    stale.mkdir(parents=True)
    (stale / "old.html").write_text("stale\n")
    proc = _run(str(app), "--export-dir", str(_export(tmp_path / "out")))
    assert proc.returncode == 0, proc.stderr
    dest = app / WEBUI_REL
    shipped = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert shipped == ["_next/static/main.js", "index.html", "models/index.html"], shipped
    assert "exported" in (dest / "index.html").read_text()
    assert "export layout ok" in proc.stdout
    assert "web UI export (3 files) at LM Warden.app/Contents/Resources/webui" in proc.stdout
    # Nothing staged is left beside it.
    assert sorted(p.name for p in (app / "Contents" / "Resources").iterdir()) == ["webui"]


def test_export_failing_the_layout_check_is_refused(tmp_path: Path) -> None:
    out = _export(tmp_path / "out")
    (out / "index.html").unlink()
    app = _app(tmp_path)
    proc = _run(str(app), "--export-dir", str(out))
    assert proc.returncode == 1
    assert "fails verify-export-layout.sh" in proc.stderr
    assert not (app / WEBUI_REL).exists()


def test_export_with_a_symlink_is_refused(tmp_path: Path) -> None:
    out = _export(tmp_path / "out")
    (out / "escape").symlink_to("/etc")
    app = _app(tmp_path)
    proc = _run(str(app), "--export-dir", str(out))
    assert proc.returncode == 1
    assert "symlinks in the export" in proc.stderr
    assert not (app / WEBUI_REL).exists()


def test_export_naming_a_build_host_path_is_refused(tmp_path: Path) -> None:
    app = _app(tmp_path)
    out = _export(tmp_path / "out")
    (out / "_next" / "static" / "leak.js").write_text(f"// {app.resolve()}\n")
    proc = _run(str(app), "--export-dir", str(out))
    assert proc.returncode == 1
    assert "build-host paths in the export" in proc.stderr
    assert "_next/static/leak.js" in proc.stderr
    assert not (app / WEBUI_REL).exists()


def test_export_naming_the_checkout_is_refused(tmp_path: Path) -> None:
    # The checkout's own path is the second leak the check looks for (a
    # source-map or a server-side path baked into a chunk).
    app = _app(tmp_path)
    out = _export(tmp_path / "out")
    (out / "_next" / "static" / "leak.js").write_text(f"// {REPO}/frontend/src/app/page.tsx\n")
    proc = _run(str(app), "--export-dir", str(out))
    assert proc.returncode == 1
    assert "build-host paths in the export" in proc.stderr
    assert "_next/static/leak.js" in proc.stderr
    assert not (app / WEBUI_REL).exists()


# --- the build path: dirty gate, HEAD only, the export env ------------------------------

FAKE_NEXT = """#!/bin/bash
# Stand-in for `next build`: refuses unless it is the export build, then writes
# an export whose index.html carries src/page.txt (the committed page).
set -eu
[ "$1" = build ] || exit 64
[ "${LMW_UI_EXPORT:-}" = 1 ] || { echo "not an export build" >&2; exit 65; }
[ "${NEXT_TELEMETRY_DISABLED:-}" = 1 ] || exit 66
[ -z "${ANALYZE:-}" ] || exit 67
[ ! -e .env.local ] || { echo "ignored .env.local reached the build" >&2; exit 68; }
mkdir -p out/_next/static
echo '// main' > out/_next/static/main.js
[ -z "${FAKE_LEAK_PWD:-}" ] || echo "// $PWD" > out/_next/static/leak.js
printf '<html><script src="/ui/_next/static/main.js"></script><body>%s</body></html>\\n' "$(cat src/page.txt)" > out/index.html
"""


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


def _repo(tmp_path: Path, node_modules: bool = True) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "macos").mkdir(parents=True)
    for s in (SCRIPT, LAYOUT):
        (repo / "scripts" / "macos" / s.name).write_bytes(s.read_bytes())
    fe = repo / "frontend"
    (fe / "src").mkdir(parents=True)
    (fe / "package.json").write_text("{}\n")
    (fe / "src" / "page.txt").write_text("COMMITTED\n")
    (repo / ".gitignore").write_text("node_modules/\n.env.local\nout/\n")
    if node_modules:
        nb = fe / "node_modules" / ".bin"
        nb.mkdir(parents=True)
        (nb / "next").write_text(FAKE_NEXT)
        (nb / "next").chmod(0o755)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _build(repo: Path, app: Path, work: Path, **extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ANALYZE": "true", **extra}
    return _run(
        str(app),
        "--work-dir",
        str(work),
        script=repo / "scripts" / "macos" / "bundle-webui.sh",
        env=env,
    )


@needs_git
@needs_node
def test_build_ships_heads_frontend_as_an_export(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # Ignored files are neither "uncommitted changes" nor part of the build.
    (repo / "frontend" / ".env.local").write_text("NEXT_PUBLIC_X=1\n")
    app, work = _app(tmp_path), tmp_path / "work"
    proc = _build(repo, app, work)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "git archive" in proc.stdout
    assert "COMMITTED" in (app / WEBUI_REL / "index.html").read_text()
    assert list(work.iterdir()) == [], "the build dir is removed afterwards"


@needs_git
@needs_node
def test_build_refuses_an_export_naming_its_build_dir(tmp_path: Path) -> None:
    # The temp build dir (outside the checkout) is the third leak looked for.
    repo = _repo(tmp_path)
    app, work = _app(tmp_path), tmp_path / "work"
    proc = _build(repo, app, work, FAKE_LEAK_PWD="1")
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "build-host paths in the export" in proc.stderr
    assert "_next/static/leak.js" in proc.stderr
    assert not (app / WEBUI_REL).exists()
    assert list(work.iterdir()) == [], "the build dir is removed on failure too"


@needs_git
@needs_node
@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_build_uses_head_even_when_status_is_blind(tmp_path: Path, flag: str) -> None:
    repo = _repo(tmp_path)
    _git(repo, "update-index", flag, "frontend/src/page.txt")
    (repo / "frontend" / "src" / "page.txt").write_text("UNREVIEWED\n")
    app = _app(tmp_path)
    proc = _build(repo, app, tmp_path / "work")
    assert proc.returncode == 0, proc.stderr
    page = (app / WEBUI_REL / "index.html").read_text()
    assert "COMMITTED" in page and "UNREVIEWED" not in page


@needs_git
@pytest.mark.parametrize("dirty", ["modified", "staged", "untracked", "deleted"])
def test_build_refuses_a_dirty_frontend_before_building(tmp_path: Path, dirty: str) -> None:
    repo = _repo(tmp_path)
    page = repo / "frontend" / "src" / "page.txt"
    if dirty == "modified":
        page.write_text("UNREVIEWED\n")
    elif dirty == "staged":
        page.write_text("UNREVIEWED\n")
        _git(repo, "add", "frontend/src/page.txt")
    elif dirty == "untracked":
        (repo / "frontend" / "src" / "new.tsx").write_text("x\n")
    else:
        page.unlink()
    app, work = _app(tmp_path), tmp_path / "work"
    proc = _build(repo, app, work)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "uncommitted changes under frontend/" in proc.stderr
    assert not work.exists(), "the gate runs before the build dir exists"
    assert not (app / WEBUI_REL).exists()


@needs_git
def test_dirty_outside_frontend_is_not_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path, node_modules=False)
    (repo / "README.md").write_text("scratch\n")
    proc = _build(repo, _app(tmp_path), tmp_path / "work")
    # Past the gate: it stops at the missing node_modules instead.
    assert "uncommitted changes" not in proc.stderr
    assert "node_modules" in proc.stderr


@needs_git
def test_build_without_node_modules_says_how_to_get_them(tmp_path: Path) -> None:
    repo = _repo(tmp_path, node_modules=False)
    app, work = _app(tmp_path), tmp_path / "work"
    proc = _build(repo, app, work)
    assert proc.returncode == 1
    assert "npm ci" in proc.stderr and "frontend-deps.sh" in proc.stderr
    assert not work.exists()


# --- make macos-bundle-python wiring --------------------------------------------------


def _make(*args: str) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VERSION", "OUT", "MAKEFLAGS", "MACOS_WEBUI", "WEBUI_EXPORT")
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
@pytest.mark.parametrize(
    ("args", "call"),
    [
        ((), 'scripts/macos/bundle-webui.sh "dist/LM Warden.app"; '),  # the default
        (("MACOS_WEBUI=1",), 'scripts/macos/bundle-webui.sh "dist/LM Warden.app"; '),
        (
            ("WEBUI_EXPORT=/x/out",),
            'scripts/macos/bundle-webui.sh "dist/LM Warden.app" --export-dir "/x/out"',
        ),
    ],
)
def test_bundle_python_bundles_the_ui_before_signing(args: tuple[str, ...], call: str) -> None:
    proc = _make("-n", "macos-bundle-python", *args)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    embed = out.find("scripts/macos/bundle-python.sh")
    webui = out.find(call)
    sign = out.find("scripts/macos/sign-bundle.sh")
    verify = out.find("scripts/macos/verify-bundle.sh")
    assert -1 < embed < webui < sign < verify, out


@needs_make
def test_bundle_python_bundles_the_ui_by_default() -> None:
    proc = _make("-n", "macos-bundle-python")
    assert proc.returncode == 0, proc.stderr
    assert 'elif [ "1" = 0 ]' in proc.stdout  # the opt-out branch is not taken
    assert 'if [ -n "" ]' in proc.stdout  # no prebuilt export


@needs_make
def test_bundle_python_opt_out_leaves_the_ui_out() -> None:
    # MACOS_WEBUI=0: the fast local build (no node, no next build).
    proc = _make("-n", "macos-bundle-python", "MACOS_WEBUI=0")
    assert proc.returncode == 0, proc.stderr
    assert 'elif [ "0" = 0 ]' in proc.stdout
    assert "web UI not bundled (MACOS_WEBUI=0)" in proc.stdout


@needs_make
@pytest.mark.parametrize("choice", ["0", "1"])
def test_macos_dmg_passes_the_ui_choice_through(choice: str) -> None:
    proc = _make("-n", "macos-dmg", "VERSION=9.9.9", "OUT=/x/y.dmg", f"MACOS_WEBUI={choice}")
    assert proc.returncode == 0, proc.stderr
    assert f'MACOS_WEBUI="{choice}"' in proc.stdout
    assert f'elif [ "{choice}" = 0 ]' in proc.stdout
    assert proc.stdout.find("scripts/macos/bundle-webui.sh") < proc.stdout.find(
        "scripts/macos/make-dmg.sh"
    )


@needs_make
def test_macos_dmg_bundles_the_ui_by_default() -> None:
    proc = _make("-n", "macos-dmg", "VERSION=9.9.9", "OUT=/x/y.dmg")
    assert proc.returncode == 0, proc.stderr
    assert 'MACOS_WEBUI="1"' in proc.stdout
