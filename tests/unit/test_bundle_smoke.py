"""GUI bundle smoke of LM Warden.app on Apple Silicon (P7, bundle-smoke).

Selected only under ``-m macos`` (see ``pyproject.toml``). It runs
``scripts/macos/smoke-bundle.sh`` — build, launch the smoke bundle
(``com.podwarden.lmwarden.smoke``) in GUI mode, probe the server child, quit
through osascript and SIGTERM, assert nothing is left running and the real
Keychain item and data dir are untouched — and fails with the script's output.

It is opt-in: it skips unless ``LMW_GUI_SMOKE=1`` is set (the M1 farm lane sets
it). Otherwise it skips only when the box cannot run a GUI smoke at all: not
macOS, not Apple Silicon, or no console (GUI) session for this user. Everything
else is a failure, never a silent skip.

The interpreter is ``LMW_SMOKE_PYTHON`` when set; else the running
``sys.executable`` when it imports ``app.main`` under ``-I``; else a throwaway
venv over ``sys.executable`` whose ``.pth`` adds the repo root and the running
interpreter's site-packages (a farm venv holds the dependencies but does not
install the app itself). If that still cannot import the app, the test fails
with the reason.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "smoke-bundle.sh"


def _imports_app(python: str) -> str | None:
    """None when ``python -I`` imports app.main, else the reason."""
    try:
        done = subprocess.run(
            [python, "-I", "-c", "import app.main"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{python}: {exc}"
    if done.returncode == 0:
        return None
    tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
    return f"{python} -I cannot import app.main: {tail}"


def _overlay_venv(tmp_path: Path) -> str:
    """A venv over sys.executable whose .pth puts the repo root and the running
    interpreter's site-packages on sys.path (a .pth is honoured under -I)."""
    venv = tmp_path / "smoke-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    python = venv / "bin" / "python3"
    purelib = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    paths = [str(REPO)]
    for key in ("purelib", "platlib"):
        site = sysconfig.get_paths()[key]
        if site not in paths:
            paths.append(site)
    (Path(purelib) / "lmw-smoke.pth").write_text("\n".join(paths) + "\n")
    return str(python)


def _interpreter(tmp_path: Path) -> str:
    override = os.environ.get("LMW_SMOKE_PYTHON", "")
    if override:
        reason = _imports_app(override)
        if reason is not None:
            pytest.fail(f"LMW_SMOKE_PYTHON: {reason}")
        return override
    if _imports_app(sys.executable) is None:
        return sys.executable
    python = _overlay_venv(tmp_path)
    reason = _imports_app(python)
    if reason is not None:
        pytest.fail(f"no interpreter imports the app under -I: {reason}")
    return python


def _gui_skip_reason() -> str | None:
    if os.environ.get("LMW_GUI_SMOKE") != "1":
        return "GUI smoke is opt-in: set LMW_GUI_SMOKE=1 (the M1 lane sets it)"
    if sys.platform != "darwin":
        return "not macOS"
    if platform.machine() != "arm64":
        return "not Apple Silicon"
    console = os.stat("/dev/console").st_uid
    if console != os.getuid():
        return "no GUI (console) session for this user"
    return None


def test_bundle_smoke(tmp_path: Path) -> None:
    reason = _gui_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    assert SCRIPT.is_file(), f"missing {SCRIPT}"

    python = _interpreter(tmp_path)
    done = subprocess.run(
        ["/bin/bash", str(SCRIPT), "--python", python],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=840,
        check=False,
    )
    output = done.stdout + done.stderr
    assert done.returncode == 0, f"smoke-bundle.sh exited {done.returncode}:\n{output}"
    assert "smoke: PASS" in output, output


def test_gui_smoke_opt_in_required(monkeypatch) -> None:
    opt_in = "GUI smoke is opt-in: set LMW_GUI_SMOKE=1 (the M1 lane sets it)"
    # A linux platform pins the ORDER: the opt-in is checked before the platform/arch/console checks.
    monkeypatch.setattr(sys, "platform", "linux")
    for value in ("", "0", "true", " 1", "1 "):
        monkeypatch.setenv("LMW_GUI_SMOKE", value)
        assert _gui_skip_reason() == opt_in
    monkeypatch.delenv("LMW_GUI_SMOKE", raising=False)
    assert _gui_skip_reason() == opt_in


def test_gui_smoke_opt_in_passes_through(monkeypatch) -> None:
    monkeypatch.setenv("LMW_GUI_SMOKE", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    assert _gui_skip_reason() == "not macOS"
