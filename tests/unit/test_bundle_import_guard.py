"""Legs for ``scripts/macos/bundle-import-guard.py`` (P11, the import net).

The guard is the regression net the P11 prune is gated on: run by the bundled
interpreter as ``<interp> -I -B scripts/macos/bundle-import-guard.py`` it
imports every module the server needs on the Mac and exits 1 naming the first
failure, so the prune (transformers ``modeling_*``/``modular_*``, PyObjCTest,
Tk, ...) can never drop something the server still imports.

Selected only under ``-m macos``. The small legs run the guard with the
running interpreter (``sys.executable``) and never touch the bundle; the
``--list`` leg skips when that interpreter lacks mlx_lm/transformers; the
real-bundle leg runs the bundled interpreter named by ``LMW_BUNDLE_APP``
(skipped unless set -- the M1 gate sets it and the farm builds an untrimmed
bundle with ``make macos-bundle-python`` for it).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.macos

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "scripts" / "macos" / "bundle-import-guard.py"

OK_RE = re.compile(r"^IMPORT-GUARD: ok ([0-9]+) modules$")
FAIL_RE = re.compile(r"^IMPORT-GUARD: FAIL ([\w.]+): \w+:")
FAILED_RE = re.compile(r"^IMPORT-GUARD: failed ([0-9]+) of ([0-9]+)$")


def _guard_args(*extra: str) -> list[str]:
    return [sys.executable, "-I", "-B", str(GUARD), *extra]


def _run(
    *extra: str, env: dict | None = None, cwd: str | None = None
) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(
        _guard_args(*extra),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=e,
        cwd=cwd,
    )


def _last_line(proc: subprocess.CompletedProcess) -> str:
    return (proc.stdout.strip().splitlines() or [""])[-1]


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


needs_mlx = pytest.mark.skipif(
    not (_has_module("mlx_lm") and _has_module("transformers")),
    reason="sys.executable lacks mlx_lm/transformers",
)


def test_only_json_ok_line_counts_exactly_one() -> None:
    proc = _run("--only", "json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    m = OK_RE.fullmatch(_last_line(proc))
    assert m is not None, proc.stdout
    assert m.group(1) == "1"


def test_only_missing_module_fails_naming_it() -> None:
    proc = _run("--only", "json,lmw_definitely_missing_mod")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    assert any(
        FAIL_RE.match(line) and "lmw_definitely_missing_mod" in line for line in lines
    ), lines
    m = FAILED_RE.fullmatch(_last_line(proc))
    assert m is not None, lines
    assert m.groups() == ("1", "2")


def test_empty_only_and_unknown_flag_are_usage_errors() -> None:
    proc = _run("--only", "")
    assert proc.returncode == 2, proc.returncode
    proc = _run("--nope")
    assert proc.returncode == 2, proc.returncode


def test_walk_reports_a_broken_submodule(tmp_path: Path) -> None:
    # A test-only --walk leg: walk just this package, no (1)-(4).
    pkg = tmp_path / "lmw_guard_probe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text("X = 1\n")
    (pkg / "bad.py").write_text("raise ImportError('boom')\n")
    # Without -I the cwd package is importable.
    e = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-B", str(GUARD), "--walk", "lmw_guard_probe"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=e,
        cwd=tmp_path,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    fails = [line for line in proc.stdout.splitlines() if FAIL_RE.match(line)]
    # Exactly the broken submodule, for its own error: the package and the
    # good submodule import (a guard that fails everything is not a guard).
    assert fails == ["IMPORT-GUARD: FAIL lmw_guard_probe.bad: ImportError: boom"], proc.stdout
    m = FAILED_RE.fullmatch(_last_line(proc))
    assert m is not None and m.groups() == ("1", "3"), proc.stdout


def test_walk_reports_a_recorded_module_deleted_from_disk(tmp_path: Path) -> None:
    # A prune that deletes a whole file leaves nothing on disk to walk: the
    # installing wheel's RECORD keeps it on the list, so its import fails.
    pkg = tmp_path / "lmw_guard_rec"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "kept.py").write_text("X = 1\n")
    dist = tmp_path / "lmw_guard_rec-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: lmw-guard-rec\nVersion: 0.1\n")
    (dist / "RECORD").write_text(
        "lmw_guard_rec/__init__.py,,\n"
        "lmw_guard_rec/kept.py,,\n"
        "lmw_guard_rec/pruned.py,,\n"
        "lmw_guard_rec-0.1.dist-info/RECORD,,\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", str(GUARD), "--walk", "lmw_guard_rec"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=dict(os.environ),
        cwd=tmp_path,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    fails = [m.group(1) for m in map(FAIL_RE.match, proc.stdout.splitlines()) if m]
    assert fails == ["lmw_guard_rec.pruned"], proc.stdout
    m = FAILED_RE.fullmatch(_last_line(proc))
    assert m is not None and m.groups() == ("1", "3"), proc.stdout


def test_walk_reports_a_recorded_module_deleted_from_a_dotted_package(tmp_path: Path) -> None:
    # The real targets are dotted (mlx_lm.models, transformers.models): the
    # dist-info sits next to the TOP package, two levels above the walked dir.
    sub = tmp_path / "lmw_guard_rec2" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "lmw_guard_rec2" / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")
    (sub / "kept.py").write_text("X = 1\n")
    dist = tmp_path / "lmw_guard_rec2-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: lmw-guard-rec2\nVersion: 0.1\n")
    (dist / "RECORD").write_text(
        "lmw_guard_rec2/__init__.py,,\n"
        "lmw_guard_rec2/sub/__init__.py,,\n"
        "lmw_guard_rec2/sub/kept.py,,\n"
        "lmw_guard_rec2/sub/pruned.py,,\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", str(GUARD), "--walk", "lmw_guard_rec2.sub"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=dict(os.environ),
        cwd=tmp_path,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    fails = [m.group(1) for m in map(FAIL_RE.match, proc.stdout.splitlines()) if m]
    assert fails == ["lmw_guard_rec2.sub.pruned"], proc.stdout
    m = FAILED_RE.fullmatch(_last_line(proc))
    assert m is not None and m.groups() == ("1", "3"), proc.stdout


@needs_mlx
def test_list_is_sorted_and_never_names_pruned_files() -> None:
    proc = _run("--list")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines == sorted(lines), "not sorted"
    for expected in (
        "app.main",
        "app.runtime.backends.mlx",
        "app.runtime.backends.llamacpp.version",
        "app.system.apple.metal",
        "mlx_lm.server",
        "transformers.models.auto.tokenization_auto",
        "transformers.models.auto.configuration_auto",
    ):
        assert expected in lines, expected
    assert any(re.fullmatch(r"transformers\.models\.\w+\.tokenization_\w+", line) for line in lines)
    assert any(
        re.fullmatch(r"transformers\.models\.\w+\.configuration_\w+", line) for line in lines
    )
    assert not any(re.search(r"\.modeling_\w+|\.modular_\w+", line) for line in lines)


def test_real_bundle_leg() -> None:
    bundle = os.environ.get("LMW_BUNDLE_APP")
    if not bundle:
        pytest.skip("set LMW_BUNDLE_APP=<path to LM Warden.app> (the M1 gate sets it)")
    interp = os.path.join(bundle, "Contents", "Resources", "python", "bin", "LM Warden Server")
    if not os.path.isfile(interp):
        pytest.skip(f"no bundled interpreter at {interp}")
    proc = subprocess.run(
        [interp, "-I", "-B", str(GUARD)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert OK_RE.fullmatch(_last_line(proc)), proc.stdout
