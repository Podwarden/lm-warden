"""Unit tests for the P11 prune in ``scripts/macos/bundle-python.sh``.

Not marked ``macos``: every leg here drives the ``--prune`` hook and the
step order against a fake python root on disk -- the hook applies
``scripts/macos/python-prune.txt`` to a throwaway tree under ``tmp_path`` and
needs only ``bash``, ``find``, ``awk`` and ``rm``. It refuses (exit 1, nothing
removed) on a stale rule (a rule that matches nothing), a malformed line and
a rule that would touch a protected path. The real bundle trim (guard on the
trimmed tree, live MLX + llama.cpp smokes, DMG size) is the ``macos``-marked
task p11-trim-m1-proof on the M1.

The fake tree mirrors the REAL python-build-standalone 3.12 macOS layout the
finished bundle ships (headers in ``include/`` at the root, the C extensions
under ``lib/python3.12/lib-dynload/``, the Tcl/Tk dylibs and data under
``lib/``, and no ``lib/python3.12/test`` -- PBS install_only does not ship
the stdlib test suite), so the committed ``python-prune.txt`` matches it
exactly:

    bin/LM Warden Server, bin/llama-server, bin/llama-server.build
    include/Python.h
    lib/{libpython3.12.dylib, libtcl9.0.dylib, libtcl9tk9.0.dylib,
    tk9.0/, tcl9.0/, tcl9/9.0/, itcl4.3.8/, thread3.0.6/}
    lib/python3.12/{os.py, json/, idlelib/, tkinter/, turtledemo/, turtle.py,
    ensurepip/, lib2to3/, pydoc_data/, config-3.12-arm64/,
    lib-dynload/_tkinter.cpython-312-darwin.so,
    site-packages/transformers/models/{qwen2,llama,auto}/{
    __init__,configuration_*,tokenization_*,modeling_*,modular_*,
    image_processing_*,video_processing_*}.py,
    numpy/{_core,linalg,polynomial}/tests/x.py,
    site-packages/app/main.py, site-packages/mlx/core.py,
    site-packages/mlx_lm/server.py, site-packages/PyObjCTest/__init__.py}
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "bundle-python.sh"
PRUNE_LIST = REPO / "scripts" / "macos" / "python-prune.txt"
# The embedded interpreter's product name (Metal design s3).
SERVER_EXE = "LM Warden Server"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _py312(root: Path) -> Path:
    d = root / "lib" / "python3.12"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _site(root: Path) -> Path:
    d = _py312(root) / "site-packages"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(p: Path, content: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fake_root(tmp_path: Path) -> Path:
    """The fake python root as it looks before the prune (a fresh stage after
    pip install and copy_app, before compileall -- no __pycache__), mirroring
    the real PBS 3.12 macOS layout."""
    root = tmp_path / "python"
    _write(root / "bin" / SERVER_EXE, "#!/bin/sh\nfake interpreter\n")
    (root / "bin" / SERVER_EXE).chmod(0o755)
    _write(root / "bin" / "llama-server", "fake llama-server\n")
    _write(root / "bin" / "llama-server.build", "b11719\n")
    _write(root / "lib" / "libpython3.12.dylib", "fake libpython\n")
    _write(root / "lib" / "libtcl9.0.dylib", "fake libtcl\n")
    _write(root / "lib" / "libtcl9tk9.0.dylib", "fake libtcl9tk\n")
    _write(root / "lib" / "tk9.0" / "tk.tcl", "tk data\n")
    _write(root / "lib" / "tcl9.0" / "init.tcl", "tcl data\n")
    _write(root / "lib" / "tcl9" / "9.0" / "http-2.10.2.tm", "tcl module\n")
    _write(root / "lib" / "itcl4.3.8" / "itcl.tcl", "itcl data\n")
    _write(root / "lib" / "thread3.0.6" / "thread.tcl", "thread data\n")
    _write(root / "include" / "Python.h", "#include\n")
    _write(root / "lib-dynload" / "_os.cpython-312-darwin.so", "fake _os\n")

    d = _py312(root)
    _write(d / "os.py", "os\n")
    _write(d / "json" / "__init__.py", "json\n")
    _write(d / "idlelib" / "idle.py", "idle\n")
    _write(d / "tkinter" / "__init__.py", "tk\n")
    _write(d / "tkinter" / "constants.py", "tk constants\n")
    _write(d / "turtledemo" / "bytedesign.py", "turtle demo\n")
    _write(d / "turtle.py", "turtle\n")
    _write(d / "ensurepip" / "__init__.py", "ensurepip\n")
    _write(d / "lib2to3" / "pygram.py", "lib2to3\n")
    _write(d / "pydoc_data" / "__init__.py", "pydoc\n")
    _write(d / "config-3.12-arm64" / "Makefile", "config\n")
    _write(d / "lib-dynload" / "_tkinter.cpython-312-darwin.so", "fake _tkinter\n")

    sp = _site(root)
    _write(sp / "app" / "main.py", "app\n")
    # The real families the server loads, plus models/auto (kept by the tree
    # rules' `! -path '*/models/auto/*'` exclusion).
    for fam in ("qwen2", "llama", "auto"):
        base = sp / "transformers" / "models" / fam
        _write(base / "__init__.py", "family\n")
        _write(base / f"configuration_{fam}.py", "config\n")
        _write(base / f"tokenization_{fam}.py", "tok\n")
        _write(base / f"modeling_{fam}.py", "modeling\n")
        _write(base / f"modular_{fam}.py", "modular\n")
        _write(base / f"image_processing_{fam}.py", "image\n")
        _write(base / f"video_processing_{fam}.py", "video\n")
    _write(sp / "numpy" / "_core" / "tests" / "x.py", "numpy core tests\n")
    _write(sp / "numpy" / "linalg" / "tests" / "y.py", "numpy linalg tests\n")
    _write(sp / "numpy" / "polynomial" / "tests" / "z.py", "numpy poly tests\n")
    _write(sp / "mlx" / "core.py", "mlx\n")
    _write(sp / "mlx_lm" / "server.py", "mlx_lm\n")
    _write(sp / "PyObjCTest" / "__init__.py", "pyobjc tests\n")
    return root


def _py312_paths(root: Path) -> list[Path]:
    """Every path python-prune.txt removes from this fake root. Computed from
    the committed list's rule kinds, so a rule or layout edit keeps the
    expectations in step."""
    d = _py312(root)
    sp = d / "site-packages"
    out: list[Path] = [
        d / "idlelib",
        d / "tkinter",
        d / "turtledemo",
        d / "turtle.py",
        d / "ensurepip",
        d / "lib2to3",
        d / "pydoc_data",
        d / "config-3.12-arm64",
        d / "lib-dynload" / "_tkinter.cpython-312-darwin.so",
        root / "include",
        root / "lib" / "libtcl9.0.dylib",
        root / "lib" / "libtcl9tk9.0.dylib",
        root / "lib" / "tk9.0",
        root / "lib" / "tcl9.0",
        root / "lib" / "tcl9",
        root / "lib" / "itcl4.3.8",
        root / "lib" / "thread3.0.6",
        sp / "PyObjCTest",
        sp / "numpy" / "_core" / "tests",
        sp / "numpy" / "linalg" / "tests",
        sp / "numpy" / "polynomial" / "tests",
    ]
    # Every non-auto family's dead modules (modeling/modular/image/video).
    for fam in ("qwen2", "llama"):
        for f in (
            f"modeling_{fam}.py",
            f"modular_{fam}.py",
            f"image_processing_{fam}.py",
            f"video_processing_{fam}.py",
        ):
            out.append(sp / "transformers" / "models" / fam / f)
    return out


def _stale_list(root: Path) -> Path:
    """python-prune.txt with one rule's target renamed so that rule matches
    nothing: on this tree the list is stale, which must refuse (exit 1)."""
    text = PRUNE_LIST.read_text()
    mangled = text.replace("path lib/python3.12/idlelib", "path lib/python3.12/definitely-not-here")
    assert mangled != text
    out = root / "stale-list.txt"
    out.write_text(mangled)
    return out


def _real_tree_root() -> Path | None:
    """The staged python root of a real Mac bundle, if LMW_BUNDLE_APP names
    one whose python/ has been populated; None otherwise (the leg skips)."""
    app = os.environ.get("LMW_BUNDLE_APP", "")
    if not app:
        return None
    root = Path(app) / "Contents" / "Resources" / "python"
    return root if (root / "lib" / "python3.12").is_dir() else None


# --- the prune removes the safe list and nothing else --------------------------


def test_prune_removes_tkinter_and_modeling_but_keeps_configuration_and_llama_server_build(
    tmp_path: Path,
) -> None:
    root = _fake_root(tmp_path)
    proc = _run("--prune", str(root))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # Per-removal lines name the path relative to the python root.
    assert "pruned: lib/python3.12/tkinter\n" in out
    assert (
        "pruned: lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py\n" in out
    )
    # Every pruned path is gone.
    for p in _py312_paths(root):
        assert not p.exists(), p
    # Kept controls survive (byte-identity is its own leg below).
    d = _py312(root)
    sp = _site(root)
    controls = [
        root / "bin" / SERVER_EXE,
        root / "bin" / "llama-server",
        root / "bin" / "llama-server.build",
        root / "lib" / "libpython3.12.dylib",
        root / "lib-dynload" / "_os.cpython-312-darwin.so",
        d / "os.py",
        d / "json" / "__init__.py",
        sp / "app" / "main.py",
        sp / "mlx" / "core.py",
        sp / "mlx_lm" / "server.py",
    ]
    for fam in ("qwen2", "llama"):
        fam_dir = sp / "transformers" / "models" / fam
        controls += [
            fam_dir / "__init__.py",
            fam_dir / f"configuration_{fam}.py",
            fam_dir / f"tokenization_{fam}.py",
        ]
    for c in controls:
        assert c.exists(), c
    # The final line names the count and a KiB figure.
    last = [ln for ln in out.splitlines() if ln]
    assert last[-1].startswith("bundle-python.sh: pruned "), last[-1]
    n = int(last[-1].split("pruned ")[1].split(" ")[0])
    assert n == len(_py312_paths(root))


def test_prune_tree_kind_reaches_nested_tests_dirs(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    _write(root / "lib" / "python3.12" / "site-packages" / "numpy" / "fft" / "tests" / "w.py")
    proc = _run("--prune", str(root))
    assert proc.returncode == 0, proc.stderr
    for sub in ("_core", "linalg", "polynomial", "fft"):
        assert f"pruned: lib/python3.12/site-packages/numpy/{sub}/tests\n" in proc.stdout
        assert not (_site(root) / "numpy" / sub / "tests").exists()
    # A sibling of the pruned dirs survives.
    assert (_site(root) / "numpy" / "_core").exists()


def test_prune_custom_list_applies_only_that_list(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    custom = tmp_path / "list.txt"
    custom.write_text("# one rule only\npath lib/python3.12/idlelib\n")
    proc = _run("--prune", str(root), "--list", str(custom))
    assert proc.returncode == 0, proc.stderr
    assert "pruned: lib/python3.12/idlelib\n" in proc.stdout
    # Everything the real list would have pruned is still here.
    d = _py312(root)
    assert (d / "tkinter").exists()
    assert (d / "turtle.py").exists()
    sp = _site(root)
    assert (sp / "PyObjCTest").exists()
    last = [ln for ln in proc.stdout.splitlines() if ln]
    # du counts the block-rounded size of the (empty) idle.py file.
    assert last[-1].startswith("bundle-python.sh: pruned 1 paths ")


def test_prune_default_list_is_the_committed_file(tmp_path: Path) -> None:
    # Without --list the hook must read $ROOT/scripts/macos/python-prune.txt
    # (opened only at the prune step, so _repo() copies stay in sync).
    assert PRUNE_LIST.is_file()
    root = _fake_root(tmp_path)
    proc = _run("--prune", str(root))
    assert proc.returncode == 0, proc.stderr
    # The list's stdlib entries are all in effect without --list.
    d = _py312(root)
    assert not (d / "idlelib").exists()
    assert not (d / "ensurepip").exists()
    assert not (d / "lib2to3").exists()
    assert not (d / "pydoc_data").exists()


def test_pruned_tree_is_byte_identical_for_every_kept_file(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    kept = [
        root / "bin" / "llama-server.build",
        root / "lib" / "libpython3.12.dylib",
        root / "lib" / "python3.12" / "os.py",
        root / "lib" / "python3.12" / "site-packages" / "app" / "main.py",
        root / "lib" / "python3.12" / "site-packages" / "mlx" / "core.py",
    ]
    before = {p: _digest(p) for p in kept}
    before.update({root / "bin" / SERVER_EXE: _digest(root / "bin" / SERVER_EXE)})
    assert _run("--prune", str(root)).returncode == 0
    for p, h in before.items():
        assert p.exists(), p
        assert _digest(p) == h, f"prune modified {p}"


# --- the real tree: the committed list matches what the M1 actually ships -------


def test_real_list_matches_the_real_tree_when_a_bundle_is_set(tmp_path: Path) -> None:
    # On the Mac, an UNTRIMMED bundle's python root is the tree the list must
    # match: a clean --prune run proves every rule is real (no stale rule) and
    # nothing protected is hit. The prune runs on a copy, never on the bundle
    # itself (it would break the bundle's signature). Skips off the M1 (no
    # bundle) and on a bundle the build already trimmed (every rule is stale
    # there by design; step 5a of that build was the check).
    real = _real_tree_root()
    if real is None:
        pytest.skip("LMW_BUNDLE_APP not set (or its python/ is absent); runs on the M1")
    if not (real / "lib" / "python3.12" / "tkinter").is_dir():
        pytest.skip("LMW_BUNDLE_APP is already trimmed (no tkinter); point it at an untrimmed one")
    root = tmp_path / "python"
    shutil.copytree(real, root, symlinks=True)
    proc = _run("--prune", str(root))
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    last = [ln for ln in proc.stdout.splitlines() if ln]
    assert last[-1].startswith("bundle-python.sh: pruned "), last


# --- a stale list fails loudly: nothing removed ---------------------------------


def test_stale_rule_refuses_and_removes_nothing(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    proc = _run("--prune", str(root), "--list", str(_stale_list(root)))
    assert proc.returncode == 1, proc.stdout
    assert "matched nothing" in proc.stderr
    # A refusal happens before the first removal: the whole list is validated
    # up front, so nothing moved, even the rules that did match.
    for p in _py312_paths(root):
        assert p.exists(), p
    d = _py312(root)
    assert (d / "tkinter").exists()
    assert (_site(root) / "PyObjCTest").exists()


def test_re_run_over_a_pruned_tree_fails(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    assert _run("--prune", str(root)).returncode == 0
    proc = _run("--prune", str(root))
    assert proc.returncode == 1
    assert "matched nothing" in proc.stderr
    # Documented behaviour: a re-run of the build starts from a fresh stage.


# --- malformed lines refuse (one leg per arm) -----------------------------------


def _malformed(root: Path, line: str, label: str) -> None:
    lst = root / f"bad-{label}.txt"
    lst.write_text(line + "\n")
    proc = _run("--prune", str(root), "--list", str(lst))
    assert proc.returncode == 1, (label, proc.stdout, proc.stderr)
    d = _py312(root)
    assert (d / "tkinter").exists(), label  # nothing removed on a malformed list


def test_malformed_unknown_kind_refuses(tmp_path: Path) -> None:
    _malformed(_fake_root(tmp_path), "bogus lib/python3.12/tkinter", "kind")


def test_malformed_missing_operand_refuses(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    # One operand where the kind needs two, and a bare kind with none.
    _malformed(root, "tree lib/python3.12", "one-operand")
    _malformed(root, "path", "no-operand")


def test_malformed_absolute_path_refuses(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    _malformed(root, "path /etc", "absolute")
    _malformed(root, "tree /usr lib", "absolute-tree")


def test_malformed_dotdot_segment_refuses(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    _malformed(root, "path lib/../lib/python3.12/tkinter", "dotdot")
    _malformed(root, "tree lib/python3.12/site-packages/.. tests", "dotdot-tree")


def test_malformed_empty_rule_refuses(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    _malformed(root, "", "empty")


# --- protected paths are never pruned ------------------------------------------


def _protected_case(root: Path, rule: str, target: Path) -> None:
    lst = root / "prot.txt"
    lst.write_text(rule + "\n")
    proc = _run("--prune", str(root), "--list", str(lst))
    assert proc.returncode == 1, (rule, proc.stdout, proc.stderr)
    assert "protected" in proc.stderr, (rule, proc.stderr)
    assert target.exists(), rule  # the protected target survived
    # Nothing else was removed: a refusal is all-or-nothing.
    d = _py312(root)
    assert (d / "tkinter").exists()


@pytest.mark.parametrize(
    "rule,target_rel",
    [
        ("path bin", "bin/llama-server.build"),
        ("path bin/LM Warden Server", "bin/LM Warden Server"),
        ("path bin/llama-server", "bin/llama-server"),
        ("path bin/llama-server.build", "bin/llama-server.build"),
        ("path lib/libpython3.12.dylib", "lib/libpython3.12.dylib"),
        # A rule matching the site-packages/app dir (contains main.py).
        ("path lib/python3.12/site-packages/app", "lib/python3.12/site-packages/app/main.py"),
        # A rule matching the mlx dir (contains core.py).
        ("path lib/python3.12/site-packages/mlx", "lib/python3.12/site-packages/mlx/core.py"),
        # A rule matching the mlx_lm dir (contains server.py).
        (
            "path lib/python3.12/site-packages/mlx_lm",
            "lib/python3.12/site-packages/mlx_lm/server.py",
        ),
        # A dir that CONTAINS a protected path several levels down: the walk
        # must descend past the immediate children to find it.
        (
            "path lib/python3.12",
            "lib/python3.12/site-packages/app/main.py",
        ),
        (
            "path lib/python3.12/site-packages/transformers",
            "lib/python3.12/site-packages/transformers/models/qwen2/configuration_qwen2.py",
        ),
        # transformers' models/auto/ is protected: the dir and a file in it.
        (
            "path lib/python3.12/site-packages/transformers/models/auto",
            "lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py",
        ),
        (
            "path lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py",
            "lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py",
        ),
    ],
)
def test_protected_entries_refuse(tmp_path: Path, rule: str, target_rel: str) -> None:
    root = _fake_root(tmp_path)
    _protected_case(root, rule, root / target_rel)


def test_protected_transformers_family_files_refuse(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    sp = _site(root)
    # __init__.py of a family: a dir-wide tree rule must stop at the file.
    _protected_case(
        root,
        "tree lib/python3.12/site-packages/transformers/models/qwen2 *",
        sp / "transformers" / "models" / "qwen2" / "__init__.py",
    )
    # A path rule that matches the whole family dir (contains the files).
    _protected_case(
        root,
        "path lib/python3.12/site-packages/transformers/models/llama",
        sp / "transformers" / "models" / "llama" / "configuration_llama.py",
    )


def test_protected_transformers_file_kinds_are_pinned(tmp_path: Path) -> None:
    # One leg per protected file kind: a rule matching the file itself
    # (the file-level protection), and a rule matching the family dir
    # (the dir-level protection, which covers all three kinds).
    root = _fake_root(tmp_path)
    sp = _site(root)
    fam = sp / "transformers" / "models" / "qwen2"
    for rule, kind in [
        ("path lib/python3.12/site-packages/transformers/models/qwen2/__init__.py", "__init__.py"),
        (
            "path lib/python3.12/site-packages/transformers/models/qwen2/configuration_qwen2.py",
            "configuration_qwen2.py",
        ),
        (
            "path lib/python3.12/site-packages/transformers/models/qwen2/tokenization_qwen2.py",
            "tokenization_qwen2.py",
        ),
        ("path lib/python3.12/site-packages/transformers/models/qwen2", "models/qwen2"),
    ]:
        lst = root / "prot-fam.txt"
        lst.write_text(rule + "\n")
        proc = _run("--prune", str(root), "--list", str(lst))
        assert proc.returncode == 1, (rule, proc.stdout, proc.stderr)
        assert kind in proc.stderr, (kind, proc.stderr)
        # The file survived the refusal.
        assert (fam / kind).exists() if kind.endswith(".py") else (fam).exists()


# --- transformers/models/auto/ survives the tree rules ---------------------------


def test_transformers_models_auto_is_kept_by_the_tree_rules(tmp_path: Path) -> None:
    root = _fake_root(tmp_path)
    sp = _site(root)
    auto = sp / "transformers" / "models" / "auto"
    auto_files = [
        auto / "__init__.py",
        auto / "configuration_auto.py",
        auto / "tokenization_auto.py",
        auto / "modeling_auto.py",
        auto / "image_processing_auto.py",
        auto / "video_processing_auto.py",
    ]
    assert all(p.exists() for p in auto_files)
    proc = _run("--prune", str(root))
    assert proc.returncode == 0, proc.stderr
    # The auto dir's own kind-matching files survive the tree rules, while a
    # non-auto family's same-named file is removed (the rules still bite).
    for p in auto_files:
        assert p.exists(), p
    qwen2 = sp / "transformers" / "models" / "qwen2"
    assert not (qwen2 / "modeling_qwen2.py").exists()
    assert not (qwen2 / "image_processing_qwen2.py").exists()
    # Nothing under models/auto/ was pruned.
    assert "/models/auto/" not in proc.stdout


# --- a `path` glob's `*` spans exactly one segment --------------------------------


def test_path_glob_wildcard_spans_one_segment_only(tmp_path: Path) -> None:
    # `find -path` lets `*` cross `/`; a bash glob's `*` must not. `path
    # lib/*.dat` removes the lib/ file but NOT a same-named file one level
    # deeper. (The name is .dat, not .dylib, so it never touches the
    # protected libpython3.12.dylib.)
    root = _fake_root(tmp_path)
    top = root / "lib" / "scratch.dat"
    nested = root / "lib" / "python3.12" / "scratch.dat"
    _write(top, "top\n")
    _write(nested, "nested\n")
    lst = root / "seg.txt"
    lst.write_text("path lib/*.dat\n")
    proc = _run("--prune", str(root), "--list", str(lst))
    assert proc.returncode == 0, proc.stderr
    assert "pruned: lib/scratch.dat\n" in proc.stdout
    assert not top.exists()
    # The nested file is one segment deeper: the one-segment wildcard must
    # not reach it.
    assert nested.exists()
    assert "pruned: lib/python3.12/scratch.dat\n" not in proc.stdout


# --- the main flow: the prune sits after pip uninstall, before compileall -------


def test_main_flow_orders_prune_between_pip_uninstall_and_compileall() -> None:
    src = SCRIPT.read_text()
    body = src[src.index("# --- 2. stage") :]
    order = [
        body.index("pip uninstall --isolated -y -q pip"),
        body.index('prune_python "$spy"'),
        body.index("compileall"),
    ]
    assert order == sorted(order)
    # The list is opened only at the prune step: the main flow passes the
    # list file to prune_python (which opens it there), and so does the
    # --prune hook. Nothing in the body before the prune step opens it --
    # the body starts at "# --- 2. stage", so the header is excluded and the
    # only reference in the body must be the step-5a call.
    before_prune = body[: body.index('prune_python "$spy"')]
    assert "python-prune.txt" not in before_prune


def test_step7_runs_the_import_guard_after_the_import_check() -> None:
    src = SCRIPT.read_text()
    body = src[src.index("# --- 2. stage") :]
    import_check = "\"$dest/bin/$SERVER_EXE\" -I -B -c 'import ssl, sqlite3, app'"
    assert import_check in body  # the existing line, unchanged
    guard = '"$dest/bin/$SERVER_EXE" -I -B "$ROOT/scripts/macos/bundle-import-guard.py"'
    assert guard in body
    assert body.index(import_check) < body.index(guard)
    assert 'die "import guard failed' in body


# --- the committed list itself ---------------------------------------------------


def test_python_prune_list_covers_the_safe_trim_and_keeps_protected() -> None:
    # Every safe-trim entry has a rule; no protected path may match.
    text = PRUNE_LIST.read_text()

    def covers(*needles: str) -> None:
        for n in needles:
            assert n in text, n

    covers(
        "path lib/python3.12/idlelib",
        "path lib/python3.12/tkinter",
        "path lib/python3.12/turtledemo",
        "path lib/python3.12/turtle.py",
        "path lib/python3.12/ensurepip",
        "path lib/python3.12/lib2to3",
        "path lib/python3.12/pydoc_data",
        "path lib/python3.12/config-3.12-*",
        "path include",
        "path lib/libtcl9.0.dylib",
        "path lib/libtcl9tk9.0.dylib",
        "path lib/tk9.0",
        "path lib/tcl9.0",
        "path lib/tcl9\n",
        "path lib/itcl4.3.8",
        "path lib/thread3.0.6",
        "path lib/python3.12/lib-dynload/_tkinter*",
        "tree lib/python3.12/site-packages PyObjCTest",
        "tree lib/python3.12/site-packages/numpy tests",
        "tree lib/python3.12/site-packages/transformers/models modeling_*",
        "tree lib/python3.12/site-packages/transformers/models modular_*",
        "tree lib/python3.12/site-packages/transformers/models image_processing_*",
        "tree lib/python3.12/site-packages/transformers/models video_processing_*",
    )
    # The real layout has no lib/python3.12/test (PBS install_only), so no
    # rule may name it: such a rule would be stale and fail the first build.
    rule_lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for ln in rule_lines:
        assert "lib/python3.12/test" not in ln, ln
    # Not in this task: CLI libraries, .py-without-.pyc, a standalone
    # processing_*.py rule (image_processing_* / video_processing_* ARE in
    # this task), the blanket site-packages tests rule. Checked against the
    # RULE lines only (a comment mentioning a name is fine).
    for banned in (
        "rich",
        "pygments",
        "typer",
        "click",
        "markdown_it",
        "tree lib/python3.12/site-packages tests",
    ):
        for ln in rule_lines:
            assert banned not in ln, (banned, ln)
    # A standalone processing_*.py rule (not image_/video_) would be a
    # transformers rule whose name is exactly processing_* or has
    # processing_ as a whole segment.
    for ln in rule_lines:
        if "transformers" in ln and "processing" in ln:
            name = ln.split()[-1]
            assert name in ("image_processing_*", "video_processing_*"), ln


def test_python_prune_list_rules_are_well_formed(tmp_path: Path) -> None:
    # The list the build runs must itself pass the malformed-line checks and
    # match on the fake tree: one clean --prune run does both at once
    # (a malformed line exits 1 before any removal; a non-matching rule
    # exits 1 as stale).
    root = _fake_root(tmp_path)
    proc = _run("--prune", str(root))
    assert proc.returncode == 0, proc.stderr
    last = [ln for ln in proc.stdout.splitlines() if ln][-1]
    assert last.startswith("bundle-python.sh: pruned ")


# --- usage -----------------------------------------------------------------------


def test_prune_usage_errors_exit_2(tmp_path: Path) -> None:
    # Missing operand, missing root, and a stray argument all exit 2.
    for args in (
        ["--prune"],
        ["--prune", str(tmp_path), "--list"],
        ["--prune", str(tmp_path / "missing"), "--list", str(tmp_path / "l")],
    ):
        proc = _run(*args)
        assert proc.returncode == 2, (args, proc.stderr)
    # --list with a missing file is a refusal (exit 1), not usage.
    proc = _run("--prune", str(_fake_root(tmp_path)), "--list", str(tmp_path / "nope"))
    assert proc.returncode == 1
    assert "list file not found" in proc.stderr
