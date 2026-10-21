"""Unit tests for ``scripts/macos/bundle-llamacpp.sh`` (P5, llama.cpp-Metal).

Not marked ``macos``: usage, ``--print-plan``, the pin-drift guards and the
``make -n`` wiring need only bash/make and run in linux CI too. The link-gate
legs compile two tiny Mach-O binaries with the system ``cc`` and need
``otool``/``lipo``, so they skip off a Mac. The real build (pinned cmake +
source, ``cmake --build --target llama-server``, gate, placement, sign, verify)
is the ``make macos-bundle-python`` run on the M1, and the bundled binary's
live serve is ``tests/integration/test_llamacpp_metal_live.py``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "macos" / "bundle-llamacpp.sh"

needs_make = pytest.mark.skipif(
    shutil.which("make") is None,
    reason="needs make (absent from the CI deps image; runs on the bare host and the M1)",
)
needs_mac_toolchain = pytest.mark.skipif(
    not (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and all(shutil.which(t) for t in ("cc", "otool", "lipo"))
    ),
    reason="the link gate needs darwin/arm64 with cc, otool and lipo",
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def _plan(tmp_path: Path) -> dict[str, str]:
    app = tmp_path / "LM Warden.app"
    (app / "Contents").mkdir(parents=True)
    proc = _run(str(app), "--print-plan", "--cache-dir", str(tmp_path / "cache"))
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


def test_usage_without_an_app_is_exit_2() -> None:
    proc = _run()
    assert proc.returncode == 2
    assert "usage: bundle-llamacpp.sh" in proc.stderr


def test_not_an_app_is_exit_2(tmp_path: Path) -> None:
    proc = _run(str(tmp_path))
    assert proc.returncode == 2
    assert "not an app bundle" in proc.stderr


def test_plan_is_a_pinned_source_build(tmp_path: Path) -> None:
    """The decision (Metal design s6): build from a pinned source tag, never
    the upstream macOS binary. Every download in the plan is an exact asset
    with a sha256, and the flags are the static, embedded-Metal set."""
    plan = _plan(tmp_path)
    assert plan["decision"] == "build-from-source"
    assert re.fullmatch(r"b\d+", plan["tag"])
    assert plan["url"].endswith(f"/archive/refs/tags/{plan['tag']}.tar.gz")
    assert re.fullmatch(r"[0-9a-f]{64}", plan["sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", plan["commit"])
    assert re.fullmatch(
        r"https://github\.com/Kitware/CMake/releases/download/v[\d.]+/"
        r"cmake-[\d.]+-macos-universal\.tar\.gz",
        plan["cmake_url"],
    )
    assert re.fullmatch(r"[0-9a-f]{64}", plan["cmake_sha256"])
    flags = plan["cmake_flags"].split()
    for want in (
        "-DGGML_METAL=ON",
        "-DGGML_METAL_EMBED_LIBRARY=ON",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "-DGGML_NATIVE=OFF",
        "-DGGML_OPENMP=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
        f"-DLLAMA_BUILD_NUMBER={plan['tag'][1:]}",
        f"-DLLAMA_BUILD_COMMIT={plan['commit'][:9]}",
    ):
        assert want in flags, (want, flags)


def test_plan_places_the_binary_next_to_the_embedded_interpreter(tmp_path: Path) -> None:
    """Contents/Resources/python/bin is dirname(sys.executable) for the
    bundled interpreter -- the first entry of the apple engine PATH."""
    plan = _plan(tmp_path)
    assert plan["dest"].endswith("/LM Warden.app/Contents/Resources/python/bin/llama-server")
    assert plan["binary"].startswith(str(tmp_path / "cache") + "/build-" + plan["tag"] + "-")


def test_plan_stamps_the_build_beside_the_binary_under_the_name_version_py_reads(
    tmp_path: Path,
) -> None:
    """On the Mac nothing sets VW_LLAMACPP_BUILD; the apple profile reads the
    tag from a stamp file beside llama-server instead. Its name is spelled in
    two places (here and version.py's STAMP_NAME) -- this pins them together,
    and the stamp's content is LLAMACPP_TAG, the one pin."""
    from app.runtime.backends.llamacpp.version import STAMP_NAME

    plan = _plan(tmp_path)
    assert plan["stamp"] == plan["dest"].rsplit("/", 1)[0] + "/" + STAMP_NAME
    text = SCRIPT.read_text()
    assert 'printf \'%s\\n\' "$LLAMACPP_TAG" > "$stamp.tmp.$$"' in text


def test_tag_matches_the_dockerfile_and_the_fixtures(tmp_path: Path) -> None:
    """One llama.cpp version for both products: the Mac pin, the Dockerfile's
    ARG LLAMACPP_BUILD and the tag the fixtures were captured at."""
    plan = _plan(tmp_path)
    docker = re.search(r"^ARG LLAMACPP_BUILD=(\S+)$", (REPO / "Dockerfile").read_text(), re.M)
    assert docker, "Dockerfile has no ARG LLAMACPP_BUILD"
    assert plan["tag"] == docker.group(1)
    readme = (REPO / "tests" / "fixtures" / "llamacpp" / "README.md").read_text()
    assert f"**`{plan['tag']}`**" in readme
    assert plan["commit"][:9] in readme


# --- the otool link gate -------------------------------------------------------


def _cc(*args: str, cwd: Path) -> None:
    subprocess.run(["cc", "-arch", "arm64", *args], cwd=cwd, check=True, timeout=120)


@needs_mac_toolchain
def test_link_gate_passes_a_system_only_binary(tmp_path: Path) -> None:
    (tmp_path / "ok.c").write_text("int main(void){return 0;}\n")
    _cc("-o", "ok", "ok.c", cwd=tmp_path)
    proc = _run("--check-links", str(tmp_path / "ok"))
    assert proc.returncode == 0, proc.stderr
    assert "links: /usr/lib/libSystem.B.dylib" in proc.stdout
    assert "link gate ok" in proc.stdout


@needs_mac_toolchain
def test_link_gate_fails_a_binary_linking_a_non_system_dylib(tmp_path: Path) -> None:
    """The teeth: a dylib outside /usr/lib and /System/Library (what a
    Homebrew libomp/openssl or a BUILD_SHARED_LIBS=ON libggml would look
    like) fails the build."""
    (tmp_path / "f.c").write_text("int f(void){return 1;}\n")
    (tmp_path / "m.c").write_text("int f(void);int main(void){return f();}\n")
    _cc(
        "-dynamiclib", "-install_name", "@rpath/libf.dylib", "-o", "libf.dylib", "f.c", cwd=tmp_path
    )
    _cc("-o", "bad", "m.c", "-L.", "-lf", f"-Wl,-rpath,{tmp_path}", cwd=tmp_path)
    proc = _run("--check-links", str(tmp_path / "bad"))
    assert proc.returncode == 1
    assert "non-system dylib(s)" in proc.stderr and "@rpath/libf.dylib" in proc.stderr


@needs_mac_toolchain
def test_link_gate_fails_a_universal_binary(tmp_path: Path) -> None:
    (tmp_path / "ok.c").write_text("int main(void){return 0;}\n")
    subprocess.run(
        ["cc", "-arch", "arm64", "-arch", "x86_64", "-o", "fat", "ok.c"],
        cwd=tmp_path,
        check=True,
        timeout=120,
    )
    proc = _run("--check-links", str(tmp_path / "fat"))
    assert proc.returncode == 1
    assert "want arm64 only" in proc.stderr


@needs_mac_toolchain
def test_link_gate_fails_a_system_only_binary_carrying_an_lc_rpath(tmp_path: Path) -> None:
    """The LC_RPATH leg on its own: every dependency is a system dylib, so only
    the rpath check can refuse it (the non-system-dylib leg above trips first)."""
    (tmp_path / "ok.c").write_text("int main(void){return 0;}\n")
    _cc("-o", "rp", "ok.c", "-Wl,-rpath,/opt/homebrew/lib", cwd=tmp_path)
    proc = _run("--check-links", str(tmp_path / "rp"))
    assert proc.returncode == 1
    assert "carries an LC_RPATH" in proc.stderr


# --- the embedded commit id ----------------------------------------------------


def _pinned_copy(tmp_path: Path, **pins: str) -> Path:
    """The script with its pins rewritten (exact `NAME="..."` lines only)."""
    text = SCRIPT.read_text()
    for name, value in pins.items():
        text, n = re.subn(rf'^{name}="[^"]*"$', f'{name}="{value}"', text, flags=re.M)
        assert n == 1, name
    out = tmp_path / "bundle-llamacpp.sh"
    out.write_text(text)
    return out


def _sha256(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def _commit_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    """A git-archive source tarball (it carries its commit id like GitHub's)
    and a stand-in cmake tarball whose cmake exits 1, both in an --offline
    cache, with pins naming them. Returns (cache, pins, the real commit)."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "CMakeLists.txt").write_text("project(x)\n")
    subprocess.run([*git, "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "-C", str(repo), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(repo), "commit", "-qm", "x"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    cache = tmp_path / "cache"
    cache.mkdir()
    src = cache / "llama.cpp-b11719.tar.gz"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar.gz",
            "--prefix=llama.cpp-b11719/",
            "-o",
            str(src),
            "HEAD",
        ],
        check=True,
    )
    stage = tmp_path / "stage" / "cmake-x" / "CMake.app" / "Contents" / "bin"
    stage.mkdir(parents=True)
    (stage / "cmake").write_text("#!/bin/sh\nexit 1\n")
    (stage / "cmake").chmod(0o755)
    cm = cache / "cmake-3.31.8-macos-universal.tar.gz"
    subprocess.run(["tar", "-czf", str(cm), "-C", str(tmp_path / "stage"), "cmake-x"], check=True)
    return cache, {"LLAMACPP_SHA256": _sha256(src), "CMAKE_SHA256": _sha256(cm)}, commit


@needs_mac_toolchain
def test_a_source_tarball_with_the_wrong_commit_id_is_refused_before_extraction(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("needs git")
    cache, pins, commit = _commit_fixture(tmp_path)
    wrong = ("0" if commit[0] != "0" else "1") + commit[1:]
    script = _pinned_copy(tmp_path, LLAMACPP_COMMIT=wrong, **pins)
    proc = subprocess.run(
        ["bash", str(script), "--build-only", "--offline", "--cache-dir", str(cache)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"commit id mismatch in {cache}/llama.cpp-b11719.tar.gz" in proc.stderr
    assert not (cache / "src-b11719").exists()
    # The control: the right commit passes the check and reaches configure.
    script = _pinned_copy(tmp_path, LLAMACPP_COMMIT=commit, **pins)
    proc = subprocess.run(
        ["bash", str(script), "--build-only", "--offline", "--cache-dir", str(cache)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert f"==> commit {commit} ok" in proc.stdout, proc.stdout + proc.stderr
    assert "cmake configure failed" in proc.stderr


def test_link_gate_refuses_a_non_macho(tmp_path: Path) -> None:
    if shutil.which("file") is None:
        pytest.skip("needs file(1)")
    f = tmp_path / "llama-server"
    f.write_text("#!/bin/sh\n")
    proc = _run("--check-links", str(f))
    assert proc.returncode == 1
    assert "not a Mach-O" in proc.stderr


# --- make macos-bundle-python wiring -------------------------------------------


def _make(*args: str) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VERSION", "OUT", "MAKEFLAGS", "MACOS_LLAMACPP", "LLAMACPP_CACHE")
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
def test_bundle_python_places_llama_server_after_python_before_signing() -> None:
    proc = _make("-n", "macos-bundle-python", "LLAMACPP_CACHE=/c/lcm")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    embed = out.find("scripts/macos/bundle-python.sh")
    lcm = out.find('scripts/macos/bundle-llamacpp.sh "dist/LM Warden.app" --cache-dir "/c/lcm"')
    sign = out.find("scripts/macos/sign-bundle.sh")
    verify = out.find("scripts/macos/verify-bundle.sh")
    assert -1 < embed < lcm < sign < verify, out
    assert 'if [ "1" = 0 ]' in out  # bundled by default


@needs_make
def test_macos_dmg_passes_the_llamacpp_choice_through() -> None:
    proc = _make("-n", "macos-dmg", "VERSION=9.9.9", "OUT=/x/y.dmg", "MACOS_LLAMACPP=0")
    assert proc.returncode == 0, proc.stderr
    assert 'MACOS_LLAMACPP="0"' in proc.stdout
    assert 'if [ "0" = 0 ]' in proc.stdout
