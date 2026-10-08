"""Does the baked binary actually run? Needs a real llama-server; not in CI.

CI's unit-tests job runs `pytest tests/unit` (plus tests/conformance) only, and
these need the built image. Run them by hand inside it:

    docker build -t warden:local .
    docker run --rm --gpus all warden:local \\
        pytest tests/integration/test_llamacpp_smoke.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("llama-server") is None, reason="llama-server is not on PATH"
)


def test_binary_runs_and_reports_the_baked_build():
    out = subprocess.run(["llama-server", "--version"], capture_output=True, text=True, cwd="/")
    combined = out.stdout + out.stderr
    assert "build" in combined.lower()
    baked = os.environ.get("VW_LLAMACPP_BUILD", "")
    if baked:
        assert baked.lstrip("b") in combined, (
            f"the image declares {baked} but the binary reports {combined!r}; "
            "the Dockerfile ARG and the built binary have drifted"
        )


@pytest.mark.parametrize(
    "which_path",
    [True, False],
    ids=["via-the-PATH-symlink", "via-the-real-path"],
)
def test_binary_resolves_its_own_shared_libraries(which_path):
    """The failure this guards against is specific and was observed upstream:
    llama.cpp's own CI images ship an ELF RUNPATH of '/app/build/bin:' -- a
    leftover build-tree path whose empty trailing element resolves against the
    CURRENT WORKING DIRECTORY. That works only because their image sets
    WORKDIR /app. Our engine is launched by a supervisor with a different cwd,
    so we set an rpath at build time instead. If that were ever lost,
    llama-server would die with 'error while loading shared libraries' from some
    directories and work from others -- the worst kind of bug.

    BOTH PATHS, and that is the point of the parametrisation. argv[0] is the
    bare name `llama-server`, so PATH resolves it to a SYMLINK. An exec of the
    symlink works on `$ORIGIN` alone -- the kernel sets /proc/self/exe to the
    real path -- but ``ldd`` expands $ORIGIN from the path it is GIVEN and
    reports "not found" through the symlink. The rpath therefore carries the
    absolute directory too, so the tool and the runtime agree; testing only the
    real path would have missed exactly that.
    """
    path = shutil.which("llama-server") if which_path else "/opt/llamacpp/llama-server"
    out = subprocess.run(["ldd", path], capture_output=True, text=True, cwd="/")
    assert "not found" not in out.stdout, f"{path}:\n{out.stdout}"


def test_cuda_backend_is_present():
    """Needs the NVIDIA driver mapped into the container -- i.e. run this inside
    the k3s pod, or under `docker run --gpus all` on a host with the nvidia
    runtime registered.

    Without a driver, ggml's dlopen of libggml-cuda.so fails on libcuda.so.1 and
    llama-server prints "compiled without support for GPU offload". That is the
    correct report for a container with no GPU, NOT a build defect -- the .so is
    present and links libcudart.so.13 / libcublas.so.13 from the base image's own
    CUDA 13 runtime. Skipped rather than failed so a CPU-only run of this file
    stays honest about what it did and did not check.
    """
    out = subprocess.run(
        ["llama-server", "--list-devices"], capture_output=True, text=True, cwd="/"
    )
    combined = out.stdout + out.stderr
    if "without support for GPU offload" in combined:
        pytest.skip("no NVIDIA driver in this container; nothing to check")
    assert "CUDA" in combined, combined


def test_the_argv_the_backend_builds_is_accepted_by_the_binary():
    """Every flag in the golden is already checked against the captured --help.
    This closes the loop the other way: the BUILT binary parses the argv this
    build's backend would actually emit. A flag that exists but is rejected in
    combination -- or a build configured without the feature behind it -- fails
    here rather than at load time."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    golden = json.loads((repo / "tests" / "fixtures" / "llamacpp_launch_golden.json").read_text())
    argv = list(golden["minimal_single_gpu"]["argv"])
    # Swap the fictional model path for a request the binary can refuse cleanly:
    # --help exits 0 after parsing everything before it.
    argv = [a for a in argv if a not in ("--model",)]
    argv = [a for a in argv if not a.endswith(".gguf")]
    out = subprocess.run(argv + ["--help"], capture_output=True, text=True, cwd="/")
    combined = out.stdout + out.stderr
    assert "error while parsing" not in combined.lower(), combined[-2000:]
    assert "invalid argument" not in combined.lower(), combined[-2000:]
