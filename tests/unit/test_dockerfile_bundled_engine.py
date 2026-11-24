"""The bundled vLLM engine is built into the Community image, on by default."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENGINE = ROOT / "deploy/engines/vllm-0.34.0rc1.dev1202-pr69165"
WHEEL = (
    "https://wheels.vllm.ai/a688ceac3979f7a9097e6d949db9bf91a5a42fde/"
    "vllm-0.34.0rc1.dev1202%2Bga688ceac3-cp38-abi3-manylinux_2_28_x86_64.whl"
)
THINK_SHA = "9bd29a9e401bd88b19c2d5dea1849023eacb18ef54c821703149a8a84769334b"
SHA = "03009ff30b39ab213c0c3501331b975f71940a26fa969b17c66c0a1c48e17909"


def test_install_script_pins_wheel_and_patch():
    s = (ENGINE / "install.sh").read_text()
    assert "set -euo pipefail" in s
    assert WHEEL in s
    assert SHA in s
    assert "patch -p1" in s
    assert THINK_SHA in s
    assert "inkling-enable-thinking.diff" in s
    assert s.index(SHA) < s.index(THINK_SHA)  # PR patch first
    assert "--forward" in s
    assert "unsafe-best-match" in s
    assert "download.pytorch.org/whl/cu132" in s
    assert "--python /usr/bin/python3.12" in s
    assert "LMW_KV_SRC" in s
    assert "e4m3" in s
    assert "Inkling2LoopForConditionalGeneration" in s
    # skips (no download) unless asked
    assert re.search(r'LMW_BUNDLED_VLLM:-0}" != "1"', s)


def test_vendored_patch_matches_the_pinned_sha():
    import hashlib

    assert hashlib.sha256((ENGINE / "pr69165-vllm.diff").read_bytes()).hexdigest() == SHA
    think = (ENGINE / "inkling-enable-thinking.diff").read_bytes()
    assert hashlib.sha256(think).hexdigest() == THINK_SHA


def test_community_dockerfile_bundles_by_default():
    s = (ROOT / "Dockerfile").read_text()
    assert "FROM vllm/vllm-openai@${VW_BASE_DIGEST} AS bundled-vllm" in s
    assert s.count("ARG LMW_BUNDLED_VLLM=1") == 1
    assert "source=deploy/engines" in s and "source=lmw_kv" in s
    assert "vllm-0.34.0rc1.dev1202-pr69165/install.sh" in s
    # the final stage copies the engines dir, before the app source (an app-only
    # commit must not invalidate it)
    final = s.split("AS bundled-vllm")[1].split("\nFROM vllm/vllm-openai@${VW_BASE_DIGEST}\n")[1]
    copy = "COPY --from=bundled-vllm /opt/lmw-engines /opt/lmw-engines"
    assert copy in final
    assert final.index(copy) < final.index("COPY app /app/app")
    # the engine stage builds from the bare base, not from a stage holding app code
    stage = s.split("AS bundled-vllm")[1].split("\nFROM ")[0]
    assert "COPY app" not in stage


def test_deploy_engines_reaches_the_build_context():
    ignore = (ROOT / ".dockerignore").read_text().split()
    assert not any(p.rstrip("/") in ("deploy", "deploy/engines", "lmw_kv") for p in ignore)


def test_no_build_passes_the_arg():
    """Every image gets the engine from the Community build; nothing re-builds it."""
    ci = (ROOT / ".gitlab-ci.yml").read_text() if (ROOT / ".gitlab-ci.yml").is_file() else ""
    assert "LMW_BUNDLED_VLLM" not in ci
    ee = ROOT / "ee/Dockerfile"
    if ee.is_file():
        assert "LMW_BUNDLED_VLLM" not in ee.read_text()
        assert "bundled-vllm" not in ee.read_text()
