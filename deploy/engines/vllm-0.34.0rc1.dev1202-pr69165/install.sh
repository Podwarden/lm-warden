#!/usr/bin/env bash
# Install the bundled vLLM engine 0.34.0rc1.dev1202-pr69165 into
# /opt/lmw-engines/<version id>. Run by the bundled-vllm stage of ee/Dockerfile.
#
# LMW_BUNDLED_VLLM=0 (the default): do nothing, leave an empty engines dir, so
# the image that does not bundle the engine does not download the wheel set.
#
# Inputs (environment):
#   LMW_BUNDLED_VLLM   1 to install, anything else to skip
#   LMW_KV_SRC         path to the repo's lmw_kv/ directory (has its own pyproject)
#   LMW_ENGINES_ROOT   install root, default /opt/lmw-engines
set -euo pipefail

ROOT="${LMW_ENGINES_ROOT:-/opt/lmw-engines}"
VERSION_ID="0.34.0rc1.dev1202-pr69165"
mkdir -p "$ROOT"

if [ "${LMW_BUNDLED_VLLM:-0}" != "1" ]; then
    echo "bundled vLLM engine: LMW_BUNDLED_VLLM is not 1, skipping (empty $ROOT)"
    exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_URL="https://wheels.vllm.ai/a688ceac3979f7a9097e6d949db9bf91a5a42fde/vllm-0.34.0rc1.dev1202%2Bga688ceac3-cp38-abi3-manylinux_2_28_x86_64.whl"
PATCH_SHA256="03009ff30b39ab213c0c3501331b975f71940a26fa969b17c66c0a1c48e17909"
THINK_PATCH_SHA256="9bd29a9e401bd88b19c2d5dea1849023eacb18ef54c821703149a8a84769334b"
VLLM_VERSION="0.34.0rc1.dev1202+ga688ceac3"
VENV="$ROOT/$VERSION_ID"

command -v uv >/dev/null 2>&1 || pip install --no-cache-dir uv
command -v patch >/dev/null 2>&1 || { apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*; }

# 1. The venv, on the image's own Python 3.12.
uv venv --python /usr/bin/python3.12 "$VENV"

# 2. The exact wheel. Never `-U vllm --pre`: that resolves to stable 0.34.0,
# which lacks the Inkling model.
uv pip install --python "$VENV/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --index-strategy unsafe-best-match \
    "$WHEEL_URL"

# 3. The vendored patch (the vllm/ part of vLLM PR 69165), checked first.
echo "$PATCH_SHA256  $HERE/pr69165-vllm.diff" | sha256sum -c -
SITE="$VENV/lib/python3.12/site-packages"
(cd "$SITE" && patch -p1 --forward < "$HERE/pr69165-vllm.diff")
if find "$SITE" -name '*.rej' -o -name '*.orig' | grep -q .; then
    echo "patch left rejected hunks" >&2
    exit 1
fi

# 3b. LM Warden's own patch: Inkling has no thinking switch, so map the
# Qwen-style chat_template_kwargs.enable_thinking=false to reasoning_effort
# "none" (vllm/renderers/inkling.py). Applied after the PR patch.
echo "$THINK_PATCH_SHA256  $HERE/inkling-enable-thinking.diff" | sha256sum -c -
(cd "$SITE" && patch -p1 --forward < "$HERE/inkling-enable-thinking.diff")
if find "$SITE" -name '*.rej' -o -name '*.orig' | grep -q .; then
    echo "thinking patch left rejected hunks" >&2
    exit 1
fi
grep -q 'reasoning_effort": "none"' "$SITE/vllm/renderers/inkling.py"

# 4. The KV cache store, so a pinned model can use it.
: "${LMW_KV_SRC:?LMW_KV_SRC must point at the lmw_kv directory}"
uv pip install --python "$VENV/bin/python" --no-deps "$LMW_KV_SRC"

# 5. Smoke.
"$VENV/bin/python" - <<PY
import vllm
assert vllm.__version__ == "$VLLM_VERSION", vllm.__version__
from vllm.model_executor.models.registry import ModelRegistry
assert "Inkling2LoopForConditionalGeneration" in ModelRegistry.get_supported_archs()
print("bundled vLLM", vllm.__version__, "ok")
PY
grep -q "e4m3" "$SITE/vllm/models/inkling2/nvidia/ops/__init__.py"
echo "bundled vLLM engine $VERSION_ID installed in $VENV"
