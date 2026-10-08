"""The only test in this suite with an EXTERNAL oracle.

Every other characterisation test is self-referential: the golden file records
whatever the code currently does, which proves a refactor changed nothing but
never proves the output was right. These three cases compare the builder's
output against argv that two live production installs actually produced on
2026-09-01.

`home` matters most and is the reason this file exists. Only `bonus` is wiped
and reinstalled -- `home` is a production dependency (15 API tokens, four in
active use) and is never touched -- so there is no post-release `home` to diff
against. It is also the only install with TP=4, a 262144 context and an fp8 KV
cache. This test is how those argv branches stay covered.

A failure here is NOT a golden-file regeneration. It means the builder no
longer reproduces what production ran.
"""

import pytest

from app.runtime.backends.vllm.args import build_vllm_args
from tests.unit.runtime.backends.corpus import LIVE_CASES, load_live_baseline

BASELINE = load_live_baseline()


@pytest.mark.parametrize("case", LIVE_CASES, ids=[c.name for c in LIVE_CASES])
def test_builder_reproduces_the_live_argv(case):
    argv = build_vllm_args(case.model, port=case.port, overrides=case.overrides, driver=case.driver)
    base = BASELINE[case.name]
    flag = "--enable-prompt-tokens-details"
    # The one deliberate addition since 2026-09-01: the builder now also asks
    # vLLM for usage.prompt_tokens_details (measured prefix-cache hits). It must
    # appear exactly once, just before extra_args, and nothing else may differ.
    extra = list(getattr(case.model, "extra_args", []) or [])
    i = len(base) - len(extra)
    assert argv.count(flag) == 1
    assert argv == base[:i] + [flag] + base[i:]


def test_the_multi_gpu_path_is_covered_by_a_real_row():
    """Named so the coverage claim is greppable.

    `home` is never reinstalled, so nothing downstream of this corpus exercises
    tensor-parallel argv against real production values. If this case is ever
    deleted, the TP path loses its only real-world check."""
    home = next(c for c in LIVE_CASES if c.name == "live_home_qwen3_8_27b_fp8")
    assert home.model.tensor_parallel_size == 4
    assert home.model.gpu_indices == [0, 1, 2, 3]
    assert home.model.max_model_len == 262144
    assert "--tensor-parallel-size" in BASELINE[home.name]
    assert BASELINE[home.name][BASELINE[home.name].index("--kv-cache-dtype") + 1] == "fp8"
