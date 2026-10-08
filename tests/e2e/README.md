# E2E smoke tests

These tests require real GPU hardware and are not part of `make test-unit` or
`make test-integration`. Run them manually after each release candidate against
the bonus node.

## test_smoke_qwen3.5-9b.sh

Reproduces the 2026-05-08 production bug regression: the wizard's GPU selection
must reach the vLLM subprocess. Boots a clean container, registers Qwen3.5-9B
with `tensor_parallel_size=2` on GPUs `[1,2]`, and asserts:

1. `/proc/<pid>/environ` shows `CUDA_VISIBLE_DEVICES=1,2` (NOT `0,1,2,3`)
2. Exactly GPUs 1 and 2 are busy in `nvidia-smi`
3. Inference returns 200 with non-empty content
4. After unload, both GPUs are idle

### Usage

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
export VW_BASE=http://bonus-node:8080
./tests/e2e/test_smoke_qwen3.5-9b.sh
```

Exit code 0 = pass. Any non-zero = regression — investigate immediately.
