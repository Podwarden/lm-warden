"""#324: the operator's remote-media allowlist reaches vLLM as --allowed-media-domains."""

from types import SimpleNamespace

from app.proxy.remote_media import ENV_VAR
from app.runtime.backends.vllm.args import build_vllm_args


def _model():
    return SimpleNamespace(
        id="m1",
        served_model_name="x",
        hf_repo="a/b",
        hf_revision=None,
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        extra_env={},
    )


def test_flag_absent_by_default(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert "--allowed-media-domains" not in build_vllm_args(_model(), port=1)


def test_flag_emitted_with_allowlist(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "img.example.com, cdn.example.org")
    args = build_vllm_args(_model(), port=1)
    i = args.index("--allowed-media-domains")
    assert args[i + 1 : i + 3] == ["img.example.com", "cdn.example.org"]
    # the next token after the hosts is another flag, never a stray value
    assert args[i + 3].startswith("--")
