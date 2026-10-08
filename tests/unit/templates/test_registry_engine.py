from app.templates.registry import (
    EngineSpec,
    ModelTemplate,
    get_builtin_template,
    list_builtin_templates,
    template_to_dict,
)


def test_engine_spec_defaults():
    s = EngineSpec(channel="cuda-stable", vllm_version="0.20.0")
    assert s.image is None


def test_builtin_gpt_oss_has_engine():
    t = get_builtin_template("gpt-oss-20b")
    assert t is not None
    assert t.source == "builtin"
    assert t.engine == EngineSpec(channel="cuda-stable", vllm_version="0.20.0")


def test_template_to_dict_serializes_engine():
    t = get_builtin_template("gpt-oss-20b")
    d = template_to_dict(t)
    assert d["source"] == "builtin"
    assert d["engine"] == {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None}


def test_template_to_dict_handles_none_engine():
    t = ModelTemplate(
        id="x",
        label="x",
        hf_repo="a/b",
        hf_revision="main",
        dtype="auto",
        max_model_len=2048,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
    )
    assert template_to_dict(t)["engine"] is None


def test_list_builtin_templates_returns_gpt_oss():
    assert any(t.id == "gpt-oss-20b" for t in list_builtin_templates())
