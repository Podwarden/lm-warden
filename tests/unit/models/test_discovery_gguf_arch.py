"""Tests for GGUF architecture inference + warning emission (#101).

These exercise the pure helper ``_infer_gguf_arch`` and assert that
``discover_repo_files`` attaches a warning when a GGUF file's inferred
arch is unsupported or unknown. End-to-end HF I/O is faked through the
existing test fixtures in ``tests/fakes/fake_hf.py`` so this stays
hermetic.
"""

from __future__ import annotations

import asyncio

import pytest

from app.models.discovery import (
    KNOWN_GGUF_ARCHES,
    DiscoveryWarning,
    _infer_gguf_arch,
    discover_repo_files,
)
from tests.fakes.fake_hf import (
    FakeHfApi,
    FakeModelInfo,
    FakeSibling,
    make_config_fetcher,
    make_hf_api_factory,
)

# --- Pure inference --------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        # Filename heuristic with longer-match-first ordering.
        ("Qwen3-7B-Q4_K_M.gguf", "qwen3"),
        ("Qwen3-MoE-A22B-Q4_K_M.gguf", "qwen3_moe"),
        ("Qwen2-7B-Q5_K_M.gguf", "qwen2"),
        ("Qwen2-MoE-14B-Q4_K_M.gguf", "qwen2_moe"),
        ("Llama-3-8B-Q4_K_M.gguf", "llama3"),
        ("Llama-2-7B-Q4_K_M.gguf", "llama2"),
        ("mistral-7b-Q4_K_M.gguf", "mistral"),
        ("Mixtral-8x7B-Q4_K_M.gguf", "mixtral"),
        ("Phi-3-mini-Q4.gguf", "phi3"),
        ("gemma-2-9b-Q4_K_M.gguf", "gemma2"),
        ("DeepSeek-V3-Q4.gguf", "deepseek_v3"),
        ("DeepSeek-V2-Q4.gguf", "deepseek_v2"),
        ("starcoder2-15b-Q4.gguf", "starcoder2"),
        ("Command-R-35B-Q4.gguf", "command_r"),
        # Nothing recognisable — None signals "unknown" to the caller.
        ("random-quant.gguf", None),
        ("MyCustomModel-Q4_K_M.gguf", None),
    ],
)
def test_infer_gguf_arch_from_filename(filename, expected):
    assert _infer_gguf_arch(filename, None) == expected


def test_infer_gguf_arch_prefers_config_general_architecture():
    """If a config provides ``general.architecture`` we use it verbatim."""
    cfg = {"general.architecture": "Qwen3_MoE"}
    # Filename would also resolve, but config takes precedence — and we
    # lower-case the returned slug for downstream comparisons.
    assert _infer_gguf_arch("Llama-3-8B-Q4_K_M.gguf", cfg) == "qwen3_moe"


def test_infer_gguf_arch_falls_back_to_model_type():
    """``model_type`` is the standard HF config key — common for raw GGUFs."""
    cfg = {"model_type": "Llama"}
    assert _infer_gguf_arch("model.gguf", cfg) == "llama"


def test_infer_gguf_arch_handles_architectures_list():
    """HF configs often expose ``architectures`` as a list — take the first."""
    cfg = {"architectures": ["MistralForCausalLM"]}
    # We do NOT strip the suffix — the matcher in KNOWN_GGUF_ARCHES uses
    # lower-case canonical names, so this lands as "mistralforcausallm",
    # outside the allowlist → triggers ``gguf_arch_unsupported`` further
    # down. That's the contract: if config explicitly says X, we don't
    # invent a heuristic on top.
    assert _infer_gguf_arch("model.gguf", cfg) == "mistralforcausallm"


def test_infer_gguf_arch_unknown_when_nothing_matches():
    assert _infer_gguf_arch("random.gguf", None) is None
    assert _infer_gguf_arch("random.gguf", {}) is None


# --- KNOWN_GGUF_ARCHES allowlist invariants --------------------------------


def test_known_gguf_arches_lowercase_canonical():
    """The allowlist is the canonical lower-case form — any uppercase entry
    would silently never match the inferred slug. Pin invariant."""
    for arch in KNOWN_GGUF_ARCHES:
        assert arch == arch.lower()
        assert arch.strip() == arch


# --- Warning emission via discover_repo_files -----------------------------


def _run_discovery(siblings, *, config=None):
    """Drive ``discover_repo_files`` with hermetic fakes.

    Returns the ``DiscoveryResult`` for assertion. We pre-bake the
    HF-side fakes here so each test case only declares its data.
    """
    info = FakeModelInfo(siblings=siblings)
    api = FakeHfApi(info)
    api_factory = make_hf_api_factory(api)
    config_fetcher = make_config_fetcher(config)
    return asyncio.run(
        discover_repo_files(
            "fake/repo",
            None,
            None,
            hf_api_factory=api_factory,
            config_fetcher=config_fetcher,
        )
    )


def test_discovery_warns_on_unknown_gguf_arch():
    """A GGUF filename that doesn't match any known arch → ``gguf_arch_unknown``."""
    siblings = [FakeSibling("RandomCustom-Q4_K_M.gguf", size=1_000_000_000)]
    result = _run_discovery(siblings)
    warn_types = [w.type for w in result.warnings]
    assert "gguf_arch_unknown" in warn_types
    warning = next(w for w in result.warnings if w.type == "gguf_arch_unknown")
    assert warning.filename == "RandomCustom-Q4_K_M.gguf"
    assert warning.arch is None


def test_discovery_warns_on_unsupported_gguf_arch():
    """A GGUF with an inferred arch NOT in the allowlist → ``gguf_arch_unsupported``."""
    # Config-driven arch outside KNOWN_GGUF_ARCHES.
    siblings = [FakeSibling("model.gguf", size=2_000_000_000)]
    config = {"general.architecture": "exoticnewmodel"}
    result = _run_discovery(siblings, config=config)
    warn_types = [w.type for w in result.warnings]
    assert "gguf_arch_unsupported" in warn_types
    warning = next(w for w in result.warnings if w.type == "gguf_arch_unsupported")
    assert warning.arch == "exoticnewmodel"


def test_discovery_no_warning_for_known_gguf_arch():
    """A GGUF whose filename resolves to a known arch → no warnings."""
    siblings = [FakeSibling("Llama-3-8B-Q4_K_M.gguf", size=4_000_000_000)]
    result = _run_discovery(siblings)
    assert result.warnings == []


def test_discovery_no_warning_for_non_gguf_files():
    """Safetensors / config / tokenizer files never trigger arch warnings."""
    siblings = [
        FakeSibling("config.json", size=2048),
        FakeSibling("model-00001-of-00002.safetensors", size=5_000_000_000),
        FakeSibling("model-00002-of-00002.safetensors", size=5_000_000_000),
        FakeSibling("tokenizer.json", size=4096),
    ]
    result = _run_discovery(siblings)
    assert result.warnings == []


def test_discovery_warning_to_dict_shape():
    """``DiscoveryWarning.to_dict`` is the wire shape the FE consumes."""
    w = DiscoveryWarning(type="gguf_arch_unknown", filename="x.gguf", arch=None)
    assert w.to_dict() == {
        "type": "gguf_arch_unknown",
        "filename": "x.gguf",
        "arch": None,
    }
