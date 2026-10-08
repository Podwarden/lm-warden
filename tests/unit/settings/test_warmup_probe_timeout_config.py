"""Config parsing for VW_WARMUP_PROBE_TIMEOUT_S (app/config.py).

#235: the 60.0s default was measured to be well below real engine init
time for current models (NVFP4/FP8 kernel autotuning, multimodal vision
tower profiling, tensor-parallel startup, hybrid-attention page-size
reconciliation) — a model that loaded and served perfectly was marked
`failed` while its engine kept running and holding GPUs. Default raised
to 600.0s.
"""

import pytest

from app.config import load_settings


@pytest.fixture
def _base_env(monkeypatch):
    # Minimum env for load_settings() to succeed.
    monkeypatch.setenv("VW_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "0")
    monkeypatch.delenv("VW_WARMUP_PROBE_TIMEOUT_S", raising=False)
    return monkeypatch


def test_warmup_probe_timeout_defaults_to_600s_when_unset(_base_env):
    s = load_settings()
    assert s.warmup_probe_timeout_s == 600.0


def test_warmup_probe_timeout_env_override_wins(_base_env):
    # Deliberately not the default: an override test that sets the value the
    # code would have picked anyway proves nothing, and silently stops proving
    # anything the moment the default moves.
    _base_env.setenv("VW_WARMUP_PROBE_TIMEOUT_S", "45")
    s = load_settings()
    assert s.warmup_probe_timeout_s == 45.0
